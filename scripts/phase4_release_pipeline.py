"""Expanded train validation and resumable versioned generation, no ML/EDA."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import audit_phase4 as audit
from robot_planner.curves import make_curve
from robot_planner.kinematics import PlanarArm2R
from robot_planner.evaluation import CircleObstacle
from robot_planner.experiments import curve_variants
from robot_planner.phase4 import _sample_arrow_schema,score_phase4_candidates,select_scenario_winners
from robot_planner.phase4_evaluation import MathematicalEvaluationConfig
from robot_planner.phase4_release import DATASET_VERSION,CODE_VERSION,evaluate_release
from robot_planner.phase4_v3 import JOINT_FIELDS
from robot_planner.refinement import differential_joint_state

ROOT=Path(__file__).resolve().parents[1]
SEED=20260912
EXP=ROOT/'outputs/phase4_v3_expansion'
FULL=ROOT/'data/phase4_v3_release'
NUMERIC={'seed','base_x','base_y','link1','link2','start_x','start_y','goal_x','goal_y','link_ratio','min_reach','max_reach','start_radius','goal_radius','chord_length','angular_separation_deg','obstacle_count'}


def json_write(path,value):
    audit.write_json(path,value)


def emit(stream,value):
    stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')


def scenarios():
    with (ROOT/'data/phase4/scenarios.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    for r in rows:
        for k in NUMERIC:
            r[k]=int(r[k]) if k in ('seed','obstacle_count') else float(r[k])
    if len({r['scenario_id'] for r in rows})!=len(rows):raise ValueError('Duplicate scenario IDs')
    return rows


def signature():
    files=['src/robot_planner/phase4_release.py','src/robot_planner/phase4_v3.py',
           'src/robot_planner/refinement.py','src/robot_planner/phase4_evaluation.py',
           'src/robot_planner/phase4.py','src/robot_planner/curves.py',
           'scripts/phase4_release_pipeline.py','docs/phase4_release_policy.md',
           'data/phase4/scenarios.csv','data/phase4/experiment_metadata.json','data/phase4/dataset_manifest.json']
    hashes={p:audit.sha(ROOT/p) for p in files}
    return dict(version=DATASET_VERSION,code_version=CODE_VERSION,hashes=hashes,
                digest=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest())


def select_expansion():
    if (EXP/'selection.json').exists():return json.loads((EXP/'selection.json').read_text(encoding='utf-8'))
    m=audit.load_train(ROOT/'data/phase4',ROOT/'outputs/phase4_quality')
    old=set(pd.read_csv(ROOT/'outputs/phase4_quality/selected_cases.csv').scenario_id)
    pool=m[~m.scenario_id.isin(old)].copy();s=pool.drop_duplicates('scenario_id').sort_values('scenario_id')
    rng=np.random.default_rng(SEED);selected={}
    def choose(ids,n,reason):
        ids=sorted(set(ids)-set(selected))
        for sid in rng.choice(ids,size=min(n,len(ids)),replace=False):selected[str(sid)]=reason
    for (typ,layout),g in s.groupby(['scenario_type','obstacle_layout'],sort=True):
        choose(g.scenario_id,1,f'stratum:{typ}/{layout}')
    choose(pool.loc[pool.min_obstacle_clearance_units.between(0,.05,inclusive='right'),'scenario_id'],8,'small_clearance')
    choose(pool.loc[pool.min_singularity_index.lt(.05),'scenario_id'],8,'near_singularity')
    choose(s.loc[(s.link1+s.link2-np.maximum(s.start_radius,s.goal_radius))/(s.link1+s.link2)<.02,'scenario_id'],8,'reach_boundary')
    choose(pool.loc[pool.fully_reachable.eq(False),'scenario_id'],8,'includes_unreachable')
    if len(selected)>96:raise ValueError('Strata exceed planned sample count')
    choose(s.scenario_id,96-len(selected),'fixed_random_fill')
    result=dict(seed=SEED,scenario_ids=sorted(selected),reasons=selected,excluded_previous=sorted(old),
                policy_sha256=audit.sha(ROOT/'docs/phase4_release_policy.md'),
                train_cache_sha256=audit.sha(ROOT/'outputs/phase4_quality/train_metrics.parquet'))
    json_write(EXP/'selection.json',result);return result


def worker(payload):
    row,destination,config_data,expanded,quarantine,sig=payload
    sid=row['scenario_id'];split=row['data_split'];base=Path(destination)/'partitions'/split/sid
    attempt=base/('attempt_'+uuid.uuid4().hex);attempt.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();cfg=MathematicalEvaluationConfig(**config_data)
    arm=PlanarArm2R(row['link1'],row['link2'],(row['base_x'],row['base_y']))
    obstacles=tuple(CircleObstacle(**o) for o in json.loads(row['obstacles_json']))
    records=[];samples=[];comparisons=[];panels_count=0;node_count=0
    with gzip.open(attempt/'adaptive_panels.jsonl.gz','wt',encoding='utf-8',compresslevel=1) as panels_file:
        for variant in curve_variants():
            curve=make_curve(variant.curve_key,(row['start_x'],row['start_y']),(row['goal_x'],row['goal_y']),shape=variant.shape)
            for branch in ('up','down'):
                rid=f'{sid}_{variant.variant_id}_{branch}'
                result=evaluate_release(arm,curve,branch,obstacles=obstacles,config=cfg,quarantine=rid in quarantine)
                identity={**row,'run_id':rid,'curve_key':variant.curve_key,'curve_name_ko':variant.curve_name_ko,'shape':variant.shape,'elbow_mode':branch}
                record={**identity,**result['metrics'],'legacy_v2':result['legacy_v2'],
                    'path_check':result['path_check'],'geometry_checks':result['geometry_checks'],
                    'joint_sample_grid_integrals':result['joint_sample_grid_integrals'],
                    'quadrature_diagnostics':result['quadrature']['diagnostics']}
                records.append(record)
                sample_id={k:identity[k] for k in ('run_id','scenario_id','data_split','dataset_partition','curve_key','shape','elbow_mode')}
                samples.extend({**sample_id,**s} for s in result['sample_rows'])
                for panel in result['quadrature']['panels']:
                    emit(panels_file,{**sample_id,'integration_status':result['quadrature']['status'],**panel})
                    panels_count+=1;node_count+=len(panel['time_s'])
                if expanded and result['quadrature']['status']=='accepted':
                    times=np.linspace(0,cfg.duration_s,2561)
                    states=[differential_joint_state(arm,curve,float(t),cfg.duration_s,branch) for t in times]
                    for name,(field,_,_,_) in JOINT_FIELDS.items():
                        value=float(np.trapezoid([math.fsum(v*v for v in s[name]) for s in states],times))
                        reference=record[field];atol,rtol=audit.TOL[{'velocity':'E_theta','acceleration':'A','jerk':'J'}[name]]
                        error=abs(value-reference)
                        comparisons.append(dict(run_id=rid,scenario_id=sid,field=field,value2561=value,adaptive=reference,
                            absolute_error=error,relative_error=error/abs(reference) if reference else None,
                            passed=error<=atol+rtol*abs(reference),atol=atol,rtol=rtol))
    scored=score_phase4_candidates(records)
    with gzip.open(attempt/'trajectory_metrics.jsonl.gz','wt',encoding='utf-8',compresslevel=1) as f:
        for r in scored:emit(f,r)
    winners=select_scenario_winners(scored);unknown=any(r['candidate_status']=='unresolved' for r in records)
    for w in winners:w.update(dataset_version=DATASET_VERSION,
        label_readiness='provisional_unresolved_candidates' if unknown else 'numerically_screened_not_model_evaluated')
    json_write(attempt/'scenario_winners.json',winners)
    schema=_sample_arrow_schema().append(pa.field('joint_sample_status',pa.string()))
    pq.write_table(pa.Table.from_pylist(samples,schema=schema),attempt/'trajectory_samples.parquet',compression='zstd')
    if expanded:json_write(attempt/'joint_comparisons.json',comparisons)
    # Input and mechanical bookkeeping only: no held-out outcome aggregation returned.
    receipt=dict(scenario_id=sid,data_split=split,signature=sig,attempt=attempt.name,
        candidates=len(records),samples=len(samples),winners=len(winners),adaptive_panels=panels_count,
        adaptive_nodes=node_count,seconds=time.perf_counter()-start,
        files=[dict(path=p.name,bytes=p.stat().st_size,sha256=audit.sha(p)) for p in sorted(attempt.iterdir())])
    json_write(base/'done.json',receipt)
    return dict(scenario_id=sid,data_split=split,seconds=receipt['seconds'])


def receipt_ok(root,row,sig):
    path=root/'partitions'/row['data_split']/row['scenario_id']/'done.json'
    if not path.exists():return False
    r=json.loads(path.read_text(encoding='utf-8'))
    if r['signature']!=sig:raise ValueError('Completed partition belongs to another code/input version')
    folder=path.parent/r['attempt']
    if not all((folder/f['path']).is_file() and (folder/f['path']).stat().st_size==f['bytes'] and audit.sha(folder/f['path'])==f['sha256'] for f in r['files']):
        raise ValueError('Completed partition hash mismatch; refusing silent regeneration')
    return True


def run(mode,workers,resume):
    sig=signature();source=scenarios()
    metadata=json.loads((ROOT/'data/phase4/experiment_metadata.json').read_text(encoding='utf-8'))
    quarantine=[]
    if mode=='expand':
        selected=select_expansion();source=[r for r in source if r['scenario_id'] in selected['scenario_ids']]
        if any(r['data_split']!='train' for r in source):raise ValueError('Expanded validation must be train only')
        destination=EXP/'dataset'
    else:
        gate=json.loads((EXP/'gate.json').read_text(encoding='utf-8'))
        if gate['signature']!=sig['digest'] or not gate['generation_allowed']:raise ValueError('Expanded validation gate is not current')
        quarantine=gate['quarantined_run_ids'];destination=FULL
    statefile=destination/'generation_state.json'
    if statefile.exists():
        state=json.loads(statefile.read_text(encoding='utf-8'))
        if not resume:raise ValueError('Existing output: explicitly use --resume')
        if state['signature']!=sig:raise ValueError('Resume source/policy/input mismatch')
    else:
        if destination.exists() and any(destination.iterdir()):raise ValueError('Output is not empty')
        destination.mkdir(parents=True,exist_ok=True)
        state=dict(signature=sig,mode=mode,status='running',config=metadata['evaluation_config'],scenario_ids=[r['scenario_id'] for r in source],
            split_counts=dict(Counter(r['data_split'] for r in source)),quarantined_run_ids=quarantine,
            dataset_version=DATASET_VERSION,created_time_utc=time.time())
        json_write(statefile,state)
    pending=[r for r in source if not receipt_ok(destination,r,sig['digest'])]
    completed=len(source)-len(pending)
    print(f'{mode}: {completed}/{len(source)} partitions already verified; workers={workers}',flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(worker,(r,str(destination),metadata['evaluation_config'],mode=='expand',quarantine,sig['digest'])):r for r in pending}
        for job in as_completed(jobs):
            result=job.result();completed+=1
            if completed%10==0 or completed==len(source):
                print(f'{mode}: {completed}/{len(source)} scenario partitions complete',flush=True)
                state.update(status='running',completed=completed,last_update_utc=time.time());json_write(statefile,state)
    state.update(status='generated_pending_verification',completed=completed,last_update_utc=time.time());json_write(statefile,state)
    if mode=='expand':expansion_report(source,destination,sig)


def expansion_report(source,destination,sig):
    results=[];counts=Counter();geometry_fail=Counter();changed161=Counter();quarantine=set();maxerrors=defaultdict_float()
    for row in source:
        base=destination/'partitions'/row['data_split']/row['scenario_id']
        r=json.loads((base/'done.json').read_text(encoding='utf-8'));folder=base/r['attempt']
        with gzip.open(folder/'trajectory_metrics.jsonl.gz','rt',encoding='utf-8') as f:
            for line in f:
                metric=json.loads(line);counts[metric['candidate_status']]+=1
                for field,c in metric['geometry_checks'].items():
                    if not c['passed']:geometry_fail[field]+=1
        checks=json.loads((folder/'joint_comparisons.json').read_text(encoding='utf-8'));results.extend(checks)
        for c in checks:
            if not c['passed']:quarantine.add(c['run_id'])
            maxerrors[c['field']]=max(maxerrors[c['field']],c['relative_error'] or 0.)
    json_write(EXP/'joint_comparison_results.json',results)
    gate=dict(signature=sig['digest'],generation_allowed=True,quarantined_run_ids=sorted(quarantine),
        scenario_count=len(source),candidates=42*len(source),candidate_status_counts=dict(counts),
        geometry_failure_counts=dict(geometry_fail),joint_comparisons=len(results),
        joint_comparison_failures=sum(not r['passed'] for r in results),max_joint_relative_error=dict(maxerrors),
        interpretation='Generation allowed with conservative per-run screening and quarantine; not universal accuracy/model readiness approval')
    json_write(EXP/'gate.json',gate)
    print(json.dumps(gate,ensure_ascii=False,indent=2),flush=True)


def defaultdict_float():
    from collections import defaultdict
    return defaultdict(float)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['expand','full'])
    p.add_argument('--workers',type=int,default=6);p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if not 1<=a.workers<=12:raise ValueError('workers must be 1..12')
    run(a.mode,a.workers,a.resume)
