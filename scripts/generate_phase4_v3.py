"""Build a fresh, train-only v3 pilot. Never overwrites/resplits v2 data."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy

import audit_phase4 as audit
from robot_planner.phase4 import _sample_arrow_schema, score_phase4_candidates, select_scenario_winners
from robot_planner.phase4_evaluation import MathematicalEvaluationConfig
from robot_planner.phase4_v3 import AdaptiveIntegralConfig, CODE_VERSION, DATASET_VERSION, evaluate_v3
from refine_phase4 import objects

ROOT=Path(__file__).resolve().parents[1]


def clean(value):
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and math.isnan(value):return None
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    return value


def emit(stream,value):
    stream.write(json.dumps(clean(value),ensure_ascii=False,allow_nan=False)+'\n')


def generate(output,limit=None):
    output=output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Output must be a new or empty directory; no overwrite/resume in pilot')
    quality=ROOT/'outputs/phase4_quality';data=ROOT/'data/phase4'
    manifest_hash=audit.sha(data/'dataset_manifest.json')
    if (quality/'train_cache_source.txt').read_text()!=manifest_hash+';python-float-roundtrip-v1':
        raise ValueError('Train cache fingerprint mismatch')
    train=audit.train_only(pd.read_parquet(quality/'train_metrics.parquet'))
    selected=sorted(pd.read_csv(quality/'selected_cases.csv').scenario_id.dropna().unique())
    if limit is not None:
        if limit<1:raise ValueError('limit must be positive')
        selected=selected[:limit]
    source=train[train.scenario_id.isin(selected)]
    if set(source.scenario_id)!=set(selected) or not source.groupby('scenario_id').size().eq(42).all():
        raise ValueError('Expected 42 train candidates per selected scenario')
    oldmeta=json.loads((data/'experiment_metadata.json').read_text(encoding='utf-8'))
    cfg=MathematicalEvaluationConfig(**oldmeta['evaluation_config']);icfg=AdaptiveIntegralConfig()
    output.mkdir(parents=True,exist_ok=True);(output/'trajectory_samples').mkdir()
    paths=[ROOT/'src/robot_planner/phase4_v3.py',ROOT/'src/robot_planner/refinement.py',
        ROOT/'src/robot_planner/phase4_evaluation.py',ROOT/'scripts/generate_phase4_v3.py',
        ROOT/'docs/phase4_v3_policy.md',quality/'train_metrics.parquet',quality/'selected_cases.csv',
        data/'experiment_metadata.json',data/'dataset_manifest.json']
    metadata=dict(dataset_version=DATASET_VERSION,code_version=CODE_VERSION,status='running',
        parent_dataset_version=oldmeta['dataset_version'],data_split='train',scenario_ids=selected,
        selection='previous 19 representative train scenarios; optional lexicographic prefix smoke',
        previous_selection_seed=audit.SEED,curve_candidates_per_scenario=42,samples_per_run=161,
        evaluation_config=asdict(cfg),adaptive_integral_config=asdict(icfg),
        calculation_methods={'E/A/J':'differential IK; QUADPACK partition + Gauss-Legendre32 weighted replay',
            'L/K_epsilon/O':'unchanged v2 trapezoid; broader accuracy not established',
            'validity':'adaptive interval bounds; unresolved is not valid'},
        hashes={str(p.relative_to(ROOT)):audit.sha(p) for p in paths},
        dependencies=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,pyarrow=pa.__version__,scipy=scipy.__version__),
        limitation='pilot only; no ML; 42-candidate optimum, not global; no physical safety or overfitting guarantee')
    audit.write_json(output/'metadata.json',metadata)
    schema=_sample_arrow_schema().append(pa.field('joint_sample_status',pa.string()))
    counts={s:0 for s in ['accepted','invalid','unresolved']};changes=[];timings=[];panel_count=0;node_count=0
    begin=time.perf_counter()
    with (output/'trajectory_metrics.jsonl').open('w',encoding='utf-8') as mf, \
         (output/'adaptive_panels.jsonl').open('w',encoding='utf-8') as pf, \
         (output/'scenario_winners.jsonl').open('w',encoding='utf-8') as wf:
        for sid in selected:
            start=time.perf_counter();newrows=[];samplerows=[]
            group=source[source.scenario_id.eq(sid)].sort_values('run_id')
            for row in group.to_dict('records'):
                arm,curve,obstacles=objects(row)
                result=evaluate_v3(arm,curve,row['elbow_mode'],obstacles=obstacles,config=cfg,samples=161,integral_config=icfg)
                record={**clean(row),**result['metrics'],'curve_name_ko':curve.display_name,
                    'legacy_v2':result['legacy_v2'],'path_check':result['path_check'],
                    'joint_sample_grid_integrals':result['joint_sample_grid_integrals'],
                    'quadrature_diagnostics':result['quadrature']['diagnostics'],
                    'quadrature_evaluations':result['quadrature']['evaluations']}
                newrows.append(record);counts[record['candidate_status']]+=1
                identity={k:row[k] for k in ('run_id','scenario_id','data_split','dataset_partition','curve_key','shape','elbow_mode')}
                samplerows.extend({**identity,**sample} for sample in result['sample_rows'])
                for panel in result['quadrature']['panels']:
                    emit(pf,{**identity,'integration_status':result['quadrature']['status'],**panel})
                    panel_count+=1;node_count+=len(panel['time_s'])
            scored=score_phase4_candidates(newrows)
            for row in scored:emit(mf,row)
            winners=select_scenario_winners(scored)
            unresolved=any(r['candidate_status']=='unresolved' for r in newrows)
            for winner in winners:
                winner.update(dataset_version=DATASET_VERSION,label_scope='42 prepared candidates',
                    label_readiness='provisional_unresolved_candidates' if unresolved else 'pilot_only_not_training_approved')
                emit(wf,winner)
            baseline=audit.winner_rows(group).set_index('objective')
            for winner in winners:
                old=baseline.loc[winner['objective']]
                changes.append(dict(scenario_id=sid,objective=winner['objective'],
                    old_winner=old.winner_run_id,new_winner=winner['winner_run_id'],
                    winner_changed=(old.winner_run_id or None)!=(winner['winner_run_id'] or None),
                    tie_set_changed=json.loads(old.tied_run_ids_json)!=json.loads(winner['tied_run_ids_json']),
                    label_readiness=winner['label_readiness']))
            pq.write_table(pa.Table.from_pylist(samplerows,schema=schema),output/'trajectory_samples'/f'{sid}.parquet',compression='zstd')
            timings.append(dict(scenario_id=sid,seconds=time.perf_counter()-start))
            print(f'{sid}: {len(scored)} candidates, {timings[-1]["seconds"]:.2f}s',flush=True)
    metadata.update(status='complete',scenario_count=len(selected),run_count=len(source),
        trajectory_sample_count=len(source)*161,winner_count=len(selected)*6,
        adaptive_panel_count=panel_count,adaptive_node_count=node_count,status_counts=counts,
        elapsed_seconds=time.perf_counter()-begin,scenario_timings=timings)
    audit.write_json(output/'metadata.json',metadata)
    audit.write_json(output/'label_changes.json',changes)
    audit.write_json(output/'data_dictionary.json',{
        'trajectory_metrics.jsonl':'one candidate; equations and v3 scores; legacy_v2 isolated audit values',
        'trajectory_samples/*.parquet':'161 time samples/run; theta and differential derivatives; not adaptive integral nodes',
        'adaptive_panels.jsonl':'final adaptive panels: sum(weights_s*integrand) reproduces accepted joint costs',
        'scenario_winners.jsonl':'v3 pilot-only labels; unresolved alternatives explicitly marked',
        'candidate_status':'accepted/invalid/unresolved; accepted = bounded_clear and all joint integrations accepted',
        'fully_reachable/collision_free':'tri-state continuous diagnostic: null when not established; see witness details',
        'joint_derivative_reliable':'legacy sampled singularity flag, not accuracy certification',
        'joint_sample_grid_integrals':'diagnostic trapezoid of 161 differential samples, NOT final score',
        'units':{'joint_velocity_effort':'rad^2/s','joint_acceleration_effort':'rad^2/s^3','joint_jerk_effort':'rad^2/s^5',
                 'time_s':'s','weights_s':'s','L':'workspace unit','K_epsilon':'1/workspace unit','O':'s/workspace unit^2'}})
    audit.write_json(output/'manifest.json',dict(dataset_version=DATASET_VERSION,files=[
        dict(path=p.relative_to(output).as_posix(),bytes=p.stat().st_size,sha256=audit.sha(p))
        for p in sorted(output.rglob('*')) if p.is_file() and p.name!='manifest.json']))
    return metadata


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'data/phase4_v3_pilot')
    parser.add_argument('--limit-scenarios',type=int)
    args=parser.parse_args()
    print(json.dumps(generate(args.output,args.limit_scenarios),ensure_ascii=False,indent=2))
