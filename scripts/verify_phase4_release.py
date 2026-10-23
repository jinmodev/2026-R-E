"""Streaming mechanical verification; numeric replay restricted to train."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import gzip
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import platform

import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import scipy
import audit_phase4 as audit
from phase4_release_pipeline import EXP,FULL,ROOT,signature,scenarios
from robot_planner.phase4 import score_phase4_candidates,select_scenario_winners
from robot_planner.experiments import curve_variants


def check_partition(payload):
    root,source,sig=payload;root=Path(root);sid=source['scenario_id'];split=source['data_split']
    base=root/'partitions'/split/sid
    receipt=json.loads((base/'done.json').read_text(encoding='utf-8'));folder=base/receipt['attempt'];errors=[]
    def check(ok,name):
        if not ok:errors.append(name)
    check(receipt['signature']==sig,'signature')
    for f in receipt['files']:
        p=folder/f['path'];check(p.stat().st_size==f['bytes'] and audit.sha(p)==f['sha256'],'hash:'+f['path'])
    with gzip.open(folder/'trajectory_metrics.jsonl.gz','rt',encoding='utf-8') as f:records=[json.loads(line) for line in f]
    ids=[r['run_id'] for r in records];check(len(ids)==len(set(ids))==42,'candidate IDs/count')
    expected_ids={f'{sid}_{v.variant_id}_{branch}' for v in curve_variants() for branch in ('up','down')}
    check(set(ids)==expected_ids,'expected candidate grid')
    check(all(all(r.get(k)==v for k,v in source.items()) for r in records),'source inputs preserved')
    check(all(r['scenario_id']==sid and r['data_split']==split for r in records),'candidate join/split')
    check(all(r['dataset_version']=='phase4-mathematics-v3' for r in records),'version')
    check(all(all(r.get(k) for k in ('theta1_t_equation','theta2_t_equation','x_t_equation','y_t_equation')) for r in records),'equation schema')
    winners=json.loads((folder/'scenario_winners.json').read_text(encoding='utf-8'))
    check(len(winners)==6 and len({w['objective'] for w in winners})==6,'winner schema/count')
    check(all(w['scenario_id']==sid and w['data_split']==split and (w['winner_run_id'] is None or w['winner_run_id'] in ids) for w in winners),'winner foreign keys')
    cols=['run_id','scenario_id','data_split','sample_index']
    if split=='train':cols+=['theta1_rad','theta2_rad','x','y','time_s']
    table=pq.read_table(folder/'trajectory_samples.parquet',columns=cols)
    check(table.num_rows==42*161,'sample row count')
    check(np.array_equal(table['run_id'].to_numpy(),np.repeat(ids,161)),'sample run order')
    check(np.array_equal(table['sample_index'].to_numpy(),np.tile(np.arange(161),42)),'sample index unique/count')
    check(all(v==sid for v in table['scenario_id'].unique().to_pylist()) and all(v==split for v in table['data_split'].unique().to_pylist()),'sample join/split')
    report=dict(scenario_id=sid,data_split=split,errors=errors,candidates=42,samples=table.num_rows,winners=len(winners),
        files=[dict(path=str((folder/f['path']).relative_to(root)).replace('\\','/'),bytes=f['bytes'],sha256=f['sha256']) for f in receipt['files']])
    # Never compute/disclose held-out value distributions, ranks or numerical diagnostics.
    if split!='train':return report
    a,b,x,y=(table[c].to_numpy() for c in ('theta1_rad','theta2_rad','x','y'))
    valid=np.isfinite(a)&np.isfinite(b)
    fx=source['base_x']+source['link1']*np.cos(a)+source['link2']*np.cos(a+b)
    fy=source['base_y']+source['link1']*np.sin(a)+source['link2']*np.sin(a+b)
    fk=float(np.max(np.hypot(fx[valid]-x[valid],fy[valid]-y[valid]),initial=0))
    check(fk<1e-9,'train FK')
    byid={r['run_id']:r for r in records};sums={};panel_count=0;node_count=0
    with gzip.open(folder/'adaptive_panels.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            p=json.loads(line);panel_count+=1;node_count+=len(p['time_s'])
            check(p['run_id'] in byid and p['scenario_id']==sid and p['data_split']=='train','panel key')
            check(len(p['weights_s'])==len(p['time_s'])==len(p['integrand'])==32,'panel shape')
            value=math.fsum(w*v for w,v in zip(p['weights_s'],p['integrand']))
            check(value==p['integral'],'panel weighted replay')
            if p['integration_status']=='accepted':sums.setdefault((p['run_id'],p['metric']),[]).append(value)
    check(panel_count==receipt['adaptive_panels'] and node_count==receipt['adaptive_nodes'],'panel/node count')
    replaymax=0
    for r in records:
        if r['selection_valid']:
            check(r['candidate_status']=='accepted' and r['path_check_status']=='bounded_clear' and r['joint_integration_status']=='accepted','train eligibility')
            check(all(c['passed'] for c in r['geometry_checks'].values()),'train convergence gate')
        if r['joint_integration_status']=='accepted':
            for field in ('joint_velocity_effort','joint_acceleration_effort','joint_jerk_effort'):
                value=math.fsum(sums.get((r['run_id'],field),[]));err=abs(value-r[field]);replaymax=max(replaymax,err)
                check(err==0,'train summary replay')
    recomputed=score_phase4_candidates(records)
    check(all(r['balanced_cost_0_100']==q['balanced_cost_0_100'] for r,q in zip(records,recomputed)),'train score replay')
    expected=select_scenario_winners(recomputed)
    check(all(a['winner_run_id']==b['winner_run_id'] and a['tied_run_ids_json']==b['tied_run_ids_json'] for a,b in zip(winners,expected)),'train label replay')
    report.update(train_status_counts=dict(Counter(r['candidate_status'] for r in records)),train_fk_max=fk,train_replay_max=replaymax)
    return report


def verify(mode,workers):
    root=EXP/'dataset' if mode=='expand' else FULL
    out=EXP/'verification' if mode=='expand' else ROOT/'outputs/phase4_v3_release_quality'
    out.mkdir(parents=True,exist_ok=True);sig=signature()
    audit.write_json(out/'verification_environment.json',dict(
        python=sys.version,executable=sys.executable,platform=platform.platform(),
        numpy=np.__version__,pyarrow=pa.__version__,scipy=scipy.__version__,
        note='Environment observed during final verification; preserve these versions for reproduction/resume'))
    state=json.loads((root/'generation_state.json').read_text(encoding='utf-8'))
    if state['signature']!=sig:raise ValueError('Source/policy fingerprint changed')
    source=[r for r in scenarios() if r['scenario_id'] in state['scenario_ids']]
    expected_ids=({r['scenario_id'] for r in scenarios()} if mode=='full' else
                  set(json.loads((EXP/'selection.json').read_text(encoding='utf-8'))['scenario_ids']))
    if set(state['scenario_ids'])!=expected_ids or len(state['scenario_ids'])!=len(expected_ids):
        raise ValueError('Missing/duplicate/extra scenario IDs in generation state')
    if dict(Counter(r['data_split'] for r in source))!=state['split_counts']:
        raise ValueError('Split counts do not match source')
    if len(list((root/'partitions').glob('*/*/done.json')))!=len(source):
        raise ValueError('Missing/extra completed partition markers')
    begin=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(check_partition,(str(root),r,sig['digest'])) for r in source]
        for job in as_completed(jobs):
            results.append(job.result())
            if len(results)%100==0:print(f'verify {len(results)}/{len(source)} partitions',flush=True)
    totals={}
    for split in ('train','validation','test','challenge'):
        subset=[r for r in results if r['data_split']==split]
        totals[split]=dict(scenarios=len(subset),candidates=sum(r['candidates'] for r in subset),samples=sum(r['samples'] for r in subset),winners=sum(r['winners'] for r in subset))
    failures=[dict(scenario_id=r['scenario_id'],errors=r['errors']) for r in results if r['errors']]
    original=json.loads((ROOT/'outputs/phase4_quality/output_manifest.json').read_text(encoding='utf-8'))
    preserved=all((ROOT/f['path']).stat().st_size==f['bytes'] and audit.sha(ROOT/f['path'])==f['sha256'] for f in original['outputs'])
    prior=json.loads((ROOT/'data/phase4_v3_pilot/manifest.json').read_text(encoding='utf-8'))
    pilot_preserved=all(audit.sha(ROOT/'data/phase4_v3_pilot'/f['path'])==f['sha256'] for f in prior['files'])
    if not preserved or not pilot_preserved:failures.append(dict(scenario_id='preservation',errors=['old artifacts modified']))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
    (out/'unit_test_log.txt').write_text(tests.stdout+tests.stderr,encoding='utf-8')
    if tests.returncode:failures.append(dict(scenario_id='tests',errors=['regression failure']))
    train_counts=Counter()
    for r in results:train_counts.update(r.get('train_status_counts',{}))
    summary=dict(mode=mode,signature=sig['digest'],partition_count=len(results),split_counts=totals,
        failed_partitions=failures,train_status_counts=dict(train_counts),previous_analysis_preserved=preserved,pilot_preserved=pilot_preserved,
        train_fk_max=max((r.get('train_fk_max',0) for r in results),default=0),
        train_replay_max=max((r.get('train_replay_max',0) for r in results),default=0),
        elapsed_seconds=time.perf_counter()-begin,regression_tests=tests.stderr.splitlines()[-4:],
        heldout_policy='only schema/IDs/row counts/partition/hash checks; no outcome statistics or rank analysis')
    audit.write_json(out/'verification.json',summary)
    audit.write_json(root/'dataset_manifest.json',dict(dataset_version=state['dataset_version'],signature=sig,
        split_counts=totals,files=[f for r in sorted(results,key=lambda r:r['scenario_id']) for f in r['files']]))
    state.update(status='verified_complete' if not failures else 'verification_failed',verified_time_utc=time.time())
    audit.write_json(root/'generation_state.json',state)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['expand','full']);p.add_argument('--workers',type=int,default=6)
    a=p.parse_args();verify(a.mode,a.workers)
