"""Read-only Phase-4 quality audit and train-only research assets.

Run with Python >=3.11 and project [data,analysis] dependencies. Source datasets
are never written. See docs/phase4_quality_analysis_protocol.md for fixed rules.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robot_planner.curves import make_curve
from robot_planner.evaluation import CircleObstacle
from robot_planner.kinematics import PlanarArm2R
from robot_planner.phase4 import RAW_OBJECTIVES, score_phase4_candidates, _winner_key, _sample_arrow_schema
from robot_planner.phase4_evaluation import MathematicalEvaluationConfig, simulate_trajectory, numerical_derivative

SEED = 20260908
FIELDS = {"L": "path_length_units", "E_theta": "joint_velocity_effort",
          "A": "joint_acceleration_effort", "J": "joint_jerk_effort",
          "K_epsilon": "regularized_curvature_energy_inv_unit", "O": "obstacle_penalty_s_per_unit2"}
TOL = dict(zip(FIELDS, [(1e-6,.005),(1e-6,.01),(1e-5,.02),(1e-3,.05),(1e-6,.02),(1e-4,.05)]))
OBJECTIVES = {**RAW_OBJECTIVES, "balanced": "balanced_cost_0_100"}
FAMILIES = ["line", "arc", "bezier", "hermite", "cycloid", "limacon"]
NAMES = dict(zip(FAMILIES, ["직선", "원호", "베지어", "에르미트", "사이클로이드", "리마송"]))
COLORS = dict(zip(FAMILIES, ["#3264D8", "#D78222", "#259067", "#CB5367", "#8864BB", "#189DAA"]))
BOOLS = ["fully_reachable", "time_grid_fully_reachable", "validation_grid_fully_reachable", "collision_free", "selection_valid", "joint_derivative_reliable", "path_regular"]
INPUTS = ["base_x","base_y","link1","link2","start_x","start_y","goal_x","goal_y","shape","duration_s","samples","obstacle_count","link_ratio","chord_length","start_radius","goal_radius"]
NUMS = list(dict.fromkeys([*INPUTS,*FIELDS.values(),"min_obstacle_clearance_units","min_singularity_index","unregularized_curvature_energy_inv_unit","curvature_speed_epsilon_units_per_u",*['normalized_'+x for x in RAW_OBJECTIVES],"balanced_cost_0_100","balanced_reward_0_100"]))
TEXT = ["run_id","scenario_id","data_split","dataset_partition","curve_key","elbow_mode","obstacles_json","curve_parameters_json","obstacle_layout","scenario_type"]

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    def clean(value):
        if isinstance(value,np.generic):return clean(value.item())
        if isinstance(value,float) and math.isnan(value):return None
        if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [clean(v) for v in value]
        return value
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def table(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = data if isinstance(data,pd.DataFrame) else pd.DataFrame(data)
    df.to_csv(path,index=False,encoding="utf-8-sig")
    return df

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def train_only(frame):
    if not frame.data_split.eq("train").all():
        raise ValueError("Research functions accept train only")
    return frame

def read_train(path, columns=None):
    # Parse strings first; held-out outcomes are never converted, aggregated or emitted.
    parts=[]
    for c in pd.read_csv(path, dtype=str, usecols=columns, chunksize=8192, keep_default_na=False):
        parts.append(c.loc[c.data_split.eq('train')].copy())
    result=pd.concat(parts,ignore_index=True)
    train_only(result)
    for c in set(BOOLS)&set(result):
        result[c]=result[c].map({'True':True,'False':False,'':None})
    for c in set(NUMS)&set(result):
        # Python's correctly rounded float parser preserves the generator's
        # binary64 values. pandas.to_numeric can change the last bit and thus
        # the deterministic ordering within an otherwise unchanged tie set.
        result[c]=result[c].map(lambda value: float(value) if value else np.nan)
    return result

def load_train(data, out):
    cache=out/'train_metrics.parquet'
    fingerprint=sha(data/'dataset_manifest.json')+';python-float-roundtrip-v1'
    tag=out/'train_cache_source.txt'
    if cache.exists() and tag.exists() and tag.read_text()==fingerprint:
        return train_only(pd.read_parquet(cache))
    m=read_train(data/'trajectory_metrics.csv',TEXT+NUMS+BOOLS)
    m.to_parquet(cache,index=False)
    tag.write_text(fingerprint)
    return m

def mechanical(data, out):
    meta=json.loads((data/'experiment_metadata.json').read_text(encoding='utf-8'))
    manifest=json.loads((data/'dataset_manifest.json').read_text(encoding='utf-8'))
    checks=[]
    def check(name, bad, total, detail=''):
        checks.append(dict(check=name,failures=int(bad),examined=int(total),status='PASS' if bad==0 else 'FAIL',detail=detail))
    s=pd.read_csv(data/'scenarios.csv',usecols=['scenario_id','data_split'])
    m=pd.read_csv(data/'trajectory_metrics.csv',usecols=['run_id','scenario_id','data_split','curve_key','shape','elbow_mode','samples'])
    w=pd.read_csv(data/'scenario_winners.csv',usecols=['scenario_id','data_split','objective'])
    check('scenario_id_unique',s.scenario_id.duplicated().sum(),len(s))
    check('run_id_unique',m.run_id.duplicated().sum(),len(m))
    check('42_candidates_per_scenario',m.groupby('scenario_id').size().ne(42).sum(),len(s))
    check('candidate_identity_unique',m.duplicated(['scenario_id','curve_key','shape','elbow_mode']).sum(),len(m))
    check('six_labels_per_scenario',w.groupby('scenario_id').size().ne(6).sum(),len(s))
    check('label_objective_key_unique',w.duplicated(['scenario_id','objective']).sum(),len(w))
    check('label_objective_domain',(~w.objective.isin(OBJECTIVES)).sum(),len(w))
    smap=s.set_index('scenario_id').data_split
    for name, f in [('metrics',m),('winners',w)]:
        check(name+'_scenario_join_split',(~f.scenario_id.isin(smap.index)|f.data_split.ne(f.scenario_id.map(smap))).sum(),len(f))
    counts=s.data_split.value_counts().to_dict()
    check('metadata_split_counts',sum(counts.get(k)!=v for k,v in meta['data_split']['counts'].items()),4)
    csv_counts={'scenarios.csv':len(s),'trajectory_metrics.csv':len(m),'scenario_winners.csv':len(w)}
    entries=[]
    for i,e in enumerate(manifest['files']):
        p=data/e['path']; exists=p.is_file()
        rows=pq.ParquetFile(p).metadata.num_rows if exists and p.suffix=='.parquet' else csv_counts.get(e['path'])
        item=dict(path=e['path'],exists=exists,bytes_match=exists and p.stat().st_size==e['bytes'],hash_match=exists and sha(p)==e['sha256'],rows=rows,
                  row_match='row_count' not in e or rows==e['row_count'],metadata_row_match=e['path'] not in meta['file_row_counts'] or rows==meta['file_row_counts'][e['path']])
        entries.append(item)
        if i%100==0: print('manifest',i,'/',len(manifest['files']),flush=True)
    table(out/'file_integrity.csv',entries)
    for key in ['exists','bytes_match','hash_match','row_match','metadata_row_match']:
        check('manifest_'+key,sum(not e[key] for e in entries),len(entries))
    known={e['path'] for e in entries}
    check('unmanifested_parquet',sum(p.relative_to(data).as_posix() not in known for p in (data/'trajectory_samples').rglob('*.parquet')),len(entries))
    index=m.set_index('run_id'); sample_counts=pd.Series(0,index=index.index,dtype='int64')
    partition_rows={}; bad_join=bad_indices=bad_schema=0
    # Each physical partition is bounded (about 169k rows); read only mechanical columns.
    keys=['run_id','scenario_id','data_split','curve_key','shape','elbow_mode','sample_index']
    for i,p in enumerate(sorted((data/'trajectory_samples').rglob('*.parquet'))):
        pf=pq.ParquetFile(p)
        bad_schema+=not pf.schema_arrow.equals(_sample_arrow_schema(),check_metadata=False)
        f=pf.read(columns=keys).to_pandas()
        mapped=index.reindex(f.run_id).reset_index(drop=True)
        for c in keys[1:-1]:
            bad_join+=int(f[c].reset_index(drop=True).ne(mapped[c]).sum())
        bad_join+=int(f.data_split.ne(p.parent.name).sum())
        g=f.groupby('run_id').sample_index.agg(['size','nunique','min','max'])
        # Unique range [0,N-1] proves exact sample index set, not just row count.
        bad_indices+=int((g['size'].ne(g['nunique'])|g['min'].ne(0)|g['max'].ne(g['size']-1)).sum())
        sample_counts=sample_counts.add(g['size'],fill_value=0)
        partition_rows[p.parent.name]=partition_rows.get(p.parent.name,0)+len(f)
        if i%100==0: print('sample keys',i,flush=True)
    check('parquet_schema',bad_schema,len(entries)-5)
    check('sample_join_identity_split',bad_join,int(sample_counts.sum()))
    check('sample_index_set',bad_indices,len(m))
    check('samples_per_run',sample_counts.ne(index.samples).sum(),len(m))
    check('split_sample_counts',sum(partition_rows.get(k)!=v*42*161 for k,v in counts.items()),4)
    table(out/'mechanical_checks.csv',checks)
    table(out/'split_counts.csv',[dict(split=k,scenarios=v,candidates=v*42,samples=partition_rows[k]) for k,v in counts.items()])
    write_json(out/'input_provenance.json',dict(dataset_version=meta['dataset_version'],code_version=meta['code_version'],evaluation_config=meta['evaluation_config'],manifest_sha256=sha(data/'dataset_manifest.json'),seed=SEED,analysis_version='phase4-audit-v1',scope='train only; holdouts mechanical only'))

def train_integrity(data,out,m):
    train_only(m)
    checks=[]
    def ck(name,bad,n): checks.append(dict(check=name,failures=int(bad),examined=int(n),status='PASS' if bad==0 else 'FAIL'))
    for f in NUMS:
        v=m[f].to_numpy(dtype=float); ck('summary_infinite_'+f,np.isinf(v).sum(),len(m))
    for f in ['joint_velocity_effort','joint_acceleration_effort','joint_jerk_effort']:
        ck('missing_condition_'+f,m[f].isna().ne(~m.time_grid_fully_reachable.astype(bool)).sum(),len(m))
    ck('missing_condition_O',m[FIELDS['O']].isna().ne((m.obstacle_count>0)&~m.time_grid_fully_reachable.astype(bool)).sum(),len(m))
    ck('missing_condition_ordinary_K',m.unregularized_curvature_energy_inv_unit.isna().ne(~m.path_regular.astype(bool)).sum(),len(m))
    ck('selection_valid_logic',m.selection_valid.ne(m.fully_reachable&m.collision_free.eq(True)).sum(),len(m))
    valid=m[m.selection_valid]
    ck('valid_finite_objectives',(~np.isfinite(valid[list(FIELDS.values())].to_numpy(dtype=float))).sum(),len(valid)*6)
    records=m.astype(object).where(pd.notna(m),None).to_dict('records')
    rescored=pd.DataFrame(score_phase4_candidates(records)).set_index('run_id')
    for c in ['balanced_cost_0_100',*['normalized_'+x for x in RAW_OBJECTIVES]]:
        a=m.set_index('run_id')[c].astype(float); b=pd.to_numeric(rescored[c]); ck('score_'+c,(~np.isclose(a,b,rtol=1e-9,atol=1e-9,equal_nan=True)).sum(),len(m))
    w=read_train(data/'scenario_winners.csv', ['scenario_id','data_split','objective','winner_run_id','tied_run_ids_json','tie_count','winner_status'])
    expected=winner_rows(m)
    joined=w.merge(expected,on=['scenario_id','objective'],suffixes=('_stored','_computed'),validate='one_to_one')
    ck('winner_ID',joined.winner_run_id_stored.ne(joined.winner_run_id_computed).sum(),len(w))
    ck('winner_tie_set',sum(set(json.loads(a))!=set(json.loads(b)) for a,b in zip(joined.tied_run_ids_json_stored,joined.tied_run_ids_json_computed)),len(w))
    ck('winner_tie_count',pd.to_numeric(joined.tie_count_stored).ne(joined.tie_count_computed).sum(),len(w))
    ck('winner_status',joined.winner_status_stored.ne(joined.winner_status_computed).sum(),len(w))
    # Stream train partitions, all runs; holdout numeric fields are never read.
    mapping={FIELDS['L']:('path_speed_units_per_u','geometry_u'), FIELDS['K_epsilon']:('regularized_curvature_integrand_inv_unit','geometry_u'),
             FIELDS['E_theta']:('joint_velocity_squared_rad2_s2','time_s'),FIELDS['A']:('joint_acceleration_squared_rad2_s4','time_s'),
             FIELDS['J']:('joint_jerk_squared_rad2_s6','time_s'),FIELDS['O']:('obstacle_penalty_integrand_per_unit2','time_s')}
    cols=list(dict.fromkeys(['run_id','sample_index','theta1_rad','theta2_rad','x','y','reachable_at_sample',*[x for pair in mapping.values() for x in pair]]))
    idx=m.set_index('run_id'); results=[]; max_fk=0.; bad_fk=nonfinite=0; numeric_nulls={}; total_samples=0
    for i,p in enumerate(sorted((data/'trajectory_samples'/'train').glob('*.parquet'))):
        f=pq.read_table(p,columns=cols).to_pandas().sort_values(['run_id','sample_index'])
        numeric=f.select_dtypes('number'); nonfinite+=int(np.isinf(numeric.to_numpy()).sum())
        for c in numeric: numeric_nulls[c]=numeric_nulls.get(c,0)+int(numeric[c].isna().sum())
        total_samples+=len(f)
        k=idx.reindex(f.run_id); th1=f.theta1_rad.to_numpy(); th2=f.theta2_rad.to_numpy()
        fx=k.base_x.to_numpy()+k.link1.to_numpy()*np.cos(th1)+k.link2.to_numpy()*np.cos(th1+th2)
        fy=k.base_y.to_numpy()+k.link1.to_numpy()*np.sin(th1)+k.link2.to_numpy()*np.sin(th1+th2)
        err=np.hypot(fx-f.x.to_numpy(),fy-f.y.to_numpy()); bad_fk+=int(np.sum(err>1e-9))
        if np.isfinite(err).any(): max_fk=max(max_fk,float(np.nanmax(err)))
        gr=f.groupby('run_id',sort=False)
        vals=pd.DataFrame(index=gr.size().index)
        for field,(integrand,coord) in mapping.items():
            prev=gr[integrand].shift(); delta=gr[coord].diff()
            pieces=(f[integrand]+prev)*delta*.5
            sums=pieces.groupby(f.run_id).sum()
            complete=gr[integrand].count().eq(gr.size())
            vals[field]=sums.where(complete)
        vals['max_fk_error']=pd.Series(err,index=f.index).groupby(f.run_id).max()
        results.append(vals.reset_index())
        if i%40==0: print('train reintegration',i,flush=True)
    r=pd.concat(results,ignore_index=True).set_index('run_id')
    ck('sample_numeric_infinite',nonfinite,total_samples)
    ck('stored_theta_FK',bad_fk,total_samples)
    table(out/'sample_null_counts.csv',[dict(field=k,null_rows=v,total_train_sample_rows=total_samples) for k,v in numeric_nulls.items()])
    details=[]
    for field in mapping:
        a=idx.loc[r.index,field].astype(float); b=r[field]; diff=abs(a-b)
        bad=~np.isclose(a,b,rtol=1e-9,atol=1e-9,equal_nan=True)
        ck('reintegrate_'+field,bad.sum(),len(r))
        details.append(dict(metric=field,runs=len(r),failures=int(bad.sum()),max_absolute_error=float(diff.max()),max_fk_error=max_fk))
    table(out/'reintegration_summary.csv',details)
    table(out/'train_integrity_checks.csv',checks)
    expected.to_parquet(out/'train_winners_recomputed.parquet',index=False)

def winner_rows(m):
    train_only(m); rows=[]
    for sid,g in m.groupby('scenario_id',sort=True):
        v=g[g.selection_valid.eq(True)]
        for obj,field in OBJECTIVES.items():
            if v.empty:
                rows.append(dict(scenario_id=sid,objective=obj,winner_run_id='',curve_key='',tie_count=0,tied_run_ids_json='[]',winner_status='no_valid_candidate')); continue
            best=v[field].min(); tied=v[np.isclose(v[field],best,rtol=1e-12,atol=1e-12)]
            # numpy isclose adds tolerances; use math.isclose exactly as generator.
            tied=tied[tied[field].map(lambda x: math.isclose(x,best,rel_tol=1e-12,abs_tol=1e-12))]
            row=min(tied.to_dict('records'),key=lambda x:_winner_key(x,field))
            rows.append(dict(scenario_id=sid,objective=obj,winner_run_id=row['run_id'],curve_key=row['curve_key'],tie_count=len(tied),tied_run_ids_json=json.dumps(sorted(tied.run_id)),winner_status='tie_resolved' if len(tied)>1 else 'ok'))
    return pd.DataFrame(rows)

def sample_null_audit(data,out,m):
    """Check all train sample columns, including expected per-row missingness."""
    train_only(m); idx=m.set_index('run_id'); counters={}; total=0
    for i,p in enumerate(sorted((data/'trajectory_samples'/'train').glob('*.parquet'))):
        f=pq.read_table(p).to_pandas(); total+=len(f)
        k=idx.reindex(f.run_id).reset_index(drop=True)
        timed=k.time_grid_fully_reachable.eq(True); has=k.obstacle_count.gt(0)
        for c in f:
            expected=pd.Series(False,index=f.index)
            if c in ['theta1_rad','theta2_rad','theta1_deg','theta2_deg']:expected=~f.reachable_at_sample
            elif c.startswith(('theta1_velocity','theta2_velocity','theta1_acceleration','theta2_acceleration','theta1_jerk','theta2_jerk','joint_velocity_','joint_acceleration_','joint_jerk_')):expected=~timed
            elif c=='curvature_inv_unit':expected=f.path_speed_units_per_u.le(f.curvature_speed_epsilon_units_per_u)
            elif c=='obstacle_clearance':expected=~(has&timed)
            elif c=='validation_obstacle_clearance':expected=~(has&k.fully_reachable.eq(True))
            elif c=='obstacle_penalty_integrand_per_unit2':expected=has&~timed
            counter=counters.setdefault(c,{'field':c,'missing':0,'unexpected_missing_or_present':0,'infinite':0})
            counter['missing']+=int(f[c].isna().sum())
            counter['unexpected_missing_or_present']+=int(f[c].isna().ne(expected).sum())
            if pd.api.types.is_numeric_dtype(f[c]):counter['infinite']+=int(np.isinf(f[c].to_numpy(dtype=float)).sum())
        if i%50==0:print('train all-column null checks',i,flush=True)
    for c in counters.values():c['train_sample_rows']=total;c['status']='PASS' if c['unexpected_missing_or_present']==c['infinite']==0 else 'FAIL'
    return table(out/'all_sample_column_checks.csv',list(counters.values()))

def stability(a,b,c,label):
    if not all(pd.notna(v) and math.isfinite(v) for v in [a,b,c]): return None
    atol,rtol=TOL[label]
    return abs(a-c)<=atol+rtol*abs(c) and abs(b-c)<=atol+rtol*abs(c)

def simulate(row,n,config):
    arm=PlanarArm2R(row['link1'],row['link2'],(row['base_x'],row['base_y']))
    curve=make_curve(row['curve_key'],(row['start_x'],row['start_y']),(row['goal_x'],row['goal_y']),shape=row['shape'],samples=n)
    # Guard against rebuilding a different parameterization from the stored inputs.
    stored=json.loads(row['curve_parameters_json'])
    for k,v in stored.items():
        if isinstance(v,(int,float)) and not math.isclose(v,curve.parameters[k],rel_tol=1e-10,abs_tol=1e-10):
            raise ValueError('Curve parameter reconstruction mismatch: '+k)
    obstacles=tuple(CircleObstacle(**o) for o in json.loads(row['obstacles_json']))
    result=simulate_trajectory(arm,curve,row['elbow_mode'],obstacles=obstacles,config=config,samples=n)
    return result,curve

def convergence(data,out,m):
    train_only(m); rng=np.random.default_rng(SEED); chosen=[]
    stable=m.selection_valid.eq(True)&m.joint_derivative_reliable.eq(True)&((m.obstacle_count==0)|(m.min_obstacle_clearance_units>.2))
    pools={**{'stable_'+k:m[stable&m.curve_key.eq(k)] for k in FAMILIES},
        'no_obstacle':m[m.obstacle_count.eq(0)],'with_obstacle':m[m.obstacle_count.gt(0)],
        'small_clearance':m[m.min_obstacle_clearance_units.between(0,.05,inclusive='right')&m.selection_valid],
        'near_singularity':m[m.min_singularity_index.lt(.05)],
        'reach_boundary':m[((m.link1+m.link2-np.maximum(m.start_radius,m.goal_radius))/(m.link1+m.link2)).lt(.02)],
        'unreachable':m[m.fully_reachable.eq(False)], 'reachable_collision':m[m.fully_reachable.eq(True)&m.collision_free.eq(False)]}
    for reason,pool in pools.items():
        if pool.empty: chosen.append(dict(reason=reason,scenario_id='',run_id='',available=0)); continue
        row=pool.sort_values('run_id').iloc[int(rng.integers(len(pool)))]
        chosen.append(dict(reason=reason,scenario_id=row.scenario_id,run_id=row.run_id,available=len(pool)))
    for sid in rng.choice(sorted(m.scenario_id.unique()),size=6,replace=False):
        chosen.append(dict(reason='random_scenario',scenario_id=sid,run_id='',available=m.scenario_id.nunique()))
    table(out/'selected_cases.csv',chosen)
    ids=sorted({c['scenario_id'] for c in chosen if c['scenario_id']})
    meta=json.loads((data/'experiment_metadata.json').read_text(encoding='utf-8'))
    config=MathematicalEvaluationConfig(**meta['evaluation_config'])
    rows=[]; analytic=[]; repro=[]
    for j,sid in enumerate(ids):
        source=m[m.scenario_id.eq(sid)]
        for row in source.to_dict('records'):
            for n in (81,161,321,641):
                result,curve=simulate(row,n,config)
                rows.append({**{k:row[k] for k in TEXT+INPUTS},**{k:v for k,v in result.metrics.items() if k in NUMS+BOOLS},'samples':n})
                if n==161:
                    for label,field in FIELDS.items():
                        a=row[field]; b=result.metrics[field]
                        repro.append(dict(run_id=row['run_id'],metric=label,pass_reproduction=(pd.isna(a) and b is None) or (pd.notna(a) and b is not None and math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9))))
                if curve.key in ['line','arc']:
                    exact={'L':math.hypot(curve.end[0]-curve.start[0],curve.end[1]-curve.start[1])}
                    if curve.key=='arc':
                        radius=curve.parameters['radius']; sweep=abs(curve.parameters['theta_delta']); speed=radius*sweep
                        exact={'L':speed,'K':sweep/radius,'K_epsilon':sweep/radius*(speed**2/(speed**2+result.metrics['curvature_speed_epsilon_units_per_u']**2))**2.5}
                    for label,truth in exact.items():
                        value=result.metrics['unregularized_curvature_energy_inv_unit' if label=='K' else FIELDS[label]]
                        analytic.append(dict(run_id=row['run_id'],curve_key=curve.key,samples=n,metric=label,value=value,analytic=truth,absolute_error=abs(value-truth),passed=math.isclose(value,truth,rel_tol=1e-9,abs_tol=1e-9)))
        print('convergence scenario',j+1,'/',len(ids),sid,flush=True)
    allrows=pd.DataFrame(rows); allrows.to_parquet(out/'recomputed_metrics.parquet',index=False)
    table(out/'analytic_checks.csv',analytic); table(out/'original_161_reproduction.csv',repro)
    detail=[]
    for rid,g in allrows.groupby('run_id'):
        g=g.set_index('samples'); flags=['fully_reachable','collision_free','selection_valid']
        change=any(g.loc[[161,321,641],f].fillna('NA').nunique()>1 for f in flags)
        for label,field in FIELDS.items():
            a,b,c=(g.loc[n,field] for n in [161,321,641]); ok=stability(a,b,c,label)
            detail.append(dict(run_id=rid,scenario_id=g.iloc[0].scenario_id,curve_key=g.iloc[0].curve_key,metric=label,value81=g.loc[81,field],value161=a,value321=b,value641=c,
                absolute_error161=abs(a-c) if pd.notna(a) and pd.notna(c) else np.nan,relative_error161=abs(a-c)/abs(c) if pd.notna(a) and pd.notna(c) and c!=0 else np.nan,
                converged=ok,hard_status_changed=change,joint_derivative_reliable161=g.loc[161,'joint_derivative_reliable'],selection_valid161=g.loc[161,'selection_valid']))
    detail=table(out/'convergence_detail.csv',detail)
    summary=[]
    for label,g in detail.groupby('metric',sort=False):
        ev=g[g.converged.notna()]
        summary.append(dict(metric=label,runs=len(g),evaluable=len(ev),converged=int(ev.converged.eq(True).sum()),not_converged=int(ev.converged.eq(False).sum()),unavailable=int(g.converged.isna().sum()),hard_changes=int(g.hard_status_changed.sum()),max_absolute_error=float(ev.absolute_error161.max()),max_relative_error=float(ev.relative_error161.max())))
    table(out/'convergence_summary.csv',summary)
    ranks=[]
    for sid,g in allrows.groupby('scenario_id'):
        scores={n:pd.DataFrame(score_phase4_candidates(g[g.samples.eq(n)].astype(object).where(pd.notna(g[g.samples.eq(n)]),None).to_dict('records'))) for n in [81,161,321,641]}
        winners={n:winner_rows(v).set_index('objective') for n,v in scores.items()}
        ds=detail[detail.scenario_id.eq(sid)]
        for obj,field in OBJECTIVES.items():
            for n in [81,161,321]:
                a=scores[n]; b=scores[641]
                both=a[a.selection_valid][['run_id',field]].merge(b[b.selection_valid][['run_id',field]],on='run_id',suffixes=('_n','_ref'))
                ra=both[field+'_n'].rank(); rb=both[field+'_ref'].rank()
                rho=float(ra.corr(rb)) if len(both)>1 and ra.nunique()>1 and rb.nunique()>1 else np.nan
                labels=list(FIELDS) if obj=='balanced' else [next(k for k,v in FIELDS.items() if v==field)]
                # Require every candidate eligible at either high-resolution grid to pass this objective.
                eligible=set(scores[161].loc[scores[161].selection_valid,'run_id'])|set(b.loc[b.selection_valid,'run_id'])
                relevant=ds[ds.run_id.isin(eligible)&ds.metric.isin(labels)]
                certified=bool(eligible) and relevant.converged.eq(True).all() and not ds.hard_status_changed.any()
                wa=winners[n].loc[obj]; wb=winners[641].loc[obj]
                ranks.append(dict(scenario_id=sid,objective=obj,samples=n,common_valid=len(both),rank_correlation=rho,winner_n=wa.winner_run_id,winner641=wb.winner_run_id,winner_same=wa.winner_run_id==wb.winner_run_id,tie_set_same=wa.tied_run_ids_json==wb.tied_run_ids_json,objective_verified=certified))
    table(out/'rank_stability.csv',ranks)
    poly=[]
    for n in [81,161,321,641]:
        t=np.linspace(0,3,n); y=2*t**5-3*t**3+7*t-4
        for order,truth,tol in [(1,10*t**4-9*t**2+7,1e-7),(2,40*t**3-18*t,1e-5),(3,120*t**2-18,1e-2)]:
            vals=numerical_derivative(y.tolist(),3/(n-1),order)
            err=float(np.max(abs(np.array(vals)-truth)))
            poly.append(dict(samples=n,order=order,max_absolute_error=err,absolute_tolerance=tol,passed=err<=tol))
    table(out/'polynomial_derivative_checks.csv',poly)

def no_valid_reason(g):
    if g.selection_valid.eq(True).any(): return 'has_valid'
    reach=g.fully_reachable.eq(True); collision=reach&g.collision_free.eq(False)
    if (~reach).all(): return 'all_unreachable'
    if collision.all(): return 'all_reachable_colliding'
    if (reach==collision).all() and (~reach).any() and collision.any(): return 'mixed_unreachable_collision'
    return 'other_calculation_or_schema'

def analysis(data,q,out,m):
    train_only(m)
    s=read_train(data/'scenarios.csv')
    for c in ['angular_separation_deg','max_reach','min_reach']: s[c]=pd.to_numeric(s[c])
    # Stored link_ratio is min(L1,L2)/max(L1,L2), not the ordered L1/L2.
    s['link1_over_link2']=s.link1/s.link2
    s['ratio_bin']=pd.cut(s.link1_over_link2,[-np.inf,.75,1.3333333333,np.inf],right=False).astype(str)
    s['motion_bin']=pd.cut(s.chord_length/(s.link1+s.link2),[0,.5,1,2.01],right=False).astype(str)
    s['angle_bin']=pd.cut(s.angular_separation_deg,[0,60,120,181],right=False).astype(str)
    for c in ['start','goal']: s[c+'_radius_bin']=pd.cut(s[c+'_radius']/(s.link1+s.link2),[0,.4,.8,1.01],right=False).astype(str)
    composition=[]
    conditions=['scenario_type','obstacle_layout','obstacle_count','ratio_bin','motion_bin','angle_bin','start_radius_bin','goal_radius_bin']
    for c in conditions:
        for k,n in s[c].value_counts().sort_index().items(): composition.append(dict(variable=c,category=str(k),scenarios=n,total_scenarios=len(s),fraction=n/len(s),small_n=n<30))
    table(out/'composition.csv',composition)
    table(out/'geometry_summary.csv',s[['link1','link2','link_ratio','start_radius','goal_radius','chord_length','angular_separation_deg']].describe().reset_index(names='statistic'))
    reasons=pd.DataFrame([dict(scenario_id=sid,reason=no_valid_reason(g),unreachable=int(g.fully_reachable.eq(False).sum()),reachable_collision=int((g.fully_reachable.eq(True)&g.collision_free.eq(False)).sum()),valid=int(g.selection_valid.sum())) for sid,g in m.groupby('scenario_id')])
    table(out/'scenario_exclusions.csv',reasons)
    table(out/'no_valid_summary.csv',reasons.groupby('reason').size().reindex(['has_valid','all_unreachable','all_reachable_colliding','mixed_unreachable_collision','other_calculation_or_schema'],fill_value=0).reset_index(name='scenarios'))
    validities=[]
    for key,g in m.groupby('curve_key'):
        reach=int(g.fully_reachable.sum()); valid=int(g.selection_valid.sum()); sn=int(g.groupby('scenario_id').selection_valid.any().sum())
        validities.append(dict(curve_key=key,curve_name=NAMES[key],variants_per_scenario=len(g)//len(s),candidates=len(g),reachable=reach,collision_free=int((g.fully_reachable&g.collision_free.eq(True)).sum()),valid=valid,scenarios=len(s),scenarios_any_valid=sn,reachable_rate=reach/len(g),conditional_collision_free_rate=valid/reach if reach else np.nan,valid_rate=valid/len(g),scenario_valid_rate=sn/len(s)))
    table(out/'validity.csv',validities)
    w=pd.read_parquet(q/'train_winners_recomputed.parquet')
    family=m.set_index('run_id').curve_key.to_dict(); winner_tables=[]
    ranks=pd.read_csv(q/'rank_stability.csv'); eligible=ranks[ranks.samples.eq(161)&ranks.objective_verified].groupby('objective').scenario_id.agg(set).to_dict()
    for scope in ['original_161_reference_only','verified_cases_only','obstacles_only_reference']:
        for obj,g in w.groupby('objective'):
            g=g[g.winner_run_id.ne('')]
            if scope=='verified_cases_only':g=g[g.scenario_id.isin(eligible.get(obj,set()))]
            if scope=='obstacles_only_reference':g=g[g.scenario_id.isin(s.loc[s.obstacle_count.gt(0),'scenario_id'])]
            fractions={k:0. for k in FAMILIES}
            for ids in g.tied_run_ids_json.map(json.loads):
                for rid in ids: fractions[family[rid]]+=1/len(ids)
            for key in FAMILIES:
                count=int(g.curve_key.eq(key).sum()); n=len(g)
                winner_tables.append(dict(scope=scope,objective=obj,curve_key=key,deterministic_wins=count,fractional_wins=fractions[key],denominator_scenarios=n,tied_scenarios=int(g.tie_count.gt(1).sum()),deterministic_rate=count/n if n else np.nan,fractional_rate=fractions[key]/n if n else np.nan))
    table(out/'winner_shares.csv',winner_tables)
    cond=[]
    for c in conditions[1:]:
        joined=m.merge(s[['scenario_id',c]].rename(columns={c:'condition'}),on='scenario_id',validate='many_to_one')
        for (b,k),g in joined.groupby(['condition','curve_key']):
            n=g.scenario_id.nunique(); success=int(g.groupby('scenario_id').selection_valid.any().sum())
            cond.append(dict(condition=c,bin=str(b),curve_key=k,scenarios=n,scenarios_any_valid=success,scenario_valid_rate=success/n,small_n=n<30))
    table(out/'conditional_validity.csv',cond)
    paired=[]
    for obj,field in OBJECTIVES.items():
        v=m[m.selection_valid]; best=v.groupby(['scenario_id','curve_key'])[field].min().unstack('curve_key')
        for a,b in itertools.combinations(FAMILIES,2):
            both=best[[a,b]].dropna(); diff=both[a]-both[b]
            paired.append(dict(objective=obj,family_a=a,family_b=b,paired_scenarios=len(both),median_a_minus_b=float(diff.median()) if len(both) else np.nan,a_lower_fraction=float((diff<0).mean()) if len(both) else np.nan,small_n=len(both)<30,status='original_161_reference_only; objective-specific family minimum'))
    table(out/'paired_objective_comparisons.csv',paired)
    # Relationships: one candidate per scenario chosen by hash of run_id, no objective selection.
    v=m[m.selection_valid].copy(); v['hash']=v.run_id.map(lambda x:hashlib.sha256((str(SEED)+x).encode()).hexdigest())
    reps=v.sort_values('hash').drop_duplicates('scenario_id').drop(columns='hash')
    table(out/'relationship_representatives.csv',reps[['scenario_id','run_id','curve_key',*FIELDS.values(),'min_obstacle_clearance_units','duration_s','link1','link2',*['normalized_'+x for x in RAW_OBJECTIVES]]])
    table(out/'analysis_flow.csv',[dict(stage='composition',included=len(s),excluded=0,reason='all train scenarios'),dict(stage='winner/relationship',included=len(reps),excluded=len(s)-len(reps),reason='no selection_valid candidate; see scenario_exclusions.csv')])
    plots(data,q,out,m,s,reps)

def plots(data,q,out,m,s,reps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts={f.name for f in font_manager.fontManager.ttflist}
    font=next((f for f in ['Malgun Gothic','Noto Sans CJK KR','NanumGothic'] if f in fonts),None)
    if not font: raise RuntimeError('A Korean font is required; install Malgun Gothic/Noto Sans CJK KR')
    plt.rcParams.update({'font.family':[font,'DejaVu Sans'],'mathtext.fontset':'dejavusans','axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'path','figure.facecolor':'white','axes.titleweight':'bold'})
    captions=[]
    def save(fig,name,caption):
        from matplotlib.ticker import FuncFormatter
        for ax in fig.axes:
            if ax.get_xscale()=='symlog':ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'))
            if ax.get_yscale()=='symlog':ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'))
        fig.savefig(out/(name+'.png'),dpi=300,bbox_inches='tight',facecolor='white'); fig.savefig(out/(name+'.svg'),bbox_inches='tight');plt.close(fig)
        captions.append(dict(figure=name,caption_ko=caption))
    f,ax=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for a,col,label in zip(ax,['link1','link_ratio','obstacle_count'],['첫째 링크 길이 (unit)','짧은 링크 / 긴 링크 (원자료 link_ratio)','장애물 수 (개)']):
        a.hist(s[col],bins=18 if col!='obstacle_count' else np.arange(s[col].max()+2)-.5,color='#3264D8',edgecolor='white');a.set(xlabel=label,ylabel='시나리오 수')
        if col=='obstacle_count':a.set_xticks(np.arange(s[col].max()+1))
    f.suptitle(f'Train 구성 · 독립 시나리오 {len(s):,}개');save(f,'01_composition',f'train {len(s):,}개 시나리오의 로봇 크기·비율·장애물 수 분포. 후보 행을 독립 표본으로 세지 않았다.')
    v=pd.read_csv(out/'validity.csv').set_index('curve_key').loc[FAMILIES]
    f,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained');x=np.arange(6)
    for offset,(col,label) in enumerate([('reachable_rate','도달 가능 / 전체 후보'),('conditional_collision_free_rate','무충돌 / 도달 가능 후보'),('valid_rate','유효 / 전체 후보')]): axes[0].bar(x+(offset-1)*.25,v[col]*100,.24,label=label)
    axes[0].set(xticks=x,xticklabels=[NAMES[k] for k in FAMILIES],ylabel='후보 단위 비율 (%)',ylim=(0,105),title='분모가 다른 세 가지 비율');axes[0].legend(fontsize=8)
    axes[1].bar(x,v.scenario_valid_rate*100,color=[COLORS[k] for k in FAMILIES]);axes[1].set(xticks=x,xticklabels=[NAMES[k] for k in FAMILIES],ylabel='시나리오 단위 비율 (%)',ylim=(0,105),title=f'유효 후보 ≥1개 / 시나리오 {len(s):,}개')
    save(f,'02_validity','왼쪽은 후보 단위 비율(직선 6,000개, 다른 종류 각각 24,000개); 오른쪽은 각 종류에 유효 후보가 하나 이상 있는 시나리오 비율. 조건부 무충돌률의 정확한 분모는 validity.csv의 reachable 열이다. 표본 검사이며 연속 안전 보증이 아니다.')
    wins=pd.read_csv(out/'winner_shares.csv'); f,ax=plt.subplots(2,3,figsize=(13,7.5),layout='constrained')
    for a,obj in zip(ax.flat,OBJECTIVES):
        z=wins[(wins.scope=='original_161_reference_only')&(wins.objective==obj)].set_index('curve_key').loc[FAMILIES]
        a.bar(x-.18,z.deterministic_rate*100,.36,color=[COLORS[k] for k in FAMILIES],label='결정적 선택')
        a.bar(x+.18,z.fractional_rate*100,.36,color=[COLORS[k] for k in FAMILIES],alpha=.38,hatch='//',label='동점 1/k 배분')
        a.set(xticks=x,xticklabels=[NAMES[k] for k in FAMILIES],ylabel='승자 비율 (%)',title=f'{obj} · n={int(z.denominator_scenarios.iloc[0]):,}');a.tick_params(axis='x',labelsize=8)
    ax.flat[0].legend(fontsize=8);f.suptitle('161표본 원자료 기술통계 — 수치 미확정 참고용, 우열 결론 금지')
    save(f,'03_winners_reference','유효 후보가 있는 train 시나리오만 분모로 한 기존 161표본 승자 집계. 연한 빗금은 동점 후보 각각 1/k씩 배분한 합계다. 직선은 후보 2개, 나머지는 8개다. safety의 무장애물 동점은 안전 우월을 뜻하지 않는다. 검증된 일부 사례 집계는 winner_shares.csv의 verified_cases_only로 분리했다.')
    f,axes=plt.subplots(1,3,figsize=(14,4.4),layout='constrained')
    pairs=[('path_length_units','joint_velocity_effort','거리 L (unit)','관절 노력도 Eθ (rad²/s)'),('joint_jerk_effort','regularized_curvature_energy_inv_unit','저크 J (rad²/s⁵)','정칙화 곡률 Kε (1/unit)'),('min_obstacle_clearance_units','obstacle_penalty_s_per_unit2','최소 링크 여유 (unit)','접근 벌점 O (s/unit²)')]
    for a,(xx,yy,xl,yl) in zip(axes,pairs):
        z=reps[[xx,yy,'curve_key']].dropna()
        for k,g in z.groupby('curve_key'): a.scatter(g[xx],g[yy],s=10,alpha=.45,color=COLORS[k],label=NAMES[k],rasterized=False)
        a.set(xlabel=xl,ylabel=yl,title=f'원 지표 · 시나리오 n={len(z):,}')
        a.set_yscale('symlog',linthresh=.01)
        if xx=='joint_jerk_effort':
            a.set_xscale('symlog',linthresh=.01)
            a.set_xticks([0,1,100,10000,1000000,100000000])
    axes[0].legend(fontsize=8);save(f,'04_relationships','각 시나리오에서 목적값과 독립적인 고정 해시로 유효 후보 1개를 골랐다. 독립 시나리오 하나가 점 하나다. 세로축은 symlog(선형 경계 0.01); J 가로축도 같다. 상관의 인과적 해석이나 수치 미확정 지표의 우열 주장은 하지 않는다.')
    c=pd.read_csv(out/'conditional_validity.csv'); z=c[c.condition.eq('obstacle_layout')].pivot(index='bin',columns='curve_key',values='scenario_valid_rate')[FAMILIES]
    f,a=plt.subplots(figsize=(10,5.5),layout='constrained');im=a.imshow(z,aspect='auto',vmin=0,vmax=1,cmap='Blues')
    counts=s.obstacle_layout.value_counts(); a.set(xticks=x,xticklabels=[NAMES[k] for k in FAMILIES],yticks=np.arange(len(z)),yticklabels=[f'{b} (n={counts[b]})' for b in z.index],title='장애물 배치별 · 해당 종류 유효 후보 ≥1개인 시나리오 비율')
    for i in range(len(z)):
        for j in range(6): a.text(j,i,f'{z.iloc[i,j]:.0%}',ha='center',va='center',color='white' if z.iloc[i,j]>.6 else '#14233A')
    f.colorbar(im,ax=a,label='시나리오 비율');save(f,'05_conditions','장애물 배치별로 동일 train 시나리오 집합에서 곡선 종류의 후보 확보율을 비교했다. 행별 n이 분모이며 환경 자체의 경로 존재율은 아니다.')
    f,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for a,condition,title in zip(axes.flat,['ratio_bin','motion_bin','start_radius_bin','goal_radius_bin'],['링크 비율 L1/L2','이동거리 / 총길이','시작 반경 / 총길이','도착 반경 / 총길이']):
        z=c[c.condition.eq(condition)]; bins=sorted(z.bin.unique()); positions=np.arange(len(bins))
        for k in FAMILIES:
            g=z[z.curve_key.eq(k)].set_index('bin').reindex(bins)
            a.plot(positions,g.scenario_valid_rate*100,'o-',color=COLORS[k],label=NAMES[k],ms=4)
        ns=z.groupby('bin').scenarios.first()
        a.set(xticks=positions,xticklabels=[b.replace('1.3333333333','1.3333')+f'\nn={ns[b]}' for b in bins],ylabel='유효 후보 확보율 (%)',ylim=(0,105),title=title)
        a.tick_params(axis='x',labelsize=8)
    axes.flat[0].legend(fontsize=8,ncol=2)
    save(f,'10_geometry_conditions','사전 고정 구간별로 해당 종류의 유효 후보가 하나 이상 있는 상황 비율. 같은 구간의 모든 종류는 동일 상황 수를 분모로 쓴다. 경계는 왼쪽 포함·오른쪽 제외, n<30은 CSV에 소표본 표시. 인과적 곡선 효과나 전체 환경 성능으로 해석하지 않는다.')
    d=pd.read_csv(q/'convergence_detail.csv');f,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for a,label in zip(axes.flat,FIELDS):
        z=d[d.metric.eq(label)]; atol,rtol=TOL[label]
        for k,g in z.groupby('curve_key'):
            for n in [81,161,321]:
                err=abs(g['value'+str(n)]-g.value641)/(atol+rtol*abs(g.value641));err=err.dropna()
                if len(err):a.scatter(np.repeat(n,len(err)),err,s=4,alpha=.14,color=COLORS[k])
        a.axhline(1,color='#D24B52',ls='--');a.set(yscale='symlog',ylim=(0,None),xticks=[81,161,321],xlabel='표본 수 N',ylabel='오차 / 허용오차',title=label)
    from matplotlib.lines import Line2D
    axes[0,0].legend(handles=[Line2D([],[],marker='o',ls='',color=COLORS[k],label=NAMES[k]) for k in FAMILIES],ncol=2,loc='center',fontsize=8)
    f.suptitle(f'고정 선정 train {d.scenario_id.nunique()}개 상황 · {d.run_id.nunique()}개 후보 · 641 대비 민감도',fontsize=12)
    save(f,'06_convergence','선정된 train 시나리오의 전 후보를 재계산했다. 세로축은 |N−641|/(atol+rtol·|641|), 빨간 선 이하는 사전 허용범위다. 641은 참값이 아니다. 목적별 161·321 두 비교와 유효성 변화를 함께 판정했다.')
    # Explain only a fully converged valid run; also show a documented invalid run.
    good=d.groupby('run_id').filter(lambda g:len(g)==6 and g.converged.eq(True).all() and not g.hard_status_changed.any() and g.selection_valid161.all()).run_id.unique()
    selected=pd.read_csv(q/'selected_cases.csv').fillna(''); selected_ids=list(selected.loc[selected.run_id.isin(good),'run_id'])
    if not selected_ids and len(good): selected_ids=[sorted(good)[0]]
    config=MathematicalEvaluationConfig(**json.loads((data/'experiment_metadata.json').read_text())['evaluation_config'])
    examples=[]
    if selected_ids: examples.append(('07_verified_example',selected_ids[0],'수치 기준을 여섯 목적 모두 통과한 train 실행'))
    for reason in ['unreachable','reachable_collision']:
        candidates=selected[selected.reason.eq(reason)].run_id.tolist()
        if candidates and candidates[0]: examples.append(('08_'+reason,candidates[0],reason+' 사전 선정 실패 사례'))
    case_json=[]
    for name,rid,reason in examples:
        row=m.set_index('run_id').loc[rid].to_dict();row['run_id']=rid
        result,curve=simulate(row,161,config);fr=pd.DataFrame(result.sample_rows)
        case_json.append(dict(run_id=rid,scenario_id=row['scenario_id'],reason=reason,inputs=row,equations=result.equations,metrics=result.metrics))
        f,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');a=axes[0,0]
        a.plot(fr.x,fr.y,color=COLORS[row['curve_key']],lw=2,label=NAMES[row['curve_key']]);a.scatter([row['start_x'],row['goal_x']],[row['start_y'],row['goal_y']],marker='x',color='#16253C')
        for label,xx,yy in [('시작',row['start_x'],row['start_y']),('도착',row['goal_x'],row['goal_y'])]:a.annotate(label,(xx,yy),xytext=(5,5),textcoords='offset points',fontsize=8)
        unreachable=fr[fr.reachable_at_sample.eq(False)]
        if len(unreachable):a.scatter(unreachable.x,unreachable.y,color='#D24B52',s=12,label='도달 불가 표본')
        for o in json.loads(row['obstacles_json']): a.add_patch(plt.Circle((o['center_x'],o['center_y']),o['radius_units'],color='#D24B52',alpha=.4))
        for i in [0,40,80,120,160]:
            z=fr.iloc[i]
            if pd.notna(z.theta1_rad):
                bx,by=row['base_x'],row['base_y'];jx=bx+row['link1']*np.cos(z.theta1_rad);jy=by+row['link1']*np.sin(z.theta1_rad)
                a.plot([bx,jx,z.x],[by,jy,z.y],'-o',alpha=.35,color='#16253C',ms=3)
        a.set(aspect='equal',xlabel='x (unit)',ylabel='y (unit)',title='경로·장애물·5개 시각의 링크');a.legend()
        axes[0,1].plot(fr.time_s,fr.theta1_rad,label='θ1(t)');axes[0,1].plot(fr.time_s,fr.theta2_rad,label='θ2(t)');axes[0,1].set(xlabel='시간 t (s)',ylabel='관절각 (rad)',title='역기구학 합성식 → 시간 표본');axes[0,1].legend()
        for col,label in [('joint_velocity_squared_rad2_s2','Eθ: rad²/s²'),('joint_acceleration_squared_rad2_s4','A: rad²/s⁴'),('joint_jerk_squared_rad2_s6','J: rad²/s⁶'),('obstacle_penalty_integrand_per_unit2','O: 1/unit²')]:axes[1,0].plot(fr.time_s,fr[col],label=label)
        axes[1,0].set(xlabel='시간 t (s)',ylabel='각 피적분함수 (서로 다른 단위)',yscale='symlog',title='시간 적분항 · 크기 직접 비교 금지');axes[1,0].legend(fontsize=8)
        axes[1,0].set_xlim(0,row['duration_s'])
        if fr.joint_velocity_squared_rad2_s2.isna().all():axes[1,0].text(.5,.5,'시간 격자 도달 불가 → 관절 적분 계산하지 않음\n공란은 0이 아닙니다.',transform=axes[1,0].transAxes,ha='center',va='center',fontsize=10)
        axes[1,1].plot(fr.geometry_u,fr.path_speed_units_per_u,label='L: unit');axes[1,1].plot(fr.geometry_u,fr.regularized_curvature_integrand_inv_unit,label='Kε: 1/unit');axes[1,1].set(xlabel='기하 매개변수 u',ylabel='각 피적분함수 (서로 다른 단위)',yscale='symlog',title='기하 적분항');axes[1,1].legend()
        values=' · '.join(f'{k}={result.metrics[v]:.4g}' if result.metrics[v] is not None else f'{k}=NA' for k,v in FIELDS.items())
        f.suptitle(rid+'\n'+values,fontsize=10)
        save(f,name,reason+f'. scenario_id={row["scenario_id"]}, run_id={rid}. 정확한 곡선식·IK 합성식은 representative_equations.json 참조. 결측은 그려지지 않으며 0이 아니다.')
    write_json(out/'representative_equations.json',case_json)
    # A genuine different-winner example restricted to verified objectives.
    ranks=pd.read_csv(q/'rank_stability.csv'); ok=ranks[(ranks.samples==161)&ranks.objective_verified&ranks.winner_n.notna()]
    trade=next((g for _,g in ok.groupby('scenario_id',sort=True) if g.winner_n.nunique()>1),None)
    if trade is not None:
        f,a=plt.subplots(figsize=(8,6),layout='constrained');items=[]
        for rid,g in trade.groupby('winner_n'):
            row=m.set_index('run_id').loc[rid].to_dict(); row['run_id']=rid;result,_=simulate(row,161,config);fr=pd.DataFrame(result.sample_rows)
            a.plot(fr.x,fr.y,label=', '.join(g.objective),color=COLORS[row['curve_key']],ls=['-','--',':','-.'][len(items)%4]);items.append(dict(run_id=rid,objectives=g.objective.tolist(),equations=result.equations))
        for o in json.loads(row['obstacles_json']):a.add_patch(plt.Circle((o['center_x'],o['center_y']),o['radius_units'],color='#D24B52',alpha=.4))
        a.set(aspect='equal',xlabel='x (unit)',ylabel='y (unit)',title='검증된 목적들의 서로 다른 선택\n'+row['scenario_id']);a.legend(fontsize=8)
        save(f,'09_objective_tradeoff','동일 시나리오에서 수렴성 기준을 통과한 목적들이 실제로 다른 후보를 선택한 사례. 팔꿈치 해만 다른 후보는 끝점 경로가 겹칠 수 있다. ID·수식은 tradeoff_equations.json 참조.');write_json(out/'tradeoff_equations.json',items)
    table(out/'figure_captions.csv',captions)
    write_json(out/'plot_style.json',dict(font=font,dpi=300,colors=COLORS,seed=SEED))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['all','research','mechanical','integrity','convergence','analysis'],default='all');p.add_argument('--data',type=Path,default=ROOT/'data/phase4');args=p.parse_args()
    q=ROOT/'outputs/phase4_quality';a=ROOT/'outputs/phase4_analysis';q.mkdir(parents=True,exist_ok=True);a.mkdir(parents=True,exist_ok=True)
    if args.stage in ['all','mechanical']:mechanical(args.data,q)
    if args.stage!='mechanical':
        m=load_train(args.data,q)
        if args.stage in ['all','research','integrity']:train_integrity(args.data,q,m)
        if args.stage in ['all','research','convergence']:convergence(args.data,q,m)
        if args.stage in ['all','research','analysis']:analysis(args.data,q,a,m)
    print('DONE',args.stage,flush=True)

if __name__=='__main__': main()
