"""Train-only opt-in diagnostics. Writes separate JSON/Markdown; never raw data.

Run: .venv/Scripts/python.exe scripts/refine_phase4.py
Dependencies: project [data,analysis,refinement]. Protocol fixed before execution.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import scipy
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

import audit_phase4 as audit
from robot_planner.curves import make_curve
from robot_planner.evaluation import CircleObstacle
from robot_planner.kinematics import PlanarArm2R
from robot_planner.phase4 import score_phase4_candidates
from robot_planner.phase4_evaluation import MathematicalEvaluationConfig, _curve_value_derivatives
from robot_planner.refinement import VERSION, adaptive_path_check, differential_joint_state

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/phase4_refinement'
JOINT={'E_theta':'velocity','A':'acceleration','J':'jerk'}
GRIDS=(161,641,1281,2561)


def objects(row):
    arm=PlanarArm2R(row['link1'],row['link2'],(row['base_x'],row['base_y']))
    curve=make_curve(row['curve_key'],(row['start_x'],row['start_y']),
                     (row['goal_x'],row['goal_y']),shape=row['shape'])
    stored=json.loads(row['curve_parameters_json'])
    for k,v in stored.items():
        if isinstance(v,(int,float)) and not math.isclose(v,curve.parameters[k],rel_tol=1e-10,abs_tol=1e-10):
            raise ValueError('Parameter reconstruction mismatch: '+k)
    obstacles=tuple(CircleObstacle(**o) for o in json.loads(row['obstacles_json']))
    return arm,curve,obstacles


def selection(m,detail):
    audit.train_only(m)
    failed=sorted(detail.loc[detail.metric.isin(['A','J'])&detail.converged.eq(False),'run_id'].unique())
    passed=detail.groupby('run_id').filter(lambda g:len(g)==6 and g.converged.eq(True).all()
                                         and not g.hard_status_changed.any()).run_id.unique()
    controls=[]
    for family in audit.FAMILIES:
        pool=m[m.run_id.isin(passed)&m.selection_valid.eq(True)&m.curve_key.eq(family)].sort_values('run_id')
        if not pool.empty:controls.append(pool.iloc[0].run_id)
    return failed,controls


def analytic_grid(row,n):
    arm,curve,_=objects(row);T=row['duration_s'];times=np.linspace(0,T,n)
    try:
        states=[differential_joint_state(arm,curve,float(t),T,row['elbow_mode']) for t in times]
        values={k:float(np.trapezoid([sum(v*v for v in s[name]) for s in states],times)) for k,name in JOINT.items()}
        if not all(math.isfinite(v) for v in values.values()):raise ValueError('nonfinite integral')
        return dict(values=values,error=None,min_sample_singularity=min(s['singularity_index'] for s in states))
    except (ValueError,OverflowError,ZeroDivisionError) as exc:
        return dict(values={k:None for k in JOINT},error=str(exc),min_sample_singularity=None)


def adaptive_joint_integrals(row):
    arm,curve,_=objects(row);T=row['duration_s'];branch=row['elbow_mode']
    def singular(t):
        s=t/T;u=10*s**3-15*s**4+6*s**5
        x,y,*_=_curve_value_derivatives(curve,u)
        return arm.inverse((x,y),branch).singularity_index
    try:
        ts=np.linspace(0,T,257);ss=[singular(t) for t in ts];points=list(np.linspace(0,T,33)[1:-1]);minima=[]
        for i in range(1,256):
            if ss[i]<ss[i-1] and ss[i]<=ss[i+1]:
                res=minimize_scalar(singular,bounds=(ts[i-1],ts[i+1]),method='bounded',options={'xatol':1e-12})
                if not res.success:raise ValueError('Singularity minimizer failed: '+res.message)
                points.append(float(res.x));minima.append(dict(t=float(res.x),singularity_index=float(res.fun)))
        result={}
        for label,name in JOINT.items():
            def integrand(t):
                state=differential_joint_state(arm,curve,t,T,branch)
                return sum(v*v for v in state[name])
            q=quad(integrand,0,T,epsabs={'E_theta':1e-7,'A':1e-5,'J':1e-3}[label],epsrel=1e-8,
                   limit=300,points=sorted(set(points)),full_output=1)
            result[label]=dict(value=q[0],estimated_absolute_error=q[1],evaluations=q[2]['neval'],
                               warning=q[3] if len(q)>3 else None)
        return dict(integrals=result,minima=minima,error=None)
    except (ValueError,OverflowError,ZeroDivisionError) as exc:
        return dict(integrals={},minima=[],error=str(exc))


def compare(value,reference,label):
    if value is None or reference is None or not all(math.isfinite(v) for v in (value,reference)):
        return dict(passed=None,absolute_error=None,relative_error=None)
    absolute=abs(value-reference);atol,rtol=audit.TOL[label]
    return dict(passed=absolute<=atol+rtol*abs(reference),absolute_error=absolute,
                relative_error=absolute/abs(reference) if reference else None)


def ranks(gridrows):
    results=[]
    for sid in sorted({r['scenario_id'] for r in gridrows}):
        source=[r for r in gridrows if r['scenario_id']==sid];scores={}
        for n in GRIDS:
            rows=[r for r in source if r['samples']==n]
            for method in ('legacy','differential_grid'):
                if method=='differential_grid' and any(r['selection_valid'] and r['analytic']['error'] for r in rows):
                    continue  # Do not quietly remove a candidate to manufacture a new winner.
                vals=[]
                for row in rows:
                    v={k:row[k] for k in ('run_id','scenario_id','curve_key','shape','elbow_mode','selection_valid','data_split')}
                    v.update(row['legacy'])
                    if method=='differential_grid':
                        v.update({audit.FIELDS[k]:value for k,value in row['analytic']['values'].items()})
                    vals.append(v)
                scores[(method,n)]=pd.DataFrame(score_phase4_candidates(vals))
        if ('legacy',161) not in scores:continue
        base=scores[('legacy',161)];bw=audit.winner_rows(base).set_index('objective')
        for (method,n),frame in scores.items():
            winners=audit.winner_rows(frame).set_index('objective')
            for obj,field in audit.OBJECTIVES.items():
                both=base[base.selection_valid][['run_id',field]].merge(frame[frame.selection_valid][['run_id',field]],on='run_id',suffixes=('_base','_new'))
                ra=both[field+'_base'].rank();rb=both[field+'_new'].rank()
                rho=float(ra.corr(rb)) if len(both)>1 and ra.nunique()>1 and rb.nunique()>1 else None
                a=bw.loc[obj];b=winners.loc[obj]
                results.append(dict(scenario_id=sid,method=method,samples=n,objective=obj,common_valid=len(both),rank_correlation=rho,
                    baseline_winner=a.winner_run_id,new_winner=b.winner_run_id,winner_changed=a.winner_run_id!=b.winner_run_id,
                    baseline_ties=json.loads(a.tied_run_ids_json),new_ties=json.loads(b.tied_run_ids_json),
                    tie_set_changed=a.tied_run_ids_json!=b.tied_run_ids_json,
                    interpretation='diagnostic only: old validity; differential E/J with same-resolution legacy L/K/O'))
    return results


def main():
    started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    data=ROOT/'data/phase4';quality=ROOT/'outputs/phase4_quality'
    expected=audit.sha(data/'dataset_manifest.json')+';python-float-roundtrip-v1'
    if (quality/'train_cache_source.txt').read_text()!=expected:raise ValueError('Stale train cache; rerun original audit first')
    m=audit.train_only(pd.read_parquet(quality/'train_metrics.parquet'))
    detail=pd.read_csv(quality/'convergence_detail.csv')
    if not set(detail.run_id)<=set(m.run_id):raise ValueError('Non-train convergence records')
    failed,controls=selection(m,detail);index=m.set_index('run_id',drop=False)
    failure_sids=sorted(index.loc[failed].scenario_id.unique())
    selected_sids=sorted(pd.read_csv(quality/'selected_cases.csv').scenario_id.dropna().unique())
    if not set(selected_sids)<=set(m.scenario_id):raise ValueError('Non-train collision scenarios')
    meta=json.loads((data/'experiment_metadata.json').read_text(encoding='utf-8'))
    config=MathematicalEvaluationConfig(**meta['evaluation_config'])
    provenance=dict(version=VERSION,input_dataset_version=meta['dataset_version'],input_code_version=meta['code_version'],
        python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,
        selection='all previous A/J failures + lexicographic all-objective-converged valid control per family; no new random sampling',
        previous_selection_seed=audit.SEED,failed_run_ids=failed,control_run_ids=controls,failure_scenarios=failure_sids,
        collision_scenarios=selected_sids,protocol_sha256=audit.sha(ROOT/'docs/phase4_refinement_protocol.md'),
        hashes={str(p.relative_to(ROOT)):audit.sha(p) for p in [data/'dataset_manifest.json',data/'experiment_metadata.json',
            quality/'train_metrics.parquet',quality/'convergence_detail.csv',quality/'selected_cases.csv',
            ROOT/'scripts/refine_phase4.py',ROOT/'src/robot_planner/refinement.py',ROOT/'src/robot_planner/phase4_evaluation.py']})
    audit.write_json(OUT/'provenance.json',provenance)
    gridrows=[]
    for sid in failure_sids:
        for row in m[m.scenario_id.eq(sid)].to_dict('records'):
            for n in GRIDS:
                result,_=audit.simulate(row,n,config)
                gridrows.append({**{k:row[k] for k in ('run_id','scenario_id','curve_key','shape','elbow_mode','data_split')},
                    'samples':n,'selection_valid':result.metrics['selection_valid'],
                    'fully_reachable':result.metrics['fully_reachable'],'collision_free':result.metrics['collision_free'],
                    'legacy':{f:result.metrics[f] for f in audit.FIELDS.values()},'analytic':analytic_grid(row,n)})
        audit.write_json(OUT/'grid_comparison.json',gridrows)
        print('grid complete',sid,flush=True)
    quadrows=[];comparisons=[]
    for rid in failed+controls:
        row=index.loc[rid].to_dict();q=adaptive_joint_integrals(row)
        quadrows.append(dict(run_id=rid,scenario_id=row['scenario_id'],curve_key=row['curve_key'],
                             selection_reason='previous_failure' if rid in failed else 'control',**q))
        available=[r for r in gridrows if r['run_id']==rid]
        if not available:
            for n in (161,2561):
                result,_=audit.simulate(row,n,config)
                available.append(dict(samples=n,legacy={f:result.metrics[f] for f in audit.FIELDS.values()},analytic=analytic_grid(row,n)))
        for r in available:
            for label in JOINT:
                ref=q['integrals'].get(label,{});reference=ref.get('value')
                warning=ref.get('warning') or q['error']
                for method,value in [('legacy',r['legacy'][audit.FIELDS[label]]),('differential_grid',r['analytic']['values'][label])]:
                    comparisons.append(dict(run_id=rid,scenario_id=row['scenario_id'],selection_reason='previous_failure' if rid in failed else 'control',
                        joint_derivative_reliable161=row['joint_derivative_reliable'],metric=label,samples=r['samples'],method=method,value=value,
                        adaptive_reference=reference,reference_warning=warning,
                        **compare(value,None if warning else reference,label)))
        print('adaptive integral',rid,flush=True)
    audit.write_json(OUT/'adaptive_integrals.json',quadrows)
    audit.write_json(OUT/'integral_accuracy.json',comparisons)
    rankrows=ranks(gridrows);audit.write_json(OUT/'rank_sensitivity.json',rankrows)
    collisionrows=[];collisionstart=time.perf_counter()
    for sid in selected_sids:
        for row in m[m.scenario_id.eq(sid)].to_dict('records'):
            arm,curve,obstacles=objects(row)
            check=adaptive_path_check(arm,curve,row['elbow_mode'],obstacles,link_radius=config.link_radius_units)
            collisionrows.append(dict(run_id=row['run_id'],scenario_id=sid,curve_key=row['curve_key'],
                legacy_selection_valid=row['selection_valid'],legacy_fully_reachable=row['fully_reachable'],
                legacy_collision_free=row['collision_free'],**check))
        print('adaptive path',sid,flush=True)
    collisionseconds=time.perf_counter()-collisionstart
    audit.write_json(OUT/'adaptive_path_checks.json',collisionrows)
    reasons=m.groupby('scenario_id').apply(audit.no_valid_reason).value_counts().to_dict()
    newly_invalid=[r['run_id'] for r in collisionrows if r['legacy_selection_valid'] and r['status']=='invalid']
    summary=dict(version=VERSION,grid_scenarios=len(failure_sids),grid_runs=len(gridrows)//4,grid_evaluations=len(gridrows),
        previous_failure_runs=len(failed),control_runs=len(controls),
        previously_valid_failure_runs=int(index.loc[failed].selection_valid.eq(True).sum()),
        grid_hard_status_changed_runs=int(pd.DataFrame(gridrows).groupby('run_id')[['fully_reachable','collision_free','selection_valid']].nunique().gt(1).any(axis=1).sum()),
        quadrature_runs=len(quadrows),quadrature_warnings=sum(bool(q['error']) or any(v['warning'] for v in q['integrals'].values()) for q in quadrows),
        collision_scenarios=len(selected_sids),collision_runs=len(collisionrows),
        path_status_counts=pd.Series([r['status'] for r in collisionrows]).value_counts().to_dict(),
        previously_valid_status_counts=pd.Series([r['status'] for r in collisionrows if r['legacy_selection_valid']]).value_counts().to_dict(),
        newly_invalid_run_ids=newly_invalid,both_failure_witnesses=sum(bool(r['collision_witnesses']) and bool(r['unreachable_witnesses']) for r in collisionrows),
        hidden_collision_on_legacy_unreachable=sum(not r['legacy_fully_reachable'] and bool(r['collision_witnesses']) for r in collisionrows),
        full_train_failure_classification=reasons,collision_seconds=collisionseconds,
        mean_path_evaluations=float(np.mean([r['evaluations'] for r in collisionrows])),max_path_evaluations=max(r['evaluations'] for r in collisionrows),
        total_seconds=time.perf_counter()-started,accuracy_by_group=[],rank_changes=[])
    for (reason,method,n,label),g in pd.DataFrame(comparisons).groupby(['selection_reason','method','samples','metric']):
        summary['accuracy_by_group'].append(dict(group=reason,method=method,samples=int(n),metric=label,runs=len(g),
            passed=int(g.passed.eq(True).sum()),failed=int(g.passed.eq(False).sum()),unavailable=int(g.passed.isna().sum()),
            max_relative_error=float(g.relative_error.max()),max_absolute_error=float(g.absolute_error.max())))
    for (method,n,obj),g in pd.DataFrame(rankrows).groupby(['method','samples','objective']):
        summary['rank_changes'].append(dict(method=method,samples=int(n),objective=obj,scenarios=len(g),
            winner_changes=int(g.winner_changed.sum()),tie_set_changes=int(g.tie_set_changed.sum())))
    audit.write_json(OUT/'summary.json',summary)
    report(summary,comparisons)
    audit.write_json(OUT/'output_manifest.json',dict(version=VERSION,files=[dict(path=p.name,bytes=p.stat().st_size,sha256=audit.sha(p)) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='output_manifest.json']))
    print(json.dumps({k:v for k,v in summary.items() if k not in ['accuracy_by_group','rank_changes']},ensure_ascii=False,indent=2),flush=True)


def report(summary,comparisons):
    lines=['# Phase 4 수치·충돌 보강 결과','',
        '기계적 검증/탐색 진단 보고서. 기존 원자료·GUI·평가 라벨은 수정하지 않았다. 새 계산은 opt-in이다.',
        '본 단계는 train만 읽었다. validation/test/challenge의 평가값을 사용하지 않았으며 모델을 학습하지 않았다.','',
        '## 범위와 사전 기준','',
        f"이전 A/J 실패 {summary['previous_failure_runs']}개와 곡선별 대조군 {summary['control_runs']}개를 적응형 적분으로 검산했다. 실패가 속한 {summary['grid_scenarios']}개 시나리오의 {summary['grid_runs']}개 후보를 161/641/1281/2561점으로 비교했다.",
        'E 1%, A 2%, J 5%와 절대 허용값은 기존 기준을 유지했다. 자세한 설정과 선택 이유는 docs/phase4_refinement_protocol.md 및 provenance.json 참조.',
        '참조는 위치 방정식을 세 번 미분한 독립 역기구학과 적응형 적분이다. 수치 적분 오차 추정은 엄밀한 보증이 아니며 2561점도 참값이 아니다.','',
        '## 적분 결과','', '|집단|계산|표본|목적|통과/평가|최대 상대오차|','|---|---|---:|---|---:|---:|']
    for r in summary['accuracy_by_group']:
        if r['samples'] in (161,2561):
            lines.append(f"|{r['group']}|{r['method']}|{r['samples']}|{r['metric']}|{r['passed']}/{r['runs']-r['unavailable']}|{r['max_relative_error']:.6g}|")
    lines+=['',f"적응형 적분 경고/실패 실행: {summary['quadrature_warnings']}. 절대오차·적분기 추정오차·특이점 위치는 integral_accuracy.json / adaptive_integrals.json에 함께 저장했다.",
        '특이점 부근에서는 관절각의 고계 미분이 짧은 시간에 크게 변할 수 있다. 표본 차분의 해상도 부족과 반올림 오차가 함께 작용할 수 있으므로 무조건 표본 수만 늘리는 정책은 채택하지 않는다.',
        '이번 실패군에서 미분법만 바꾼 161점 적분도 A/J 일부가 허용범위를 벗어났다. 즉 미분 공식 개선만으로 충분하지 않고, 피적분함수가 급변하는 구간의 적분 해상도도 확보해야 한다.',
        f"이전 실패 실행 중 selection_valid였던 수는 {summary['previously_valid_failure_runs']}개다. 충돌/도달 실패 후보를 포함한 수치 진단이므로 10개 실패를 곧바로 10개 잘못된 승자 라벨로 해석하지 않는다.",
        'joint_derivative_reliable은 특이점 진단값일 뿐 적분 정확도 보증이 아니다. 실행별 실제 오차와 함께 확인해야 한다.',
        '후보 순위·승자 민감도는 rank_sensitivity.json. 기존 유효성 라벨을 고정한 진단이며, 새 충돌 판정과 섞어 최종 라벨로 사용하지 않는다.',
        f"고해상도 grid의 도달/충돌/유효성 상태가 달라진 실행 수: {summary['grid_hard_status_changed_runs']}.",
        f"비교된 시나리오·목적·방법·해상도 조합 중 결정론적 승자 변화 {sum(r['winner_changes'] for r in summary['rank_changes'])}건, 동점 집합 변화 {sum(r['tie_set_changes'] for r in summary['rank_changes'])}건. 일부 사례의 승자가 유지되었다고 잘못된 적분값을 정당화할 수는 없다.",
        '종합점수에는 같은 해상도의 기존 L/K/O와 새 E/J를 사용한 진단값을 별도 표기했다. A는 기존 종합점수의 구성요소가 아니다.','',
        '## 연속 구간에 대한 적응형 충돌 점검','',
        f"19개 시나리오 {summary['collision_runs']}개 후보: {summary['path_status_counts']}.",
        f"기존 유효 후보의 새 상태: {summary['previously_valid_status_counts']}. 새로 위반을 찾은 실행: {len(summary['newly_invalid_run_ids'])}.",
        f"도달 불가와 충돌을 모두 목격한 실행: {summary['both_failure_witnesses']}. 기존 도달 불가 실행 중 충돌도 목격한 수: {summary['hidden_collision_on_legacy_unreachable']}.",
        'invalid는 위반 위치를 발견했다는 뜻이다. bounded_clear는 현재 2R·원형 장애물·고정 팔꿈치 가지 모델에서 구간 이동 상계가 여유보다 작음을 확인했다는 뜻이다. unresolved는 검사 한도/특이점 등으로 결론을 내리지 못했다는 뜻이며 유효 후보로 승인하지 않는다.',
        '구간 상계는 binary64 계산이며 바깥쪽 반올림 구간 산술이 아니므로 엄밀한 수치 인증이나 실제 로봇의 안전 보증은 아니다. 위반 구간의 세부 검사는 중단하므로 발견되지 않은 다른 원인의 부재를 주장하지 않는다.',
        f"검사시간 {summary['collision_seconds']:.2f}초, 평균 평가점 {summary['mean_path_evaluations']:.1f}, 최대 {summary['max_path_evaluations']}점. 기기·경로 구성에 따라 비용은 달라진다.",
        '', '## 유효 후보가 없는 train 환경','',str(summary['full_train_failure_classification']),
        '기존 분류는 도달 불가 우선의 상호 배타적 요약이다. 도달 가능한 후보의 충돌만 세므로 도달 불가 후보의 다른 구간 충돌이 가려질 수 있다. 새 목격 목록은 원인 중첩을 별도로 보여준다.',
        '853개 환경에 준비된 42개 후보 중 유효 후보가 없다는 뜻이지, 환경에 경로 자체가 없다는 뜻이 아니다. 후보 보강 여부는 이후 별도 설계가 필요하다.','',
        '## 데이터 버전 및 다음 단계','',
        '원자료는 phase4-mathematics-v2, 새 진단은 phase4-refinement-v1이다. 신뢰성이 확인되지 않은 구버전 J/A나 종합 라벨을 확정 학습 정답으로 사용하지 않는다.',
        '이번에 검산한 일부 train 결과를 전체 데이터의 정확도나 과적합 방지 보증으로 일반화하지 않는다. 최적의 범위는 준비된 42개 후보 안이다.',
        '다음에는 생성기의 v3 정책(독립 미분/적응형 적분, 미확정 유효성 처리)을 정한 후 별도 경로에 버전 데이터와 라벨을 재생성해야 한다. 기존 GUI 기본값은 계속 구버전 계산을 사용한다.',
        '전체 재생성 비용은 곡선·특이점 분포에 따라 달라진다. 이번 실행시간과 평가점 수는 파일에 기록했지만 대표 사례 편향이 있으므로 전체 시간의 정확한 예측으로 쓰지 않는다. 먼저 train의 고정 소규모 파일럿으로 처리시간·실패율을 측정하고 범위를 확정한다.',
        '', '## 재실행','', '```powershell', '.\\.venv\\Scripts\\python.exe -m pip install -e ".[data,analysis,refinement]"',
        '.\\.venv\\Scripts\\python.exe scripts/refine_phase4.py',
        '.\\.venv\\Scripts\\python.exe scripts/verify_phase4_refinement.py',
        '.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v','```','',
        '새 JSON/Markdown은 outputs/phase4_refinement에 저장된다. 코드·입력 해시, 라이브러리 버전은 provenance.json; 파일 해시는 output_manifest.json.']
    (OUT/'refinement_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
