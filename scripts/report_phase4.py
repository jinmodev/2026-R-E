"""Generate Korean reports and a typed workbook payload from completed audit CSVs."""
from pathlib import Path
import json
import sys
import platform
import unittest
import io
import pandas as pd
import numpy as np
import matplotlib
import pyarrow

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from audit_phase4 import FIELDS, FAMILIES, NAMES, TOL, table, write_json, SEED, sample_null_audit
Q=ROOT/'outputs/phase4_quality'; A=ROOT/'outputs/phase4_analysis'

def md(df):
    def val(v):
        if pd.isna(v):return 'NA'
        if isinstance(v,float):return f'{v:.5g}'
        return str(v).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(val(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))

def main():
    mech=pd.read_csv(Q/'mechanical_checks.csv'); integ=pd.read_csv(Q/'train_integrity_checks.csv')
    conv=pd.read_csv(Q/'convergence_summary.csv'); d=pd.read_csv(Q/'convergence_detail.csv'); ranks=pd.read_csv(Q/'rank_stability.csv')
    analytic=pd.read_csv(Q/'analytic_checks.csv'); poly=pd.read_csv(Q/'polynomial_derivative_checks.csv'); reproduction=pd.read_csv(Q/'original_161_reproduction.csv')
    rein=pd.read_csv(Q/'reintegration_summary.csv');selected=pd.read_csv(Q/'selected_cases.csv');m=pd.read_parquet(Q/'train_metrics.parquet')
    v=pd.read_csv(A/'validity.csv');no=pd.read_csv(A/'no_valid_summary.csv');wins=pd.read_csv(A/'winner_shares.csv');flow=pd.read_csv(A/'analysis_flow.csv')
    composition=pd.read_csv(A/'composition.csv');conditional=pd.read_csv(A/'conditional_validity.csv')
    paired=pd.read_csv(A/'paired_objective_comparisons.csv');caps=pd.read_csv(A/'figure_captions.csv')
    representative=pd.read_csv(A/'relationship_representatives.csv')
    relationships=[]
    for left,right in [('path_length_units','joint_velocity_effort'),('joint_jerk_effort','regularized_curvature_energy_inv_unit'),('min_obstacle_clearance_units','obstacle_penalty_s_per_unit2')]:
        paired_values=representative[[left,right]].dropna()
        relationships.append(dict(x=left,y=right,independent_scenarios=len(paired_values),spearman_rho=paired_values[left].rank().corr(paired_values[right].rank()),scope='original_161_reference_only; one hash-selected valid candidate per scenario'))
    relationships=table(A/'relationship_summary.csv',relationships)
    provenance=json.loads((Q/'input_provenance.json').read_text(encoding='utf-8'))
    sample_checks=sample_null_audit(ROOT/'data/phase4',Q,m)
    byflag=[]
    for (metric,flag),g in d.groupby(['metric','joint_derivative_reliable161'],dropna=False):
        byflag.append(dict(metric=metric,diagnostic_flag=str(flag),runs=len(g),evaluable=int(g.converged.notna().sum()),converged=int(g.converged.eq(True).sum()),not_converged=int(g.converged.eq(False).sum())))
    table(Q/'derivative_flag_vs_convergence.csv',byflag)
    recomputed=pd.read_parquet(Q/'recomputed_metrics.parquet')
    status=table(Q/'hard_status_by_resolution.csv',recomputed.groupby('samples').agg(runs=('run_id','size'),reachable=('fully_reachable','sum'),valid=('selection_valid','sum')).reset_index())
    nulls=[]
    for field in [*FIELDS.values(),'unregularized_curvature_energy_inv_unit','min_obstacle_clearance_units','min_singularity_index']:
        for (time,reach),g in m.groupby(['time_grid_fully_reachable','fully_reachable']):
            nulls.append(dict(field=field,time_grid_fully_reachable=bool(time),fully_reachable=bool(reach),runs=len(g),missing=int(g[field].isna().sum()),finite=int(np.isfinite(g[field]).sum())))
    table(Q/'summary_missing_conditions.csv',nulls)
    stream=io.StringIO(); suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (Q/'unit_test_log.txt').write_text(stream.getvalue(),encoding='utf-8')
    env=dict(python=platform.python_version(),pandas=pd.__version__,numpy=np.__version__,matplotlib=matplotlib.__version__,pyarrow=pyarrow.__version__,seed=SEED,tests_run=result.testsRun,tests_passed=result.wasSuccessful())
    write_json(Q/'execution_environment.json',env)
    n=m.scenario_id.nunique();nv=int(flow.loc[flow.stage.eq('winner/relationship'),'included'].iloc[0]);nr=selected.scenario_id.dropna().nunique()
    flips=d.drop_duplicates('run_id').hard_status_changed.sum()
    uncertainty=ranks[ranks.samples.eq(161)].groupby('objective').agg(cases=('scenario_id','size'),verified=('objective_verified','sum'),same_winner=('winner_same','sum'),same_tie_set=('tie_set_same','sum')).reset_index()
    table(Q/'objective_rank_summary.csv',uncertainty)
    quality=f'''# Phase 4 수치 품질 검증 보고서

입력 `{provenance['dataset_version']}` / `{provenance['code_version']}`.
원자료 변경 없음. 분석 버전 phase4-audit-v1, 시드 {SEED}.
사전 기준: [검증 계획](../../docs/phase4_quality_analysis_protocol.md).

## 1. 실제 수행 범위

전체 분할은 기계적 검증만 수행했다. train {n:,}개 시나리오·{len(m):,}개 후보와
20,286,000개 저장 시간표본에 대해서만 평가값 검산·탐색을 수행했다.
메타데이터의 평가값 분포나 held-out 승자 통계는 분석하지 않았다.
전체 시간표본을 한꺼번에 적재하지 않고 Parquet 파일별 필요한 열만 읽었다.
원자료 CSV는 청크를 읽고 train만 숫자로 변환했다.

- 기계적 검증: {len(mech)}개 검사, FAIL {int(mech.status.eq('FAIL').sum())}개.
- train 지표·라벨 검증: {len(integ)}개 검사, FAIL {int(integ.status.eq('FAIL').sum())}개.
- train 전체 표본 열 결측조건·무한대 검사: {len(sample_checks)}개 열, FAIL {int(sample_checks.status.eq('FAIL').sum())}개.
- 코드 회귀 테스트: {result.testsRun}개, 전체 통과={result.wasSuccessful()}.
- 해석적 길이·곡률 검사: {len(analytic)}건, 실패 {int((~analytic.passed).sum())}건.
- 다항 관절각 미분 검사: {len(poly)}건, 실패 {int((~poly.passed).sum())}건.
- 선택 사례 161표본 원자료 재현: {len(reproduction)}건, 실패 {int((~reproduction.pass_reproduction).sum())}건.

개별 실패를 포함한 모든 검사 항목: mechanical_checks.csv, train_integrity_checks.csv.
저장된 관절각을 독립적인 sin/cos 순기구학으로 복원한 최대 오차는
{rein.max_fk_error.max():.6g} unit. 좌표 복원이 좋다는 것은 고차 미분 정확도 보증은 아니다.

### 저장된 피적분함수의 재적분

{md(rein[['metric','runs','failures','max_absolute_error']])}

재적분 일치는 저장·요약의 일관성 검증이며, 사다리꼴 이산화 자체의 정확도 증명이 아니다.

## 2. 표본 수 민감도

고정 규칙으로 선정한 {nr}개 train 시나리오의 전 후보 {nr*42}개를
81·161·321·641 표본으로 재계산했다. input/equation/branch/T/epsilon은 고정했다.
빈 선정 층은 selected_cases.csv에 available=0으로 남긴다.
641은 고해상도 비교 기준이며 참값이 아니다. 비확률 층화 대표 사례이므로
아래 통과율을 전체 train의 정확도 확률로 일반화하지 않는다.

{md(conv[['metric','runs','evaluable','converged','not_converged','unavailable','max_relative_error']])}

converged는 161→641과 321→641 모두 `atol+rtol*abs(reference)` 이내인 경우다.
상대오차와 절대오차는 convergence_detail.csv에 함께 기록했다. 기준값 0인
상대오차는 NA이며 절대오차로 판정했다. 계산 불가 NA는 실패로 억지 대체하지 않았다.
목적별 기준은 다음과 같다.

{md(pd.DataFrame([dict(metric=k,atol=a,rtol=r) for k,(a,r) in TOL.items()]))}

161·321·641 사이 하드 유효성 상태가 달라진 실행은 {int(flips)}개다.

{md(status)}

### 목적별 순위 안정성

{md(uncertainty)}

verified는 161 또는 641에서 유효한 모든 후보가 그 목적의 수렴성 기준을 통과하고,
시나리오 전 후보의 하드 상태가 161·321·641에서 바뀌지 않은 경우다.
종합은 다섯 목적과 보조 A까지 통과하도록 보수적으로 판정했다.
same_winner는 같은 ID를 뜻하며, 양쪽 모두 유효 후보 없음도 포함하므로 이 열만으로
정확도나 해의 존재를 주장하면 안 된다. 공통 유효 후보의 순위 상관·동점 집합은
rank_stability.csv에 있다. 원값 수렴과 미세한 동점의 ID 순서 안정은 별개다.

`joint_derivative_reliable`과 실제 수렴의 교차표는 derivative_flag_vs_convergence.csv다.
이 플래그는 |sin(theta2)|의 표본 최솟값 기준일 뿐 수치 정확도 인증이 아니다.
통과 사례에서도 표본 사이의 더 좁은 특이점·충돌 구간을 배제할 수 없다.
이번 검증에서는 진단 플래그 true인 실행에서도 J 수렴 실패가
{int((d.metric.eq('J')&d.joint_derivative_reliable161.eq(True)&d.converged.eq(False)).sum())}건 관찰됐다.
따라서 플래그만으로 저크 정확성을 확정하면 안 된다.

## 3. 해석 검산 방법

- 직선 L = hypot(끝점−시작점).
- 원호 반지름 R, 총각도 Δ이면 L=R|Δ|, 보통 K=|Δ|/R.
- 원호 속도 v=R|Δ|이면 Kε=(|Δ|/R)·[v²/(v²+ε²)]^(5/2).
  보통 K와 정칙화 Kε를 서로 다른 해석값에 비교했다.
- θ(t)=2t⁵−3t³+7t−4를 t∈[0,3]에서 평가했다.
  θ′=10t⁴−9t²+7, θ″=40t³−18t, θ‴=120t²−18.
  표본을 무한히 늘리면 반올림 오차가 증폭될 수 있어 무조건 단조 개선을 요구하지 않았다.

## 4. 결측·실패의 의미와 문서 차이

E/A/J는 **시간 격자 전체** 도달 가능할 때 계산된다. uniform-u 격자만 실패하면
지표가 남아 있어도 selection_valid=false다. 기존 문서의 '어떤 표본에서든
도달 불가면 항상 관절 지표 공란'은 실제 코드보다 강한 설명이다.
무장애물 O=0, 최소 여유=NA; 비정칙 경로의 보통 K=NA는 의도된 값이다.
summary_missing_conditions.csv, sample_null_counts.csv에 조건과 분모를 기록했다.
저장된 link_ratio는 min(L1,L2)/max(L1,L2)이다. 초기 분석 표시의 L1/L2와
혼동되지 않도록 원자료 분포 그림은 짧은/긴 링크로 표시하고, 사전 계획의
방향 있는 비율 구간은 link1/link2를 새로 계산했다. 원본 열은 바꾸지 않았다.
모든 표본 열의 예상 결측 패턴과 무한대 검사는 all_sample_column_checks.csv에 있다.
collision_at_sample=false는 미검사 도달 불가 행의 기본값일 수도 있으므로,
reachable 및 실행 전체 검사 상태 없이 단독으로 안전 라벨로 사용하지 않는다.

새 분석기의 초기 CSV 파서에서 pd.to_numeric의 마지막 비트 반올림 때문에
동점 집합은 같지만 결정적 승자 ID가 3건 달라졌다. Python float 왕복 파서로
수정하고 원자료를 다시 읽어 검증했다. 이는 분석 입력 파서 수정이지 원자료
값/라벨 변경이 아니다. 회귀 테스트로 해당 소수 문자열을 보존한다.

기존 실패 분류는 도달 불가를 먼저 판정하여 배타적 범주를 만든다.
시간 격자 도달 실패이면 링크 충돌을 계산하지 않아 충돌 여부가 가려진다.
따라서 all_unreachable은 충돌이 없다는 뜻이 아니다. 무효 시나리오의 원인은
train 분석 보고서의 후보군 실패 분류를 참조한다.

## 5. 영향 및 후속 조치

이번 작업은 계산 코어·원자료·승자 라벨을 수정하지 않았다. 재계산 결과는
outputs/phase4_quality/recomputed_metrics.parquet의 별도 비교자료이며 원자료와
혼합하지 않았다. 코드 회귀 실패와 표본 민감도 실패를 구분한다.
수렴 실패한 지표는 전체 train 우열 결론에서 제외하고 원자료 기술통계 참고로만 제시한다.
L 등 통과 지표도 검증한 사례 범위의 재현성만 확인된 것이다.

추가 작업: 불안정한 run_id를 대상으로 1281 이상·적응형 적분·독립 미분법을 비교하고,
충돌 검사에는 구간 세분화 또는 연속 검사 보강이 필요하다. 이를 검증한 뒤에만
새 계산 버전과 별도 data 폴더로 평가값·라벨 재생성을 검토한다. 지금 전면 재생성은
하지 않았다. 641표본 전체 생성은 표본 수만 약 4배(148,071,000행)이며 실제 시간·
압축 용량은 100시나리오 벤치마크로 다시 산정해야 한다. 기존 161표본 41.3분의
단순 비례 예상은 약 164분이지만 I/O·CPU·병렬성에 따라 달라지는 거친 추정이다.

## 재실행

```powershell
py -m pip install -e ".[data,analysis]"
py scripts/audit_phase4.py --stage all
py scripts/report_phase4.py
```

중간 재실행은 --stage mechanical / integrity / convergence / analysis.
train 캐시는 원본 manifest SHA-256로 묶이며 다른 입력 버전이면 다시 읽는다.
환경·시드·테스트는 execution_environment.json, input_provenance.json,
unit_test_log.txt에 보존했다. 외부 모델이나 튜닝은 사용하지 않았다.
'''
    (Q/'quality_report.md').write_text(quality,encoding='utf-8')
    best=v.sort_values('scenario_valid_rate',ascending=False).iloc[0]
    report=f'''# Phase 4 train 탐색 분석 — R&E 개별 자료

## 분석 대상과 제외

train {n:,}개 시나리오, {len(m):,}개 후보. 시작·도착·로봇·장애물이 같은
42개 후보는 서로 독립적인 표본이 아니다. 머신러닝 학습·가중치 조정·추천기·
최종 포스터는 이번 산출물에 포함하지 않는다. validation/test/challenge 결과는
탐색에 사용하지 않았다. 입력 버전은 `{provenance['dataset_version']}`다.

{md(flow)}

유효 후보가 없는 시나리오도 구성·유효성의 분모에는 남긴다. 목적별 승자와
관계 그림에서는 유효 후보가 있는 {nv:,}개 시나리오만 포함한다.

## 1. 구성과 유효성

{md(v[['curve_name','variants_per_scenario','candidates','reachable','valid','scenarios_any_valid','scenario_valid_rate']])}

변형과 팔꿈치 해를 합쳐 직선 2개, 나머지 종류 각각 8개다. 변형이 많은 종류는
하나 이상의 유효 후보 및 목적별 최소값을 확보할 탐색 기회가 더 많다.
원자료에서 유효 후보 확보율이 가장 큰 종류는 {NAMES[best.curve_key]}이며
{int(best.scenarios_any_valid):,}/{n:,} ({best.scenario_valid_rate:.1%})다.
이는 **이 생성 분포·준비된 후보군·161표본 검사**에서의 관찰이지
모든 환경에서 해당 곡선이 우월하다는 결론이 아니다.
도달률/도달가능 중 무충돌률/전체 유효률은 분모가 서로 다르다. validity.csv 참조.

## 2. 유효 후보 없음의 원인

{md(no)}

all_unreachable: 42개 모두 도달 실패. all_reachable_colliding: 모두 도달하지만 충돌.
mixed_unreachable_collision: 도달 실패 후보와 도달 가능한 충돌 후보가 함께 있음.
other_calculation_or_schema: 이들로 설명되지 않는 경우. 분류는 배타적이지만
도달 불가가 선행되면 장애물 충돌이 가려질 수 있다. 후보군에 해가 없음은
**환경 자체에 경로가 존재하지 않는다는 증명과 다르다**. 후보군/회피 전략 보강이 필요하다.

## 3. 목적별 승자 — 참고 통계와 검증 사례 분리

winner_shares.csv의 세 scope:

- original_161_reference_only: 전체 유효 train 시나리오의 기존 값 기술통계.
- obstacles_only_reference: 장애물이 있는 시나리오만 같은 방식으로 재집계.
- verified_cases_only: 사전 수치 기준을 목적별로 통과한 일부 사례만 집계.

3번 그림은 첫 범위의 **수치 미확정 참고 결과**다. 저크 및 종합을 포함하여
검증되지 않은 전체 자료에서 확정적 우열을 결론내리지 않는다. 결정적 승자는
원 목적값, L, curve_key, shape, elbow_mode, run_id 순으로 정렬한다.
동점은 math.isclose 상대·절대 1e-12이고 분수 집계는 동점 후보 각각 1/k다.
후보별 분수 배분도 변형 수의 차이에 영향받는다.
무장애물에서는 O=0 동점이므로 결정적 safety 승자를 안전 우월로 해석하지 않는다.

{md(uncertainty)}

verified 열은 목적별 검증 시나리오 수다. 선정 규칙상 극단 사례가 포함되어 있으므로
이 작은 집합의 승자 비율도 전체 환경 모집단의 비율로 해석하지 않는다.

## 4. 지표의 관계와 대응 비교

관계 그림은 각 시나리오의 유효 후보 중 고정 해시 순서로 하나를 선택하여
점 하나=시나리오 하나로 만들었다. 값에 유리하게 대표를 고른 것이 아니다.
원 지표와 normalized_*은 다른 열이다. 본 그림은 원 지표다.

{md(relationships[['x','y','independent_scenarios','spearman_rho']])}

이 값은 원 지표 순위 사이의 Spearman 기술통계다. p값·유의성 결론은 산출하지 않았다.
L–Eθ의 양의 연관, 여유–벌점의 음의 연관은 이 표본에서 관찰된 관계이며,
시간·로봇 크기·경로 형태와 후보 선택 과정의 영향을 제거한 인과효과가 아니다.
J–Kε 관계는 수치 미확정 J를 포함하는 참고값으로 한정한다.

길이 unit, Eθ rad²/s, A rad²/s³, J rad²/s⁵, Kε 1/unit, O s/unit².
Eθ는 줄(J) 단위의 실제 에너지가 아니다. 곡률과 각저크도 다른 개념이다.

본 원자료 이동시간은 {sorted(m.duration_s.unique().tolist())}초다. 같은 공간 경로의
시간을 늘리면 Eθ∝T⁻¹, A∝T⁻³, J∝T⁻⁵, O∝T인 차이가 생긴다.
L과 Kε는 시간과 무관하다. 로봇·환경을 같은 비율 a로 확대하면 보통 L∝a,
Kε∝a⁻¹이나, O는 고정 링크 반지름·clearance epsilon 때문에 단순 축척 법칙이
그대로 성립하지 않을 수 있다. 원 지표의 시나리오 간 크기 차이를 곡선 효과만으로
해석하지 않는다. normalized_*은 같은 시나리오의 유효 후보 범위에 상대적이다.

paired_objective_comparisons.csv는 **각 목적별** family 최소값을 대표로 골라,
같은 시나리오에서 두 종류 모두 유효할 때만 대응 비교했다. 차이는 A−B의
원 지표 차이며 음수는 family_a가 작다는 뜻이다. 한 목적의 최소 후보를 다른
목적에서도 공정한 대표로 간주하지 않는다. 표본 수는 paired_scenarios다.
수치 미확정 참고표로 유지했으며 유의성 검정이나 독립 후보 가정의 CI를 쓰지 않았다.

## 5. 조건별 비교

conditional_validity.csv에 장애물 배치/수, 링크 비율, 이동거리/총길이,
각도차, 시작·도착 반경/총길이별 후보 확보율을 저장했다.
구간은 사전 계획의 고정 경계이며 왼쪽 포함·오른쪽 제외다.
ratio는 <0.75, [0.75,1.3333333333), 그 이상;
여기서는 link1/link2를 새로 계산했다. 원자료 link_ratio=짧은 링크/긴 링크와 다르다.
motion은 [0,.5), [.5,1), [1,2.01); 반경은 [0,.4), [.4,.8), [.8,1.01);
각도차는 [0,60), [60,120), [120,181)이다.
n<30은 small_n=true. 빈 구간은 관측이 없으며 성능 0으로 채우지 않는다.
같은 조건 안에서도 다른 기하 요소가 함께 달라져 인과효과가 아니다.

## 6. 사례와 보고서에 쓸 수식

representative_equations.json에 scenario_id/run_id, 실제 입력, x(u),y(u),
x(t),y(t), 역기구학 합성식, 적분값을 보존했다. 긴 수식은 Excel에 잘라 넣지 않는다.
q(t)=10(t/T)³−15(t/T)⁴+6(t/T)⁵, p(t)=p(q(t)).
X=x(t)−bx, Y=y(t)−by, c₂=(X²+Y²−L₁²−L₂²)/(2L₁L₂).
θ₂=±acos(c₂), θ₁=atan2(Y,X)−atan2(L₂sinθ₂,L₁+L₂cosθ₂).
주값 계산 후 시간 표본은 2π 배수로 unwrap한다.

L=∫₀¹√(x′²+y′²)du; Eθ=∫₀ᵀ(θ̇₁²+θ̇₂²)dt;
A=∫₀ᵀ(θ̈₁²+θ̈₂²)dt; J=∫₀ᵀ(θ⃛₁²+θ⃛₂²)dt;
Kε=∫₀¹(x′y″−y′x″)²/(x′²+y′²+ε²)^(5/2)du;
O=∫₀ᵀ1/max(c(t),εc)²dt (무장애물은 0).
종합 C=100(.25nL+.20nE+.20nJ+.20nK+.15nO), 보상 R=100−C.
모든 원 비용은 작을수록 좋다. A는 종합에 들어가지 않는다.

## 그래프별 한글 캡션

'''+ '\n\n'.join(f'### {r.figure}\n\n{r.caption_ko}\n\n![{r.figure}]({r.figure}.png)' for r in caps.itertuples())+'''

## 재현과 향후 모델 학습 전 보완

```powershell
py -m pip install -e ".[data,analysis]"
py scripts/audit_phase4.py --stage all
py scripts/report_phase4.py
node scripts/build_phase4_workbook.mjs
py scripts/verify_phase4_outputs.py
```

Excel 빌더는 @oai/artifact-tool이 있는 Node 환경이 필요하다. 기존 GUI·이전 포스터·
원자료는 보존한다. 모든 표의 기계 판독 원본은 이 폴더 CSV, 수치 검증은
../phase4_quality/quality_report.md다. 그림은 300dpi PNG와 SVG다.

1. 불안정 지표의 독립 미분/적분·더 높은 해상도 검증을 먼저 수행한다.
2. 연속 충돌 검사와 no_valid_candidate의 회피 후보군 보강을 검토한다.
3. 바뀐 평가값/라벨은 새 데이터 버전에 기록하고 구버전과 혼합하지 않는다.
4. 모델 특징은 입력 기하만 사용한다. 평가값·승자·정규화·도달/충돌 결과의 누출을 차단한다.
5. scenario_id 분할을 유지하고 validation은 다음 단계 모델 선택에만, test는 최종 평가에만 쓴다.
6. 여러 생성 시드·새 장애물 배치로 일반화를 확인한다. 데이터가 많다는 이유만으로
   과적합 해결을 주장하지 않는다. 모든 최적은 현재 준비된 42개 후보 안에서다.
'''
    (A/'analysis_report.md').write_text(report,encoding='utf-8')
    dictionary=[
      ['scenario_id','독립 상황 ID','없음','동일 ID 후보는 독립 표본 아님'],['run_id','후보 실행 ID','없음','시나리오×곡선 변형×팔꿈치'],
      ['selection_valid','도달 및 무충돌','bool','두 이산 격자 검사, 연속 안전 아님'],['L','끝점 경로 길이','unit','기하 격자 사다리꼴 적분'],
      ['E_theta','관절 속도 제곱 적분','rad²/s','실제 에너지 아님'],['A','관절 각가속도 제곱 적분','rad²/s³','보조값; 종합 제외'],
      ['J','관절 각저크 제곱 적분','rad²/s⁵','고차 미분 수렴 확인 필요'],['K_epsilon','정칙화 곡률 제곱 적분','1/unit','일반 K와 다름'],
      ['O','장애물 접근 벌점','s/unit²','무장애물=0; 여유는 NA'],['normalized_*','시나리오 내 min-max','무차원','원값과 구분; 무효 후보 NA'],
      ['balanced_cost','가중 정규화 비용','0–100','낮을수록 좋음; 무효 1e6은 제외 표시'],['joint_derivative_reliable','특이점 표본 진단','bool','정확도 보증 아님'],
      ['converged','161/321→641 허용오차 통과','bool/NA','641은 비교기준, 참값 아님'],['fractional_wins','동점 후보별 1/k 합계','시나리오','종류별 변형 수에 영향받음'],
      ['paired_scenarios','두 종류가 모두 유효','시나리오','목적별 최소 후보 대응 비교'],['small_n','구간 n<30','bool','추론 확대 주의'],
      ['곡선·관절각 합성식','원본 JSON 참조','rad / unit','representative_equations.json'],
      ['입력 원본','data/phase4','phase4-mathematics-v2','data_dictionary.md + experiment_metadata.json'],
      ['수치 기준','docs/phase4_quality_analysis_protocol.md','v1','분석 전에 고정; 결과에 맞춰 조정 안 함'],
      ['재현 시드',str(SEED),'고정','manifest 해시는 input_provenance.json'],
    ]
    table(A/'analysis_dictionary.csv',pd.DataFrame(dictionary,columns=['field','meaning_ko','unit','notes']))
    payload={'sheets':[]}
    def sheet(name,title,note,headers,rows,widths,formats=None):
        clean=pd.DataFrame(rows).astype(object).where(pd.notna(pd.DataFrame(rows)),None).values.tolist()
        payload['sheets'].append(dict(name=name,title=title,note=note,headers=headers,rows=clean,widths=widths,formats=formats or {}))
    sheet('유효성','Train 후보 유효성',f'독립 시나리오 {n:,}개 · 161표본 검사 · 42개 후보 범위',
          ['곡선','전체 후보','도달 후보','유효 후보','도달률','도달 중 무충돌','유효률','유효≥1 상황','상황 확보율'],
          [[r.curve_name,r.candidates,r.reachable,r.valid,None,None,None,r.scenarios_any_valid,None] for r in v.itertuples()], [130,125,125,125,110,140,110,140,130],{'E':'0.0%','F':'0.0%','G':'0.0%','I':'0.0%'})
    payload['scenario_count']=n
    payload['no_valid_counts']=no.to_dict('records')
    sheet('구성','Train 시나리오 구성','n<30 구간은 소표본 · 후보 수가 아닌 상황 수',
          ['조건','범주 / 고정 구간','상황 수 n','분모','비율','소표본'],composition[['variable','category','scenarios','total_scenarios','fraction','small_n']].values.tolist(),[200,310,110,110,100,100],{'E':'0.0%'})
    scope={'original_161_reference_only':'원자료 참고','verified_cases_only':'검증 사례 한정','obstacles_only_reference':'장애물 있음 참고'}
    sheet('목적별선택','목적별 선택 · 수치 미확정 참고와 검증 사례 분리','동점은 후보별 1/k 배분 · 무장애물 O 동점은 안전 우월 아님',
          ['분석 범위','목적','곡선','결정적 승수','동점 분수 승수','분모 상황 n','결정적 비율','동점 배분 비율'],
          [[scope[r.scope],r.objective,NAMES[r.curve_key],r.deterministic_wins,r.fractional_wins,r.denominator_scenarios,r.deterministic_rate,r.fractional_rate] for r in wins.itertuples()], [175,130,125,120,145,125,130,140],{'E':'0.00','G':'0.0%','H':'0.0%'})
    sheet('대응비교','동일 상황 · 목적별 최소 후보 대응 비교','원자료 161표본 참고값 · 차이는 A−B · 한 목적의 대표를 다른 목적에 재사용하지 않음',
          ['목적','곡선 A','곡선 B','대응 상황 n','차이 중앙값 (목적 단위)','A가 작은 비율','소표본'],
          [[r.objective,NAMES[r.family_a],NAMES[r.family_b],r.paired_scenarios,r.median_a_minus_b,r.a_lower_fraction,r.small_n] for r in paired.itertuples()], [140,130,130,130,240,160,100],{'E':'0.0000','F':'0.0%'})
    sheet('조건별확보율','조건별 유효 후보 확보율','같은 조건·종류에 유효 후보가 하나 이상 있는 상황 수 / 그 조건의 전체 상황 수',
          ['조건','구간 / 배치','곡선','상황 n','유효≥1 상황','확보율','소표본'],
          [[r.condition,r.bin,NAMES[r.curve_key],r.scenarios,r.scenarios_any_valid,r.scenario_valid_rate,r.small_n] for r in conditional.itertuples()], [195,300,125,100,135,110,100],{'F':'0.0%'})
    sheet('수치검증','목적별 수렴성 · 641은 참값이 아닌 비교 기준',f'{nr}개 선정 train 상황의 전 후보 · 161→641 및 321→641 모두 사전 기준 통과 필요',
          ['지표','검사 실행','계산 가능','수렴 통과','수렴 실패','계산 불가','최대 절대오차','최대 상대오차'],
          conv[['metric','runs','evaluable','converged','not_converged','unavailable','max_absolute_error','max_relative_error']].values.tolist(),[145,115,115,115,115,115,170,170],{'G':'0.000E+00','H':'0.000E+00'})
    sheet('데이터사전','정의·단위·출처','전체 시간표본 및 긴 수식은 담지 않음 · CSV/JSON과 보고서에서 추적',
          ['필드','의미','단위 / 버전','해석 / 출처'],dictionary,[205,355,200,420])
    write_json(A/'workbook_payload.json',payload)
    print('Reports and workbook payload written; tests:',result.testsRun,result.wasSuccessful())

if __name__=='__main__':main()
