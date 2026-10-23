"""Build Korean research report and typed workbook inputs from train-only tables."""
from pathlib import Path
import json
import sys
import pandas as pd
from analyze_phase4_v3 import OUT, ROOT, FAMILIES, NAMES, COLORS, OBJECTIVES, LABELS, UNITS, SEED, table, write_json, plot_setup


def read(name):return pd.read_csv(OUT/(name+'.csv'),float_precision='round_trip')


def md(frame):
    def fmt(v):
        if pd.isna(v):return 'NA'
        if isinstance(v,float):return f'{v:.6g}'
        return str(v).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(frame.columns)+' |\n|'+'|'.join(['---']*len(frame.columns))+'|\n'+'\n'.join('| '+' | '.join(map(fmt,row))+' |' for row in frame.itertuples(index=False,name=None))


def main():
    check=json.loads((OUT/'analysis_checks.json').read_text(encoding='utf-8'));assert check['passed']
    s=read('scenario_exclusions');v=read('validity');wins=read('winner_shares');rel=read('relationships')
    paired=read('paired_objective_comparisons');caps=read('figure_captions');composition=read('composition');conditions=read('conditional_validity')
    m=pd.read_parquet(OUT/'train_analysis_metrics.parquet');grid=[]
    for k in FAMILIES:
        g=m[m.curve_key.eq(k)]
        reachable=g.time_grid_fully_reachable.eq(True)&g.validation_grid_fully_reachable.eq(True)
        clear=reachable&g.time_collision_samples.eq(0)&g.validation_collision_samples.eq(0)
        nr=int(reachable.sum());nc=int(clear.sum())
        grid.append(dict(curve_key=k,candidates=len(g),grid_reachable=nr,grid_clear=nc,
            grid_reachable_rate=nr/len(g),grid_clear_given_reachable=nc/nr if nr else None,
            grid_clear_rate=nc/len(g)))
    grid=table('sampled_grid_diagnostics',grid)
    plt=plot_setup();f,axes=plt.subplots(1,3,figsize=(13,4.4),layout='constrained')
    for ax,col,title in zip(axes,['grid_reachable_rate','grid_clear_given_reachable','grid_clear_rate'],
        ['격자 도달 / 전체 후보','격자 무충돌 / 격자 도달 후보','격자 도달·무충돌 / 전체 후보']):
        bars=ax.bar(range(6),grid[col]*100,color=[COLORS[k] for k in FAMILIES]);ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=8)
        ax.set(xticks=range(6),xticklabels=[NAMES[k] for k in FAMILIES],ylim=(0,110),ylabel='표본 격자 진단 비율 (%)',title=title,)
        ax.tick_params(axis='x',labelsize=8)
    f.suptitle('시간 161점 + 검증 161점 진단 — 최종 유효성·연속 안전 보증 아님')
    name='10_sampled_grid_diagnostics'
    f.savefig(OUT/(name+'.png'),dpi=300,bbox_inches='tight');f.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(f)
    caps=pd.concat([caps[caps.figure.ne(name)],pd.DataFrame([dict(figure=name,caption_ko='적응형 검사의 조기 종료 영향을 설명하기 위한 보조 표본 진단. 두 격자 모두 도달 가능한 후보만 조건부 무충돌 분모에 넣었다. 후보 분모는 직선 6,000개, 나머지 각 24,000개이며 조건부 분모는 sampled_grid_diagnostics.csv에 있다. 표본 사이 충돌을 놓칠 수 있어 이 값으로 최종 승자를 고르지 않는다.')])],ignore_index=True)
    table('figure_captions',caps)
    n=len(s);nm=check['main_scenarios'];nu=check['excluded_unresolved'];nn=check['excluded_no_valid']
    # Condition-specific winner shares use the same complete-case cohort, not the all-train denominator.
    ww=read('main_winners');cw=[]
    for col in ['obstacle_layout','ratio_bin','motion_bin','angle_bin','start_radius_bin','goal_radius_bin']:
        joined=ww.merge(s[['scenario_id',col]],on='scenario_id',validate='many_to_one')
        for (b,obj),g in joined.groupby([col,'objective'],sort=True):
            for k in FAMILIES:
                count=int(g.curve_key.eq(k).sum())
                cw.append(dict(condition=col,bin=str(b),objective=obj,curve_key=k,wins=count,scenarios=len(g),rate=count/len(g),small_n=len(g)<30))
    table('conditional_winners',cw)
    gs=read('geometry_convergence');gsummary=[]
    for field,g in gs.groupby('field'):
        gsummary.append(dict(field=field,checked=len(g),passed=int(g.passed.sum()),failed=int((~g.passed).sum()),
            max_abs_difference=g.absolute_difference.max(),max_relative_difference=g.relative_difference.max()))
    table('geometry_check_summary',gsummary)
    overview=v[['curve_name','variants_per_scenario','candidates','reachable','valid','scenarios_any_valid','scenario_valid_rate']].copy()
    overview.columns=['곡선','상황당 후보','전체 후보','도달 확인','유효 후보','유효≥1 상황','상황 확보율']
    headline=[]
    for obj in OBJECTIVES:
        z=wins[wins.scope.eq('main')&wins.objective.eq(obj)]
        a=z.loc[z.deterministic_rate.idxmax()];b=z.loc[z.fractional_rate.idxmax()]
        headline.append(dict(목적=LABELS[obj],결정적_최다=NAMES[a.curve_key],결정적_비율=a.deterministic_rate,
            동점배분_최다=NAMES[b.curve_key],동점배분_비율=b.fractional_rate,동점_상황=int(a.tied_scenarios)))
    table('headline_winners',headline)
    best=v.loc[v.scenario_valid_rate.idxmax()]
    equations=json.loads((OUT/'representative_equations.json').read_text(encoding='utf-8'))
    trade=json.loads((OUT/'tradeoff_equations.json').read_text(encoding='utf-8'))
    report=fr'''# Phase 4 v3 train 분석 결과

## 핵심 결과

새 데이터 버전 `phase4-mathematics-v3`의 train **{n:,}개 시나리오, {check['candidate_count']:,}개 후보**를 분석했다.
구성·유효성에는 전체 {n:,}개 상황을 사용했다. 주 승자·대응 비교·관계 분석에는 **{nm:,}개 상황**을 사용했다.
미확정 대안이 있는 {nu}개 상황과, 미확정은 없지만 유효 후보가 없는 {nn}개 상황을 주 비교에서 제외했다.
이 둘은 서로 겹치지 않는다. 후보 단위 미확정은 16개다. 제외된 상황에서의 곡선 성능을 이 결과로 대신 설명하지 않는다.

각 곡선 종류에 유효 후보가 하나 이상 있는 상황 비율이 가장 큰 종류는 **{best.curve_name}**이다.
{int(best.scenarios_any_valid):,}/{n:,} = {best.scenario_valid_rate:.1%}였다. 이는 생성된 환경 분포·주어진 변형 범위에서의 후보 확보율이다.
모든 환경에서 그 곡선이 더 좋다는 결론이 아니다.

{md(pd.DataFrame(headline))}

위 표는 0–1 비율이다. 승자는 준비된 **42개 후보 안에서**의 계산 결과다.
결정적 집계는 원값 차이가 극히 작아도 정렬 순서에 민감할 수 있다. 동점 배분도 수치 오차를 반영한 확률이 아니다.
목적별 수치 허용오차보다 작은 후보 간 차이의 순위까지 검증한 결과로 해석하지 않는다.

## 1. 분석 범위와 재현성

- 사전 규칙: `docs/phase4_v3_analysis_protocol.md`. 사례/관계 선택 시드 {SEED}.
- 완료된 train receipt가 가리키는 파일만 읽었다. validation/test/challenge 결과를 열거나 분석하지 않았다.
- 동일 scenario_id의 42개 후보는 같은 조건의 대응 자료다. 후보 126,000개를 독립 표본 126,000개라고 하지 않는다.
- ML 학습, 가중치 조정, 추천 기능, 최종 포스터 제작은 수행하지 않았다.
- 기존 원자료·GUI·이전 분석·검증 파일은 수정하지 않았다. 이전 v2 통계는 이 결과와 섞지 않았다.
- 원자료의 한글 곡선명과 표준 curve_key 표시명이 다른 행은 {check['source_curve_label_mismatches']:,}개였다.
  이는 ‘3차 베지어/3차 에르미트’를 짧게 ‘베지어/에르미트’로 표시한 48,000행이며 원자료 오류가 아니다. 원본 필드는 보존했다.

## 2. 구성과 후보 유효성

{md(overview)}

직선은 팔꿈치 해까지 2개, 다른 종류는 각 8개 후보이다. 변형이 많을수록 유효 후보나 낮은 비용을 찾을 기회가 많다.
이 탐색 예산 차이를 통제한 곡선 고유 우월성 실험은 아니다.

`validity.csv` 분모:

- 도달 확인률 = fully_reachable=true 후보 / 전체 후보.
- 도달 중 무충돌 확인률 = fully_reachable=true AND collision_free=true / fully_reachable=true.
- 유효률 = selection_valid=true / 전체 후보. 기하·적분 수치 검사도 포함하므로 단순 무충돌률과 다르다.
- 상황 확보율 = 그 종류의 유효 후보가 1개 이상 있는 상황 / 전체 train 상황.

null은 미확정으로 별도 집계했다. 확인률은 미확정을 성공으로 간주하지 않는 보수적인 기술통계다.
도달 검사 도중 위반이 발견되면 남은 구간·충돌 검사가 생략될 수 있으므로 reach_unknown을 도달 불가로 바꾸면 안 된다.
**이 버전은 bounded_clear일 때에만 fully_reachable=true와 collision_free=true를 함께 설정하므로 위 조건부 무충돌률은 모두 100%다.
이는 검사 상태 정의의 결과이며 곡선의 안전 성능 차이를 보여주는 지표가 아니다.** 도달 확인률 역시 ‘충돌과 무관하게 도달 가능한 전체 후보 비율’과 다르다.
보조 sampled_grid_diagnostics.csv/그림은 저장된 시간 161점과 검증 161점에서의 도달·충돌을 분리해 집계한다.
주 분석 후 이 표현상의 문제를 발견하여 설명용 진단을 추가했으며 사전 제외·동점·승자 규칙은 바꾸지 않았다.
추가 근거는 docs/phase4_v3_analysis_diagnostic_note.md다. 격자 진단을 최종 선택의 유효성으로 대체하지 않는다.
구성표에는 시나리오 유형, 배치, 장애물 수, 링크 비율, 이동거리, 시작·도착 반경과 각도 구간을 담았다.
링크 길이와 비율의 기술통계는 geometry_summary.csv다.

## 3. 유효 후보 없는 상황과 미확정

{md(read('no_valid_summary'))}

분류는 배타적인 보고용 범주다. `all_unreachable_witness`는 모든 후보에서 도달 위반을 관측,
`all_collision_witness`는 모든 후보에 충돌 witness가 있고 도달 위반 witness는 없는 경우다.
후자의 이름은 전체 경로의 도달 가능성까지 증명했다는 뜻이 아니다.
`mixed_unreachable_collision`은 두 종류 위반이 관측된 상황, `unresolved_or_calculation`은 확정 실패만으로 정리할 수 없는 상황이다.
하나의 후보에 둘 다 있을 수도 있고 먼저 검사한 위반 때문에 다른 위반이 가려질 수도 있다.
**후보군에 유효 경로가 없다는 사실은 환경 자체에 경로가 없다는 증명이 아니다.**

scenario_exclusions.csv에 모든 상황의 accepted/unresolved/각 witness 수와 분석 집합을 기록했다.
conditional_exclusions.csv는 어려운 조건이 제외 집합에 집중되는지 읽을 수 있는 조건별 분모를 제공한다.
미확정 16개를 0점이나 유효 후보로 처리하지 않았으며, 미확정 대안이 있는 상황의 잠정 라벨은 주 승자 비율에서 제외했다.

## 4. 목적별 승자와 동점

winner_shares.csv의 main은 주 집합, obstacles_only는 그중 장애물이 있는 상황이다.
동점 판정은 원 구현과 같은 math.isclose(relative=absolute=1e-12), 선택 순서는 원 목적값, L, curve_key, shape, elbow_mode, run_id다.
동점 후보 k개에 각각 1/k를 부여하는 집계와 결정적으로 선택한 한 후보의 집계를 함께 기록했다.
동점 배분도 종류별 변형 개수의 영향을 받는다. 무장애물 O=0은 모든 유효 후보가 동점이므로 특정 곡선의 안전성 우월로 해석하지 않는다.
종합점수는 기존 가중치와 시나리오 내부 min–max를 그대로 사용했다. 가중치를 분석 결과에 맞춰 조정하지 않았다.

## 5. 지표의 관계와 대응 비교

{md(rel)}

각 상황에서 목적값과 무관한 SHA256 순서로 유효 후보 하나를 골랐다. 점 하나가 상황 하나다.
Spearman 상관은 순위의 기술통계이며 p값이나 유의성 검정은 수행하지 않았다. 신뢰구간도 계산하지 않았다.
raw는 원 지표이고 within_scenario는 같은 상황의 유효 후보 범위에 대한 min–max 값이다.
원값/정규화/축의 로그 표시는 각각 다른 개념이다. 큰 상관을 원인 관계나 곡선 선택의 보편 법칙으로 읽지 않는다.
장애물 없는 경우 최소 여유는 정의하지 않으므로 여유–벌점 쌍에서만 빠진다.

paired_objective_comparisons.csv는 각 목적에 대해 종류별 최소 비용 후보를 고른 후 같은 상황에서 두 종류가 모두 유효할 때만 비교했다.
차이는 A−B이고 음수면 A가 작다. a_lower_fraction은 위 동점 규칙으로 동점이 아닌 엄격한 우세 비율이다.
한 목적의 대표 후보를 다른 목적에도 공정한 대표라고 간주하지 않았다. 이 비교 역시 후보 탐색 개수 차이를 제거하지 않는다.

단위: L은 unit, Eθ는 rad²/s, A는 rad²/s³, J는 rad²/s⁵, Kε는 1/unit, O는 s/unit².
Eθ는 실제 소비 에너지(Joule)가 아니라 관절 운동 노력도이다. 각저크는 곡률과 다르다.
이 데이터의 이동시간은 {sorted(s.duration_s.unique().tolist())}초다. 같은 공간 경로에서 T를 바꾸면 Eθ∝T⁻¹, A∝T⁻³, J∝T⁻⁵, O∝T다.
L과 기하 Kε는 시간에 무관하다. 로봇 크기도 원 지표의 크기에 영향을 준다. 고정 링크 반경·여유 epsilon 때문에 장애물 벌점의 단순 축척 법칙은 성립하지 않을 수 있다.

## 6. 조건별 비교

conditional_validity.csv는 전체 train 상황을 분모로 한다. conditional_winners.csv는 해당 구간의 주 비교 상황만 분모로 한다.
두 분모를 혼동하지 않는다. 후자는 결정적 승자 집계이고 동점 보정 결과가 아니다.
L1/L2 구간은 <.75, [.75,4/3), >=4/3. 원 link_ratio=짧은/긴 링크와 구분한다.
직선거리/총길이는 [0,.5), [.5,1), [1,2.01), 반경/총길이는 [0,.4), [.4,.8), [.8,1.01),
각도차는 [0,60), [60,120), [120,181)도다. 왼쪽 포함·오른쪽 제외이고 CSV의 구간 표시 소수는 반올림된 표시일 뿐 실제 경계는 사전 규칙이다.
n<30은 small_n=true로 표시했다. 조건부 차이는 다른 기하 요소와 함께 변한 결과여서 인과효과가 아니다.

## 7. 사례와 보고서용 수식

{md(read('case_selection'))}

6종류의 accepted 사례와 도달 불가/충돌 witness 사례를 별도로 제시했다.
곡선·관절 합성식 및 입력은 representative_equations.json에 전체 길이로 보존했다.
각 사례의 161 시간표본은 case_*_samples.csv, E/A/J 적응형 적분점은 case_adaptive_nodes.csv다.
서로 다른 목적의 동점 집합이 겹치지 않고 경로 형태도 다른 train 상황은 {check['tradeoff_eligible_scenarios']:,}개였다.
그중 고정 해시로 고른 사례는 {trade[0]['scenario_id'] if trade else '없음'}이다. 모든 상황에서 목적 간 충돌이 있다는 의미는 아니다.

$$s=t/T,\qquad q(s)=10s^3-15s^4+6s^5,\qquad \mathbf{{p}}(t)=\mathbf{{p}}(q(s)).$$

$$X=x(t)-b_x,\quad Y=y(t)-b_y,\quad c_2=\frac{{X^2+Y^2-L_1^2-L_2^2}}{{2L_1L_2}},$$
$$\theta_2(t)=\pm\arccos c_2,\quad \theta_1(t)=\operatorname{{atan2}}(Y,X)-\operatorname{{atan2}}(L_2\sin\theta_2,L_1+L_2\cos\theta_2).$$

역기구학 해는 도달 가능한 영역에서만 정의한다. 시간 표본의 각도는 2π 배수로 unwrap하여 연속성을 맞춘다.

$$L=\int_0^1\sqrt{{x_u'^2+y_u'^2}}\,du,\qquad E_\theta=\int_0^T(\dot\theta_1^2+\dot\theta_2^2)\,dt,$$
$$A=\int_0^T(\ddot\theta_1^2+\ddot\theta_2^2)\,dt,\qquad J=\int_0^T(\dddot\theta_1^2+\dddot\theta_2^2)\,dt,$$
$$K_\epsilon=\int_0^1\frac{{(x_u'y_u''-y_u'x_u'')^2}}{{(x_u'^2+y_u'^2+\epsilon^2)^{{5/2}}}}\,du,$$
$$O=\int_0^T\frac{{1}}{{\max(c(t),\epsilon_c)^2}}\,dt\quad\text{{(장애물 없음: 0)}}.$$

Kε는 정칙화된 곡률 제곱 비용이다. 단순 ∫κ ds와 같지 않다. c(t)는 두 링크 표면과 장애물 사이의 최소 여유이다.
종합 비용 C=100(0.25 nL+0.20 nE+0.20 nJ+0.20 nK+0.15 nO), 보상 R=100−C. A는 종합에서 제외한다.
원 비용은 작을수록 좋다. 무효 후보의 1e6은 벌점 표시이며 정상 비용 통계에 넣지 않는다.

## 8. 기계 검증과 수치적 한계

이번 분석에서 {check['winner_labels_replayed']:,}개 주 집합 승자·동점 라벨을 원 저장 라벨과 대조했고 일치했다.
6종류 대표 사례의 L/Eθ/A/J/Kε/O 재적분 36건은 저장된 요약과 일치했다(허용오차는 스크립트에 기록).
보존 대상 {check['preserved_files']}개 파일의 작업 전후 SHA256이 같았다. 별도 artifact 검증 결과는 artifact_checks.json에 기록한다.
이는 탐색적 결과의 통계적 확실성을 보증하는 검사가 아니다.

이전 완료 검증은 5,500개 파티션의 기계적 무결성, train 가중 재적분·FK, 96개 확대 사례의 E/A/J 교차 검산을 포함했다.
해당 근거는 ../phase4_v3_release_quality/release_report.md 및 verification.json이다. 이번에 전체 수치 검증을 새로 수행했다고 주장하지 않는다.
161점 E/A/J 사다리꼴 적분은 최종 값이 아니다. 새 관절 지표는 적응형 가중합을 사용해야 한다.
L/Kε/O는 161점 값을 쓰되 321·641점과 고정 허용오차 비교를 통과한 후보만 유효로 사용했다.
641점이나 적응형 값도 참값이라고 단정하지 않는다. 극단적인 특이점, 더 좁은 장애물 구간, 물리 모델 차이는 추가 검증 대상이다.
joint_derivative_reliable은 특이점 표본 진단일 뿐 정확도 보증이 아니다.

## 9. 재실행과 향후 단계

```powershell
.\.venv\Scripts\python.exe scripts/analyze_phase4_v3.py
.\.venv\Scripts\python.exe scripts/report_phase4_v3_analysis.py
node scripts/build_phase4_v3_workbook.mjs
.\.venv\Scripts\python.exe scripts/verify_phase4_v3_analysis.py
```

Python 의존성: numpy, pandas, pyarrow, matplotlib. Excel 생성은 번들 Node의 @oai/artifact-tool을 사용한다.
실제 Python/라이브러리 버전, 입력 manifest·규칙·스크립트 해시는 provenance.json에 남겼다.
Excel은 요약표와 사전만 담으며 긴 식과 전체 시간표본은 넣지 않았다. 계산형 비율은 Excel 수식으로 재계산한다.

다음 모델 단계 전: 미확정 상황 처리 정책 유지, 기하 입력만 특징으로 사용, 평가값/유효 여부/정규화/승자 정보의 특징 누출 차단,
scenario_id별 분할 유지, validation에서 모델 선택 후 test/challenge를 최종 평가에만 사용해야 한다.
새로운 생성 시드·환경에서의 일반화를 확인하고 입력 범위를 벗어나면 재검사/추천 보류가 필요하다.
많은 데이터가 과적합을 자동으로 해결하지 않는다. 추천 경로는 별도 시뮬레이션과 수치·충돌 재검사를 거쳐야 한다.

## 그림별 한글 캡션

'''
    report+='\n\n'.join(f'### {r.figure}\n\n{r.caption_ko}\n\n![{r.figure}]({r.figure}.png)' for r in caps.itertuples())
    (OUT/'analysis_report.md').write_text(report,encoding='utf-8')
    dictionary=[
        ['scenario_id','동일 환경의 ID','없음','후보들은 대응 자료'],['run_id','곡선 변형 × 팔꿈치 해','없음','버전과 함께 결합'],
        ['selection_valid','수치·경로 검사 통과','bool','accepted만 true; 정확도/물리 안전 인증 아님'],
        ['cohort','main / excluded_unresolved / excluded_no_valid','분석 집합','구성·유효성은 전체, 우열 비교는 main'],
        ['fully_reachable','전체 경로 도달 검사 결과','true/false/null','null은 미확정, false가 아님'],
        ['collision_free','링크 무충돌 검사 결과','true/false/null','검사 순서 때문에 다른 위반이 가려질 수 있음'],
        ['L','끝점 경로 길이','unit','161 기하 표본; 321/641 비교 통과'],
        ['E_theta','관절 속도 제곱 적분','rad²/s','실제 에너지 아님; 적응형 가중합'],
        ['A','관절 가속도 제곱 적분','rad²/s³','보조 지표, 종합에서 제외'],
        ['J','관절 각저크 제곱 적분','rad²/s⁵','곡률과 다름; 적응형 가중합'],
        ['K_epsilon','정칙화 곡률 제곱 비용','1/unit','단순 총곡률과 다름'],
        ['O','장애물 접근 벌점','s/unit²','장애물 없음은 0, 여유는 NA'],
        ['balanced_cost','가중 정규화 비용','0–100','작을수록 좋음; 무효 1e6 제외'],
        ['fractional_wins','동점 후보별 1/k 배분 합','상황','확률/오차 보정 아님'],
        ['paired_scenarios','두 종류가 모두 유효한 같은 상황','상황','목적별 최소 비용 후보 대응'],
        ['small_n','분모 상황 수 <30','bool','소표본 표시, 유의성 검사 아님'],
        ['곡선·관절각 식','representative_equations.json','CSV/JSON 원본 참조','긴 식은 Excel에서 생략'],
        ['출처','data/phase4_v3_release train','phase4-mathematics-v3','완료 receipt와 manifest는 provenance.json 참조'],
        ['분석 규칙','docs/phase4_v3_analysis_protocol.md',str(SEED),'고정 해시 사례 선택; held-out 미사용'],
    ]
    table('analysis_dictionary',pd.DataFrame(dictionary,columns=['field','meaning_ko','unit','notes']))
    payload={'sheets':[],'scenario_count':n,'main_scenarios':nm}
    def sheet(name,title,note,headers,rows,widths,formats=None,ratios=None):
        df=pd.DataFrame(rows).astype(object);rows=df.where(pd.notna(df),None).values.tolist()
        payload['sheets'].append(dict(name=name,title=title,note=note,headers=headers,rows=rows,widths=widths,formats=formats or {},ratios=ratios or {}))
    sheet('유효성','곡선별 후보 확보율',f'Train {n:,} 상황. 주 비교 {nm:,}. 미확정 상황 {nu}, 유효 없음 {nn} 제외.',
        ['곡선','전체 후보','도달 확인','무충돌 확인¹','유효 후보','도달률','조건부 무충돌률¹','유효률','유효≥1 상황','상황 분모','상황 확보율'],
        [[r.curve_name,r.candidates,r.reachable,r.collision_free_among_reachable,r.valid,None,None,None,r.scenarios_any_valid,n,None] for r in v.itertuples()],
        [135,110,110,130,110,100,155,100,130,110,120],{'F':'0.0%','G':'0.0%','H':'0.0%','K':'0.0%'}, {'F':['C','B'],'G':['D','C'],'H':['E','B'],'K':['I','J']})
    sheet('목적별선택','평가기준별 선택과 동점','main: 미확정 대안 없는 유효 상황. obstacles_only: 그중 장애물 있음. 동점은 후보별 1/k.',
        ['범위','목적','곡선','결정적 승수','동점 배분 승수','상황 분모','결정적 비율','동점 배분 비율'],
        [[r.scope,LABELS[r.objective],NAMES[r.curve_key],r.deterministic_wins,r.fractional_wins,r.denominator_scenarios,None,None] for r in wins.itertuples()],
        [175,180,130,120,150,115,140,145],{'E':'0.00','G':'0.0%','H':'0.0%'},{'G':['D','F'],'H':['E','F']})
    sheet('격자진단','시간·검증 격자의 보조 진단','각 161점. 표본 사이 위반을 놓칠 수 있으므로 최종 유효성·연속 안전성과 구분.',
        ['곡선','전체 후보','격자 도달','격자 무충돌¹','격자 도달률','조건부 무충돌률¹','도달·무충돌률'],
        [[NAMES[r.curve_key],r.candidates,r.grid_reachable,r.grid_clear,None,None,None] for r in grid.itertuples()],
        [155,120,120,145,145,170,160],{'E':'0.0%','F':'0.0%','G':'0.0%'},{'E':['C','B'],'F':['D','C'],'G':['D','B']})
    sheet('구성과제외','분석 집합과 시나리오 구성','구성은 전체 train 기준. 후보가 아닌 상황 단위. 작은 구간은 별도 표시.',
        ['변수','범주 / 구간','상황 수','전체 상황 분모','구성 비율','소표본 n<30'],
        composition[['variable','category','scenarios','denominator','fraction','small_n']].values.tolist(),
        [205,310,115,150,120,130],{'E':'0.0%'},{'E':['C','D']})
    sheet('대응비교','같은 상황의 목적별 최소 비용 비교','차이 = A − B (목적 단위). 동점은 1e-12. n은 두 종류 모두 유효한 상황 수.',
        ['목적','단위','곡선 A','곡선 B','대응 상황 n','차이 중앙값','A 우세 비율','동점 비율','소표본'],
        [[LABELS[r.objective],r.unit,NAMES[r.family_a],NAMES[r.family_b],r.paired_scenarios,r.median_a_minus_b,r.a_lower_fraction,r.tie_fraction,r.small_n] for r in paired.itertuples()],
        [175,115,130,130,130,170,130,120,100],{'F':'0.000E+00','G':'0.0%','H':'0.0%'})
    sheet('조건별확보율','조건별 유효 후보 확보율','분모: 해당 조건의 모든 train 상황. 미확정/실패 상황도 분모에 포함.',
        ['조건','배치 / 구간','곡선','상황 분모','유효≥1 상황','확보율','소표본'],
        [[r.condition,r.bin,NAMES[r.curve_key],r.scenarios,r.scenarios_any_valid,None,r.small_n] for r in conditions.itertuples()],
        [195,280,135,120,140,110,100],{'F':'0.0%'},{'F':['E','D']})
    sheet('관계','시나리오당 후보 하나의 지표 관계','고정 해시 선택. 상관은 기술통계이며 인과관계/통계적 유의성 주장이 아니다.',
        ['범위','x','y','상황 n','Spearman ρ'],rel.values.tolist(),[180,390,390,110,140],{'E':'0.000'})
    sheet('데이터사전','정의·단위·분석 범위','원자료는 수정하지 않음. 전체 시간표본과 긴 수식은 CSV/JSON 참조.',
        ['필드','의미 / 출처','단위 / 버전','해석'],dictionary,[215,375,240,440])
    write_json(OUT/'workbook_payload.json',payload)
    print('Report and workbook payload ready',flush=True)


if __name__=='__main__':main()
