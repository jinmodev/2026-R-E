# Phase 4 수치 품질 검증 보고서

입력 `phase4-mathematics-v2` / `phase4-generator-v4`.
원자료 변경 없음. 분석 버전 phase4-audit-v1, 시드 20260908.
사전 기준: [검증 계획](../../docs/phase4_quality_analysis_protocol.md).

## 1. 실제 수행 범위

전체 분할은 기계적 검증만 수행했다. train 3,000개 시나리오·126,000개 후보와
20,286,000개 저장 시간표본에 대해서만 평가값 검산·탐색을 수행했다.
메타데이터의 평가값 분포나 held-out 승자 통계는 분석하지 않았다.
전체 시간표본을 한꺼번에 적재하지 않고 Parquet 파일별 필요한 열만 읽었다.
원자료 CSV는 청크를 읽고 train만 숫자로 변환했다.

- 기계적 검증: 21개 검사, FAIL 0개.
- train 지표·라벨 검증: 58개 검사, FAIL 0개.
- train 전체 표본 열 결측조건·무한대 검사: 48개 열, FAIL 0개.
- 코드 회귀 테스트: 55개, 전체 통과=True.
- 해석적 길이·곡률 검사: 1976건, 실패 0건.
- 다항 관절각 미분 검사: 12건, 실패 0건.
- 선택 사례 161표본 원자료 재현: 4788건, 실패 0건.

개별 실패를 포함한 모든 검사 항목: mechanical_checks.csv, train_integrity_checks.csv.
저장된 관절각을 독립적인 sin/cos 순기구학으로 복원한 최대 오차는
6.5346e-14 unit. 좌표 복원이 좋다는 것은 고차 미분 정확도 보증은 아니다.

### 저장된 피적분함수의 재적분

| metric | runs | failures | max_absolute_error |
| --- | --- | --- | --- |
| path_length_units | 126000 | 0 | 7.1054e-15 |
| regularized_curvature_energy_inv_unit | 126000 | 0 | 1.4211e-14 |
| joint_velocity_effort | 126000 | 0 | 3.2401e-12 |
| joint_acceleration_effort | 126000 | 0 | 3.4459e-08 |
| joint_jerk_effort | 126000 | 0 | 2.0981e-05 |
| obstacle_penalty_s_per_unit2 | 126000 | 0 | 2.9843e-13 |

재적분 일치는 저장·요약의 일관성 검증이며, 사다리꼴 이산화 자체의 정확도 증명이 아니다.

## 2. 표본 수 민감도

고정 규칙으로 선정한 19개 train 시나리오의 전 후보 798개를
81·161·321·641 표본으로 재계산했다. input/equation/branch/T/epsilon은 고정했다.
빈 선정 층은 selected_cases.csv에 available=0으로 남긴다.
641은 고해상도 비교 기준이며 참값이 아니다. 비확률 층화 대표 사례이므로
아래 통과율을 전체 train의 정확도 확률로 일반화하지 않는다.

| metric | runs | evaluable | converged | not_converged | unavailable | max_relative_error |
| --- | --- | --- | --- | --- | --- | --- |
| L | 798 | 798 | 798 | 0 | 0 | 6.0237e-05 |
| E_theta | 798 | 568 | 568 | 0 | 230 | 0.0067695 |
| A | 798 | 568 | 564 | 4 | 230 | 0.053799 |
| J | 798 | 568 | 558 | 10 | 230 | 0.56328 |
| K_epsilon | 798 | 798 | 798 | 0 | 0 | 0.0045899 |
| O | 798 | 662 | 662 | 0 | 136 | 0.020839 |

converged는 161→641과 321→641 모두 `atol+rtol*abs(reference)` 이내인 경우다.
상대오차와 절대오차는 convergence_detail.csv에 함께 기록했다. 기준값 0인
상대오차는 NA이며 절대오차로 판정했다. 계산 불가 NA는 실패로 억지 대체하지 않았다.
목적별 기준은 다음과 같다.

| metric | atol | rtol |
| --- | --- | --- |
| L | 1e-06 | 0.005 |
| E_theta | 1e-06 | 0.01 |
| A | 1e-05 | 0.02 |
| J | 0.001 | 0.05 |
| K_epsilon | 1e-06 | 0.02 |
| O | 0.0001 | 0.05 |

161·321·641 사이 하드 유효성 상태가 달라진 실행은 0개다.

| samples | runs | reachable | valid |
| --- | --- | --- | --- |
| 81 | 798 | 568 | 289 |
| 161 | 798 | 568 | 289 |
| 321 | 798 | 568 | 289 |
| 641 | 798 | 568 | 289 |

### 목적별 순위 안정성

| objective | cases | verified | same_winner | same_tie_set |
| --- | --- | --- | --- | --- |
| balanced | 19 | 14 | 19 | 19 |
| curvature | 19 | 15 | 19 | 19 |
| distance | 19 | 15 | 19 | 19 |
| jerk | 19 | 14 | 19 | 19 |
| joint_effort | 19 | 15 | 19 | 19 |
| safety | 19 | 15 | 19 | 19 |

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
4건 관찰됐다.
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
