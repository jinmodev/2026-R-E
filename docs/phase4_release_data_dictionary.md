# Phase 4 v3 릴리스 데이터 사용법

## 저장 구조

`data/phase4_v3_release/partitions/{split}/{scenario_id}/done.json`이
해당 시나리오의 완료 attempt와 파일 크기·SHA-256을 가리킨다.
미완료 attempt가 있더라도 done.json이 지정하지 않은 파일은 분석에 쓰지 않는다.
`generation_state.json`의 status가 `verified_complete`인 경우에만 완료 데이터다.

각 attempt:

- `trajectory_metrics.jsonl.gz`: 후보 42행. 입력, 곡선식, theta(t) 합성식, 비용, 상태 및 검산값.
- `trajectory_samples.parquet`: 후보별 161행. 시간, 끝점, 관절각, 독립 미분과 격자 피적분함수.
- `adaptive_panels.jsonl.gz`: 관절 적분의 가변 구간. metric별 time_s, weights_s, integrand.
- `scenario_winners.json`: 목적별 6행. distance / joint_effort / jerk / curvature / safety / balanced.

조인 키는 `(dataset_version, scenario_id, run_id)`이며 시간표본에는 sample_index가 추가된다.
Parquet에는 버전 열을 반복하지 않으므로 반드시 상위 데이터 버전과 함께 읽는다.
원 v2·v3-pilot과 run_id가 같아도 서로 다른 계산 결과다. 설명 없이 합치지 않는다.

## 중요한 열

| 열 | 의미·단위 |
|---|---|
| path_length_units | L, 작업공간 길이 단위 |
| joint_velocity_effort | Eθ, rad²/s. 실제 소비 에너지 아님 |
| joint_acceleration_effort | A, rad²/s³. 보조 지표이며 종합점수 구성요소 아님 |
| joint_jerk_effort | J, rad²/s⁵ |
| regularized_curvature_energy_inv_unit | Kε, 작업공간 길이 단위의 역수 |
| obstacle_penalty_s_per_unit2 | O, s/작업공간 길이 단위² |
| selection_valid | 공통 후보군의 수치/도달/충돌 검사에 통과했는가 |
| candidate_status | accepted / invalid / unresolved |
| geometry_checks | L/Kε/O의 161·321·641점 값과 고정 허용오차, 통과 여부 |
| expanded_validation_quarantine | 확대 train 검산 불일치로 보류된 실행인가 |
| joint_integration_status | accepted / unresolved / not_evaluated |
| joint_sample_grid_integrals | 161 시간표본 사다리꼴 진단값. 최종 적분값과 구별 |
| legacy_v2 | 원 계산법의 재계산 진단값. 새 점수와 분리 |
| label_readiness | 미확정 대안이 있으면 provisional_unresolved_candidates |

accepted는 현재 모델·검사 기준을 통과했다는 뜻이지 엄밀한 수학적 정확도나 물리 안전 인증이 아니다.
invalid는 도달/충돌 위반을 목격했다는 뜻이며 관절 목적 적분은 생략하고 null을 남긴다.
unresolved는 특이점·적분 경고·수렴성 실패 등이다. 이미 구한 유한값이 남아 있어도 진단용이며
selection_valid=false이다. null을 0으로 채우지 않는다.
시나리오에 unresolved가 있으면 선택된 승자는 확인된 후보 중의 잠정 결과다.
no_valid_candidate는 준비된 42개 안에서 찾지 못했다는 뜻이며 환경의 경로 부재를 증명하지 않는다.

## 적분 재현

관절 지표는 같은 run_id/metric의 accepted 적분 구간에서
`sum(weights_s[i] * integrand[i])`를 합한 값이다.
L/Kε/O는 기존 161점 표본의 사다리꼴 적분이며 별도 geometry_checks로 수렴성을 확인한다.
관절 지표에 161점 사다리꼴 값을 대신 사용하면 새 점수를 재현할 수 없다.

## train 스트리밍 읽기

```python
from robot_planner.release_io import iter_release_metrics

for row in iter_release_metrics('data/phase4_v3_release'):
    if row['selection_valid']:
        # 여기서 필요한 train 열만 선택한다. 실제 분석은 다음 단계다.
        pass
```

독자는 기본으로 train만 읽는다. held-out은 명시적 별도 평가 권한 없이 열지 않는다.
전체 test/challenge 수치분포나 곡선 순위 분석은 이번 작업에서 수행하지 않았다.
