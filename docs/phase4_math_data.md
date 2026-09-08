# 4단계 수학 중심 경로 데이터 명세

## 1. 이 단계의 목적과 범위

4단계의 목적은 2차원 2관절 로봇팔의 후보 경로를 **같은 수학적 기준으로 반복 평가하고, 계산 과정을 시간 표본까지 남기는 것**이다. 질량, 중력, 모터 상수와 토크를 가정하는 3단계 공학 모델과 달리, 4단계의 핵심 지표는 곡선과 관절각 함수의 미분·적분만으로 정의한다.

이 단계가 만드는 것은 향후 분석에 쓸 원자료와 목적별 정답 레이블이다. 다음 기능은 아직 하지 않는다.

- 새로운 사용자 입력에 대한 머신러닝 예측
- 후보군 밖의 곡선을 찾는 연속 최적화
- 통계 분석, 최종 표·그래프 또는 포스터 자동 작성
- 실제 모터 전력·토크·열 발생량의 예측
- 3차원 또는 원통좌표계 로봇팔 계산

GUI 역시 42개 후보를 자동 비교하는 화면이 아니라, 사용자가 고른 **곡선 하나와 팔꿈치 해 하나**를 자세히 시뮬레이션하는 화면이다.

## 2. 경로, 시간함수와 관절각

### 2.1 끝점 곡선

끝점 경로를 다음 매개곡선으로 둔다.

\[
\mathbf p(u)=(x(u),y(u)),\qquad 0\le u\le 1.
\]

후보 곡선 종류는 직선, 원호, 3차 베지어, 3차 에르미트, 사이클로이드, 리마송이다. 데이터 생성에서는 직선 1개와 나머지 다섯 종류의 굽힘값

\[
-0.8,\;-0.4,\;0.4,\;0.8
\]

을 사용한다. 따라서 곡선 변형은 `1 + 5 x 4 = 21개`이고, 각 곡선에서 팔꿈치 위·아래 해를 계산하므로 시나리오당 후보는 `21 x 2 = 42개`이다.

### 2.2 최소 저크 시간함수

같은 곡선이라도 이동시간에 따라 관절 속도·가속도·저크 값이 달라진다. 모든 후보는 총 이동시간 \(T\) 동안 다음 진행률을 공통으로 사용한다.

\[
s=\frac{t}{T},\qquad
q(s)=10s^3-15s^4+6s^5,\qquad
\mathbf p(t)=\mathbf p(q(t)),\qquad 0\le t\le T.
\]

이 5차 함수는 시작과 끝에서 속도와 가속도가 0이 되는 최소 저크 보간 함수다. 기본값은 \(T=3\,\mathrm{s}\)이며 CLI에서 바꿀 수 있다. GUI의 재생 속도 슬라이더는 화면 재생만 빠르게 하거나 느리게 하며 평가에 쓰는 \(T\)는 바꾸지 않는다.

### 2.3 2R 역기구학과 `theta(t)`

밑부분을 \((b_x,b_y)\), 두 팔 길이를 \(L_1,L_2\)라 하고

\[
X(t)=x(t)-b_x,\qquad Y(t)=y(t)-b_y
\]

로 두면

\[
c_2(t)=\frac{X(t)^2+Y(t)^2-L_1^2-L_2^2}{2L_1L_2},
\]

\[
\theta_2(t)=\pm\arccos(c_2(t)),
\]

\[
\theta_1(t)=\operatorname{atan2}(Y(t),X(t))-
\operatorname{atan2}\!\left(L_2\sin\theta_2(t),
L_1+L_2\cos\theta_2(t)\right).
\]

구현에서 팔꿈치 위는 \(-\arccos(c_2)\), 팔꿈치 아래는 \(+\arccos(c_2)\)를 사용한다. `atan2(y,x)`의 쉼표는 두 입력 \(y,x\)를 구분하며, 사분면 정보를 보존하기 위해 한 변수짜리 \(\arctan(y/x)\) 대신 사용한다.

출력 수식은 `acos`의 주값과 `atan2`의 주값으로 정의한다. 실제 시간 표본의 관절각은 이 주값을 계산한 뒤, 인접 표본에서 \(\pi\)보다 큰 불연속 점프가 생기면 \(2\pi\)의 정수배를 더하거나 빼서 연속적으로 펼친다. 이 정책은 `branch_policy`, `unwrap_policy`, `equation_descriptor_json`에도 기록된다.

## 3. 평가함수

모든 핵심 지표는 **작을수록 좋은 비용**이다. 아래에서 `unit`은 입력 좌표와 팔 길이에 공통으로 사용한 작업공간 단위다.

### 3.1 경로 길이 \(L\)

\[
L=\int_0^1\left\lVert\mathbf p'(u)\right\rVert\,du
=\int_0^1\sqrt{x'(u)^2+y'(u)^2}\,du.
\]

`path_length_units`에 저장된다. 평면의 속도 벡터 \((x'(u),y'(u))\)의 유클리드 노름을 적분한 값이며 단위는 `unit`이다.

### 3.2 관절 운동 노력도 \(E_\theta\)

\[
E_\theta=\int_0^T\left(\dot\theta_1(t)^2+
\dot\theta_2(t)^2\right)dt.
\]

`joint_velocity_effort`에 저장되며 단위 표기는 \(\mathrm{rad^2/s}\)이다. 관절각이 시간에 따라 얼마나 빠르게 많이 변하는지를 재는 수학적 대리지표다.

이 값에는 토크 \(\tau\), 전력 \(\tau\dot\theta\), 질량, 중력 또는 모터 효율이 들어가지 않는다. 따라서 **줄(J) 단위의 기계적 에너지나 전기에너지가 아니며**, 보고서에서도 `에너지`라고 단정하지 말고 `관절 운동 노력도`라고 부르는 것이 정확하다.

### 3.3 각가속도 제곱 적분 \(A\)

\[
A=\int_0^T\left(\ddot\theta_1(t)^2+
\ddot\theta_2(t)^2\right)dt.
\]

`joint_acceleration_effort`에 저장되며 단위 표기는 \(\mathrm{rad^2/s^3}\)이다. 급격한 각속도 변화를 설명하는 **보조 지표**다. 현재 종합 점수의 가중치와 목적별 승자 레이블에는 포함되지 않는다.

### 3.4 각저크 제곱 적분 \(J\)

\[
J=\int_0^T\left(\dddot\theta_1(t)^2+
\dddot\theta_2(t)^2\right)dt.
\]

`joint_jerk_effort`에 저장되며 단위 표기는 \(\mathrm{rad^2/s^5}\)이다. 저크는 각가속도 자체가 아니라 **각가속도의 시간 미분**, 즉 관절각의 3차 시간미분이다. 이 값이 작으면 관절 운동의 급격한 변화가 상대적으로 적다.

### 3.5 곡률 제곱 적분 \(K\)와 정칙화 \(K_\varepsilon\)

평면 매개곡선의 곡률은

\[
\kappa(u)=\frac{|x'(u)y''(u)-y'(u)x''(u)|}
{(x'(u)^2+y'(u)^2)^{3/2}}
\]

이고, 보통의 곡률 제곱 적분은

\[
K=\int\kappa^2\,ds
=\int_0^1\frac{(x'y''-y'x'')^2}
{(x'^2+y'^2)^{5/2}}\,du
\]

이다. 여기서 \(x'y''-y'x''\)는 2차원 벡터를 억지로 3차원으로 바꾸지 않아도 계산할 수 있는 행렬식이다.

사이클로이드 후보는 양 끝에서 \(\lVert\mathbf p'(u)\rVert=0\)인 첨점을 가지므로 보통의 \(K\)가 발산할 수 있다. 0으로 나누는 문제를 숨기기 위해 0을 대입하지 않고, 모든 후보의 비교에는 다음 정칙화 값을 사용한다.

\[
K_\varepsilon=
\int_0^1\frac{(x'y''-y'x'')^2}
{(x'^2+y'^2+\varepsilon_v^2)^{5/2}}\,du,
\qquad
\varepsilon_v=r_\varepsilon\,\lVert\mathbf p(1)-\mathbf p(0)\rVert.
\]

기본 \(r_\varepsilon\)은 \(10^{-3}\)이다. 선택에 쓰는 열은 `regularized_curvature_energy_inv_unit`이고, GUI 호환 표시 열 `curvature_energy_inv_unit`도 같은 값을 담는다. 단위는 `1/unit`이다.

`unregularized_curvature_energy_inv_unit`에는 경로가 정칙이라고 판정된 경우에만 보통의 \(K\)를 저장한다. 사이클로이드는 `path_regular=false`, 보통의 \(K\)는 빈값이고 경고에 `non_regular_curve_regularized_curvature`가 남는다. 다른 곡선도 표본의 최소 속도가 \(\varepsilon_v\) 이하이면 보통의 \(K\)를 기록하지 않는다. 따라서 보고서의 후보 비교 그래프에서는 이 지표를 그냥 \(K\)라고 쓰기보다 **정칙화 곡률 비용 \(K_\varepsilon\)**이라고 밝혀야 한다.

### 3.6 장애물 접근 벌점 \(O\)

시각 \(t\)에서 두 링크 선분과 모든 원형 장애물 사이의 최소 부호 있는 여유를 \(c(t)\)라 둔다. 구현에서는

\[
c(t)=\min(\text{장애물 중심-링크 선분 거리}
-r_{\mathrm{obs}}-r_{\mathrm{link}})
\]

로 계산한다. \(c(t)\le 0\)이면 해당 표본에서 링크가 장애물과 충돌한 것이다. 접근 벌점은

\[
O=\int_0^T\frac{1}{\max(c(t),\varepsilon_c)^2}\,dt
\]

이며 `obstacle_penalty_s_per_unit2`에 저장된다. 기본 링크 반지름과 \(\varepsilon_c\)는 각각 `0.05 unit`이고, 단위는 `s/unit²`이다. 장애물이 없으면 \(O=0\)이며 최소 여유는 빈값이다.

충돌 후보도 유한한 \(O\)를 가질 수 있지만, 충돌 여부는 점수보다 우선하는 하드 제약이므로 승자가 될 수 없다.

## 4. 수치계산 방법과 재검산

표본 수를 \(N\)이라 할 때 기본값은 161이다. 4단계 계산기는 내장된 여섯 곡선의 생성식을 직접 계산하고, 그 식의 1·2차 도함수를 해석적으로 사용한다. 따라서 곡선 좌표의 보간오차가 곡률에 증폭되는 문제를 줄였다. 다음 세 격자를 구분한다.

- 기하 적분 격자: \(s_i=i/(N-1)\), `geometry_u=(1-cos(pi*s_i))/2`인 끝점 밀집 격자에서 `geometry_x`, `geometry_y`, \(x'\), \(y'\), \(x''\), \(y''\)를 구하고 \(L\)과 \(K_\varepsilon\)를 적분한다. 사이클로이드 첨점 근처의 좁은 정칙화 구간을 균일 격자보다 잘 포착한다.
- 시간 격자: `time_s=iT/(N-1)`에서 `progress_u=q(t/T)`를 적용해 `x`, `y`, 관절각, \(E_\theta,A,J,O\)를 계산한다.
- 하드 제약 검증 격자: `validation_u=i/(N-1)`에서 도달 가능성과 링크 충돌을 한 번 더 검사한다. 최소 저크 시간 격자가 곡선 중간의 짧은 위반 구간을 건너뛸 위험을 줄이기 위한 별도 균일 \(u\) 검사다.

따라서 같은 행에 있는 `(x, y)`는 애니메이션용 시간 위치이고 `(geometry_x, geometry_y)`는 기하 적분용 끝점 밀집 \(u\) 위치다. 중간 표본에서 둘이 같지 않은 것은 오류가 아니다.

관절각의 1·2·3차 미분은 기본 7점 국소 다항 유한차분으로 계산하며, 끝에서는 한쪽 스텐실을 사용한다. 모든 적분은 복합 사다리꼴법을 쓴다. `trajectory_samples/`의 피적분함수 열을 사다리꼴법으로 다시 적분하면 요약 지표를 재현할 수 있게 설계되어 있다.

저장된 곡선식은 내장 곡선 생성식 자체를 사람이 읽을 수 있게 나타낸 것이다. 좌표와 도함수도 같은 식에서 직접 계산하며, 정적분에는 유한한 격자의 복합 사다리꼴법을 쓰므로 해석적 적분값과 작은 이산화 오차는 남는다.

## 5. 하드 제약과 점수

### 5.1 유효 후보

후보는 다음 두 조건을 모두 만족할 때만 `selection_valid=true`다.

1. 저장되는 최소 저크 시간 격자와 별도 균일 \(u\) 검증 격자 모두에서 끝점이 2R 작업공간 안에 있다.
2. 두 격자 모두에서 두 링크가 원형 장애물과 충돌하지 않는다.

한 표본이라도 도달 불가능하면 \(E_\theta,A,J\)와, 장애물이 있을 때의 \(O\)처럼 연속 관절 경로가 필요한 값은 빈값이며, 개별적으로 도달 가능한 표본의 관절각만 진단용으로 남는다. 장애물이 없는 경우의 \(O\)는 정의대로 0을 유지한다. 도달 불가능 또는 충돌 후보는 목적별 승자와 종합 승자에서 제외된다.

이 검사는 서로 다른 두 종류의 161개 이산 표본에 대한 검사다. `collision_free=true`는 **그 표본 사이까지 연속적으로 충돌이 없다는 수학적 증명**이 아니다. `collision_at_sample`은 저장 시간 격자, `validation_collision_at_sample`은 균일 \(u\) 검증 격자의 상태다.

### 5.2 시나리오별 최소-최대 정규화

서로 단위와 크기가 다른 다섯 비용 \(L,E_\theta,J,K_\varepsilon,O\)는 같은 시나리오의 유효 후보끼리만 정규화한다. 목적 \(j\)의 원값을 \(z_j\)라 하면

\[
n_j=\frac{z_j-z_{j,\min}}{z_{j,\max}-z_{j,\min}}.
\]

즉 그 시나리오에서 최소값은 0, 최소·최대의 정확한 중간값은 0.5, 최대값은 1이다. 이는 `0, 0.5, 1` 세 등급으로 양자화하는 방식이 아니라, 그 사이의 연속값도 갖는 최소-최대 정규화다. 모든 유효 후보의 값이 같으면 분모를 0으로 두지 않고 모두 0으로 처리한다.

무효 후보의 `normalized_*` 열은 빈값이고, 저장 편의를 위해 종합 비용은 `1,000,000`, 종합 보상은 0으로 둔다. 이 백만이라는 값은 물리량이나 유효 후보와 비교할 점수가 아니라 **선택 제외 표시**다.

### 5.3 종합 비용과 보상

\[
C=100\left(
0.25n_L+0.20n_{E_\theta}+0.20n_J+
0.20n_{K_\varepsilon}+0.15n_O
\right),
\]

\[
R=100-C.
\]

`balanced_cost_0_100`은 작을수록, `balanced_reward_0_100`은 클수록 좋다. 가중치는 거리 0.25, 관절 운동 노력도 0.20, 저크 0.20, 정칙화 곡률 0.20, 안전 0.15다. \(A\)는 보조 기록이라 이 식에 들어가지 않는다.

`scenario_winners.csv`는 시나리오마다 `distance`, `joint_effort`, `jerk`, `curvature`, `safety`, `balanced`의 6개 레이블을 긴 형식으로 저장한다. 유효 후보가 없으면 `winner_status=no_valid_candidate`다. 최솟값이 상대·절대 허용오차 \(10^{-12}\) 안에서 같으면 `tie_resolved`와 후보 수·ID를 남기고, 목적값, 경로 길이, 곡선 키, 굽힘값, 팔꿈치 모드, 실행 ID 순으로 결정적 순서를 적용한다.

정규화와 종합 점수는 42개 후보 집합에 상대적이다. 후보 곡선이나 굽힘값을 바꾸면 같은 물리 상황에서도 점수가 달라질 수 있다.

## 6. 시나리오, 분할과 과적합 방지

기본 실행은 난수 시드 `20260904`로 다음 데이터를 만든다.

| 구분 | 시나리오 수 | 분할 | 역할 |
|---|---:|---|---|
| 일반 분포 | 5,000 | train 3,000 / validation 1,000 / test 1,000 | 향후 모델 학습·선정·최종 평가 |
| OOD challenge | 500 | challenge 500 | 경계·극단 조건에 대한 별도 점검 |
| 합계 | 5,500 | 서로 겹치지 않음 | 후보 231,000개 |

일반 시나리오 인덱스는 기하 생성용 난수와 분리된 시드로 먼저 섞은 뒤 정확히 60%/20%/20%로 나눈다. 이 방식은 시나리오 유형의 주기와 분할 번호가 결합되는 편향을 막는다. 같은 시나리오의 42개 후보가 서로 다른 분할로 흩어지지 않으므로 후보 단위 무작위 분할에서 생기는 데이터 누출도 막는다.

일반 장애물 배치는 `no_obstacle`, `chord_blocker`, `double_offset`, `start_near`, `goal_near`, `narrow_passage`, `random_field`를 순환한다. 도전 세트는 `narrow_passage`, `near_base_gate`, `dense_random`, `chord_blocker`와 작업공간 경계·극단 팔 길이 비율 상황을 사용한다.

이 구조는 과적합 위험을 줄이지만 없애지는 않는다. 특히 다음을 별도로 확인해야 한다.

- test와 challenge는 모델·가중치 선택에 사용하지 않는다.
- 최종 모델 입력에는 시나리오 기하, 장애물, 곡선 종류·굽힘값만 사용한다.
- 도달·충돌 결과, 원 평가값, 정규화값, 점수, 승자 레이블을 입력 특징에 섞지 않는다.
- 여러 난수 시드와 사람이 설계한 새 장애물 배치에서도 결과를 재검증한다.

`challenge`는 코드가 만든 어려운 분포일 뿐 현실의 모든 OOD 상황을 대표한다는 보장은 없다. 데이터가 많다는 사실만으로 일반화가 증명되지는 않는다.

## 7. 출력 구조와 열

### 7.1 파일 구조

```text
data/phase4/
├─ scenarios.csv
├─ trajectory_metrics.csv
├─ scenario_winners.csv
├─ trajectory_samples/
│  ├─ train/part-*.parquet
│  ├─ validation/part-*.parquet
│  ├─ test/part-*.parquet
│  └─ challenge/part-*.parquet
├─ experiment_metadata.json
├─ data_dictionary.md
├─ dataset_manifest.json
├─ generation_state.json
└─ .batches/
   ├─ metrics-*.csv
   └─ winners-*.csv
```

`--sample-format csv`를 선택하면 표본 파일 확장자는 `.csv.gz`가 된다. 요약 CSV는 Excel에서 한글을 읽기 쉽도록 UTF-8 BOM으로 저장한다. `dataset_manifest.json`은 최종 공개 파일의 바이트 수와 SHA-256 해시를 기록한다. `.batches/`와 `generation_state.json`은 재개용 내부 파일이라 매니페스트에서 제외된다.

### 7.2 `scenarios.csv`

한 행이 독립된 로봇·환경 상황이다.

- 식별: `scenario_id`, `scenario_type`, `seed`, `data_split`, `dataset_partition`
- 로봇: `base_x`, `base_y`, `link1`, `link2`, `link_ratio`, `min_reach`, `max_reach`
- 끝점: `start_x`, `start_y`, `goal_x`, `goal_y`, `start_radius`, `goal_radius`, `chord_length`, `angular_separation_deg`
- 장애물: `obstacle_layout`, `obstacle_count`, `obstacles_json`

### 7.3 `trajectory_metrics.csv`

한 행이 곡선 변형 하나와 팔꿈치 해 하나의 전체 실행이다.

- 결합 키와 입력: `run_id`, `scenario_id`, `scenario_type`, `seed`, `data_split`, `dataset_partition`, 밑부분·팔 길이·시작점·도착점 열, `obstacle_layout`, `obstacle_count`, `obstacles_json`
- 후보: `curve_key`, `curve_name_ko`, `shape`, `elbow_mode`, `curve_parameters_json`
- 핵심 적분: `path_length_units`, `joint_velocity_effort`, `joint_jerk_effort`, `regularized_curvature_energy_inv_unit`, `obstacle_penalty_s_per_unit2`
- 보조·검산: `path_length_polyline_units`, `joint_acceleration_effort`, `curvature_energy_inv_unit`, `unregularized_curvature_energy_inv_unit`, `curvature_speed_epsilon_units_per_u`, `min_geometric_speed_units_per_u`, `path_regular`
- 유효성: `fully_reachable`, `time_grid_fully_reachable`, `validation_grid_fully_reachable`, 시간 격자와 균일 검증 격자의 도달 가능 표본 수·비율, `collision_free`, `time_collision_samples`, `validation_collision_samples`, `collision_samples`, `collision_ratio`, `min_obstacle_clearance_units`, `max_forward_kinematics_error_units`, `selection_valid`, `warning`
- 특이점 진단: `min_singularity_index`는 전체 시간 표본에서의 최소 \(|\sin\theta_2|\)이고, `joint_derivative_reliable`은 이 값이 설정 임계값 이상인지 나타낸다. 이는 고차 관절 미분의 수치 민감도를 걸러 보기 위한 진단값이지 엄밀한 오차 보증은 아니다.
- 계산법: `duration_s`, `samples`, `integration_method`, `curvature_definition`, `joint_derivative_method`, `time_law`
- 수식: `curve_equation_template`, `curve_equation`, `progress_equation`, `x_u_equation`, `y_u_equation`, `x_t_equation`, `y_t_equation`, `ik_cosine_equation`, `theta1_t_equation`, `theta2_t_equation`, `branch_policy`, `unwrap_policy`, `equation_format_version`, `equation_descriptor_json`
- 시나리오 상대 점수: `normalized_distance`, `normalized_joint_effort`, `normalized_jerk`, `normalized_curvature`, `normalized_safety`, `balanced_cost_0_100`, `balanced_reward_0_100`

`theta1_t_equation`과 `theta2_t_equation`은 억지로 회귀한 다항식이 아니라 곡선·시간함수·역기구학의 합성식을 문자열로 보존한다. 기계가 안정적으로 다시 읽어야 할 때는 버전이 붙은 `equation_descriptor_json`을 사용한다.

### 7.4 `trajectory_samples/`

한 행이 한 실행의 한 표본이다. 큰 파일에서 반복 문자열을 줄이기 위해 요약표보다 결합 키가 간결하다.

- 결합 키: `run_id`, `scenario_id`, `data_split`, `dataset_partition`, `curve_key`, `shape`, `elbow_mode`
- 시간 위치: `sample_index`, `time_s`, `normalized_time`, `progress_u`, `x`, `y`
- 기하 적분 격자: `geometry_u`, `geometry_x`, `geometry_y`, `dx_du`, `dy_du`, `d2x_du2`, `d2y_du2`, `path_speed_units_per_u`, `curvature_inv_unit`, `regularized_curvature_integrand_inv_unit`, `curvature_speed_epsilon_units_per_u`, `path_regular`
- 균일 검증 격자: `validation_u`, `validation_x`, `validation_y`, `validation_obstacle_clearance`, `validation_collision_at_sample`, `validation_reachable_at_sample`
- 관절각: `theta1_rad`, `theta2_rad`, `theta1_deg`, `theta2_deg`
- 시간미분: `theta1_velocity_rad_s`, `theta2_velocity_rad_s`, `theta1_acceleration_rad_s2`, `theta2_acceleration_rad_s2`, `theta1_jerk_rad_s3`, `theta2_jerk_rad_s3`
- 적분항: `joint_velocity_squared_rad2_s2`, `joint_acceleration_squared_rad2_s4`, `joint_jerk_squared_rad2_s6`, `obstacle_penalty_integrand_per_unit2`
- 상태: `obstacle_clearance`, `collision_at_sample`, `reachable_at_sample`

### 7.5 `scenario_winners.csv`

한 행이 한 시나리오의 한 목적 레이블이다. 주요 열은 `scenario_id`, `objective`, `winner_status`, `tie_count`, `tied_run_ids_json`, `winner_run_id`, 후보 종류·굽힘·팔꿈치 해, 원값·정규화값·종합점수, 곡선식과 관절각 합성식이다. 유효 후보가 하나도 없어도 6개의 빈 승자 행을 보존하므로, 단순히 행 수만 보고 모든 시나리오에 해가 있었다고 해석하면 안 된다.

## 8. 생성 명령과 재개

Python 3.11 이상에서 Parquet 의존성을 설치한다.

```powershell
py -m pip install -e ".[data]"
```

20개와 100개 점검 데이터로 실행·스키마·용량을 먼저 확인할 수 있다.

```powershell
py scripts/generate_phase4_data.py --output data/phase4_smoke_20 --scenarios 20 --challenge-scenarios 0 --samples 161 --batch-size 10 --workers 4 --sample-format parquet
py scripts/generate_phase4_data.py --output data/phase4_smoke_100 --scenarios 100 --challenge-scenarios 0 --samples 161 --batch-size 25 --workers 8 --sample-format parquet
```

기본 전체 데이터는 다음과 같다.

```powershell
py scripts/generate_phase4_data.py --output data/phase4 --workers 8 --sample-format parquet
```

기본값은 일반 5,000개, challenge 500개, 시드 20260904, 경로당 161표본, 배치당 25시나리오, 이동시간 3초다. 사용 가능한 옵션은 다음 명령으로 확인한다.

```powershell
py scripts/generate_phase4_data.py --help
```

주요 옵션은 `--scenarios`, `--challenge-scenarios`, `--seed`, `--samples`, `--batch-size`, `--workers`, `--duration`, `--link-radius`, `--clearance-epsilon`, `--curvature-epsilon-ratio`, `--sample-format`, `--output`, `--resume`이다. `--samples`는 최소 7이어야 하나, 3차 미분과 충돌 검사의 안정성을 위해 본 실험에서는 기본 161을 유지하는 편이 좋다. `--workers 1`은 메모리를 가장 적게 쓰고, 여러 코어가 있는 PC에서는 값을 높여 생성 시간을 줄일 수 있다.

출력 폴더가 비어 있지 않으면 실수로 덮어쓰지 않고 중단한다. 중단된 실행은 처음과 동일한 모든 옵션에 `--resume`을 추가한다.

```powershell
py scripts/generate_phase4_data.py --output data/phase4 --workers 8 --sample-format parquet --resume
```

데이터 버전, 개수, 시드, 표본 수, 배치 크기, 실제 저장 형식과 모든 평가 설정이 기존 `generation_state.json`과 같아야 재개된다. `auto`는 설치 상태에 따라 Parquet 또는 CSV로 달라질 수 있으므로 장기 실행에는 형식을 명시하는 것이 안전하다. 완료된 배치는 건너뛰고 미완료 배치부터 이어 간다.

기본 전체 실행은 231,000개 후보와 37,191,000개 표본 행을 쓴다. 압축 Parquet도 수 GB가 될 수 있고 CSV 계열은 더 커질 수 있으므로, 본 실행 전 100개 점검 결과를 바탕으로 시간과 디스크를 추정해야 한다. 배치 크기를 줄이면 재개 단위와 메모리 부담이 작아지지만 파일 수가 늘어난다.

### 8.1 완료된 전체 생성 결과

2026-09-04에 기본 시드와 161표본, 8개 작업자로 `data/phase4/` 전체 생성을 완료했다.

- 시나리오: 일반 5,000개 + challenge/OOD 500개 = 5,500개
- 후보 요약: 231,000행(각 시나리오당 42개)
- 승자 라벨: 33,000행(각 시나리오당 6개 목적)
- 시간표본: 37,191,000행
- 표본 분할: train 20,286,000 / validation 6,762,000 / test 6,762,000 / challenge 3,381,000행
- 유효 후보: 42,692개
- 하드 제약 실패: 도달 불가 84,866개 / 충돌 103,442개 / 기타 계산 실패 0개
- 유효 후보가 하나도 없는 시나리오: 1,726개이며, 이에 해당하는 빈 승자 라벨 10,356행도 보존됨
- 저장 크기: 약 7.33 GB(6.82 GiB), 실행 시간 약 41.3분

독립 검산에서 `scenario_id`와 `run_id` 중복 0건, 분할 누출 0건, 시작·끝점 불일치 0건, 유효 후보의 비유한 핵심 지표 0건을 확인했다. 매니페스트의 공개 파일 623개는 SHA-256 재계산 결과 모두 일치했고 전체 48개 단위 테스트가 통과했다. 이 수치는 데이터 품질 검증 결과이지 곡선별 우열에 대한 통계적 결론은 아니다.

## 9. GUI 사용법

```powershell
.\start_app.cmd
```

또는 다음과 같이 실행한다.

```powershell
py app.py
```

1. 밑부분 좌표, 두 팔 길이, 끝점 시작·도착 좌표와 이동시간을 입력한다.
2. 곡선 종류, 굽힘값과 팔꿈치 위·아래 해를 고른다.
3. 원형 장애물의 중심 `(x,y)`와 반지름을 입력해 여러 개를 추가하거나 선택·전체 삭제한다.
4. `경로 다시 계산`을 누르고 유효성, 경로 길이, 최소 여유와 작업공간 그림을 확인한다.
5. 재생·일시정지·처음 버튼과 시간 슬라이더로 두 링크 자세, 현재 \(t,u,\theta_1,\theta_2\), 충돌 상태를 살펴본다.
6. 오른쪽 탭에서 곡선식·시간함수·관절각 합성식, 여섯 적분값과 관절각 그래프를 확인한다.
7. JSON은 입력·장애물·수식·지표·모든 표본을, CSV는 현재 실행의 표본 행을 저장한다.

GUI는 161표본과 수학 코어의 기본 링크 반지름·정칙화 설정을 사용한다. 화면의 `K`는 보통의 \(K\)가 아니라 데이터 선택과 동일한 `curvature_energy_inv_unit = K_\varepsilon`이다. `Eθ`도 실제 줄(J)이 아닌 관절 운동 노력도다.

## 10. 보고서에 쓸 때의 해석 원칙

의미 있게 주장할 수 있는 내용은 다음과 같다.

- 동일한 시작·도착점에서도 곡선 종류와 팔꿈치 해에 따라 \(L,E_\theta,J,K_\varepsilon,O\)의 순위가 달라진다.
- 한 목적의 최적 후보가 다른 목적에서도 최적이라는 보장은 없으므로 상충관계를 비교할 수 있다.
- 시나리오별 정규화와 고정 가중치가 종합 선택에 어떤 영향을 주는지 분석할 수 있다.
- 시간별 \(\theta_i(t)\), 미분값과 피적분함수를 남겼으므로 대표 사례의 수치적분을 재검산할 수 있다.

추천할 만한 포스터 자료는 곡선 종류별 유효률, 목적별 승자 비율, 분할별 원 지표 분포, \(L\)-\(E_\theta\) 또는 \(J\)-\(K_\varepsilon\) 산점도, 장애물 여유와 종합점수의 관계, 대표 시나리오의 경로·관절각·피적분함수 그래프다. 통계의 독립 표본 단위는 후보 행이 아니라 `scenario_id`로 잡아야 한다.

다음 표현은 피해야 한다.

- `E_theta가 작으므로 실제 소비 전력이 몇 % 감소했다.`
- `161개 표본에서 충돌이 없으므로 연속 경로의 안전이 증명되었다.`
- `42개 중 최상이므로 가능한 모든 곡선 중 전역 최적이다.`
- `한 시드의 test 성능만으로 새로운 환경 일반화가 보장된다.`
- `정칙화 K_epsilon을 사이클로이드의 보통 곡률 적분 K라고 동일시한다.`

특이점 근처에서는 역기구학 관절각의 고차 미분이 매우 커지고 수치 저크가 표본 수에 민감할 수 있다. 따라서 `joint_derivative_reliable=false`인 후보의 \(A,J\)는 순위를 그대로 일반화하지 말고 별도 민감도 분석 대상으로 다뤄야 한다. 또한 `collision_samples`와 `collision_ratio`는 최소 저크 시간 격자와 균일 \(u\) 검증 격자, 즉 총 \(2N\)개의 판정 결과를 합친 값이다. 결과 보고서에는 표본 수, 이동시간, 유한차분, 사다리꼴법, 충돌의 이산 검사, 곡률 정칙화와 후보군 제한을 함께 밝혀야 한다.
