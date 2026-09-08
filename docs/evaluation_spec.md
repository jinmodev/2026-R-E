# 3단계 다목적 평가함수 명세

## 연구 질문

같은 시작점과 끝점을 연결하는 곡선이라도 거리, 모델 추정 에너지, 저크, 곡률, 특이점 위험, 장애물 안전성에서 서로 다른 성능을 보이는가? 사용 목적의 가중치가 바뀌면 최적 곡선 계열은 어떻게 달라지는가?

## 공정한 시간 비교

GUI와 기존 데이터의 좌표는 무차원 작업공간 단위다. 물리량 계산에서는 기본적으로 `1 좌표 단위 = 0.1 m`로 환산한다. 따라서 길이 2~5인 링크는 실제 모델에서 0.2~0.5 m로 해석한다. 이 값은 `workspace_unit_m`으로 변경할 수 있고 데이터 메타데이터에 저장된다.

곡선의 기하학적 매개변수 `u`와 실제 시간 `t`를 구분한다. 모든 후보에 같은 이동 시간 `T`와 다음 최소-저크 진행률을 적용한다.

```text
s = t / T
u(s) = 10s³ - 15s⁴ + 6s⁵
```

이 시간법칙은 시작과 끝에서 속도와 가속도가 0이다. 관절각 `theta(u)`는 자연 3차 스플라인으로 보간한 뒤 동일한 시간격자에서 속도, 가속도, 저크를 계산한다. 후보마다 이동 시간을 다르게 하면 에너지와 저크를 공정하게 비교할 수 없으므로 기본 데이터에서는 `T=3 s`로 고정한다.

## 1. 끝점 경로 거리

```text
L = integral(||p'(u)|| du),  0 <= u <= 1
```

CSV 열: `curve_length`(작업공간 단위), `curve_length_m`(m)

## 2. 모델 추정 에너지

로봇팔은 균일한 가느다란 막대 두 개, 관절은 직접 구동 모터로 가정한다. 기본값은 수평면 운동이므로 중력항을 제외하고, 수직면 옵션에서는 중력항을 포함한다.

```text
tau = M(theta) theta_ddot + C(theta, theta_dot) + G(theta) + B theta_dot
```

양의 기계적 일과 모터 권선의 구리손을 구분한다.

```text
E_mech = integral(sum(max(tau_i * theta_dot_i, 0)) dt)
tau_i = K_t I_i
E_copper = integral(sum(I_i² R) dt)
         = (R / K_t²) integral(sum(tau_i²) dt)
E_model = E_mech + E_copper
```

CSV 열:

- `positive_mechanical_energy_j`
- `motor_copper_loss_j`
- `model_electrical_energy_j`
- `rms_joint_torque_nm`, `max_joint_torque_nm`

이는 모터·감속기·드라이버를 실측한 값이 아니다. 포스터와 보고서에서는 반드시 **모델 추정 에너지**라고 표기한다. 실제 장치가 준비되면 전압과 전류를 측정한 `integral(VI dt)`로 검증한다.

## 3. 저크 기반 부드러움

저크는 관절각을 시간에 대해 세 번 미분한 값이다. 저크가 일정한지를 목표로 삼지 않고, 저크의 크기와 변동이 작은지를 평가한다.

```text
J = integral(theta1_jerk² + theta2_jerk² dt)
```

CSV 열:

- `joint_jerk_effort_rad2_per_s5`
- `joint_jerk_rms_rad_per_s3`
- `joint_jerk_variation_rad_per_s3`

## 4. 곡률

곡률은 시간과 무관한 경로의 기하학적 굽음이다. 연속식 `kappa = |x'y''-y'x''|/(x'²+y'²)^(3/2)`에 대응하도록 세 점의 외접원 곡률을 사용한다.

```text
K = integral(kappa(s)² ds)
```

CSV 열:

- `mean_abs_curvature_inv_m`
- `max_curvature_inv_m`
- `curvature_energy_inv_m`
- `curvature_variation_inv_m`

## 5. 장애물 안전성

장애물은 원, 로봇 링크는 반지름을 가진 선분으로 모델링한다. 모든 표본 자세에서 밑부분–팔꿈치, 팔꿈치–끝점 두 링크와 장애물 사이의 부호 있는 최소 여유 거리를 계산한다.

```text
C = integral(1 / max(clearance, epsilon)² dt)
```

CSV 열:

- `min_obstacle_clearance_m`
- `clearance_penalty_s_per_m2`
- `collision_samples`, `collision_ratio`, `collision_free`

충돌 경로는 가중치와 관계없이 최적 후보에서 제외한다.

## 6. 특이점 안정성

2R 로봇팔에서 `|sin(theta2)|`가 0에 가까우면 팔이 완전히 펴지거나 접힌 특이 자세에 가깝다.

```text
S = integral(1 / (|sin(theta2)| + epsilon)² dt)
```

CSV 열: `min_singularity_index`, `singularity_penalty_s`

## 7. 종합비용과 보상

단위가 서로 다른 원자료를 같은 시나리오의 유효 후보 안에서 0~1 최소–최대 정규화한다.

```text
f_k* = (f_k - min(f_k)) / (max(f_k) - min(f_k))
Cost = 100 * sum(w_k f_k*) / sum(w_k)
Reward = 100 - Cost
```

비용은 낮을수록, 보상은 높을수록 좋다. 도달 불가능하거나 충돌한 경로는 `Cost=1,000,000`, `Reward=0`으로 처리한다.

기본 프로필:

- `balanced`: 거리 20%, 에너지 25%, 저크 20%, 곡률 10%, 특이점 10%, 안전성 15%
- `shortest`: 거리 55%
- `energy_saving`: 에너지 55%
- `smooth_precision`: 저크 40%, 곡률 20%
- `safety_first`: 안전성 50%, 특이점 15%

종합점수 하나만 보고 결론을 내리지 않는다. 원자료의 상관관계, 프로필별 최적 곡선 변화, 파레토 전선을 함께 분석해야 가중치에 의한 결론 왜곡을 확인할 수 있다.

## 데이터 분할과 누출 방지

같은 `scenario_id`에서 생성된 42개 곡선·팔꿈치 후보는 서로 독립된 표본이 아니다. 따라서 개별 실행 행을 무작위로 나누지 않고 시나리오 전체를 `train 60% / validation 20% / test 20%`로 고정한다. `data_split` 열에 분할이 기록되며 같은 시나리오의 모든 후보는 반드시 같은 분할에 속한다. 향후 예측 모델의 정규화 범위와 하이퍼파라미터는 train에서 정하고 validation에서 선택한 뒤 test는 마지막 성능 보고에 한 번만 사용한다.

## 현재 한계

- 링크 밀도·반지름·모터 상수는 비교용 가정값이며 실물 파라미터가 아니다.
- 원형 장애물과 링크 충돌은 이산 표본에서 검사한다.
- 후보 집합 내부 정규화이므로 곡선 변형을 추가하면 종합점수가 달라질 수 있다.
- 현재 모델에는 관절각·속도·토크 제한, 감속기 효율, 회생제동, 반도체 스위칭 손실이 없다.
- 현재 종합점수는 한 시나리오 내부 후보 선택용이다. 여러 시나리오를 학습하는 예측 모델에서는 정규화 범위를 train에서만 추정해야 한다.
