# Phase 4 수치·충돌 보강 결과

기계적 검증/탐색 진단 보고서. 기존 원자료·GUI·평가 라벨은 수정하지 않았다. 새 계산은 opt-in이다.
본 단계는 train만 읽었다. validation/test/challenge의 평가값을 사용하지 않았으며 모델을 학습하지 않았다.

## 범위와 사전 기준

이전 A/J 실패 10개와 곡선별 대조군 6개를 적응형 적분으로 검산했다. 실패가 속한 4개 시나리오의 168개 후보를 161/641/1281/2561점으로 비교했다.
E 1%, A 2%, J 5%와 절대 허용값은 기존 기준을 유지했다. 자세한 설정과 선택 이유는 docs/phase4_refinement_protocol.md 및 provenance.json 참조.
참조는 위치 방정식을 세 번 미분한 독립 역기구학과 적응형 적분이다. 수치 적분 오차 추정은 엄밀한 보증이 아니며 2561점도 참값이 아니다.

## 적분 결과

|집단|계산|표본|목적|통과/평가|최대 상대오차|
|---|---|---:|---|---:|---:|
|control|differential_grid|161|A|6/6|4.01575e-08|
|control|differential_grid|161|E_theta|6/6|3.97919e-12|
|control|differential_grid|161|J|6/6|0.000197237|
|control|differential_grid|2561|A|6/6|6.13488e-13|
|control|differential_grid|2561|E_theta|6/6|3.34008e-16|
|control|differential_grid|2561|J|6/6|7.70865e-07|
|control|legacy|161|A|6/6|2.05733e-05|
|control|legacy|161|E_theta|6/6|3.10359e-06|
|control|legacy|161|J|6/6|0.0084523|
|control|legacy|2561|A|6/6|1.7618e-10|
|control|legacy|2561|E_theta|6/6|2.01295e-13|
|control|legacy|2561|J|6/6|7.42561e-07|
|previous_failure|differential_grid|161|A|5/10|0.090113|
|previous_failure|differential_grid|161|E_theta|10/10|0.00424264|
|previous_failure|differential_grid|161|J|5/10|0.390368|
|previous_failure|differential_grid|2561|A|10/10|3.05746e-14|
|previous_failure|differential_grid|2561|E_theta|10/10|2.23941e-14|
|previous_failure|differential_grid|2561|J|10/10|4.70448e-11|
|previous_failure|legacy|161|A|6/10|0.0538577|
|previous_failure|legacy|161|E_theta|10/10|0.0067755|
|previous_failure|legacy|161|J|0/10|0.570668|
|previous_failure|legacy|2561|A|10/10|1.74151e-08|
|previous_failure|legacy|2561|E_theta|10/10|1.65286e-09|
|previous_failure|legacy|2561|J|10/10|7.98291e-05|

적응형 적분 경고/실패 실행: 0. 절대오차·적분기 추정오차·특이점 위치는 integral_accuracy.json / adaptive_integrals.json에 함께 저장했다.
특이점 부근에서는 관절각의 고계 미분이 짧은 시간에 크게 변할 수 있다. 표본 차분의 해상도 부족과 반올림 오차가 함께 작용할 수 있으므로 무조건 표본 수만 늘리는 정책은 채택하지 않는다.
이번 실패군에서 미분법만 바꾼 161점 적분도 A/J 일부가 허용범위를 벗어났다. 즉 미분 공식 개선만으로 충분하지 않고, 피적분함수가 급변하는 구간의 적분 해상도도 확보해야 한다.
이전 실패 실행 중 selection_valid였던 수는 2개다. 충돌/도달 실패 후보를 포함한 수치 진단이므로 10개 실패를 곧바로 10개 잘못된 승자 라벨로 해석하지 않는다.
joint_derivative_reliable은 특이점 진단값일 뿐 적분 정확도 보증이 아니다. 실행별 실제 오차와 함께 확인해야 한다.
후보 순위·승자 민감도는 rank_sensitivity.json. 기존 유효성 라벨을 고정한 진단이며, 새 충돌 판정과 섞어 최종 라벨로 사용하지 않는다.
고해상도 grid의 도달/충돌/유효성 상태가 달라진 실행 수: 0.
비교된 시나리오·목적·방법·해상도 조합 중 결정론적 승자 변화 0건, 동점 집합 변화 0건. 일부 사례의 승자가 유지되었다고 잘못된 적분값을 정당화할 수는 없다.
종합점수에는 같은 해상도의 기존 L/K/O와 새 E/J를 사용한 진단값을 별도 표기했다. A는 기존 종합점수의 구성요소가 아니다.

## 연속 구간에 대한 적응형 충돌 점검

19개 시나리오 798개 후보: {'invalid': 509, 'bounded_clear': 289}.
기존 유효 후보의 새 상태: {'bounded_clear': 289}. 새로 위반을 찾은 실행: 0.
도달 불가와 충돌을 모두 목격한 실행: 78. 기존 도달 불가 실행 중 충돌도 목격한 수: 78.
invalid는 위반 위치를 발견했다는 뜻이다. bounded_clear는 현재 2R·원형 장애물·고정 팔꿈치 가지 모델에서 구간 이동 상계가 여유보다 작음을 확인했다는 뜻이다. unresolved는 검사 한도/특이점 등으로 결론을 내리지 못했다는 뜻이며 유효 후보로 승인하지 않는다.
구간 상계는 binary64 계산이며 바깥쪽 반올림 구간 산술이 아니므로 엄밀한 수치 인증이나 실제 로봇의 안전 보증은 아니다. 위반 구간의 세부 검사는 중단하므로 발견되지 않은 다른 원인의 부재를 주장하지 않는다.
검사시간 1.47초, 평균 평가점 57.5, 최대 634점. 기기·경로 구성에 따라 비용은 달라진다.

## 유효 후보가 없는 train 환경

{'has_valid': 2147, 'mixed_unreachable_collision': 821, 'all_reachable_colliding': 32}
기존 분류는 도달 불가 우선의 상호 배타적 요약이다. 도달 가능한 후보의 충돌만 세므로 도달 불가 후보의 다른 구간 충돌이 가려질 수 있다. 새 목격 목록은 원인 중첩을 별도로 보여준다.
853개 환경에 준비된 42개 후보 중 유효 후보가 없다는 뜻이지, 환경에 경로 자체가 없다는 뜻이 아니다. 후보 보강 여부는 이후 별도 설계가 필요하다.

## 데이터 버전 및 다음 단계

원자료는 phase4-mathematics-v2, 새 진단은 phase4-refinement-v1이다. 신뢰성이 확인되지 않은 구버전 J/A나 종합 라벨을 확정 학습 정답으로 사용하지 않는다.
이번에 검산한 일부 train 결과를 전체 데이터의 정확도나 과적합 방지 보증으로 일반화하지 않는다. 최적의 범위는 준비된 42개 후보 안이다.
다음에는 생성기의 v3 정책(독립 미분/적응형 적분, 미확정 유효성 처리)을 정한 후 별도 경로에 버전 데이터와 라벨을 재생성해야 한다. 기존 GUI 기본값은 계속 구버전 계산을 사용한다.
전체 재생성 비용은 곡선·특이점 분포에 따라 달라진다. 이번 실행시간과 평가점 수는 파일에 기록했지만 대표 사례 편향이 있으므로 전체 시간의 정확한 예측으로 쓰지 않는다. 먼저 train의 고정 소규모 파일럿으로 처리시간·실패율을 측정하고 범위를 확정한다.

## 재실행

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[data,analysis,refinement]"
.\.venv\Scripts\python.exe scripts/refine_phase4.py
.\.venv\Scripts\python.exe scripts/verify_phase4_refinement.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

새 JSON/Markdown은 outputs/phase4_refinement에 저장된다. 코드·입력 해시, 라이브러리 버전은 provenance.json; 파일 해시는 output_manifest.json.
