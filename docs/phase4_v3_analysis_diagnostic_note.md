# 분석 중 확인한 검사 순서의 영향

2026-09-11. 첫 집계 후 추가한 보조 진단이며 사전 주 분석 집합·동점·순위 규칙은 바꾸지 않는다.

v3의 fully_reachable=true와 collision_free=true는 모두 path_check_status=bounded_clear일 때 설정된다.
위반을 먼저 발견한 후보에서는 다른 상태가 null일 수 있다. 따라서 fully_reachable=true 후보 안에서
collision_free=true 비율이 100%인 것은 계산기 상태 정의에 따른 결과이며 충돌 회피 성능 비교가 아니다.
주 validity 표는 상태를 충실하게 유지하되 이 한계를 명시한다.

보조 sampled_grid_diagnostics는 v3에 저장된 기존 격자 진단 열만 사용한다.
두 격자 도달 확인 = time_grid_fully_reachable AND validation_grid_fully_reachable.
그중 두 격자 무충돌 = time_collision_samples=0 AND validation_collision_samples=0.
도달 불가 표본의 기본 충돌 false를 무충돌로 세지 않도록 반드시 두 격자 도달 조건을 먼저 요구한다.
이는 161점 시간 격자 및 161점 검증 격자의 진단이다. 표본 사이의 위반을 놓칠 수 있으므로
selection_valid, 최종 승자, 적응형 경로 검사, 연속 안전성의 대용으로 사용하지 않는다.
