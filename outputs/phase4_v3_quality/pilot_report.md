# Phase 4 v3 파일럿 결과

버전: phase4-mathematics-v3-pilot. train 19개 시나리오, 후보 798개, 시간표본 128,478개.
상태: {'accepted': 289, 'invalid': 509, 'unresolved': 0}. 총 생성 60.22초.
적응형 적분 구간 28,435개, 저장 적분점 909,920개. 161 시간표본과 별도 저장.
검증 55개 통과, 0개 실패. 가중 재적분 최대 절대오차 0, FK 최대 오차 1.12347e-14.
v2 대비 결정론적 승자 변경 0건, 동점집합 변경 0건 / 114개 시나리오-목적 조합.

## 바뀐 계산

Eθ/A/J는 독립 관절 미분과 적응형 구간 적분이다. 저장한 weights_s와 integrand의 가중합이 최종 점수다.
L/Kε/O는 기존 표본 적분을 명시적으로 유지했다. 161개 시간표본을 사다리꼴 적분한 관절 지표는 진단용이며 최종 점수와 다를 수 있다.
QUADPACK과 최종 32점 Gauss-Legendre 재적분의 일치 및 오차 추정치를 확인했다. 이는 엄밀한 정확도 인증은 아니다.
invalid는 관절 적분을 생략하고 null을 저장한다. unresolved는 유효 후보로 승인하지 않으며 대안 중 unresolved가 있으면 승자도 잠정 표시한다.

## 보존·해석·다음 단계

기존 GUI와 v2 생성기/데이터를 변경하지 않았다. 곡선식·theta(t) 합성식은 v3 데이터에도 포함된다.
기존 19개 대표 train 사례의 파일럿이다. 무작위 전체 모집단 분석도, 머신러닝 학습도 아니다. validation/test/challenge 평가값은 읽지 않았다.
후보 42개 안에서의 비교일 뿐 전역 최적이나 과적합 방지 보증이 아니다. 기존 데이터와는 (dataset_version, run_id)로 구별한다.
전체 생성 전에 더 넓은 고정 train 파일럿으로 L/Kε/O와 추가 특이점·미확정 처리를 검증해야 한다. 이번 속도를 전체 데이터에 그대로 외삽하지 않는다.
반도체/모터의 실제 에너지 모델로 바뀐 것이 아니다. Eθ는 여전히 관절 운동 노력도다.

## 파일과 실행

데이터: C:\Users\LG\Documents\2026 R&E\data\phase4_v3_pilot
검증: C:\Users\LG\Documents\2026 R&E\outputs\phase4_v3_quality
정책: docs/phase4_v3_policy.md. 의존성·입력 해시·설정은 데이터 metadata.json.
```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[data,analysis,refinement]"
.\.venv\Scripts\python.exe scripts/generate_phase4_v3.py --output data/phase4_v3_pilot_new
.\.venv\Scripts\python.exe scripts/verify_phase4_v3.py --data data/phase4_v3_pilot_new --output outputs/phase4_v3_quality_new
```

생성기는 비어 있지 않은 출력 폴더를 거부한다. 기존 파일럿은 재생성 없이 검증 명령만 실행할 수 있다.
