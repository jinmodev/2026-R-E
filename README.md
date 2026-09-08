# 2026 R&E

**Robot Path Lab — 로봇팔 경로의 수학적 탐구**

GitHub 저장소: [jinmodev/2026-R-E](https://github.com/jinmodev/2026-R-E) (비공개)

2차원 2관절 로봇팔이 여러 매개곡선을 따라갈 때의 도달 가능성, 장애물 충돌, 경로 길이와 수학적 운동 비용을 탐구하는 R&E용 프로그램입니다. 기존 2·3단계 데이터 파이프라인을 보존하면서, 4단계에는 시간별 관절각을 남기는 대규모 데이터 생성기와 장애물 대응 단일 경로 시뮬레이터가 추가되었습니다.

## 현재 할 수 있는 것

단계별 구현, 데이터 생성·검증 기록과 다음 작업 계획은 [프로젝트 진행 기록](docs/project_progress.md)에 정리되어 있습니다.

- 로봇 밑부분, 두 팔 길이, 끝점의 시작·도착 위치 입력
- 직선, 원호, 3차 베지어, 3차 에르미트, 사이클로이드, 리마송 구간 생성
- 팔꿈치 위/아래 역기구학 해 비교
- 표본별 도달 가능 여부와 원형 장애물-두 링크 충돌 확인
- 로봇팔 자세 재생·일시정지·초기화와 시간 슬라이더
- 곡선 매개방정식, 최소 저크 시간함수, `theta1(t)`, `theta2(t)` 합성식 확인
- 경로 길이, 관절 운동 노력도, 각가속도, 저크, 곡률, 장애물 접근 벌점 확인
- 현재 단일 시뮬레이션의 JSON 또는 표본별 CSV 저장
- 42개 후보를 시나리오마다 비교하는 재현 가능한 4단계 데이터 생성

현재 GUI는 **사용자가 선택한 곡선 하나를 평가**합니다. 새 입력에 최적 곡선을 자동 추천하는 머신러닝·최종 분석 기능은 아직 포함하지 않습니다.

## 실행

Python 3.11 이상이 필요합니다. GUI 앱 자체는 표준 라이브러리만 사용하므로 별도 패키지를 설치할 필요가 없습니다. 포스터 분석과 4단계 Parquet 출력에만 선택 패키지가 필요합니다. Windows에서는 `start_app.cmd`를 더블클릭하는 것이 가장 간단합니다.

```powershell
.\start_app.cmd
```

Python이 PATH에 등록되어 있다면 직접 실행해도 됩니다.

```powershell
py app.py
```

또는 패키지를 개발 모드로 설치한 뒤 명령으로 실행할 수 있습니다.

```powershell
py -m pip install -e .
robot-path-lab
```

## 테스트

```powershell
py -m unittest discover -s tests -v
```

## 2단계 실험 데이터 생성

GUI는 그대로 두고 수학 코어를 반복 호출해 분석용 CSV를 생성할 수 있습니다. 기본 설정은 240개 상황, 21개 곡선 변형, 팔꿈치 해 2개로 총 10,080개 실행입니다.

```powershell
py scripts/generate_experiment_data.py
```

난수 시드와 실험 크기를 바꾸려면 다음처럼 실행합니다.

```powershell
py scripts/generate_experiment_data.py --scenarios 300 --seed 20260825 --samples 161
```

결과는 `data/phase2/`의 `scenarios.csv`, `trajectory_metrics.csv`, `experiment_metadata.json`에 저장됩니다. `joint_velocity_effort`와 `joint_acceleration_effort`는 이동 시간을 1로 둔 정규화 지표이며 실제 모터 에너지가 아닙니다.

## 포스터용 분석 자료 생성

240개 시나리오를 독립 표본으로 유지하면서 곡선별 성공률, 95% 신뢰구간, 경로 길이–관절 이동량 관계, 상황별 히트맵과 대표 사례를 생성합니다. GUI는 변경하지 않습니다.

```powershell
py -m pip install -e ".[analysis]"
py scripts/generate_poster_assets.py
```

그림은 `outputs/poster_assets/phase2_baseline/`에 300dpi PNG와 SVG로 저장됩니다. 비교 규칙과 해석 한계는 [`docs/poster_assets.md`](docs/poster_assets.md)에 정리되어 있습니다.

## 3단계 다목적 평가 데이터

기존 2단계 CSV와 GUI를 보존하면서 장애물과 다음 평가함수를 포함한 별도 데이터셋을 생성합니다.

- 끝점 경로 거리
- 2R 동역학·모터 구리손 기반 모델 추정 에너지
- 관절 저크의 크기와 변동
- 끝점 경로의 곡률
- 두 링크 전체의 장애물 충돌·여유 거리
- 특이점 위험
- 목적별 5개 가중치 프로필의 종합비용·보상

```powershell
py scripts/generate_phase3_data.py
```

기본 결과는 `data/phase3/`에 저장됩니다. 이동 시간이나 운동 평면을 바꿀 수도 있습니다.

같은 시나리오의 후보가 학습 세트와 시험 세트에 동시에 들어가는 데이터 누출을 막기 위해 `data_split`은 시나리오 단위의 `train 60% / validation 20% / test 20%`로 미리 고정됩니다.

```powershell
py scripts/generate_phase3_data.py --duration 5 --motion-plane vertical
```

에너지는 실측값이 아니라 명시된 물리 가정에 따른 모델 추정치입니다. 수식, 단위, 가중치와 한계는 [`docs/evaluation_spec.md`](docs/evaluation_spec.md)에 정리되어 있습니다.

## 4단계 수학 중심 데이터와 장애물 시뮬레이터

4단계는 토크·모터 모델 대신, 표본에서 다시 적분해 검산할 수 있는 수학적 비용을 기록합니다. 기본 실행은 일반 시나리오 5,000개와 별도 OOD 도전 시나리오 500개를 생성합니다. 각 시나리오에서 `21개 곡선 변형 x 팔꿈치 해 2개 = 42개` 후보를 계산하므로 총 231,000개 후보와, 표본 수가 161일 때 37,191,000개의 시간 표본이 생깁니다.

먼저 압축 Parquet 출력을 위한 선택 의존성을 설치합니다.

```powershell
py -m pip install -e ".[data]"
```

짧은 점검 실행은 새 출력 폴더에 만드는 것이 안전합니다.

```powershell
py scripts/generate_phase4_data.py --output data/phase4_smoke_20 --scenarios 20 --challenge-scenarios 0 --samples 161 --batch-size 10 --sample-format parquet
```

기본 전체 데이터는 다음 명령으로 생성합니다.

```powershell
py scripts/generate_phase4_data.py --output data/phase4 --workers 8 --sample-format parquet
```

현재 작업공간의 `data/phase4/`에는 이 기본 설정으로 생성한 전체 데이터가 있습니다. 일반 5,000개와 challenge 500개, 후보 231,000행, 승자 라벨 33,000행, 시간표본 37,191,000행이며 압축 데이터와 재개용 내부 파일을 합쳐 약 7.33 GB(6.82 GiB)입니다. 2026-09-04 실행에서는 8개 작업자로 약 41.3분이 걸렸습니다.

이 전체 데이터는 생성한 PC에 보관되어 있고 GitHub에는 포함하지 않습니다. 저장소를 새로 내려받으면 위 명령으로 원자료를 생성할 수 있습니다. 기존 Phase 2·3 데이터와 포스터 출력도 로컬 산출물이며, 저장소에는 해당 생성 스크립트와 명세를 포함합니다.

중단된 생성을 이어갈 때는 처음과 **완전히 같은 옵션**에 `--resume`만 추가합니다.

```powershell
py scripts/generate_phase4_data.py --output data/phase4 --workers 8 --sample-format parquet --resume
```

출력 폴더에는 `scenarios.csv`, `trajectory_metrics.csv`, `scenario_winners.csv`, 분할·배치별 `trajectory_samples/`, 메타데이터, 데이터 사전과 SHA-256 매니페스트가 저장됩니다. 전체 기본 데이터는 수천만 행이라 수 GB가 될 수 있으므로 디스크 여유를 먼저 확인하세요. `--sample-format auto`는 `pyarrow`가 있으면 Parquet, 없으면 압축 CSV를 선택합니다. `--workers`는 시나리오 계산에 사용할 프로세스 수이며, 메모리가 부족하면 1로 낮출 수 있습니다. 재개 가능성을 분명히 하려면 저장 형식과 작업자 수를 명시하는 편이 좋습니다.

평가식, 정규화, 열 구성, 곡률 정칙화와 해석상의 한계는 [`docs/phase4_math_data.md`](docs/phase4_math_data.md)에 정리되어 있습니다.

## 수학적 범위

밑부분에서 끝점까지의 거리 `r`이 다음 조건을 만족해야 한다.

```text
|L1 - L2| ≤ r ≤ L1 + L2
```

자세한 가정과 식은 [`docs/research_spec.md`](docs/research_spec.md)에 정리되어 있습니다.

## 프로젝트 구조

```text
app.py                         GUI 실행 진입점
src/robot_planner/kinematics.py 2R 순기구학·역기구학
src/robot_planner/curves.py      매개곡선 생성기
src/robot_planner/gui.py         Tk 데스크톱 GUI
src/robot_planner/analysis.py    시나리오 단위 통계·후보 선택
src/robot_planner/poster.py      포스터용 그림·표 생성
src/robot_planner/evaluation.py 다목적 평가·2R 동역학·충돌 판정
src/robot_planner/phase3.py     장애물 시나리오·종합점수·CSV 생성
src/robot_planner/phase4_evaluation.py 시간 표본·수학적 적분·재적분 검산
src/robot_planner/phase4.py     4단계 시나리오·점수·승자·분할 저장
scripts/generate_poster_assets.py 재현 가능한 분석 실행 진입점
scripts/generate_phase3_data.py 3단계 데이터 생성 진입점
scripts/generate_phase4_data.py 4단계 스트리밍 데이터 생성 진입점
tests/                           수학 코어 단위 테스트
docs/research_spec.md            연구 명세
docs/poster_assets.md             포스터 분석 규칙·해석 범위
docs/evaluation_spec.md           평가함수·에너지 모델·가중치 명세
docs/phase4_math_data.md          4단계 수학식·스키마·실행·한계
```
