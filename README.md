# 2026 R&E

**Robot Path Lab — 로봇팔 경로의 수학적 탐구**

## 최신 사례 자료와 GitHub 저장 범위 (2026-10-23)

- [우승 곡선 전체 자료 PDF](outputs/case1_winner_bundle_20261023/우승곡선_전체자료.pdf): 기존 `분석자료_1`의 곡선식·구간·개별 그래프·적분·점수, 곡률 공동 우승 포함 6개 후보.
- [전체 ZIP](outputs/case1_winner_bundle_20261023.zip): 복사용 TXT, 정밀 원본 식, 구간별 적분 CSV, 그래프 PNG·SVG, 원자료·코드 스냅샷 포함.
- [GitHub 저장 범위와 재검사](docs/github_backup_20261023.md): 코드·문서·분석 자료를 저장하되, 기존에 제외된 약 22 GB 원자료의 원격 백업은 별도입니다. 전체 원자료 백업 완료로 해석하지 마세요.
- 저장 대상 해시 및 제외 원자료 목록: `outputs/github_backup_20261023/`.

## 보고서 연계 프로그램과 분석용 원자료 (2026-10-21)

GUI에 보고서의 원형 곡선식·미분·곡률·적분 설명, 2관절 토크·기계적 일·모형 에너지,
같은 입력의 42개 후보 비교와 CSV 원자료 저장 기능을 추가했습니다.
‘보고서 사례’에서 P4S000691의 리마송·사이클로이드 조건을 불러올 수 있습니다.
‘42개 후보 비교’를 누르면 오른쪽 ‘후보 비교’ 탭이 자동으로 열립니다.
계산 후 평가기준을 고르고 ‘우승 곡선 적용’을 누르면 해당 후보의 곡선·굽힘값·팔꿈치 방향과
경로 그림·곡선식·관절각 식·평가값이 함께 적용됩니다. 기본 기준은 종합점수이며,
동점은 기존 결정 규칙을 사용합니다. 미확정·유효 후보 없음·입력 변경 시에는 적용하지 않습니다.
적용 후에도 비교 결과와 원자료 저장 기능은 유지되며, 재생 버튼으로 움직임을 확인할 수 있습니다.
화면은 검증된 v3 release 계산을 사용하며, 새 물리 모형은 report-v4로 구분합니다.

- 실행: start_app.cmd 또는 .venv/Scripts/python.exe app.py
- [수식·모형·점수·결측 처리와 사용법](docs/report_alignment.md)
- 분석용 전체 파일: outputs/report_aligned_v4_20261021 및 같은 이름의 ZIP.
- CSV.gz/Parquet에 상황·장애물·후보 요약·곡선/관절각 식·시간 표본·기하 표본을 나눠 저장합니다.
- 새 자료는 train만 사용하며 validation/test/challenge 평가값과 기존 원자료는 보존합니다.
- 그래프나 통계적 우열 분석은 생성하지 않습니다. 같은 scenario_id 후보는 독립 표본이 아닙니다.

    .\.venv\Scripts\python.exe scripts/build_report_dataset.py --workers 6
    .\.venv\Scripts\python.exe scripts/prepare_report_handoff.py
    .\.venv\Scripts\python.exe scripts/verify_report_dataset.py

아래 절들은 이전 단계의 기록입니다. 기존 단일 경로 GUI 설명은 위의 보고서 연계 기능으로 확장됐습니다.

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

## 수치·충돌 보강 검사 (2026-09-11)

기존 기본 계산과 원자료는 그대로 두고, 독립 관절 미분·적응형 적분 및
구간 이동 상계를 이용한 충돌 검사 API를 추가했습니다. 모델 학습은 아직 하지 않습니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[data,analysis,refinement]"
.\.venv\Scripts\python.exe scripts/refine_phase4.py
.\.venv\Scripts\python.exe scripts/verify_phase4_refinement.py
```

앞 단계에서 생성한 train 검증 캐시가 필요합니다. 새 결과는
`outputs/phase4_refinement`의 JSON과 [보강 보고서](outputs/phase4_refinement/refinement_report.md)입니다.
검사 범위·고정 기준은 [사전 프로토콜](docs/phase4_refinement_protocol.md)을 따릅니다.
새 충돌 API는 `invalid`, `bounded_clear`, `unresolved`를 구분하며 미확정을 안전으로 처리하지 않습니다.
GUI는 여전히 구버전 기본 계산을 사용합니다. 새 계산의 데이터 생성기 적용은 아래 v3 파일럿에 별도로 구현했습니다.

## v3 데이터 생성기와 train 파일럿

관절 운동 노력도·각가속도·저크에 독립 미분과 적응형 적분을 적용한 별도 생성기입니다.
기존 v2 생성기·원자료·GUI는 바꾸지 않습니다. L/Kε/O는 명시적으로 기존 계산을 유지합니다.

```powershell
.\.venv\Scripts\python.exe scripts/generate_phase4_v3.py --output data/phase4_v3_pilot_new
.\.venv\Scripts\python.exe scripts/verify_phase4_v3.py --data data/phase4_v3_pilot_new --output outputs/phase4_v3_quality_new
```

이미 생성된 로컬 `data/phase4_v3_pilot`에는 train 19개 시나리오의 후보 798개,
시간표본 128,478개, 적응형 적분점 909,920개가 있습니다. 비어 있지 않은 출력 폴더는 거부합니다.
이 파일럿 이후 전체 재생성도 완료했습니다. 아래 릴리스 절을 참조하세요. 모델 학습은 아직 하지 않았습니다.

- [파일럿 결과와 검증](outputs/phase4_v3_quality/pilot_report.md)
- [버전별 계산·결측·미확정 처리 정책](docs/phase4_v3_policy.md)

후보 요약·곡선식·관절각 합성식은 JSONL, 시간표본은 Parquet로 저장합니다.
적응형 적분점은 `adaptive_panels.jsonl`에 시간·가중치·피적분함수로 남기므로
저장된 가중합으로 최종 관절 평가값을 재계산할 수 있습니다.
161개 시간표본의 사다리꼴 적분은 별도 진단값이며 최종 적분값과 혼동하면 안 됩니다.

## v3 전체 데이터와 검증 확대 완료

새로운 train 96개 시나리오로 검증 범위를 넓힌 뒤, 원래 시나리오와 분할을 유지한
`phase4-mathematics-v3` 데이터를 별도 생성했습니다. 총 5,500개 시나리오,
후보 231,000개, 시간표본 37,191,000개, 목적별 라벨 33,000개입니다.
로컬 `data/phase4_v3_release`에 압축 저장되어 있으며 주 데이터 파일은 약 8.44 GiB입니다.

- [최종 검증·완료 보고서](outputs/phase4_v3_release_quality/release_report.md)
- [데이터 구조·단위·미확정 처리·읽기 방법](docs/phase4_release_data_dictionary.md)
- [사전에 고정한 검증·생성 정책](docs/phase4_release_policy.md)

train에서는 24,692개 후보가 검사 기준을 통과했고 16개는 미확정으로 분리했습니다.
미확정 대안이 있는 시나리오의 승자는 잠정값입니다. 기존 표본 검사에서 유효했던
train 경로 3개는 새 구간 검사에서 충돌이 발견돼 제외했습니다.
validation/test/challenge 결과 분포·순위 분석과 머신러닝은 하지 않았습니다.
새 train 포스터용 개별 자료는 아래 분석 단계에서 별도로 갱신했습니다.
기존 GUI·v2 원자료·v3 파일럿·이전 분석 결과는 보존했습니다.

## v3 train 보고서·포스터용 개별 자료

새 버전 train 3,000개 상황·126,000개 후보를 분석했습니다. 주 승자·대응 비교에는
2,136개 상황을 사용했고, 미확정 대안이 있는 12개 상황과 나머지 유효 후보 없는 852개 상황을 분리했습니다.
각 곡선의 경로·관절각·피적분함수, 목적별 실제 선택 차이, CSV 요약표와 Excel을 제공합니다.
최종 포스터와 추천 모델은 만들지 않았습니다.

- [분석 보고서와 그림 캡션](outputs/phase4_v3_analysis/analysis_report.md)
- [요약 Excel](outputs/phase4_v3_analysis/phase4_v3_summary.xlsx)
- [사전 분석 규칙](docs/phase4_v3_analysis_protocol.md)
- [검사 순서와 보조 격자 진단 해석](docs/phase4_v3_analysis_diagnostic_note.md)

```powershell
.\.venv\Scripts\python.exe scripts/analyze_phase4_v3.py
.\.venv\Scripts\python.exe scripts/report_phase4_v3_analysis.py
node scripts/build_phase4_v3_workbook.mjs
.\.venv\Scripts\python.exe scripts/verify_phase4_v3_analysis.py
```

Excel 빌더에는 번들 Node의 `@oai/artifact-tool`이 필요합니다. 기존 생성 데이터를 다시 만들 필요는 없습니다.
분석은 train 완료 파일만 읽으며 기존 GUI·원자료·과거 분석 결과를 변경하지 않습니다.
모든 최적은 준비된 42개 후보 안에서의 계산 결과이고 물리적 안전·과적합 해결의 보증이 아닙니다.

```powershell
.\.venv\Scripts\python.exe scripts/verify_phase4_release.py full --workers 6
```

이미 완료한 데이터는 다시 생성할 필요 없습니다. 중단된 생성만 소스·입력·환경을 유지한 채
`scripts/phase4_release_pipeline.py full --workers 6 --resume`으로 재개합니다.
완료 기록이 지정한 attempt만 읽어야 하며, 기본 읽기 API는 train만 허용합니다.

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

## Phase 4 수치 검증 및 train 분석 (2026-09-09)

원자료와 GUI를 보존하고 수치 검증·R&E 개별 자료를 별도로 생성합니다.
머신러닝 학습이나 최종 추천기는 포함하지 않습니다. test·challenge는
해시·행 수·키·스키마만 검사하며 연구 분석은 train에 한정됩니다.

```powershell
py -m pip install -e ".[data,analysis]"
py scripts/audit_phase4.py --stage all
py scripts/report_phase4.py
node scripts/build_phase4_workbook.mjs
py scripts/verify_phase4_outputs.py
```

Excel 빌더는 `@oai/artifact-tool`이 설치된 Node 환경을 사용합니다.
이 작업공간의 `scripts/node_modules`는 Codex 번들 패키지에 대한 junction이며,
다른 PC에서는 동일 패키지를 제공하는 환경을 준비해야 합니다. Python 검증·표·그림
생성은 Node 없이 재실행할 수 있습니다. Python 단계는 `--stage mechanical`,
`integrity`, `convergence`, `analysis`, 또는 `research`(구조 검사 이후 전 단계)로
나누어 재실행할 수 있습니다.

- 사전 기준: [수치 검증·분석 계획](docs/phase4_quality_analysis_protocol.md)
- 수치 검증: [quality_report.md](outputs/phase4_quality/quality_report.md)
- train 분석·그림·캡션: [analysis_report.md](outputs/phase4_analysis/analysis_report.md)
- 요약 Excel: `outputs/phase4_analysis/phase4_summary.xlsx`
- 표 원본: 위 두 출력 폴더의 CSV, 수식 원본은 `representative_equations.json`

19개 train 시나리오의 798개 후보에 대한 81/161/321/641표본 비교에서
일부 A·J가 사전 수렴성 기준을 넘었습니다. 전체 원자료의 승자 집계는
수치 미확정 참고 통계로 분리합니다. 최적의 범위는 준비된 42개 후보이며,
641표본도 정확한 참값이나 연속 충돌 안전 보증이 아닙니다.

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
