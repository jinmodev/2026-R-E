# 2단계 베이스라인 포스터 자료 생성

이 분석은 `trajectory_metrics.csv`의 각 행을 독립 표본으로 취급하지 않는다. 같은 로봇팔·시작점·끝점을 공유하는 실행은 하나의 시나리오에 속하므로, 통계의 독립 단위는 240개 `scenario_id`이다.

## 비교 규칙

각 `시나리오 × 곡선 계열`에서 다음 순서로 대표 후보 하나를 선택한다.

1. 경로의 모든 표본점이 도달 가능한 후보만 남긴다.
2. 관절공간 경로 길이 `Q = joint_motion_l2_rad`가 가장 작은 후보를 고른다.
3. 동률이면 끝점 경로 길이가 짧은 후보를 고른다.
4. 다시 동률이면 최소 특이점 지수가 큰 후보를 고른다.

성공률의 95% 신뢰구간은 개별 실행 행이 아니라 시나리오 전체를 복원추출하는 부트스트랩으로 계산한다. 직선 대비 `Q` 비교도 같은 시나리오에서 두 곡선이 모두 성공한 경우만 짝지어 수행한다.

## 실행

분석용 선택 의존성을 설치한다.

```powershell
py -m pip install -e ".[analysis]"
```

포스터 자료를 생성한다.

```powershell
py scripts/generate_poster_assets.py
```

결과는 `outputs/poster_assets/phase2_baseline/`에 저장된다. 그림과 표는 편집 가능한 SVG와 300dpi PNG로 동시에 만들어지며, 표의 원자료 CSV, 캡션 초안, LaTeX 수식, 분석 매니페스트도 포함된다.

## 생성 파일

- `fig00_method_model`: 2R 기구학, 작업공간, 평가식
- `fig01_feasibility_by_curve`: 곡선 계열별 성공률과 95% 신뢰구간
- `fig02_path_vs_joint_motion`: 끝점 경로 길이와 관절 이동량의 관계
- `fig03_scenario_heatmap`: 시나리오 유형별 성공률
- `fig04_joint_motion_vs_line`: 같은 시나리오에서 직선 대비 관절 이동량
- `fig05_representative_case`: 직선은 실패하고 곡선은 성공한 사례
- `table01_curve_summary`: 포스터용 핵심 결과 표
- `captions.md`: 포스터 캡션과 해석 한계
- `method_equations.tex`: 수식 원본
- `analysis_manifest.json`: 입력 해시와 분석 규칙

## 해석 범위

현재 자료는 장애물이 없는 2차원 2관절 베이스라인이다. `Q`는 실제 전력이나 모터 에너지가 아니라 관절각 변화 벡터의 누적 길이다. 따라서 포스터에서는 이를 “에너지”라고 쓰지 않고 “관절 이동량”으로 명시한다. 장애물 충돌률, 최소 장애물 여유 거리, 동역학 기반 에너지는 다음 실험 단계의 별도 지표로 추가한다.
