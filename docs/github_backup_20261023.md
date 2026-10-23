# GitHub 저장 범위와 대용량 원자료 (2026-10-23)

현재 작업공간의 Git 저장 대상(코드, 테스트, 문서, Phase 2·3 데이터, 분석 결과,
`분석자료_1`, 최신 우승 곡선 PDF·CSV·PNG·SVG·ZIP)을 기존 비공개 저장소에 반영합니다.
기존 원격 파일과 과거 커밋은 삭제하지 않습니다. 강제 푸시는 하지 않습니다.

## 별도로 보존하는 원자료

전체 연구 파일은 약 22.35 GB입니다. 기존 `.gitignore`에 따라 Phase 4 원자료,
v3 파일럿·릴리스 데이터, report-v4 전체 출력 등 약 22 GB는 로컬에 남습니다.
이 저장을 전체 원자료의 원격 백업 완료라고 해석하면 안 됩니다.
가상환경, 라이브러리, 캐시, 임시 파일, `.git` 내부 파일은 연구 백업에서 제외합니다.

정확한 저장 대상 경로·크기·해시는 `outputs/github_backup_20261023/included_files.csv`,
제외한 원자료의 경로·크기는 같은 폴더의 `excluded_large_data_files.csv`에 기록합니다.
목록 폴더는 순환 해시를 피하기 위해 자신의 목록에서 제외합니다.
목록을 만들었다는 사실만으로 원격 업로드 성공을 보증하지 않습니다.
원격 커밋의 전체 파일 목록과 Git blob SHA를 로컬 스냅샷과 대조해야 합니다.

대용량 원자료의 별도 백업 방식·요금은 사용자와 결정합니다.
새 유료 서비스, LFS 예산·결제, 공개 데이터 배포는 승인 없이 설정하지 않습니다.

## 재검사

```powershell
New-Item -ItemType Directory -Force -Path tmp/github_backup_checks
$env:TEMP = (Join-Path (Get-Location) 'tmp/github_backup_checks')
$env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
.\.venv\Scripts\python.exe scripts/prepare_github_backup.py
```

2026-10-23 검사: 단위 테스트 97개 통과. 기본 샌드박스 임시 경로에서는 권한 오류가
발생했으며, 작업공간의 임시 경로로 지정한 재실행에서 모두 통과했습니다.
앱 계산 로직은 이 저장 작업에서 변경하지 않았습니다.

최신 단일 사례 자료는 `outputs/case1_winner_bundle_20261023/우승곡선_전체자료.pdf`와
`outputs/case1_winner_bundle_20261023.zip`입니다. 42개 후보와 9개 우승 라벨을 재현했고,
곡률 공동 우승을 포함한 6개 후보를 담았습니다. 에너지는 실측이 아닌 가정한 모형값입니다.
