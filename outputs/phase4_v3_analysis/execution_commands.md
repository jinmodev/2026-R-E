# 실행과 변경 범위

작업 폴더: `C:\Users\LG\Documents\2026 R&E`.
완성된 보고서·Excel·그림을 보기만 할 때는 재실행할 필요 없다.

```powershell
.\.venv\Scripts\python.exe scripts/analyze_phase4_v3.py
.\.venv\Scripts\python.exe scripts/report_phase4_v3_analysis.py
& 'C:/Users/LG/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe' scripts/build_phase4_v3_workbook.mjs
.\.venv\Scripts\python.exe scripts/verify_phase4_v3_analysis.py
```

첫 명령은 새 train 요약·그림·사례를 계산하고 두 번째는 보고서 및 보조 격자 진단을 만든다.
세 번째는 Excel, 네 번째는 원본 파일 대조·산출물 검증·회귀 테스트를 실행한다.
의존성: 프로젝트 Python 환경의 numpy/pandas/pyarrow/matplotlib/Pillow 및 번들 Node의 @oai/artifact-tool.
맑은 고딕 Windows 글꼴을 사용한다. `scripts/node_modules`는 번들 Node 패키지 경로를 가리킨다.
다른 PC에서는 Python 패키지·한글 글꼴·Node 패키지 경로를 맞춰야 한다.

레이아웃만 다시 그릴 때는 첫 명령에 `--refresh-overview-figures` 또는 `--refresh-case-figures`를 쓸 수 있다.
기존 분석 캐시의 train 집합과 manifest를 확인하고 선정 기준·평가값은 바꾸지 않는다.
그 후 최종 검증 명령을 다시 실행하여 파일 해시 목록을 갱신한다.

## 변경 파일

- scripts/analyze_phase4_v3.py: train 집합·제외·유효성·승자·동점·대응·관계·사례 및 그림.
- scripts/report_phase4_v3_analysis.py: 한글 보고서·격자 진단·Excel 입력표.
- scripts/build_phase4_v3_workbook.mjs: 8시트 Excel과 재계산·렌더링.
- scripts/verify_phase4_v3_analysis.py: 원본 해시·집계·이미지·Excel 검증.
- tests/test_phase4_v3_analysis.py: 제외·동점·경계·재현성 회귀 테스트 6개.
- docs/phase4_v3_analysis_protocol.md 및 phase4_v3_analysis_diagnostic_note.md: 사전 규칙과 후속 보조 진단의 구분.
- README.md 및 docs/project_progress.md: 새 산출물 위치와 진행 상태.

새 결과는 이 폴더에만 저장한다. GUI, 수학 계산 코어, 기존 원자료, 이전 포스터/분석/검증 폴더는 보존한다.
GitHub 업로드, 모델 학습, 최종 포스터는 이번 작업에 포함하지 않았다.
