"""Check saved Excel XML, source preservation, then package the verified case."""
import hashlib,json,shutil,zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/analysis_case1_20261021'
NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
    'c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

checks=json.loads((OUT/'verification.json').read_text(encoding='utf-8'))
for relative,digest in checks['source_hashes'].items():
    assert sha(ROOT/'분석자료_1'/relative)==digest
    assert sha(OUT/'provenance_inputs'/relative)==digest

with zipfile.ZipFile(OUT/'사례1_비교표.xlsx') as z:
    assert z.testzip() is None
    wb=ET.fromstring(z.read('xl/workbook.xml'))
    assert len(wb.findall('s:sheets/s:sheet',NS))==3
    charts=[n for n in z.namelist() if '/charts/chart' in n and n.endswith('.xml')]
    assert len(charts)==2
    chart_roots=[ET.fromstring(z.read(n)) for n in charts]
    scatter=next(r for r in chart_roots if r.find('.//c:scatterChart',NS) is not None)
    xref=scatter.find('.//c:xVal//c:f',NS).text;yref=scatter.find('.//c:yVal//c:f',NS).text
    assert '$B$51:$B$62' in xref and '$C$51:$C$62' in yref
    for r in chart_roots:
        for series in r.findall('.//c:ser',NS):
            assert series.find('.//c:numRef/c:f',NS) is not None
    s=ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
    assert len(s.findall('.//s:f',NS))==35
    assert not [c for c in s.findall('.//s:c',NS) if c.get('t')=='e']
    settings=ET.fromstring(z.read('xl/worksheets/sheet3.xml'))
    assert settings.find(".//s:c[@r='B5']",NS).get('t') not in ('s','inlineStr')
    summary={c.get('r'):float(c.find('s:v',NS).text) for c in s.findall('.//s:c',NS) if c.find('s:f',NS) is not None}
    payload=json.loads((OUT/'workbook_payload.json').read_text(encoding='utf-8'))
    for i,row in enumerate(payload['summary_rows'],7):
        for col,value in zip('CDEFGHI',row[2:]): assert abs(summary[f'{col}{i}']-value)<1e-8

(OUT/'시작하기.md').write_text('''# 분석자료 1 결과물

1. 사례1_비교표.xlsx를 열면 후보 5개의 비교표, 기준별 승자, 편집 가능한 차트 2개를 볼 수 있습니다.
2. figures에는 포스터용 300dpi PNG와 편집 가능한 SVG가 각각 3개 있습니다.
3. 분석설명.md에는 관찰 결과, 그림 캡션, 비교 범위와 한계가 있습니다.
4. 팀원용_곡선식.md에는 선택된 곡선의 간단한 식과 관절각 합성식이 있습니다.
5. 직접 분석하려면 all_42_candidates.csv, selected_5_candidates.csv 및 표본 CSV를 사용하세요.

원자료와 새 정리 자료는 구분했습니다. provenance_inputs는 사용자 제공 원본의 해시가 일치하는 사본입니다.
제외 후보와 결측값을 지우거나 0으로 바꾸지 않았습니다. 상세 원본 식은 selected_equations.csv에 있습니다.
곡률 우승은 동점 후보 2개 중 기존 규칙으로 선택한 후보입니다.
모형 에너지는 실측값이 아니며, 최적은 준비된 42개 후보 안에서의 최적입니다.
엑셀 입력을 고쳐도 로봇 물리나 승자 라벨은 다시 계산되지 않습니다. 새 조건은 앱에서 다시 비교해 저장하세요.

## 재실행

작업공간 루트에서 .venv/Scripts/python.exe scripts/analyze_case1.py --output outputs/새폴더이름
엑셀은 tmp/case1_workbook/build_workbook.mjs에 위 출력 경로를 전달해 제공된 Node 런타임으로 생성합니다.
numpy, pandas, matplotlib, scipy와 프로젝트 패키지를 사용합니다. 난수 추출·모델 학습·유의성 검정은 하지 않았습니다.
입력 데이터는 GUI export report-v4, 계산은 보존된 v3 수학 평가+report-v4 물리 모형입니다.
''',encoding='utf-8')
(OUT/'code').mkdir(exist_ok=True)
for name in ('analyze_case1.py','finalize_case1.py'):shutil.copy2(ROOT/'scripts'/name,OUT/'code'/name)
shutil.copy2(ROOT/'tmp/case1_workbook/build_workbook.mjs',OUT/'code/build_workbook.mjs')
final=dict(status='passed',source_files_unchanged=True,source_copies_hash_match=True,
           xlsx_crc_passed=True,sheets=3,native_charts=2,scatter_points=12,
           saved_summary_formula_count=35,saved_cached_values_match=True,settings_numbers_typed=True,
           summary_formula_change_test=True,images_visually_reviewed=7,
           standalone_figures=3,formats=['300dpi PNG','SVG'],
           limitation='Excel native UI was not opened; used artifact rendering and independent saved XML verification.')
(OUT/'final_checks.json').write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf-8')
excluded={'workbook_payload.json','사례1_비교표.xlsx.inspect.ndjson','manifest.json'}
files=[p for p in OUT.rglob('*') if p.is_file() and p.name not in excluded and 'previews' not in p.parts]
manifest={str(p.relative_to(OUT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files}
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
archive=OUT.with_suffix('.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files+[OUT/'manifest.json']:z.write(p,str(p.relative_to(OUT)))
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
print(json.dumps({'status':'passed','archive':str(archive),'bytes':archive.stat().st_size,'files':len(files)+1,'xlsx_charts':2,'figures':3},ensure_ascii=False))
