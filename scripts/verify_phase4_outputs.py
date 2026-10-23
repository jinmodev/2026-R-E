"""Read-only artifact QA plus a SHA-256 inventory of the delivered files."""
from pathlib import Path
import json
import math
import sys
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from audit_phase4 import sha,write_json,table
Q=ROOT/'outputs/phase4_quality';A=ROOT/'outputs/phase4_analysis'

def main():
    checks=[]
    def check(name,ok,detail=''):checks.append(dict(check=name,status='PASS' if ok else 'FAIL',detail=detail))
    for name in ['mechanical_checks','train_integrity_checks','all_sample_column_checks']:
        f=pd.read_csv(Q/(name+'.csv'));check(name,f.status.eq('PASS').all(),f'{len(f)} checks')
    for file,col in [('analytic_checks','passed'),('polynomial_derivative_checks','passed'),('original_161_reproduction','pass_reproduction')]:
        f=pd.read_csv(Q/(file+'.csv'));check(file,f[col].all(),f'{len(f)} comparisons')
    w=pd.read_csv(A/'winner_shares.csv')
    for (scope,obj),g in w.groupby(['scope','objective']):
        n=g.denominator_scenarios.iloc[0]
        check('winner_denominator_'+scope+'_'+obj,math.isclose(g.deterministic_wins.sum(),n) and math.isclose(g.fractional_wins.sum(),n,rel_tol=1e-10,abs_tol=1e-10))
    caps=pd.read_csv(A/'figure_captions.csv')
    for name in caps.figure:
        p=A/(name+'.png');svg=A/(name+'.svg')
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            dpi=im.info.get('dpi',(0,0));check('image_'+name,all(abs(d-300)<.1 for d in dpi) and min(im.size)>=1000,f'{im.size}, dpi={dpi}')
        check('svg_'+name,ET.parse(svg).getroot().tag.endswith('svg'))
    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(A/'phase4_summary.xlsx') as z:
        wb=ET.fromstring(z.read('xl/workbook.xml'));sheets=wb.findall('s:sheets/s:sheet',ns)
        check('xlsx_sheet_count',len(sheets)==7)
        formulas=0;errors=0;longtext=0
        for file in z.namelist():
            if file.startswith('xl/worksheets/sheet') and file.endswith('.xml'):
                xml=ET.fromstring(z.read(file));formulas+=len(xml.findall('.//s:f',ns));errors+=len(xml.findall('.//s:c[@t="e"]',ns))
        if 'xl/sharedStrings.xml' in z.namelist():
            strings=ET.fromstring(z.read('xl/sharedStrings.xml'));longtext=sum(len(''.join(s.itertext()))>32767 for s in strings)
        check('xlsx_native_formulas',formulas==24,f'{formulas} ratio formulas')
        check('xlsx_error_cells',errors==0);check('xlsx_cell_limit',longtext==0)
        xml=ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
        values={c.attrib['r']:c.findtext('s:v',namespaces=ns) for c in xml.findall('.//s:c',ns)}
        v=pd.read_csv(A/'validity.csv')
        for i,r in enumerate(v.itertuples(),6):
            for col,expected in [('B',r.candidates),('C',r.reachable),('D',r.valid),('E',r.reachable_rate),('F',r.conditional_collision_free_rate),('G',r.valid_rate),('H',r.scenarios_any_valid),('I',r.scenario_valid_rate)]:
                value=values.get(f'{col}{i}')
                check('xlsx_cached_'+f'{col}{i}',value is not None and math.isclose(float(value),expected,rel_tol=1e-12,abs_tol=1e-12))
    table(Q/'artifact_checks.csv',checks)
    files=[]
    for folder in [Q,A]:
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.name!='output_manifest.json': files.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    sources=['scripts/audit_phase4.py','scripts/report_phase4.py','scripts/build_phase4_workbook.mjs','scripts/verify_phase4_outputs.py','tests/test_phase4_audit.py','docs/phase4_quality_analysis_protocol.md']
    write_json(Q/'output_manifest.json',dict(input_manifest_sha256=sha(ROOT/'data/phase4/dataset_manifest.json'),source_files=[dict(path=p,sha256=sha(ROOT/p)) for p in sources],outputs=files))
    failures=[c for c in checks if c['status']=='FAIL']
    print('Artifact checks:',len(checks),'Failures:',len(failures))
    if failures:print(failures);raise SystemExit(1)

if __name__=='__main__':main()
