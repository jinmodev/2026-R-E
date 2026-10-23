"""Read-only source/artifact checks; writes QA only to the new analysis folder."""
import json
import math
from pathlib import Path
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from analyze_phase4_v3 import ROOT,OUT,DATA,sha,write_json,FAMILIES,OBJECTIVES


def main():
    checks=[]
    def check(name,passed,detail=''):
        checks.append(dict(check=name,passed=bool(passed),detail=detail))
    audit=json.loads((OUT/'analysis_checks.json').read_text(encoding='utf-8'))
    provenance=json.loads((OUT/'provenance.json').read_text(encoding='utf-8'))
    check('source_manifest_unchanged',sha(DATA/'dataset_manifest.json')==provenance['dataset_manifest_sha256'])
    check('protected_files_unchanged',all(sha(ROOT/p)==h for p,h in audit['protected_hashes'].items()))
    count=0
    for marker in sorted((DATA/'partitions/train').glob('*/done.json')):
        r=json.loads(marker.read_text(encoding='utf-8'))
        assert r['data_split']=='train'
        for entry in r['files']:
            if entry['path'] not in ('trajectory_metrics.jsonl.gz','scenario_winners.json'):continue
            p=marker.parent/r['attempt']/entry['path']
            if p.stat().st_size!=entry['bytes'] or sha(p)!=entry['sha256']:raise AssertionError(p)
            count+=1
    check('train_summary_and_label_hashes',count==6000,str(count))
    # Only the few illustrated sample/panel partitions are opened and hashed.
    examples=json.loads((OUT/'representative_equations.json').read_text(encoding='utf-8'))
    for sid in set(r['scenario_id'] for r in examples):
        p=DATA/'partitions/train'/sid;r=json.loads((p/'done.json').read_text(encoding='utf-8'))
        check('case_source_hash_'+sid,all(sha(p/r['attempt']/e['path'])==e['sha256'] for e in r['files']))
    s=pd.read_csv(OUT/'scenario_exclusions.csv');m=pd.read_parquet(OUT/'train_analysis_metrics.parquet')
    check('train_only',set(m.data_split)=={'train'} and not audit['heldout_outcomes_read'])
    check('scenario_count_unique',len(s)==3000 and s.scenario_id.is_unique)
    check('candidate_count_and_keys',len(m)==126000 and m.run_id.is_unique and m.groupby('scenario_id').size().eq(42).all())
    check('analysis_cohort',int(s.cohort.eq('main').sum())==audit['main_scenarios'] and not s.loc[s.cohort.eq('main'),'unresolved'].any())
    v=pd.read_csv(OUT/'validity.csv');check('validity_candidate_total',v.candidates.sum()==126000)
    check('validity_accepted_total',v.valid.sum()==24692)
    check('validity_not_conflated_with_collision',all(v.collision_free_among_reachable>=v.valid))
    for r in v.itertuples():
        g=m[m.curve_key.eq(r.curve_key)]
        check('family_validity_'+r.curve_key,
            int(g.fully_reachable.eq(True).sum())==r.reachable and int(g.selection_valid.sum())==r.valid
            and int(g.groupby('scenario_id').selection_valid.any().sum())==r.scenarios_any_valid)
    wins=pd.read_csv(OUT/'winner_shares.csv')
    for (scope,obj),g in wins.groupby(['scope','objective']):
        n=g.denominator_scenarios.iloc[0]
        check('wins_'+scope+'_'+obj,g.deterministic_wins.sum()==n and math.isclose(g.fractional_wins.sum(),n,abs_tol=1e-8))
    cases=pd.read_csv(OUT/'case_integral_replay.csv')
    check('case_integrals',len(cases)==36 and all(abs(cases.replayed-cases.summary)<=1e-8+1e-10*abs(cases.summary)))
    caps=pd.read_csv(OUT/'figure_captions.csv')
    for name in caps.figure:
        p=OUT/(name+'.png')
        with Image.open(p) as im:im.verify()
        with Image.open(p) as im:
            check('png_'+name,min(im.size)>=1000 and all(abs(d-300)<.1 for d in im.info.get('dpi',(0,0))),str(im.size))
        check('svg_'+name,ET.parse(OUT/(name+'.svg')).getroot().tag.endswith('svg'))
    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    payload=json.loads((OUT/'workbook_payload.json').read_text(encoding='utf-8'))
    cached_checks=0;error_cells=0;formulas=0;long_cells=0
    with zipfile.ZipFile(OUT/'phase4_v3_summary.xlsx') as z:
        wb=ET.fromstring(z.read('xl/workbook.xml'))
        check('xlsx_sheet_names',[r.attrib['name'] for r in wb.findall('s:sheets/s:sheet',ns)]==[p['name'] for p in payload['sheets']])
        for idx,p in enumerate(payload['sheets'],1):
            xml=ET.fromstring(z.read(f'xl/worksheets/sheet{idx}.xml'))
            error_cells+=len(xml.findall('.//s:c[@t="e"]',ns));formulas+=len(xml.findall('.//s:f',ns))
            cells={c.attrib['r']:c for c in xml.findall('.//s:c',ns)}
            for i,row in enumerate(p['rows'],6):
                for j,expected in enumerate(row):
                    col=chr(65+j)
                    if col in p['ratios']:
                        num,den=p['ratios'][col];expected=row[ord(num)-65]/row[ord(den)-65]
                        assert cells[col+str(i)].find('s:f',ns) is not None
                    if isinstance(expected,(int,float)) and not isinstance(expected,bool):
                        actual=cells.get(col+str(i));value=actual.findtext('s:v',namespaces=ns) if actual is not None else None
                        if value is None or not math.isclose(float(value),expected,rel_tol=1e-11,abs_tol=1e-11):raise AssertionError((p['name'],col,i,value,expected))
                        cached_checks+=1
        if 'xl/sharedStrings.xml' in z.namelist():
            long_cells=sum(len(''.join(c.itertext()))>32767 for c in ET.fromstring(z.read('xl/sharedStrings.xml')))
    check('xlsx_formula_errors',error_cells==0);check('xlsx_cell_length',long_cells==0)
    check('xlsx_numeric_caches',cached_checks>0,str(cached_checks))
    calc=json.loads((OUT/'excel_previews/recalculation.json').read_text(encoding='utf-8'))
    check('xlsx_live_recalculation',calc['passed'] and calc['ratioChecks']==formulas and calc['zeroDenominator']=='n.a.')
    # Complete automated regression run captured in the new output only.
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,text=True,capture_output=True)
    (OUT/'unit_test_log.txt').write_text(result.stdout+result.stderr,encoding='utf-8')
    check('unit_tests',result.returncode==0,(result.stdout+result.stderr).splitlines()[-4:])
    # Contact sheets are QA aids, not scientific figures or additional deliverables.
    preview=OUT/'visual_review';preview.mkdir(exist_ok=True)
    paths=[OUT/(n+'.png') for n in caps.figure]+sorted((OUT/'excel_previews').glob('*.png'))
    for page in range(0,len(paths),4):
        canvas=Image.new('RGB',(1800,1200),'white');draw=ImageDraw.Draw(canvas)
        for i,p in enumerate(paths[page:page+4]):
            with Image.open(p) as im:
                im=im.convert('RGB');im.thumbnail((880,550));x=(i%2)*900;y=(i//2)*600
                canvas.paste(im,(x,y+30));draw.text((x+5,y+5),p.stem,fill='#20314D',font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',14))
        canvas.save(preview/f'page_{page//4+1:02}.png')
    write_json(OUT/'artifact_checks.json',dict(passed=all(c['passed'] for c in checks),checks=checks,
        numeric_cache_checks=cached_checks,formula_count=formulas,scientific_figures=len(caps),
        native_excel_tested=False,visual_review='Rendered; human/model inspection recorded separately in visual_review.md'))
    files=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
           for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='output_manifest.json']
    write_json(OUT/'output_manifest.json',dict(input_manifest_sha256=provenance['dataset_manifest_sha256'],
        sources={p:sha(ROOT/p) for p in ['scripts/analyze_phase4_v3.py','scripts/report_phase4_v3_analysis.py',
            'scripts/build_phase4_v3_workbook.mjs','scripts/verify_phase4_v3_analysis.py',
            'tests/test_phase4_v3_analysis.py','docs/phase4_v3_analysis_protocol.md','docs/phase4_v3_analysis_diagnostic_note.md']},outputs=files))
    failures=[c for c in checks if not c['passed']]
    print('Checks:',len(checks),'Failures:',len(failures),'Numeric Excel values:',cached_checks,'Figures:',len(caps),flush=True)
    if failures:print(failures);raise SystemExit(1)


if __name__=='__main__':main()
