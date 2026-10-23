"""Check refinement artifacts and run all regression tests, without raw writes."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys

import audit_phase4 as audit

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/phase4_refinement'


def main():
    checks=[]
    def check(name,passed,detail=''):
        checks.append(dict(check=name,passed=bool(passed),detail=detail))
    def read(name):
        return json.loads((OUT/name).read_text(encoding='utf-8'))
    provenance=read('provenance.json');summary=read('summary.json')
    for rel,expected in provenance['hashes'].items():
        check('unchanged input/source: '+rel,audit.sha(ROOT/rel)==expected)
    check('protocol unchanged',audit.sha(ROOT/'docs/phase4_refinement_protocol.md')==provenance['protocol_sha256'])
    old=json.loads((ROOT/'outputs/phase4_quality/output_manifest.json').read_text(encoding='utf-8'))
    preserved=[]
    for item in old['outputs']:
        p=ROOT/item['path']
        preserved.append(p.stat().st_size==item['bytes'] and audit.sha(p)==item['sha256'])
    check('previous quality/analysis artifacts preserved',all(preserved),f'{len(preserved)} files')
    for item in read('output_manifest.json')['files']:
        p=OUT/item['path']
        check('output hash/size: '+item['path'],p.stat().st_size==item['bytes'] and audit.sha(p)==item['sha256'])
    grid=read('grid_comparison.json');paths=read('adaptive_path_checks.json');accuracy=read('integral_accuracy.json')
    check('grid count',len(grid)==4*42*len(provenance['failure_scenarios']))
    check('grid unique keys',len({(r['run_id'],r['samples']) for r in grid})==len(grid))
    check('train grid only',all(r['data_split']=='train' for r in grid))
    check('42 paths/scenario',all(sum(r['scenario_id']==s for r in paths)==42 for s in provenance['collision_scenarios']))
    check('path run IDs unique',len({r['run_id'] for r in paths})==len(paths))
    check('bounded-clear contains no witnesses or unknown intervals',all(
        not r['unreachable_witnesses'] and not r['collision_witnesses'] and r['unresolved_intervals']==0
        for r in paths if r['status']=='bounded_clear'))
    check('invalid has a witness',all(r['unreachable_witnesses'] or r['collision_witnesses'] for r in paths if r['status']=='invalid'))
    check('evaluation budget respected',all(r['evaluations']<=2049 for r in paths))
    check('accuracy reference run IDs match',set(r['run_id'] for r in accuracy)==set(provenance['failed_run_ids']+provenance['control_run_ids']))
    check('reference warnings not counted as verified',all(r['passed'] is not True for r in accuracy if r['reference_warning']))
    check('summary collision count',len(paths)==summary['collision_runs'])
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,
                          capture_output=True,text=True,encoding='utf-8',errors='replace')
    (OUT/'unit_test_log.txt').write_text(result.stdout+result.stderr,encoding='utf-8')
    check('full regression suite',result.returncode==0,result.stderr.splitlines()[-4:])
    audit.write_json(OUT/'verification.json',dict(checks=checks,passed=sum(c['passed'] for c in checks),failed=sum(not c['passed'] for c in checks)))
    audit.write_json(OUT/'output_manifest.json',dict(version=provenance['version'],files=[dict(path=p.name,bytes=p.stat().st_size,sha256=audit.sha(p)) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='output_manifest.json']))
    print(json.dumps(checks,ensure_ascii=False,indent=2))
    if any(not c['passed'] for c in checks):raise SystemExit(1)


if __name__=='__main__':main()
