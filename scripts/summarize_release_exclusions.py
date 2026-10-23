"""Numerical exclusion diagnostics on train only; no curve ranking analysis."""
from collections import Counter
from pathlib import Path
import sys

import audit_phase4 as audit
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from robot_planner.release_io import iter_release_metrics


def main():
    unresolved=[];new_invalid=[]
    for r in iter_release_metrics(ROOT/'data/phase4_v3_release'):
        if r['candidate_status']=='unresolved':
            unresolved.append(dict(run_id=r['run_id'],scenario_id=r['scenario_id'],warning=r['warning'],
                path_check_status=r['path_check_status'],joint_integration_status=r['joint_integration_status'],
                joint_integration_error=r['joint_integration_error'],
                failed_geometry={k:v for k,v in r['geometry_checks'].items() if not v['passed']},
                path_check=r['path_check']))
        if r['candidate_status']=='invalid' and r['legacy_v2']['selection_valid']:
            new_invalid.append(dict(run_id=r['run_id'],scenario_id=r['scenario_id'],path_check=r['path_check']))
    result=dict(split='train',unresolved_count=len(unresolved),
        reasons=dict(Counter(r['warning'] for r in unresolved)),
        failed_geometry_counts=dict(Counter(k for r in unresolved for k in r['failed_geometry'])),
        unresolved=unresolved,newly_invalid_count=len(new_invalid),newly_invalid=new_invalid)
    audit.write_json(ROOT/'outputs/phase4_v3_release_quality/train_exclusions.json',result)
    print({k:v for k,v in result.items() if k not in ('unresolved','newly_invalid')})


if __name__=='__main__':main()
