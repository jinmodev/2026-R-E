"""Independent flat-file audit of the delivered single-scenario dossier."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=2e-10, abs_tol=1e-9), (a, b)

def audit(out):
    data = json.loads((out / 'dossier.json').read_text(encoding='utf-8'))
    summary = {r['run_id']: r for r in read(out / '우승후보_전체평가값.csv')}
    assert len(summary) == 6
    totals = defaultdict(list)
    cumulative = defaultdict(float)
    traps = read(out / 'integration_steps/사다리꼴_구간별_계산.csv')
    for r in traps:
        key = (r['run_id'], r['metric'])
        value = (float(r['right']) - float(r['left'])) * (
            float(r['integrand_left']) + float(r['integrand_right'])) / 2
        close(value, r['contribution'])
        totals[key].append(value)
        cumulative[key] += value
        close(cumulative[key], r['cumulative'])
    for rid, row in summary.items():
        for metric, field in [('L', 'path_length_units'), ('Kepsilon', 'regularized_curvature_energy_inv_unit'), ('O', 'obstacle_penalty_s_per_unit2')]:
            close(math.fsum(totals[rid, metric]), row[field])
        energy = math.fsum(totals[rid, 'positive_power_w']) + math.fsum(totals[rid, 'copper_loss_power_w'])
        close(energy, row['model_electrical_energy_j'])
        close(math.sqrt(math.fsum(totals[rid, 'torque_squared_nm2']) / 3), row['rms_joint_torque_nm'])
    nodes = read(out / 'integration_steps/관절적응형_가중적분_계산.csv')
    weighted = defaultdict(list)
    for r in nodes:
        value = float(r['weight_s']) * float(r['integrand'])
        close(value, r['contribution'])
        weighted[r['run_id'], r['metric']].append(value)
    for (rid, field), values in weighted.items():
        close(math.fsum(values), summary[rid][field])
    deductions = defaultdict(list)
    score_rows = read(out / '종합점수_계산과정.csv')
    for r in score_rows:
        assert float(r['denominator_candidates']) == 12
        span = float(r['maximum']) - float(r['minimum'])
        normalized = (float(r['raw_value']) - float(r['minimum'])) / span if span else 0
        close(normalized, r['normalized'])
        close(100 * float(r['weight']) * normalized, r['deduction_points'])
        deductions[r['run_id']].append(float(r['deduction_points']))
    for rid, values in deductions.items():
        close(math.fsum(values), summary[rid]['report_balanced_cost_0_100'])
        close(100 - math.fsum(values), summary[rid]['report_balanced_reward_0_100'])
    labels = read(out / '기준별_우승라벨.csv')
    assert len(labels) == 9
    selected = set()
    for r in labels:
        ties = json.loads(r['tied_run_ids_json'])
        assert len(ties) == int(float(r['tie_count']))
        assert r['winner_run_id'] in ties
        selected.update(ties)
    assert selected == set(summary)
    for name, expected in data['source_hashes'].items():
        for path in [ROOT / '분석자료_1' / name, out / 'provenance_inputs' / name]:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    csv_files = list(out.rglob('*.csv'))
    for path in csv_files:
        for r in read(path):
            for value in r.values():
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                assert math.isfinite(number), (path, value)
    result = dict(status='passed', trap_rows=len(traps), weighted_nodes=len(nodes),
                  score_rows=len(score_rows), labels=len(labels), selected_candidates=len(summary),
                  csv_files_checked=len(csv_files), all_contributions_recomputed=True,
                  all_scores_recomputed=True, original_and_copied_source_hashes_match=True)
    (out / 'independent_flatfile_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/case1_winner_bundle_20261023')
    audit(parser.parse_args().output)
