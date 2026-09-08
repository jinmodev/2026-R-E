from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robot_planner.analysis import (  # noqa: E402
    CURVE_ORDER,
    load_phase2_data,
    paired_with_line,
    select_best_candidates,
    select_representative_scenario,
    summarize_curves,
)


def _row(
    scenario_id: str,
    curve_key: str,
    candidate: int,
    *,
    reachable: bool,
    joint_motion: float,
    curve_length: float,
) -> dict[str, object]:
    return {
        "run_id": f"{scenario_id}-{curve_key}-{candidate}",
        "scenario_id": scenario_id,
        "scenario_type": "cross_inner",
        "curve_key": curve_key,
        "shape": 0.4 if candidate else -0.4,
        "elbow_mode": "up" if candidate else "down",
        "fully_reachable": reachable,
        "reachable_ratio": 1.0 if reachable else 0.7,
        "curve_length": curve_length,
        "path_to_chord_ratio": curve_length / 2.0,
        "joint_motion_l2_rad": joint_motion,
        "total_joint_rotation_deg": joint_motion * 57.3,
        "min_singularity_index": 0.8 if candidate else 0.5,
    }


def _metrics() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario_id in ("S1", "S2"):
        for curve_key in CURVE_ORDER:
            line_failure = scenario_id == "S2" and curve_key == "line"
            rows.append(
                _row(
                    scenario_id,
                    curve_key,
                    0,
                    reachable=not line_failure,
                    joint_motion=2.0,
                    curve_length=2.4,
                )
            )
            rows.append(
                _row(
                    scenario_id,
                    curve_key,
                    1,
                    reachable=not line_failure,
                    joint_motion=1.5,
                    curve_length=2.5,
                )
            )
    return pd.DataFrame(rows)


class AnalysisTests(unittest.TestCase):
    def test_selects_one_minimum_joint_motion_candidate_per_family(self) -> None:
        best = select_best_candidates(_metrics())
        self.assertEqual(len(best), 2 * len(CURVE_ORDER))
        selected = best.loc[
            (best["scenario_id"] == "S1") & (best["curve_key"] == "arc")
        ].iloc[0]
        self.assertEqual(selected["run_id"], "S1-arc-1")
        self.assertAlmostEqual(selected["joint_motion_l2_rad"], 1.5)

        failed_line = best.loc[
            (best["scenario_id"] == "S2") & (best["curve_key"] == "line")
        ].iloc[0]
        self.assertFalse(bool(failed_line["family_feasible"]))
        self.assertTrue(pd.isna(failed_line["run_id"]))

    def test_summary_uses_scenario_counts_not_raw_candidate_rows(self) -> None:
        best = select_best_candidates(_metrics())
        summary = summarize_curves(best, bootstrap_iterations=200, seed=7)
        line = summary.set_index("curve_key").loc["line"]
        arc = summary.set_index("curve_key").loc["arc"]
        self.assertEqual(line["scenario_count"], 2)
        self.assertEqual(line["feasible_count"], 1)
        self.assertAlmostEqual(line["feasible_rate"], 0.5)
        self.assertAlmostEqual(arc["feasible_rate"], 1.0)

    def test_paired_comparison_keeps_only_scenarios_where_line_is_feasible(self) -> None:
        best = select_best_candidates(_metrics())
        paired = paired_with_line(best)
        self.assertEqual(set(paired["scenario_id"]), {"S1"})
        self.assertEqual(len(paired), len(CURVE_ORDER) - 1)
        self.assertTrue((paired["joint_motion_ratio_to_line"] == 1.0).all())

    def test_representative_case_requires_curves_that_avoid_line_failure(self) -> None:
        metrics = _metrics()
        best = select_best_candidates(metrics)
        selected = select_representative_scenario(metrics, best)
        self.assertEqual(selected, "S2")

    def test_csv_loader_parses_boolean_strings_safely(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = _metrics().iloc[:2].copy()
            metrics["fully_reachable"] = ["False", "True"]
            metrics.to_csv(root / "metrics.csv", index=False)
            pd.DataFrame([{"scenario_id": "S1"}]).to_csv(
                root / "scenarios.csv", index=False
            )
            loaded, _ = load_phase2_data(root / "metrics.csv", root / "scenarios.csv")
            self.assertEqual(loaded["fully_reachable"].tolist(), [False, True])


if __name__ == "__main__":
    unittest.main()
