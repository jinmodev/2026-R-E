from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robot_planner.experiments import curve_variants, generate_scenarios, run_experiments, write_dataset


class ExperimentTests(unittest.TestCase):
    def test_scenario_generation_is_deterministic(self) -> None:
        first = generate_scenarios(4, seed=1234)
        second = generate_scenarios(4, seed=1234)
        self.assertEqual(first, second)
        self.assertEqual(
            {item.scenario_type for item in first[:3]},
            {"local_move", "cross_inner", "boundary_sweep"},
        )

    def test_all_scenario_endpoints_are_reachable(self) -> None:
        for scenario in generate_scenarios(30, seed=55):
            self.assertGreaterEqual(scenario.start_radius, scenario.min_reach)
            self.assertLessEqual(scenario.start_radius, scenario.max_reach)
            self.assertGreaterEqual(scenario.goal_radius, scenario.min_reach)
            self.assertLessEqual(scenario.goal_radius, scenario.max_reach)

    def test_run_count_matches_experiment_grid(self) -> None:
        scenarios = generate_scenarios(2, seed=7)
        records = list(run_experiments(scenarios, samples=21))
        self.assertEqual(len(records), 2 * len(curve_variants()) * 2)
        self.assertEqual(len({record["run_id"] for record in records}), len(records))

    def test_line_path_ratio_is_one(self) -> None:
        scenario = generate_scenarios(1, seed=8)[0]
        records = [
            record for record in run_experiments([scenario], samples=31)
            if record["curve_key"] == "line"
        ]
        self.assertEqual(len(records), 2)
        for record in records:
            self.assertAlmostEqual(record["path_to_chord_ratio"], 1.0, places=9)

    def test_write_dataset_creates_auditable_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metadata = write_dataset(directory, scenario_count=3, seed=9, samples=21)
            destination = Path(directory)
            self.assertEqual(metadata["run_count"], 3 * len(curve_variants()) * 2)
            with (destination / "scenarios.csv").open(encoding="utf-8-sig") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 3)
            with (destination / "trajectory_metrics.csv").open(encoding="utf-8-sig") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), metadata["run_count"])
            loaded = json.loads(
                (destination / "experiment_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(loaded["seed"], 9)


if __name__ == "__main__":
    unittest.main()

