from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robot_planner.experiments import curve_variants  # noqa: E402
from robot_planner.phase3 import (  # noqa: E402
    generate_phase3_scenarios,
    run_phase3_experiments,
    score_scenario_candidates,
    write_phase3_dataset,
)


class Phase3Tests(unittest.TestCase):
    def test_obstacle_generation_is_deterministic_and_avoids_endpoints(self) -> None:
        first = generate_phase3_scenarios(5, seed=123)
        second = generate_phase3_scenarios(5, seed=123)
        self.assertEqual(first, second)
        for scenario in first:
            robot = scenario.robot
            for obstacle in scenario.obstacles:
                start_distance = math.hypot(
                    obstacle.center_x - robot.start_x,
                    obstacle.center_y - robot.start_y,
                )
                goal_distance = math.hypot(
                    obstacle.center_x - robot.goal_x,
                    obstacle.center_y - robot.goal_y,
                )
                self.assertGreater(start_distance, obstacle.radius_units)
                self.assertGreater(goal_distance, obstacle.radius_units)

    def test_phase3_grid_has_scores_and_expected_run_count(self) -> None:
        scenarios = generate_phase3_scenarios(1, seed=8)
        raw = list(run_phase3_experiments(scenarios, samples=21))
        scored = score_scenario_candidates(raw)
        self.assertEqual(len(scored), len(curve_variants()) * 2)
        self.assertEqual(len({row["run_id"] for row in scored}), len(scored))
        self.assertTrue(all("balanced_reward_0_100" in row for row in scored))
        self.assertTrue(all("model_electrical_energy_j" in row for row in scored))

    def test_data_split_is_assigned_by_whole_scenario(self) -> None:
        scenarios = generate_phase3_scenarios(10, seed=11)
        counts = {
            name: sum(item.data_split == name for item in scenarios)
            for name in ("train", "validation", "test")
        }
        self.assertEqual(counts, {"train": 6, "validation": 2, "test": 2})
        records = list(run_phase3_experiments(scenarios[:2], samples=21))
        by_scenario: dict[str, set[str]] = {}
        for row in records:
            by_scenario.setdefault(str(row["scenario_id"]), set()).add(
                str(row["data_split"])
            )
        self.assertTrue(all(len(values) == 1 for values in by_scenario.values()))

    def test_write_phase3_dataset_is_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metadata = write_phase3_dataset(
                directory,
                scenario_count=2,
                seed=9,
                samples=21,
            )
            destination = Path(directory)
            with (destination / "trajectory_metrics.csv").open(
                encoding="utf-8-sig"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2 * len(curve_variants()) * 2)
            loaded = json.loads(
                (destination / "experiment_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(loaded["dataset_version"], "phase3-multicriteria-v1")
            self.assertIn("evaluation_config", loaded)
            self.assertEqual(metadata["run_count"], len(rows))


if __name__ == "__main__":
    unittest.main()
