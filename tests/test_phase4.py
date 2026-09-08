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
from robot_planner.phase4 import (  # noqa: E402
    CODE_VERSION,
    RAW_OBJECTIVES,
    SAMPLE_SCHEMA_VERSION,
    WINNER_FIELDS,
    evaluate_phase4_variant,
    generate_phase4_scenarios,
    score_phase4_candidates,
    select_scenario_winners,
    write_phase4_dataset,
)


class Phase4PipelineTests(unittest.TestCase):
    def test_scenario_generation_is_deterministic_diverse_and_endpoint_safe(self) -> None:
        first = generate_phase4_scenarios(30, seed=321)
        second = generate_phase4_scenarios(30, seed=321)
        self.assertEqual(first, second)
        self.assertEqual(len({item.robot.scenario_id for item in first}), 30)
        self.assertGreaterEqual(len({item.obstacle_layout for item in first}), 7)
        self.assertEqual(
            {name: sum(item.data_split == name for item in first) for name in ("train", "validation", "test")},
            {"train": 18, "validation": 6, "test": 6},
        )
        for scenario in first:
            robot = scenario.robot
            for obstacle in scenario.obstacles:
                for point in (
                    (robot.base_x, robot.base_y),
                    (robot.start_x, robot.start_y),
                    (robot.goal_x, robot.goal_y),
                ):
                    self.assertGreater(
                        math.dist(obstacle.center, point), obstacle.radius_units
                    )

    def test_seeded_split_does_not_confound_scenario_types_or_layouts(self) -> None:
        scenarios = generate_phase4_scenarios(200, seed=321)
        expected = {"train", "validation", "test"}
        splits_by_type: dict[str, set[str]] = {}
        splits_by_layout: dict[str, set[str]] = {}
        for scenario in scenarios:
            splits_by_type.setdefault(scenario.robot.scenario_type, set()).add(
                scenario.data_split
            )
            splits_by_layout.setdefault(scenario.obstacle_layout, set()).add(
                scenario.data_split
            )
        self.assertTrue(all(value == expected for value in splits_by_type.values()))
        self.assertTrue(all(value == expected for value in splits_by_layout.values()))

    def test_candidate_row_contains_equations_samples_and_math_metrics(self) -> None:
        scenario = generate_phase4_scenarios(1, seed=8)[0]
        run = evaluate_phase4_variant(
            scenario, curve_variants()[0], "up", samples=21
        )
        self.assertEqual(len(run.sample_rows), 21)
        self.assertIn("theta1_t_equation", run.metric_row)
        self.assertIn("equation_descriptor_json", run.metric_row)
        self.assertIn("joint_velocity_effort", run.metric_row)
        self.assertIn("regularized_curvature_energy_inv_unit", run.metric_row)
        self.assertIn("theta1_rad", run.sample_rows[0])
        self.assertIn("theta1_jerk_rad_s3", run.sample_rows[0])

    def _fake_row(self, scenario_id: str, run_id: str, value: float, valid: bool = True) -> dict[str, object]:
        row: dict[str, object] = {
            "scenario_id": scenario_id,
            "run_id": run_id,
            "data_split": "train",
            "dataset_partition": "in_distribution",
            "selection_valid": valid,
            "curve_key": "line",
            "curve_name_ko": "직선",
            "shape": 0.0,
            "elbow_mode": "up" if run_id.endswith("a") else "down",
            "curve_equation_template": "p(u)",
            "curve_equation": "p(u)=(x(u),y(u))",
            "curve_parameters_json": "{}",
            "x_t_equation": "x(t)",
            "y_t_equation": "y(t)",
            "theta1_t_equation": "theta1(t)",
            "theta2_t_equation": "theta2(t)",
            "equation_format_version": "phase4-equations-v1",
            "equation_descriptor_json": "{}",
        }
        for field in RAW_OBJECTIVES.values():
            row[field] = value
        return row

    def test_scoring_winners_ties_and_no_valid_schema(self) -> None:
        records = [
            self._fake_row("A_no_valid", "bad", 1.0, valid=False),
            self._fake_row("B_tie", "candidate_a", 1.0),
            self._fake_row("B_tie", "candidate_b", 1.0),
        ]
        scored = score_phase4_candidates(records)
        winners = select_scenario_winners(scored)
        self.assertEqual(len(winners), 12)
        self.assertTrue(all(tuple(row) == WINNER_FIELDS for row in winners))
        invalid = [row for row in winners if row["scenario_id"] == "A_no_valid"]
        self.assertTrue(all(row["winner_status"] == "no_valid_candidate" for row in invalid))
        tied = [row for row in winners if row["scenario_id"] == "B_tie"]
        self.assertTrue(all(row["winner_status"] == "tie_resolved" for row in tied))
        self.assertTrue(all(row["tie_count"] == 2 for row in tied))
        self.assertTrue(all(row["winner_run_id"] == "candidate_b" for row in tied))

    def test_scoring_rejects_nonfinite_objective_on_valid_candidate(self) -> None:
        row = self._fake_row("finite_guard", "candidate_a", 1.0)
        row[RAW_OBJECTIVES["jerk"]] = math.nan
        with self.assertRaisesRegex(ValueError, "finite values"):
            score_phase4_candidates([row])

    def test_small_parquet_dataset_and_resume_are_auditable(self) -> None:
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as directory:
            progress_messages: list[str] = []
            metadata = write_phase4_dataset(
                directory,
                scenario_count=2,
                challenge_count=1,
                seed=44,
                samples=21,
                batch_size=1,
                sample_format="parquet",
                progress_callback=progress_messages.append,
            )
            destination = Path(directory)
            with (destination / "scenarios.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                scenarios = list(csv.DictReader(handle))
            with (destination / "trajectory_metrics.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                metrics = list(csv.DictReader(handle))
            with (destination / "scenario_winners.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                winners = list(csv.DictReader(handle))
            self.assertEqual(len(metrics), 3 * len(curve_variants()) * 2)
            self.assertEqual(len(winners), 3 * 6)
            self.assertEqual(len({row["scenario_id"] for row in scenarios}), 3)
            self.assertEqual(len({row["run_id"] for row in metrics}), len(metrics))
            self.assertEqual(metadata["sample_storage_format"], "parquet")
            parquet_files = sorted((destination / "trajectory_samples").rglob("*.parquet"))
            self.assertTrue(parquet_files)
            self.assertEqual(
                sum(pq.ParquetFile(path).metadata.num_rows for path in parquet_files),
                len(metrics) * 21,
            )
            sample_rows = [
                row
                for path in parquet_files
                for row in pq.ParquetFile(path).read().to_pylist()
            ]
            self.assertEqual(
                len({(row["run_id"], row["sample_index"]) for row in sample_rows}),
                len(sample_rows),
            )

            scenario_geometry = {
                row["scenario_id"]: (
                    float(row["start_x"]),
                    float(row["start_y"]),
                    float(row["goal_x"]),
                    float(row["goal_y"]),
                )
                for row in scenarios
            }
            endpoints: dict[str, dict[int, dict[str, object]]] = {}
            for row in sample_rows:
                sample_index = int(row["sample_index"])
                if sample_index in {0, 20}:
                    endpoints.setdefault(str(row["run_id"]), {})[sample_index] = row
            self.assertEqual(len(endpoints), len(metrics))
            for run_id, endpoint_rows in endpoints.items():
                scenario_id = str(endpoint_rows[0]["scenario_id"])
                start_x, start_y, goal_x, goal_y = scenario_geometry[scenario_id]
                self.assertAlmostEqual(float(endpoint_rows[0]["x"]), start_x, places=10)
                self.assertAlmostEqual(float(endpoint_rows[0]["y"]), start_y, places=10)
                self.assertAlmostEqual(float(endpoint_rows[20]["x"]), goal_x, places=10)
                self.assertAlmostEqual(float(endpoint_rows[20]["y"]), goal_y, places=10)

            split_by_scenario: dict[str, set[str]] = {}
            for row in metrics:
                split_by_scenario.setdefault(row["scenario_id"], set()).add(row["data_split"])
            for row in winners:
                split_by_scenario.setdefault(row["scenario_id"], set()).add(row["data_split"])
            for row in sample_rows:
                split_by_scenario.setdefault(str(row["scenario_id"]), set()).add(
                    str(row["data_split"])
                )
            self.assertTrue(all(len(values) == 1 for values in split_by_scenario.values()))

            valid_run_ids: set[str] = set()
            finite_metric_fields = [
                *RAW_OBJECTIVES.values(),
                *(f"normalized_{name}" for name in RAW_OBJECTIVES),
                "balanced_cost_0_100",
                "balanced_reward_0_100",
            ]
            for row in metrics:
                if row["selection_valid"] != "True":
                    continue
                valid_run_ids.add(row["run_id"])
                for field in finite_metric_fields:
                    self.assertTrue(math.isfinite(float(row[field])), (row["run_id"], field))
            finite_sample_fields = (
                "theta1_rad",
                "theta2_rad",
                "theta1_velocity_rad_s",
                "theta2_velocity_rad_s",
                "theta1_acceleration_rad_s2",
                "theta2_acceleration_rad_s2",
                "theta1_jerk_rad_s3",
                "theta2_jerk_rad_s3",
            )
            for row in sample_rows:
                if row["run_id"] in valid_run_ids:
                    for field in finite_sample_fields:
                        self.assertTrue(math.isfinite(float(row[field])), (row["run_id"], field))

            self.assertIn("obstacle_generation", metadata)
            self.assertIn("link_length_generation", metadata)
            self.assertIn("actual_by_dataset_partition", metadata["obstacle_generation"])
            self.assertIn("actual_by_dataset_partition", metadata["link_length_generation"])
            self.assertEqual(metadata["code_version"], CODE_VERSION)
            self.assertEqual(metadata["sample_schema_version"], SAMPLE_SCHEMA_VERSION)
            self.assertEqual(metadata["trajectory_sample_row_count"], len(sample_rows))
            self.assertTrue(
                any(
                    "overall=100.0%" in message
                    and "expected total=" in message
                    and "valid=" in message
                    and "reachability:" in message
                    and "collision:" in message
                    for message in progress_messages
                )
            )
            first_manifest = json.loads(
                (destination / "dataset_manifest.json").read_text(encoding="utf-8")
            )
            row_counts = metadata["file_row_counts"]
            for item in first_manifest["files"]:
                if item["path"] in row_counts:
                    self.assertEqual(item["row_count"], row_counts[item["path"]])
            first_hashes = {
                item["path"]: item["sha256"]
                for item in first_manifest["files"]
                if item["path"] != "experiment_metadata.json"
            }
            write_phase4_dataset(
                directory,
                scenario_count=2,
                challenge_count=1,
                seed=44,
                samples=21,
                batch_size=1,
                sample_format="parquet",
                resume=True,
            )
            second_manifest = json.loads(
                (destination / "dataset_manifest.json").read_text(encoding="utf-8")
            )
            second_hashes = {
                item["path"]: item["sha256"]
                for item in second_manifest["files"]
                if item["path"] != "experiment_metadata.json"
            }
            self.assertEqual(first_hashes, second_hashes)

            # A state created before explicit schema/code versioning must never
            # silently mix its old partitions with the current sample schema.
            state_path = destination / "generation_state.json"
            old_state = json.loads(state_path.read_text(encoding="utf-8"))
            old_state["signature"].pop("code_version")
            old_state["signature"].pop("sample_schema_version")
            state_path.write_text(
                json.dumps(old_state, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Resume parameters"):
                write_phase4_dataset(
                    directory,
                    scenario_count=2,
                    challenge_count=1,
                    seed=44,
                    samples=21,
                    batch_size=1,
                    sample_format="parquet",
                    resume=True,
                )

    def test_fresh_datasets_with_same_seed_have_identical_data_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            roots = [Path(directory) / name for name in ("first", "second")]
            hash_sets: list[dict[str, str]] = []
            for root in roots:
                write_phase4_dataset(
                    root,
                    scenario_count=1,
                    challenge_count=0,
                    seed=90210,
                    samples=7,
                    batch_size=1,
                    sample_format="parquet",
                )
                manifest = json.loads(
                    (root / "dataset_manifest.json").read_text(encoding="utf-8")
                )
                hash_sets.append(
                    {
                        item["path"]: item["sha256"]
                        for item in manifest["files"]
                        if item["path"] != "experiment_metadata.json"
                    }
                )
            self.assertEqual(hash_sets[0], hash_sets[1])


if __name__ == "__main__":
    unittest.main()
