from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robot_planner.curves import circular_arc, line  # noqa: E402
from robot_planner.evaluation import (  # noqa: E402
    CircleObstacle,
    EvaluationConfig,
    discrete_curvature_metrics,
    evaluate_trajectory,
    minimum_jerk_progress,
    pose_obstacle_clearance,
)
from robot_planner.kinematics import PlanarArm2R  # noqa: E402
from robot_planner.phase3 import score_scenario_candidates  # noqa: E402


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.arm = PlanarArm2R(2.0, 1.0)
        self.path = line((1.5, 0.4), (1.1, 1.5), samples=81)

    def test_minimum_jerk_time_law_has_expected_endpoints(self) -> None:
        self.assertEqual(minimum_jerk_progress(0.0), 0.0)
        self.assertEqual(minimum_jerk_progress(1.0), 1.0)
        self.assertAlmostEqual(minimum_jerk_progress(0.5), 0.5)
        with self.assertRaises(ValueError):
            minimum_jerk_progress(1.01)

    def test_line_curvature_is_zero_and_arc_curvature_is_positive(self) -> None:
        straight = discrete_curvature_metrics(self.path.points)
        curved = discrete_curvature_metrics(
            circular_arc((1.5, 0.4), (1.1, 1.5), bulge=0.3, samples=81).points
        )
        self.assertLess(straight.max_curvature_inv_m, 1e-9)
        self.assertGreater(curved.mean_abs_curvature_inv_m, 0.0)
        self.assertGreater(curved.curvature_energy_inv_m, 0.0)

    def test_full_link_collision_is_detected(self) -> None:
        arm = PlanarArm2R(1.0, 1.0)
        solution = arm.inverse((2.0, 0.0), "down")
        obstacle = CircleObstacle("O1", 0.5, 0.0, 0.1)
        clearance = pose_obstacle_clearance(
            arm,
            solution,
            [obstacle],
            link_radius_units=0.05,
        )
        self.assertIsNotNone(clearance)
        self.assertLessEqual(float(clearance), 0.0)

    def test_evaluation_produces_nonnegative_model_energy_and_jerk(self) -> None:
        result = evaluate_trajectory(
            self.arm,
            self.path,
            "up",
            config=EvaluationConfig(duration_s=3.0),
        )
        self.assertTrue(result.fully_reachable)
        self.assertTrue(result.valid_trajectory)
        self.assertGreater(float(result.model_electrical_energy_j), 0.0)
        self.assertGreaterEqual(float(result.motor_copper_loss_j), 0.0)
        self.assertGreaterEqual(float(result.joint_jerk_effort_rad2_per_s5), 0.0)
        self.assertTrue(math.isfinite(float(result.model_electrical_energy_j)))

    def test_slower_horizontal_motion_uses_less_model_energy(self) -> None:
        fast = evaluate_trajectory(
            self.arm,
            self.path,
            "up",
            config=EvaluationConfig(duration_s=2.0),
        )
        slow = evaluate_trajectory(
            self.arm,
            self.path,
            "up",
            config=EvaluationConfig(duration_s=5.0),
        )
        self.assertLess(
            float(slow.model_electrical_energy_j),
            float(fast.model_electrical_energy_j),
        )

    def test_jerk_estimate_is_stable_under_finer_sampling(self) -> None:
        coarse = evaluate_trajectory(
            self.arm,
            line((1.5, 0.4), (1.1, 1.5), samples=81),
            "up",
        )
        fine = evaluate_trajectory(
            self.arm,
            line((1.5, 0.4), (1.1, 1.5), samples=161),
            "up",
        )
        ratio = (
            float(fine.joint_jerk_effort_rad2_per_s5)
            / float(coarse.joint_jerk_effort_rad2_per_s5)
        )
        self.assertGreater(ratio, 0.7)
        self.assertLess(ratio, 1.3)

    def test_scoring_uses_hard_constraints_and_profile_weights(self) -> None:
        base = {
            "scenario_id": "S1",
            "valid_trajectory": True,
            "curve_length": 1.0,
            "curve_length_m": 0.1,
            "model_electrical_energy_j": 5.0,
            "joint_jerk_effort_rad2_per_s5": 3.0,
            "curvature_energy_inv_m": 2.0,
            "singularity_penalty_s": 4.0,
            "clearance_penalty_s_per_m2": 6.0,
        }
        longer = dict(base, run_id="long", curve_length=2.0, curve_length_m=0.2)
        shorter = dict(base, run_id="short", curve_length=1.0, curve_length_m=0.1)
        invalid = dict(base, run_id="invalid", valid_trajectory=False)
        scored = score_scenario_candidates([longer, shorter, invalid])
        by_id = {str(row["run_id"]): row for row in scored}
        self.assertLess(
            float(by_id["short"]["shortest_cost_0_100"]),
            float(by_id["long"]["shortest_cost_0_100"]),
        )
        self.assertEqual(by_id["invalid"]["balanced_reward_0_100"], 0.0)
        self.assertEqual(by_id["invalid"]["balanced_cost_0_100"], 1_000_000.0)


if __name__ == "__main__":
    unittest.main()
