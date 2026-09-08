from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robot_planner.curves import circular_arc, cycloid_arc, line  # noqa: E402
from robot_planner.evaluation import CircleObstacle  # noqa: E402
from robot_planner.kinematics import PlanarArm2R  # noqa: E402
from robot_planner.phase4_evaluation import (  # noqa: E402
    numerical_derivative,
    reintegrate_sample_objectives,
    simulate_trajectory,
)


class Phase4EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.arm = PlanarArm2R(2.0, 2.0, base=(0.25, -0.5))

    def test_line_integrals_and_exported_samples_reintegrate(self) -> None:
        curve = line((1.25, 0.5), (2.25, 0.5), samples=161)
        result = simulate_trajectory(self.arm, curve, "up", samples=161)
        self.assertAlmostEqual(result.L, 1.0, places=10)
        self.assertAlmostEqual(result.K, 0.0, places=12)
        self.assertTrue(bool(result.metrics["path_regular"]))
        self.assertTrue(bool(result.metrics["selection_valid"]))
        recomputed = reintegrate_sample_objectives(result.sample_rows)
        for field in (
            "path_length_units",
            "joint_velocity_effort",
            "joint_acceleration_effort",
            "joint_jerk_effort",
            "regularized_curvature_energy_inv_unit",
            "obstacle_penalty_s_per_unit2",
        ):
            self.assertAlmostEqual(
                float(recomputed[field]), float(result.metrics[field]), places=10
            )

    def test_arc_converges_to_analytic_length_and_curvature(self) -> None:
        errors: list[float] = []
        for samples in (81, 161, 321):
            curve = circular_arc(
                (1.25, 0.5), (2.25, 0.5), bulge=0.3, samples=samples
            )
            result = simulate_trajectory(self.arm, curve, "up", samples=samples)
            exact_length = float(curve.parameters["radius"]) * abs(
                float(curve.parameters["theta_delta"])
            )
            exact_curvature_energy = abs(float(curve.parameters["theta_delta"])) / float(
                curve.parameters["radius"]
            )
            self.assertLess(abs(result.L - exact_length) / exact_length, 2e-4)
            unregularized = float(
                result.metrics["unregularized_curvature_energy_inv_unit"]
            )
            errors.append(abs(unregularized - exact_curvature_energy))
        # Exact built-in curve derivatives make the circular-arc integrands
        # constant, so all three grids are accurate to floating-point noise.
        self.assertTrue(all(error < 1e-12 for error in errors))

    def test_zero_bulge_arc_exports_its_straight_limit_equation(self) -> None:
        curve = circular_arc(
            (1.25, 0.5), (2.25, 0.5), bulge=0.0, samples=41
        )
        result = simulate_trajectory(self.arm, curve, "up", samples=41)
        self.assertIn("(1-u)", result.equations["x_u_equation"])
        self.assertIn("u*(2.25)", result.equations["x_u_equation"])
        self.assertAlmostEqual(result.L, 1.0, places=10)

    def test_cycloid_cusp_uses_explicit_regularization(self) -> None:
        curve = cycloid_arc(
            (1.25, 0.5), (2.25, 0.5), height_ratio=0.3, samples=161
        )
        result = simulate_trajectory(self.arm, curve, "up", samples=161)
        self.assertFalse(bool(result.metrics["path_regular"]))
        self.assertIsNone(result.metrics["unregularized_curvature_energy_inv_unit"])
        self.assertTrue(all(row["path_regular"] is False for row in result.sample_rows))
        recomputed = reintegrate_sample_objectives(result.sample_rows)
        self.assertIsNone(recomputed["unregularized_curvature_energy_inv_unit"])
        self.assertTrue(math.isfinite(result.K))
        self.assertIn("regularized", str(result.metrics["warning"]))

    def test_cycloid_regularized_curvature_converges_on_clustered_grid(self) -> None:
        values: list[float] = []
        for samples in (81, 161, 321, 641):
            curve = cycloid_arc(
                (1.25, 0.5), (2.25, 0.5), height_ratio=0.4, samples=samples
            )
            values.append(simulate_trajectory(self.arm, curve, "up", samples=samples).K)
        reference = values[-1]
        errors = [abs(value - reference) for value in values[:-1]]
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])
        self.assertLess(errors[1] / reference, 0.01)

    def test_uniform_u_validation_catches_gap_missed_by_time_grid(self) -> None:
        from robot_planner.experiments import curve_variants
        from robot_planner.phase4 import evaluate_phase4_variant, generate_phase4_scenarios

        scenario = generate_phase4_scenarios(64)[63]
        variant = next(
            item for item in curve_variants() if item.variant_id == "bezier_m40"
        )
        result = evaluate_phase4_variant(scenario, variant, "up", samples=161)
        self.assertTrue(bool(result.metric_row["time_grid_fully_reachable"]))
        self.assertFalse(bool(result.metric_row["validation_grid_fully_reachable"]))
        self.assertFalse(bool(result.metric_row["selection_valid"]))
        self.assertIn("uniform_u", str(result.metric_row["warning"]))

    def test_local_polynomial_derivatives_are_exact_for_polynomial(self) -> None:
        dt = 0.05
        times = [index * dt for index in range(61)]
        values = [2.0 * t**5 - 3.0 * t**3 + 7.0 * t - 4.0 for t in times]
        first = numerical_derivative(values, dt, 1)
        second = numerical_derivative(values, dt, 2)
        third = numerical_derivative(values, dt, 3)
        for t, d1, d2, d3 in zip(times, first, second, third):
            self.assertAlmostEqual(d1, 10.0 * t**4 - 9.0 * t**2 + 7.0, places=7)
            self.assertAlmostEqual(d2, 40.0 * t**3 - 18.0 * t, places=6)
            self.assertAlmostEqual(d3, 120.0 * t**2 - 18.0, places=5)

    def test_equations_include_base_branch_and_unwrap_policy(self) -> None:
        result = simulate_trajectory(
            self.arm,
            line((1.25, 0.5), (2.25, 0.5), samples=41),
            "up",
            samples=41,
        )
        equations = result.equations
        self.assertIn("x(t)-(0.25)", equations["theta1_t_equation"])
        self.assertIn("y(t)-(-0.5)", equations["theta1_t_equation"])
        self.assertIn("theta2(t)=-acos", equations["theta2_t_equation"])
        self.assertIn("principal", equations["branch_policy"])
        self.assertIn("2*pi", equations["unwrap_policy"])
        self.assertIn('"base": [0.25, -0.5]', equations["equation_descriptor_json"])

    def test_full_link_collision_is_a_hard_invalidity(self) -> None:
        arm = PlanarArm2R(1.0, 1.0)
        curve = line((1.8, 0.0), (1.6, 0.2), samples=41)
        obstacle = CircleObstacle("O1", 0.45, -0.22, 0.12)
        result = simulate_trajectory(
            arm, curve, "down", obstacles=(obstacle,), samples=41
        )
        self.assertFalse(bool(result.metrics["collision_free"]))
        self.assertFalse(bool(result.metrics["selection_valid"]))
        self.assertGreater(int(result.metrics["collision_samples"]), 0)
        self.assertTrue(any(bool(row["collision_at_sample"]) for row in result.sample_rows))


if __name__ == "__main__":
    unittest.main()
