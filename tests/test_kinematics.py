from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robot_planner.kinematics import PlanarArm2R, UnreachablePointError


class PlanarArm2RTests(unittest.TestCase):
    def setUp(self) -> None:
        self.arm = PlanarArm2R(3.0, 2.0)

    def test_workspace_bounds(self) -> None:
        self.assertTrue(self.arm.is_reachable((5.0, 0.0)))
        self.assertTrue(self.arm.is_reachable((1.0, 0.0)))
        self.assertFalse(self.arm.is_reachable((5.1, 0.0)))
        self.assertFalse(self.arm.is_reachable((0.5, 0.0)))

    def test_inverse_forward_round_trip_for_both_branches(self) -> None:
        target = (3.2, 1.1)
        for mode in ("up", "down"):
            solution = self.arm.inverse(target, mode)
            pose = self.arm.forward(solution.theta1, solution.theta2)
            self.assertAlmostEqual(pose.end[0], target[0], places=9)
            self.assertAlmostEqual(pose.end[1], target[1], places=9)

    def test_two_branches_have_opposite_second_joint_signs(self) -> None:
        up, down = self.arm.inverse_all((3.0, 0.5))
        self.assertLessEqual(up.theta2, 0.0)
        self.assertGreaterEqual(down.theta2, 0.0)

    def test_unreachable_point_raises(self) -> None:
        with self.assertRaises(UnreachablePointError):
            self.arm.inverse((7.0, 0.0))

    def test_path_failure_reports_index(self) -> None:
        with self.assertRaises(UnreachablePointError) as context:
            self.arm.inverse_path([(3.0, 0.0), (7.0, 0.0)])
        self.assertEqual(context.exception.index, 1)

    def test_path_angles_are_unwrapped(self) -> None:
        points = [
            (3.5 * math.cos(angle), 3.5 * math.sin(angle))
            for angle in (2.9, 3.0, 3.1, 3.2, 3.3)
        ]
        solutions = self.arm.inverse_path(points, "up")
        jumps = [
            abs(right.theta1 - left.theta1)
            for left, right in zip(solutions, solutions[1:])
        ]
        self.assertLess(max(jumps), 0.2)


if __name__ == "__main__":
    unittest.main()

