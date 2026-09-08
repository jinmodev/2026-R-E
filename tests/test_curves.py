from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robot_planner.curves import circular_arc, line, make_curve


class CurveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.start = (1.2, -0.4)
        self.end = (4.1, 2.3)

    def assert_point_close(self, actual, expected) -> None:
        self.assertAlmostEqual(actual[0], expected[0], places=8)
        self.assertAlmostEqual(actual[1], expected[1], places=8)

    def test_all_curve_families_match_endpoints(self) -> None:
        for kind in ("line", "arc", "bezier", "hermite", "cycloid", "limacon"):
            with self.subTest(kind=kind):
                curve = make_curve(kind, self.start, self.end, shape=0.4, samples=101)
                self.assertEqual(len(curve.points), 101)
                self.assert_point_close(curve.start, self.start)
                self.assert_point_close(curve.end, self.end)
                self.assertTrue(curve.equation)
                self.assertGreaterEqual(curve.length, math.dist(self.start, self.end) - 1e-8)

    def test_line_length_is_chord_length(self) -> None:
        curve = line(self.start, self.end)
        self.assertAlmostEqual(curve.length, math.dist(self.start, self.end), places=9)

    def test_circular_arc_has_constant_radius(self) -> None:
        curve = circular_arc(self.start, self.end, bulge=0.3)
        center = curve.parameters["center_x"], curve.parameters["center_y"]
        radius = curve.parameters["radius"]
        for point in curve.points[::20]:
            self.assertAlmostEqual(math.dist(point, center), radius, places=8)

    def test_zero_bulge_is_well_defined(self) -> None:
        curve = circular_arc(self.start, self.end, bulge=0.0)
        self.assert_point_close(curve.start, self.start)
        self.assert_point_close(curve.end, self.end)

    def test_equal_endpoints_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_curve("bezier", self.start, self.start)


if __name__ == "__main__":
    unittest.main()

