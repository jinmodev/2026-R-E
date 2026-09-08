"""Forward and inverse kinematics for a planar two-link (2R) robot arm."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin, sqrt
from typing import Iterable, Literal, Sequence


Point = tuple[float, float]
ElbowMode = Literal["up", "down"]


class UnreachablePointError(ValueError):
    """Raised when a requested end-effector point is outside the workspace."""

    def __init__(self, point: Point, index: int | None = None) -> None:
        self.point = point
        self.index = index
        location = f" at path index {index}" if index is not None else ""
        super().__init__(f"Point {point!r}{location} is outside the 2R workspace.")


@dataclass(frozen=True, slots=True)
class ArmPose:
    """World coordinates of the base, elbow, and end effector."""

    base: Point
    elbow: Point
    end: Point


@dataclass(frozen=True, slots=True)
class JointSolution:
    """A single inverse-kinematics branch in radians."""

    theta1: float
    theta2: float
    elbow_mode: ElbowMode

    @property
    def degrees(self) -> tuple[float, float]:
        factor = 180.0 / pi
        return self.theta1 * factor, self.theta2 * factor

    @property
    def singularity_index(self) -> float:
        """Return |sin(theta2)|; zero means a fully folded/extended singularity."""

        return abs(sin(self.theta2))


def _unwrap(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    result = [values[0]]
    for value in values[1:]:
        adjusted = value
        previous = result[-1]
        while adjusted - previous > pi:
            adjusted -= 2.0 * pi
        while adjusted - previous < -pi:
            adjusted += 2.0 * pi
        result.append(adjusted)
    return result


@dataclass(frozen=True, slots=True)
class PlanarArm2R:
    """A fixed-base planar arm with two revolute joints."""

    link1: float
    link2: float
    base: Point = (0.0, 0.0)

    def __post_init__(self) -> None:
        if self.link1 <= 0.0 or self.link2 <= 0.0:
            raise ValueError("Both link lengths must be positive.")

    @property
    def min_reach(self) -> float:
        return abs(self.link1 - self.link2)

    @property
    def max_reach(self) -> float:
        return self.link1 + self.link2

    def radial_distance(self, point: Point) -> float:
        return hypot(point[0] - self.base[0], point[1] - self.base[1])

    def is_reachable(self, point: Point, tolerance: float = 1e-9) -> bool:
        radius = self.radial_distance(point)
        return self.min_reach - tolerance <= radius <= self.max_reach + tolerance

    def forward(self, theta1: float, theta2: float) -> ArmPose:
        bx, by = self.base
        ex = bx + self.link1 * cos(theta1)
        ey = by + self.link1 * sin(theta1)
        px = ex + self.link2 * cos(theta1 + theta2)
        py = ey + self.link2 * sin(theta1 + theta2)
        return ArmPose(base=self.base, elbow=(ex, ey), end=(px, py))

    def inverse(self, point: Point, elbow_mode: ElbowMode = "up") -> JointSolution:
        if elbow_mode not in ("up", "down"):
            raise ValueError("elbow_mode must be 'up' or 'down'.")
        if not self.is_reachable(point):
            raise UnreachablePointError(point)

        x = point[0] - self.base[0]
        y = point[1] - self.base[1]
        cosine2 = (
            x * x + y * y - self.link1 * self.link1 - self.link2 * self.link2
        ) / (2.0 * self.link1 * self.link2)
        cosine2 = max(-1.0, min(1.0, cosine2))
        sine_magnitude = sqrt(max(0.0, 1.0 - cosine2 * cosine2))

        # With a target on the positive x-axis, negative theta2 places the
        # elbow above the base-target line, hence the branch names below.
        sine2 = -sine_magnitude if elbow_mode == "up" else sine_magnitude
        theta2 = atan2(sine2, cosine2)
        theta1 = atan2(y, x) - atan2(
            self.link2 * sine2,
            self.link1 + self.link2 * cosine2,
        )
        return JointSolution(theta1=theta1, theta2=theta2, elbow_mode=elbow_mode)

    def inverse_all(self, point: Point) -> tuple[JointSolution, JointSolution]:
        return self.inverse(point, "up"), self.inverse(point, "down")

    def inverse_path(
        self,
        points: Iterable[Point],
        elbow_mode: ElbowMode = "up",
        *,
        unwrap: bool = True,
    ) -> list[JointSolution]:
        raw: list[JointSolution] = []
        for index, point in enumerate(points):
            try:
                raw.append(self.inverse(point, elbow_mode))
            except UnreachablePointError as error:
                raise UnreachablePointError(error.point, index=index) from error

        if not unwrap or not raw:
            return raw

        theta1 = _unwrap([solution.theta1 for solution in raw])
        theta2 = _unwrap([solution.theta2 for solution in raw])
        return [
            JointSolution(a1, a2, elbow_mode)
            for a1, a2 in zip(theta1, theta2, strict=True)
        ]

