"""Mathematics-first trajectory sampling and numerical evaluation.

This module deliberately keeps the Phase-4 objective functions independent
from the rigid-body and motor model in :mod:`robot_planner.evaluation`.
Every reported integral is reconstructed from the returned sample rows, which
makes the data suitable for classroom explanation and later auditing.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from typing import Iterable, Mapping, Sequence

from .curves import CurvePath
from .evaluation import CircleObstacle, minimum_jerk_progress, pose_obstacle_clearance
from .kinematics import JointSolution, PlanarArm2R


Number = float | int


@dataclass(frozen=True, slots=True)
class MathematicalEvaluationConfig:
    """Numerical assumptions for the mathematics-first objectives."""

    duration_s: float = 3.0
    link_radius_units: float = 0.05
    clearance_epsilon_units: float = 0.05
    curvature_speed_epsilon_ratio: float = 1e-3
    derivative_stencil: int = 7
    singularity_warning_threshold: float = 0.05
    equation_precision: int = 12

    def __post_init__(self) -> None:
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive.")
        if self.link_radius_units < 0.0:
            raise ValueError("link_radius_units cannot be negative.")
        if self.clearance_epsilon_units <= 0.0:
            raise ValueError("clearance_epsilon_units must be positive.")
        if self.curvature_speed_epsilon_ratio <= 0.0:
            raise ValueError("curvature_speed_epsilon_ratio must be positive.")
        if self.derivative_stencil < 5 or self.derivative_stencil % 2 == 0:
            raise ValueError("derivative_stencil must be an odd integer of at least 5.")
        if not 0.0 < self.singularity_warning_threshold < 1.0:
            raise ValueError("singularity_warning_threshold must lie between 0 and 1.")
        if self.equation_precision < 6:
            raise ValueError("equation_precision must be at least 6.")


@dataclass(frozen=True, slots=True)
class MathematicalTrajectoryResult:
    """One fully auditable trajectory evaluation for GUI and data export."""

    sample_rows: tuple[dict[str, object], ...]
    metrics: dict[str, object]
    equations: dict[str, str]

    @property
    def frames(self) -> tuple[dict[str, object], ...]:
        """Alias used by animation code."""

        return self.sample_rows

    @property
    def L(self) -> float:
        return float(self.metrics["path_length_units"])

    @property
    def E_theta(self) -> float | None:
        value = self.metrics["joint_velocity_effort"]
        return None if value is None else float(value)

    @property
    def A(self) -> float | None:
        value = self.metrics["joint_acceleration_effort"]
        return None if value is None else float(value)

    @property
    def J(self) -> float | None:
        value = self.metrics["joint_jerk_effort"]
        return None if value is None else float(value)

    @property
    def K(self) -> float:
        return float(self.metrics["regularized_curvature_energy_inv_unit"])

    @property
    def O(self) -> float | None:
        value = self.metrics["obstacle_penalty_s_per_unit2"]
        return None if value is None else float(value)


def _fmt(value: Number, precision: int) -> str:
    rounded = 0.0 if abs(float(value)) < 10.0 ** (-precision) else float(value)
    return format(rounded, f".{precision}g")


def _natural_cubic_second_derivatives(values: Sequence[float]) -> list[float]:
    """Return natural-cubic second derivatives for uniform u in [0, 1]."""

    count = len(values)
    if count <= 2:
        return [0.0] * count
    interior = count - 2
    step = 1.0 / (count - 1)
    diagonal = [4.0] * interior
    upper = [1.0] * max(0, interior - 1)
    right = [
        6.0 * (values[index + 1] - 2.0 * values[index] + values[index - 1])
        / step**2
        for index in range(1, count - 1)
    ]
    for index in range(1, interior):
        factor = 1.0 / diagonal[index - 1]
        diagonal[index] -= factor * upper[index - 1]
        right[index] -= factor * right[index - 1]
    solved = [0.0] * interior
    solved[-1] = right[-1] / diagonal[-1]
    for index in range(interior - 2, -1, -1):
        solved[index] = (
            right[index] - upper[index] * solved[index + 1]
        ) / diagonal[index]
    return [0.0, *solved, 0.0]


def _spline_value_derivatives(
    values: Sequence[float],
    second_derivatives: Sequence[float],
    progress: float,
) -> tuple[float, float, float]:
    """Evaluate a uniform natural cubic and its first two u-derivatives."""

    progress = max(0.0, min(1.0, progress))
    step = 1.0 / (len(values) - 1)
    position = progress / step
    left = min(int(math.floor(position)), len(values) - 2)
    fraction = position - left
    a = 1.0 - fraction
    b = fraction
    m0 = second_derivatives[left]
    m1 = second_derivatives[left + 1]
    value = (
        a * values[left]
        + b * values[left + 1]
        + ((a**3 - a) * m0 + (b**3 - b) * m1) * step**2 / 6.0
    )
    first = (
        (values[left + 1] - values[left]) / step
        + step * ((-3.0 * a * a + 1.0) * m0 + (3.0 * b * b - 1.0) * m1) / 6.0
    )
    second = a * m0 + b * m1
    return value, first, second


def _curve_value_derivatives(
    curve: CurvePath,
    progress: float,
) -> tuple[float, float, float, float, float, float]:
    """Evaluate every built-in curve and its first two exact u-derivatives.

    ``CurvePath`` intentionally stores sampled points for the legacy GUI.  The
    Phase-4 integrals, however, should follow the generating equation rather
    than derivatives of an interpolating spline.  Keeping the formulas here
    also resolves the sharp regularized-curvature layer at cycloid cusps.
    """

    u = max(0.0, min(1.0, float(progress)))
    x0, y0 = curve.start
    x1, y1 = curve.end
    parameters = curve.parameters

    if curve.key == "line" or (
        curve.key == "arc" and "center_x" not in parameters
    ):
        dx = x1 - x0
        dy = y1 - y0
        return x0 + u * dx, y0 + u * dy, dx, dy, 0.0, 0.0

    if curve.key == "arc":
        center_x = float(parameters["center_x"])
        center_y = float(parameters["center_y"])
        radius = float(parameters["radius"])
        theta_start = float(parameters["theta_start"])
        theta_delta = float(parameters["theta_delta"])
        angle = theta_start + theta_delta * u
        cosine = math.cos(angle)
        sine = math.sin(angle)
        return (
            center_x + radius * cosine,
            center_y + radius * sine,
            -radius * theta_delta * sine,
            radius * theta_delta * cosine,
            -radius * theta_delta**2 * cosine,
            -radius * theta_delta**2 * sine,
        )

    if curve.key == "bezier":
        c1 = (
            float(parameters["control1_x"]),
            float(parameters["control1_y"]),
        )
        c2 = (
            float(parameters["control2_x"]),
            float(parameters["control2_y"]),
        )
        one_minus = 1.0 - u
        value_weights = (
            one_minus**3,
            3.0 * one_minus**2 * u,
            3.0 * one_minus * u**2,
            u**3,
        )
        controls = ((x0, y0), c1, c2, (x1, y1))
        x = math.fsum(weight * point[0] for weight, point in zip(value_weights, controls))
        y = math.fsum(weight * point[1] for weight, point in zip(value_weights, controls))
        first_weights = (
            -3.0 * one_minus**2,
            3.0 * one_minus**2 - 6.0 * one_minus * u,
            6.0 * one_minus * u - 3.0 * u**2,
            3.0 * u**2,
        )
        second_weights = (
            6.0 * one_minus,
            -12.0 * one_minus + 6.0 * u,
            6.0 * one_minus - 12.0 * u,
            6.0 * u,
        )
        dx = math.fsum(weight * point[0] for weight, point in zip(first_weights, controls))
        dy = math.fsum(weight * point[1] for weight, point in zip(first_weights, controls))
        ddx = math.fsum(weight * point[0] for weight, point in zip(second_weights, controls))
        ddy = math.fsum(weight * point[1] for weight, point in zip(second_weights, controls))
        return x, y, dx, dy, ddx, ddy

    if curve.key == "hermite":
        tangent0 = (
            float(parameters["tangent0_x"]),
            float(parameters["tangent0_y"]),
        )
        tangent1 = (
            float(parameters["tangent1_x"]),
            float(parameters["tangent1_y"]),
        )
        h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
        h10 = u**3 - 2.0 * u**2 + u
        h01 = -2.0 * u**3 + 3.0 * u**2
        h11 = u**3 - u**2
        dh00, dh10 = 6.0 * u**2 - 6.0 * u, 3.0 * u**2 - 4.0 * u + 1.0
        dh01, dh11 = -dh00, 3.0 * u**2 - 2.0 * u
        ddh00, ddh10 = 12.0 * u - 6.0, 6.0 * u - 4.0
        ddh01, ddh11 = -ddh00, 6.0 * u - 2.0
        bases = ((x0, y0), tangent0, (x1, y1), tangent1)
        values = (h00, h10, h01, h11)
        firsts = (dh00, dh10, dh01, dh11)
        seconds = (ddh00, ddh10, ddh01, ddh11)
        return (
            math.fsum(weight * point[0] for weight, point in zip(values, bases)),
            math.fsum(weight * point[1] for weight, point in zip(values, bases)),
            math.fsum(weight * point[0] for weight, point in zip(firsts, bases)),
            math.fsum(weight * point[1] for weight, point in zip(firsts, bases)),
            math.fsum(weight * point[0] for weight, point in zip(seconds, bases)),
            math.fsum(weight * point[1] for weight, point in zip(seconds, bases)),
        )

    if curve.key == "cycloid":
        chord_x = x1 - x0
        chord_y = y1 - y0
        chord = math.hypot(chord_x, chord_y)
        normal_x, normal_y = -chord_y / chord, chord_x / chord
        height = float(parameters["height"])
        angle = 2.0 * math.pi * u
        forward = (angle - math.sin(angle)) / (2.0 * math.pi)
        transverse = (1.0 - math.cos(angle)) / 2.0
        forward_first = 1.0 - math.cos(angle)
        transverse_first = math.pi * math.sin(angle)
        forward_second = 2.0 * math.pi * math.sin(angle)
        transverse_second = 2.0 * math.pi**2 * math.cos(angle)
        return (
            x0 + chord_x * forward + normal_x * height * transverse,
            y0 + chord_y * forward + normal_y * height * transverse,
            chord_x * forward_first + normal_x * height * transverse_first,
            chord_y * forward_first + normal_y * height * transverse_first,
            chord_x * forward_second + normal_x * height * transverse_second,
            chord_y * forward_second + normal_y * height * transverse_second,
        )

    if curve.key == "limacon":
        a = float(parameters["a"])
        b = float(parameters["b"])
        phase = float(parameters["phase"])
        sweep = float(parameters["sweep"])
        side = float(parameters["side"])
        scale = float(parameters["scale"])
        rotation = float(parameters["rotation"])

        def raw(phi: float) -> tuple[float, float, float, float, float, float]:
            cosine = math.cos(phi)
            sine = math.sin(phi)
            radius = a + b * cosine
            radius_first = -b * sine
            radius_second = -b * cosine
            return (
                radius * cosine,
                side * radius * sine,
                radius_first * cosine - radius * sine,
                side * (radius_first * sine + radius * cosine),
                radius_second * cosine - 2.0 * radius_first * sine - radius * cosine,
                side * (radius_second * sine + 2.0 * radius_first * cosine - radius * sine),
            )

        raw_start = raw(phase)
        current = raw(phase + sweep * u)
        cosine = math.cos(rotation)
        sine = math.sin(rotation)

        def transform(x_value: float, y_value: float) -> tuple[float, float]:
            return (
                scale * (cosine * x_value - sine * y_value),
                scale * (sine * x_value + cosine * y_value),
            )

        offset_x, offset_y = transform(
            current[0] - raw_start[0], current[1] - raw_start[1]
        )
        first_x, first_y = transform(sweep * current[2], sweep * current[3])
        second_x, second_y = transform(
            sweep**2 * current[4], sweep**2 * current[5]
        )
        return x0 + offset_x, y0 + offset_y, first_x, first_y, second_x, second_y

    raise KeyError(f"Unsupported curve key for numerical evaluation: {curve.key!r}")


def _finite_difference_weights(nodes: Sequence[float], order: int) -> list[float]:
    """Fornberg weights at x=0 for arbitrary relative sample nodes."""

    count = len(nodes)
    if order >= count:
        raise ValueError("Derivative order must be smaller than the stencil.")
    coefficients = [[0.0] * (order + 1) for _ in range(count)]
    coefficients[0][0] = 1.0
    c1 = 1.0
    c4 = nodes[0]
    for index in range(1, count):
        maximum_order = min(index, order)
        c2 = 1.0
        c5 = c4
        c4 = nodes[index]
        for previous in range(index):
            c3 = nodes[index] - nodes[previous]
            if abs(c3) <= 1e-18:
                raise ValueError("Finite-difference nodes must be distinct.")
            c2 *= c3
            if previous == index - 1:
                for derivative in range(maximum_order, 0, -1):
                    coefficients[index][derivative] = c1 * (
                        derivative * coefficients[index - 1][derivative - 1]
                        - c5 * coefficients[index - 1][derivative]
                    ) / c2
                coefficients[index][0] = (
                    -c1 * c5 * coefficients[index - 1][0] / c2
                )
            for derivative in range(maximum_order, 0, -1):
                coefficients[previous][derivative] = (
                    c4 * coefficients[previous][derivative]
                    - derivative * coefficients[previous][derivative - 1]
                ) / c3
            coefficients[previous][0] = (
                c4 * coefficients[previous][0] / c3
            )
        c1 = c2
    return [row[order] for row in coefficients]


def numerical_derivative(
    values: Sequence[float],
    dt: float,
    order: int,
    *,
    stencil: int = 7,
) -> list[float]:
    """Differentiate sampled values with centered/one-sided local stencils."""

    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if order < 1:
        raise ValueError("order must be positive.")
    if len(values) <= order:
        raise ValueError("Not enough samples for the requested derivative.")
    width = min(stencil, len(values))
    if width <= order:
        width = order + 1
    stencils = _cached_difference_stencils(len(values), dt, order, width)
    return [
        sum(weight * values[index] for weight, index in zip(weights, indices))
        for indices, weights in stencils
    ]


@lru_cache(maxsize=64)
def _cached_difference_stencils(
    count: int,
    dt: float,
    order: int,
    width: int,
) -> tuple[tuple[tuple[int, ...], tuple[float, ...]], ...]:
    half = width // 2
    result: list[tuple[tuple[int, ...], tuple[float, ...]]] = []
    for index in range(count):
        start = max(0, min(index - half, count - width))
        indices = tuple(range(start, start + width))
        nodes = [(sample_index - index) * dt for sample_index in indices]
        weights = tuple(_finite_difference_weights(nodes, order))
        result.append((indices, weights))
    return tuple(result)


def _variable_trapezoid(values: Sequence[float], coordinates: Sequence[float]) -> float:
    if len(values) != len(coordinates):
        raise ValueError("values and coordinates must have the same length.")
    return sum(
        0.5 * (left_value + right_value) * (right_x - left_x)
        for left_value, right_value, left_x, right_x in zip(
            values, values[1:], coordinates, coordinates[1:]
        )
    )


def _uniform_trapezoid(values: Sequence[float], dt: float) -> float:
    return _variable_trapezoid(values, [index * dt for index in range(len(values))])


def _curve_coordinate_expressions(
    curve: CurvePath,
    *,
    precision: int,
) -> tuple[str, str]:
    """Return reproducible x(u), y(u) expressions for every built-in curve."""

    x0, y0 = curve.start
    x1, y1 = curve.end
    p = curve.parameters
    f = lambda value: _fmt(float(value), precision)
    chord_length = math.hypot(x1 - x0, y1 - y0)
    arc_parameter_names = {
        "center_x",
        "center_y",
        "radius",
        "theta_start",
        "theta_delta",
    }
    arc_is_straight_limit = curve.key == "arc" and (
        not arc_parameter_names.issubset(p)
        or abs(float(p.get("bulge", 0.0))) * chord_length < 1e-8
    )
    if curve.key == "line" or arc_is_straight_limit:
        return (
            f"(1-u)*({f(x0)}) + u*({f(x1)})",
            f"(1-u)*({f(y0)}) + u*({f(y1)})",
        )
    if curve.key == "arc":
        cx, cy = float(p["center_x"]), float(p["center_y"])
        radius = float(p["radius"])
        theta0, delta = float(p["theta_start"]), float(p["theta_delta"])
        angle = f"({f(theta0)} + ({f(delta)})*u)"
        return (
            f"{f(cx)} + ({f(radius)})*cos{angle}",
            f"{f(cy)} + ({f(radius)})*sin{angle}",
        )
    if curve.key == "bezier":
        c1x, c1y = float(p["control1_x"]), float(p["control1_y"])
        c2x, c2y = float(p["control2_x"]), float(p["control2_y"])
        return (
            f"(1-u)^3*({f(x0)}) + 3*(1-u)^2*u*({f(c1x)}) + "
            f"3*(1-u)*u^2*({f(c2x)}) + u^3*({f(x1)})",
            f"(1-u)^3*({f(y0)}) + 3*(1-u)^2*u*({f(c1y)}) + "
            f"3*(1-u)*u^2*({f(c2y)}) + u^3*({f(y1)})",
        )
    if curve.key == "hermite":
        m0x, m0y = float(p["tangent0_x"]), float(p["tangent0_y"])
        m1x, m1y = float(p["tangent1_x"]), float(p["tangent1_y"])
        h00, h10 = "(2*u^3-3*u^2+1)", "(u^3-2*u^2+u)"
        h01, h11 = "(-2*u^3+3*u^2)", "(u^3-u^2)"
        return (
            f"{h00}*({f(x0)}) + {h10}*({f(m0x)}) + {h01}*({f(x1)}) + {h11}*({f(m1x)})",
            f"{h00}*({f(y0)}) + {h10}*({f(m0y)}) + {h01}*({f(y1)}) + {h11}*({f(m1y)})",
        )
    if curve.key == "cycloid":
        dx, dy = x1 - x0, y1 - y0
        chord = math.hypot(dx, dy)
        nx, ny = -dy / chord, dx / chord
        height = float(p["height"])
        forward = "((2*pi*u-sin(2*pi*u))/(2*pi))"
        transverse = "((1-cos(2*pi*u))/2)"
        return (
            f"{f(x0)} + ({f(dx)})*{forward} + ({f(nx * height)})*{transverse}",
            f"{f(y0)} + ({f(dy)})*{forward} + ({f(ny * height)})*{transverse}",
        )
    if curve.key == "limacon":
        a, b = float(p["a"]), float(p["b"])
        phase, sweep = float(p["phase"]), float(p["sweep"])
        side = float(p["side"])
        scale, rotation = float(p["scale"]), float(p["rotation"])
        raw_radius0 = a + b * math.cos(phase)
        raw_x0 = raw_radius0 * math.cos(phase)
        raw_y0 = side * raw_radius0 * math.sin(phase)
        phi = f"({f(phase)} + ({f(sweep)})*u)"
        radius = f"({f(a)} + ({f(b)})*cos{phi})"
        raw_x = f"({radius}*cos{phi} - ({f(raw_x0)}))"
        raw_y = f"(({f(side)})*{radius}*sin{phi} - ({f(raw_y0)}))"
        cosine, sine = math.cos(rotation), math.sin(rotation)
        transformed_x = f"(({f(cosine)})*{raw_x} - ({f(sine)})*{raw_y})"
        transformed_y = f"(({f(sine)})*{raw_x} + ({f(cosine)})*{raw_y})"
        return (
            f"{f(x0)} + ({f(scale)})*{transformed_x}",
            f"{f(y0)} + ({f(scale)})*{transformed_y}",
        )
    raise KeyError(f"Unsupported curve key for equation export: {curve.key!r}")


def trajectory_equations(
    arm: PlanarArm2R,
    curve: CurvePath,
    elbow_mode: str,
    *,
    duration_s: float = 3.0,
    precision: int = 12,
) -> dict[str, str]:
    """Build explicit, reproducible curve/time/IK composition strings."""

    if elbow_mode not in {"up", "down"}:
        raise ValueError("elbow_mode must be 'up' or 'down'.")
    x_u, y_u = _curve_coordinate_expressions(curve, precision=precision)
    f = lambda value: _fmt(float(value), precision)
    progress = (
        f"q(t)=10*(t/{f(duration_s)})^3-15*(t/{f(duration_s)})^4+"
        f"6*(t/{f(duration_s)})^5, 0<=t<={f(duration_s)}"
    )
    x_t = f"x(t)=x_u(q(t)), where x_u(u)={x_u}"
    y_t = f"y(t)=y_u(q(t)), where y_u(u)={y_u}"
    relative_x = f"(x(t)-({f(arm.base[0])}))"
    relative_y = f"(y(t)-({f(arm.base[1])}))"
    cosine = (
        f"c(t)=({relative_x}^2+{relative_y}^2-({f(arm.link1)})^2-"
        f"({f(arm.link2)})^2)/(2*({f(arm.link1)})*({f(arm.link2)}))"
    )
    sign = "-" if elbow_mode == "up" else "+"
    theta2 = f"theta2(t)={sign}acos(c(t))"
    theta1 = (
        f"theta1(t)=atan2({relative_y},{relative_x})-atan2("
        f"({f(arm.link2)})*sin(theta2(t)),({f(arm.link1)})+"
        f"({f(arm.link2)})*cos(theta2(t)))"
    )
    descriptor = {
        "equation_format_version": "phase4-equations-v1",
        "curve": {
            "key": curve.key,
            "start": list(curve.start),
            "end": list(curve.end),
            "parameters": dict(curve.parameters),
            "x_u": x_u,
            "y_u": y_u,
        },
        "time_law": {
            "duration_s": duration_s,
            "normalized_time": "s=t/T",
            "progress": "q=10*s^3-15*s^4+6*s^5",
        },
        "inverse_kinematics": {
            "base": list(arm.base),
            "link1": arm.link1,
            "link2": arm.link2,
            "elbow_mode": elbow_mode,
            "principal_branches": "acos in [0,pi], atan2 in (-pi,pi]",
            "unwrap": "sampled theta1 and theta2 are unwrapped by +/-2*pi for temporal continuity",
        },
    }
    return {
        "equation_format_version": "phase4-equations-v1",
        "curve_equation_template": curve.equation,
        "curve_equation": f"p(u)=({x_u}, {y_u}), 0<=u<=1",
        "progress_equation": progress,
        "x_u_equation": f"x_u(u)={x_u}",
        "y_u_equation": f"y_u(u)={y_u}",
        "x_t_equation": x_t,
        "y_t_equation": y_t,
        "ik_cosine_equation": cosine,
        "theta1_t_equation": theta1,
        "theta2_t_equation": theta2,
        "branch_policy": "principal acos/atan2 values followed by temporal angle unwrapping",
        "unwrap_policy": "successive jumps above pi are shifted by integer multiples of 2*pi",
        "equation_descriptor_json": json.dumps(
            descriptor, ensure_ascii=False, sort_keys=True
        ),
    }


def _apply_joint_derivatives(
    samples: list[dict[str, object]],
    solutions: Sequence[JointSolution],
    dt: float,
    stencil: int,
) -> tuple[list[float], list[float], list[float]]:
    theta1 = [item.theta1 for item in solutions]
    theta2 = [item.theta2 for item in solutions]
    velocity1 = numerical_derivative(theta1, dt, 1, stencil=stencil)
    velocity2 = numerical_derivative(theta2, dt, 1, stencil=stencil)
    acceleration1 = numerical_derivative(theta1, dt, 2, stencil=stencil)
    acceleration2 = numerical_derivative(theta2, dt, 2, stencil=stencil)
    jerk1 = numerical_derivative(theta1, dt, 3, stencil=stencil)
    jerk2 = numerical_derivative(theta2, dt, 3, stencil=stencil)
    velocity_squared: list[float] = []
    acceleration_squared: list[float] = []
    jerk_squared: list[float] = []
    for row, solution, q1d, q2d, q1dd, q2dd, q1ddd, q2ddd in zip(
        samples,
        solutions,
        velocity1,
        velocity2,
        acceleration1,
        acceleration2,
        jerk1,
        jerk2,
    ):
        v2 = q1d**2 + q2d**2
        a2 = q1dd**2 + q2dd**2
        j2 = q1ddd**2 + q2ddd**2
        row.update(
            {
                "theta1_rad": solution.theta1,
                "theta2_rad": solution.theta2,
                "theta1_deg": math.degrees(solution.theta1),
                "theta2_deg": math.degrees(solution.theta2),
                "theta1_velocity_rad_s": q1d,
                "theta2_velocity_rad_s": q2d,
                "theta1_acceleration_rad_s2": q1dd,
                "theta2_acceleration_rad_s2": q2dd,
                "theta1_jerk_rad_s3": q1ddd,
                "theta2_jerk_rad_s3": q2ddd,
                "joint_velocity_squared_rad2_s2": v2,
                "joint_acceleration_squared_rad2_s4": a2,
                "joint_jerk_squared_rad2_s6": j2,
            }
        )
        velocity_squared.append(v2)
        acceleration_squared.append(a2)
        jerk_squared.append(j2)
    return velocity_squared, acceleration_squared, jerk_squared


def simulate_trajectory(
    arm: PlanarArm2R,
    curve: CurvePath,
    elbow_mode: str,
    *,
    obstacles: Iterable[CircleObstacle] = (),
    duration_s: float = 3.0,
    samples: int | None = None,
    config: MathematicalEvaluationConfig | None = None,
) -> MathematicalTrajectoryResult:
    """Sample and evaluate one selected curve with mathematics-first metrics.

    The public result is intentionally presentation-neutral: a GUI can animate
    ``sample_rows`` while the data pipeline can add scenario/run identifiers to
    those same rows before writing them.
    """

    selected = config or MathematicalEvaluationConfig(duration_s=duration_s)
    if config is not None and not math.isclose(duration_s, 3.0) and not math.isclose(
        duration_s, config.duration_s
    ):
        raise ValueError("Pass duration_s or config.duration_s, not conflicting values.")
    duration = selected.duration_s
    sample_count = len(curve.points) if samples is None else samples
    if sample_count < 7:
        raise ValueError("At least seven samples are required for jerk evaluation.")
    if elbow_mode not in {"up", "down"}:
        raise ValueError("elbow_mode must be 'up' or 'down'.")

    obstacle_list = tuple(obstacles)
    dt = duration / (sample_count - 1)
    normalized_times = [index / (sample_count - 1) for index in range(sample_count)]
    times = [duration * value for value in normalized_times]
    progress_values = [minimum_jerk_progress(value) for value in normalized_times]
    # A cosine-clustered grid resolves the narrow, finite K_epsilon boundary
    # layer at cycloid endpoint cusps far better than uniform-u sampling while
    # retaining exactly ``sample_count`` auditable geometry rows.
    geometry_u_values = [
        0.5 * (1.0 - math.cos(math.pi * value)) for value in normalized_times
    ]
    chord_scale = math.hypot(curve.end[0] - curve.start[0], curve.end[1] - curve.start[1])
    curvature_speed_epsilon = selected.curvature_speed_epsilon_ratio * max(
        chord_scale, 1e-12
    )

    rows: list[dict[str, object]] = []
    timed_points: list[tuple[float, float]] = []
    geometry_points: list[tuple[float, float]] = []
    validation_points: list[tuple[float, float]] = []
    speeds_du: list[float] = []
    curvature_values: list[float | None] = []
    regularized_curvature_integrands: list[float] = []
    for index, (time_s, normalized_time, progress_u, geometry_u) in enumerate(
        zip(times, normalized_times, progress_values, geometry_u_values)
    ):
        x, y, _, _, _, _ = _curve_value_derivatives(curve, progress_u)
        geometry_x, geometry_y, dx, dy, ddx, ddy = _curve_value_derivatives(
            curve, geometry_u
        )
        validation_u = normalized_time
        validation_x, validation_y, _, _, _, _ = _curve_value_derivatives(
            curve, validation_u
        )
        speed = math.hypot(dx, dy)
        cross = dx * ddy - dy * ddx
        kappa = (
            None
            if speed <= curvature_speed_epsilon
            else abs(cross) / speed**3
        )
        regularized_integrand = cross**2 / (
            speed**2 + curvature_speed_epsilon**2
        ) ** 2.5
        point = (x, y)
        timed_points.append(point)
        geometry_points.append((geometry_x, geometry_y))
        validation_points.append((validation_x, validation_y))
        speeds_du.append(speed)
        curvature_values.append(kappa)
        regularized_curvature_integrands.append(regularized_integrand)
        rows.append(
            {
                "sample_index": index,
                "time_s": time_s,
                "normalized_time": normalized_time,
                "progress_u": progress_u,
                "x": x,
                "y": y,
                "geometry_u": geometry_u,
                "geometry_x": geometry_x,
                "geometry_y": geometry_y,
                "validation_u": validation_u,
                "validation_x": validation_x,
                "validation_y": validation_y,
                "dx_du": dx,
                "dy_du": dy,
                "d2x_du2": ddx,
                "d2y_du2": ddy,
                "path_speed_units_per_u": speed,
                "curvature_inv_unit": kappa,
                "regularized_curvature_integrand_inv_unit": regularized_integrand,
                "curvature_speed_epsilon_units_per_u": curvature_speed_epsilon,
                "theta1_rad": None,
                "theta2_rad": None,
                "theta1_deg": None,
                "theta2_deg": None,
                "theta1_velocity_rad_s": None,
                "theta2_velocity_rad_s": None,
                "theta1_acceleration_rad_s2": None,
                "theta2_acceleration_rad_s2": None,
                "theta1_jerk_rad_s3": None,
                "theta2_jerk_rad_s3": None,
                "joint_velocity_squared_rad2_s2": None,
                "joint_acceleration_squared_rad2_s4": None,
                "joint_jerk_squared_rad2_s6": None,
                "obstacle_clearance": None,
                "obstacle_penalty_integrand_per_unit2": 0.0 if not obstacle_list else None,
                "collision_at_sample": False,
                "reachable_at_sample": arm.is_reachable(point),
                "validation_obstacle_clearance": None,
                "validation_collision_at_sample": False,
                "validation_reachable_at_sample": arm.is_reachable(
                    (validation_x, validation_y)
                ),
            }
        )

    path_length = _variable_trapezoid(speeds_du, geometry_u_values)
    path_regular = (
        curve.key != "cycloid"
        and min(speeds_du, default=0.0) > curvature_speed_epsilon
    )
    for row in rows:
        # Repeating this run-level flag in the sample export keeps standalone
        # partitions self-describing and prevents a cusp path from being
        # mistakenly assigned a finite ordinary integral during reintegration.
        row["path_regular"] = path_regular
    unregularized_curvature_energy = (
        _variable_trapezoid(
            [
                float(kappa) ** 2 * speed
                for kappa, speed in zip(curvature_values, speeds_du)
            ],
            geometry_u_values,
        )
        if path_regular
        else None
    )
    regularized_curvature_energy = _variable_trapezoid(
        regularized_curvature_integrands, geometry_u_values
    )
    reachable_count = sum(bool(row["reachable_at_sample"]) for row in rows)
    validation_reachable_count = sum(
        bool(row["validation_reachable_at_sample"]) for row in rows
    )
    time_fully_reachable = reachable_count == sample_count
    validation_fully_reachable = validation_reachable_count == sample_count
    fully_reachable = time_fully_reachable and validation_fully_reachable
    equations = trajectory_equations(
        arm,
        curve,
        elbow_mode,
        duration_s=duration,
        precision=selected.equation_precision,
    )

    joint_velocity_effort: float | None = None
    joint_acceleration_effort: float | None = None
    joint_jerk_effort: float | None = None
    max_forward_error: float | None = None
    min_singularity_index: float | None = None
    minimum_clearance: float | None = None
    obstacle_penalty: float | None = 0.0 if not obstacle_list else None
    time_collision_count = 0
    validation_collision_count = 0
    collision_free: bool | None = True if fully_reachable else None
    warning_parts: list[str] = []

    if time_fully_reachable:
        solutions = arm.inverse_path(timed_points, elbow_mode)  # type: ignore[arg-type]
        min_singularity_index = min(
            (abs(math.sin(solution.theta2)) for solution in solutions),
            default=0.0,
        )
        velocity_squared, acceleration_squared, jerk_squared = _apply_joint_derivatives(
            rows, solutions, dt, selected.derivative_stencil
        )
        joint_velocity_effort = _uniform_trapezoid(velocity_squared, dt)
        joint_acceleration_effort = _uniform_trapezoid(acceleration_squared, dt)
        joint_jerk_effort = _uniform_trapezoid(jerk_squared, dt)
        forward_errors = [
            math.hypot(
                arm.forward(solution.theta1, solution.theta2).end[0] - point[0],
                arm.forward(solution.theta1, solution.theta2).end[1] - point[1],
            )
            for solution, point in zip(solutions, timed_points)
        ]
        max_forward_error = max(forward_errors, default=0.0)

        if obstacle_list:
            clearances = [
                float(
                    pose_obstacle_clearance(
                        arm,
                        solution,
                        obstacle_list,
                        link_radius_units=selected.link_radius_units,
                    )
                )
                for solution in solutions
            ]
            penalty_integrand = [
                1.0 / max(clearance, selected.clearance_epsilon_units) ** 2
                for clearance in clearances
            ]
            obstacle_penalty = _uniform_trapezoid(penalty_integrand, dt)
            minimum_clearance = min(clearances)
            time_collision_count = sum(clearance <= 0.0 for clearance in clearances)
            for row, clearance, penalty in zip(rows, clearances, penalty_integrand):
                row["obstacle_clearance"] = clearance
                row["obstacle_penalty_integrand_per_unit2"] = penalty
                row["collision_at_sample"] = clearance <= 0.0

            if validation_fully_reachable:
                validation_solutions = arm.inverse_path(
                    validation_points, elbow_mode  # type: ignore[arg-type]
                )
                validation_clearances = [
                    float(
                        pose_obstacle_clearance(
                            arm,
                            solution,
                            obstacle_list,
                            link_radius_units=selected.link_radius_units,
                        )
                    )
                    for solution in validation_solutions
                ]
                validation_collision_count = sum(
                    clearance <= 0.0 for clearance in validation_clearances
                )
                minimum_clearance = min(minimum_clearance, min(validation_clearances))
                for row, clearance in zip(rows, validation_clearances):
                    row["validation_obstacle_clearance"] = clearance
                    row["validation_collision_at_sample"] = clearance <= 0.0

            if fully_reachable:
                collision_free = (
                    time_collision_count + validation_collision_count == 0
                )
    else:
        # Keep individually reachable angles for visual diagnosis, but do not
        # differentiate a trajectory with missing samples.
        for row, point in zip(rows, timed_points):
            if not bool(row["reachable_at_sample"]):
                continue
            solution = arm.inverse(point, elbow_mode)  # type: ignore[arg-type]
            row["theta1_rad"] = solution.theta1
            row["theta2_rad"] = solution.theta2
            row["theta1_deg"] = math.degrees(solution.theta1)
            row["theta2_deg"] = math.degrees(solution.theta2)

    if not time_fully_reachable:
        warning_parts.append("unreachable_time_grid")
    if not validation_fully_reachable:
        warning_parts.append("unreachable_uniform_u_validation_grid")
    if collision_free is False:
        warning_parts.append("collision")
        if time_collision_count == 0 and validation_collision_count > 0:
            warning_parts.append("collision_detected_on_uniform_u_validation_grid_only")
    if (
        min_singularity_index is not None
        and min_singularity_index < selected.singularity_warning_threshold
    ):
        warning_parts.append("near_kinematic_singularity_derivatives_sensitive")
    if not path_regular:
        warning_parts.append("non_regular_curve_regularized_curvature")
    selection_valid = fully_reachable and collision_free is True
    collision_count = time_collision_count + validation_collision_count
    metrics: dict[str, object] = {
        "duration_s": duration,
        "samples": sample_count,
        "path_length_units": path_length,
        "path_length_polyline_units": sum(
            math.hypot(right[0] - left[0], right[1] - left[1])
            for left, right in zip(geometry_points, geometry_points[1:])
        ),
        "joint_velocity_effort": joint_velocity_effort,
        "joint_acceleration_effort": joint_acceleration_effort,
        "joint_jerk_effort": joint_jerk_effort,
        # The regularized value is the Phase-4 selection objective.  The alias
        # is retained for simple GUI display, while the explicit field and
        # metadata prevent it being mistaken for an improper cusp integral.
        "curvature_energy_inv_unit": regularized_curvature_energy,
        "regularized_curvature_energy_inv_unit": regularized_curvature_energy,
        "unregularized_curvature_energy_inv_unit": unregularized_curvature_energy,
        "curvature_speed_epsilon_units_per_u": curvature_speed_epsilon,
        "min_geometric_speed_units_per_u": min(speeds_du, default=0.0),
        "path_regular": path_regular,
        "obstacle_penalty_s_per_unit2": obstacle_penalty,
        "fully_reachable": fully_reachable,
        "time_grid_fully_reachable": time_fully_reachable,
        "validation_grid_fully_reachable": validation_fully_reachable,
        "reachable_samples": reachable_count,
        "unreachable_samples": sample_count - reachable_count,
        "reachable_ratio": reachable_count / sample_count,
        "validation_reachable_samples": validation_reachable_count,
        "validation_unreachable_samples": sample_count - validation_reachable_count,
        "validation_reachable_ratio": validation_reachable_count / sample_count,
        "hard_validation_grid": "minimum-jerk time grid plus uniform-u validation grid",
        "collision_free": collision_free,
        "collision_samples": collision_count if fully_reachable else None,
        "time_collision_samples": time_collision_count if time_fully_reachable else None,
        "validation_collision_samples": (
            validation_collision_count if fully_reachable else None
        ),
        "collision_ratio": (
            collision_count / (2 * sample_count) if fully_reachable else None
        ),
        "min_obstacle_clearance_units": minimum_clearance,
        "max_forward_kinematics_error_units": max_forward_error,
        "min_singularity_index": min_singularity_index,
        "joint_derivative_reliable": (
            None
            if min_singularity_index is None
            else min_singularity_index >= selected.singularity_warning_threshold
        ),
        "selection_valid": selection_valid,
        "integration_method": (
            "composite trapezoid; exact curve derivatives on cosine-clustered-u "
            "geometry grid, joints on uniform-t grid"
        ),
        "curvature_definition": (
            "K_epsilon=integral cross(p',p'')^2/"
            "(||p'||^2+epsilon^2)^(5/2) du"
        ),
        "joint_derivative_method": (
            f"local polynomial finite differences, stencil={selected.derivative_stencil}"
        ),
        "time_law": "minimum jerk q(s)=10s^3-15s^4+6s^5",
        "warning": ";".join(warning_parts),
        "curve_parameters_json": json.dumps(
            dict(curve.parameters), ensure_ascii=False, sort_keys=True
        ),
        **equations,
    }
    return MathematicalTrajectoryResult(tuple(rows), metrics, equations)


def reintegrate_sample_objectives(
    sample_rows: Sequence[Mapping[str, object]],
) -> dict[str, float | None]:
    """Recompute Phase-4 objectives directly from exported sample rows."""

    if len(sample_rows) < 2:
        raise ValueError("At least two sample rows are required.")
    geometry_u = [float(row["geometry_u"]) for row in sample_rows]
    time = [float(row["time_s"]) for row in sample_rows]
    speed = [float(row["path_speed_units_per_u"]) for row in sample_rows]
    regularized_curvature_integrand = [
        float(row["regularized_curvature_integrand_inv_unit"])
        for row in sample_rows
    ]
    result: dict[str, float | None] = {
        "path_length_units": _variable_trapezoid(speed, geometry_u),
        "regularized_curvature_energy_inv_unit": _variable_trapezoid(
            regularized_curvature_integrand, geometry_u
        ),
    }
    result["curvature_energy_inv_unit"] = result[
        "regularized_curvature_energy_inv_unit"
    ]
    curvature = [row.get("curvature_inv_unit") for row in sample_rows]
    regularity = [row.get("path_regular") for row in sample_rows]
    result["unregularized_curvature_energy_inv_unit"] = (
        None
        if any(value is None for value in curvature)
        or any(_value is not True for _value in regularity)
        else _variable_trapezoid(
            [
                float(kappa) ** 2 * path_speed
                for kappa, path_speed in zip(curvature, speed)
            ],
            geometry_u,
        )
    )
    mappings = {
        "joint_velocity_effort": "joint_velocity_squared_rad2_s2",
        "joint_acceleration_effort": "joint_acceleration_squared_rad2_s4",
        "joint_jerk_effort": "joint_jerk_squared_rad2_s6",
        "obstacle_penalty_s_per_unit2": "obstacle_penalty_integrand_per_unit2",
    }
    for objective, field in mappings.items():
        values = [row.get(field) for row in sample_rows]
        result[objective] = (
            None
            if any(value is None for value in values)
            else _variable_trapezoid([float(value) for value in values], time)
        )
    return result
