"""Endpoint-normalized parametric curve families for trajectory exploration."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin, sqrt
from typing import Callable, Mapping


Point = tuple[float, float]


def _add(a: Point, b: Point) -> Point:
    return a[0] + b[0], a[1] + b[1]


def _sub(a: Point, b: Point) -> Point:
    return a[0] - b[0], a[1] - b[1]


def _scale(value: float, point: Point) -> Point:
    return value * point[0], value * point[1]


def _distance(a: Point, b: Point) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _sample(function: Callable[[float], Point], samples: int) -> tuple[Point, ...]:
    if samples < 2:
        raise ValueError("samples must be at least 2.")
    return tuple(function(index / (samples - 1)) for index in range(samples))


def _validate_endpoints(start: Point, end: Point) -> tuple[Point, float, Point]:
    chord = _sub(end, start)
    length = hypot(*chord)
    if length <= 1e-12:
        raise ValueError("Start and end points must be different.")
    unit = chord[0] / length, chord[1] / length
    normal = -unit[1], unit[0]
    return unit, length, normal


@dataclass(frozen=True, slots=True)
class CurvePath:
    """A sampled path plus the equation and parameters that generated it."""

    key: str
    display_name: str
    points: tuple[Point, ...]
    equation: str
    parameters: Mapping[str, float | str]

    @property
    def start(self) -> Point:
        return self.points[0]

    @property
    def end(self) -> Point:
        return self.points[-1]

    @property
    def length(self) -> float:
        return sum(_distance(a, b) for a, b in zip(self.points, self.points[1:]))


def line(start: Point, end: Point, samples: int = 161) -> CurvePath:
    _validate_endpoints(start, end)

    def point(t: float) -> Point:
        return (
            (1.0 - t) * start[0] + t * end[0],
            (1.0 - t) * start[1] + t * end[1],
        )

    return CurvePath(
        key="line",
        display_name="직선",
        points=_sample(point, samples),
        equation="p(t) = (1-t)P₀ + tP₁,   0 ≤ t ≤ 1",
        parameters={},
    )


def circular_arc(
    start: Point,
    end: Point,
    *,
    bulge: float = 0.25,
    samples: int = 161,
) -> CurvePath:
    _, chord_length, normal = _validate_endpoints(start, end)
    height = bulge * chord_length
    if abs(height) < 1e-8:
        straight = line(start, end, samples)
        return CurvePath(
            key="arc",
            display_name="원호 (직선 한계)",
            points=straight.points,
            equation="bulge = 0 이므로 p(t) = (1-t)P₀ + tP₁",
            parameters={"bulge": bulge},
        )

    midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
    arc_midpoint = _add(midpoint, _scale(height, normal))
    center_offset = (height * height - (chord_length / 2.0) ** 2) / (2.0 * height)
    center = _add(midpoint, _scale(center_offset, normal))
    radius = _distance(center, start)
    start_angle = atan2(start[1] - center[1], start[0] - center[0])
    end_angle = atan2(end[1] - center[1], end[0] - center[0])

    ccw_delta = (end_angle - start_angle) % (2.0 * pi)
    clockwise_delta = ccw_delta - 2.0 * pi

    def candidate_midpoint(delta: float) -> Point:
        angle = start_angle + 0.5 * delta
        return center[0] + radius * cos(angle), center[1] + radius * sin(angle)

    delta = min(
        (ccw_delta, clockwise_delta),
        key=lambda value: _distance(candidate_midpoint(value), arc_midpoint),
    )

    def point(t: float) -> Point:
        angle = start_angle + delta * t
        return center[0] + radius * cos(angle), center[1] + radius * sin(angle)

    return CurvePath(
        key="arc",
        display_name="원호",
        points=_sample(point, samples),
        equation=(
            "p(t) = C + R(cos(θ₀ + Δθt), sin(θ₀ + Δθt)),   0 ≤ t ≤ 1"
        ),
        parameters={
            "bulge": bulge,
            "center_x": center[0],
            "center_y": center[1],
            "radius": radius,
            "theta_start": start_angle,
            "theta_delta": delta,
        },
    )


def cubic_bezier(
    start: Point,
    end: Point,
    *,
    control1_offset: float = 0.2,
    control2_offset: float = 0.2,
    samples: int = 161,
) -> CurvePath:
    unit, length, normal = _validate_endpoints(start, end)
    control1 = _add(
        _add(start, _scale(length / 3.0, unit)),
        _scale(control1_offset * length, normal),
    )
    control2 = _add(
        _add(start, _scale(2.0 * length / 3.0, unit)),
        _scale(control2_offset * length, normal),
    )

    def point(t: float) -> Point:
        u = 1.0 - t
        weights = (u**3, 3.0 * u * u * t, 3.0 * u * t * t, t**3)
        controls = (start, control1, control2, end)
        return (
            sum(weight * control[0] for weight, control in zip(weights, controls)),
            sum(weight * control[1] for weight, control in zip(weights, controls)),
        )

    return CurvePath(
        key="bezier",
        display_name="3차 베지어",
        points=_sample(point, samples),
        equation=(
            "p(t) = (1-t)³P₀ + 3(1-t)²tC₁ + 3(1-t)t²C₂ + t³P₁"
        ),
        parameters={
            "control1_x": control1[0],
            "control1_y": control1[1],
            "control2_x": control2[0],
            "control2_y": control2[1],
        },
    )


def cubic_hermite(
    start: Point,
    end: Point,
    *,
    tangent_scale: float = 1.0,
    angle_degrees: float = 30.0,
    samples: int = 161,
) -> CurvePath:
    unit, length, _ = _validate_endpoints(start, end)
    base_angle = atan2(unit[1], unit[0])
    angle = angle_degrees * pi / 180.0
    magnitude = tangent_scale * length
    tangent0 = magnitude * cos(base_angle + angle), magnitude * sin(base_angle + angle)
    tangent1 = magnitude * cos(base_angle - angle), magnitude * sin(base_angle - angle)

    def point(t: float) -> Point:
        h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
        h10 = t**3 - 2.0 * t**2 + t
        h01 = -2.0 * t**3 + 3.0 * t**2
        h11 = t**3 - t**2
        return (
            h00 * start[0] + h10 * tangent0[0] + h01 * end[0] + h11 * tangent1[0],
            h00 * start[1] + h10 * tangent0[1] + h01 * end[1] + h11 * tangent1[1],
        )

    return CurvePath(
        key="hermite",
        display_name="3차 에르미트",
        points=_sample(point, samples),
        equation="p(t) = h₀₀P₀ + h₁₀M₀ + h₀₁P₁ + h₁₁M₁",
        parameters={
            "tangent0_x": tangent0[0],
            "tangent0_y": tangent0[1],
            "tangent1_x": tangent1[0],
            "tangent1_y": tangent1[1],
            "angle_degrees": angle_degrees,
        },
    )


def cycloid_arc(
    start: Point,
    end: Point,
    *,
    height_ratio: float = 0.25,
    samples: int = 161,
) -> CurvePath:
    unit, length, normal = _validate_endpoints(start, end)
    height = height_ratio * length

    def point(t: float) -> Point:
        theta = 2.0 * pi * t
        forward = (theta - sin(theta)) / (2.0 * pi)
        transverse = 0.5 * height * (1.0 - cos(theta))
        return _add(start, _add(_scale(forward * length, unit), _scale(transverse, normal)))

    return CurvePath(
        key="cycloid",
        display_name="사이클로이드 변형",
        points=_sample(point, samples),
        equation=(
            "θ=2πt,  p(t)=P₀ + D(θ-sinθ)/(2π) + n·h(1-cosθ)/2"
        ),
        parameters={"height_ratio": height_ratio, "height": height},
    )


def limacon_arc(
    start: Point,
    end: Point,
    *,
    a: float = 1.0,
    b: float = 0.65,
    phase: float = -0.6 * pi,
    sweep: float = 1.2 * pi,
    side: float = 1.0,
    samples: int = 161,
) -> CurvePath:
    _validate_endpoints(start, end)

    def raw(t: float) -> Point:
        phi = phase + sweep * t
        radius = a + b * cos(phi)
        return radius * cos(phi), side * radius * sin(phi)

    raw_start = raw(0.0)
    raw_end = raw(1.0)
    raw_chord = _sub(raw_end, raw_start)
    target_chord = _sub(end, start)
    raw_length = hypot(*raw_chord)
    target_length = hypot(*target_chord)
    if raw_length <= 1e-10:
        raise ValueError("The selected limacon segment has coincident endpoints.")

    scale = target_length / raw_length
    rotation = atan2(target_chord[1], target_chord[0]) - atan2(raw_chord[1], raw_chord[0])
    cosine = cos(rotation)
    sine = sin(rotation)

    def point(t: float) -> Point:
        qx, qy = _sub(raw(t), raw_start)
        rotated = cosine * qx - sine * qy, sine * qx + cosine * qy
        return _add(start, _scale(scale, rotated))

    return CurvePath(
        key="limacon",
        display_name="리마송 구간",
        points=_sample(point, samples),
        equation=(
            "r(φ)=a+b cosφ,  φ=φ₀+Δφt; 두 끝점을 P₀,P₁에 맞춘 닮음변환"
        ),
        parameters={
            "a": a,
            "b": b,
            "phase": phase,
            "sweep": sweep,
            "side": side,
            "scale": scale,
            "rotation": rotation,
        },
    )


CURVE_LABELS: dict[str, str] = {
    "직선": "line",
    "원호": "arc",
    "3차 베지어": "bezier",
    "3차 에르미트": "hermite",
    "사이클로이드": "cycloid",
    "리마송": "limacon",
}


def make_curve(
    kind: str,
    start: Point,
    end: Point,
    *,
    shape: float = 0.35,
    samples: int = 161,
) -> CurvePath:
    """Create a curve from a compact GUI-friendly shape parameter in [-1, 1]."""

    normalized = CURVE_LABELS.get(kind, kind.lower().strip())
    shape = max(-1.0, min(1.0, shape))
    if normalized == "line":
        return line(start, end, samples)
    if normalized == "arc":
        return circular_arc(start, end, bulge=0.42 * shape, samples=samples)
    if normalized == "bezier":
        return cubic_bezier(
            start,
            end,
            control1_offset=0.42 * shape,
            control2_offset=0.42 * shape,
            samples=samples,
        )
    if normalized == "hermite":
        return cubic_hermite(
            start,
            end,
            angle_degrees=65.0 * shape,
            samples=samples,
        )
    if normalized == "cycloid":
        return cycloid_arc(start, end, height_ratio=0.5 * shape, samples=samples)
    if normalized == "limacon":
        side = -1.0 if shape < 0.0 else 1.0
        b = 0.35 + 0.55 * abs(shape)
        return limacon_arc(start, end, b=b, side=side, samples=samples)
    raise KeyError(f"Unknown curve kind: {kind!r}")

