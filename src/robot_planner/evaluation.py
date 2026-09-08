"""Multi-objective trajectory evaluation for a planar two-link robot arm.

The energy values in this module are model estimates, not measurements.  A
slender-link rigid-body model, a shared minimum-jerk time law, and a simplified
direct-drive motor model are used so that curve families can be compared under
the same stated assumptions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import fmean
from typing import Iterable, Sequence

from .curves import CurvePath
from .kinematics import JointSolution, PlanarArm2R


Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class CircleObstacle:
    obstacle_id: str
    center_x: float
    center_y: float
    radius_units: float

    def __post_init__(self) -> None:
        if self.radius_units <= 0.0:
            raise ValueError("Obstacle radius must be positive.")

    @property
    def center(self) -> Point:
        return self.center_x, self.center_y

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Physical and numerical assumptions shared by every candidate path."""

    duration_s: float = 3.0
    workspace_unit_m: float = 0.1
    link_linear_density_kg_per_m: float = 0.35
    link_radius_m: float = 0.005
    joint_viscous_damping_nms_per_rad: float = 0.03
    motor_torque_constant_nm_per_a: float = 0.12
    motor_resistance_ohm: float = 0.8
    motion_plane: str = "horizontal"
    gravity_m_per_s2: float = 9.80665
    singularity_epsilon: float = 0.05
    clearance_epsilon_m: float = 0.01

    def __post_init__(self) -> None:
        positive_fields = (
            self.duration_s,
            self.workspace_unit_m,
            self.link_linear_density_kg_per_m,
            self.link_radius_m,
            self.motor_torque_constant_nm_per_a,
            self.motor_resistance_ohm,
            self.gravity_m_per_s2,
            self.singularity_epsilon,
            self.clearance_epsilon_m,
        )
        if any(value <= 0.0 for value in positive_fields):
            raise ValueError("Physical parameters and epsilons must be positive.")
        if self.joint_viscous_damping_nms_per_rad < 0.0:
            raise ValueError("Joint damping cannot be negative.")
        if self.motion_plane not in {"horizontal", "vertical"}:
            raise ValueError("motion_plane must be 'horizontal' or 'vertical'.")


@dataclass(frozen=True, slots=True)
class CurvatureMetrics:
    mean_abs_curvature_inv_m: float
    max_curvature_inv_m: float
    curvature_energy_inv_m: float
    curvature_variation_inv_m: float


@dataclass(frozen=True, slots=True)
class TrajectoryEvaluation:
    duration_s: float
    curve_length: float
    curve_length_m: float
    path_to_chord_ratio: float
    fully_reachable: bool
    reachable_samples: int
    reachable_ratio: float
    unreachable_samples: int
    collision_free: bool | None
    collision_samples: int | None
    collision_ratio: float | None
    min_obstacle_clearance_m: float | None
    clearance_penalty_s_per_m2: float | None
    mean_abs_curvature_inv_m: float
    max_curvature_inv_m: float
    curvature_energy_inv_m: float
    curvature_variation_inv_m: float
    joint_motion_l2_rad: float | None
    total_joint_rotation_deg: float | None
    min_singularity_index: float | None
    mean_singularity_index: float | None
    singularity_penalty_s: float | None
    joint_jerk_effort_rad2_per_s5: float | None
    joint_jerk_rms_rad_per_s3: float | None
    joint_jerk_variation_rad_per_s3: float | None
    rms_joint_torque_nm: float | None
    max_joint_torque_nm: float | None
    positive_mechanical_energy_j: float | None
    motor_copper_loss_j: float | None
    model_electrical_energy_j: float | None
    valid_trajectory: bool
    warning: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def minimum_jerk_progress(normalized_time: float) -> float:
    """Return 10s^3-15s^4+6s^5 for s in [0, 1]."""

    if not 0.0 <= normalized_time <= 1.0:
        raise ValueError("normalized_time must lie in [0, 1].")
    s = normalized_time
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


def _distance(left: Point, right: Point) -> float:
    return math.hypot(right[0] - left[0], right[1] - left[1])


def _trapezoid(values: Sequence[float], dt: float) -> float:
    if len(values) < 2:
        return 0.0
    return dt * (0.5 * values[0] + sum(values[1:-1]) + 0.5 * values[-1])


def _derivative(values: Sequence[float], dt: float) -> list[float]:
    if len(values) < 2:
        return [0.0 for _ in values]
    result = [(values[1] - values[0]) / dt]
    result.extend(
        (values[index + 1] - values[index - 1]) / (2.0 * dt)
        for index in range(1, len(values) - 1)
    )
    result.append((values[-1] - values[-2]) / dt)
    return result


def _interpolate_uniform(values: Sequence[float], progress: float) -> float:
    position = progress * (len(values) - 1)
    left = min(int(math.floor(position)), len(values) - 2)
    fraction = position - left
    return (1.0 - fraction) * values[left] + fraction * values[left + 1]


def _natural_cubic_second_derivatives(values: Sequence[float]) -> list[float]:
    """Solve a natural cubic spline on equally spaced parameter samples."""

    count = len(values)
    if count <= 2:
        return [0.0] * count
    interior = count - 2
    step = 1.0 / (count - 1)
    diagonal = [4.0] * interior
    upper = [1.0] * max(interior - 1, 0)
    right = [
        6.0 * (values[index + 1] - 2.0 * values[index] + values[index - 1])
        / (step * step)
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


def _cubic_interpolate_uniform(
    values: Sequence[float],
    second_derivatives: Sequence[float],
    progress: float,
) -> float:
    step = 1.0 / (len(values) - 1)
    position = progress / step
    left = min(int(math.floor(position)), len(values) - 2)
    fraction = position - left
    a = 1.0 - fraction
    b = fraction
    return (
        a * values[left]
        + b * values[left + 1]
        + (
            (a**3 - a) * second_derivatives[left]
            + (b**3 - b) * second_derivatives[left + 1]
        )
        * step**2
        / 6.0
    )


def _resample_with_time_law(
    values: Sequence[float],
    *,
    cubic: bool = False,
) -> list[float]:
    if len(values) < 2:
        return list(values)
    last = len(values) - 1
    second_derivatives = (
        _natural_cubic_second_derivatives(values) if cubic else None
    )
    progress_values = [minimum_jerk_progress(index / last) for index in range(len(values))]
    if second_derivatives is not None:
        return [
            _cubic_interpolate_uniform(values, second_derivatives, progress)
            for progress in progress_values
        ]
    return [
        _interpolate_uniform(values, progress) for progress in progress_values
    ]


def discrete_curvature_metrics(points: Sequence[Point]) -> CurvatureMetrics:
    """Estimate geometry-only curvature from consecutive triples of points."""

    if len(points) < 3:
        raise ValueError("At least three points are required for curvature.")
    curvatures: list[float] = []
    weights: list[float] = []
    for left, middle, right in zip(points, points[1:], points[2:]):
        a = _distance(left, middle)
        b = _distance(middle, right)
        c = _distance(left, right)
        denominator = a * b * c
        if denominator <= 1e-15:
            curvature = 0.0
        else:
            cross = abs(
                (middle[0] - left[0]) * (right[1] - left[1])
                - (middle[1] - left[1]) * (right[0] - left[0])
            )
            curvature = 2.0 * cross / denominator
        curvatures.append(curvature)
        weights.append(0.5 * (a + b))

    total_weight = sum(weights)
    if total_weight <= 1e-15:
        return CurvatureMetrics(0.0, 0.0, 0.0, 0.0)
    mean = sum(value * weight for value, weight in zip(curvatures, weights)) / total_weight
    energy = sum(value * value * weight for value, weight in zip(curvatures, weights))
    variance = sum(
        (value - mean) ** 2 * weight for value, weight in zip(curvatures, weights)
    ) / total_weight
    return CurvatureMetrics(
        mean_abs_curvature_inv_m=mean,
        max_curvature_inv_m=max(curvatures, default=0.0),
        curvature_energy_inv_m=energy,
        curvature_variation_inv_m=math.sqrt(max(0.0, variance)),
    )


def _point_to_segment_distance(point: Point, start: Point, end: Point) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-15:
        return _distance(point, start)
    projection = (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
    ) / length_squared
    projection = max(0.0, min(1.0, projection))
    nearest = start[0] + projection * dx, start[1] + projection * dy
    return _distance(point, nearest)


def pose_obstacle_clearance(
    arm: PlanarArm2R,
    solution: JointSolution,
    obstacles: Iterable[CircleObstacle],
    *,
    link_radius_units: float,
) -> float | None:
    """Return minimum signed link-to-obstacle clearance for one arm pose."""

    obstacle_list = tuple(obstacles)
    if not obstacle_list:
        return None
    pose = arm.forward(solution.theta1, solution.theta2)
    clearances: list[float] = []
    for obstacle in obstacle_list:
        first = _point_to_segment_distance(obstacle.center, pose.base, pose.elbow)
        second = _point_to_segment_distance(obstacle.center, pose.elbow, pose.end)
        clearances.append(
            min(first, second) - obstacle.radius_units - link_radius_units
        )
    return min(clearances)


def _joint_motion_metrics(
    solutions: Sequence[JointSolution],
) -> tuple[float, float]:
    deltas = [
        (
            right.theta1 - left.theta1,
            right.theta2 - left.theta2,
        )
        for left, right in zip(solutions, solutions[1:])
    ]
    motion = sum(math.hypot(delta1, delta2) for delta1, delta2 in deltas)
    total_rotation = math.degrees(
        sum(abs(delta1) + abs(delta2) for delta1, delta2 in deltas)
    )
    return motion, total_rotation


def _dynamic_metrics(
    arm: PlanarArm2R,
    solutions: Sequence[JointSolution],
    config: EvaluationConfig,
) -> dict[str, float]:
    samples = len(solutions)
    if samples < 7:
        raise ValueError("At least seven samples are required for jerk evaluation.")
    dt = config.duration_s / (samples - 1)
    theta1 = _resample_with_time_law(
        [item.theta1 for item in solutions], cubic=True
    )
    theta2 = _resample_with_time_law(
        [item.theta2 for item in solutions], cubic=True
    )
    velocity1 = _derivative(theta1, dt)
    velocity2 = _derivative(theta2, dt)
    acceleration1 = _derivative(velocity1, dt)
    acceleration2 = _derivative(velocity2, dt)
    jerk1 = _derivative(acceleration1, dt)
    jerk2 = _derivative(acceleration2, dt)

    link1_m = config.workspace_unit_m * arm.link1
    link2_m = config.workspace_unit_m * arm.link2
    link1_mass = config.link_linear_density_kg_per_m * link1_m
    link2_mass = config.link_linear_density_kg_per_m * link2_m
    center1 = link1_m / 2.0
    center2 = link2_m / 2.0
    inertia1 = link1_mass * link1_m**2 / 12.0
    inertia2 = link2_mass * link2_m**2 / 12.0
    damping = config.joint_viscous_damping_nms_per_rad

    torque1: list[float] = []
    torque2: list[float] = []
    for q1, q2, q1d, q2d, q1dd, q2dd in zip(
        theta1,
        theta2,
        velocity1,
        velocity2,
        acceleration1,
        acceleration2,
    ):
        cosine2 = math.cos(q2)
        sine2 = math.sin(q2)
        coupling = link2_mass * link1_m * center2
        mass11 = (
            inertia1
            + inertia2
            + link1_mass * center1**2
            + link2_mass
            * (link1_m**2 + center2**2 + 2.0 * link1_m * center2 * cosine2)
        )
        mass12 = inertia2 + link2_mass * (
            center2**2 + link1_m * center2 * cosine2
        )
        mass22 = inertia2 + link2_mass * center2**2
        coriolis1 = -coupling * sine2 * (2.0 * q1d * q2d + q2d**2)
        coriolis2 = coupling * sine2 * q1d**2

        gravity1 = 0.0
        gravity2 = 0.0
        if config.motion_plane == "vertical":
            gravity1 = (
                (link1_mass * center1 + link2_mass * link1_m)
                * config.gravity_m_per_s2
                * math.cos(q1)
                + link2_mass
                * center2
                * config.gravity_m_per_s2
                * math.cos(q1 + q2)
            )
            gravity2 = (
                link2_mass
                * center2
                * config.gravity_m_per_s2
                * math.cos(q1 + q2)
            )

        torque1.append(
            mass11 * q1dd
            + mass12 * q2dd
            + coriolis1
            + gravity1
            + damping * q1d
        )
        torque2.append(
            mass12 * q1dd
            + mass22 * q2dd
            + coriolis2
            + gravity2
            + damping * q2d
        )

    jerk_squared = [a * a + b * b for a, b in zip(jerk1, jerk2)]
    jerk_norms = [math.sqrt(value) for value in jerk_squared]
    jerk_effort = _trapezoid(jerk_squared, dt)
    jerk_mean = fmean(jerk_norms)
    jerk_variation = math.sqrt(
        fmean((value - jerk_mean) ** 2 for value in jerk_norms)
    )

    torque_squared = [a * a + b * b for a, b in zip(torque1, torque2)]
    positive_power = [
        max(tau1 * speed1, 0.0) + max(tau2 * speed2, 0.0)
        for tau1, tau2, speed1, speed2 in zip(
            torque1, torque2, velocity1, velocity2
        )
    ]
    copper_power = [
        config.motor_resistance_ohm
        * value
        / config.motor_torque_constant_nm_per_a**2
        for value in torque_squared
    ]
    positive_mechanical = _trapezoid(positive_power, dt)
    copper_loss = _trapezoid(copper_power, dt)
    return {
        "joint_jerk_effort_rad2_per_s5": jerk_effort,
        "joint_jerk_rms_rad_per_s3": math.sqrt(jerk_effort / config.duration_s),
        "joint_jerk_variation_rad_per_s3": jerk_variation,
        "rms_joint_torque_nm": math.sqrt(_trapezoid(torque_squared, dt) / config.duration_s),
        "max_joint_torque_nm": max(
            (math.sqrt(value) for value in torque_squared), default=0.0
        ),
        "positive_mechanical_energy_j": positive_mechanical,
        "motor_copper_loss_j": copper_loss,
        "model_electrical_energy_j": positive_mechanical + copper_loss,
    }


def evaluate_trajectory(
    arm: PlanarArm2R,
    curve: CurvePath,
    elbow_mode: str,
    *,
    obstacles: Iterable[CircleObstacle] = (),
    config: EvaluationConfig | None = None,
) -> TrajectoryEvaluation:
    """Evaluate one endpoint curve and inverse-kinematics branch."""

    selected_config = config or EvaluationConfig()
    if len(curve.points) < 7:
        raise ValueError("At least seven path samples are required.")
    if elbow_mode not in {"up", "down"}:
        raise ValueError("elbow_mode must be 'up' or 'down'.")

    obstacle_list = tuple(obstacles)
    physical_points = tuple(
        (
            point[0] * selected_config.workspace_unit_m,
            point[1] * selected_config.workspace_unit_m,
        )
        for point in curve.points
    )
    curvature = discrete_curvature_metrics(physical_points)
    chord_length = _distance(curve.start, curve.end)
    reachable_flags = [arm.is_reachable(point) for point in curve.points]
    reachable_samples = sum(reachable_flags)
    fully_reachable = reachable_samples == len(curve.points)

    common: dict[str, object] = {
        "duration_s": selected_config.duration_s,
        "curve_length": curve.length,
        "curve_length_m": curve.length * selected_config.workspace_unit_m,
        "path_to_chord_ratio": curve.length / chord_length,
        "fully_reachable": fully_reachable,
        "reachable_samples": reachable_samples,
        "reachable_ratio": reachable_samples / len(curve.points),
        "unreachable_samples": len(curve.points) - reachable_samples,
        **asdict(curvature),
    }
    if not fully_reachable:
        return TrajectoryEvaluation(
            **common,
            collision_free=None,
            collision_samples=None,
            collision_ratio=None,
            min_obstacle_clearance_m=None,
            clearance_penalty_s_per_m2=None,
            joint_motion_l2_rad=None,
            total_joint_rotation_deg=None,
            min_singularity_index=None,
            mean_singularity_index=None,
            singularity_penalty_s=None,
            joint_jerk_effort_rad2_per_s5=None,
            joint_jerk_rms_rad_per_s3=None,
            joint_jerk_variation_rad_per_s3=None,
            rms_joint_torque_nm=None,
            max_joint_torque_nm=None,
            positive_mechanical_energy_j=None,
            motor_copper_loss_j=None,
            model_electrical_energy_j=None,
            valid_trajectory=False,
            warning="unreachable_path",
        )

    solutions = arm.inverse_path(curve.points, elbow_mode)  # type: ignore[arg-type]
    motion, total_rotation = _joint_motion_metrics(solutions)
    singularity = [item.singularity_index for item in solutions]
    resampled_singularity = _resample_with_time_law(singularity)
    dt = selected_config.duration_s / (len(solutions) - 1)
    singularity_penalty = _trapezoid(
        [
            1.0 / (value + selected_config.singularity_epsilon) ** 2
            for value in resampled_singularity
        ],
        dt,
    )

    collision_free: bool | None = True
    collision_samples: int | None = 0
    collision_ratio: float | None = 0.0
    minimum_clearance: float | None = None
    clearance_penalty: float | None = 0.0
    if obstacle_list:
        clearances = [
            pose_obstacle_clearance(
                arm,
                solution,
                obstacle_list,
                link_radius_units=(
                    selected_config.link_radius_m / selected_config.workspace_unit_m
                ),
            )
            for solution in solutions
        ]
        numeric_clearances = [float(value) for value in clearances if value is not None]
        minimum_clearance = min(numeric_clearances) * selected_config.workspace_unit_m
        collision_samples = sum(value <= 0.0 for value in numeric_clearances)
        collision_ratio = collision_samples / len(numeric_clearances)
        collision_free = collision_samples == 0
        resampled_clearance = _resample_with_time_law(
            [
                value * selected_config.workspace_unit_m
                for value in numeric_clearances
            ]
        )
        clearance_penalty = _trapezoid(
            [
                1.0 / max(value, selected_config.clearance_epsilon_m) ** 2
                for value in resampled_clearance
            ],
            dt,
        )

    dynamic = _dynamic_metrics(arm, solutions, selected_config)
    warnings: list[str] = []
    if min(singularity) < selected_config.singularity_epsilon:
        warnings.append("near_singularity")
    if collision_free is False:
        warnings.append("collision")
    return TrajectoryEvaluation(
        **common,
        collision_free=collision_free,
        collision_samples=collision_samples,
        collision_ratio=collision_ratio,
        min_obstacle_clearance_m=minimum_clearance,
        clearance_penalty_s_per_m2=clearance_penalty,
        joint_motion_l2_rad=motion,
        total_joint_rotation_deg=total_rotation,
        min_singularity_index=min(singularity),
        mean_singularity_index=fmean(singularity),
        singularity_penalty_s=singularity_penalty,
        **dynamic,
        valid_trajectory=fully_reachable and bool(collision_free),
        warning=";".join(warnings),
    )
