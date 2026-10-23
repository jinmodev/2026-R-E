"""Reproducible phase-2 experiment generation and metric extraction.

The data in this module intentionally describes geometry and normalized joint
motion only. It does not claim to measure motor energy or obstacle avoidance.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from .curves import CurvePath, make_curve
from .kinematics import JointSolution, PlanarArm2R


DEFAULT_SEED = 20260825
DEFAULT_SAMPLES = 161
SHAPE_LEVELS = (-0.8, -0.4, 0.4, 0.8)


@dataclass(frozen=True, slots=True)
class Scenario:
    scenario_id: str
    scenario_type: str
    seed: int
    base_x: float
    base_y: float
    link1: float
    link2: float
    start_x: float
    start_y: float
    goal_x: float
    goal_y: float

    @property
    def min_reach(self) -> float:
        return abs(self.link1 - self.link2)

    @property
    def max_reach(self) -> float:
        return self.link1 + self.link2

    @property
    def link_ratio(self) -> float:
        return min(self.link1, self.link2) / max(self.link1, self.link2)

    @property
    def start_radius(self) -> float:
        return math.hypot(self.start_x - self.base_x, self.start_y - self.base_y)

    @property
    def goal_radius(self) -> float:
        return math.hypot(self.goal_x - self.base_x, self.goal_y - self.base_y)

    @property
    def chord_length(self) -> float:
        return math.hypot(self.goal_x - self.start_x, self.goal_y - self.start_y)

    @property
    def angular_separation_deg(self) -> float:
        start_angle = math.atan2(self.start_y - self.base_y, self.start_x - self.base_x)
        goal_angle = math.atan2(self.goal_y - self.base_y, self.goal_x - self.base_x)
        delta = (goal_angle - start_angle + math.pi) % (2.0 * math.pi) - math.pi
        return abs(math.degrees(delta))

    def as_row(self) -> dict[str, object]:
        return {
            **asdict(self),
            "link_ratio": self.link_ratio,
            "min_reach": self.min_reach,
            "max_reach": self.max_reach,
            "start_radius": self.start_radius,
            "goal_radius": self.goal_radius,
            "chord_length": self.chord_length,
            "angular_separation_deg": self.angular_separation_deg,
        }


@dataclass(frozen=True, slots=True)
class CurveVariant:
    curve_key: str
    curve_name_ko: str
    shape: float

    @property
    def variant_id(self) -> str:
        if self.curve_key == "line":
            return "line"
        sign = "p" if self.shape >= 0.0 else "m"
        return f"{self.curve_key}_{sign}{round(abs(self.shape) * 100):02d}"


SCENARIO_FIELDS = [
    "scenario_id", "scenario_type", "seed", "base_x", "base_y", "link1", "link2",
    "link_ratio", "min_reach", "max_reach", "start_x", "start_y", "goal_x", "goal_y",
    "start_radius", "goal_radius", "chord_length", "angular_separation_deg",
]


METRIC_FIELDS = [
    "run_id", "scenario_id", "scenario_type", "seed", "link1", "link2", "link_ratio",
    "start_x", "start_y", "goal_x", "goal_y", "start_radius", "goal_radius", "chord_length",
    "curve_key", "curve_name_ko", "shape", "elbow_mode", "samples", "curve_parameters_json",
    "curve_length", "path_to_chord_ratio", "fully_reachable", "reachable_samples",
    "reachable_ratio", "unreachable_samples", "theta1_min_deg", "theta1_max_deg",
    "theta1_range_deg", "theta2_min_deg", "theta2_max_deg", "theta2_range_deg",
    "total_abs_theta1_deg", "total_abs_theta2_deg", "total_joint_rotation_deg",
    "joint_motion_l2_rad", "joint_velocity_effort", "joint_acceleration_effort",
    "max_joint_speed_rad", "min_singularity_index", "mean_singularity_index", "warning",
]


def curve_variants() -> tuple[CurveVariant, ...]:
    variants = [CurveVariant("line", "직선", 0.0)]
    names = {
        "arc": "원호",
        "bezier": "3차 베지어",
        "hermite": "3차 에르미트",
        "cycloid": "사이클로이드",
        "limacon": "리마송",
    }
    variants.extend(
        CurveVariant(key, names[key], shape)
        for key in names
        for shape in SHAPE_LEVELS
    )
    return tuple(variants)


def _radius(minimum: float, maximum: float, fraction: float) -> float:
    return minimum + fraction * (maximum - minimum)


def _point(radius: float, angle: float) -> tuple[float, float]:
    return radius * math.cos(angle), radius * math.sin(angle)


def _make_scenario(index: int, seed: int, rng: random.Random) -> Scenario:
    scenario_type = ("local_move", "cross_inner", "boundary_sweep")[index % 3]

    if scenario_type == "cross_inner":
        link1 = rng.uniform(3.4, 5.0)
        link2 = rng.uniform(1.5, 3.0)
    else:
        link1 = rng.uniform(2.0, 5.0)
        link2 = rng.uniform(2.0, 5.0)

    minimum = abs(link1 - link2)
    maximum = link1 + link2
    start_angle = rng.uniform(-math.pi, math.pi)

    if scenario_type == "local_move":
        start_fraction = rng.uniform(0.25, 0.85)
        goal_fraction = min(0.95, max(0.05, start_fraction + rng.uniform(-0.18, 0.18)))
        separation = rng.choice((-1.0, 1.0)) * rng.uniform(0.25, 1.15)
    elif scenario_type == "cross_inner":
        start_fraction = rng.uniform(0.68, 0.96)
        goal_fraction = rng.uniform(0.68, 0.96)
        separation = rng.choice((-1.0, 1.0)) * rng.uniform(2.35, math.pi)
    else:
        start_fraction = rng.choice((rng.uniform(0.03, 0.16), rng.uniform(0.86, 0.97)))
        goal_fraction = rng.uniform(0.08, 0.94)
        separation = rng.choice((-1.0, 1.0)) * rng.uniform(0.7, 2.4)

    start_radius = _radius(minimum, maximum, start_fraction)
    goal_radius = _radius(minimum, maximum, goal_fraction)
    start = _point(start_radius, start_angle)
    goal = _point(goal_radius, start_angle + separation)

    return Scenario(
        scenario_id=f"S{index + 1:04d}",
        scenario_type=scenario_type,
        seed=seed,
        base_x=0.0,
        base_y=0.0,
        link1=link1,
        link2=link2,
        start_x=start[0],
        start_y=start[1],
        goal_x=goal[0],
        goal_y=goal[1],
    )


def generate_scenarios(count: int, seed: int = DEFAULT_SEED) -> list[Scenario]:
    if count <= 0:
        raise ValueError("count must be positive.")
    rng = random.Random(seed)
    return [_make_scenario(index, seed, rng) for index in range(count)]


def _differences(values: list[float]) -> list[float]:
    return [right - left for left, right in zip(values, values[1:])]


def _joint_metrics(solutions: list[JointSolution], samples: int) -> dict[str, float]:
    theta1 = [solution.theta1 for solution in solutions]
    theta2 = [solution.theta2 for solution in solutions]
    delta1 = _differences(theta1)
    delta2 = _differences(theta2)
    degrees = 180.0 / math.pi
    dt = 1.0 / max(samples - 1, 1)

    velocity_effort = sum(
        (d1 / dt) ** 2 + (d2 / dt) ** 2 for d1, d2 in zip(delta1, delta2)
    ) * dt
    speeds = [math.hypot(d1, d2) / dt for d1, d2 in zip(delta1, delta2)]
    acceleration_effort = 0.0
    for index in range(1, len(theta1) - 1):
        accel1 = (theta1[index + 1] - 2.0 * theta1[index] + theta1[index - 1]) / (dt * dt)
        accel2 = (theta2[index + 1] - 2.0 * theta2[index] + theta2[index - 1]) / (dt * dt)
        acceleration_effort += (accel1 * accel1 + accel2 * accel2) * dt

    singularity = [solution.singularity_index for solution in solutions]
    return {
        "theta1_min_deg": min(theta1) * degrees,
        "theta1_max_deg": max(theta1) * degrees,
        "theta1_range_deg": (max(theta1) - min(theta1)) * degrees,
        "theta2_min_deg": min(theta2) * degrees,
        "theta2_max_deg": max(theta2) * degrees,
        "theta2_range_deg": (max(theta2) - min(theta2)) * degrees,
        "total_abs_theta1_deg": sum(abs(value) for value in delta1) * degrees,
        "total_abs_theta2_deg": sum(abs(value) for value in delta2) * degrees,
        "total_joint_rotation_deg": sum(abs(value) for value in delta1 + delta2) * degrees,
        "joint_motion_l2_rad": sum(math.hypot(d1, d2) for d1, d2 in zip(delta1, delta2)),
        "joint_velocity_effort": velocity_effort,
        "joint_acceleration_effort": acceleration_effort,
        "max_joint_speed_rad": max(speeds, default=0.0),
        "min_singularity_index": min(singularity),
        "mean_singularity_index": sum(singularity) / len(singularity),
    }


def evaluate_variant(
    scenario: Scenario,
    variant: CurveVariant,
    elbow_mode: str,
    samples: int = DEFAULT_SAMPLES,
) -> dict[str, object]:
    arm = PlanarArm2R(scenario.link1, scenario.link2, (scenario.base_x, scenario.base_y))
    curve: CurvePath = make_curve(
        variant.curve_key,
        (scenario.start_x, scenario.start_y),
        (scenario.goal_x, scenario.goal_y),
        shape=variant.shape,
        samples=samples,
    )
    reachable = [arm.is_reachable(point) for point in curve.points]
    reachable_samples = sum(reachable)
    fully_reachable = reachable_samples == len(curve.points)
    metrics: dict[str, object] = {field: None for field in METRIC_FIELDS}
    metrics.update({
        "run_id": f"{scenario.scenario_id}_{variant.variant_id}_{elbow_mode}",
        "scenario_id": scenario.scenario_id,
        "scenario_type": scenario.scenario_type,
        "seed": scenario.seed,
        "link1": scenario.link1,
        "link2": scenario.link2,
        "link_ratio": scenario.link_ratio,
        "start_x": scenario.start_x,
        "start_y": scenario.start_y,
        "goal_x": scenario.goal_x,
        "goal_y": scenario.goal_y,
        "start_radius": scenario.start_radius,
        "goal_radius": scenario.goal_radius,
        "chord_length": scenario.chord_length,
        "curve_key": variant.curve_key,
        "curve_name_ko": variant.curve_name_ko,
        "shape": variant.shape,
        "elbow_mode": elbow_mode,
        "samples": samples,
        "curve_parameters_json": json.dumps(dict(curve.parameters), ensure_ascii=False, sort_keys=True),
        "curve_length": curve.length,
        "path_to_chord_ratio": curve.length / scenario.chord_length,
        "fully_reachable": fully_reachable,
        "reachable_samples": reachable_samples,
        "reachable_ratio": reachable_samples / len(curve.points),
        "unreachable_samples": len(curve.points) - reachable_samples,
    })

    if fully_reachable:
        solutions = arm.inverse_path(curve.points, elbow_mode)  # type: ignore[arg-type]
        joint_metrics = _joint_metrics(solutions, samples)
        metrics.update(joint_metrics)
        metrics["warning"] = (
            "near_singularity" if joint_metrics["min_singularity_index"] < 0.05 else ""
        )
    else:
        metrics["warning"] = "unreachable_path"
    return metrics


def run_experiments(
    scenarios: Iterable[Scenario], *, samples: int = DEFAULT_SAMPLES
) -> Iterator[dict[str, object]]:
    for scenario in scenarios:
        for variant in curve_variants():
            for elbow_mode in ("up", "down"):
                yield evaluate_variant(scenario, variant, elbow_mode, samples)


def _write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_dataset(
    output_dir: str | Path,
    *,
    scenario_count: int = 240,
    seed: int = DEFAULT_SEED,
    samples: int = DEFAULT_SAMPLES,
) -> dict[str, object]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    scenarios = generate_scenarios(scenario_count, seed)
    scenario_rows = [scenario.as_row() for scenario in scenarios]
    metric_rows = list(run_experiments(scenarios, samples=samples))

    _write_csv(destination / "scenarios.csv", SCENARIO_FIELDS, scenario_rows)
    _write_csv(destination / "trajectory_metrics.csv", METRIC_FIELDS, metric_rows)

    metadata = {
        "dataset_version": "phase2-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "scenario_count": scenario_count,
        "curve_variant_count": len(curve_variants()),
        "elbow_modes": 2,
        "run_count": len(metric_rows),
        "samples_per_path": samples,
        "shape_levels": list(SHAPE_LEVELS),
        "scenario_types": ["local_move", "cross_inner", "boundary_sweep"],
        "limitations": [
            "No obstacles or collision checks are included.",
            "Joint effort metrics use normalized parameter time T=1 and are not motor energy.",
            "Reachability is checked at discrete path samples, not proved continuously.",
        ],
    }
    (destination / "experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata

