"""Phase-4 reproducible mathematics-first dataset generation.

Phase 2 and Phase 3 remain unchanged.  This module adds a separately versioned
pipeline whose labels are based on path length, joint-motion effort, jerk,
curvature and obstacle clearance rather than motor dynamics.
"""

from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
from itertools import repeat
from typing import Callable, Iterable, Iterator, Mapping

from .curves import make_curve
from .evaluation import CircleObstacle
from .experiments import CurveVariant, Scenario, curve_variants
from .kinematics import PlanarArm2R
from .phase4_evaluation import (
    MathematicalEvaluationConfig,
    MathematicalTrajectoryResult,
    simulate_trajectory,
)


DEFAULT_PHASE4_SEED = 20260904
DEFAULT_PHASE4_SAMPLES = 161
DATASET_VERSION = "phase4-mathematics-v2"
EQUATION_FORMAT_VERSION = "phase4-equations-v1"
CODE_VERSION = "phase4-generator-v4"
SAMPLE_SCHEMA_VERSION = "phase4-samples-v3"
SPLIT_POLICY_VERSION = "separate-seed-permutation-60-20-20-v1"
_SPLIT_SEED_OFFSET = 8_000_009

RAW_OBJECTIVES: dict[str, str] = {
    "distance": "path_length_units",
    "joint_effort": "joint_velocity_effort",
    "jerk": "joint_jerk_effort",
    "curvature": "regularized_curvature_energy_inv_unit",
    "safety": "obstacle_penalty_s_per_unit2",
}

BALANCED_WEIGHTS: dict[str, float] = {
    "distance": 0.25,
    "joint_effort": 0.20,
    "jerk": 0.20,
    "curvature": 0.20,
    "safety": 0.15,
}

WINNER_FIELDS = (
    "scenario_id",
    "data_split",
    "dataset_partition",
    "objective",
    "winner_status",
    "tie_count",
    "tied_run_ids_json",
    "tie_tolerance",
    "winner_run_id",
    "curve_key",
    "curve_name_ko",
    "shape",
    "elbow_mode",
    "raw_metric_name",
    "raw_metric_value",
    "normalized_metric_value",
    "balanced_cost_0_100",
    "balanced_reward_0_100",
    "curve_equation_template",
    "curve_equation",
    "curve_parameters_json",
    "x_t_equation",
    "y_t_equation",
    "theta1_t_equation",
    "theta2_t_equation",
    "equation_format_version",
    "equation_descriptor_json",
)
WINNER_TIE_REL_TOL = 1e-12
WINNER_TIE_ABS_TOL = 1e-12


@dataclass(frozen=True, slots=True)
class Phase4Scenario:
    robot: Scenario
    obstacles: tuple[CircleObstacle, ...]
    obstacle_layout: str
    data_split: str
    dataset_partition: str = "in_distribution"

    def as_row(self) -> dict[str, object]:
        return {
            **self.robot.as_row(),
            "data_split": self.data_split,
            "dataset_partition": self.dataset_partition,
            "obstacle_layout": self.obstacle_layout,
            "obstacle_count": len(self.obstacles),
            "obstacles_json": json.dumps(
                [obstacle.as_dict() for obstacle in self.obstacles],
                ensure_ascii=False,
                sort_keys=True,
            ),
        }


@dataclass(frozen=True, slots=True)
class Phase4Run:
    metric_row: dict[str, object]
    sample_rows: tuple[dict[str, object], ...]


def _split_counts(count: int) -> dict[str, int]:
    """Return the closest integer 60/20/20 allocation summing to ``count``."""

    names_and_weights = (("train", 6), ("validation", 2), ("test", 2))
    counts = {name: count * weight // 10 for name, weight in names_and_weights}
    remainder = count - sum(counts.values())
    allocation_order = sorted(
        range(len(names_and_weights)),
        key=lambda index: (
            -(count * names_and_weights[index][1] % 10),
            index,
        ),
    )
    for index in allocation_order[:remainder]:
        counts[names_and_weights[index][0]] += 1
    return counts


def _standard_split_assignment(count: int, seed: int) -> dict[int, str]:
    """Shuffle scenario indices with an RNG independent of geometry generation."""

    permutation = list(range(count))
    random.Random(seed + _SPLIT_SEED_OFFSET).shuffle(permutation)
    counts = _split_counts(count)
    assignment: dict[int, str] = {}
    cursor = 0
    for split in ("train", "validation", "test"):
        right = cursor + counts[split]
        for index in permutation[cursor:right]:
            assignment[index] = split
        cursor = right
    return assignment


def _polar_point(base: tuple[float, float], radius: float, angle: float) -> tuple[float, float]:
    return base[0] + radius * math.cos(angle), base[1] + radius * math.sin(angle)


def _scenario_fingerprint(robot: Scenario) -> tuple[int, ...]:
    values = (
        robot.base_x,
        robot.base_y,
        robot.link1,
        robot.link2,
        robot.start_x,
        robot.start_y,
        robot.goal_x,
        robot.goal_y,
    )
    return tuple(round(value * 10_000) for value in values)


def _make_robot(
    index: int,
    seed: int,
    rng: random.Random,
    *,
    challenge: bool,
) -> Scenario:
    if challenge:
        scenario_types = (
            "near_inner_boundary",
            "near_outer_boundary",
            "extreme_link_ratio",
            "wide_sweep",
            "boundary_to_boundary",
        )
    else:
        scenario_types = (
            "local_move",
            "radial_change",
            "cross_inner",
            "wide_sweep",
            "boundary_sweep",
        )
    scenario_type = scenario_types[index % len(scenario_types)]
    base = (rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5))

    if scenario_type in {"extreme_link_ratio", "cross_inner"}:
        link1 = rng.uniform(3.7, 5.5)
        link2 = rng.uniform(1.1, 2.2)
        if rng.random() < 0.5:
            link1, link2 = link2, link1
    else:
        link1 = rng.uniform(2.0, 5.5)
        link2 = rng.uniform(2.0, 5.5)

    minimum = abs(link1 - link2)
    maximum = link1 + link2
    span = maximum - minimum
    start_angle = rng.uniform(-math.pi, math.pi)
    direction = rng.choice((-1.0, 1.0))

    if scenario_type == "local_move":
        start_fraction = rng.uniform(0.18, 0.85)
        goal_fraction = min(0.94, max(0.06, start_fraction + rng.uniform(-0.18, 0.18)))
        separation = direction * rng.uniform(0.18, 1.05)
    elif scenario_type == "radial_change":
        start_fraction = rng.uniform(0.08, 0.32)
        goal_fraction = rng.uniform(0.68, 0.94)
        if rng.random() < 0.5:
            start_fraction, goal_fraction = goal_fraction, start_fraction
        separation = direction * rng.uniform(0.25, 1.35)
    elif scenario_type in {"cross_inner", "wide_sweep"}:
        start_fraction = rng.uniform(0.55, 0.94)
        goal_fraction = rng.uniform(0.55, 0.94)
        separation = direction * rng.uniform(2.15, math.pi - 0.04)
    elif scenario_type == "near_inner_boundary":
        start_fraction = rng.uniform(0.005, 0.045)
        goal_fraction = rng.uniform(0.08, 0.35)
        separation = direction * rng.uniform(1.5, 3.05)
    elif scenario_type == "near_outer_boundary":
        start_fraction = rng.uniform(0.955, 0.995)
        goal_fraction = rng.uniform(0.82, 0.99)
        separation = direction * rng.uniform(0.9, 2.6)
    elif scenario_type == "boundary_to_boundary":
        start_fraction = rng.uniform(0.005, 0.06)
        goal_fraction = rng.uniform(0.94, 0.995)
        separation = direction * rng.uniform(1.3, 2.9)
    else:
        start_fraction = rng.choice((rng.uniform(0.015, 0.12), rng.uniform(0.88, 0.98)))
        goal_fraction = rng.uniform(0.08, 0.94)
        separation = direction * rng.uniform(0.75, 2.45)

    start_radius = minimum + start_fraction * span
    goal_radius = minimum + goal_fraction * span
    start = _polar_point(base, start_radius, start_angle)
    goal = _polar_point(base, goal_radius, start_angle + separation)
    prefix = "P4C" if challenge else "P4S"
    return Scenario(
        scenario_id=f"{prefix}{index + 1:06d}",
        scenario_type=scenario_type,
        seed=seed,
        base_x=base[0],
        base_y=base[1],
        link1=link1,
        link2=link2,
        start_x=start[0],
        start_y=start[1],
        goal_x=goal[0],
        goal_y=goal[1],
    )


def _move_away_from_protected_points(
    center: tuple[float, float],
    radius: float,
    protected: Iterable[tuple[float, float]],
    fallback_direction: tuple[float, float],
) -> tuple[float, float]:
    x, y = center
    required = radius + 0.10
    protected_points = tuple(protected)
    for _ in range(8):
        moved = False
        for point in protected_points:
            dx, dy = x - point[0], y - point[1]
            distance = math.hypot(dx, dy)
            if distance >= required:
                continue
            if distance <= 1e-12:
                dx, dy = fallback_direction
                distance = math.hypot(dx, dy)
            scale = (required - distance + 0.02) / distance
            x += dx * scale
            y += dy * scale
            moved = True
        if not moved:
            break
    return x, y


def _make_obstacle(
    robot: Scenario,
    obstacle_id: str,
    center: tuple[float, float],
    radius: float,
    normal: tuple[float, float],
) -> CircleObstacle:
    adjusted = _move_away_from_protected_points(
        center,
        radius,
        (
            (robot.base_x, robot.base_y),
            (robot.start_x, robot.start_y),
            (robot.goal_x, robot.goal_y),
        ),
        normal,
    )
    return CircleObstacle(obstacle_id, adjusted[0], adjusted[1], radius)


def _generate_obstacles(
    robot: Scenario,
    rng: random.Random,
    layout: str,
) -> tuple[CircleObstacle, ...]:
    if layout == "no_obstacle":
        return ()
    start = (robot.start_x, robot.start_y)
    goal = (robot.goal_x, robot.goal_y)
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    chord = math.hypot(dx, dy)
    ux, uy = dx / chord, dy / chord
    nx, ny = -uy, ux
    reach = robot.max_reach

    specifications: list[tuple[float, float, float]] = []
    if layout == "chord_blocker":
        specifications.append((rng.uniform(0.38, 0.62), rng.uniform(-0.035, 0.035), min(0.10 * chord, 0.065 * reach)))
    elif layout == "double_offset":
        radius = min(0.075 * chord, 0.05 * reach)
        specifications.extend(((0.35, 0.10, radius), (0.65, -0.10, radius)))
    elif layout == "start_near":
        radius = min(0.075 * chord, 0.045 * reach)
        specifications.append((0.12, rng.choice((-1.0, 1.0)) * (radius / chord + 0.04), radius))
    elif layout == "goal_near":
        radius = min(0.075 * chord, 0.045 * reach)
        specifications.append((0.88, rng.choice((-1.0, 1.0)) * (radius / chord + 0.04), radius))
    elif layout == "narrow_passage":
        radius = min(0.095 * chord, 0.06 * reach)
        offset = radius / chord + rng.uniform(0.055, 0.09)
        specifications.extend(((0.50, offset, radius), (0.50, -offset, radius)))
    elif layout == "near_base_gate":
        radius = min(0.06 * reach, 0.11 * chord)
        specifications.extend(((0.28, 0.08, radius), (0.28, -0.08, radius)))
    elif layout in {"random_field", "dense_random"}:
        count = rng.randint(1, 3) if layout == "random_field" else rng.randint(3, 5)
        for _ in range(count):
            specifications.append(
                (
                    rng.uniform(0.16, 0.84),
                    rng.uniform(-0.28, 0.28),
                    min(rng.uniform(0.035, 0.075) * reach, 0.12 * chord),
                )
            )
    else:
        raise KeyError(f"Unknown obstacle layout: {layout!r}")

    obstacles: list[CircleObstacle] = []
    for obstacle_index, (progress, offset_ratio, radius) in enumerate(specifications):
        center = (
            start[0] + progress * dx + offset_ratio * chord * nx,
            start[1] + progress * dy + offset_ratio * chord * ny,
        )
        obstacles.append(
            _make_obstacle(
                robot,
                f"{robot.scenario_id}_O{obstacle_index + 1}",
                center,
                max(radius, 0.04),
                (nx, ny),
            )
        )
    return tuple(obstacles)


def generate_phase4_scenarios(
    count: int,
    *,
    seed: int = DEFAULT_PHASE4_SEED,
    challenge: bool = False,
) -> list[Phase4Scenario]:
    """Generate deterministic, non-duplicate standard or OOD scenarios."""

    if count < 0:
        raise ValueError("count cannot be negative.")
    if count == 0:
        return []
    rng = random.Random(seed + (4_000_003 if challenge else 0))
    standard_layouts = (
        "no_obstacle",
        "chord_blocker",
        "double_offset",
        "start_near",
        "goal_near",
        "narrow_passage",
        "random_field",
    )
    challenge_layouts = (
        "narrow_passage",
        "near_base_gate",
        "dense_random",
        "chord_blocker",
    )
    layouts = challenge_layouts if challenge else standard_layouts
    split_assignment = {} if challenge else _standard_split_assignment(count, seed)
    seen: set[tuple[int, ...]] = set()
    scenarios: list[Phase4Scenario] = []
    attempts = 0
    while len(scenarios) < count:
        attempts += 1
        if attempts > count * 20:
            raise RuntimeError("Unable to generate enough distinct scenarios.")
        index = len(scenarios)
        robot = _make_robot(index, seed, rng, challenge=challenge)
        fingerprint = _scenario_fingerprint(robot)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        layout = layouts[index % len(layouts)]
        obstacles = _generate_obstacles(robot, rng, layout)
        scenarios.append(
            Phase4Scenario(
                robot=robot,
                obstacles=obstacles,
                obstacle_layout=layout,
                data_split="challenge" if challenge else split_assignment[index],
                dataset_partition="challenge_ood" if challenge else "in_distribution",
            )
        )
    return scenarios


def evaluate_phase4_variant(
    scenario: Phase4Scenario,
    variant: CurveVariant,
    elbow_mode: str,
    *,
    samples: int = DEFAULT_PHASE4_SAMPLES,
    config: MathematicalEvaluationConfig | None = None,
) -> Phase4Run:
    """Evaluate one candidate and attach identifiers to summary and samples."""

    selected = config or MathematicalEvaluationConfig()
    robot = scenario.robot
    arm = PlanarArm2R(robot.link1, robot.link2, (robot.base_x, robot.base_y))
    curve = make_curve(
        variant.curve_key,
        (robot.start_x, robot.start_y),
        (robot.goal_x, robot.goal_y),
        shape=variant.shape,
        samples=samples,
    )
    result: MathematicalTrajectoryResult = simulate_trajectory(
        arm,
        curve,
        elbow_mode,
        obstacles=scenario.obstacles,
        config=selected,
        samples=samples,
    )
    run_id = f"{robot.scenario_id}_{variant.variant_id}_{elbow_mode}"
    identity: dict[str, object] = {
        "run_id": run_id,
        "scenario_id": robot.scenario_id,
        "scenario_type": robot.scenario_type,
        "seed": robot.seed,
        "data_split": scenario.data_split,
        "dataset_partition": scenario.dataset_partition,
        "base_x": robot.base_x,
        "base_y": robot.base_y,
        "link1": robot.link1,
        "link2": robot.link2,
        "link_ratio": robot.link_ratio,
        "start_x": robot.start_x,
        "start_y": robot.start_y,
        "goal_x": robot.goal_x,
        "goal_y": robot.goal_y,
        "start_radius": robot.start_radius,
        "goal_radius": robot.goal_radius,
        "chord_length": robot.chord_length,
        "obstacle_layout": scenario.obstacle_layout,
        "obstacle_count": len(scenario.obstacles),
        "obstacles_json": json.dumps(
            [item.as_dict() for item in scenario.obstacles],
            ensure_ascii=False,
            sort_keys=True,
        ),
        "curve_key": variant.curve_key,
        "curve_name_ko": variant.curve_name_ko,
        "shape": variant.shape,
        "elbow_mode": elbow_mode,
    }
    metric_row = {**identity, **result.metrics}
    sample_identity = {
        "run_id": run_id,
        "scenario_id": robot.scenario_id,
        "data_split": scenario.data_split,
        "dataset_partition": scenario.dataset_partition,
        "curve_key": variant.curve_key,
        "shape": variant.shape,
        "elbow_mode": elbow_mode,
    }
    sample_rows = tuple({**sample_identity, **row} for row in result.sample_rows)
    return Phase4Run(metric_row, sample_rows)


def run_phase4_experiments(
    scenarios: Iterable[Phase4Scenario],
    *,
    samples: int = DEFAULT_PHASE4_SAMPLES,
    config: MathematicalEvaluationConfig | None = None,
) -> Iterator[Phase4Run]:
    for scenario in scenarios:
        for variant in curve_variants():
            for elbow_mode in ("up", "down"):
                yield evaluate_phase4_variant(
                    scenario,
                    variant,
                    elbow_mode,
                    samples=samples,
                    config=config,
                )


def _evaluate_scenario_runs(
    scenario: Phase4Scenario,
    samples: int,
    config: MathematicalEvaluationConfig,
) -> tuple[Phase4Run, ...]:
    """Pickle-friendly unit of work used by the optional process pool."""

    return tuple(
        run_phase4_experiments((scenario,), samples=samples, config=config)
    )


def _validate_weights(weights: Mapping[str, float]) -> None:
    if set(weights) != set(RAW_OBJECTIVES):
        raise ValueError(f"Weights must define exactly {sorted(RAW_OBJECTIVES)}.")
    if any(value < 0.0 for value in weights.values()):
        raise ValueError("Weights cannot be negative.")
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12):
        raise ValueError("Weights must sum to 1.")


def score_phase4_candidates(
    records: Iterable[Mapping[str, object]],
    *,
    weights: Mapping[str, float] = BALANCED_WEIGHTS,
) -> list[dict[str, object]]:
    """Normalize objectives within each scenario and add balanced scores."""

    _validate_weights(weights)
    rows = [dict(record) for record in records]
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["scenario_id"]), []).append(row)
    for candidates in grouped.values():
        valid = [row for row in candidates if bool(row.get("selection_valid"))]
        bounds: dict[str, tuple[float, float]] = {}
        for objective, field in RAW_OBJECTIVES.items():
            values = [float(row[field]) for row in valid if row.get(field) is not None]
            if len(values) != len(valid):
                raise ValueError(f"Valid candidates are missing {field!r}.")
            if any(not math.isfinite(value) for value in values):
                raise ValueError(
                    f"Valid candidates must have finite values for {field!r}."
                )
            bounds[objective] = (min(values), max(values)) if values else (0.0, 0.0)
        for row in candidates:
            if not bool(row.get("selection_valid")):
                for objective in RAW_OBJECTIVES:
                    row[f"normalized_{objective}"] = None
                row["balanced_cost_0_100"] = 1_000_000.0
                row["balanced_reward_0_100"] = 0.0
                continue
            normalized: dict[str, float] = {}
            for objective, field in RAW_OBJECTIVES.items():
                low, high = bounds[objective]
                value = float(row[field])
                scaled = 0.0 if high - low <= 1e-15 else (value - low) / (high - low)
                normalized[objective] = max(0.0, min(1.0, scaled))
                row[f"normalized_{objective}"] = normalized[objective]
            cost = 100.0 * sum(weights[name] * normalized[name] for name in weights)
            row["balanced_cost_0_100"] = cost
            row["balanced_reward_0_100"] = 100.0 - cost
    return rows


def _winner_key(row: Mapping[str, object], field: str) -> tuple[object, ...]:
    return (
        float(row[field]),
        float(row["path_length_units"]),
        str(row["curve_key"]),
        float(row["shape"]),
        str(row["elbow_mode"]),
        str(row["run_id"]),
    )


def select_scenario_winners(
    scored_records: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return six deterministic label rows per scenario."""

    rows = [dict(row) for row in scored_records]
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["scenario_id"]), []).append(row)
    winners: list[dict[str, object]] = []
    objective_fields = {**RAW_OBJECTIVES, "balanced": "balanced_cost_0_100"}
    for scenario_id in sorted(grouped):
        candidates = grouped[scenario_id]
        valid = [row for row in candidates if bool(row.get("selection_valid"))]
        example = candidates[0]
        for objective, field in objective_fields.items():
            output = {name: None for name in WINNER_FIELDS}
            output.update(
                {
                    "scenario_id": scenario_id,
                    "data_split": example["data_split"],
                    "dataset_partition": example["dataset_partition"],
                    "objective": objective,
                    "tie_tolerance": (
                        f"rel={WINNER_TIE_REL_TOL:g},abs={WINNER_TIE_ABS_TOL:g}"
                    ),
                }
            )
            if not valid:
                output.update(
                    {
                        "winner_status": "no_valid_candidate",
                        "tie_count": 0,
                        "tied_run_ids_json": "[]",
                    }
                )
                winners.append(output)
                continue
            best_value = min(float(row[field]) for row in valid)
            tied = [
                row
                for row in valid
                if math.isclose(
                    float(row[field]),
                    best_value,
                    rel_tol=WINNER_TIE_REL_TOL,
                    abs_tol=WINNER_TIE_ABS_TOL,
                )
            ]
            winner = min(tied, key=lambda row: _winner_key(row, field))
            raw_field = RAW_OBJECTIVES.get(objective)
            output.update(
                {
                    "data_split": winner["data_split"],
                    "dataset_partition": winner["dataset_partition"],
                    "winner_status": "tie_resolved" if len(tied) > 1 else "ok",
                    "tie_count": len(tied),
                    "tied_run_ids_json": json.dumps(
                        sorted(str(row["run_id"]) for row in tied)
                    ),
                    "winner_run_id": winner["run_id"],
                    "curve_key": winner["curve_key"],
                    "curve_name_ko": winner["curve_name_ko"],
                    "shape": winner["shape"],
                    "elbow_mode": winner["elbow_mode"],
                    "raw_metric_name": raw_field if raw_field else field,
                    "raw_metric_value": winner[raw_field] if raw_field else winner[field],
                    "normalized_metric_value": (
                        winner.get(f"normalized_{objective}")
                        if objective in RAW_OBJECTIVES
                        else None
                    ),
                    "balanced_cost_0_100": winner["balanced_cost_0_100"],
                    "balanced_reward_0_100": winner["balanced_reward_0_100"],
                    "curve_equation_template": winner["curve_equation_template"],
                    "curve_equation": winner["curve_equation"],
                    "curve_parameters_json": winner["curve_parameters_json"],
                    "x_t_equation": winner["x_t_equation"],
                    "y_t_equation": winner["y_t_equation"],
                    "theta1_t_equation": winner["theta1_t_equation"],
                    "theta2_t_equation": winner["theta2_t_equation"],
                    "equation_format_version": winner["equation_format_version"],
                    "equation_descriptor_json": winner["equation_descriptor_json"],
                }
            )
            winners.append(output)
    return winners


def _write_csv(path: Path, rows: list[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _combine_csv_parts(parts: list[Path], destination: Path) -> None:
    if not parts:
        raise ValueError("No CSV parts to combine.")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as output:
        for index, part in enumerate(parts):
            with part.open("rb") as source:
                content = source.read()
            if index > 0:
                bom = b"\xef\xbb\xbf"
                if content.startswith(bom):
                    content = content[len(bom):]
                newline = content.find(b"\n")
                content = content[newline + 1 :]
            output.write(content)
    temporary.replace(destination)


def _parquet_available() -> bool:
    return importlib.util.find_spec("pyarrow") is not None


def _resolve_sample_format(requested: str) -> str:
    if requested not in {"auto", "parquet", "csv"}:
        raise ValueError("sample_format must be 'auto', 'parquet', or 'csv'.")
    if requested == "auto":
        return "parquet" if _parquet_available() else "csv"
    if requested == "parquet" and not _parquet_available():
        raise RuntimeError("Parquet export requires pyarrow; use sample_format='csv'.")
    return requested


def _sample_arrow_schema():
    import pyarrow as pa

    string_fields = {
        "run_id",
        "scenario_id",
        "data_split",
        "dataset_partition",
        "curve_key",
        "elbow_mode",
    }
    integer_fields = {"sample_index"}
    boolean_fields = {
        "collision_at_sample",
        "reachable_at_sample",
        "validation_collision_at_sample",
        "validation_reachable_at_sample",
        "path_regular",
    }
    all_fields = (
        "run_id", "scenario_id", "data_split", "dataset_partition",
        "curve_key", "shape", "elbow_mode", "sample_index", "time_s", "normalized_time",
        "progress_u", "x", "y", "geometry_u", "geometry_x", "geometry_y",
        "validation_u", "validation_x", "validation_y",
        "dx_du", "dy_du", "d2x_du2", "d2y_du2",
        "path_speed_units_per_u", "curvature_inv_unit",
        "regularized_curvature_integrand_inv_unit",
        "curvature_speed_epsilon_units_per_u", "path_regular",
        "theta1_rad", "theta2_rad",
        "theta1_deg", "theta2_deg", "theta1_velocity_rad_s",
        "theta2_velocity_rad_s", "theta1_acceleration_rad_s2",
        "theta2_acceleration_rad_s2", "theta1_jerk_rad_s3",
        "theta2_jerk_rad_s3", "joint_velocity_squared_rad2_s2",
        "joint_acceleration_squared_rad2_s4", "joint_jerk_squared_rad2_s6",
        "obstacle_clearance", "obstacle_penalty_integrand_per_unit2",
        "collision_at_sample", "reachable_at_sample",
        "validation_obstacle_clearance", "validation_collision_at_sample",
        "validation_reachable_at_sample",
    )
    fields = []
    for name in all_fields:
        data_type = (
            pa.string()
            if name in string_fields
            else pa.int64()
            if name in integer_fields
            else pa.bool_()
            if name in boolean_fields
            else pa.float64()
        )
        fields.append(pa.field(name, data_type, nullable=True))
    return pa.schema(fields)


class _SamplePartitionStream:
    """Atomic, incremental writer for one batch/split sample partition."""

    def __init__(self, path_stem: Path, sample_format: str) -> None:
        path_stem.parent.mkdir(parents=True, exist_ok=True)
        self.sample_format = sample_format
        self.path = path_stem.with_suffix(
            ".parquet" if sample_format == "parquet" else ".csv.gz"
        )
        self.temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        self._writer = None
        self._handle = None
        self._fieldnames: list[str] | None = None
        self._buffer: list[dict[str, object]] = []
        self._buffer_row_limit = 10_000

    def write(self, rows: Iterable[Mapping[str, object]]) -> None:
        row_list = [dict(row) for row in rows]
        if not row_list:
            return
        if self.sample_format == "parquet":
            self._buffer.extend(row_list)
            if len(self._buffer) >= self._buffer_row_limit:
                self._flush_parquet()
            return
        if self._writer is None:
            self._handle = gzip.open(
                self.temporary, "wt", newline="", encoding="utf-8-sig"
            )
            self._fieldnames = list(row_list[0])
            self._writer = csv.DictWriter(
                self._handle,
                fieldnames=self._fieldnames,
                extrasaction="ignore",
                lineterminator="\n",
            )
            self._writer.writeheader()
        self._writer.writerows(row_list)

    def _flush_parquet(self) -> None:
        if not self._buffer:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        schema = _sample_arrow_schema()
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self.temporary,
                schema,
                compression="zstd",
                use_dictionary=True,
            )
        table = pa.Table.from_pylist(self._buffer, schema=schema)
        self._writer.write_table(table)
        self._buffer.clear()

    def close(self, *, commit: bool) -> None:
        if commit and self.sample_format == "parquet":
            self._flush_parquet()
        else:
            self._buffer.clear()
        if self._writer is not None and self.sample_format == "parquet":
            self._writer.close()
        if self._handle is not None:
            self._handle.close()
        if commit and self.temporary.exists():
            self.temporary.replace(self.path)
        elif self.temporary.exists():
            self.temporary.unlink()


def _state_signature(
    *,
    scenario_count: int,
    challenge_count: int,
    seed: int,
    samples: int,
    batch_size: int,
    sample_format: str,
    config: MathematicalEvaluationConfig,
) -> dict[str, object]:
    return {
        "dataset_version": DATASET_VERSION,
        "code_version": CODE_VERSION,
        "sample_schema_version": SAMPLE_SCHEMA_VERSION,
        "split_policy_version": SPLIT_POLICY_VERSION,
        "scenario_count": scenario_count,
        "challenge_count": challenge_count,
        "seed": seed,
        "samples": samples,
        "batch_size": batch_size,
        "sample_format": sample_format,
        "evaluation_config": asdict(config),
    }


def _save_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_csv_rows(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def _count_gzip_csv_rows(path: Path) -> int:
    """Count data rows without constructing dictionaries for a large CSV-gzip."""

    line_count = 0
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            line_count += chunk.count(b"\n")
    return max(0, line_count - 1)


def _tabular_row_count(path: Path) -> int | None:
    """Return a cheap, format-aware row count for generated tabular files."""

    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    if path.name.endswith(".csv.gz"):
        return _count_gzip_csv_rows(path)
    if path.suffix == ".csv":
        return _count_csv_rows(path)
    return None


_OUTCOME_COUNT_FIELDS = (
    "total",
    "valid",
    "reachability_failure",
    "collision_failure",
    "other_invalid",
)


def _as_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _candidate_outcome_counts(
    rows: Iterable[Mapping[str, object]],
) -> dict[str, int]:
    """Count mutually exclusive candidate outcomes, including CSV-loaded rows."""

    counts = {name: 0 for name in _OUTCOME_COUNT_FIELDS}
    for row in rows:
        counts["total"] += 1
        if _as_bool(row.get("selection_valid")) is True:
            counts["valid"] += 1
        elif _as_bool(row.get("fully_reachable")) is not True:
            counts["reachability_failure"] += 1
        elif _as_bool(row.get("collision_free")) is False:
            counts["collision_failure"] += 1
        else:
            counts["other_invalid"] += 1
    return counts


def _candidate_outcome_counts_from_csv(path: Path) -> dict[str, int]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return _candidate_outcome_counts(csv.DictReader(handle))


def _add_outcome_counts(
    target: dict[str, int], source: Mapping[str, object]
) -> None:
    for name in _OUTCOME_COUNT_FIELDS:
        target[name] += int(source.get(name, 0))


def _numeric_summary(values: Iterable[float]) -> dict[str, float | int | list[float]]:
    items = [float(value) for value in values]
    if not items:
        return {"count": 0, "actual_range": []}
    return {
        "count": len(items),
        "minimum": min(items),
        "maximum": max(items),
        "mean": math.fsum(items) / len(items),
        "actual_range": [min(items), max(items)],
    }


def _frequency_table(values: Iterable[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _generation_distribution_metadata(
    scenarios: list[Phase4Scenario],
) -> dict[str, object]:
    by_partition: dict[str, list[Phase4Scenario]] = {}
    for scenario in scenarios:
        by_partition.setdefault(scenario.dataset_partition, []).append(scenario)

    actual_links: dict[str, object] = {}
    for partition, subset in sorted(by_partition.items()):
        robots = [item.robot for item in subset]
        actual_links[partition] = {
            "link1_units": _numeric_summary(robot.link1 for robot in robots),
            "link2_units": _numeric_summary(robot.link2 for robot in robots),
            "either_link_units": _numeric_summary(
                length
                for robot in robots
                for length in (robot.link1, robot.link2)
            ),
            "short_to_long_link_ratio": _numeric_summary(
                robot.link_ratio for robot in robots
            ),
        }

    all_obstacles = [
        obstacle for scenario in scenarios for obstacle in scenario.obstacles
    ]
    obstacle_actual_by_partition = {
        partition: {
            "scenario_count": len(subset),
            "layout_counts": _frequency_table(
                item.obstacle_layout for item in subset
            ),
            "obstacle_count_per_scenario": _numeric_summary(
                len(item.obstacles) for item in subset
            ),
            "obstacle_count_frequency": _frequency_table(
                len(item.obstacles) for item in subset
            ),
            "total_obstacles": sum(len(item.obstacles) for item in subset),
            "radius_units": _numeric_summary(
                obstacle.radius_units
                for item in subset
                for obstacle in item.obstacles
            ),
        }
        for partition, subset in sorted(by_partition.items())
    }
    return {
        "link_length_generation": {
            "sampling_rules": {
                "ordinary_scenario": {
                    "link1_units": "uniform[2.0, 5.5]",
                    "link2_units": "uniform[2.0, 5.5]",
                },
                "cross_inner_or_extreme_link_ratio": {
                    "long_link_units": "uniform[3.7, 5.5]",
                    "short_link_units": "uniform[1.1, 2.2]",
                    "assignment": "random 50/50 swap between link1 and link2",
                },
            },
            "actual_by_dataset_partition": actual_links,
        },
        "obstacle_generation": {
            "layout_assignment": "deterministic cyclic assignment within each partition",
            "standard_layout_cycle": [
                "no_obstacle",
                "chord_blocker",
                "double_offset",
                "start_near",
                "goal_near",
                "narrow_passage",
                "random_field",
            ],
            "challenge_layout_cycle": [
                "narrow_passage",
                "near_base_gate",
                "dense_random",
                "chord_blocker",
            ],
            "parameter_rules": {
                "reference_frame": "centers are sampled relative to the start-goal chord",
                "random_field_count": "discrete uniform integer 1..3",
                "dense_random_count": "discrete uniform integer 3..5",
                "random_field_progress": "uniform[0.16, 0.84] along the chord",
                "random_field_normal_offset_ratio": "uniform[-0.28, 0.28]",
                "random_radius_rule": "min(uniform[0.035,0.075]*max_reach, 0.12*chord), then at least 0.04",
                "protected_points": "centers are displaced to clear base, start and goal by radius+0.10",
            },
            "actual_total_obstacles": len(all_obstacles),
            "actual_radius_units": _numeric_summary(
                obstacle.radius_units for obstacle in all_obstacles
            ),
            "actual_by_dataset_partition": obstacle_actual_by_partition,
        },
    }


def _write_data_dictionary(path: Path) -> None:
    content = """# Phase 4 data dictionary

## Files

- `scenarios.csv`: one row per independent robot/environment scenario.
- `trajectory_metrics.csv`: one row per candidate curve and elbow branch.
- `scenario_winners.csv`: one row per scenario and objective label.
- `trajectory_samples/`: time samples, partitioned by split and generation batch.

## Leakage boundary

Future model inputs may use scenario geometry, obstacles, curve identity and shape.
They must not use reachability/collision outcomes, raw or normalized objective values,
winner labels, costs or rewards as predictors.

## Mathematical objectives (all minimized)

- `path_length_units`: L = integral ||p'(u)|| du.
- `joint_velocity_effort`: E_theta = integral (theta1_dot^2 + theta2_dot^2) dt.
- `joint_acceleration_effort`: A = integral (theta1_ddot^2 + theta2_ddot^2) dt (auxiliary).
- `joint_jerk_effort`: J = integral (theta1_jerk^2 + theta2_jerk^2) dt.
- `regularized_curvature_energy_inv_unit`: K_epsilon = integral
  cross(p',p'')^2/(||p'||^2+epsilon^2)^(5/2) du. This is the selection
  objective and remains finite at cycloid endpoint cusps.
- `unregularized_curvature_energy_inv_unit`: ordinary K = integral kappa^2 ds,
  recorded only when the sampled path is regular (||p'|| stays above epsilon).
- `path_regular`: repeated in every sample row so standalone partitions preserve
  the ordinary-curvature eligibility rule; non-regular paths reintegrate to null.
- `obstacle_penalty_s_per_unit2`: O = integral 1/max(clearance, epsilon)^2 dt.

## Numerical grids and hard constraints

- Saved time rows use uniform t and minimum-jerk progress q(t).
- L and K_epsilon use exact built-in curve derivatives on the endpoint-clustered
  grid u_i=(1-cos(pi*i/(N-1)))/2.
- Reachability and collision are hard-validated on both the saved time grid and
  a separate uniform-u grid. The latter is stored in `validation_u/x/y`,
  `validation_reachable_at_sample`, `validation_obstacle_clearance`, and
  `validation_collision_at_sample`.
- `min_singularity_index=|sin(theta2)|` and `joint_derivative_reliable` flag
  paths whose high-order numerical derivatives need extra caution.

`joint_velocity_effort` is a mathematical motion-effort proxy, not measured energy
and not joules. Collision and incomplete reachability are hard invalidity rules.
"""
    path.write_text(content, encoding="utf-8")


def write_phase4_dataset(
    output_dir: str | Path,
    *,
    scenario_count: int = 5_000,
    challenge_count: int = 500,
    seed: int = DEFAULT_PHASE4_SEED,
    samples: int = DEFAULT_PHASE4_SAMPLES,
    batch_size: int = 25,
    workers: int = 1,
    resume: bool = False,
    sample_format: str = "auto",
    config: MathematicalEvaluationConfig | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Generate an auditable dataset incrementally, with batch-level resume."""

    if scenario_count <= 0:
        raise ValueError("scenario_count must be positive.")
    if challenge_count < 0:
        raise ValueError("challenge_count cannot be negative.")
    if samples < 7:
        raise ValueError("samples must be at least 7.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if workers <= 0:
        raise ValueError("workers must be positive.")
    selected = config or MathematicalEvaluationConfig()
    selected_format = _resolve_sample_format(sample_format)
    destination = Path(output_dir)
    state_path = destination / "generation_state.json"
    signature = _state_signature(
        scenario_count=scenario_count,
        challenge_count=challenge_count,
        seed=seed,
        samples=samples,
        batch_size=batch_size,
        sample_format=selected_format,
        config=selected,
    )
    if destination.exists() and any(destination.iterdir()) and not resume:
        raise FileExistsError(
            f"{destination} is not empty. Choose a new output or pass resume=True."
        )
    destination.mkdir(parents=True, exist_ok=True)
    batch_dir = destination / ".batches"
    batch_dir.mkdir(exist_ok=True)

    completed_batches: set[int] = set()
    stored_batch_outcomes: dict[int, dict[str, int]] = {}
    if resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("signature") != signature:
            raise ValueError("Resume parameters do not match the existing dataset state.")
        completed_batches = {int(value) for value in state.get("completed_batches", [])}
        for key, value in dict(state.get("batch_outcome_counts", {})).items():
            stored_batch_outcomes[int(key)] = {
                name: int(dict(value).get(name, 0))
                for name in _OUTCOME_COUNT_FIELDS
            }

    standard = generate_phase4_scenarios(scenario_count, seed=seed)
    challenge = generate_phase4_scenarios(challenge_count, seed=seed, challenge=True)
    scenarios = [*standard, *challenge]
    _write_csv(destination / "scenarios.csv", [item.as_row() for item in scenarios])

    batches = [
        scenarios[index : index + batch_size]
        for index in range(0, len(scenarios), batch_size)
    ]
    notify = progress_callback or (lambda _message: None)
    candidates_per_scenario = len(curve_variants()) * 2
    expected_candidate_rows = len(scenarios) * candidates_per_scenario
    expected_sample_rows = expected_candidate_rows * samples
    cumulative_outcomes = {name: 0 for name in _OUTCOME_COUNT_FIELDS}
    processed_scenarios = 0

    # Older resumable states did not persist outcome counters. Reconstruct each
    # missing batch once from its compact metric part, never from sample rows.
    for batch_index in sorted(completed_batches):
        if batch_index in stored_batch_outcomes:
            continue
        metric_part = batch_dir / f"metrics-{batch_index:06d}.csv"
        if not metric_part.exists():
            raise RuntimeError(
                f"Resume state marks batch {batch_index} complete, but {metric_part} is missing."
            )
        stored_batch_outcomes[batch_index] = _candidate_outcome_counts_from_csv(
            metric_part
        )

    def progress_message(batch_index: int, status: str) -> str:
        processed_candidates = cumulative_outcomes["total"]
        processed_samples = processed_candidates * samples
        percentage = 100.0 * processed_scenarios / len(scenarios)
        return (
            f"Phase 4 batch {batch_index + 1}/{len(batches)} {status}; "
            f"overall={percentage:.1f}%; "
            f"scenarios={processed_scenarios}/{len(scenarios)}; "
            f"candidates={processed_candidates}/{expected_candidate_rows} "
            f"(expected total={expected_candidate_rows}); "
            f"sample_rows={processed_samples}/{expected_sample_rows} "
            f"(expected total={expected_sample_rows}); "
            f"valid={cumulative_outcomes['valid']}; "
            "candidate_failures="
            f"reachability:{cumulative_outcomes['reachability_failure']},"
            f"collision:{cumulative_outcomes['collision_failure']},"
            f"other:{cumulative_outcomes['other_invalid']}"
        )

    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    for batch_index, batch in enumerate(batches):
        if batch_index in completed_batches:
            _add_outcome_counts(
                cumulative_outcomes, stored_batch_outcomes[batch_index]
            )
            processed_scenarios += len(batch)
            notify(progress_message(batch_index, "resumed/skipped"))
            continue
        notify(progress_message(batch_index, f"starting ({len(batch)} scenarios)"))
        scored_batch: list[dict[str, object]] = []
        winner_batch: list[dict[str, object]] = []
        sample_streams: dict[str, _SamplePartitionStream] = {}
        try:
            run_groups: Iterable[tuple[Phase4Run, ...]]
            if executor is None:
                run_groups = (
                    _evaluate_scenario_runs(scenario, samples, selected)
                    for scenario in batch
                )
            else:
                run_groups = executor.map(
                    _evaluate_scenario_runs,
                    batch,
                    repeat(samples),
                    repeat(selected),
                    chunksize=1,
                )
            for scenario_runs in run_groups:
                scenario_metrics: list[dict[str, object]] = []
                for run in scenario_runs:
                    scenario_metrics.append(run.metric_row)
                    split = str(run.metric_row["data_split"])
                    if split not in sample_streams:
                        sample_streams[split] = _SamplePartitionStream(
                            destination
                            / "trajectory_samples"
                            / split
                            / f"part-{batch_index:06d}",
                            selected_format,
                        )
                    sample_streams[split].write(run.sample_rows)
                scored_scenario = score_phase4_candidates(scenario_metrics)
                scored_batch.extend(scored_scenario)
                winner_batch.extend(select_scenario_winners(scored_scenario))
            for stream in sample_streams.values():
                stream.close(commit=True)
        except BaseException:
            for stream in sample_streams.values():
                stream.close(commit=False)
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
            raise
        _write_csv(batch_dir / f"metrics-{batch_index:06d}.csv", scored_batch)
        _write_csv(batch_dir / f"winners-{batch_index:06d}.csv", winner_batch)
        stored_batch_outcomes[batch_index] = _candidate_outcome_counts(scored_batch)
        completed_batches.add(batch_index)
        _save_json_atomic(
            state_path,
            {
                "signature": signature,
                "completed_batches": sorted(completed_batches),
                "total_batches": len(batches),
                "batch_outcome_counts": {
                    str(index): stored_batch_outcomes[index]
                    for index in sorted(stored_batch_outcomes)
                    if index in completed_batches
                },
            },
        )
        _add_outcome_counts(cumulative_outcomes, stored_batch_outcomes[batch_index])
        processed_scenarios += len(batch)
        notify(progress_message(batch_index, "completed"))

    if executor is not None:
        executor.shutdown()

    metric_parts = [batch_dir / f"metrics-{index:06d}.csv" for index in range(len(batches))]
    winner_parts = [batch_dir / f"winners-{index:06d}.csv" for index in range(len(batches))]
    missing = [str(path) for path in [*metric_parts, *winner_parts] if not path.exists()]
    if missing:
        raise RuntimeError(f"Dataset generation is incomplete; missing {missing[:3]}")
    _combine_csv_parts(metric_parts, destination / "trajectory_metrics.csv")
    _combine_csv_parts(winner_parts, destination / "scenario_winners.csv")
    _write_data_dictionary(destination / "data_dictionary.md")

    split_counts = {
        name: sum(item.data_split == name for item in scenarios)
        for name in ("train", "validation", "test", "challenge")
    }
    run_count = _count_csv_rows(destination / "trajectory_metrics.csv")
    winner_count = _count_csv_rows(destination / "scenario_winners.csv")
    tabular_files = [
        destination / "scenarios.csv",
        destination / "trajectory_metrics.csv",
        destination / "scenario_winners.csv",
        *sorted((destination / "trajectory_samples").rglob("*.parquet")),
        *sorted((destination / "trajectory_samples").rglob("*.csv.gz")),
    ]
    file_row_counts = {
        path.relative_to(destination).as_posix(): int(row_count)
        for path in tabular_files
        if (row_count := _tabular_row_count(path)) is not None
    }
    sample_row_counts_by_split: dict[str, int] = {}
    for relative_path, row_count in file_row_counts.items():
        parts = Path(relative_path).parts
        if len(parts) >= 3 and parts[0] == "trajectory_samples":
            split = parts[1]
            sample_row_counts_by_split[split] = (
                sample_row_counts_by_split.get(split, 0) + row_count
            )
    actual_sample_rows = sum(sample_row_counts_by_split.values())
    if actual_sample_rows != run_count * samples:
        raise RuntimeError(
            "Sample partition row count does not match candidate rows times samples: "
            f"{actual_sample_rows} != {run_count}*{samples}."
        )
    generation_distribution = _generation_distribution_metadata(scenarios)
    metadata: dict[str, object] = {
        "dataset_version": DATASET_VERSION,
        "code_version": CODE_VERSION,
        "sample_schema_version": SAMPLE_SCHEMA_VERSION,
        "equation_format_version": EQUATION_FORMAT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "scenario_count": scenario_count,
        "challenge_scenario_count": challenge_count,
        "total_scenario_count": len(scenarios),
        "curve_variant_count": len(curve_variants()),
        "elbow_modes": 2,
        "run_count": run_count,
        "winner_label_count": winner_count,
        "samples_per_path": samples,
        "trajectory_sample_row_count": actual_sample_rows,
        "sample_row_counts_by_split": dict(sorted(sample_row_counts_by_split.items())),
        "file_row_counts": dict(sorted(file_row_counts.items())),
        "sample_storage_format": selected_format,
        "sample_compression": "zstd" if selected_format == "parquet" else "gzip",
        "batch_size_scenarios": batch_size,
        "generation_workers": workers,
        "data_split": {
            "unit": "scenario_id",
            "policy_version": SPLIT_POLICY_VERSION,
            "policy": (
                "independent seeded permutation with exact 60/20/20 counts before "
                "candidate generation; challenge kept separate"
            ),
            "counts": split_counts,
        },
        "evaluation_config": asdict(selected),
        "time_law": "q(s)=10s^3-15s^4+6s^5, s=t/T",
        "integration_method": "composite trapezoid",
        "integration_grids": {
            "geometry": (
                "cosine-clustered u_i=(1-cos(pi*i/(N-1)))/2 with exact built-in "
                "curve derivatives for L and curvature"
            ),
            "joint_motion": "uniform t grid with q(t) for E_theta, A, J and O",
            "hard_constraint_validation": (
                "union of the saved minimum-jerk time grid and an additional "
                "uniform-u grid, each with N samples"
            ),
        },
        "curvature_regularization": {
            "formula": "K_epsilon=integral cross^2/(speed^2+epsilon^2)^(5/2) du",
            "epsilon": "curvature_speed_epsilon_ratio times endpoint chord length",
            "selection_field": "regularized_curvature_energy_inv_unit",
            "cusp_policy": "cycloid endpoints are marked path_regular=false; ordinary K is null",
        },
        "raw_objectives": RAW_OBJECTIVES,
        "balanced_weights": BALANCED_WEIGHTS,
        "tie_breaking": (
            "objective value, path length, curve_key, shape, elbow_mode, run_id"
        ),
        "tie_tolerance": {
            "relative": WINNER_TIE_REL_TOL,
            "absolute": WINNER_TIE_ABS_TOL,
        },
        "hard_constraints": [
            "all saved time-grid and uniform-u validation-grid samples reachable",
            "no link-obstacle collision on either validation grid",
        ],
        "obstacle_layouts": sorted({item.obstacle_layout for item in scenarios}),
        "scenario_types": sorted({item.robot.scenario_type for item in scenarios}),
        **generation_distribution,
        "limitations": [
            "Discrete samples do not prove continuous-time reachability or collision freedom.",
            "Joint-motion effort is not measured electrical energy and has no joule unit.",
            "High-order numerical derivatives are sensitive near singular configurations.",
            "Scenario-relative min-max scores depend on the candidate curve pool.",
            "Large datasets reduce but cannot guarantee elimination of overfitting.",
        ],
    }
    _save_json_atomic(destination / "experiment_metadata.json", metadata)

    manifest_files = [
        path
        for path in destination.rglob("*")
        if path.is_file()
        and ".batches" not in path.parts
        and path.name not in {"dataset_manifest.json", "generation_state.json"}
    ]
    manifest = {
        "dataset_version": DATASET_VERSION,
        "code_version": CODE_VERSION,
        "sample_schema_version": SAMPLE_SCHEMA_VERSION,
        "reproducibility_scope": (
            "Data files are deterministic for the same signature; "
            "experiment_metadata.json includes a generation timestamp."
        ),
        "files": [],
    }
    for path in sorted(manifest_files):
        relative_path = path.relative_to(destination).as_posix()
        entry: dict[str, object] = {
            "path": relative_path,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        if relative_path in file_row_counts:
            entry["row_count"] = file_row_counts[relative_path]
        manifest["files"].append(entry)
    _save_json_atomic(destination / "dataset_manifest.json", manifest)
    notify(
        f"Completed {len(scenarios)} scenarios, {run_count} candidate runs, "
        f"and {actual_sample_rows} sample rows; valid={cumulative_outcomes['valid']}; "
        "candidate_failures="
        f"reachability:{cumulative_outcomes['reachability_failure']},"
        f"collision:{cumulative_outcomes['collision_failure']},"
        f"other:{cumulative_outcomes['other_invalid']}"
    )
    return metadata
