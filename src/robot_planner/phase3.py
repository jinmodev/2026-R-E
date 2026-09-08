"""Phase-3 obstacle and multi-objective experiment pipeline."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Iterable, Iterator, Mapping

from .curves import make_curve
from .evaluation import CircleObstacle, EvaluationConfig, evaluate_trajectory
from .experiments import (
    DEFAULT_SAMPLES,
    DEFAULT_SEED,
    CurveVariant,
    Scenario,
    curve_variants,
    generate_scenarios,
)
from .kinematics import PlanarArm2R


RAW_OBJECTIVES = {
    "distance": "curve_length_m",
    "energy": "model_electrical_energy_j",
    "jerk": "joint_jerk_effort_rad2_per_s5",
    "curvature": "curvature_energy_inv_m",
    "singularity": "singularity_penalty_s",
    "safety": "clearance_penalty_s_per_m2",
}

DEFAULT_SCORE_PROFILES: dict[str, dict[str, float]] = {
    "balanced": {
        "distance": 0.20,
        "energy": 0.25,
        "jerk": 0.20,
        "curvature": 0.10,
        "singularity": 0.10,
        "safety": 0.15,
    },
    "shortest": {
        "distance": 0.55,
        "energy": 0.10,
        "jerk": 0.10,
        "curvature": 0.05,
        "singularity": 0.10,
        "safety": 0.10,
    },
    "energy_saving": {
        "distance": 0.10,
        "energy": 0.55,
        "jerk": 0.15,
        "curvature": 0.05,
        "singularity": 0.05,
        "safety": 0.10,
    },
    "smooth_precision": {
        "distance": 0.10,
        "energy": 0.10,
        "jerk": 0.40,
        "curvature": 0.20,
        "singularity": 0.10,
        "safety": 0.10,
    },
    "safety_first": {
        "distance": 0.10,
        "energy": 0.10,
        "jerk": 0.10,
        "curvature": 0.05,
        "singularity": 0.15,
        "safety": 0.50,
    },
}


@dataclass(frozen=True, slots=True)
class Phase3Scenario:
    robot: Scenario
    obstacles: tuple[CircleObstacle, ...]
    obstacle_layout: str
    data_split: str

    def as_row(self) -> dict[str, object]:
        return {
            **self.robot.as_row(),
            "data_split": self.data_split,
            "obstacle_layout": self.obstacle_layout,
            "obstacle_count": len(self.obstacles),
            "obstacles_json": json.dumps(
                [item.as_dict() for item in self.obstacles],
                ensure_ascii=False,
                sort_keys=True,
            ),
        }


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(right[0] - left[0], right[1] - left[1])


def _generate_obstacles(
    scenario: Scenario,
    rng: random.Random,
    *,
    count: int,
    offset_range: tuple[float, float],
    radius_scale_range: tuple[float, float],
    chord_radius_cap: float,
) -> tuple[CircleObstacle, ...]:
    start = scenario.start_x, scenario.start_y
    goal = scenario.goal_x, scenario.goal_y
    chord_x = goal[0] - start[0]
    chord_y = goal[1] - start[1]
    chord = math.hypot(chord_x, chord_y)
    unit_x, unit_y = chord_x / chord, chord_y / chord
    normal_x, normal_y = -unit_y, unit_x
    obstacles: list[CircleObstacle] = []
    for obstacle_index in range(count):
        radius = min(
            rng.uniform(*radius_scale_range) * scenario.max_reach,
            chord_radius_cap * chord,
        )
        if count == 1:
            progress = rng.uniform(0.38, 0.62)
        else:
            progress = (obstacle_index + 1) / (count + 1) + rng.uniform(-0.04, 0.04)
        side = rng.choice((-1.0, 1.0))
        offset = side * rng.uniform(*offset_range) * chord
        center = (
            start[0] + progress * chord_x + offset * normal_x,
            start[1] + progress * chord_y + offset * normal_y,
        )

        base = scenario.base_x, scenario.base_y
        required_base_clearance = radius + 0.10
        base_distance = _distance(center, base)
        if base_distance < required_base_clearance:
            shift = required_base_clearance - base_distance + 0.04
            center = (
                center[0] + side * shift * normal_x,
                center[1] + side * shift * normal_y,
            )

        obstacles.append(
            CircleObstacle(
                obstacle_id=f"{scenario.scenario_id}_O{obstacle_index + 1}",
                center_x=center[0],
                center_y=center[1],
                radius_units=radius,
            )
        )
    return tuple(obstacles)


def generate_phase3_scenarios(
    count: int,
    *,
    seed: int = DEFAULT_SEED,
) -> list[Phase3Scenario]:
    robots = generate_scenarios(count, seed)
    rng = random.Random(seed + 300_003)
    scenarios: list[Phase3Scenario] = []
    for index, robot in enumerate(robots):
        layout_index = index % 4
        if layout_index == 0:
            layout = "no_obstacle"
            obstacles: tuple[CircleObstacle, ...] = ()
        elif layout_index in {1, 2}:
            layout = "chord_blocker"
            obstacles = _generate_obstacles(
                robot,
                rng,
                count=1,
                offset_range=(-0.02, 0.065),
                radius_scale_range=(0.03, 0.065),
                chord_radius_cap=0.095,
            )
        else:
            layout = "double_offset"
            obstacles = _generate_obstacles(
                robot,
                rng,
                count=2,
                offset_range=(0.08, 0.18),
                radius_scale_range=(0.02, 0.045),
                chord_radius_cap=0.07,
            )
        split_index = index % 5
        data_split = (
            "train" if split_index < 3 else "validation" if split_index == 3 else "test"
        )
        scenarios.append(Phase3Scenario(robot, obstacles, layout, data_split))
    return scenarios


def evaluate_phase3_variant(
    scenario: Phase3Scenario,
    variant: CurveVariant,
    elbow_mode: str,
    *,
    samples: int = DEFAULT_SAMPLES,
    config: EvaluationConfig | None = None,
) -> dict[str, object]:
    selected_config = config or EvaluationConfig()
    robot = scenario.robot
    arm = PlanarArm2R(robot.link1, robot.link2, (robot.base_x, robot.base_y))
    curve = make_curve(
        variant.curve_key,
        (robot.start_x, robot.start_y),
        (robot.goal_x, robot.goal_y),
        shape=variant.shape,
        samples=samples,
    )
    evaluation = evaluate_trajectory(
        arm,
        curve,
        elbow_mode,
        obstacles=scenario.obstacles,
        config=selected_config,
    )
    return {
        "run_id": f"{robot.scenario_id}_{variant.variant_id}_{elbow_mode}",
        "scenario_id": robot.scenario_id,
        "scenario_type": robot.scenario_type,
        "seed": robot.seed,
        "link1": robot.link1,
        "link2": robot.link2,
        "link_ratio": robot.link_ratio,
        "link1_mass_kg": (
            selected_config.link_linear_density_kg_per_m
            * selected_config.workspace_unit_m
            * robot.link1
        ),
        "link2_mass_kg": (
            selected_config.link_linear_density_kg_per_m
            * selected_config.workspace_unit_m
            * robot.link2
        ),
        "start_x": robot.start_x,
        "start_y": robot.start_y,
        "goal_x": robot.goal_x,
        "goal_y": robot.goal_y,
        "start_radius": robot.start_radius,
        "goal_radius": robot.goal_radius,
        "chord_length": robot.chord_length,
        "data_split": scenario.data_split,
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
        "samples": samples,
        "curve_parameters_json": json.dumps(
            dict(curve.parameters), ensure_ascii=False, sort_keys=True
        ),
        **evaluation.as_dict(),
    }


def run_phase3_experiments(
    scenarios: Iterable[Phase3Scenario],
    *,
    samples: int = DEFAULT_SAMPLES,
    config: EvaluationConfig | None = None,
) -> Iterator[dict[str, object]]:
    for scenario in scenarios:
        for variant in curve_variants():
            for elbow_mode in ("up", "down"):
                yield evaluate_phase3_variant(
                    scenario,
                    variant,
                    elbow_mode,
                    samples=samples,
                    config=config,
                )


def _validate_profiles(profiles: Mapping[str, Mapping[str, float]]) -> None:
    expected = set(RAW_OBJECTIVES)
    for name, weights in profiles.items():
        if set(weights) != expected:
            raise ValueError(f"Profile {name!r} must define exactly {sorted(expected)}.")
        if any(value < 0.0 for value in weights.values()):
            raise ValueError(f"Profile {name!r} cannot contain negative weights.")
        if sum(weights.values()) <= 0.0:
            raise ValueError(f"Profile {name!r} must have a positive total weight.")


def score_scenario_candidates(
    records: Iterable[dict[str, object]],
    *,
    profiles: Mapping[str, Mapping[str, float]] = DEFAULT_SCORE_PROFILES,
) -> list[dict[str, object]]:
    """Add scenario-relative normalized objectives, costs, and rewards.

    Normalization is performed within a scenario because different arm sizes
    and endpoint separations have different physical scales.  Invalid paths
    receive a hard cost of 1,000,000 and zero reward.
    """

    _validate_profiles(profiles)
    rows = [dict(record) for record in records]
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["scenario_id"]), []).append(row)

    for candidates in grouped.values():
        valid = [row for row in candidates if bool(row.get("valid_trajectory"))]
        bounds: dict[str, tuple[float, float]] = {}
        for objective, field in RAW_OBJECTIVES.items():
            values = [float(row[field]) for row in valid if row.get(field) is not None]
            if len(values) != len(valid):
                raise ValueError(f"Valid candidates are missing objective {objective!r}.")
            bounds[objective] = (min(values), max(values)) if values else (0.0, 0.0)

        for row in candidates:
            row["selection_valid"] = bool(row.get("valid_trajectory"))
            if not row["selection_valid"]:
                for objective in RAW_OBJECTIVES:
                    row[f"normalized_{objective}"] = None
                for profile_name in profiles:
                    row[f"{profile_name}_cost_0_100"] = 1_000_000.0
                    row[f"{profile_name}_reward_0_100"] = 0.0
                continue

            normalized: dict[str, float] = {}
            for objective, field in RAW_OBJECTIVES.items():
                low, high = bounds[objective]
                value = float(row[field])
                scaled = 0.0 if high - low <= 1e-15 else (value - low) / (high - low)
                normalized[objective] = max(0.0, min(1.0, scaled))
                row[f"normalized_{objective}"] = normalized[objective]

            for profile_name, weights in profiles.items():
                total_weight = sum(weights.values())
                cost = 100.0 * sum(
                    weights[objective] * normalized[objective]
                    for objective in RAW_OBJECTIVES
                ) / total_weight
                row[f"{profile_name}_cost_0_100"] = cost
                row[f"{profile_name}_reward_0_100"] = 100.0 - cost
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty CSV dataset.")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_phase3_dataset(
    output_dir: str | Path,
    *,
    scenario_count: int = 240,
    seed: int = DEFAULT_SEED,
    samples: int = DEFAULT_SAMPLES,
    config: EvaluationConfig | None = None,
) -> dict[str, object]:
    selected_config = config or EvaluationConfig()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    scenarios = generate_phase3_scenarios(scenario_count, seed=seed)
    raw_records = list(
        run_phase3_experiments(
            scenarios,
            samples=samples,
            config=selected_config,
        )
    )
    metric_rows = score_scenario_candidates(raw_records)
    _write_csv(destination / "scenarios.csv", [item.as_row() for item in scenarios])
    _write_csv(destination / "trajectory_metrics.csv", metric_rows)

    metadata = {
        "dataset_version": "phase3-multicriteria-v1",
        "seed": seed,
        "scenario_count": scenario_count,
        "curve_variant_count": len(curve_variants()),
        "elbow_modes": 2,
        "run_count": len(metric_rows),
        "samples_per_path": samples,
        "data_split": {
            "unit": "scenario_id",
            "policy": "deterministic 60/20/20 train/validation/test split",
            "counts": {
                name: sum(item.data_split == name for item in scenarios)
                for name in ("train", "validation", "test")
            },
        },
        "obstacle_model": (
            "balanced deterministic layouts: no_obstacle, chord_blocker, "
            "and double_offset circular obstacles"
        ),
        "evaluation_config": asdict(selected_config),
        "raw_objectives": RAW_OBJECTIVES,
        "score_profiles": DEFAULT_SCORE_PROFILES,
        "hard_constraints": ["full path reachability", "full-link obstacle collision"],
        "limitations": [
            "Energy is a rigid-body/direct-drive model estimate, not a physical measurement.",
            "All candidates share a minimum-jerk time law and the same duration.",
            "Circular obstacles and link capsules are checked at discrete samples.",
            "Scenario-relative min-max scores change if the candidate pool changes.",
        ],
    }
    (destination / "experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata
