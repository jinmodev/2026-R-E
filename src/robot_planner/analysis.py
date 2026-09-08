"""Scenario-level aggregation and statistics for poster/report analysis.

Rows from the same scenario are not independent.  This module first selects
one feasible candidate per scenario and curve family, then computes uncertainty
by resampling whole scenarios rather than individual trajectory rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CURVE_ORDER = ("line", "arc", "bezier", "hermite", "cycloid", "limacon")
CURVE_NAMES_KO = {
    "line": "직선",
    "arc": "원호",
    "bezier": "3차 베지어",
    "hermite": "3차 에르미트",
    "cycloid": "사이클로이드",
    "limacon": "리마송",
}


REQUIRED_METRIC_COLUMNS = {
    "run_id",
    "scenario_id",
    "scenario_type",
    "curve_key",
    "shape",
    "elbow_mode",
    "fully_reachable",
    "reachable_ratio",
    "curve_length",
    "path_to_chord_ratio",
    "joint_motion_l2_rad",
    "total_joint_rotation_deg",
    "min_singularity_index",
}


@dataclass(frozen=True, slots=True)
class AnalysisData:
    metrics: pd.DataFrame
    scenarios: pd.DataFrame
    best_candidates: pd.DataFrame
    curve_summary: pd.DataFrame


def load_phase2_data(
    metrics_path: str | Path,
    scenarios_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(metrics_path)
    scenarios = pd.read_csv(scenarios_path)
    missing = REQUIRED_METRIC_COLUMNS.difference(metrics.columns)
    if missing:
        raise ValueError(f"Missing metric columns: {sorted(missing)}")
    if metrics["run_id"].duplicated().any():
        raise ValueError("run_id must be unique.")
    if scenarios["scenario_id"].duplicated().any():
        raise ValueError("scenario_id must be unique in scenarios.csv.")
    if metrics["fully_reachable"].dtype == object:
        normalized = metrics["fully_reachable"].astype(str).str.strip().str.lower()
        valid = normalized.isin({"true", "false"})
        if not valid.all():
            bad_values = sorted(metrics.loc[~valid, "fully_reachable"].astype(str).unique())
            raise ValueError(f"Invalid fully_reachable values: {bad_values}")
        metrics["fully_reachable"] = normalized.eq("true")
    else:
        metrics["fully_reachable"] = metrics["fully_reachable"].astype(bool)
    return metrics, scenarios


def select_best_candidates(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return one row for each scenario and curve family.

    A family is feasible if any tested shape/elbow candidate is fully
    reachable.  Among feasible candidates, the row with the smallest
    joint-space path length is selected.  Ties prefer the shorter endpoint
    path and then the larger singularity index.
    """

    missing = REQUIRED_METRIC_COLUMNS.difference(metrics.columns)
    if missing:
        raise ValueError(f"Missing metric columns: {sorted(missing)}")

    group_columns = ["scenario_id", "curve_key"]
    base = (
        metrics.groupby(group_columns, as_index=False, observed=True)
        .agg(
            scenario_type=("scenario_type", "first"),
            family_feasible=("fully_reachable", "any"),
            max_reachable_ratio=("reachable_ratio", "max"),
            candidate_count=("run_id", "size"),
        )
    )

    feasible = metrics.loc[metrics["fully_reachable"]].copy()
    feasible = feasible.sort_values(
        [
            "scenario_id",
            "curve_key",
            "joint_motion_l2_rad",
            "curve_length",
            "min_singularity_index",
            "run_id",
        ],
        ascending=[True, True, True, True, False, True],
        kind="mergesort",
    )
    chosen = feasible.drop_duplicates(group_columns, keep="first").copy()
    chosen_columns = [
        "scenario_id",
        "curve_key",
        "run_id",
        "shape",
        "elbow_mode",
        "curve_length",
        "path_to_chord_ratio",
        "joint_motion_l2_rad",
        "total_joint_rotation_deg",
        "min_singularity_index",
        "curve_parameters_json",
    ]
    chosen_columns = [column for column in chosen_columns if column in chosen.columns]
    best = base.merge(chosen[chosen_columns], on=group_columns, how="left", validate="one_to_one")

    expected = metrics["scenario_id"].nunique() * metrics["curve_key"].nunique()
    if len(best) != expected:
        raise AssertionError(f"Expected {expected} scenario-family rows, got {len(best)}.")
    return best


def _bootstrap_rate_interval(
    values: np.ndarray,
    *,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("values must be a non-empty one-dimensional array.")
    sample_indexes = rng.integers(0, len(values), size=(iterations, len(values)))
    estimates = values[sample_indexes].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def summarize_curves(
    best_candidates: pd.DataFrame,
    *,
    bootstrap_iterations: int = 4000,
    seed: int = 20260825,
) -> pd.DataFrame:
    if bootstrap_iterations < 200:
        raise ValueError("bootstrap_iterations must be at least 200.")
    rng = np.random.default_rng(seed)
    scenario_count = best_candidates["scenario_id"].nunique()

    feasible = best_candidates.loc[best_candidates["family_feasible"]].copy()
    winner_rows = feasible.loc[
        feasible.groupby("scenario_id", observed=True)["joint_motion_l2_rad"].idxmin()
    ]
    winner_counts = winner_rows["curve_key"].value_counts()

    rows: list[dict[str, object]] = []
    for curve_key in CURVE_ORDER:
        family = best_candidates.loc[best_candidates["curve_key"] == curve_key].copy()
        if family.empty:
            raise ValueError(f"No rows found for curve family {curve_key!r}.")
        flags = family["family_feasible"].astype(float).to_numpy()
        low, high = _bootstrap_rate_interval(
            flags,
            iterations=bootstrap_iterations,
            rng=rng,
        )
        successful = family.loc[family["family_feasible"]]
        rows.append(
            {
                "curve_key": curve_key,
                "curve_name_ko": CURVE_NAMES_KO[curve_key],
                "scenario_count": len(family),
                "feasible_count": int(family["family_feasible"].sum()),
                "feasible_rate": float(flags.mean()),
                "feasible_ci_low": low,
                "feasible_ci_high": high,
                "median_path_ratio": float(successful["path_to_chord_ratio"].median()),
                "median_joint_motion_rad": float(successful["joint_motion_l2_rad"].median()),
                "median_min_singularity_index": float(
                    successful["min_singularity_index"].median()
                ),
                "joint_motion_wins": int(winner_counts.get(curve_key, 0)),
                "joint_motion_win_rate": float(winner_counts.get(curve_key, 0) / scenario_count),
            }
        )
    return pd.DataFrame(rows)


def scenario_type_feasibility(best_candidates: pd.DataFrame) -> pd.DataFrame:
    table = best_candidates.pivot_table(
        index="scenario_type",
        columns="curve_key",
        values="family_feasible",
        aggfunc="mean",
        observed=True,
    )
    return table.reindex(columns=CURVE_ORDER)


def paired_with_line(best_candidates: pd.DataFrame) -> pd.DataFrame:
    feasible = best_candidates.loc[best_candidates["family_feasible"]].copy()
    line = feasible.loc[
        feasible["curve_key"] == "line",
        ["scenario_id", "joint_motion_l2_rad"],
    ].rename(columns={"joint_motion_l2_rad": "line_joint_motion_rad"})
    paired = feasible.loc[feasible["curve_key"] != "line"].merge(
        line,
        on="scenario_id",
        how="inner",
        validate="many_to_one",
    )
    paired["joint_motion_ratio_to_line"] = (
        paired["joint_motion_l2_rad"] / paired["line_joint_motion_rad"]
    )
    return paired


def select_representative_scenario(
    metrics: pd.DataFrame,
    best_candidates: pd.DataFrame,
    required_curves: Iterable[str] = ("arc", "bezier"),
) -> str:
    line_rows = metrics.loc[metrics["curve_key"] == "line"]
    line_status = (
        line_rows.groupby("scenario_id", as_index=False, observed=True)
        .agg(
            line_feasible=("fully_reachable", "any"),
            line_reachable_ratio=("reachable_ratio", "max"),
        )
    )
    availability = best_candidates.pivot_table(
        index="scenario_id",
        columns="curve_key",
        values="family_feasible",
        aggfunc="first",
        fill_value=False,
        observed=True,
    )
    availability = availability.reindex(columns=CURVE_ORDER, fill_value=False)
    availability["alternative_count"] = availability.drop(columns="line").sum(axis=1)
    candidates = line_status.loc[~line_status["line_feasible"]].set_index("scenario_id")
    candidates = candidates.join(availability, how="inner")
    for curve in required_curves:
        candidates = candidates.loc[candidates[curve]]
    if candidates.empty:
        raise ValueError("No representative scenario satisfies the requested curve constraints.")
    candidates["selection_score"] = (
        1.0 - candidates["line_reachable_ratio"]
        + 0.12 * candidates["alternative_count"]
    )
    return str(candidates["selection_score"].idxmax())


def build_analysis_data(
    metrics_path: str | Path,
    scenarios_path: str | Path,
    *,
    bootstrap_iterations: int = 4000,
    seed: int = 20260825,
) -> AnalysisData:
    metrics, scenarios = load_phase2_data(metrics_path, scenarios_path)
    best = select_best_candidates(metrics)
    summary = summarize_curves(
        best,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )
    return AnalysisData(metrics=metrics, scenarios=scenarios, best_candidates=best, curve_summary=summary)
