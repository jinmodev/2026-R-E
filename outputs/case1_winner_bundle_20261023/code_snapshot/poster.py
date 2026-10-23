"""Poster-ready visual outputs for the phase-2 baseline experiment.

The figures intentionally describe only the current evidence: a 2-D, fixed-base
two-link arm without obstacles or a motor dynamics model.  They are designed to
be reproducible from the committed CSV files and suitable for a Korean R&E
poster after exporting both SVG and 300 dpi PNG versions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager, patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from .analysis import (
    AnalysisData,
    CURVE_NAMES_KO,
    CURVE_ORDER,
    paired_with_line,
    scenario_type_feasibility,
    select_representative_scenario,
)
from .curves import make_curve
from .kinematics import PlanarArm2R


CURVE_COLORS = {
    "line": "#475569",
    "arc": "#2563EB",
    "bezier": "#0F9D8A",
    "hermite": "#7C3AED",
    "cycloid": "#EA8C00",
    "limacon": "#D94682",
}
SCENARIO_NAMES_KO = {
    "local_move": "근거리 이동",
    "cross_inner": "내부 공백 횡단",
    "boundary_sweep": "경계 부근 이동",
}


@dataclass(frozen=True, slots=True)
class PosterOutputs:
    output_dir: Path
    generated_files: tuple[Path, ...]
    representative_scenario_id: str


def _configure_style() -> None:
    candidates = (
        Path(r"C:\Windows\Fonts\NotoSansKR-VF.ttf"),
        Path(r"C:\Windows\Fonts\malgun.ttf"),
    )
    family = "DejaVu Sans"
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            family = font_manager.FontProperties(fname=str(path)).get_name()
            break
    mpl.rcParams.update(
        {
            "font.family": family,
            "axes.unicode_minus": False,
            "svg.fonttype": "path",
            "svg.hashsalt": "phase2-poster-v1",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#334155",
            "axes.titlecolor": "#0F172A",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "text.color": "#1E293B",
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": False,
        }
    )


def _clean_axes(axis: plt.Axes, *, grid_axis: str | None = "y") -> None:
    axis.spines[["top", "right"]].set_visible(False)
    if grid_axis:
        axis.grid(axis=grid_axis, color="#E2E8F0", linewidth=0.8, alpha=0.85)
        axis.set_axisbelow(True)


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    png_path = output_dir / f"{stem}.png"
    svg_path = output_dir / f"{stem}.svg"
    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(
        svg_path,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Date": None},
    )
    plt.close(figure)
    return [png_path, svg_path]


def _subtitle(figure: plt.Figure, text: str) -> None:
    figure.text(0.5, 0.925, text, ha="center", va="top", fontsize=9.5, color="#64748B")


def _method_model_figure() -> plt.Figure:
    figure, (axis_arm, axis_eq) = plt.subplots(
        1,
        2,
        figsize=(12.6, 5.6),
        gridspec_kw={"width_ratios": [1.05, 1.3]},
    )
    figure.suptitle("2관절 로봇팔의 경로 비교 모형", fontsize=18, fontweight="bold", y=0.98)
    _subtitle(figure, "끝점 경로를 관절공간으로 변환해 도달 가능성과 이동량을 비교한다")

    arm = PlanarArm2R(2.8, 2.1)
    theta1, theta2 = np.deg2rad(38.0), np.deg2rad(-73.0)
    pose = arm.forward(float(theta1), float(theta2))
    base = np.asarray(pose.base)
    elbow = np.asarray(pose.elbow)
    end = np.asarray(pose.end)

    axis_arm.add_patch(patches.Circle(base, arm.max_reach, fill=False, ec="#94A3B8", ls="--"))
    axis_arm.add_patch(
        patches.Circle(base, arm.min_reach, fc="#FEE2E2", ec="#F87171", alpha=0.7)
    )
    axis_arm.plot([base[0], elbow[0]], [base[1], elbow[1]], color="#2563EB", lw=8, solid_capstyle="round")
    axis_arm.plot([elbow[0], end[0]], [elbow[1], end[1]], color="#0F9D8A", lw=8, solid_capstyle="round")
    axis_arm.scatter(*base, s=100, c="#0F172A", zorder=5)
    axis_arm.scatter(*elbow, s=130, c="white", edgecolors="#0F172A", linewidths=2.2, zorder=5)
    axis_arm.scatter(*end, s=100, c="#D94682", edgecolors="white", linewidths=1.5, zorder=5)

    arc1 = patches.Arc(base, 1.45, 1.45, theta1=0, theta2=38, ec="#334155", lw=1.6)
    axis_arm.add_patch(arc1)
    axis_arm.text(0.82, 0.22, r"$\theta_1$", fontsize=13)
    axis_arm.text(*(0.52 * (base + elbow) + np.array([-0.05, 0.2])), r"$l_1$", fontsize=13)
    axis_arm.text(*(0.52 * (elbow + end) + np.array([0.05, 0.22])), r"$l_2$", fontsize=13)
    axis_arm.text(end[0] + 0.12, end[1] + 0.1, r"$p(t)$", fontsize=13, color="#BE185D")
    axis_arm.text(0, -0.55, "도달 불가 내부 공백", ha="center", color="#B91C1C", fontsize=10)
    axis_arm.set_xlim(-5.3, 5.3)
    axis_arm.set_ylim(-5.3, 5.3)
    axis_arm.set_aspect("equal")
    axis_arm.set_title("2R 기구학과 작업공간")
    axis_arm.set_xlabel("x")
    axis_arm.set_ylabel("y")
    _clean_axes(axis_arm, grid_axis=None)

    axis_eq.axis("off")
    equations = [
        (
            "① 순기구학",
            "$x=l_1\\cos\\theta_1+l_2\\cos(\\theta_1+\\theta_2)$\n"
            "$y=l_1\\sin\\theta_1+l_2\\sin(\\theta_1+\\theta_2)$",
        ),
        ("② 도달 가능 조건", r"$|l_1-l_2|\leq \|p(t)-b\|\leq l_1+l_2$"),
        ("③ 끝점 경로 길이", r"$L=\int_0^1\|p'(t)\|\,dt$"),
        ("④ 관절 이동량", r"$Q\approx\sum_{i=0}^{N-2}\sqrt{(\Delta\theta_{1,i})^2+(\Delta\theta_{2,i})^2}$"),
    ]
    y_positions = [0.85, 0.60, 0.38, 0.16]
    for (label, equation), y in zip(equations, y_positions, strict=True):
        axis_eq.text(0.0, y + 0.09, label, fontsize=11, weight="bold", color="#334155", transform=axis_eq.transAxes)
        axis_eq.text(0.03, y, equation, fontsize=13.5, transform=axis_eq.transAxes, va="center")
    axis_eq.text(
        0.0,
        -0.04,
        "현재 2단계 범위: 장애물·충돌·모터 동역학 없음  |  Q는 실제 에너지가 아닌 관절공간 경로 길이",
        fontsize=9.5,
        color="#B45309",
        transform=axis_eq.transAxes,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#FFFBEB", "edgecolor": "#FCD34D"},
    )
    figure.subplots_adjust(top=0.84, wspace=0.22)
    return figure


def _feasibility_figure(summary: pd.DataFrame) -> plt.Figure:
    ordered = summary.set_index("curve_key").loc[list(CURVE_ORDER)].reset_index()
    scenario_count = int(ordered["scenario_count"].max())
    figure, axis = plt.subplots(figsize=(10.8, 6.0))
    x = np.arange(len(ordered))
    rates = ordered["feasible_rate"].to_numpy()
    errors = np.vstack(
        [rates - ordered["feasible_ci_low"].to_numpy(), ordered["feasible_ci_high"].to_numpy() - rates]
    )
    bars = axis.bar(
        x,
        rates * 100,
        color=[CURVE_COLORS[key] for key in ordered["curve_key"]],
        width=0.68,
        yerr=errors * 100,
        capsize=4,
        error_kw={"ecolor": "#334155", "elinewidth": 1.2},
    )
    for bar, rate, count in zip(bars, rates, ordered["feasible_count"], strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3.0,
            f"{rate * 100:.1f}%\n({int(count)}/{scenario_count})",
            ha="center",
            va="bottom",
            fontsize=9.5,
            weight="bold",
        )
    axis.set_ylim(0, 112)
    axis.set_xticks(x, ordered["curve_name_ko"])
    axis.set_ylabel("도달 가능한 시나리오 비율 (%)")
    axis.set_title("곡선 계열별 경로 생성 성공률", pad=34)
    axis.text(
        0.0,
        1.015,
        f"오차막대: 시나리오 단위 부트스트랩 95% 신뢰구간 (n={scenario_count})",
        transform=axis.transAxes,
        color="#64748B",
        fontsize=9.5,
    )
    _clean_axes(axis)
    figure.tight_layout()
    return figure


def _tradeoff_figure(best: pd.DataFrame) -> plt.Figure:
    feasible = best.loc[best["family_feasible"]].copy()
    figure, axis = plt.subplots(figsize=(10.8, 6.5))
    for curve_key in CURVE_ORDER:
        family = feasible.loc[feasible["curve_key"] == curve_key]
        axis.scatter(
            family["path_to_chord_ratio"],
            family["joint_motion_l2_rad"],
            s=18,
            alpha=0.34,
            color=CURVE_COLORS[curve_key],
            edgecolors="none",
            label=CURVE_NAMES_KO[curve_key],
        )
        axis.scatter(
            family["path_to_chord_ratio"].median(),
            family["joint_motion_l2_rad"].median(),
            marker="D",
            s=88,
            color=CURVE_COLORS[curve_key],
            edgecolors="white",
            linewidths=1.4,
            zorder=5,
        )
    axis.set_xlim(0.94, min(4.25, float(feasible["path_to_chord_ratio"].quantile(0.995)) + 0.15))
    axis.set_xlabel("끝점 경로 길이 / 시작–끝 직선거리")
    axis.set_ylabel(r"관절 이동량 $Q$ (rad)")
    axis.set_title("짧은 끝점 경로가 항상 작은 관절 이동을 뜻하지는 않는다", pad=34)
    axis.text(
        0.0,
        1.015,
        "점: 각 시나리오에서 선택된 최적 후보  |  ◆: 곡선 계열 중앙값",
        transform=axis.transAxes,
        color="#64748B",
        fontsize=9.5,
    )
    axis.legend(ncol=3, loc="upper right")
    _clean_axes(axis, grid_axis="both")
    figure.tight_layout()
    return figure


def _scenario_heatmap_figure(best: pd.DataFrame) -> plt.Figure:
    table = scenario_type_feasibility(best).reindex(
        index=["local_move", "cross_inner", "boundary_sweep"]
    )
    values = table.to_numpy(dtype=float) * 100
    cmap = LinearSegmentedColormap.from_list("phase2", ["#FEF3C7", "#93C5FD", "#0F766E"])
    figure, axis = plt.subplots(figsize=(10.6, 4.7))
    image = axis.imshow(values, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            color = "white" if values[row, column] >= 62 else "#0F172A"
            axis.text(column, row, f"{values[row, column]:.1f}%", ha="center", va="center", color=color, weight="bold")
    axis.set_xticks(np.arange(len(CURVE_ORDER)), [CURVE_NAMES_KO[key] for key in CURVE_ORDER])
    scenario_counts = best.groupby("scenario_type", observed=True)["scenario_id"].nunique()
    axis.set_yticks(
        np.arange(len(table.index)),
        [
            f"{SCENARIO_NAMES_KO.get(key, key)} (n={int(scenario_counts.get(key, 0))})"
            for key in table.index
        ],
    )
    axis.set_title("상황 유형에 따라 유리한 곡선이 달라진다")
    axis.set_xlabel("곡선 계열")
    axis.set_ylabel("시나리오 유형")
    colorbar = figure.colorbar(image, ax=axis, pad=0.025, shrink=0.85)
    colorbar.set_label("도달 가능률 (%)")
    figure.tight_layout()
    return figure


def _paired_ratio_figure(best: pd.DataFrame, *, seed: int) -> plt.Figure:
    paired = paired_with_line(best)
    keys = [key for key in CURVE_ORDER if key != "line"]
    samples = [
        paired.loc[paired["curve_key"] == key, "joint_motion_ratio_to_line"].to_numpy()
        for key in keys
    ]
    figure, axis = plt.subplots(figsize=(10.8, 6.1))
    box = axis.boxplot(
        samples,
        positions=np.arange(1, len(keys) + 1),
        widths=0.54,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 2.0},
        whiskerprops={"color": "#64748B"},
        capprops={"color": "#64748B"},
    )
    for patch, key in zip(box["boxes"], keys, strict=True):
        patch.set_facecolor(CURVE_COLORS[key])
        patch.set_alpha(0.9)
        patch.set_edgecolor("none")
    rng = np.random.default_rng(seed)
    for position, (key, values) in enumerate(zip(keys, samples, strict=True), start=1):
        jitter = rng.normal(position, 0.055, len(values))
        axis.scatter(jitter, values, s=9, alpha=0.13, color=CURVE_COLORS[key], edgecolors="none")
        axis.text(position, 8.25, f"n={len(values)}", ha="center", va="bottom", fontsize=8.5, color="#64748B")
    axis.axhline(1.0, color="#0F172A", ls="--", lw=1.2)
    axis.text(5.45, 1.04, "직선과 동일", ha="right", va="bottom", fontsize=9, color="#334155")
    axis.set_yscale("log", base=2)
    axis.set_ylim(0.22, 10.0)
    ticks = [0.25, 0.5, 1, 2, 4, 8]
    axis.set_yticks(ticks, ["0.25×", "0.5×", "1×", "2×", "4×", "8×"])
    axis.set_xticks(np.arange(1, len(keys) + 1), [CURVE_NAMES_KO[key] for key in keys])
    axis.set_ylabel("관절 이동량 비율  " + r"$Q_c/Q_{line}$")
    axis.set_title("직선도 가능한 경우의 관절 이동량 상대 비교", pad=34)
    axis.text(
        0.0,
        1.015,
        "1보다 작으면 해당 곡선이 직선보다 관절 이동량이 작다 (짝지은 시나리오만 비교)",
        transform=axis.transAxes,
        color="#64748B",
        fontsize=9.5,
    )
    _clean_axes(axis)
    figure.tight_layout()
    return figure


def _plot_masked_segments(
    axis: plt.Axes,
    points: np.ndarray,
    mask: np.ndarray,
    **plot_kwargs: object,
) -> None:
    start = None
    for index, enabled in enumerate(mask):
        if enabled and start is None:
            start = index
        if start is not None and (not enabled or index == len(mask) - 1):
            stop = index if not enabled else index + 1
            if stop - start >= 2:
                axis.plot(points[start:stop, 0], points[start:stop, 1], **plot_kwargs)
            start = None


def _representative_figure(data: AnalysisData, scenario_id: str) -> plt.Figure:
    scenario = data.scenarios.set_index("scenario_id").loc[scenario_id]
    selected = data.best_candidates.loc[
        (data.best_candidates["scenario_id"] == scenario_id)
        & (data.best_candidates["family_feasible"])
    ].set_index("curve_key")
    start = float(scenario["start_x"]), float(scenario["start_y"])
    goal = float(scenario["goal_x"]), float(scenario["goal_y"])
    arm = PlanarArm2R(
        float(scenario["link1"]),
        float(scenario["link2"]),
        base=(float(scenario["base_x"]), float(scenario["base_y"])),
    )

    figure, axis = plt.subplots(figsize=(8.4, 7.4))
    axis.add_patch(patches.Circle(arm.base, arm.max_reach, fill=False, ec="#94A3B8", ls="--", lw=1.3))
    axis.add_patch(
        patches.Circle(arm.base, arm.min_reach, fc="#FEE2E2", ec="#F87171", alpha=0.72, lw=1.2)
    )

    line_path = make_curve("line", start, goal, samples=241)
    line_points = np.asarray(line_path.points)
    reachable = np.asarray([arm.is_reachable(tuple(point)) for point in line_points])
    _plot_masked_segments(axis, line_points, reachable, color="#64748B", lw=2.3, alpha=0.8, label="직선: 도달 가능 구간")
    _plot_masked_segments(axis, line_points, ~reachable, color="#DC2626", lw=3.0, ls="--", label="직선: 도달 불가 구간")

    alternatives = [key for key in ("arc", "bezier", "hermite", "cycloid") if key in selected.index]
    drawn_paths: dict[str, np.ndarray] = {}
    for key in alternatives[:3]:
        row = selected.loc[key]
        path = make_curve(key, start, goal, shape=float(row["shape"]), samples=241)
        points = np.asarray(path.points)
        drawn_paths[key] = points
        axis.plot(points[:, 0], points[:, 1], color=CURVE_COLORS[key], lw=3.0, label=f"{CURVE_NAMES_KO[key]} (Q={row['joint_motion_l2_rad']:.2f})")

    best_key = selected.loc[alternatives, "joint_motion_l2_rad"].idxmin()
    best_row = selected.loc[best_key]
    path_points = [tuple(point) for point in drawn_paths[best_key]]
    solutions = arm.inverse_path(path_points, str(best_row["elbow_mode"]))
    pose_points: list[tuple[float, float]] = []
    for index in np.linspace(0, len(solutions) - 1, 4, dtype=int):
        pose = arm.forward(solutions[index].theta1, solutions[index].theta2)
        pose_points.extend([pose.base, pose.elbow, pose.end])
        axis.plot(
            [pose.base[0], pose.elbow[0], pose.end[0]],
            [pose.base[1], pose.elbow[1], pose.end[1]],
            color=CURVE_COLORS[best_key],
            lw=1.1,
            alpha=0.24,
            marker="o",
            markersize=3,
        )

    axis.scatter(*arm.base, s=75, c="#0F172A", zorder=6, label="로봇 밑부분")
    axis.scatter(*start, s=110, c="#22C55E", edgecolors="white", linewidths=1.5, zorder=7, label="시작점")
    axis.scatter(*goal, s=120, c="#F43F5E", marker="*", edgecolors="white", linewidths=1.0, zorder=7, label="끝점")
    axis.text(arm.base[0], arm.base[1] - arm.min_reach * 0.15, "내부 공백", ha="center", va="top", color="#B91C1C", fontsize=9)

    relevant = [arm.base, start, goal, *pose_points]
    for points in drawn_paths.values():
        relevant.extend((float(x), float(y)) for x, y in points)
    bx, by = arm.base
    relevant.extend(
        [
            (bx - arm.min_reach, by),
            (bx + arm.min_reach, by),
            (bx, by - arm.min_reach),
            (bx, by + arm.min_reach),
        ]
    )
    relevant_array = np.asarray(relevant)
    lower = relevant_array.min(axis=0)
    upper = relevant_array.max(axis=0)
    center = (lower + upper) / 2.0
    span = max(float((upper - lower).max()), 1.0) * 1.18
    axis.set_xlim(center[0] - span / 2.0, center[0] + span / 2.0)
    axis.set_ylim(center[1] - span / 2.0, center[1] + span / 2.0)
    axis.set_aspect("equal")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(
        f"대표 사례 {scenario_id}: 직선은 실패하지만 곡선은 성공",
        pad=34,
    )
    axis.text(
        0.0,
        1.015,
        f"{SCENARIO_NAMES_KO.get(str(scenario['scenario_type']), scenario['scenario_type'])}  |  l1={arm.link1:.2f}, l2={arm.link2:.2f}, 내부 반지름={arm.min_reach:.2f}",
        transform=axis.transAxes,
        color="#64748B",
        fontsize=9.5,
    )
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axis.legend(
        unique.values(),
        unique.keys(),
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8.5,
    )
    _clean_axes(axis, grid_axis="both")
    figure.tight_layout()
    return figure


def _summary_table_figure(summary: pd.DataFrame) -> plt.Figure:
    ordered = summary.set_index("curve_key").loc[list(CURVE_ORDER)].reset_index()
    scenario_count = int(ordered["scenario_count"].max())
    values: list[list[str]] = []
    for row in ordered.itertuples(index=False):
        values.append(
            [
                str(row.curve_name_ko),
                f"{row.feasible_rate * 100:.1f}%\n[{row.feasible_ci_low * 100:.1f}, {row.feasible_ci_high * 100:.1f}]",
                f"{row.median_path_ratio:.3f}",
                f"{row.median_joint_motion_rad:.3f}",
                f"{row.joint_motion_wins}회\n({row.joint_motion_win_rate * 100:.1f}%)",
            ]
        )
    columns = ["곡선 계열", "도달 가능률\n[95% CI]", "경로비\n중앙값", "Q (rad)\n중앙값", "Q 최소\n시나리오"]
    figure, axis = plt.subplots(figsize=(11.6, 4.5))
    axis.axis("off")
    table = axis.table(
        cellText=values,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.0, 1.0, 0.84],
        colWidths=[0.19, 0.24, 0.16, 0.18, 0.19],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 2.0)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_facecolor("#0F766E")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#F1F5F9" if row % 2 else "#E2E8F0")
            if column == 0:
                cell.get_text().set_weight("bold")
                cell.get_text().set_color(CURVE_COLORS[CURVE_ORDER[row - 1]])
    figure.suptitle(
        f"곡선 계열별 핵심 결과 요약 (독립 시나리오 n={scenario_count})",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    figure.subplots_adjust(top=0.94, bottom=0.04, left=0.02, right=0.98)
    return figure


def _write_summary_csv(summary: pd.DataFrame, path: Path) -> None:
    ordered = summary.set_index("curve_key").loc[list(CURVE_ORDER)].reset_index()
    ordered.to_csv(path, index=False, encoding="utf-8-sig")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text_outputs(
    data: AnalysisData,
    output_dir: Path,
    *,
    metrics_path: Path,
    scenarios_path: Path,
    seed: int,
    bootstrap_iterations: int,
    representative_scenario_id: str,
) -> list[Path]:
    captions_path = output_dir / "captions.md"
    captions_path.write_text(
        """# 2단계 베이스라인 포스터 캡션 초안

## 그림 0 — 연구 모형

2차원 2관절 로봇팔의 끝점 경로를 역기구학으로 관절공간에 대응시켰다. 경로 전체가 작업공간 안에 있는지 확인하고, 끝점 경로 길이와 관절공간 경로 길이 Q를 비교 지표로 사용하였다.

## 그림 1 — 곡선 계열별 성공률

각 시나리오·곡선 계열에서 관절 이동량 Q가 가장 작은 도달 가능 후보를 선택했다. 막대는 240개 독립 시나리오의 성공률, 오차막대는 시나리오 단위 부트스트랩 95% 신뢰구간이다.

## 그림 2 — 경로 길이와 관절 이동량

끝점 경로가 짧더라도 역기구학 변환 후의 관절 이동량이 반드시 작지는 않았다. 따라서 곡선의 우수성을 하나의 지표만으로 판단하기보다 도달 가능성·경로 길이·관절 이동량을 함께 비교할 필요가 있다.

## 그림 3 — 상황별 성공률

근거리 이동, 내부 공백 횡단, 작업공간 경계 부근 이동에서 곡선 계열의 성공률이 서로 다르게 나타났다. 이는 입력 조건에 따라 적합한 곡선을 선택하는 규칙을 학습할 가능성을 뒷받침한다.

## 그림 4 — 직선 대비 관절 이동량

직선과 해당 곡선이 모두 도달 가능한 같은 시나리오만 짝지어 Q의 비를 계산했다. 1보다 작으면 해당 곡선의 관절 이동량이 더 작다.

## 그림 5 — 대표 사례

두 끝점 자체는 도달 가능하지만 직선 구간이 로봇팔의 내부 도달 불가 영역을 통과하는 사례다. 일부 곡선은 내부 공백을 우회하여 연속적인 관절 경로를 생성한다.

## 해석 시 반드시 함께 적을 한계

- 이 결과는 장애물이 없는 2차원 2관절 모델의 베이스라인이다.
- Q는 관절각 변화의 누적 길이이며 실제 모터 소비 에너지가 아니다.
- 도달 가능성은 161개 이산 표본점에서 검사했으므로 연속 구간에 대한 수학적 증명은 아니다.
- 21개 사전 정의 변형 중 최적 후보를 고른 결과이며, 각 곡선의 전 매개변수 공간을 최적화한 결과는 아니다.
- 다음 단계에서는 장애물 충돌·최소 여유 거리·속도/가속도·동역학 기반 에너지 지표를 추가해야 한다.
""",
        encoding="utf-8",
    )
    equations_path = output_dir / "method_equations.tex"
    equations_path.write_text(
        r"""% Poster equation source (phase-2 baseline)
\begin{align}
\mathbf{p}(\boldsymbol{\theta}) &=
\begin{bmatrix}
l_1\cos\theta_1+l_2\cos(\theta_1+\theta_2)\\
l_1\sin\theta_1+l_2\sin(\theta_1+\theta_2)
\end{bmatrix},\\
|l_1-l_2| &\le \|\mathbf{p}(t)-\mathbf{b}\| \le l_1+l_2,\\
L &= \int_0^1 \|\mathbf{p}'(t)\|\,dt,\\
Q &\approx \sum_{i=0}^{N-2}\sqrt{(\theta_{1,i+1}-\theta_{1,i})^2+(\theta_{2,i+1}-\theta_{2,i})^2}.
\end{align}
""",
        encoding="utf-8",
    )
    manifest_path = output_dir / "analysis_manifest.json"
    manifest = {
        "analysis_version": "phase2-poster-v1",
        "source_files": {
            "metrics": {
                "path": metrics_path.name,
                "sha256": _sha256(metrics_path),
            },
            "scenarios": {
                "path": scenarios_path.name,
                "sha256": _sha256(scenarios_path),
            },
        },
        "independent_scenario_count": int(data.scenarios["scenario_id"].nunique()),
        "raw_run_count": int(len(data.metrics)),
        "curve_families": list(CURVE_ORDER),
        "candidate_selection_rule": (
            "Within each scenario and curve family, keep fully reachable candidates; "
            "minimize joint_motion_l2_rad, then curve_length, then maximize "
            "min_singularity_index."
        ),
        "bootstrap": {
            "unit": "scenario",
            "iterations": bootstrap_iterations,
            "confidence_level": 0.95,
            "seed": seed,
        },
        "representative_scenario_id": representative_scenario_id,
        "limitations": [
            "No obstacles or collision checks.",
            "Joint-space path length Q is not physical motor energy.",
            "Reachability is sampled discretely at 161 points.",
            "Curve parameters are a finite preset grid, not continuous optimization.",
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return [captions_path, equations_path, manifest_path]


def generate_poster_assets(
    data: AnalysisData,
    output_dir: str | Path,
    *,
    metrics_path: str | Path,
    scenarios_path: str | Path,
    seed: int = 20260825,
    bootstrap_iterations: int = 4000,
) -> PosterOutputs:
    """Generate all poster assets and return their paths."""

    _configure_style()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    representative = select_representative_scenario(data.metrics, data.best_candidates)

    generated: list[Path] = []
    figure_specs: Iterable[tuple[str, plt.Figure]] = (
        ("fig00_method_model", _method_model_figure()),
        ("fig01_feasibility_by_curve", _feasibility_figure(data.curve_summary)),
        ("fig02_path_vs_joint_motion", _tradeoff_figure(data.best_candidates)),
        ("fig03_scenario_heatmap", _scenario_heatmap_figure(data.best_candidates)),
        ("fig04_joint_motion_vs_line", _paired_ratio_figure(data.best_candidates, seed=seed)),
        ("fig05_representative_case", _representative_figure(data, representative)),
        ("table01_curve_summary", _summary_table_figure(data.curve_summary)),
    )
    for stem, figure in figure_specs:
        generated.extend(_save_figure(figure, destination, stem))

    csv_path = destination / "table01_curve_summary.csv"
    _write_summary_csv(data.curve_summary, csv_path)
    generated.append(csv_path)
    generated.extend(
        _write_text_outputs(
            data,
            destination,
            metrics_path=Path(metrics_path),
            scenarios_path=Path(scenarios_path),
            seed=seed,
            bootstrap_iterations=bootstrap_iterations,
            representative_scenario_id=representative,
        )
    )
    return PosterOutputs(destination, tuple(generated), representative)
