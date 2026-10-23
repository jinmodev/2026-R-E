"""Read-only train example selection; generate a checked hand-calculation figure.

Run: .venv/Scripts/python.exe scripts/prepare_handcalculation_case.py
Existing datasets, analysis results, and GUI files are never modified.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robot_planner.curves import make_curve

OUT = ROOT / "outputs/rne_activity_2026_10_02"
CACHE = ROOT / "outputs/phase4_v3_analysis/train_analysis_metrics.parquet"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Deliberately illustrative, not a representative sample or significance test.
    df = pd.read_parquet(CACHE)
    eligible = df.loc[(df.cohort == "main") & df.selection_valid & (df.obstacle_count == 0)]
    choices = []
    for sid, group in eligible.groupby("scenario_id", sort=True):
        best = group.sort_values(["joint_velocity_effort", "run_id"]).iloc[0]
        lines = group[group.curve_key == "line"].sort_values(["joint_velocity_effort", "run_id"])
        if best.curve_key != "bezier" or lines.empty:
            continue
        line = lines.iloc[0]
        gain = 1 - best.joint_velocity_effort / line.joint_velocity_effort
        if gain >= 0.20:
            choices.append((sid, line.run_id, best.run_id))
    sid, line_id, bezier_id = choices[0]
    if sid != "P4S000057":
        raise RuntimeError("Selected case changed; review the accompanying activity handout.")
    folder = ROOT / "data/phase4_v3_release/partitions/train" / sid
    receipt_path = folder / "done.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    attempt = folder / receipt["attempt"]
    source_files = []
    for name in ("trajectory_metrics.jsonl.gz", "scenario_winners.json"):
        p = attempt / name
        expected = next(f for f in receipt["files"] if f["path"] == name)
        actual = {"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size,
                  "sha256": digest(p)}
        assert actual["bytes"] == expected["bytes"]
        assert actual["sha256"] == expected["sha256"]
        source_files.append(actual)
    with gzip.open(attempt / "trajectory_metrics.jsonl.gz", "rt", encoding="utf-8") as f:
        rows = [json.loads(s) for s in f]
    winners = json.loads((attempt / "scenario_winners.json").read_text(encoding="utf-8"))
    winner = {w["objective"]: w for w in winners}
    line = next(r for r in rows if r["run_id"] == line_id)
    bezier = next(r for r in rows if r["run_id"] == bezier_id)
    assert len(rows) == receipt["candidates"] == 42
    assert all(r["data_split"] == "train" for r in rows)
    assert all(r["candidate_status"] != "unresolved" for r in rows)
    assert line_id in json.loads(winner["distance"]["tied_run_ids_json"])
    assert winner["joint_effort"]["winner_run_id"] == bezier_id
    assert winner["joint_effort"]["tie_count"] == 1
    assert winner["balanced"]["winner_run_id"] == line_id
    for r in (line, bezier):
        assert r["selection_valid"] and r["candidate_status"] == "accepted"
        assert r["path_check_status"] == "bounded_clear"
        assert r["geometry_convergence_status"] == "passed"
        assert r["joint_integration_status"] == "accepted"
        assert r["elbow_mode"] == "up" and r["duration_s"] == 3.0
        assert r["obstacles_json"] == "[]"
    p0 = np.array([line["start_x"], line["start_y"]])
    p1 = np.array([line["goal_x"], line["goal_y"]])
    delta = p1 - p0
    D = float(np.linalg.norm(delta))
    tangent = delta / D
    normal = np.array([-tangent[1], tangent[0]])
    a = 1.26 * bezier["shape"]
    assert math.isclose(a, -0.504, abs_tol=1e-15)
    u = np.linspace(0, 1, 1001)
    y_normalized = a * u * (1 - u)
    polynomial = p0 + u[:, None] * delta + (D * y_normalized)[:, None] * normal
    curve = make_curve("bezier", tuple(p0), tuple(p1), shape=bezier["shape"], samples=len(u))
    max_error = float(np.max(np.linalg.norm(polynomial - np.asarray(curve.points), axis=1)))
    assert max_error < 1e-12
    params = json.loads(bezier["curve_parameters_json"])
    assert all(math.isclose(curve.parameters[k], v, abs_tol=1e-13) for k, v in params.items())
    ctrl1 = np.array([params["control1_x"], params["control1_y"]])
    ctrl2 = np.array([params["control2_x"], params["control2_y"]])
    cubic_residual = -p0 + 3 * ctrl1 - 3 * ctrl2 + p1
    assert np.max(np.abs(cubic_residual)) < 1e-12
    recovered = np.column_stack(((polynomial - p0) @ tangent / D,
                                 (polynomial - p0) @ normal / D))
    assert np.allclose(recovered, np.column_stack((u, y_normalized)), atol=1e-13, rtol=0)
    analytic_L = D / 2 * (math.sqrt(1 + a*a) + math.asinh(abs(a)) / abs(a))
    quad_L, quad_error = quad(lambda x: D * math.sqrt(1 + (a - 2*a*x)**2), 0, 1,
                              epsabs=1e-12, epsrel=1e-12)
    assert abs(quad_L - analytic_L) < 1e-12
    stored_L = bezier["path_length_units"]
    assert abs(stored_L - analytic_L) < 1e-4  # Display check, not a new selection gate.
    selected_fields = ["scenario_id", "run_id", "dataset_version", "data_split", "curve_key",
                       "shape", "elbow_mode", "base_x", "base_y", "link1", "link2", "start_x",
                       "start_y", "goal_x", "goal_y", "duration_s", "obstacles_json", "selection_valid",
                       "candidate_status", "path_check_status", "geometry_convergence_status",
                       "joint_integration_status", "path_length_units", "joint_velocity_effort",
                       "balanced_reward_0_100", "equation_descriptor_json", "geometry_checks",
                       "quadrature_diagnostics"]
    evidence = {
        "scenario_id": sid, "dataset_version": line["dataset_version"],
        "selection_rule": "train main cohort; no obstacles; best effort candidate is bezier; "
                          "best-effort valid line exists; effort reduction >=20%; lowest scenario ID",
        "selection_warning": "Post-hoc pedagogical selection, not population inference or held-out testing.",
        "qualifying_scenarios": len(choices), "randomness": "none; deterministic sorting",
        "source_files": source_files,
        "receipt_sha256": digest(receipt_path), "train_cache_sha256": digest(CACHE),
        "curve_implementation_sha256": digest(ROOT / "src/robot_planner/curves.py"),
        "candidates": {r["run_id"]: {k: r[k] for k in selected_fields} for r in (line, bezier)},
        "winners": [{k: w[k] for k in ("objective", "winner_run_id", "tie_count", "tied_run_ids_json")}
                    for w in winners],
        "derivation": {"D": D, "delta": delta.tolist(), "tangent": tangent.tolist(),
                       "normal": normal.tolist(), "normalized_Y_linear": a, "normalized_Y_quadratic": -a,
                       "physical_local_Y_quadratic": -a/D,
                       "original_xy_coefficients_ascending": [p0.tolist(), (delta+a*D*normal).tolist(),
                                                               (-a*D*normal).tolist()],
                       "analytic_length": analytic_L, "stored_161_length": stored_L,
                       "stored_length_absolute_error": abs(stored_L-analytic_L),
                       "stored_length_relative_error": abs(stored_L-analytic_L)/analytic_L,
                       "analytic_length_increase_percent": 100*(analytic_L/D-1),
                       "effort_reduction_percent": 100*(1-bezier["joint_velocity_effort"]/line["joint_velocity_effort"])},
        "checks": {"status": "PASS", "polynomial_max_point_error": max_error,
                   "cubic_coefficient_residual": cubic_residual.tolist(),
                   "independent_quad_length_difference": abs(quad_L-analytic_L),
                   "independent_quad_error_estimate": quad_error},
    }
    (OUT / "selected_case_evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")

    font = Path("C:/Windows/Fonts/malgun.ttf")
    if not font.exists():
        raise RuntimeError("Korean font missing; configure a Korean-capable font before rendering.")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False,
                         "svg.fonttype": "path", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.0), gridspec_kw={"width_ratios": [1, 1.35]})
    straight = p0 + u[:, None] * delta
    axes[0].plot(*straight.T, color="#3264D8", lw=2.6, label="직선 · 거리 동점 최적")
    axes[0].plot(*polynomial.T, color="#259067", lw=2.6, label="베지어 · 노력도 최적")
    base = np.array([line["base_x"], line["base_y"]])
    for fraction, alpha in ((0, 0.8), (1, 0.4)):
        p = p0 if fraction == 0 else p1
        v = p - base
        l1, l2 = line["link1"], line["link2"]
        th2 = -math.acos((v@v-l1*l1-l2*l2)/(2*l1*l2))
        th1 = math.atan2(v[1], v[0])-math.atan2(l2*math.sin(th2), l1+l2*math.cos(th2))
        joint = base+l1*np.array([math.cos(th1), math.sin(th1)])
        arm = np.vstack((base, joint, p))
        axes[0].plot(*arm.T, "o--", color="#778399", alpha=alpha, lw=1.4, zorder=0)
    axes[0].scatter(*base, c="#25324D", marker="s", s=45)
    axes[0].annotate("밑부분", base, xytext=(-5, -20), textcoords="offset points")
    axes[0].annotate("출발", p0, xytext=(-33, -10), textcoords="offset points")
    axes[0].annotate("도착", p1, xytext=(7, 0), textcoords="offset points")
    axes[0].set(xlabel="x [unit]", ylabel="y [unit]", title="원래 데이터의 로봇팔과 경로")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend(loc="upper left", fontsize=10)
    axes[0].margins(.20)
    axes[1].plot(u, np.zeros_like(u), color="#3264D8", lw=2.8)
    axes[1].plot(u, y_normalized, color="#259067", lw=2.8)
    axes[1].scatter([0, 1], [0, 0], color="#25324D", s=38, zorder=5)
    axes[1].scatter([.5], [-.126], color="#259067", s=35)
    axes[1].text(.5, .03, r"$Y=0$", color="#3264D8", ha="center", fontsize=15)
    axes[1].text(.5, -.203, r"$Y=0.504X^2-0.504X$", color="#259067", ha="center", fontsize=15)
    axes[1].annotate("꼭짓점 (0.5, -0.126)", (.5, -.126), xytext=(.68, -.16), fontsize=10,
                     arrowprops={"arrowstyle": "-", "color": "#778399"})
    axes[1].set(xlabel="X [무차원]", ylabel="Y [무차원]", title="이동·회전·길이 정규화 후 같은 경로",
                xlim=(-.08, 1.08), ylim=(-.26, .13))
    axes[1].set_aspect("equal", adjustable="box")
    for ax in axes:
        ax.grid(alpha=.15)
    fig.suptitle("손계산 사례: 짧은 경로가 관절 노력도도 최소일까?", fontsize=17, fontweight="bold", y=.97)
    fig.text(.5, .065, "P4S000057 · train · 장애물 없음 · 두 경로 모두 up 해 · 이동시간 3 s", ha="center", fontsize=11)
    fig.text(.5, .025, "42개 준비 후보 중 유효 34개 비교 | 오른쪽 길이에 D = 4.685244746을 곱하면 원래 길이", ha="center", fontsize=10, color="#555E70")
    fig.subplots_adjust(left=.07, right=.97, top=.87, bottom=.19, wspace=.28)
    fig.savefig(OUT / "line_and_parabola.png", dpi=300)
    fig.savefig(OUT / "line_and_parabola.svg")
    plt.close(fig)
    print(json.dumps({"output": str(OUT), "scenario": sid, "checks": evidence["checks"],
                      "derivation": evidence["derivation"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
