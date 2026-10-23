"""Extract a train-only cycloid/limacon teaching case without editing raw data."""
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
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robot_planner.curves import make_curve

OUT = ROOT / "outputs/rne_activity_2026_10_02/advanced_curves"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "outputs/phase4_v3_analysis/train_analysis_metrics.parquet"
    cols = ["scenario_id", "run_id", "curve_key", "cohort", "selection_valid",
            "obstacle_count", "path_length_units", "balanced_cost_0_100"]
    df = pd.read_parquet(cache, columns=cols)
    valid = df[df.selection_valid & df.cohort.eq("main")]
    distance = valid.sort_values(["path_length_units", "run_id"]).groupby("scenario_id").first()
    balanced = valid.sort_values(["balanced_cost_0_100", "run_id"]).groupby("scenario_id").first()
    ids = sorted(set(distance[distance.curve_key.eq("cycloid")].index)
                 & set(balanced[balanced.curve_key.eq("limacon")].index))
    sid = ids[0]
    assert sid == "P4S000691", "Source selection changed; review handout."
    partition = ROOT / "data/phase4_v3_release/partitions/train" / sid
    receipt_file = partition / "done.json"
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    folder = partition / receipt["attempt"]
    sources = []
    for name in ("trajectory_metrics.jsonl.gz", "scenario_winners.json"):
        p = folder / name
        f = next(x for x in receipt["files"] if x["path"] == name)
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        assert p.stat().st_size == f["bytes"] and digest == f["sha256"]
        sources.append(dict(path=p.relative_to(ROOT).as_posix(), sha256=digest, bytes=f["bytes"]))
    with gzip.open(folder / "trajectory_metrics.jsonl.gz", "rt", encoding="utf-8") as f:
        rows = [json.loads(s) for s in f]
    winners = json.loads((folder / "scenario_winners.json").read_text(encoding="utf-8"))
    w = {r["objective"]: r for r in winners}
    indexed = {r["run_id"]: r for r in rows}
    cyc = indexed[sid + "_cycloid_p80_up"]
    lim = indexed[sid + "_limacon_m80_up"]
    accepted = [r for r in rows if r["selection_valid"]]
    assert len(rows) == 42 and len(accepted) == 3
    assert all(r["data_split"] == "train" and r["candidate_status"] != "unresolved" for r in rows)
    for obj in ("distance", "joint_effort", "jerk"):
        assert w[obj]["winner_run_id"] == cyc["run_id"] and w[obj]["tie_count"] == 1
    for obj in ("curvature", "safety", "balanced"):
        assert w[obj]["winner_run_id"] == lim["run_id"] and w[obj]["tie_count"] == 1
    for r in accepted:
        assert r["candidate_status"] == "accepted" and r["path_check_status"] == "bounded_clear"
        assert r["geometry_convergence_status"] == "passed" and r["joint_integration_status"] == "accepted"
    p0 = np.array([cyc["start_x"], cyc["start_y"]])
    p1 = np.array([cyc["goal_x"], cyc["goal_y"]])
    delta = p1 - p0
    D = float(np.linalg.norm(delta))
    e = delta/D
    n = np.array([-e[1], e[0]])
    u = np.linspace(0, 1, 1001)
    phi = 2*np.pi*u
    A, B = D/(2*np.pi), .2*D
    c = math.sqrt(B*B-A*A)
    cycloid_length = 4*(B+A*A/c*math.log((c+B)/A))
    cycloid_quad = quad(lambda v: math.hypot(A*(1-math.cos(v)), B*math.sin(v)),
                        0, 2*math.pi, epsabs=1e-11, epsrel=1e-12)[0]
    assert abs(cycloid_length-cycloid_quad) < 1e-11
    cycloid_points = p0 + (A*(phi-np.sin(phi)))[:, None]*e + (B*(1-np.cos(phi)))[:, None]*n
    cp = np.asarray(make_curve("cycloid", tuple(p0), tuple(p1), shape=.8, samples=len(u)).points)
    cyc_error = float(np.max(np.linalg.norm(cp-cycloid_points, axis=1)))
    assert cyc_error < 1e-12
    prm = json.loads(lim["curve_parameters_json"])
    b, lam, beta, alpha = prm["b"], prm["scale"], prm["rotation"], 3*math.pi/5
    assert b == .79 and prm["side"] == -1 and prm["a"] == 1
    assert math.isclose(lam, D/(2*(1+b*math.cos(alpha))*math.sin(alpha)), rel_tol=1e-14)
    origin = p0-lam*(1+b*math.cos(alpha))*np.array([math.cos(beta+alpha), math.sin(beta+alpha)])
    ang = -alpha+2*alpha*u
    radius = 1+b*np.cos(ang)
    limacon_points = origin + lam*radius[:, None]*np.column_stack((np.cos(beta-ang), np.sin(beta-ang)))
    lp = np.asarray(make_curve("limacon", tuple(p0), tuple(p1), shape=-.8, samples=len(u)).points)
    lim_error = float(np.max(np.linalg.norm(lp-limacon_points, axis=1)))
    assert lim_error < 1e-12
    limacon_length, limacon_quad_error = quad(lambda v: lam*math.sqrt(1+b*b+2*b*math.cos(v)),
                                            -alpha, alpha, epsabs=1e-11, epsrel=1e-12)
    area_cyc = 3*math.pi*A*B
    area_lim = lam*lam*((1+b*b/2)*alpha+2*b*math.sin(alpha)+b*b/4*math.sin(2*alpha))
    assert abs(area_cyc-quad(lambda v: A*B*(1-math.cos(v))**2, 0, 2*math.pi)[0]) < 1e-10
    assert abs(area_lim-quad(lambda v: .5*lam*lam*(1+b*math.cos(v))**2, -alpha, alpha)[0]) < 1e-10
    # Check resolution sensitivity within the same three accepted candidates.
    weights = {"path_length_units": .25, "joint_velocity_effort": .20, "joint_jerk_effort": .20,
               "regularized_curvature_energy_inv_unit": .20, "obstacle_penalty_s_per_unit2": .15}
    scores = {}
    for size in ("161", "321", "641"):
        costs = np.zeros(len(accepted))
        for field, weight in weights.items():
            values = np.array([r["geometry_checks"][field]["values_by_n"][size]
                               if field in r["geometry_checks"] else r[field] for r in accepted])
            span = float(np.max(values)-np.min(values))
            costs += 100*weight*((values-values.min())/span if span > 0 else np.zeros_like(values))
        scores[size] = {r["run_id"]: float(100-cost) for r, cost in zip(accepted, costs)}
    assert all(max(z, key=z.get) == lim["run_id"] for z in scores.values())
    assert all(abs(scores["161"][r["run_id"]]-r["balanced_reward_0_100"]) < 1e-10 for r in accepted)
    keep = ["run_id", "dataset_version", "base_x", "base_y", "link1", "link2", "start_x", "start_y",
            "goal_x", "goal_y", "shape", "elbow_mode", "duration_s", "obstacles_json",
            "path_length_units", "joint_velocity_effort", "joint_acceleration_effort", "joint_jerk_effort",
            "regularized_curvature_energy_inv_unit", "obstacle_penalty_s_per_unit2",
            "balanced_reward_0_100", "candidate_status", "path_check", "geometry_checks",
            "quadrature_diagnostics", "equation_descriptor_json"]
    evidence = dict(
        scenario_id=sid, sources=sources,
        selection_rule="train main cohort; distance winner cycloid and balanced winner limacon; lowest scenario_id; educational post-hoc selection",
        candidate_count=42, accepted_count=3, randomness="none",
        candidates=[{k:r[k] for k in keep} for r in accepted],
        straight_exclusions=[dict(run_id=r["run_id"], path_check=r["path_check"])
                             for r in rows if r["curve_key"] == "line"],
        winners=[{k:r[k] for k in ("objective", "winner_run_id", "tie_count")} for r in winners],
        derivation=dict(D=D, A=A, B=B, c=c, b=b, scale=lam, rotation=beta, polar_origin=origin.tolist(),
                        cycloid_length_analytic=cycloid_length, limacon_length_quad=limacon_length,
                        cycloid_length_stored_error=cyc["path_length_units"]-cycloid_length,
                        limacon_length_stored_error=lim["path_length_units"]-limacon_length,
                        cycloid_area=area_cyc, limacon_sector_area=area_lim),
        scores_by_geometry_grid=scores,
        checks=dict(status="PASS", cycloid_point_error=cyc_error, limacon_point_error=lim_error,
                    cycloid_length_quad_difference=abs(cycloid_length-cycloid_quad),
                    limacon_length_quad_error_estimate=limacon_quad_error),
        limitations="No new full-data audit; not held-out performance; areas are educational and not existing objectives; same candidate set and fixed weights for grid sensitivity, not universal rank robustness.")
    (OUT / "case_evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")

    font_manager.fontManager.addfont("C:/Windows/Fonts/malgun.ttf")
    plt.rcParams.update({"font.family":"Malgun Gothic", "axes.unicode_minus":False,
                         "svg.fonttype":"path", "font.size":11})
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), gridspec_kw={"width_ratios":[1.35, 1, 1]})
    colors = {"cycloid":"#8864BB", "limacon":"#189DAA"}
    ax = axes[0]
    ax.plot(*cp.T, color=colors["cycloid"], lw=2.5, label="사이클로이드 변형")
    ax.plot(*lp.T, color=colors["limacon"], lw=2.5, label="리마송 구간")
    ax.plot([p0[0],p1[0]], [p0[1],p1[1]], "--", color="#A0A8B5", lw=1.2, label="직선 · 링크 충돌로 제외")
    base = (cyc["base_x"],cyc["base_y"])
    ax.scatter(*base, marker="s", color="#25324D", s=35)
    ax.annotate("밑부분", base, xytext=(6,7), textcoords="offset points", fontsize=9)
    for label, p in (("출발",p0),("도착",p1)):
        ax.scatter(*p, color="#25324D", s=30)
        ax.annotate(label,p,xytext=(-28,-14),textcoords="offset points",fontsize=10)
    for obs in json.loads(cyc["obstacles_json"]):
        ax.add_patch(Circle((obs["center_x"],obs["center_y"]),obs["radius_units"],facecolor="#F7D1D1",edgecolor="#D95B5B"))
    ax.set(title="같은 환경 · 서로 다른 기준의 승자", xlabel="x [unit]", ylabel="y [unit]")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5,.935),
               ncol=3, fontsize=10, frameon=False)
    ax.margins(.18)
    axes[1].plot((phi-np.sin(phi))/(2*np.pi), .2*(1-np.cos(phi)), color=colors["cycloid"], lw=2.5)
    axes[1].set(title="사이클로이드 변형 · 정규화 좌표",xlabel="X [무차원]",ylabel="Y [무차원]",ylim=(-.1,.65))
    axes[1].text(.5,.55,r"$X=(\phi-\sin\phi)/(2\pi)$",ha="center",fontsize=12)
    axes[1].text(.5,.46,r"$Y=0.2(1-\cos\phi)$",ha="center",fontsize=12)
    rx, ry = radius*np.cos(ang), radius*np.sin(ang)
    axes[2].fill(np.r_[0,rx,0], np.r_[0,ry,0], color=colors["limacon"], alpha=.1)
    axes[2].plot(rx,ry,color=colors["limacon"],lw=2.5)
    axes[2].plot([rx[0],0,rx[-1]],[ry[0],0,ry[-1]],"--",color="#778399",lw=1.1)
    axes[2].scatter([0],[0],color="#25324D",s=20)
    axes[2].set(title="리마송 · 닮음변환 전 표준좌표",xlabel="원형식 가로좌표 [무차원]",ylabel="원형식 세로좌표 [무차원]")
    axes[2].text(.82,0,r"$r=1+0.79\cos\phi$",ha="center",fontsize=12)
    for axis in axes:
        axis.set_aspect("equal",adjustable="box")
        axis.grid(alpha=.15)
    fig.suptitle("배운 곡선의 적분으로 설명하는 로봇팔 경로",fontsize=18,fontweight="bold",y=.98)
    fig.text(.5,.095,"P4S000691 · train · 장애물 1개 · 3 s · 두 선택 후보 모두 up · 42개 후보 중 유효 3개",ha="center")
    fig.text(.5,.045,"가운데 길이 × D(6.940081) / 오른쪽 길이 × λ(4.827000) = 원래 길이 · 음영은 극좌표 면적(평가함수 아님)",ha="center",fontsize=10,color="#596579")
    fig.subplots_adjust(left=.055,right=.98,top=.81,bottom=.21,wspace=.4)
    fig.savefig(OUT / "cycloid_limacon.png",dpi=300)
    fig.savefig(OUT / "cycloid_limacon.svg")
    plt.close(fig)
    print(json.dumps(dict(output=str(OUT),derivation=evidence["derivation"],checks=evidence["checks"],
                         scores_by_geometry_grid=scores), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
