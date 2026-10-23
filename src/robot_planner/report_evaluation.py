"""Report-facing evaluator, conservative candidate labels, and portable exports."""
from __future__ import annotations
from dataclasses import asdict
import csv
import json
import math
from pathlib import Path

from .curves import make_curve
from .phase4 import _winner_key
from .phase4_evaluation import MathematicalEvaluationConfig, MathematicalTrajectoryResult
from .phase4_release import evaluate_release
from .report_math import report_equations
from .report_physics import ReportPhysicsConfig, evaluate_physics, METRICS

DATASET_VERSION = "phase4-report-v4"
REPORT_WEIGHTS = {
    "path_length_units": .25, "model_electrical_energy_j": .20,
    "joint_jerk_effort": .20, "regularized_curvature_energy_inv_unit": .20,
    "obstacle_penalty_s_per_unit2": .15,
}
OBJECTIVES = {
    "distance": ("path_length_units", None),
    "joint_effort": ("joint_velocity_effort", None),
    "acceleration": ("joint_acceleration_effort", None),
    "jerk": ("joint_jerk_effort", None),
    "curvature": ("regularized_curvature_energy_inv_unit", None),
    "obstacle_penalty": ("obstacle_penalty_s_per_unit2", None),
    "model_energy": ("model_electrical_energy_j", "energy_selection_ready"),
    "rms_torque": ("rms_joint_torque_nm", "torque_selection_ready"),
    "report_balanced": ("report_balanced_cost_0_100", "energy_selection_ready"),
}


def skipped_physics(reason):
    return dict(metrics={**{k:None for k in METRICS},"physics_status":"not_evaluated",
                         "energy_selection_ready":False,"torque_selection_ready":False,
                         "physics_error":reason,"work_energy_balance_passed":False},
                grids={},samples=None)


def physical_sample_rows(physics):
    if physics["samples"] is None:
        return []
    state,values=physics["samples"]
    result=[]
    for i,t in enumerate(state["time_s"]):
        result.append(dict(sample_index=i,time_s=float(t),
                           **{k:float(v[i]) for k,v in values.items()}))
    return result


def simulate_report_trajectory(arm,curve,elbow_mode="up",*,obstacles=(),duration_s=3.,samples=161,physics_config=None):
    """Use verified v3 mathematical costs, then append explicitly versioned physics."""
    if samples != 161:
        raise ValueError("Report GUI export uses 161 samples; convergence uses 321/641 internally")
    cfg=physics_config or ReportPhysicsConfig()
    old=evaluate_release(arm,curve,elbow_mode,obstacles=obstacles,
                         config=MathematicalEvaluationConfig(duration_s=duration_s))
    metrics=dict(old["metrics"])
    metrics["source_calculation_version"]=metrics["calculation_version"]
    metrics["source_dataset_version"]=metrics["dataset_version"]
    metrics["dataset_version"]=DATASET_VERSION
    physics=(evaluate_physics(arm,curve,duration_s,elbow_mode,cfg) if metrics["selection_valid"]
             else skipped_physics("mathematical_candidate_"+metrics["candidate_status"]))
    metrics.update(physics["metrics"])
    metrics["physics_config_json"]=json.dumps(asdict(cfg),sort_keys=True)
    metrics["physics_grids_json"]=json.dumps(physics["grids"],sort_keys=True)
    metrics["geometry_checks_json"]=json.dumps(old["geometry_checks"],sort_keys=True)
    equations={**old["equations"],**report_equations(curve,duration_s)}
    extras=physical_sample_rows(physics)
    rows=[]
    for i,row in enumerate(old["sample_rows"]):
        row=dict(row)
        if extras:
            row.update(extras[i])
        else:
            for key in ("torque1_nm","torque2_nm","joint1_power_w","joint2_power_w","net_power_w",
                        "positive_power_w","absorbed_power_w","torque_squared_nm2","copper_loss_power_w",
                        "damping_power_w","kinetic_energy_j","potential_energy_j"):
                row[key]=None
        row["physics_sample_status"]="computed" if extras else "not_evaluated"
        rows.append(row)
    return MathematicalTrajectoryResult(tuple(rows),metrics,equations)


def score_report_candidates(records):
    """New energy composite; keep raw scores and quarantine unresolved alternatives.

    Numerical tie tolerance matches the existing 1e-12 rule. This score is
    scenario-relative; its energy term replaces the old effort term explicitly.
    """
    grouped={}
    for r in records:
        grouped.setdefault(r["scenario_id"],[]).append(r)
    labels=[]
    for sid,rows in grouped.items():
        mathematical=[r for r in rows if r.get("selection_valid") is True]
        energy=[r for r in mathematical if r.get("energy_selection_ready") is True]
        unresolved=any(r.get("candidate_status")=="unresolved" for r in rows)
        for r in rows:
            r["report_balanced_cost_0_100"]=None
            r["report_balanced_reward_0_100"]=None
            r["report_selection_valid"]=r in energy
            for field in REPORT_WEIGHTS:
                r["report_normalized_"+field]=None
        if energy:
            spans={f:(min(float(r[f]) for r in energy),max(float(r[f]) for r in energy))
                   for f in REPORT_WEIGHTS}
            for r in energy:
                cost=0.
                for field,weight in REPORT_WEIGHTS.items():
                    low,high=spans[field]
                    norm=(float(r[field])-low)/(high-low) if high>low else 0.
                    r["report_normalized_"+field]=norm
                    cost+=100*weight*norm
                r["report_balanced_cost_0_100"]=cost
                r["report_balanced_reward_0_100"]=100-cost
        for objective,(field,readiness) in OBJECTIVES.items():
            pool=[r for r in mathematical if readiness is None or r.get(readiness) is True]
            missing=any(r.get(field) is None or not math.isfinite(float(r[field])) for r in pool)
            status=("unresolved_source_alternative" if unresolved else "no_valid_candidate" if not mathematical
                    else "unresolved_physics_alternative" if len(pool)!=len(mathematical) or missing else "ready")
            label=dict(scenario_id=sid,objective=objective,metric=field,label_status=status,
                       source_valid_candidates=len(mathematical),eligible_candidates=len(pool),
                       winner_run_id=None,tie_count=None,tied_run_ids_json=None,
                       raw_metric_value=None,dataset_version=DATASET_VERSION)
            if status=="ready":
                best=min(float(r[field]) for r in pool)
                tied=[r for r in pool if math.isclose(float(r[field]),best,rel_tol=1e-12,abs_tol=1e-12)]
                winner=min(tied,key=lambda r:_winner_key(r,field))
                label.update(winner_run_id=winner["run_id"],tie_count=len(tied),
                             tied_run_ids_json=json.dumps(sorted(r["run_id"] for r in tied)),
                             raw_metric_value=float(winner[field]))
            labels.append(label)
        scenario_ready=not unresolved and len(energy)==len(mathematical) and bool(mathematical)
        for r in rows:
            r["report_scenario_labels_ready"]=scenario_ready
    return labels


def compare_candidates(arm,start,end,*,obstacles=(),duration_s=3.,physics_config=None,progress=None):
    """Exhaustive fixed 42-candidate comparison, no machine learning."""
    from .experiments import curve_variants
    records,results=[],{}
    variants=curve_variants()
    total=len(variants)*2
    for variant in variants:
        curve=make_curve(variant.curve_key,start,end,shape=variant.shape,samples=161)
        for mode in ("up","down"):
            rid=f"interactive_{variant.variant_id}_{mode}"
            result=simulate_report_trajectory(arm,curve,mode,obstacles=obstacles,duration_s=duration_s,
                                              physics_config=physics_config)
            record=dict(result.metrics,scenario_id="interactive",run_id=rid,
                        curve_key=curve.key,shape=variant.shape,elbow_mode=mode,
                        base_x=arm.base[0],base_y=arm.base[1],link1=arm.link1,link2=arm.link2,
                        start_x=start[0],start_y=start[1],goal_x=end[0],goal_y=end[1])
            records.append(record)
            results[rid]=result
            if progress:
                progress(len(records),total)
    return dict(records=records,labels=score_report_candidates(records),results=results)


def write_csv(path,rows):
    """Flat UTF-8 BOM CSV; None is blank, booleans are explicit true/false."""
    rows=list(rows)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:("true" if v else "false") if isinstance(v,bool) else v for k,v in row.items()})


def export_comparison(directory,comparison):
    """Separate inputs/summary/equations/samples, no charts."""
    root=Path(directory)
    root.mkdir(parents=True,exist_ok=False)
    write_csv(root/"candidate_metrics.csv",comparison["records"])
    write_csv(root/"objective_winners.csv",comparison["labels"])
    write_csv(root/"equations.csv",[
        dict(run_id=rid,**result.equations) for rid,result in comparison["results"].items()])
    all_rows=[dict(run_id=rid,**row) for rid,result in comparison["results"].items() for row in result.sample_rows]
    geometry_keys={"sample_index","geometry_u","geometry_x","geometry_y","dx_du","dy_du","d2x_du2","d2y_du2",
                   "path_speed_units_per_u","curvature_inv_unit","regularized_curvature_integrand_inv_unit",
                   "curvature_speed_epsilon_units_per_u","path_regular"}
    write_csv(root/"geometry_samples.csv",[
        {k:v for k,v in r.items() if k=="run_id" or k in geometry_keys} for r in all_rows])
    write_csv(root/"time_samples.csv",[
        {k:v for k,v in r.items() if k not in geometry_keys or k=="sample_index"} for r in all_rows])
    (root/"README.txt").write_text(
        "42 prepared candidates; no learned recommendation. Match files on run_id.\n"
        "Only label_status=ready supports a winner. Blank is unavailable, not zero.\n"
        "geometry_u/geometry_x/dx_du etc use a cosine-clustered geometric grid; time_s/x/theta use a uniform time grid.\n"
        "Physics headline integrals use 641 points; sample export has 161, so re-integration is approximate.\n"
        "Energy is a model estimate under physics_config_json. Source mathematical effort is not J.\n"
        "report_balanced replaces the effort weight with model energy; other weights remain 0.25/0.20/0.20/0.15.\n",
        encoding="utf-8")
