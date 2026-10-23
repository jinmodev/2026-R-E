"""Train-only cleaned raw export plus versioned report physics. Never plots."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from robot_planner.curves import make_curve
from robot_planner.kinematics import PlanarArm2R
from robot_planner.report_math import report_equations, VERSION as EQUATION_VERSION
from robot_planner.report_physics import ReportPhysicsConfig,evaluate_physics,VERSION as PHYSICS_VERSION,METRICS
from robot_planner.report_evaluation import DATASET_VERSION,REPORT_WEIGHTS,score_report_candidates,skipped_physics

IDENTITY=["scenario_id","run_id","data_split"]
SOURCE_FIELDS=["curve_key","shape","elbow_mode","duration_s","samples","candidate_status","selection_valid",
               "path_check_status","joint_integration_status","geometry_convergence_status",
               "joint_derivative_reliable","path_regular","min_singularity_index",
               "path_length_units","joint_velocity_effort","joint_acceleration_effort","joint_jerk_effort",
               "regularized_curvature_energy_inv_unit","unregularized_curvature_energy_inv_unit",
               "obstacle_penalty_s_per_unit2","min_obstacle_clearance_units",
               "curvature_speed_epsilon_units_per_u"]
PHYSICS_FIELDS=["physics_version","physics_status","energy_selection_ready","torque_selection_ready",
                "work_energy_balance_passed","physics_error",*METRICS]
SUMMARY_FIELDS=[*IDENTITY,"dataset_version","source_dataset_version",*SOURCE_FIELDS,
                "source_balanced_cost_0_100","source_balanced_reward_0_100",
                *PHYSICS_FIELDS,"report_selection_valid","report_scenario_labels_ready",
                "report_balanced_cost_0_100","report_balanced_reward_0_100",
                *["report_normalized_"+k for k in REPORT_WEIGHTS],
                "physical_replay_consistent","max_velocity_replay_abs_error","max_acceleration_replay_abs_error",
                *[f"physics_{n}_{k}" for n in (161,321,641) for k in METRICS],
                "source_unreachable_witness","source_collision_witness"]
SCENARIO_FIELDS=["scenario_id","data_split","dataset_partition","scenario_type","seed","obstacle_layout",
                 "obstacle_count","base_x","base_y","link1","link2","start_x","start_y","goal_x","goal_y",
                 "link_ratio","chord_length","duration_s","candidate_count","source_valid_candidate_count",
                 "source_unresolved_candidate_count","report_energy_ready_count","report_torque_ready_count",
                 "report_scenario_labels_ready"]
EQUATION_FIELDS=[*IDENTITY,"curve_key","shape","elbow_mode","curve_equation","curve_parameters_json",
                 "equation_descriptor_json","x_u_equation","y_u_equation","progress_equation",
                 "x_t_equation","y_t_equation","theta1_t_equation","theta2_t_equation","ik_cosine_equation",
                 "report_equation_version","native_equation","native_first","native_second","native_length",
                 "native_curvature","native_area","analytic_length_units","native_area_units2","native_warning",
                 "report_equations_json"]
PHYS_SAMPLE=["torque1_nm","torque2_nm","joint1_power_w","joint2_power_w","net_power_w","positive_power_w",
             "absorbed_power_w","torque_squared_nm2","copper_loss_power_w","damping_power_w",
             "kinetic_energy_j","potential_energy_j"]
TIME_SOURCE=["sample_index","time_s","normalized_time","progress_u","x","y","theta1_rad","theta2_rad",
             "theta1_velocity_rad_s","theta2_velocity_rad_s","theta1_acceleration_rad_s2",
             "theta2_acceleration_rad_s2","theta1_jerk_rad_s3","theta2_jerk_rad_s3",
             "joint_velocity_squared_rad2_s2","joint_acceleration_squared_rad2_s4","joint_jerk_squared_rad2_s6",
             "obstacle_clearance","obstacle_penalty_integrand_per_unit2","joint_sample_status"]
TIME_FIELDS=[*IDENTITY,*TIME_SOURCE,*PHYS_SAMPLE,"physics_sample_status","energy_selection_ready","torque_selection_ready"]
GEOMETRY_SOURCE=["sample_index","geometry_u","geometry_x","geometry_y","dx_du","dy_du","d2x_du2","d2y_du2",
                 "path_speed_units_per_u","curvature_inv_unit","regularized_curvature_integrand_inv_unit",
                 "curvature_speed_epsilon_units_per_u","path_regular"]
GEOMETRY_FIELDS=[*IDENTITY,*GEOMETRY_SOURCE,"ordinary_curvature_inv_unit","ordinary_curvature_defined",
                 "source_candidate_status"]
LABEL_FIELDS=["scenario_id","objective","metric","label_status","source_valid_candidates","eligible_candidates",
              "winner_run_id","tie_count","tied_run_ids_json","raw_metric_value","dataset_version"]
OBSTACLE_FIELDS=["scenario_id","obstacle_id","center_x","center_y","radius_units"]
SOURCE_LABEL_FIELDS=["scenario_id","objective","winner_run_id","winner_status","tie_count","tied_run_ids_json",
                     "raw_metric_name","raw_metric_value","source_dataset_version"]

STRINGS=set(IDENTITY+["dataset_version","source_dataset_version","dataset_partition","scenario_type","obstacle_layout",
 "curve_key","elbow_mode","candidate_status","path_check_status","joint_integration_status","geometry_convergence_status",
 "physics_version","physics_status","physics_error","joint_sample_status","physics_sample_status","source_candidate_status",
 "objective","metric","label_status","winner_run_id","tied_run_ids_json","obstacle_id","winner_status","raw_metric_name",
 *[k for k in EQUATION_FIELDS if k not in ("shape","analytic_length_units","native_area_units2")]])
BOOLS=set(["selection_valid","joint_derivative_reliable","path_regular","energy_selection_ready","torque_selection_ready",
           "work_energy_balance_passed","report_selection_valid","report_scenario_labels_ready","physical_replay_consistent",
           "source_unreachable_witness","source_collision_witness","ordinary_curvature_defined"])
INTS=set(["samples","seed","obstacle_count","candidate_count","source_valid_candidate_count","source_unresolved_candidate_count",
          "report_energy_ready_count","report_torque_ready_count","sample_index","source_valid_candidates","eligible_candidates","tie_count"])


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def save_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


class TableWriter:
    def __init__(self,path,fields):
        self.path=Path(path);self.fields=fields;self.count=0
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.schema=pa.schema([(k,pa.string() if k in STRINGS else pa.bool_() if k in BOOLS else pa.int64() if k in INTS else pa.float64()) for k in fields])
        self.parquet=pq.ParquetWriter(self.path.with_suffix(".parquet"),self.schema,compression="zstd")
        self.csv=gzip.open(self.path.with_suffix(".csv.gz"),"wt",encoding="utf-8-sig",newline="",compresslevel=1)

    def append(self,rows):
        frame=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
        if frame.empty:return
        frame=frame.reindex(columns=self.fields)
        # Nullable Arrow types prevent an all-blank first batch from fixing a null schema.
        table=pa.Table.from_pandas(frame,schema=self.schema,preserve_index=False,safe=True)
        self.parquet.write_table(table)
        # Round-trippable numbers; one variable per column, no unit strings in numeric cells.
        frame.to_csv(self.csv,index=False,header=self.count==0,na_rep="",float_format="%.17g")
        self.count+=len(frame)

    def close(self):
        if self.count==0:
            pd.DataFrame(columns=self.fields).to_csv(self.csv,index=False)
        self.parquet.close();self.csv.close()


def build(source,out,limit=None,make_zip=True,shard_index=0,shard_count=1):
    source,out=Path(source),Path(out)
    if out.exists():
        raise ValueError("Choose a new output directory; existing data will not be overwritten")
    if shutil.disk_usage(out.parent if out.parent.exists() else ROOT).free < 5*1024**3:
        raise ValueError("At least 5 GiB free disk space required")
    state=json.loads((source/"generation_state.json").read_text(encoding="utf-8"))
    if state["status"]!="verified_complete" or state["dataset_version"]!="phase4-mathematics-v3":
        raise ValueError("Expected mechanically verified v3 source")
    markers=sorted((source/"partitions/train").glob("*/done.json"))
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Invalid shard index/count")
    markers=markers[shard_index::shard_count]
    if limit:markers=markers[:limit]
    if not markers:raise ValueError("No train partitions")
    out.mkdir(parents=True)
    cfg=ReportPhysicsConfig()
    code_hashes={p:sha(ROOT/p) for p in ["scripts/export_report_data.py","src/robot_planner/report_math.py",
                 "src/robot_planner/report_physics.py","src/robot_planner/report_evaluation.py"]}
    start=time.monotonic()
    common={
        "candidate_metrics":SUMMARY_FIELDS,"scenarios":SCENARIO_FIELDS,"equations":EQUATION_FIELDS,
        "objective_winners":LABEL_FIELDS,"source_winners":SOURCE_LABEL_FIELDS,"obstacles":OBSTACLE_FIELDS,
    }
    writers={name:TableWriter(out/name,fields) for name,fields in common.items()}
    time_writer=geometry_writer=None
    source_files=[]
    counts=dict(scenarios=0,candidates=0,source_accepted=0,source_invalid=0,source_unresolved=0,
                time_samples=0,geometry_samples=0,physics_screened=0,physics_unresolved=0,
                replay_inconsistent=0,energy_ready=0,torque_ready=0)
    scenario_ids=set();run_ids=set()
    try:
        for index,marker in enumerate(markers):
            receipt=json.loads(marker.read_text(encoding="utf-8"))
            sid=receipt["scenario_id"]
            if receipt["data_split"]!="train" or sid in scenario_ids:
                raise ValueError("Split mismatch/duplicate scenario")
            scenario_ids.add(sid)
            folder=marker.parent/receipt["attempt"]
            for name in ("trajectory_metrics.jsonl.gz","trajectory_samples.parquet","scenario_winners.json"):
                p=folder/name;expected=next(f for f in receipt["files"] if f["path"]==name)
                actual=dict(path=p.relative_to(source).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
                if actual["bytes"]!=expected["bytes"] or actual["sha256"]!=expected["sha256"]:
                    raise ValueError("Source integrity mismatch: "+str(p))
                source_files.append(actual)
            with gzip.open(folder/"trajectory_metrics.jsonl.gz","rt",encoding="utf-8") as f:
                source_rows=[json.loads(z) for z in f]
            if len(source_rows)!=42 or len({r["run_id"] for r in source_rows})!=42:
                raise ValueError("Expected 42 unique candidates")
            for r in source_rows:
                if r["data_split"]!="train" or r["scenario_id"]!=sid or r["run_id"] in run_ids:
                    raise ValueError("Candidate split/key mismatch")
                run_ids.add(r["run_id"])
            frame=pq.read_table(folder/"trajectory_samples.parquet",columns=list(dict.fromkeys(IDENTITY+TIME_SOURCE+GEOMETRY_SOURCE))).to_pandas()
            if len(frame)!=receipt["samples"] or set(frame.run_id)!=set(r["run_id"] for r in source_rows):
                raise ValueError("Sample count/join mismatch")
            if frame.duplicated(["run_id","sample_index"]).any():
                raise ValueError("Duplicate time sample key")
            if not frame.data_split.eq("train").all() or not frame.scenario_id.eq(sid).all():
                raise ValueError("Sample split/scenario mismatch")
            if index%50==0:
                if time_writer:
                    time_writer.close();geometry_writer.close()
                part=f"part-{index//50:03d}"
                time_writer=TableWriter(out/"time_samples"/part,TIME_FIELDS)
                geometry_writer=TableWriter(out/"geometry_samples"/part,GEOMETRY_FIELDS)
            records,equations,physical_by_id=[],[],{}
            first=source_rows[0]
            arm=PlanarArm2R(first["link1"],first["link2"],(first["base_x"],first["base_y"]))
            for old in source_rows:
                rid=old["run_id"]
                curve=make_curve(old["curve_key"],(old["start_x"],old["start_y"]),(old["goal_x"],old["goal_y"]),
                                 shape=old["shape"],samples=161)
                base={k:old.get(k) for k in IDENTITY+SOURCE_FIELDS}
                base.update(dataset_version=DATASET_VERSION,source_dataset_version=old["dataset_version"],
                            source_balanced_cost_0_100=old["balanced_cost_0_100"],
                            source_balanced_reward_0_100=old["balanced_reward_0_100"],
                            source_unreachable_witness=bool(old["path_check"]["unreachable_witnesses"]),
                            source_collision_witness=bool(old["path_check"]["collision_witnesses"]))
                counts["source_"+old["candidate_status"]]+=1
                physical=(evaluate_physics(arm,curve,old["duration_s"],old["elbow_mode"],cfg)
                          if old["selection_valid"] else skipped_physics("source_candidate_"+old["candidate_status"]))
                base.update(physical["metrics"])
                base.pop("physics_config_json",None)
                base.pop("physics_integration_method",None)
                for size,values in physical["grids"].items():
                    for key,value in values.items():base[f"physics_{size}_{key}"]=value
                samples=frame.loc[frame.run_id.eq(rid)].sort_values("sample_index")
                if len(samples)!=old["samples"] or not np.array_equal(samples.sample_index,np.arange(old["samples"])):
                    raise ValueError("Per-run sample count/order mismatch")
                if not np.allclose(samples.time_s,np.linspace(0,old["duration_s"],old["samples"]),rtol=0,atol=1e-12):
                    raise ValueError("Nonuniform source time grid")
                if old["selection_valid"]:
                    times=samples[IDENTITY+TIME_SOURCE].copy()
                    replay=False
                    if physical["samples"]:
                        st,vals=physical["samples"]
                        oldv=samples[["theta1_velocity_rad_s","theta2_velocity_rad_s"]].to_numpy(dtype=float)
                        olda=samples[["theta1_acceleration_rad_s2","theta2_acceleration_rad_s2"]].to_numpy(dtype=float)
                        base["max_velocity_replay_abs_error"]=float(np.max(np.abs(oldv-st["velocity"])))
                        base["max_acceleration_replay_abs_error"]=float(np.max(np.abs(olda-st["acceleration"])))
                        replay=np.allclose(oldv,st["velocity"],rtol=1e-6,atol=1e-7) and np.allclose(olda,st["acceleration"],rtol=1e-6,atol=1e-6)
                        for key in PHYS_SAMPLE:times[key]=vals[key]
                    else:
                        for key in PHYS_SAMPLE:times[key]=None
                    base["physical_replay_consistent"]=bool(replay)
                    if not replay:
                        base.update(energy_selection_ready=False,torque_selection_ready=False,physics_status="unresolved",
                                    physics_error="source derivative replay mismatch or missing physics samples")
                        counts["replay_inconsistent"]+=1
                    times["physics_sample_status"]="computed" if replay else "unresolved_replay"
                    times["energy_selection_ready"]=base["energy_selection_ready"]
                    times["torque_selection_ready"]=base["torque_selection_ready"]
                    time_writer.append(times)
                    counts["time_samples"]+=len(times)
                    counts["physics_"+base["physics_status"]]+=1
                    counts["energy_ready"]+=int(base["energy_selection_ready"])
                    counts["torque_ready"]+=int(base["torque_selection_ready"])
                records.append(base)
                eq=report_equations(curve,old["duration_s"])
                equation={k:old.get(k) for k in EQUATION_FIELDS}
                equation.update({k:eq.get(k) for k in EQUATION_FIELDS if k in eq})
                equation["report_equations_json"]=json.dumps(eq,ensure_ascii=False,sort_keys=True)
                for key in ("analytic_length_units","native_area_units2"):
                    if equation.get(key) is not None:equation[key]=float(equation[key])
                equations.append(equation)
            labels=score_report_candidates(records)
            writers["candidate_metrics"].append(records)
            writers["objective_winners"].append(labels)
            writers["equations"].append(equations)
            geom=frame[IDENTITY+GEOMETRY_SOURCE].copy()
            speeds=geom.path_speed_units_per_u.to_numpy()
            cross=geom.dx_du.to_numpy()*geom.d2y_du2.to_numpy()-geom.dy_du.to_numpy()*geom.d2x_du2.to_numpy()
            defined=speeds>1e-12*max(first["chord_length"],1e-12)
            k=np.full(len(geom),np.nan)
            np.divide(np.abs(cross),speeds**3,out=k,where=defined)
            geom["ordinary_curvature_inv_unit"]=k
            geom["ordinary_curvature_defined"]=defined
            geom["source_candidate_status"]=geom.run_id.map({r["run_id"]:r["candidate_status"] for r in source_rows})
            geometry_writer.append(geom)
            counts["geometry_samples"]+=len(geom)
            obs=json.loads(first["obstacles_json"])
            writers["obstacles"].append([dict(scenario_id=sid,**o) for o in obs])
            scenario={k:first.get(k) for k in SCENARIO_FIELDS}
            scenario.update(candidate_count=len(records),source_valid_candidate_count=sum(r["selection_valid"] for r in records),
                            source_unresolved_candidate_count=sum(r["candidate_status"]=="unresolved" for r in records),
                            report_energy_ready_count=sum(r.get("energy_selection_ready",False) for r in records),
                            report_torque_ready_count=sum(r.get("torque_selection_ready",False) for r in records),
                            report_scenario_labels_ready=records[0]["report_scenario_labels_ready"])
            writers["scenarios"].append([scenario])
            source_labels=json.loads((folder/"scenario_winners.json").read_text(encoding="utf-8"))
            writers["source_winners"].append([dict({k:r.get(k) for k in SOURCE_LABEL_FIELDS},
                                                   source_dataset_version=r["dataset_version"]) for r in source_labels])
            counts["scenarios"]+=1;counts["candidates"]+=42
            if index%50==0 or index+1==len(markers):
                print(json.dumps(dict(completed=index+1,total=len(markers),seconds=round(time.monotonic()-start,1),
                                      accepted=counts["source_accepted"],physics_unresolved=counts["physics_unresolved"])),flush=True)
    finally:
        for writer in writers.values():writer.close()
        if time_writer:time_writer.close();geometry_writer.close()
    if counts["candidates"]!=42*len(markers) or counts["time_samples"]!=161*counts["source_accepted"]:
        raise ValueError("Final row counts do not reconcile")
    if counts["geometry_samples"]!=161*counts["candidates"]:
        raise ValueError("Geometry row count does not reconcile")
    save_json(out/"source_manifest.json",dict(source_dataset_version=state["dataset_version"],
              source_generation_signature=state["signature"]["digest"],files=source_files))
    save_json(out/"experiment_metadata.json",dict(dataset_version=DATASET_VERSION,source_dataset_version=state["dataset_version"],
              scope="train_first_n_smoke" if limit else "train_shard" if shard_count>1 else "train_only",
              shard_index=shard_index,shard_count=shard_count,randomness="none; deterministic source order",
              physics_config=asdict(cfg),physics_version=PHYSICS_VERSION,equation_version=EQUATION_VERSION,
              headline_physics_samples=641,physics_screening_samples=[161,321,641],exported_time_samples_per_run=161,
              report_weights=REPORT_WEIGHTS,tie_tolerance=dict(rtol=1e-12,atol=1e-12),
              missing_value="blank/Arrow null; never numeric zero as a substitute",
              time_scope="source selection_valid candidates only; all candidate outcomes remain in candidate_metrics",
              geometry_scope="all candidates; geometry grid differs from time grid",
              code_sha256=code_hashes,
              source_read_policy="no validation/test/challenge outcomes read",python=platform.python_version()))
    dictionaries=[]
    for name,fields in {**common,"time_samples":TIME_FIELDS,"geometry_samples":GEOMETRY_FIELDS}.items():
        for key in fields:
            unit=unit_of(key)
            dictionaries.append(dict(table=name,column=key,dtype="string" if key in STRINGS else "boolean" if key in BOOLS else "integer" if key in INTS else "number",
                                     unit=unit,description=description_of(key),join_key="scenario_id" if key=="scenario_id" else "run_id" if key=="run_id" else "",
                                     missing_reason=missing_reason(key,name)))
    pd.DataFrame(dictionaries).to_csv(out/"data_dictionary.csv",index=False,encoding="utf-8-sig")
    save_json(out/"data_dictionary.json",dictionaries)
    save_json(out/"quality_checks.json",dict(status="passed",counts=counts,elapsed_seconds=time.monotonic()-start,
              source_hashes_checked=len(source_files),scenario_ids_unique=len(scenario_ids),run_ids_unique=len(run_ids),
              sample_keys_unique=True,source_split_leakage=False,source_values_preserved=True,
              finite_physics_failures_not_imputed=True,charts_generated=0,
              limitations="Finite-grid physics convergence is not an exact-error bound; unreliable added objectives have no ready winner. No inferential analysis performed."))
    (out/"README.md").write_text(readme(counts),encoding="utf-8")
    (out/"load_data.py").write_text(LOADER,encoding="utf-8")
    files=[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
           for p in sorted(out.rglob("*")) if p.is_file()]
    save_json(out/"manifest.json",dict(dataset_version=DATASET_VERSION,files=files))
    if make_zip:
        archive=out.with_suffix(".zip")
        if archive.exists():raise ValueError("Archive already exists")
        with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_STORED) as z:
            for p in sorted(out.rglob("*")):
                if p.is_file():z.write(p,p.relative_to(out.parent))
        print("ARCHIVE "+str(archive),flush=True)
    print(json.dumps(dict(output=str(out),counts=counts,seconds=round(time.monotonic()-start,1))),flush=True)
    return counts


def unit_of(key):
    if "normalized" in key or key in ("progress_u","geometry_u","shape","min_singularity_index","link_ratio"):return "dimensionless"
    if key.endswith("_nm2"):return "(N*m)^2"
    if key.endswith("_nm"):return "N*m"
    if key.endswith("_j"):return "J (model)"
    if key.endswith("_w"):return "W (model)"
    if "rad2_s6" in key:return "rad^2/s^6"
    if "rad2_s4" in key:return "rad^2/s^4"
    if "rad2_s2" in key:return "rad^2/s^2"
    if key.endswith("rad_s3"):return "rad/s^3"
    if key.endswith("rad_s2"):return "rad/s^2"
    if key.endswith("rad_s"):return "rad/s"
    if key.endswith("_rad"):return "rad"
    if key in ("duration_s","time_s"):return "s"
    if key=="joint_velocity_effort":return "rad^2/s"
    if key=="joint_acceleration_effort":return "rad^2/s^3"
    if key=="joint_jerk_effort":return "rad^2/s^5"
    if key=="obstacle_penalty_s_per_unit2":return "s/unit^2"
    if "inv_unit" in key:return "1/unit"
    if key=="obstacle_penalty_integrand_per_unit2":return "1/unit^2"
    if "cost_0_100" in key or "reward_0_100" in key:return "score"
    if key=="native_area_units2":return "unit^2 (additional geometry, not an objective)"
    if key in ("base_x","base_y","start_x","start_y","goal_x","goal_y","link1","link2","chord_length",
               "center_x","center_y","radius_units","path_length_units","analytic_length_units","min_obstacle_clearance_units",
               "x","y","geometry_x","geometry_y","obstacle_clearance"):return "workspace unit"
    if key in ("dx_du","dy_du","d2x_du2","d2y_du2","path_speed_units_per_u","curvature_speed_epsilon_units_per_u"):return "unit per dimensionless u"
    return ""


def description_of(key):
    descriptions={
        "selection_valid":"Original v3 mathematical/path candidate validity; independent of added physics screening.",
        "report_selection_valid":"Original valid candidate whose model energy passed the added numerical screen.",
        "report_scenario_labels_ready":"No unresolved source candidate and all source-valid candidates energy-ready.",
        "physics_status":"screened / unresolved / not_evaluated. Screening is finite-grid evidence.",
        "energy_selection_ready":"Positive work and copper/total model energy converge 321-to-641 and work-energy identity passes.",
        "torque_selection_ready":"Torque RMS converges 321-to-641 and work-energy identity passes.",
        "net_mechanical_work_j":"Signed integral of sum(tau_j*theta_j_dot), 641 uniform-t points.",
        "positive_mechanical_work_j":"Integral of sum(max(tau_j*theta_j_dot,0)), by joint, no regenerative credit.",
        "absorbed_mechanical_work_j":"Integral of sum(max(-tau_j*theta_j_dot,0)); positive magnitude.",
        "model_electrical_energy_j":"Positive mechanical work plus resistive copper loss, under experiment_metadata assumptions.",
        "rms_joint_torque_nm":"sqrt(integral(tau1^2+tau2^2)dt/T), 641 points; vector RMS, not sum of individual RMS.",
        "work_energy_residual_j":"Signed work minus Delta(kinetic+potential) minus viscous damping loss.",
        "curvature_inv_unit":"Original v3 curvature column: blank when geometric speed is below source epsilon.",
        "ordinary_curvature_inv_unit":"Pointwise ordinary curvature from determinant/speed^3; near-zero speed -> null.",
        "ordinary_curvature_defined":"Geometric speed > 1e-12 times chord scale; does not imply finite total ordinary bending cost.",
        "path_regular":"Original run-level flag; cycloid cusps make this false even when path is selection_valid.",
        "geometry_u":"Cosine-clustered geometric parameter; distinct from progress_u at the same sample_index.",
        "progress_u":"q(time_s/T), using the shared quintic time law.",
        "dx_du":"Geometry derivative evaluated at geometry_u, NOT at progress_u.",
        "dy_du":"Geometry derivative evaluated at geometry_u, NOT at progress_u.",
        "d2x_du2":"Second geometry derivative at geometry_u.",
        "d2y_du2":"Second geometry derivative at geometry_u.",
        "regularized_curvature_integrand_inv_unit":"cross(p_u,p_uu)^2/(||p_u||^2+epsilon^2)^(5/2), integrate over geometry_u.",
        "label_status":"ready / unresolved_source_alternative / unresolved_physics_alternative / no_valid_candidate.",
        "raw_metric_value":"Winner's metric only when label_status=ready; units given by metric field.",
        "report_balanced_cost_0_100":"New weighted min-max cost, with modeled energy instead of source effort; see metadata.",
        "report_balanced_reward_0_100":"100 minus new cost; not comparable across candidate pools/scenarios.",
        "source_balanced_cost_0_100":"Unmodified v3 mathematical composite; source effort term is not physical energy.",
        "source_balanced_reward_0_100":"Unmodified v3 reward, retained separately from new energy composite.",
        "native_area_units2":"Additional hand-calculation geometry; not a path-planning objective or swept robot area.",
        "theta1_t_equation":"Source composed IK formula, with x(t),y(t), time law and continuity policy in descriptor.",
        "theta2_t_equation":"Source branch-specific inverse cosine IK composition.",
        "physical_replay_consistent":"Vectorized exact derivatives agree with source v3 161-point derivatives at documented tolerances.",
        "source_unreachable_witness":"Adaptive path check found a witness. False does not prove global reachability.",
        "source_collision_witness":"Adaptive path check found a collision witness. Early stopping can mask other failures.",
    }
    if key in descriptions:return descriptions[key]
    if key.startswith("physics_") and key.split("_")[1] in ("161","321","641"):
        bits=key.split("_");return f"{bits[1]}-point estimate of "+description_of("_".join(bits[2:]))
    if key.startswith("report_normalized_"):return "Scenario-relative min-max version of "+key[len("report_normalized_"):]
    if key.startswith("theta") and "velocity" in key:return "Exact time derivative from source differential IK, not slope-parameter derivative."
    if key.startswith("theta") and "acceleration" in key:return "Exact second time derivative from source differential IK."
    if key.startswith("theta") and "jerk" in key:return "Exact third time derivative from source differential IK."
    if key.endswith("_power_w"):return "Instantaneous modeled power at time_s; integrate over time, not geometry_u."
    if key in ("kinetic_energy_j","potential_energy_j"):return "Instantaneous modeled energy relative to robot base height; endpoint difference is physically relevant."
    if key.endswith("_json"):return "JSON text preserving exact structured source/equation metadata; parse with json.loads."
    return key.replace("_"," ")+". See README and experiment_metadata for table grain, source and conventions."


def missing_reason(key,table):
    if table=="objective_winners":return "No ready label -> winner/tie/value blank; inspect label_status."
    if table=="equations":return "Family-specific equation or closed-form result not applicable."
    if key in PHYSICS_FIELDS or key.startswith("physics_"):return "Source invalid/unresolved -> not_evaluated; singularity/numerical failure -> unresolved. Inspect physics_error."
    if key=="unregularized_curvature_energy_inv_unit":return "Cusp/nonregular path -> unavailable; do not replace with finite K_epsilon."
    if key in ("min_obstacle_clearance_units","obstacle_clearance"):return "No obstacle -> not applicable."
    if "curvature" in key:return "Zero/near-zero geometric speed or nonregular total integral; inspect flags."
    return "Blank means unavailable/not applicable, not numeric zero. Invalid candidate joint costs remain blank."


def readme(counts):
    return f"""# 보고서 연계 정제 원자료

버전 {DATASET_VERSION}. 원자료 phase4-mathematics-v3의 train만 사용했다.
시나리오 {counts['scenarios']:,}개, 후보 {counts['candidates']:,}개.
이 패키지는 원자료와 계산·상태를 제공하며 그래프나 통계적 우열 분석을 만들지 않는다.

## 파일

- scenarios: 상황 1개당 1행. 팔 길이·위치·장애물 수·유효 후보 수.
- obstacles: 장애물 1개당 1행. scenario_id로 연결.
- candidate_metrics: 후보 1개당 1행. 실패·미확정도 보존. source_* 점수와 새 report_* 점수를 구분.
- equations: 후보 1개당 1행. 곡선·관절각 합성식, 정밀 파라미터, 보고서용 원형식·미분식.
- objective_winners: 상황·목적당 1행. label_status=ready일 때만 승자 해석 가능.
- source_winners: 기존 v3 라벨을 그대로 별도 보존.
- time_samples/part-*: v3 selection_valid 후보만, 실행당 161행. 총 {counts['time_samples']:,}행.
- geometry_samples/part-*: 모든 후보, 실행당 161행. 총 {counts['geometry_samples']:,}행.
- 각 표는 typed Parquet 및 UTF-8 BOM CSV.gz로 제공한다. 압축을 풀거나 pandas.read_csv로 바로 읽을 수 있다.
- data_dictionary.csv/JSON: 열별 단위·타입·의미·결측 사유.
- experiment_metadata.json: 가정·가중치·수치 기준·소스 버전·코드 해시.
- source_manifest.json, manifest.json, quality_checks.json: 출처/파일 해시와 기계적 검사 기록.

## 분석할 때

후보 요약과 식은 run_id로, 입력·장애물은 scenario_id로 연결한다.
한 scenario_id의 후보들은 대응 비교 대상이며 독립 표본이 아니다.
후보 수가 많다는 사실이 과적합을 해결하지 않는다.
최적은 준비된 42개 후보 중 유효 후보라는 범위에 한정된다.
validation/test/challenge 평가값은 열지 않았다.

## 시간과 기하 좌표

time_samples의 time_s, progress_u, x/y, theta는 uniform-t 표본이다.
geometry_samples의 geometry_u, geometry_x/y, dx_du 등은 cosine-clustered-u 표본이다.
두 표의 같은 sample_index가 같은 경로 위치를 뜻하지 않는다.
시간 비용은 time_s로, 길이·K_epsilon은 geometry_u로 적분한다.
ordinary_curvature_inv_unit는 올바른 외적 곡률이고, curvature_inv_unit는 원자료 epsilon 마스킹값이다.
사이클로이드 끝점의 일반 곡률은 비어 있다. 그 경로의 일반 곡률 제곱 적분과 정칙화 적분은 서로 다르다.

## 토크·에너지 가정

수직 평면의 2관절 가는 강체 링크와 점성마찰·직결 모터를 가정한다.
1 unit=0.1 m, 선밀도 0.35 kg/m, 감쇠 0.03 N*m*s/rad, Kt=0.12 N*m/A,
저항 0.8 ohm, 중력 9.80665 m/s^2. 실측값이 아닌 명시적 모형 가정이다.
관절 운동 노력도 E_theta는 rad^2/s이고 실제 에너지 J가 아니다.
W_net는 부호 있는 기계적 일, W_positive는 관절별 양의 일만 더한 값이다.
모델 전기에너지=W_positive+저항 발열이며 회생에너지 회수는 가정하지 않았다.
기계적 일은 Delta(운동+위치에너지)+마찰손실과 비교해 검산했다.
일은 참고량으로 보존하며, 부호 있는 일을 최소화하는 승자를 따로 만들지 않았다.

새 물리지표는 161/321/641점으로 계산했다. 대표값은 641점,
321→641 변화는 에너지 2%, RMS 토크 1%, 절대허용 1e-6과 일-에너지 검사를 사용했다.
이 기준은 생성 전 코드에 고정했다. 641점도 정확한 참값은 아니다.
time_samples의 새 토크·전력은 161점이므로 요약의 641점 적분과 근사 차이가 날 수 있다.
정확한 비교는 candidate_metrics의 physics_161_* 값과 하며, 수렴성은 321/641을 함께 확인한다.
불안정·실패는 채워 넣지 않고 원인과 readiness를 남긴다.

## 종합점수

원래 v3 점수는 source_balanced_*에 보존했다.
새 점수는 유효하고 에너지 검증을 통과한 후보 안에서 각 비용을 min-max 정규화했다.
거리 .25, 모형 전기에너지 .20, 관절 저크 .20, 정칙화 곡률 .20, 접근 벌점 .15이다.
가중치는 기존 노력도 항 .20을 명시적 에너지 항으로 바꾼 것으로, 통계적으로 학습한 값이 아니다.
후보군마다 정규화 범위가 다르므로 점수를 시나리오 간 절대 성능으로 해석하지 않는다.
미확정 원자료 후보나 해당 목적의 물리 계산 미해결 후보가 남으면 확정 승자를 발표하지 않는다.
그 경우 후보별 새 점수는 잠정 비교값이며 report_scenario_labels_ready=false다.

## 실행과 보존

프로젝트에서 .venv/Scripts/python.exe scripts/export_report_data.py --output 새폴더 로 재생성한다.
기존 폴더에는 덮어쓰지 않는다. 원자료 후보·시간법칙·분할·기존 평가값을 바꾸지 않았다.
source_manifest의 해시는 실제로 읽은 train 입력 파일에 대응한다.
CSV의 빈 셀은 0이 아니다. 크기가 큰 표는 Excel 한 시트에 들어가지 않으므로 Parquet 또는 분할 CSV를 사용한다.
load_data.py에는 파일 읽기·결합 예시만 있으며 통계 분석·그래프 생성은 실행하지 않는다.
"""


LOADER='''"""Read/join examples only. No statistical analysis or figures."""
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent

def candidates():
    return pd.read_parquet(ROOT/"candidate_metrics.parquet")

def inputs():
    return pd.read_parquet(ROOT/"scenarios.parquet")

def equations():
    return pd.read_parquet(ROOT/"equations.parquet")

def candidate_inputs():
    return candidates().merge(inputs(),on=["scenario_id","data_split"],validate="many_to_one")

def iter_time_samples(columns=None):
    for path in sorted((ROOT/"time_samples").glob("*.parquet")):
        yield pd.read_parquet(path,columns=columns)

def iter_geometry_samples(columns=None):
    for path in sorted((ROOT/"geometry_samples").glob("*.parquet")):
        yield pd.read_parquet(path,columns=columns)

if __name__=="__main__":
    print("Import this module or call the functions to load data. No analysis is run automatically.")
'''


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=ROOT/"data/phase4_v3_release")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/report_aligned_v4_20261021")
    parser.add_argument("--limit",type=int,help="Development smoke run only; final export omits this.")
    parser.add_argument("--no-zip",action="store_true")
    parser.add_argument("--shard-index",type=int,default=0)
    parser.add_argument("--shard-count",type=int,default=1)
    args=parser.parse_args()
    if args.limit is not None and args.limit<=0:parser.error("--limit must be positive")
    build(args.source,args.output,args.limit,not args.no_zip,args.shard_index,args.shard_count)
