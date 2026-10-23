"""Independent mechanical/reintegration checks; no curve-performance analysis."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"scripts"))
import numpy as np
import pandas as pd
from export_report_data import sha,save_json
from robot_planner.report_evaluation import score_report_candidates


def check_close(actual,expected,name,rtol=1e-9,atol=1e-9):
    a,b=np.asarray(actual,dtype=float),np.asarray(expected,dtype=float)
    if not np.allclose(a,b,rtol=rtol,atol=atol,equal_nan=True):
        raise AssertionError(name+": maximum discrepancy "+str(np.nanmax(np.abs(a-b))))


def grid_blocks(frame):
    if len(frame)%161:raise ValueError("Incomplete run blocks")
    ids=frame.run_id.to_numpy().reshape(-1,161)
    if not (ids==ids[:,0,None]).all():raise ValueError("Run block order differs")
    index=frame.sample_index.to_numpy().reshape(-1,161)
    if not (index==np.arange(161)).all():raise ValueError("Sample indices differ")
    return ids[:,0]


def verify(folder,refresh_archive=True):
    folder=Path(folder).resolve()
    checks={}
    manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    for f in manifest["files"]:
        path=folder/f["path"]
        if path.stat().st_size!=f["bytes"] or sha(path)!=f["sha256"]:
            raise AssertionError("Manifest mismatch "+f["path"])
    checks["manifest_files"]=len(manifest["files"])
    if any(p.suffix.lower() in (".png",".svg",".jpg",".pdf") for p in folder.rglob("*")):
        raise AssertionError("Unexpected visual output in raw data package")
    candidates=pd.read_parquet(folder/"candidate_metrics.parquet")
    scenarios=pd.read_parquet(folder/"scenarios.parquet")
    equations=pd.read_parquet(folder/"equations.parquet",columns=["run_id","scenario_id"])
    labels=pd.read_parquet(folder/"objective_winners.parquet")
    checks.update(scenarios=len(scenarios),candidates=len(candidates),objective_labels=len(labels))
    assert len(candidates)==42*len(scenarios)
    assert len(labels)==9*len(scenarios)
    assert not candidates.run_id.duplicated().any() and not scenarios.scenario_id.duplicated().any()
    assert set(candidates.run_id)==set(equations.run_id)
    assert candidates.data_split.eq("train").all() and scenarios.data_split.eq("train").all()
    numeric=candidates.select_dtypes(include="number").to_numpy(dtype=float)
    assert not np.isinf(numeric).any()
    old=pd.read_parquet(ROOT/"outputs/phase4_v3_analysis/train_analysis_metrics.parquet")
    old=old.set_index("run_id").loc[candidates.run_id]
    fields=["path_length_units","joint_velocity_effort","joint_acceleration_effort","joint_jerk_effort",
            "regularized_curvature_energy_inv_unit","unregularized_curvature_energy_inv_unit",
            "obstacle_penalty_s_per_unit2"]
    for field in fields:
        check_close(candidates[field],old[field],field,rtol=0,atol=0)
    np.testing.assert_array_equal(candidates.selection_valid,old.selection_valid)
    checks["source_metric_values_preserved"]=True
    byrun=candidates.set_index("run_id")
    time_count=geometry_count=0
    time_ids=set();geometry_ids=set()
    for path in sorted((folder/"time_samples").glob("*.parquet")):
        f=pd.read_parquet(path)
        ids=grid_blocks(f)
        if time_ids.intersection(ids):raise AssertionError("Repeated time run")
        time_ids.update(ids)
        expected=byrun.loc[ids]
        assert expected.selection_valid.all()
        t=f.time_s.to_numpy().reshape(-1,161)
        def integral(key):
            a=f[key].to_numpy(dtype=float).reshape(-1,161)
            return np.sum((a[:,1:]+a[:,:-1])*np.diff(t)/2,axis=1)
        for sample_key,metric in (
            ("net_power_w","net_mechanical_work_j"),
            ("positive_power_w","positive_mechanical_work_j"),
            ("copper_loss_power_w","motor_copper_loss_j")):
            check_close(integral(sample_key),expected["physics_161_"+metric],sample_key,rtol=2e-9,atol=1e-8)
        rms=np.sqrt(np.maximum(0,integral("torque_squared_nm2")/(t[:,-1]-t[:,0])))
        check_close(rms,expected.physics_161_rms_joint_torque_nm,"RMS replay")
        assert not any(k in f for k in ("geometry_u","dx_du"))
        time_count+=len(f)
    for path in sorted((folder/"geometry_samples").glob("*.parquet")):
        f=pd.read_parquet(path)
        ids=grid_blocks(f)
        if geometry_ids.intersection(ids):raise AssertionError("Repeated geometry run")
        geometry_ids.update(ids)
        u=f.geometry_u.to_numpy().reshape(-1,161)
        for key,metric in (
            ("path_speed_units_per_u","path_length_units"),
            ("regularized_curvature_integrand_inv_unit","regularized_curvature_energy_inv_unit")):
            a=f[key].to_numpy().reshape(-1,161)
            replay=np.sum((a[:,1:]+a[:,:-1])*np.diff(u)/2,axis=1)
            check_close(replay,byrun.loc[ids,metric],"Geometry "+metric,rtol=1e-10,atol=1e-8)
        assert "time_s" not in f
        geometry_count+=len(f)
    assert geometry_ids==set(candidates.run_id)
    assert time_ids==set(candidates.loc[candidates.selection_valid,"run_id"])
    assert time_count==161*int(candidates.selection_valid.sum())
    assert geometry_count==161*len(candidates)
    checks.update(time_samples=time_count,geometry_samples=geometry_count,grid_reintegration_passed=True)
    rebuilt=score_report_candidates(candidates.to_dict("records"))
    rebuilt=pd.DataFrame(rebuilt).sort_values(["scenario_id","objective"]).reset_index(drop=True)
    stored=labels.sort_values(["scenario_id","objective"]).reset_index(drop=True)
    for field in ("label_status","winner_run_id","tied_run_ids_json"):
        np.testing.assert_array_equal(rebuilt[field].fillna(""),stored[field].fillna(""))
    check_close(rebuilt.raw_metric_value,stored.raw_metric_value,"Label raw metric")
    checks["labels_replayed"]=len(labels)
    csv=pd.read_csv(folder/"candidate_metrics.csv.gz",nrows=12,float_precision="round_trip")
    source=candidates.set_index("run_id").loc[csv.run_id]
    for field in fields:check_close(csv[field],source[field],"CSV "+field,rtol=0,atol=0)
    result=dict(status="passed",checks=checks,charts_generated=0)
    save_json(folder/"independent_verification.json",result)
    paths=[p for p in folder.rglob("*") if p.is_file() and p.name!="manifest.json"]
    save_json(folder/"manifest.json",dict(dataset_version=manifest["dataset_version"],
              files=[dict(path=p.relative_to(folder).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)]))
    if refresh_archive:
        archive=folder.with_suffix(".zip")
        temp_archive=folder.parent/(folder.name+".verified.zip")
        with zipfile.ZipFile(temp_archive,"w",compression=zipfile.ZIP_STORED) as z:
            for p in sorted(folder.rglob("*")):
                if p.is_file():z.write(p,p.relative_to(folder.parent))
        temp_archive.replace(archive)
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:raise AssertionError("ZIP CRC failure")
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=ROOT/"outputs/report_aligned_v4_20261021")
    parser.add_argument("--no-zip",action="store_true")
    args=parser.parse_args()
    verify(args.data,not args.no_zip)
