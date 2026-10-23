"""Build disjoint train shards concurrently, then package one tidy dataset."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from export_report_data import TableWriter,readme,save_json,sha
import pyarrow.parquet as pq
import pandas as pd


def build(output,workers=6,source=None):
    output=Path(output).resolve()
    source=Path(source or ROOT/"data/phase4_v3_release").resolve()
    if output.exists():raise ValueError("Final directory already exists; use a new directory")
    working=ROOT/"tmp"/(output.name+"_shards")
    if working.exists():raise ValueError("Working directory exists; inspect it before restarting")
    working.mkdir(parents=True)
    started=time.monotonic()
    processes=[]
    env=dict(os.environ,OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
    for i in range(workers):
        log=(working/f"shard-{i:02d}.log").open("w",encoding="utf-8")
        folder=working/f"shard-{i:02d}"
        command=[sys.executable,str(ROOT/"scripts/export_report_data.py"),"--source",str(source),
                 "--output",str(folder),"--no-zip","--shard-index",str(i),"--shard-count",str(workers)]
        proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        processes.append((proc,log,folder))
    try:
        while True:
            statuses=[p.poll() for p,_,_ in processes]
            if any(s is not None and s!=0 for s in statuses):
                raise RuntimeError("A shard failed; see "+str(working))
            completed=sum(s==0 for s in statuses)
            print(json.dumps(dict(stage="export",completed_shards=completed,total_shards=workers,
                                  elapsed_seconds=round(time.monotonic()-started))),flush=True)
            if completed==workers:break
            time.sleep(20)
    finally:
        for proc,log,_ in processes:
            if proc.poll() is None:
                proc.terminate()
            log.close()
    folders=[f for _,_,f in processes]
    output.mkdir(parents=True)
    metadata=[json.loads((f/"experiment_metadata.json").read_text(encoding="utf-8")) for f in folders]
    for m in metadata[1:]:
        for key in ("physics_config","report_weights","code_sha256","source_dataset_version"):
            if m[key]!=metadata[0][key]:raise ValueError("Shard metadata differs: "+key)
    code=output/"provenance/code";code.mkdir(parents=True)
    for relative,expected in metadata[0]["code_sha256"].items():
        path=ROOT/relative
        if sha(path)!=expected:raise ValueError("Evaluation code changed during generation: "+relative)
        shutil.copy2(path,code/path.name)
    shutil.copy2(ROOT/"scripts/build_report_dataset.py",code/"build_report_dataset.py")
    counts={}
    for folder in folders:
        quality=json.loads((folder/"quality_checks.json").read_text(encoding="utf-8"))
        if quality["status"]!="passed":raise ValueError("Shard quality failure")
        for key,value in quality["counts"].items():counts[key]=counts.get(key,0)+value
    expected=json.loads((source/"generation_state.json").read_text(encoding="utf-8"))["split_counts"]["train"]
    if counts["scenarios"]!=expected or counts["candidates"]!=42*expected:
        raise ValueError("Combined export is incomplete")
    names=["candidate_metrics","scenarios","equations","objective_winners","source_winners","obstacles"]
    for name in names:
        fields=pq.ParquetFile(folders[0]/(name+".parquet")).schema_arrow.names
        writer=TableWriter(output/name,fields)
        try:
            for folder in folders:
                for batch in pq.ParquetFile(folder/(name+".parquet")).iter_batches(batch_size=2048):
                    writer.append(batch.to_pandas())
        finally:writer.close()
        print("MERGED "+name,flush=True)
    for name in ("time_samples","geometry_samples"):
        target=output/name;target.mkdir()
        for i,folder in enumerate(folders):
            for file in sorted((folder/name).iterdir()):
                shutil.copy2(file,target/f"shard-{i:02d}-{file.name}")
    merged_source=[]
    for folder in folders:
        merged_source.extend(json.loads((folder/"source_manifest.json").read_text(encoding="utf-8"))["files"])
    if len({r["path"] for r in merged_source})!=len(merged_source):
        raise ValueError("Shards overlap source partitions")
    sm=json.loads((folders[0]/"source_manifest.json").read_text(encoding="utf-8"));sm["files"]=merged_source
    save_json(output/"source_manifest.json",sm)
    finalmeta=metadata[0]
    finalmeta.update(scope="train_only",shard_count=workers,shard_index=None,
                     generation_method="disjoint scenario_id shards; deterministic merge; no held-out outcomes")
    save_json(output/"experiment_metadata.json",finalmeta)
    for name in ("data_dictionary.csv","data_dictionary.json","load_data.py"):
        shutil.copy2(folders[0]/name,output/name)
    # Verify joins and finite summary fields without inspecting outcome distributions.
    candidates=pd.read_parquet(output/"candidate_metrics.parquet")
    scenarios=pd.read_parquet(output/"scenarios.parquet")
    equations=pd.read_parquet(output/"equations.parquet",columns=["run_id","scenario_id"])
    if candidates.run_id.duplicated().any() or scenarios.scenario_id.duplicated().any():
        raise ValueError("Duplicate merged IDs")
    if set(candidates.run_id)!=set(equations.run_id) or set(candidates.scenario_id)!=set(scenarios.scenario_id):
        raise ValueError("Merged key mismatch")
    if not candidates.data_split.eq("train").all():raise ValueError("Unexpected split")
    save_json(output/"quality_checks.json",dict(status="passed",counts=counts,
              elapsed_seconds=time.monotonic()-started,source_hashes_checked=len(merged_source),
              scenario_ids_unique=len(scenarios),run_ids_unique=len(candidates),sample_keys_unique=True,
              source_split_leakage=False,source_values_preserved=True,charts_generated=0,
              limitations="Finite-grid convergence is not an exact-error bound. Physics-unresolved alternatives retain no ready label. No inferential analysis."))
    (output/"README.md").write_text(readme(counts),encoding="utf-8")
    files=[dict(path=p.relative_to(output).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
           for p in sorted(output.rglob("*")) if p.is_file()]
    save_json(output/"manifest.json",dict(dataset_version=finalmeta["dataset_version"],files=files))
    archive=output.with_suffix(".zip")
    if archive.exists():raise ValueError("Archive already exists")
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_STORED) as z:
        for p in sorted(output.rglob("*")):
            if p.is_file():z.write(p,p.relative_to(output.parent))
    print(json.dumps(dict(stage="complete",output=str(output),archive=str(archive),counts=counts,
                          elapsed_seconds=time.monotonic()-started)),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/report_aligned_v4_20261021")
    parser.add_argument("--workers",type=int,default=6)
    parser.add_argument("--source",type=Path,default=ROOT/"data/phase4_v3_release")
    args=parser.parse_args()
    if not 1<=args.workers<=12:parser.error("workers must be in [1,12]")
    build(args.output,args.workers,args.source)
