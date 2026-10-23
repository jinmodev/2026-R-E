"""Create smaller flat entry tables and report-case extracts; no statistics/plots."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import pandas as pd
from export_report_data import description_of,unit_of,sha,save_json


def prepare(folder):
    folder=Path(folder)
    candidates=pd.read_parquet(folder/"candidate_metrics.parquet")
    scenarios=pd.read_parquet(folder/"scenarios.parquet")
    candidate_columns=["scenario_id","run_id","data_split","curve_key","shape","elbow_mode","duration_s",
        "candidate_status","selection_valid","physics_status","energy_selection_ready","torque_selection_ready",
        "report_selection_valid","report_scenario_labels_ready","source_unreachable_witness","source_collision_witness",
        "path_length_units","joint_velocity_effort","joint_acceleration_effort","joint_jerk_effort",
        "regularized_curvature_energy_inv_unit","obstacle_penalty_s_per_unit2",
        "net_mechanical_work_j","positive_mechanical_work_j","motor_copper_loss_j",
        "model_electrical_energy_j","rms_joint_torque_nm","source_balanced_reward_0_100",
        "report_balanced_reward_0_100"]
    scenario_columns=["scenario_id","data_split","base_x","base_y","link1","link2","start_x","start_y",
                      "goal_x","goal_y","obstacle_count","obstacle_layout"]
    ready=candidates[candidate_columns].merge(scenarios[scenario_columns],
                                             on=["scenario_id","data_split"],validate="many_to_one")
    assert len(ready)==len(candidates)
    ready.to_csv(folder/"analysis_ready.csv",index=False,encoding="utf-8-sig",float_format="%.17g")
    ready.to_parquet(folder/"analysis_ready.parquet",index=False,compression="zstd")
    for name in ("scenarios","obstacles","objective_winners"):
        pd.read_parquet(folder/(name+".parquet")).to_csv(folder/(name+".csv"),index=False,
            encoding="utf-8-sig",float_format="%.17g")
    for suffix in ("csv","json"):
        target=folder/("data_dictionary."+suffix)
        if suffix=="csv":
            dictionary=pd.read_csv(target).fillna("")
            if not dictionary.table.eq("analysis_ready").any():
                rows=[]
                for c in ready:
                    original=dictionary.loc[dictionary.column.eq(c)]
                    source=original.iloc[0].to_dict() if not original.empty else dict(column=c,dtype=str(ready[c].dtype),
                        unit=unit_of(c),description=description_of(c),join_key="",missing_reason="")
                    source["table"]="analysis_ready";rows.append(source)
                dictionary=pd.concat([dictionary,pd.DataFrame(rows)],ignore_index=True)
                dictionary.to_csv(target,index=False,encoding="utf-8-sig")
                (folder/"data_dictionary.json").write_text(dictionary.to_json(orient="records",force_ascii=False,indent=2),encoding="utf-8")
    case=folder/"report_cases";case.mkdir(exist_ok=True)
    ids={"P4S000691","P4S000057"}
    selected=set(candidates.loc[candidates.scenario_id.isin(ids),"run_id"])
    for name in ("candidate_metrics","equations","objective_winners","scenarios","obstacles"):
        f=pd.read_parquet(folder/(name+".parquet"))
        f[f.scenario_id.isin(ids)].to_csv(case/(name+".csv"),index=False,encoding="utf-8-sig",float_format="%.17g")
    for name in ("time_samples","geometry_samples"):
        pieces=[]
        for path in sorted((folder/name).glob("*.parquet")):
            import pyarrow.parquet as pq
            table=pq.read_table(path,filters=[("scenario_id","in",sorted(ids))])
            if len(table):pieces.append(table.to_pandas())
        f=pd.concat(pieces,ignore_index=True)
        f.to_csv(case/(name+".csv"),index=False,encoding="utf-8-sig",float_format="%.17g")
    (folder/"먼저_읽어주세요.md").write_text(
        """# 직접 분석을 시작하는 순서

1. analysis_ready.csv를 먼저 연다. 상황 입력과 후보별 비용을 한 줄로 합친 가벼운 표다.
2. 곡선식·theta(t)가 필요하면 equations.parquet 또는 equations.csv.gz를 run_id로 연결한다.
3. 시간에 따른 움직임·토크·전력을 보려면 time_samples를, 길이·곡률 적분을 보려면 geometry_samples를 읽는다.
4. 상세 열 설명·단위·빈칸 사유는 data_dictionary.csv에 있다.

analysis_ready.csv에는 train의 모든 후보가 포함된다. 실패 후보를 임의로 지우지 않았다.
candidate_status/selection_valid는 기존 수학·경로 검사 상태다.
physics_status와 energy_selection_ready/torque_selection_ready는 새 물리 계산의 사용 가능 상태다.
report_scenario_labels_ready=false이면 새 종합점수를 확정 승자 판단에 사용하지 않는다.
목적별 확정 여부는 objective_winners의 label_status에서 확인한다.

source_balanced_reward는 기존 수학 종합점수, report_balanced_reward는 에너지를 포함한 새 점수다.
두 점수는 기준이 다르며, 시나리오마다 정규화 범위도 달라진다.
에너지는 명시한 강체·모터 모형의 추정값이다. E_theta(관절 노력도)와 구분한다.

CSV의 빈칸은 0이 아니다. 장애물이 없는 상황의 여유 값이나 실패한 계산이 비어 있을 수 있다.
같은 scenario_id의 후보들은 같은 환경에서 비교한 대응 자료다.
theta 식은 도달 가능한 구간에서 해당 가지로 해석한다. 식이 저장돼 있어도 경로가 유효하다는 뜻은 아니다.

report_cases 폴더에는 앞서 사용한 두 상황의 실제 원자료만 따로 모았다.
전체 자료는 train 3,000개 상황이며, report_cases만으로 전체 성능을 결론 내리지 않는다.

Python에서는 pandas.read_csv('analysis_ready.csv')로 시작할 수 있다.
압축 CSV는 pandas.read_csv('candidate_metrics.csv.gz')로 바로 읽는다.
Parquet는 pandas.read_parquet('candidate_metrics.parquet')로 읽는다.
큰 시간·기하 표는 load_data.py의 반복 읽기 함수를 사용한다.

이 패키지에는 그래프나 통계적 우열 결론이 없다.
""",encoding="utf-8")
    shutil.copy2(ROOT/"docs/report_alignment.md",folder/"수식과_계산방법.md")
    main_readme=folder/"README.md"
    original=main_readme.read_text(encoding="utf-8")
    if "기존 Eθ/A/J 적분" not in original:
        main_readme.write_text(original+"""

## 기존 Eθ/A/J 적분과 표시 정밀도

기존 Eθ/A/J 요약값은 v3 적응형 적분 결과다. 161점 시간 표본의 사다리꼴 재적분은 별도 근사이며 정확히 같을 필요는 없다.
원래 적응형 가중 노드는 기존 data/phase4_v3_release의 adaptive_panels 파일에 보존돼 있고, 이 패키지에 대용량 중복 복사하지 않았다.
새 물리지표는 physics_161_*와 161점 표본을 직접 대조하고, 321·641값으로 수렴성을 확인한다.
화면용 곡선식은 표시 정밀도를 사용한다. 정밀한 재계산에는 equations의 equation_descriptor_json/curve_parameters_json을 사용한다.
analysis_ready.csv는 기본 입력과 후보별 비용만 모은 시작용 표다. 상세 표·열 사전·원형식은 그대로 별도 보존했다.
""",encoding="utf-8")
    if (ROOT/"tmp/report_v4_regression.log").exists():
        target=folder/"provenance"
        target.mkdir(exist_ok=True)
        shutil.copy2(ROOT/"tmp/report_v4_regression.log",target/"regression_tests.txt")
    manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    paths=[p for p in folder.rglob("*") if p.is_file() and p.name!="manifest.json"]
    save_json(folder/"manifest.json",dict(dataset_version=manifest["dataset_version"],files=[
        dict(path=p.relative_to(folder).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)]))
    print(f"HANDOFF rows={len(ready)} columns={len(ready.columns)}")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=ROOT/"outputs/report_aligned_v4_20261021")
    prepare(parser.parse_args().data)
