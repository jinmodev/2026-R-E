"""Small hidden-window integration check; no user input or scientific figures."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import tempfile
import time
import tkinter as tk
from tkinter import ttk

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from robot_planner.gui import PlannerApp, COMPARISON_OBJECTIVES
from robot_planner.report_evaluation import export_comparison


def main():
    root=tk.Tk()
    root.withdraw()
    app=PlannerApp(root)
    try:
        assert app.simulation is not None
        assert app.simulation.metrics["dataset_version"]=="phase4-report-v4"
        app.case_choice.set("P4S000691 · 리마송/사이클로이드")
        app._load_report_case()
        assert len(app.obstacles)==1
        assert app.curve.key=="cycloid"
        assert app.simulation.metrics["candidate_status"]=="accepted"
        assert "r(phi)" not in app.equation_text.get("1.0","end")  # Cycloid preset.
        report=app.report_text.get("1.0","end")
        assert "A*(phi-sin(phi))" in report
        assert "b*(2*v^2-1)/(1+v^2)^(5/2)" in report
        assert app.physics_metric_vars["model_electrical_energy_j"].get()!="—"
        app.curve_kind.set("리마송");app.shape.set(-.8)
        assert app._recompute()
        assert app.simulation.metrics["candidate_status"]=="accepted"
        assert "r(phi)" in app.report_text.get("1.0","end")
        payload=app._payload()
        json.dumps(payload,ensure_ascii=False,allow_nan=False)
        assert payload["physics_model"]["motion_plane"]=="vertical"
        assert str(app.apply_winner_button["state"])=="disabled"
        assert app._apply_comparison_winner() is False
        app._start_comparison()
        assert app.details_tabs.select()==str(app.comparison_tab)
        assert str(app.apply_winner_button["state"])=="disabled"
        deadline=time.monotonic()+90
        while app._comparison_busy and time.monotonic()<deadline:
            root.update()
            time.sleep(.05)
        assert not app._comparison_busy,"comparison did not complete"
        assert app._comparison is not None,app.comparison_status.get()
        assert len(app._comparison["records"])==42
        assert len(app._comparison["labels"])==9
        comparison=app._comparison
        applied=[]
        for name,objective in COMPARISON_OBJECTIVES.items():
            app.winner_objective.set(name)
            app._update_winner_controls()
            label=next(row for row in comparison["labels"] if row["objective"]==objective)
            if label["label_status"]!="ready":
                assert str(app.apply_winner_button["state"])=="disabled"
                continue
            assert str(app.apply_winner_button["state"])=="normal"
            assert app.apply_winner_button.invoke()==1
            rid=label["winner_run_id"]
            record=next(row for row in comparison["records"] if row["run_id"]==rid)
            assert app.curve.key==record["curve_key"]
            assert app.shape.get()==record["shape"]
            assert app.elbow_mode.get()==("팔꿈치 위" if record["elbow_mode"]=="up" else "팔꿈치 아래")
            assert app.simulation is comparison["results"][rid]
            assert app.simulation.equations["x_u_equation"] in app.equation_text.get("1.0","end")
            assert app.simulation.equations["theta1_t_equation"] in app.ik_text.get("1.0","end")
            assert app.pose_index.get()==0
            assert app._comparison is comparison
            assert str(app.comparison_export_button["state"])=="normal"
            applied.append(objective)
        assert "distance" in applied
        # Typed inputs must be checked even before the user presses recompute.
        old=app.link1.get();previous_simulation=app.simulation
        app.link1.set("4.25")
        assert app._apply_comparison_winner() is False
        assert app.simulation is previous_simulation
        assert str(app.apply_winner_button["state"])=="disabled"
        app.link1.set(old)
        label=next(row for row in comparison["labels"] if row["objective"]=="distance")
        saved=dict(label)
        app.winner_objective.set("거리")
        for status in ("no_valid_candidate","unresolved_source_alternative","unresolved_physics_alternative"):
            label.update(label_status=status,winner_run_id=None)
            app._update_winner_controls()
            assert str(app.apply_winner_button["state"])=="disabled"
            assert app._apply_comparison_winner() is False
            assert app.simulation is previous_simulation
        label.update(saved)
        app._update_winner_controls()
        with tempfile.TemporaryDirectory(dir=ROOT/"tmp") as folder:
            target=Path(folder)/"comparison"
            export_comparison(target,app._comparison)
            assert (target/"candidate_metrics.csv").exists()
            assert (target/"time_samples.csv").exists()
            assert (target/"geometry_samples.csv").exists()
        app.motion_plane.set("수평 평면")
        assert app._recompute()
        assert app._comparison is None
        assert str(app.apply_winner_button["state"])=="disabled"
        assert json.loads(app.simulation.metrics["physics_config_json"])["motion_plane"]=="horizontal"
        root.update_idletasks()
        def visit(parent):
            for child in parent.winfo_children():
                yield child
                yield from visit(child)
        notebooks=[w for w in visit(root) if isinstance(w,ttk.Notebook)]
        assert len(notebooks)==1 and len(notebooks[0].tabs())==6
        print(json.dumps(dict(status="passed",checks=[
            "startup","train_case_preset","exact_report_formula","physical_metrics","limacon_switch",
            "42_candidate_background_comparison","clean_comparison_export","stale_comparison_cleared",
            "horizontal_model_switch","JSON_finite_export","six_report_tabs",
            "comparison_auto_tab","winner_applies_curve_shape_elbow_equations",
            "comparison_preserved_after_apply","stale_input_winner_blocked",
            "unresolved_and_no_valid_winner_blocked"],
            winner_objectives_applied=applied,
            source_case="P4S000691",scientific_figures_generated=0),ensure_ascii=False))
    finally:
        app._stop_animation()
        root.destroy()


if __name__=="__main__":
    main()
