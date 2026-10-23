import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from robot_planner.curves import make_curve
from robot_planner.kinematics import PlanarArm2R
from robot_planner.report_evaluation import simulate_report_trajectory,score_report_candidates,export_comparison


class ReportEvaluationTests(unittest.TestCase):
    def row(self,rid,energy=1.,effort=2.,**changes):
        r=dict(scenario_id="s",run_id=rid,curve_key="line",shape=0.,elbow_mode="up",
               candidate_status="accepted",selection_valid=True,energy_selection_ready=True,torque_selection_ready=True,
               path_length_units=1.,joint_velocity_effort=effort,joint_acceleration_effort=2.,
               joint_jerk_effort=3.,regularized_curvature_energy_inv_unit=0.,
               obstacle_penalty_s_per_unit2=0.,model_electrical_energy_j=energy,rms_joint_torque_nm=energy)
        r.update(changes)
        return r

    def test_energy_replaces_effort_only_in_new_score(self):
        rows=[self.row("low_effort",energy=10,effort=1),self.row("low_energy",energy=1,effort=10)]
        labels=score_report_candidates(rows)
        by={r["objective"]:r for r in labels}
        self.assertEqual(by["joint_effort"]["winner_run_id"],"low_effort")
        self.assertEqual(by["model_energy"]["winner_run_id"],"low_energy")
        self.assertEqual(by["report_balanced"]["winner_run_id"],"low_energy")
        self.assertEqual(rows[0]["joint_velocity_effort"],1)
        self.assertEqual(rows[0]["report_balanced_cost_0_100"],20)

    def test_unknown_energy_alternative_blocks_energy_but_not_mathematics(self):
        rows=[self.row("good"),self.row("unresolved_energy",energy_selection_ready=False,torque_selection_ready=False)]
        labels={r["objective"]:r for r in score_report_candidates(rows)}
        self.assertEqual(labels["distance"]["label_status"],"ready")
        self.assertEqual(labels["model_energy"]["label_status"],"unresolved_physics_alternative")
        self.assertIsNone(labels["model_energy"]["winner_run_id"])
        self.assertFalse(rows[0]["report_scenario_labels_ready"])

    def test_unknown_source_alternative_blocks_labels(self):
        rows=[self.row("good"),self.row("unknown",candidate_status="unresolved",selection_valid=False)]
        self.assertTrue(all(r["winner_run_id"] is None for r in score_report_candidates(rows)))

    def test_no_valid_candidate_keeps_blank_scores(self):
        rows=[self.row("bad",candidate_status="invalid",selection_valid=False)]
        labels=score_report_candidates(rows)
        self.assertTrue(all(r["label_status"]=="no_valid_candidate" for r in labels))
        self.assertIsNone(rows[0]["report_balanced_cost_0_100"])

    def test_single_simulation_uses_release_math_and_preserves_equations(self):
        result=simulate_report_trajectory(PlanarArm2R(2,2),make_curve("line",(1,1),(2,1)))
        self.assertEqual(result.metrics["source_dataset_version"],"phase4-mathematics-v3")
        self.assertEqual(result.metrics["dataset_version"],"phase4-report-v4")
        self.assertEqual(result.metrics["geometry_convergence_status"],"passed")
        self.assertIn("theta1_t_equation",result.equations)
        self.assertIn("corrected_note_derivatives",result.equations)
        self.assertEqual(len(result.sample_rows),161)
        self.assertEqual(result.sample_rows[0]["theta1_velocity_rad_s"],0)
        self.assertIsNotNone(result.sample_rows[40]["torque1_nm"])

    def test_comparison_export_separates_grid_coordinates(self):
        result=simulate_report_trajectory(PlanarArm2R(2,2),make_curve("line",(1,1),(2,1)))
        records=[self.row("r")]
        comparison=dict(records=records,labels=score_report_candidates(records),results={"r":result})
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/"tmp") as folder:
            target=Path(folder)/"new"
            export_comparison(target,comparison)
            with (target/"time_samples.csv").open(encoding="utf-8-sig") as f:
                times=list(csv.DictReader(f))
            with (target/"geometry_samples.csv").open(encoding="utf-8-sig") as f:
                geometry=list(csv.DictReader(f))
            self.assertEqual(len(times),161)
            self.assertEqual(len(geometry),161)
            self.assertNotIn("dx_du",times[0])
            self.assertNotIn("time_s",geometry[0])
            self.assertIn("time_s",times[0])
            self.assertIn("geometry_u",geometry[0])
            self.assertFalse(list(target.glob("*.png")))


if __name__=="__main__":
    unittest.main()
