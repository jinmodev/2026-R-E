from pathlib import Path
import sys
import unittest
import math

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from robot_planner.curves import make_curve
from robot_planner.kinematics import PlanarArm2R
from robot_planner.evaluation import CircleObstacle
from robot_planner.phase4_evaluation import MathematicalEvaluationConfig,simulate_trajectory
from robot_planner.phase4_release import geometry_values,evaluate_release,GEOMETRY_TOL


class ReleaseTests(unittest.TestCase):
    def test_geometry_reproduces_original_three_fields(self):
        cfg=MathematicalEvaluationConfig();arm=PlanarArm2R(3,2)
        obstacles=(CircleObstacle('far',-3,2,.1),)
        for family in ('line','arc','bezier','hermite','cycloid','limacon'):
            curve=make_curve(family,(1.5,1.2),(2.8,1.5),shape=.4)
            for n in (161,321,641):
                a=geometry_values(arm,curve,'up',obstacles,cfg,n)
                b=simulate_trajectory(arm,curve,'up',obstacles=obstacles,config=cfg,samples=n).metrics
                for field in GEOMETRY_TOL:self.assertTrue(math.isclose(a[field],b[field],rel_tol=1e-12,abs_tol=1e-10))

    def test_quarantine_prevents_selection_without_erasing_values(self):
        result=evaluate_release(PlanarArm2R(2,2),make_curve('line',(1,1),(2,1)),quarantine=True)
        self.assertEqual(result['metrics']['candidate_status'],'unresolved')
        self.assertFalse(result['metrics']['selection_valid'])
        self.assertIsNotNone(result['metrics']['joint_jerk_effort'])

    def test_obstacle_iterable_not_lost(self):
        result=evaluate_release(PlanarArm2R(2,2),make_curve('line',(1,1),(2,1)),obstacles=iter([CircleObstacle('hit',1.5,1,.1)]))
        self.assertEqual(result['metrics']['candidate_status'],'invalid')


if __name__=='__main__':unittest.main()
