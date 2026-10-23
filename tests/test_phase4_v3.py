import math
from pathlib import Path
import sys
import unittest
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from robot_planner.curves import make_curve, CurvePath
from robot_planner.kinematics import PlanarArm2R
from robot_planner.evaluation import CircleObstacle
from robot_planner.phase4_v3 import adaptive_joint_costs, evaluate_v3, AdaptiveIntegralConfig


class V3Tests(unittest.TestCase):
    def test_weighted_replay_and_circle_analytic_costs(self):
        # theta1=.8*q(t/3)+constant; theta2 constant. Polynomial integrals exact.
        import numpy as np
        from numpy.polynomial import Polynomial
        d=.8;T=3.;r=2.;a=.2
        curve=CurvePath('arc','circle',((r*math.cos(a),r*math.sin(a)),(r*math.cos(a+d),r*math.sin(a+d))),
            'fixture',dict(center_x=0.,center_y=0.,radius=r,theta_start=a,theta_delta=d))
        result=adaptive_joint_costs(PlanarArm2R(2,2),curve,T,'down')
        self.assertEqual(result['status'],'accepted')
        theta=Polynomial([0,0,0,10,-15,6])*d
        for order,field in enumerate(('joint_velocity_effort','joint_acceleration_effort','joint_jerk_effort'),1):
            derivative=theta.deriv(order)/T**order
            integral=(derivative*derivative).integ()
            truth=T*(integral(1)-integral(0))
            replay=math.fsum(math.fsum(w*v for w,v in zip(p['weights_s'],p['integrand'])) for p in result['panels'] if p['metric']==field)
            self.assertAlmostEqual(result['costs'][field],truth,delta=1e-9)
            self.assertEqual(replay,result['costs'][field])

    def test_invalid_has_no_joint_score_and_no_stale_derivatives(self):
        arm=PlanarArm2R(2,2);curve=make_curve('line',(1,1),(2,1))
        result=evaluate_v3(arm,curve,obstacles=[CircleObstacle('hit',1.5,1,.1)])
        self.assertEqual(result['metrics']['candidate_status'],'invalid')
        self.assertIsNone(result['metrics']['joint_jerk_effort'])
        self.assertFalse(result['metrics']['selection_valid'])
        self.assertEqual(result['sample_rows'][0]['theta1_velocity_rad_s'],0)

    def test_singular_is_unresolved_not_finite_fallback(self):
        arm=PlanarArm2R(2,2);curve=make_curve('line',(4,0),(2,1))
        result=adaptive_joint_costs(arm,curve,3,'up')
        self.assertEqual(result['status'],'unresolved')
        self.assertTrue(all(v is None for v in result['costs'].values()))

    def test_old_and_new_are_separate(self):
        arm=PlanarArm2R(2,2);curve=make_curve('line',(1,1),(2,1))
        result=evaluate_v3(arm,curve)
        self.assertEqual(result['metrics']['candidate_status'],'accepted')
        self.assertIn('finite differences',result['legacy_v2']['joint_derivative_method'])
        self.assertIn('differential',result['metrics']['joint_derivative_method'])
        self.assertIn('theta1_t_equation',result['equations'])
        self.assertEqual(len(result['sample_rows']),161)

    def test_invalid_config(self):
        with self.assertRaises(ValueError):AdaptiveIntegralConfig(relative_tolerance=0)

    def test_unresolved_candidate_not_selected(self):
        result=evaluate_v3(PlanarArm2R(2,2),make_curve('line',(4,0),(2,1)))
        self.assertEqual(result['metrics']['candidate_status'],'unresolved')
        self.assertFalse(result['metrics']['selection_valid'])
        self.assertIsNone(result['metrics']['joint_velocity_effort'])

    def test_generator_refuses_existing_output(self):
        from generate_phase4_v3 import generate
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);sentinel=p/'preserve.txt';sentinel.write_text('keep')
            with self.assertRaises(ValueError):generate(p)
            self.assertEqual(sentinel.read_text(),'keep')


if __name__=='__main__':unittest.main()
