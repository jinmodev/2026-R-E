"""Independent analytic fixtures for the opt-in refinement, not raw-data edits."""
import math
import unittest
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

from robot_planner.curves import CurvePath, make_curve
from robot_planner.evaluation import CircleObstacle, pose_obstacle_clearance
from robot_planner.kinematics import PlanarArm2R
from robot_planner.phase4_evaluation import (
    MathematicalEvaluationConfig, _curve_value_derivatives,
    numerical_derivative, simulate_trajectory,
)
from robot_planner.refinement import (
    adaptive_path_check, curve_speed_bound, differential_joint_state,
    third_curve_derivative,
)


class RefinementTests(unittest.TestCase):
    def test_third_derivative_all_families(self):
        for family in ('line','arc','bezier','hermite','cycloid','limacon'):
            for shape in (-.8,.4):
                curve=make_curve(family,(1.2,1.5),(3.,1.),shape=shape)
                for u in (.13,.41,.77):
                    h=1e-5
                    plus=_curve_value_derivatives(curve,u+h)[4:]
                    minus=_curve_value_derivatives(curve,u-h)[4:]
                    for a,b,c in zip(third_curve_derivative(curve,u),plus,minus):
                        self.assertAlmostEqual(a,(b-c)/(2*h),delta=1e-5*max(1,abs(a)))

    def test_manufactured_circle_joint_derivatives(self):
        # Centered circle has constant theta2 and theta1 = constant + .8*q(t/T).
        r=2.; a=.2; d=.8; T=3.
        curve=CurvePath('arc','circle',((r*math.cos(a),r*math.sin(a)),
                        (r*math.cos(a+d),r*math.sin(a+d))), 'fixture',
                        dict(center_x=0.,center_y=0.,radius=r,theta_start=a,theta_delta=d))
        arm=PlanarArm2R(2.,2.)
        for branch in ('up','down'):
            for s in (0.,.13,.5,.87,1.):
                result=differential_joint_state(arm,curve,s*T,T,branch)
                truth=(d*(30*s*s-60*s**3+30*s**4)/T,
                       d*(60*s-180*s*s+120*s**3)/T**2,
                       d*(60-360*s+360*s*s)/T**3)
                for name,value in zip(('velocity','acceleration','jerk'),truth):
                    self.assertAlmostEqual(result[name][0],value,delta=1e-12)
                    self.assertAlmostEqual(result[name][1],0.,delta=1e-12)

    def test_speed_bounds(self):
        for family in ('line','arc','bezier','hermite','cycloid','limacon'):
            for shape in (-.8,-.4,.4,.8):
                curve=make_curve(family,(1.2,1.5),(3.,1.),shape=shape)
                bound=curve_speed_bound(curve)
                for i in range(1001):
                    value=_curve_value_derivatives(curve,i/1000)
                    self.assertLessEqual(math.hypot(*value[2:4]),bound+1e-10)

    def test_joint_derivatives_against_independent_local_differences(self):
        # Nonconstant theta2: this exercises cross terms absent from the circle fixture.
        arm=PlanarArm2R(3,2);T=3.;h=.001
        for family in ('line','arc','bezier','hermite','cycloid','limacon'):
            curve=make_curve(family,(1.5,1.2),(2.8,1.5),shape=.4)
            for branch in ('up','down'):
                for t in (.7,1.4,2.3):
                    angles=[[],[]]
                    for k in range(-3,4):
                        s=(t+k*h)/T;u=10*s**3-15*s**4+6*s**5
                        x,y,*_=_curve_value_derivatives(curve,u)
                        pose=arm.inverse((x,y),branch)
                        angles[0].append(pose.theta1);angles[1].append(pose.theta2)
                    exact=differential_joint_state(arm,curve,t,T,branch)
                    for order,name,tol in [(1,'velocity',1e-7),(2,'acceleration',1e-5),(3,'jerk',1e-4)]:
                        for j in (0,1):
                            finite=numerical_derivative(angles[j],h,order)[3]
                            self.assertAlmostEqual(finite,exact[name][j],delta=tol*max(1,abs(exact[name][j])))

    def test_clear_path_and_budget(self):
        arm=PlanarArm2R(2,2);curve=make_curve('line',(1,1),(2,1))
        self.assertEqual(adaptive_path_check(arm,curve)['status'],'bounded_clear')
        self.assertEqual(adaptive_path_check(arm,curve,max_evaluations=1)['status'],'unresolved')

    def test_collision_and_reach_failure_both_recorded(self):
        arm=PlanarArm2R(2,1);curve=make_curve('line',(-2,0),(2,0))
        result=adaptive_path_check(arm,curve,obstacles=[CircleObstacle('test',-2,0,.1)])
        self.assertEqual(result['status'],'invalid')
        self.assertTrue(result['unreachable_witnesses'])
        self.assertTrue(result['collision_witnesses'])

    def test_narrow_between_sample_collision(self):
        arm=PlanarArm2R(2,2);curve=make_curve('line',(1,1),(2,1))
        obstacle=CircleObstacle('tiny',1.53125,1.,1e-5)
        coarse=simulate_trajectory(arm,curve,'up',obstacles=[obstacle],
            config=MathematicalEvaluationConfig(link_radius_units=0),samples=9)
        self.assertTrue(coarse.metrics['selection_valid'])
        self.assertEqual(adaptive_path_check(arm,curve,obstacles=[obstacle],link_radius=0)['status'],'invalid')

    def test_bounded_clear_dense_crosscheck(self):
        arm=PlanarArm2R(2,2)
        obstacles=[CircleObstacle('far',-1,2,.2)]
        count=0
        for family in ('line','arc','bezier','hermite','cycloid','limacon'):
            curve=make_curve(family,(1,1),(2,1),shape=.4)
            for branch in ('up','down'):
                result=adaptive_path_check(arm,curve,branch,obstacles)
                if result['status']!='bounded_clear':continue
                count+=1
                for i in range(2049):
                    x,y,*_=_curve_value_derivatives(curve,i/2048)
                    self.assertTrue(arm.is_reachable((x,y),tolerance=0))
                    self.assertGreater(pose_obstacle_clearance(arm,arm.inverse((x,y),branch),obstacles,link_radius_units=.05),0)
        self.assertGreater(count,0)

    def test_singular_and_invalid_inputs(self):
        arm=PlanarArm2R(2,2);curve=make_curve('line',(4,0),(2,1))
        with self.assertRaises(ValueError):differential_joint_state(arm,curve,0,3)
        for kwargs in ({'link_radius':float('nan')},{'initial_intervals':1.5},{'max_depth':-1}):
            with self.assertRaises(ValueError):adaptive_path_check(arm,curve,**kwargs)
        with self.assertRaises(ValueError):differential_joint_state(arm,curve,1,3,singularity_floor=-1)


if __name__=='__main__':unittest.main()
