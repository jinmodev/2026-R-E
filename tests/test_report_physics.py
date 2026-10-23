import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from robot_planner.curves import make_curve, CurvePath
from robot_planner.kinematics import PlanarArm2R
from robot_planner.phase4_evaluation import _curve_value_derivatives
from robot_planner.refinement import differential_joint_state
from robot_planner.report_math import corrected_sine_limacon, cycloid_length, report_equations
from robot_planner.report_physics import (ReportPhysicsConfig, curve_arrays, joint_arrays,
                                        torque_arrays, evaluate_physics)


class ReportPhysicsTests(unittest.TestCase):
    def test_note_derivative_typo_regression(self):
        result=corrected_sine_limacon(0,1,1)
        self.assertEqual(result["ddx"],-1)
        self.assertAlmostEqual(result["curvature"],3/(2*math.sqrt(2)))
        for v in (-1.2,.4,1.8):
            h=1e-5
            numeric=(corrected_sine_limacon(v+h,.79,1)["dx"]-
                     corrected_sine_limacon(v-h,.79,1)["dx"])/(2*h)
            self.assertAlmostEqual(corrected_sine_limacon(v,.79,1)["ddx"],numeric,delta=1e-8)

    def test_cycloid_length_hand_integral_and_general_cases(self):
        from scipy.integrate import quad
        for A,B in ((1,1),(1,.3),(1,1.4),(1,-1.4),(1,0)):
            reference=quad(lambda p:math.hypot(A*(1-math.cos(p)),B*math.sin(p)),0,2*math.pi,epsabs=1e-11)[0]
            self.assertAlmostEqual(cycloid_length(A,B),reference,delta=1e-10)

    def test_vectorized_curves_and_joint_derivatives_match_independent_existing_code(self):
        arm=PlanarArm2R(4,3)
        times=np.linspace(0,3,19)
        for family in ("line","arc","bezier","hermite","cycloid","limacon"):
            curve=make_curve(family,(2,1),(3,1),shape=.4)
            us=np.linspace(0,1,19)
            pos,first,second=curve_arrays(curve,us)
            for i,u in enumerate(us):
                old=_curve_value_derivatives(curve,u)
                np.testing.assert_allclose(np.r_[pos[i],first[i],second[i]],old,atol=1e-12,rtol=1e-12)
            for mode in ("up","down"):
                state=joint_arrays(arm,curve,times,3,mode)
                for i,t in enumerate(times):
                    old=differential_joint_state(arm,curve,float(t),3,mode)
                    np.testing.assert_allclose(state["velocity"][i],old["velocity"],atol=1e-10)
                    np.testing.assert_allclose(state["acceleration"][i],old["acceleration"],atol=1e-9)

    def test_static_vertical_gravity_and_zero_power(self):
        arm=PlanarArm2R(4,3)
        cfg=ReportPhysicsConfig()
        q=np.array([[0.,0.],[math.pi/2,0.]])
        z=np.zeros_like(q)
        r=torque_arrays(arm,q,z,z,cfg)
        l1,l2=.4,.3
        m1,m2=.35*l1,.35*l2
        expected=((m1*l1/2+m2*l1)+m2*l2/2)*cfg.gravity_m_per_s2
        self.assertAlmostEqual(r["torque1_nm"][0],expected)
        self.assertAlmostEqual(r["torque1_nm"][1],0,delta=1e-12)
        np.testing.assert_array_equal(r["net_power_w"],[0,0])
        self.assertGreater(r["copper_loss_power_w"][0],0)  # Holding consumes model current.

    def circle(self):
        radius,angle,delta=2.,.2,.8
        return CurvePath("arc","circle",
                         ((radius*math.cos(angle),radius*math.sin(angle)),
                          (radius*math.cos(angle+delta),radius*math.sin(angle+delta))),
                         "fixture",dict(center_x=0.,center_y=0.,radius=radius,theta_start=angle,theta_delta=delta))

    def test_work_energy_identity_and_signed_work(self):
        arm=PlanarArm2R(2,2)
        cfg=ReportPhysicsConfig()
        result=evaluate_physics(arm,self.circle(),config=cfg)
        self.assertEqual(result["metrics"]["physics_status"],"screened")
        m=result["metrics"]
        damping=cfg.viscous_damping_nms_per_rad*.8**2*10/(7*3)
        self.assertAlmostEqual(m["damping_loss_j"],damping,delta=1e-10)
        self.assertAlmostEqual(m["net_mechanical_work_j"],
                               m["potential_energy_change_j"]+damping,delta=1e-8)
        self.assertAlmostEqual(m["net_mechanical_work_j"],
                               m["positive_mechanical_work_j"]-m["absorbed_mechanical_work_j"],delta=1e-12)
        self.assertAlmostEqual(m["model_electrical_energy_j"],
                               m["positive_mechanical_work_j"]+m["motor_copper_loss_j"])

    def test_torque_rms_analytic_polynomial_circle(self):
        arm=PlanarArm2R(2,2)
        cfg=ReportPhysicsConfig(motion_plane="horizontal",viscous_damping_nms_per_rad=0)
        result=evaluate_physics(arm,self.circle(),config=cfg)
        theta2=arm.inverse(self.circle().start,"up").theta2
        l,m=.2,.07
        I=m*l*l/12
        M11=2*I+m*(l/2)**2+m*(l*l+(l/2)**2+2*l*(l/2)*math.cos(theta2))
        M12=I+m*((l/2)**2+l*(l/2)*math.cos(theta2))
        # Even with constant relative theta2, tau2 includes centripetal load.
        # The cross integral int q''*(q')^2 dt vanishes at stationary endpoints.
        H=m*l*(l/2)*math.sin(theta2)
        fourth=30**4*math.factorial(8)**2/math.factorial(17)
        truth=math.sqrt(.8**2*120/7*(M11*M11+M12*M12)+H*H*.8**4*fourth)/3**2
        self.assertAlmostEqual(result["metrics"]["rms_joint_torque_nm"],truth,delta=1e-10)
        self.assertAlmostEqual(result["metrics"]["net_mechanical_work_j"],0,delta=1e-12)

    def test_singular_state_is_unresolved(self):
        result=evaluate_physics(PlanarArm2R(2,2),make_curve("line",(4,0),(2,1)))
        self.assertEqual(result["metrics"]["physics_status"],"unresolved")
        self.assertFalse(result["metrics"]["energy_selection_ready"])
        self.assertIsNone(result["metrics"]["model_electrical_energy_j"])

    def test_nonfinite_config_rejected(self):
        with self.assertRaises(ValueError):ReportPhysicsConfig(workspace_unit_m=float("nan"))

    def test_equations_keep_local_scale_and_polar_area_separate(self):
        c=make_curve("limacon",(2,1),(3,1),shape=-.8)
        e=report_equations(c)
        self.assertIn("scale",e["native_curvature"])
        self.assertIn("not",e["native_area"])
        c=make_curve("cycloid",(2,1),(3,1),shape=.8)
        self.assertAlmostEqual(float(report_equations(c)["analytic_length_units"]),1.387198211478156,places=12)


if __name__=="__main__":
    unittest.main()
