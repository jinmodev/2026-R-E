"""Explicit 2R torque/work model with exact differential IK and grid checks.

This versioned extension never relabels the existing mathematical effort as J.
Defaults describe a vertical, slender-link, direct-drive *model*, not hardware.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
import math

import numpy as np

from .report_math import polynomial_coefficients

VERSION = "report-2r-physics-v1"
METRICS = (
    "net_mechanical_work_j", "positive_mechanical_work_j", "absorbed_mechanical_work_j",
    "motor_copper_loss_j", "model_electrical_energy_j", "rms_joint_torque_nm",
    "damping_loss_j", "kinetic_energy_change_j", "potential_energy_change_j",
    "work_energy_residual_j",
)


@dataclass(frozen=True)
class ReportPhysicsConfig:
    workspace_unit_m: float = .1
    link_linear_density_kg_per_m: float = .35
    viscous_damping_nms_per_rad: float = .03
    torque_constant_nm_per_a: float = .12
    motor_resistance_ohm: float = .8
    gravity_m_per_s2: float = 9.80665
    motion_plane: str = "vertical"
    singularity_floor: float = 1e-10
    energy_rtol: float = .02
    torque_rtol: float = .01
    absolute_tolerance: float = 1e-6
    balance_rtol: float = .005

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key == "motion_plane":
                if value not in ("vertical","horizontal"):
                    raise ValueError("motion_plane must be vertical or horizontal")
            elif not math.isfinite(value) or value < 0 or (value == 0 and key != "viscous_damping_nms_per_rad"):
                raise ValueError(key+" must be finite and positive (damping may be zero)")
        if self.singularity_floor >= 1:
            raise ValueError("singularity_floor must be below 1")
        if max(self.energy_rtol, self.torque_rtol, self.balance_rtol) >= 1:
            raise ValueError("relative tolerances must be below 1")


def curve_arrays(curve, u):
    """p, p_u, p_uu arrays; exact formulas for all built-in families."""
    u = np.asarray(u, dtype=float)
    if u.ndim != 1 or not np.isfinite(u).all() or (u < 0).any() or (u > 1).any():
        raise ValueError("u must be a finite one-dimensional array in [0,1]")
    p = curve.parameters
    start, end = np.asarray(curve.start), np.asarray(curve.end)
    coefficients = polynomial_coefficients(curve)
    if coefficients is not None:
        c = np.asarray(coefficients).T
        pos = c[0]+u[:,None]*(c[1]+u[:,None]*(c[2]+u[:,None]*c[3]))
        first = c[1]+u[:,None]*(2*c[2]+u[:,None]*3*c[3])
        second = np.broadcast_to(2*c[2],pos.shape)+u[:,None]*6*c[3]
        return pos, first, second
    if curve.key == "arc":
        phi = float(p["theta_start"])+float(p["theta_delta"])*u
        r, sweep = float(p["radius"]), float(p["theta_delta"])
        pos = r*np.column_stack((np.cos(phi),np.sin(phi)))
        first = r*sweep*np.column_stack((-np.sin(phi),np.cos(phi)))
        second = -sweep*sweep*pos
        return pos+np.array([p["center_x"],p["center_y"]]),first,second
    if curve.key == "cycloid":
        d = end-start
        normal = np.array([-d[1],d[0]])/np.linalg.norm(d)
        h, phi = float(p["height"]), 2*np.pi*u
        pos = start+((phi-np.sin(phi))/(2*np.pi))[:,None]*d+(.5*h*(1-np.cos(phi)))[:,None]*normal
        first = (1-np.cos(phi))[:,None]*d+(h*np.pi*np.sin(phi))[:,None]*normal
        second = (2*np.pi*np.sin(phi))[:,None]*d+(2*h*np.pi**2*np.cos(phi))[:,None]*normal
        return pos,first,second
    if curve.key == "limacon":
        a,b,phase,sweep,side,scale,rotation = (float(p[k]) for k in ("a","b","phase","sweep","side","scale","rotation"))
        phi = phase+sweep*u
        raw = np.column_stack((a*np.cos(phi)+b*(1+np.cos(2*phi))/2,
                               side*(a*np.sin(phi)+b*np.sin(2*phi)/2)))
        raw0 = np.array([(a+b*math.cos(phase))*math.cos(phase),
                         side*(a+b*math.cos(phase))*math.sin(phase)])
        first = np.column_stack((-a*np.sin(phi)-b*np.sin(2*phi),
                                 side*(a*np.cos(phi)+b*np.cos(2*phi))))
        second = np.column_stack((-a*np.cos(phi)-2*b*np.cos(2*phi),
                                  side*(-a*np.sin(phi)-2*b*np.sin(2*phi))))
        R = np.array([[math.cos(rotation),-math.sin(rotation)],
                      [math.sin(rotation), math.cos(rotation)]])
        return start+scale*(raw-raw0)@R.T,scale*sweep*first@R.T,scale*sweep*sweep*second@R.T
    raise ValueError("Unsupported built-in curve: "+curve.key)


def joint_arrays(arm, curve, times, duration_s, elbow_mode="up", config=None):
    """Analytic first/second time derivatives, vectorized 2R differential IK."""
    cfg = config or ReportPhysicsConfig()
    times = np.asarray(times,dtype=float)
    if not math.isfinite(duration_s) or duration_s <= 0 or times.ndim != 1 or len(times) < 2:
        raise ValueError("positive duration and at least two times required")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or times[0] < 0 or times[-1] > duration_s:
        raise ValueError("times must increase within [0,T]")
    if elbow_mode not in ("up","down"):
        raise ValueError("unknown elbow mode")
    s = times/duration_s
    u = np.clip(10*s**3-15*s**4+6*s**5,0,1)
    du = 30*s*s*(1-s)**2/duration_s
    ddu = (60*s-180*s*s+120*s**3)/duration_s**2
    pos, pu, puu = curve_arrays(curve,u)
    v = pos-np.asarray(arm.base)
    cosine = (np.sum(v*v,axis=1)-arm.link1**2-arm.link2**2)/(2*arm.link1*arm.link2)
    if np.max(np.abs(cosine)) > 1+1e-10:
        raise ValueError("unreachable state")
    cosine = np.clip(cosine,-1,1)
    sine = np.sqrt(np.maximum(0,1-cosine*cosine))*(-1 if elbow_mode=="up" else 1)
    if np.min(np.abs(sine)) <= cfg.singularity_floor:
        raise ValueError("singular differential IK")
    q2 = np.arctan2(sine,cosine)
    q1 = np.arctan2(v[:,1],v[:,0])-np.arctan2(arm.link2*sine,arm.link1+arm.link2*cosine)
    q1,q2 = np.unwrap(q1),np.unwrap(q2)
    e1 = arm.link1*np.column_stack((np.cos(q1),np.sin(q1)))
    e2 = arm.link2*np.column_stack((np.cos(q1+q2),np.sin(q1+q2)))
    n1,n2 = np.column_stack((-e1[:,1],e1[:,0])),np.column_stack((-e2[:,1],e2[:,0]))
    det = n1[:,0]*n2[:,1]-n1[:,1]*n2[:,0]
    def solve(rhs):
        return np.column_stack(((rhs[:,0]*n2[:,1]-rhs[:,1]*n2[:,0])/det,
                                (n1[:,0]*rhs[:,1]-n1[:,1]*rhs[:,0])/det))
    absolute_v = solve(pu*du[:,None])
    absolute_a = solve(puu*du[:,None]**2+pu*ddu[:,None]+e1*absolute_v[:,0,None]**2+e2*absolute_v[:,1,None]**2)
    velocity = np.column_stack((absolute_v[:,0],absolute_v[:,1]-absolute_v[:,0]))
    acceleration = np.column_stack((absolute_a[:,0],absolute_a[:,1]-absolute_a[:,0]))
    return dict(time_s=times,progress_u=u,position=pos,theta=np.column_stack((q1,q2)),
                velocity=velocity,acceleration=acceleration,
                singularity_index=np.abs(sine),geometry_first=pu,geometry_second=puu)


def torque_arrays(arm, theta, velocity, acceleration, config=None):
    """Rigid-link inverse dynamics from relative joint coordinates, in SI."""
    cfg = config or ReportPhysicsConfig()
    theta,velocity,acceleration = (np.asarray(z,dtype=float) for z in (theta,velocity,acceleration))
    if theta.ndim != 2 or theta.shape[1] != 2 or any(z.shape != theta.shape for z in (velocity,acceleration)):
        raise ValueError("theta/velocity/acceleration must have shape (N,2)")
    if not all(np.isfinite(z).all() for z in (theta,velocity,acceleration)):
        raise ValueError("nonfinite joint input")
    l1,l2 = cfg.workspace_unit_m*arm.link1,cfg.workspace_unit_m*arm.link2
    m1,m2 = cfg.link_linear_density_kg_per_m*l1,cfg.link_linear_density_kg_per_m*l2
    c1,c2 = l1/2,l2/2
    I1,I2 = m1*l1*l1/12,m2*l2*l2/12
    a,b = theta.T
    da,db = velocity.T
    dda,ddb = acceleration.T
    coupling = m2*l1*c2
    M11 = I1+I2+m1*c1*c1+m2*(l1*l1+c2*c2+2*l1*c2*np.cos(b))
    M12 = I2+m2*(c2*c2+l1*c2*np.cos(b))
    M22 = I2+m2*c2*c2
    C1 = -coupling*np.sin(b)*(2*da*db+db*db)
    C2 = coupling*np.sin(b)*da*da
    gravity = cfg.gravity_m_per_s2 if cfg.motion_plane=="vertical" else 0.
    G1 = gravity*((m1*c1+m2*l1)*np.cos(a)+m2*c2*np.cos(a+b))
    G2 = gravity*m2*c2*np.cos(a+b)
    tau1 = M11*dda+M12*ddb+C1+G1+cfg.viscous_damping_nms_per_rad*da
    tau2 = M12*dda+M22*ddb+C2+G2+cfg.viscous_damping_nms_per_rad*db
    P1,P2 = tau1*da,tau2*db
    torque2 = tau1*tau1+tau2*tau2
    return dict(torque1_nm=tau1,torque2_nm=tau2,joint1_power_w=P1,joint2_power_w=P2,
                net_power_w=P1+P2,positive_power_w=np.maximum(P1,0)+np.maximum(P2,0),
                absorbed_power_w=np.maximum(-P1,0)+np.maximum(-P2,0),
                torque_squared_nm2=torque2,
                copper_loss_power_w=cfg.motor_resistance_ohm*torque2/cfg.torque_constant_nm_per_a**2,
                damping_power_w=cfg.viscous_damping_nms_per_rad*(da*da+db*db),
                kinetic_energy_j=.5*M11*da*da+M12*da*db+.5*M22*db*db,
                potential_energy_j=gravity*((m1*c1+m2*l1)*np.sin(a)+m2*c2*np.sin(a+b)))


def integrate_physics(times, values):
    times = np.asarray(times,dtype=float)
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("strictly increasing time grid required")
    def integ(key):
        return float(np.trapezoid(values[key],times))
    work,positive,absorbed = integ("net_power_w"),integ("positive_power_w"),integ("absorbed_power_w")
    copper,damping = integ("copper_loss_power_w"),integ("damping_power_w")
    kinetic = float(values["kinetic_energy_j"][-1]-values["kinetic_energy_j"][0])
    potential = float(values["potential_energy_j"][-1]-values["potential_energy_j"][0])
    return dict(net_mechanical_work_j=work,positive_mechanical_work_j=positive,
                absorbed_mechanical_work_j=absorbed,motor_copper_loss_j=copper,
                model_electrical_energy_j=positive+copper,
                rms_joint_torque_nm=math.sqrt(max(0,integ("torque_squared_nm2")/(times[-1]-times[0]))),
                damping_loss_j=damping,kinetic_energy_change_j=kinetic,
                potential_energy_change_j=potential,work_energy_residual_j=work-kinetic-potential-damping)


def evaluate_physics(arm,curve,duration_s=3.,elbow_mode="up",config=None):
    """Expose estimates at 161/321/641; readiness is objective-specific.

    Agreement is a numerical screen, not an exact-error bound. No failed value
    is replaced with zero. 641 is a comparison resolution, not a true solution.
    """
    cfg = config or ReportPhysicsConfig()
    grids, samples = {}, None
    try:
        for n in (161,321,641):
            state = joint_arrays(arm,curve,np.linspace(0,duration_s,n),duration_s,elbow_mode,cfg)
            physical = torque_arrays(arm,state["theta"],state["velocity"],state["acceleration"],cfg)
            grids[str(n)] = integrate_physics(state["time_s"],physical)
            if n == 161:
                samples = (state,physical)
        small,big = grids["321"],grids["641"]
        if not all(math.isfinite(v) for g in grids.values() for v in g.values()):
            raise ValueError("nonfinite physical estimate")
        def converged(field,rtol):
            return abs(small[field]-big[field]) <= cfg.absolute_tolerance+rtol*abs(big[field])
        scale = abs(big["net_mechanical_work_j"])+abs(big["kinetic_energy_change_j"])+abs(big["potential_energy_change_j"])+abs(big["damping_loss_j"])
        balance = abs(big["work_energy_residual_j"]) <= cfg.absolute_tolerance+cfg.balance_rtol*scale
        energy_ok = balance and all(converged(k,cfg.energy_rtol) for k in ("positive_mechanical_work_j","motor_copper_loss_j","model_electrical_energy_j"))
        torque_ok = balance and converged("rms_joint_torque_nm",cfg.torque_rtol)
        metrics = dict(big,physics_version=VERSION,physics_status="screened" if energy_ok and torque_ok else "unresolved",
                   energy_selection_ready=energy_ok,torque_selection_ready=torque_ok,
                   work_energy_balance_passed=balance,physics_error=None,
                   physics_integration_method="uniform-t composite trapezoid; analytic differential IK; 161/321/641 screening; headline=641",
                   physics_config_json=json.dumps(asdict(cfg),sort_keys=True))
        return dict(metrics=metrics,grids=grids,samples=samples)
    except (ValueError,OverflowError,ZeroDivisionError,FloatingPointError) as exc:
        return dict(metrics=dict({k:None for k in METRICS},physics_version=VERSION,physics_status="unresolved",
                             energy_selection_ready=False,torque_selection_ready=False,work_energy_balance_passed=False,
                             physics_error=str(exc),physics_config_json=json.dumps(asdict(cfg),sort_keys=True),
                             physics_integration_method="no finite fallback on failure"),
                    grids=grids,samples=samples)
