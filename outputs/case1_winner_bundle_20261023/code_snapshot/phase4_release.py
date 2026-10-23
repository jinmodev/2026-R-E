"""Frozen release evaluation with per-run geometry convergence gating."""
from __future__ import annotations
import math
from .phase4_v3 import evaluate_v3
from .phase4_evaluation import _curve_value_derivatives, MathematicalEvaluationConfig
from .evaluation import pose_obstacle_clearance

DATASET_VERSION='phase4-mathematics-v3'
CODE_VERSION='release-generator-v1'
GEOMETRY_TOL={
    'path_length_units':(1e-6,.005),
    'regularized_curvature_energy_inv_unit':(1e-6,.02),
    'obstacle_penalty_s_per_unit2':(1e-4,.05),
}


def geometry_values(arm,curve,branch,obstacles,cfg,n):
    us=[(1-math.cos(math.pi*i/(n-1)))/2 for i in range(n)]
    epsilon=cfg.curvature_speed_epsilon_ratio*max(math.dist(curve.start,curve.end),1e-12)
    speeds=[];curv=[]
    for u in us:
        _,_,dx,dy,ddx,ddy=_curve_value_derivatives(curve,u)
        speed=math.hypot(dx,dy);speeds.append(speed)
        curv.append((dx*ddy-dy*ddx)**2/(speed**2+epsilon**2)**2.5)
    def trap(vals,coords):
        return math.fsum((a+b)*(v-u)/2 for a,b,u,v in zip(vals,vals[1:],coords,coords[1:]))
    penalty=0.
    if obstacles:
        vals=[];times=[cfg.duration_s*i/(n-1) for i in range(n)]
        for t in times:
            s=t/cfg.duration_s;u=10*s**3-15*s**4+6*s**5
            x,y,*_=_curve_value_derivatives(curve,u)
            clearance=pose_obstacle_clearance(arm,arm.inverse((x,y),branch),obstacles,link_radius_units=cfg.link_radius_units)
            vals.append(1/max(clearance,cfg.clearance_epsilon_units)**2)
        penalty=trap(vals,times)
    return dict(path_length_units=trap(speeds,us),regularized_curvature_energy_inv_unit=trap(curv,us),obstacle_penalty_s_per_unit2=penalty)


def evaluate_release(arm,curve,branch='up',*,obstacles=(),config=None,quarantine=False):
    cfg=config or MathematicalEvaluationConfig();obstacles=tuple(obstacles)
    result=evaluate_v3(arm,curve,branch,obstacles=obstacles,config=cfg)
    metrics=result['metrics'];checks={}
    if metrics['candidate_status']=='accepted':
        try:
            grids={161:{f:metrics[f] for f in GEOMETRY_TOL}}
            grids.update({n:geometry_values(arm,curve,branch,obstacles,cfg,n) for n in (321,641)})
            for field,(atol,rtol) in GEOMETRY_TOL.items():
                vals=[grids[n][field] for n in (161,321,641)]
                ok=all(math.isfinite(v) for v in vals) and all(abs(v-vals[2])<=atol+rtol*abs(vals[2]) for v in vals[:2])
                checks[field]=dict(values_by_n={str(n):grids[n][field] for n in grids},passed=ok,atol=atol,rtol=rtol)
            sample_ok=all(r['joint_sample_status']=='computed' for r in result['sample_rows'])
            if not all(c['passed'] for c in checks.values()) or not sample_ok or quarantine:
                metrics.update(candidate_status='unresolved',selection_valid=False,warning='numerical_convergence_or_quarantine')
        except (ValueError,OverflowError,ZeroDivisionError) as exc:
            metrics.update(candidate_status='unresolved',selection_valid=False,warning=str(exc))
    metrics.update(dataset_version=DATASET_VERSION,calculation_version=CODE_VERSION,
        geometry_convergence_status='passed' if checks and all(c['passed'] for c in checks.values()) else 'failed_or_not_evaluated',
        expanded_validation_quarantine=quarantine)
    result['geometry_checks']=checks
    return result
