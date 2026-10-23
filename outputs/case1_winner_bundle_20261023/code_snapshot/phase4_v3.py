"""Versioned opt-in generator math; the v2 GUI/default pipeline is untouched."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .phase4_evaluation import MathematicalEvaluationConfig, simulate_trajectory
from .refinement import adaptive_path_check, differential_joint_state

DATASET_VERSION = 'phase4-mathematics-v3-pilot'
CODE_VERSION = 'phase4-v3-generator-v1'
JOINT_FIELDS = {
    'velocity': ('joint_velocity_effort', 'joint_velocity_squared_rad2_s2', 'rad_s', 1e-7),
    'acceleration': ('joint_acceleration_effort', 'joint_acceleration_squared_rad2_s4', 'rad_s2', 1e-5),
    'jerk': ('joint_jerk_effort', 'joint_jerk_squared_rad2_s6', 'rad_s3', 1e-3),
}


@dataclass(frozen=True)
class AdaptiveIntegralConfig:
    relative_tolerance: float = 1e-8
    subdivision_limit: int = 300
    singularity_floor: float = 1e-10

    def __post_init__(self):
        if not 0 < self.relative_tolerance < 1:
            raise ValueError('relative_tolerance must be in (0,1)')
        if type(self.subdivision_limit) is not int or self.subdivision_limit < 33:
            raise ValueError('subdivision_limit must be an integer >= 33')
        if not 0 < self.singularity_floor < 1:
            raise ValueError('singularity_floor must be in (0,1)')


def adaptive_joint_costs(arm, curve, duration, elbow_mode, config=None):
    """Export replayable weighted integrands, with independent QUADPACK check.

    Lazy optional imports keep the original standard-library GUI usable.
    No fallback to finite differences or unverified finite scores on failure.
    """
    import numpy as np
    from scipy.integrate import quad
    from scipy.optimize import minimize_scalar

    cfg = config or AdaptiveIntegralConfig()
    cache = {}

    def state(t):
        t = float(t)
        if t not in cache:
            cache[t] = differential_joint_state(
                arm, curve, t, duration, elbow_mode, cfg.singularity_floor)
        return cache[t]

    costs = {fields[0]: None for fields in JOINT_FIELDS.values()}
    panels, diagnostics = [], {}
    try:
        ts = np.linspace(0, duration, 257)
        ss = [state(t)['singularity_index'] for t in ts]
        points = list(np.linspace(0, duration, 33)[1:-1])
        for i in range(1, 256):
            if ss[i] < ss[i-1] and ss[i] <= ss[i+1]:
                result = minimize_scalar(lambda t: state(t)['singularity_index'],
                    bounds=(ts[i-1], ts[i+1]), method='bounded', options={'xatol': 1e-12})
                if not result.success:
                    raise ValueError('Singularity search did not converge')
                points.append(float(result.x))
        nodes, weights = np.polynomial.legendre.leggauss(32)
        for name, (field, _, _, atol) in JOINT_FIELDS.items():
            def integrand(t):
                return math.fsum(v*v for v in state(t)[name])
            q = quad(integrand, 0, duration, points=sorted(set(points)),
                     epsabs=atol, epsrel=cfg.relative_tolerance,
                     limit=cfg.subdivision_limit, full_output=1)
            if len(q) > 3:
                raise ValueError('QUADPACK warning: ' + str(q[3]))
            info = q[2]
            objective_panels = []
            for i in range(info['last']):
                lo, hi = float(info['alist'][i]), float(info['blist'][i])
                half = (hi-lo)/2
                times = ((lo+hi)/2 + half*nodes).tolist()
                actual_weights = (half*weights).tolist()
                values = [integrand(t) for t in times]
                value = math.fsum(w*v for w,v in zip(actual_weights,values))
                objective_panels.append(dict(metric=field,lo=lo,hi=hi,time_s=times,
                    weights_s=actual_weights,integrand=values,integral=value,
                    quad_integral=float(info['rlist'][i]),quad_error_estimate=float(info['elist'][i])))
            value = math.fsum(p['integral'] for p in objective_panels)
            tolerance = atol + cfg.relative_tolerance*abs(q[0])
            if not all(math.isfinite(v) for v in (value,q[0],q[1])):
                raise ValueError('Nonfinite quadrature result')
            if abs(value-q[0]) > tolerance or q[1] > tolerance:
                raise ValueError('Independent quadrature disagreement/error: ' + field)
            costs[field] = value
            diagnostics[field] = dict(quad_reference=q[0],quad_error_estimate=q[1],
                replay_difference=abs(value-q[0]),tolerance=tolerance,panels=len(objective_panels))
            panels.extend(objective_panels)
        return dict(status='accepted',costs=costs,panels=panels,diagnostics=diagnostics,
                    evaluations=len(cache),error=None)
    except (ValueError,OverflowError,ZeroDivisionError) as exc:
        return dict(status='unresolved',costs={k:None for k in costs},panels=panels,
                    diagnostics=diagnostics,evaluations=len(cache),error=str(exc))


def evaluate_v3(arm, curve, elbow_mode='up', *, obstacles=(), config=None,
                samples=161, integral_config=None):
    """New candidate result; v2 values isolated under legacy_v2, never mixed labels."""
    cfg = config or MathematicalEvaluationConfig()
    icfg = integral_config or AdaptiveIntegralConfig()
    old = simulate_trajectory(arm,curve,elbow_mode,obstacles=obstacles,config=cfg,samples=samples)
    path = adaptive_path_check(arm,curve,elbow_mode,obstacles,link_radius=cfg.link_radius_units)
    if path['status'] == 'bounded_clear':
        integral = adaptive_joint_costs(arm,curve,cfg.duration_s,elbow_mode,icfg)
    else:
        integral = dict(status='not_evaluated',costs={f[0]:None for f in JOINT_FIELDS.values()},
            panels=[],diagnostics={},evaluations=0,error='path_'+path['status'])
    rows = []
    for oldrow in old.sample_rows:
        row = dict(oldrow)
        # Always erase the old finite-difference columns, including failed samples.
        for name, (_, squared, unit, _) in JOINT_FIELDS.items():
            row[squared] = None
            for j in (1,2):
                row[f'theta{j}_{name}_{unit}'] = None
        try:
            values = differential_joint_state(arm,curve,row['time_s'],cfg.duration_s,
                                               elbow_mode,icfg.singularity_floor)
            for name, (_, squared, unit, _) in JOINT_FIELDS.items():
                row[squared] = math.fsum(v*v for v in values[name])
                for j,value in enumerate(values[name],1):
                    row[f'theta{j}_{name}_{unit}'] = value
            row['joint_sample_status'] = 'computed'
        except (ValueError,OverflowError,ZeroDivisionError) as exc:
            row['joint_sample_status'] = str(exc)
        rows.append(row)
    metrics = dict(old.metrics)
    metrics.update(integral['costs'])
    status = ('invalid' if path['status']=='invalid' else
              'accepted' if path['status']=='bounded_clear' and integral['status']=='accepted' else 'unresolved')
    metrics.update(dataset_version=DATASET_VERSION,calculation_version=CODE_VERSION,
        candidate_status=status,selection_valid=status=='accepted',path_check_status=path['status'],
        joint_integration_status=integral['status'],joint_integration_error=integral['error'],
        integration_method='E/A/J: adaptive QUADPACK partitions + replayable Gauss-Legendre32; L/K/O: v2 trapezoid',
        joint_derivative_method='differential inverse kinematics; exact curve/time derivatives',
        hard_validation_grid='bounded adaptive intervals; see path_check_status',
        fully_reachable=True if path['status']=='bounded_clear' else False if path['unreachable_witnesses'] else None,
        collision_free=True if path['status']=='bounded_clear' else False if path['collision_witnesses'] else None,
        warning=integral['error'] or '',
        legacy_sample_diagnostics_note='sample counts/minima/reliability flags are diagnostic, not continuous guarantees')
    grid_integrals = {}
    for _, (field,squared,_,_) in JOINT_FIELDS.items():
        vals = [r[squared] for r in rows]
        grid_integrals[field] = (math.fsum((a+b)*(r['time_s']-l['time_s'])/2
            for a,b,l,r in zip(vals,vals[1:],rows,rows[1:])) if all(v is not None for v in vals) else None)
    return dict(metrics=metrics,sample_rows=rows,equations=old.equations,path_check=path,
                quadrature=integral,joint_sample_grid_integrals=grid_integrals,
                legacy_v2=dict(old.metrics))
