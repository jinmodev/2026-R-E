"""Opt-in differential IK and bounded adaptive collision diagnostics.

No changes to Phase-4 default metrics or labels. Binary64 bounds below are
conservative analytic bounds, not outward-rounded interval arithmetic or a
physical robot safety certificate.
"""
from __future__ import annotations
import math
from .curves import CurvePath
from .evaluation import pose_obstacle_clearance
from .phase4_evaluation import _curve_value_derivatives

VERSION='phase4-refinement-v1'

def third_curve_derivative(curve: CurvePath,u: float):
    p=curve.parameters; x0,y0=curve.start;x1,y1=curve.end
    if curve.key=='line' or (curve.key=='arc' and 'center_x' not in p):return 0.,0.
    if curve.key=='arc':
        angle=p['theta_start']+p['theta_delta']*u;v=p['radius']*p['theta_delta']**3
        return v*math.sin(angle),-v*math.cos(angle)
    if curve.key=='bezier':
        return (6*(-x0+3*p['control1_x']-3*p['control2_x']+x1),6*(-y0+3*p['control1_y']-3*p['control2_y']+y1))
    if curve.key=='hermite':
        return (12*x0+6*p['tangent0_x']-12*x1+6*p['tangent1_x'],12*y0+6*p['tangent0_y']-12*y1+6*p['tangent1_y'])
    if curve.key=='cycloid':
        dx,dy=x1-x0,y1-y0;c=math.hypot(dx,dy);a=2*math.pi*u
        forward=4*math.pi**2*math.cos(a);trans=-4*math.pi**3*math.sin(a)*p['height']
        return dx*forward-dy/c*trans,dy*forward+dx/c*trans
    if curve.key=='limacon':
        a=p['phase']+p['sweep']*u
        rx=p['a']*math.sin(a)+4*p['b']*math.sin(2*a)
        ry=p['side']*(-p['a']*math.cos(a)-4*p['b']*math.cos(2*a))
        scale=p['scale']*p['sweep']**3;c,s=math.cos(p['rotation']),math.sin(p['rotation'])
        return scale*(c*rx-s*ry),scale*(s*rx+c*ry)
    raise ValueError('Unsupported curve')

def differential_joint_state(arm,curve,t,duration,elbow_mode='up',singularity_floor=1e-10):
    """Return joint first/second/third t derivatives without finite differences."""
    if not math.isfinite(duration) or duration<=0:raise ValueError('duration must be positive and finite')
    if not 0<singularity_floor<1:raise ValueError('singularity_floor must be in (0,1)')
    if not 0<=t<=duration:raise ValueError('time outside [0,T]')
    s=t/duration;u=10*s**3-15*s**4+6*s**5
    du=(30*s**2-60*s**3+30*s**4)/duration
    ddu=(60*s-180*s**2+120*s**3)/duration**2
    dddu=(60-360*s+360*s**2)/duration**3
    x,y,dx,dy,ddx,ddy=_curve_value_derivatives(curve,u)
    tx,ty=third_curve_derivative(curve,u)
    joint=arm.inverse((x,y),elbow_mode)
    if abs(math.sin(joint.theta2))<=singularity_floor:raise ValueError('singular differential IK')
    a,b=joint.theta1,joint.theta1+joint.theta2
    e1=(arm.link1*math.cos(a),arm.link1*math.sin(a));e2=(arm.link2*math.cos(b),arm.link2*math.sin(b))
    n1=(-e1[1],e1[0]);n2=(-e2[1],e2[0]);det=n1[0]*n2[1]-n1[1]*n2[0]
    def solve(rx,ry):return (rx*n2[1]-ry*n2[0])/det,(n1[0]*ry-n1[1]*rx)/det
    v1,v2=solve(dx*du,dy*du)
    a1,a2=solve(ddx*du**2+dx*ddu+e1[0]*v1*v1+e2[0]*v2*v2,ddy*du**2+dy*ddu+e1[1]*v1*v1+e2[1]*v2*v2)
    j1,j2=solve(tx*du**3+3*ddx*du*ddu+dx*dddu+3*(e1[0]*v1*a1+e2[0]*v2*a2)+n1[0]*v1**3+n2[0]*v2**3,
                 ty*du**3+3*ddy*du*ddu+dy*dddu+3*(e1[1]*v1*a1+e2[1]*v2*a2)+n1[1]*v1**3+n2[1]*v2**3)
    return {'velocity':(v1,v2-v1),'acceleration':(a1,a2-a1),'jerk':(j1,j2-j1),'singularity_index':abs(math.sin(joint.theta2))}

def curve_speed_bound(curve):
    """Global upper bound for ||dp/du|| on [0,1], all six built-in families."""
    p=curve.parameters;a,b=curve.start,curve.end;chord=math.dist(a,b)
    if curve.key=='line' or (curve.key=='arc' and 'center_x' not in p):return chord
    if curve.key=='arc':return abs(p['radius']*p['theta_delta'])
    if curve.key in ['bezier','hermite']:
        if curve.key=='bezier':controls=[a,(p['control1_x'],p['control1_y']),(p['control2_x'],p['control2_y']),b]
        else:controls=[a,(a[0]+p['tangent0_x']/3,a[1]+p['tangent0_y']/3),(b[0]-p['tangent1_x']/3,b[1]-p['tangent1_y']/3),b]
        return 3*max(math.dist(x,y) for x,y in zip(controls,controls[1:]))
    if curve.key=='cycloid':return 2*chord+math.pi*abs(p['height'])
    if curve.key=='limacon':return abs(p['scale']*p['sweep'])*(abs(p['a'])+2*abs(p['b']))
    raise ValueError('Unsupported curve')

def adaptive_path_check(arm,curve,elbow_mode='up',obstacles=(),*,link_radius=.05,initial_intervals=16,max_depth=10,max_evaluations=2049):
    """Return invalid / bounded_clear / unresolved; never silently accept limits.

    For an interval of half-width h, p stays within M*h of its midpoint.
    Strictly interior radial bounds give a lower bound s for |sin(theta2)|.
    Elbow speed <= M/s, so every link segment's Hausdorff displacement from
    its midpoint pose <= M*h/s. Distance to fixed obstacles is 1-Lipschitz.
    """
    if any(type(v) is not int for v in (initial_intervals,max_depth,max_evaluations)) or initial_intervals<1 or max_depth<0 or max_evaluations<1 or not math.isfinite(link_radius) or link_radius<0:raise ValueError('Invalid adaptive limits')
    obstacles=tuple(obstacles);M=curve_speed_bound(curve);cache={};unreachable=[];collisions=[];unresolved=[];bounded=0
    def evaluate(u):
        if u in cache:return cache[u]
        if len(cache)>=max_evaluations:return None
        x,y,*_=_curve_value_derivatives(curve,u);r=arm.radial_distance((x,y))
        reach=arm.is_reachable((x,y),tolerance=0.)
        clearance=None
        if reach:
            solution=arm.inverse((x,y),elbow_mode)
            if obstacles:clearance=pose_obstacle_clearance(arm,solution,obstacles,link_radius_units=link_radius)
            if clearance is not None and clearance<=0:collisions.append(u)
        else:unreachable.append(u)
        cache[u]=(r,reach,clearance);return cache[u]
    # Endpoints are always checked; unknown endpoints make the result unresolved.
    for u in [0.,1.]:
        if evaluate(u) is None:unresolved.append((u,u))
    stack=[(i/initial_intervals,(i+1)/initial_intervals,0) for i in reversed(range(initial_intervals))]
    while stack:
        lo,hi,depth=stack.pop();mid=(lo+hi)*.5;h=(hi-lo)*.5;state=evaluate(mid)
        if state is None:unresolved.append((lo,hi));continue
        r,reach,clearance=state
        if not reach or (clearance is not None and clearance<=0):continue
        rlo=max(0.,r-M*h);rhi=r+M*h
        radial_inside=rlo>arm.min_reach+1e-12 and rhi<arm.max_reach-1e-12
        safe=False
        if radial_inside:
            if not obstacles:safe=True
            else:
                denom=2*arm.link1*arm.link2;const=arm.link1**2+arm.link2**2
                cmax=max(abs((rlo*rlo-const)/denom),abs((rhi*rhi-const)/denom))
                smin=math.sqrt(max(0.,1-cmax*cmax))
                safe=smin>1e-12 and clearance>M*h/smin+1e-12
        if safe:bounded+=1
        elif depth>=max_depth:unresolved.append((lo,hi))
        else:stack.extend([(mid,hi,depth+1),(lo,mid,depth+1)])
    status='invalid' if unreachable or collisions else 'unresolved' if unresolved else 'bounded_clear'
    return dict(version=VERSION,status=status,evaluations=len(cache),bounded_intervals=bounded,unresolved_intervals=len(unresolved),unreachable_witnesses=sorted(unreachable)[:16],collision_witnesses=sorted(collisions)[:16],
                unresolved_examples=unresolved[:16],min_sample_clearance=min((z[2] for z in cache.values() if z[2] is not None),default=None),
                interpretation='binary64 analytic bounds under model assumptions; no physical safety certification')
