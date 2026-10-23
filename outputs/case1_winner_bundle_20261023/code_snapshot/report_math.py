"""Corrected classroom mathematics shared by the report UI and data exports."""
from __future__ import annotations

import json
import math

from .curves import CurvePath

VERSION = "report-equations-v1"


def corrected_sine_limacon(v: float, amplitude: float, offset: float) -> dict:
    """r=amplitude*sin(phi)+offset, v=tan(phi), cos(phi)>0.

    v is a slope parameter, not time. Singular points have no ordinary curvature.
    """
    a, b, q = amplitude, offset, 1 + v*v
    x = a*v/q+b/math.sqrt(q)
    y = a*v*v/q+b*v/math.sqrt(q)
    dx = a*(1-v*v)/q**2-b*v/q**1.5
    dy = 2*a*v/q**2+b/q**1.5
    ddx = (2*a*v**3-6*a*v)/q**3+b*(2*v*v-1)/q**2.5
    ddy = (2*a-6*a*v*v)/q**3-3*b*v/q**2.5
    speed2 = dx*dx+dy*dy
    return dict(x=x, y=y, dx=dx, dy=dy, ddx=ddx, ddy=ddy,
                cross=dx*ddy-dy*ddx,
                curvature=None if speed2 == 0 else abs(dx*ddy-dy*ddx)/speed2**1.5)


def cycloid_length(A: float, B: float) -> float:
    """Length of (A*(phi-sin(phi)), B*(1-cos(phi))), phi in [0,2*pi]."""
    if not math.isfinite(A) or not math.isfinite(B) or A <= 0:
        raise ValueError("A must be positive; coefficients must be finite")
    B = abs(B)
    if math.isclose(A, B, rel_tol=1e-12, abs_tol=0):
        return 8*A
    if B > A:
        c = math.sqrt(B*B-A*A)
        return 4*(B+A*A/c*math.asinh(c/A))
    c = math.sqrt(A*A-B*B)
    return 4*(B+A*A/c*math.asin(min(1., c/A)))


def polynomial_coefficients(curve: CurvePath):
    """Ascending power coefficients, per coordinate, for polynomial families."""
    p = curve.parameters
    start, end = curve.start, curve.end
    if curve.key == "line" or (curve.key == "arc" and "radius" not in p):
        return [(a, b-a, 0., 0.) for a, b in zip(start, end)]
    if curve.key == "bezier":
        result = []
        for j, axis in enumerate(("x", "y")):
            a, b = start[j], end[j]
            c, d = float(p["control1_"+axis]), float(p["control2_"+axis])
            result.append((a, 3*(c-a), 3*(a-2*c+d), -a+3*c-3*d+b))
        return result
    if curve.key == "hermite":
        result = []
        for j, axis in enumerate(("x", "y")):
            a, b = start[j], end[j]
            c, d = float(p["tangent0_"+axis]), float(p["tangent1_"+axis])
            result.append((a, c, -3*a-2*c+3*b-d, 2*a+c-2*b+d))
        return result
    return None


def report_equations(curve: CurvePath, duration_s: float = 3.) -> dict[str, str]:
    """Human-readable equations, no sampled fit and no change to source metrics."""
    D = math.dist(curve.start, curve.end)
    p = curve.parameters
    result = dict(
        report_equation_version=VERSION,
        variable_conventions="u in [0,1]: path progress; phi: curve angle; t in [0,T]: seconds; theta1/theta2: joint angles; v=tan(phi) only on a stated branch",
        length_integral="L=int_0^1 sqrt(x_u'(u)^2+y_u'(u)^2) du",
        ordinary_curvature="kappa(u)=abs(x_u'(u)*y_u''(u)-y_u'(u)*x_u''(u))/(x_u'(u)^2+y_u'(u)^2)^(3/2); undefined at zero speed",
        regularized_curvature="K_epsilon=int_0^1 (x'*y''-y'*x'')^2/(x'^2+y'^2+epsilon^2)^(5/2) du; same fixed epsilon as source",
        effort_integral="E_theta=int_0^T (theta1_dot^2+theta2_dot^2) dt [rad^2/s]",
        acceleration_integral="A=int_0^T (theta1_ddot^2+theta2_ddot^2) dt",
        jerk_integral="J=int_0^T (theta1_third^2+theta2_third^2) dt; not curvature",
        signed_work_integral="W_net=int_0^T (tau1*theta1_dot+tau2*theta2_dot) dt [J]",
        positive_work_integral="W_positive=int_0^T (max(tau1*theta1_dot,0)+max(tau2*theta2_dot,0)) dt [J]",
        rms_torque_integral="tau_rms=sqrt(int_0^T (tau1^2+tau2^2) dt / T) [N*m]",
        motor_energy_integral="E_model=W_positive+int_0^T R*(tau1^2+tau2^2)/Kt^2 dt; no regenerative recovery; model estimate",
        torque_equation="tau=M(theta)*theta_ddot+C(theta,theta_dot)+G(theta)+damping*theta_dot; coupled 2R slender-link model",
        work_energy_identity="W_net=Delta(kinetic+potential)+int damping*(theta1_dot^2+theta2_dot^2) dt",
        one_joint_reference="For a separate ideal 1R vertical example: tau=I*theta_ddot+m*g*lc*cos(theta); int tau*theta_dot dt=[I*theta_dot^2/2+m*g*lc*sin(theta)]_0^T",
        corrected_note_derivatives="For r=a*sin(phi)+b, v=tan(phi), cos(phi)>0: x''=(2*a*v^3-6*a*v)/(1+v^2)^3+b*(2*v^2-1)/(1+v^2)^(5/2); y''=(2*a-6*a*v^2)/(1+v^2)^3-3*b*v/(1+v^2)^(5/2)",
        corrected_note_curvature="kappa(v)=abs(2*a^2+b^2+3*a*b*v/sqrt(1+v^2))/(a^2+b^2+2*a*b*v/sqrt(1+v^2))^(3/2); only regular points on the chosen branch",
        progress_equation=f"u=q(t)=10*(t/{duration_s:.15g})^3-15*(t/{duration_s:.15g})^4+6*(t/{duration_s:.15g})^5",
    )
    coefficients = polynomial_coefficients(curve)
    if coefficients:
        for axis, c in zip(("x", "y"), coefficients):
            result[axis+"_u_simplified"] = f"{axis}(u)=" + "+".join(f"({v:.15g})*u^{i}" for i, v in enumerate(c))
            result[axis+"_u_first"] = f"d{axis}/du=({c[1]:.15g})+({2*c[2]:.15g})*u+({3*c[3]:.15g})*u^2"
            result[axis+"_u_second"] = f"d2{axis}/du2=({2*c[2]:.15g})+({6*c[3]:.15g})*u"
    if curve.key == "cycloid":
        A, B = D/(2*math.pi), float(p["height"])/2
        result.update(
            native_equation=f"X_local=A*(phi-sin(phi)); Y_local=B*(1-cos(phi)); phi=2*pi*u; A={A:.15g}; B={B:.15g}",
            native_first="dX/dphi=A*(1-cos(phi)); dY/dphi=B*sin(phi)",
            native_second="d2X/dphi2=A*sin(phi); d2Y/dphi2=B*cos(phi)",
            native_length="L=8*int_0^1 sqrt(A^2+(B^2-A^2)*z^2) dz, z=cos(phi/2); ordinary 8*A formula requires abs(B)=A",
            native_area="signed area between path and chord=int Y dX=3*pi*A*B; additional geometry, not an objective",
            analytic_length_units=format(cycloid_length(A, B), ".17g"),
            native_area_units2=format(3*math.pi*A*B, ".17g"),
            native_warning="Height-adjusted cycloid, generally A!=abs(B); zero geometric speed at both endpoints, ordinary curvature undefined there",
        )
    elif curve.key == "limacon":
        a, b, phase, sweep, scale = (float(p[k]) for k in ("a","b","phase","sweep","scale"))
        end = phase+sweep
        def area_primitive(phi):
            return .5*(a*a+b*b/2)*phi+a*b*math.sin(phi)+b*b*math.sin(2*phi)/8
        result.update(
            native_equation=f"r(phi)=({a:.15g})+({b:.15g})*cos(phi); phi=({phase:.15g})+({sweep:.15g})*u",
            native_first=f"dr/dphi=-({b:.15g})*sin(phi)",
            native_second=f"d2r/dphi2=-({b:.15g})*cos(phi)",
            native_transform=json.dumps(dict(p), sort_keys=True),
            native_length=f"L=({scale:.15g})*abs(int_({phase:.15g})^({end:.15g}) sqrt(a^2+b^2+2*a*b*cos(phi)) dphi)",
            native_curvature="kappa=abs(a^2+2*b^2+3*a*b*cos(phi))/(scale*(a^2+b^2+2*a*b*cos(phi))^(3/2)); undefined at zero speed",
            native_area="sector area=scale^2*abs([0.5*(a^2+b^2/2)*phi+a*b*sin(phi)+b^2*sin(2*phi)/8] at angular endpoints); extra geometry, not swept robot area or an objective",
            native_area_units2=format(scale*scale*abs(area_primitive(end)-area_primitive(phase)), ".17g"),
            native_warning="The native pole differs from the robot base; side reflection, rotation, scale and translation must all be retained",
        )
    elif curve.key == "arc" and "radius" in p:
        result["analytic_length_units"] = format(abs(float(p["radius"])*float(p["theta_delta"])), ".17g")
        result["native_equation"] = "p(u)=center+radius*(cos(theta_start+theta_delta*u),sin(theta_start+theta_delta*u))"
    elif curve.key == "line":
        result["analytic_length_units"] = format(D, ".17g")
    return result


def format_report_equations(equations: dict[str, str]) -> str:
    sections = [
        ("기호와 시간", ("variable_conventions","progress_equation")),
        ("곡선의 원형식·미분", ("native_equation","native_first","native_second","x_u_simplified","y_u_simplified","x_u_first","y_u_first","x_u_second","y_u_second","native_transform")),
        ("곡률과 적분", ("ordinary_curvature","native_curvature","length_integral","native_length","analytic_length_units","regularized_curvature","native_area","native_area_units2","native_warning")),
        ("관절 운동과 토크·에너지", ("effort_integral","acceleration_integral","jerk_integral","torque_equation","signed_work_integral","positive_work_integral","rms_torque_integral","motor_energy_integral","work_energy_identity","one_joint_reference")),
        ("팀원 노트 교정식", ("corrected_note_derivatives","corrected_note_curvature")),
    ]
    return "\n\n".join(title+"\n"+"\n\n".join(equations[k] for k in keys if k in equations)
                       for title, keys in sections)
