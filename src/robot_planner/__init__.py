"""Mathematical core for the 2D robot-arm path explorer."""

from .curves import CurvePath, make_curve
from .evaluation import CircleObstacle, EvaluationConfig, TrajectoryEvaluation, evaluate_trajectory
from .kinematics import ArmPose, JointSolution, PlanarArm2R, UnreachablePointError
from .phase4_evaluation import (
    MathematicalEvaluationConfig,
    MathematicalTrajectoryResult,
    simulate_trajectory,
)

__all__ = [
    "ArmPose",
    "CircleObstacle",
    "CurvePath",
    "EvaluationConfig",
    "JointSolution",
    "MathematicalEvaluationConfig",
    "MathematicalTrajectoryResult",
    "PlanarArm2R",
    "TrajectoryEvaluation",
    "UnreachablePointError",
    "evaluate_trajectory",
    "make_curve",
    "simulate_trajectory",
]
