"""Generate reproducible, poster-ready phase-2 analysis assets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from robot_planner.analysis import build_analysis_data  # noqa: E402
from robot_planner.poster import generate_poster_assets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=PROJECT_ROOT / "data" / "phase2" / "trajectory_metrics.csv",
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=PROJECT_ROOT / "data" / "phase2" / "scenarios.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "poster_assets" / "phase2_baseline",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = build_analysis_data(
        args.metrics,
        args.scenarios,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    outputs = generate_poster_assets(
        data,
        args.output,
        metrics_path=args.metrics,
        scenarios_path=args.scenarios,
        seed=args.seed,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    print(f"Generated {len(outputs.generated_files)} files in {outputs.output_dir}")
    print(f"Representative scenario: {outputs.representative_scenario_id}")


if __name__ == "__main__":
    main()
