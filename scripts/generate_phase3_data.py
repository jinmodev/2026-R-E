"""Generate obstacle-aware multi-objective phase-3 experiment data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from robot_planner.evaluation import EvaluationConfig  # noqa: E402
from robot_planner.phase3 import write_phase3_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "phase3",
    )
    parser.add_argument("--scenarios", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--samples", type=int, default=161)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument(
        "--motion-plane",
        choices=("horizontal", "vertical"),
        default="horizontal",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = EvaluationConfig(
        duration_s=args.duration,
        motion_plane=args.motion_plane,
    )
    metadata = write_phase3_dataset(
        args.output,
        scenario_count=args.scenarios,
        seed=args.seed,
        samples=args.samples,
        config=config,
    )
    print(f"Generated {metadata['run_count']} runs in {args.output}")
    print(f"Dataset version: {metadata['dataset_version']}")


if __name__ == "__main__":
    main()
