"""Generate Phase-4 mathematical trajectory data and time-sample partitions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from robot_planner.phase4 import (  # noqa: E402
    DEFAULT_PHASE4_SEED,
    write_phase4_dataset,
)
from robot_planner.phase4_evaluation import MathematicalEvaluationConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "phase4",
        help="New output directory. Existing data requires --resume.",
    )
    parser.add_argument("--scenarios", type=int, default=5_000)
    parser.add_argument("--challenge-scenarios", type=int, default=500)
    parser.add_argument("--seed", type=int, default=DEFAULT_PHASE4_SEED)
    parser.add_argument("--samples", type=int, default=161)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel scenario processes. Use 1 for the lowest memory footprint.",
    )
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--link-radius", type=float, default=0.05)
    parser.add_argument("--clearance-epsilon", type=float, default=0.05)
    parser.add_argument("--curvature-epsilon-ratio", type=float, default=1e-3)
    parser.add_argument(
        "--sample-format",
        choices=("auto", "parquet", "csv"),
        default="auto",
        help="auto prefers compressed Parquet when pyarrow is installed.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume batches only when all generation parameters match.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MathematicalEvaluationConfig(
        duration_s=args.duration,
        link_radius_units=args.link_radius,
        clearance_epsilon_units=args.clearance_epsilon,
        curvature_speed_epsilon_ratio=args.curvature_epsilon_ratio,
    )
    metadata = write_phase4_dataset(
        args.output,
        scenario_count=args.scenarios,
        challenge_count=args.challenge_scenarios,
        seed=args.seed,
        samples=args.samples,
        batch_size=args.batch_size,
        workers=args.workers,
        resume=args.resume,
        sample_format=args.sample_format,
        config=config,
        progress_callback=print,
    )
    print(f"Dataset version: {metadata['dataset_version']}")
    print(f"Candidate rows: {metadata['run_count']}")
    print(f"Winner labels: {metadata['winner_label_count']}")
    print(f"Sample format: {metadata['sample_storage_format']}")
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
