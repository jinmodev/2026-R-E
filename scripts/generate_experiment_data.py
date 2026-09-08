"""Generate phase-2 CSV data for later Python analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from robot_planner.experiments import DEFAULT_SAMPLES, DEFAULT_SEED, write_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=int, default=240)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "phase2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = write_dataset(
        args.output_dir,
        scenario_count=args.scenarios,
        seed=args.seed,
        samples=args.samples,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

