#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bio_fly.grasp_lateralization import GraspAnalysisConfig, run_grasp_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze paired left/right GRASP signal without assuming direction.")
    parser.add_argument("--input", type=Path, required=True, help="CSV with fly_id,left_signal,right_signal")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experimental-group", default="experimental")
    parser.add_argument("--control-group", default="control")
    parser.add_argument("--bootstrap-repeats", type=int, default=20_000)
    parser.add_argument("--permutation-repeats", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GraspAnalysisConfig(
        experimental_group=args.experimental_group,
        control_group=args.control_group or None,
        bootstrap_repeats=args.bootstrap_repeats,
        permutation_repeats=args.permutation_repeats,
        random_seed=args.seed,
    )
    paths = run_grasp_analysis(args.input, args.output_dir, config)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
