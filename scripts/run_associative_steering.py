#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bio_fly.associative_steering import AssociativeSteeringConfig, run_associative_steering


def parse_args() -> argparse.Namespace:
    defaults = AssociativeSteeringConfig()
    parser = argparse.ArgumentParser(
        description="Replay odor-specific KC-to-MBON32 plasticity into bilateral DNa02 steering."
    )
    parser.add_argument("--annotation-path", type=Path, default=defaults.annotation_path)
    parser.add_argument("--connectivity-path", type=Path, default=defaults.connectivity_path)
    parser.add_argument("--kc-nt-inputs-path", type=Path, default=defaults.kc_nt_inputs_path)
    parser.add_argument("--male-edge-component-path", type=Path, default=defaults.male_edge_component_path)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(defaults.seeds))
    parser.add_argument("--validation-seed-start", type=int, default=defaults.validation_seed_start)
    parser.add_argument("--n-odors", type=int, default=defaults.n_odors)
    parser.add_argument("--null-repeats", type=int, default=defaults.null_repeats)
    parser.add_argument("--depression-fraction", type=float, default=defaults.depression_fraction)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AssociativeSteeringConfig(
        annotation_path=args.annotation_path,
        connectivity_path=args.connectivity_path,
        kc_nt_inputs_path=args.kc_nt_inputs_path,
        male_edge_component_path=args.male_edge_component_path,
        output_dir=args.output_dir,
        device=args.device,
        seeds=tuple(args.seeds),
        validation_seed_start=args.validation_seed_start,
        n_odors=args.n_odors,
        null_repeats=args.null_repeats,
        depression_fraction=args.depression_fraction,
    )
    paths = run_associative_steering(config)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
