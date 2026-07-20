#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bio_fly.lateralized_steering import LateralizedSteeringConfig, run_lateralized_steering


def parse_args() -> argparse.Namespace:
    defaults = LateralizedSteeringConfig()
    parser = argparse.ArgumentParser(
        description="Test the KC chemical gate against the bilateral DNa02 steering readout."
    )
    parser.add_argument("--annotation-path", type=Path, default=defaults.annotation_path)
    parser.add_argument("--connectivity-path", type=Path, default=defaults.connectivity_path)
    parser.add_argument("--kc-nt-inputs-path", type=Path, default=defaults.kc_nt_inputs_path)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(defaults.seeds))
    parser.add_argument("--validation-seed-start", type=int, default=defaults.validation_seed_start)
    parser.add_argument("--n-odors", type=int, default=defaults.n_odors)
    parser.add_argument("--n-mediation-odors", type=int, default=defaults.n_mediation_odors)
    parser.add_argument("--null-repeats", type=int, default=defaults.null_repeats)
    parser.add_argument("--steps", type=int, default=defaults.propagation_steps)
    parser.add_argument("--max-active", type=int, default=defaults.max_active)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = LateralizedSteeringConfig(
        annotation_path=args.annotation_path,
        connectivity_path=args.connectivity_path,
        kc_nt_inputs_path=args.kc_nt_inputs_path,
        output_dir=args.output_dir,
        device=args.device,
        seeds=tuple(args.seeds),
        validation_seed_start=args.validation_seed_start,
        n_odors=args.n_odors,
        n_mediation_odors=args.n_mediation_odors,
        null_repeats=args.null_repeats,
        propagation_steps=args.steps,
        max_active=args.max_active,
    )
    paths = run_lateralized_steering(cfg)
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
