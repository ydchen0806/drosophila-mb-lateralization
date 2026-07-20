#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bio_fly.nt_analysis import run_kc_nt_analysis
from bio_fly.paths import DEFAULT_OUTPUT_ROOT, RAW_DATA_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze KC neurotransmitter input lateralization from FlyWire connections.")
    parser.add_argument(
        "--connections",
        type=Path,
        default=RAW_DATA_ROOT / "zenodo_10676866" / "proofread_connections_783.feather",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "kc_nt_lateralization")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_kc_nt_analysis(output_dir=args.output_dir, connections_path=args.connections)
    effects = pd.read_csv(paths["fraction_stats"])
    top = effects.reindex(effects["cohens_d"].abs().sort_values(ascending=False).index).head(12)
    direction = pd.read_csv(paths["direction_tests"])
    print(
        json.dumps(
            {
                "outputs": {key: str(value) for key, value in paths.items()},
                "direction_tests": direction.to_dict(orient="records"),
                "top_abs_effects": top[
                    [
                        "hemibrain_type",
                        "cell_type",
                        "nt",
                        "left_mean_fraction",
                        "right_mean_fraction",
                        "right_laterality_index",
                        "cohens_d",
                        "fdr_q",
                    ]
                ].to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
