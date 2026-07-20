from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA = ROOT / "data" / "source_data"
DEFAULT_SOURCE = ROOT / "outputs" / "kc_nt_lateralization" / "serotonin_dominant_upstream_by_class.csv"
CORE_SUBTYPES = [
    "KCa'b'-ap1",
    "KCa'b'-ap2",
    "KCa'b'-m",
    "KCab-c",
    "KCab-m",
    "KCab-p",
    "KCab-s",
    "KCg-d",
    "KCg-m",
]


def build_compartment_switch(source: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    upstream = pd.read_csv(source)
    dpm = upstream[
        upstream["pre_cell_class"].eq("MBIN")
        & upstream["pre_cell_type"].eq("DPM")
        & upstream["kc_hemibrain_type"].isin(CORE_SUBTYPES)
    ].copy()
    pivot = (
        dpm.pivot_table(
            index="kc_hemibrain_type",
            columns="kc_side",
            values=["n_edges", "syn_count"],
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(CORE_SUBTYPES)
        .fillna(0)
    )
    rows: list[dict[str, object]] = []
    for subtype in CORE_SUBTYPES:
        left_syn = int(pivot.loc[subtype, ("syn_count", "left")])
        right_syn = int(pivot.loc[subtype, ("syn_count", "right")])
        rows.append(
            {
                "kc_subtype": subtype,
                "compartment": "alpha_prime_beta_prime" if subtype.startswith("KCa'b'") else "other_core_kc",
                "left_high_confidence_dpm_5ht_label_synapses": left_syn,
                "right_high_confidence_dpm_5ht_label_synapses": right_syn,
                "right_minus_left_synapses": right_syn - left_syn,
                "left_connections": int(pivot.loc[subtype, ("n_edges", "left")]),
                "right_connections": int(pivot.loc[subtype, ("n_edges", "right")]),
                "inference_boundary": (
                    "within-connectome subtype pattern among DPM-to-KC edges with ser_avg>=0.5; "
                    "not biological replication or direct serotonin release"
                ),
            }
        )
    summary = pd.DataFrame.from_records(rows)
    alpha = summary[summary["compartment"].eq("alpha_prime_beta_prime")]
    other = summary[summary["compartment"].eq("other_core_kc")]
    observed_delta = int(alpha["right_minus_left_synapses"].sum())
    all_deltas = summary.set_index("kc_subtype")["right_minus_left_synapses"].to_dict()
    null_sums = [
        sum(int(all_deltas[subtype]) for subtype in combination)
        for combination in itertools.combinations(CORE_SUBTYPES, len(alpha))
    ]
    alpha_positive = int(alpha["right_minus_left_synapses"].gt(0).sum())
    other_positive = int(other["right_minus_left_synapses"].gt(0).sum())
    stats = {
        "edge_definition": "DPM-to-KC connections with ser_avg>=0.5",
        "alpha_prime_beta_prime_left_synapses": int(alpha["left_high_confidence_dpm_5ht_label_synapses"].sum()),
        "alpha_prime_beta_prime_right_synapses": int(alpha["right_high_confidence_dpm_5ht_label_synapses"].sum()),
        "alpha_prime_beta_prime_right_minus_left": observed_delta,
        "other_core_kc_left_synapses": int(other["left_high_confidence_dpm_5ht_label_synapses"].sum()),
        "other_core_kc_right_synapses": int(other["right_high_confidence_dpm_5ht_label_synapses"].sum()),
        "alpha_prime_beta_prime_positive_subtypes": alpha_positive,
        "other_positive_subtypes": other_positive,
        "subtype_label_permutation_n": len(null_sums),
        "subtype_label_permutation_p_ge_observed": sum(value >= observed_delta for value in null_sums) / len(null_sums),
        "direction_enrichment_fisher_p_greater": float(
            fisher_exact(
                [[alpha_positive, len(alpha) - alpha_positive], [other_positive, len(other) - other_positive]],
                alternative="greater",
            ).pvalue
        ),
        "boundary": (
            "Exact tests quantify an internal compartment pattern in one connectome. "
            "KC subtypes and synapses are not independent animals."
        ),
    }
    return summary, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the DPM-to-KC high-confidence 5-HT-label compartment audit.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-csv", type=Path, default=SOURCE_DATA / "dpm_5ht_compartment_switch.csv")
    parser.add_argument("--output-json", type=Path, default=SOURCE_DATA / "dpm_5ht_compartment_switch_statistics.json")
    args = parser.parse_args()
    summary, stats = build_compartment_switch(args.source)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    args.output_json.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
