#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "source_data"
HASHES = ROOT / "data" / "source_data.sha256"


def verify_hashes() -> int:
    checked = 0
    for line in HASHES.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = ROOT / relative
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise AssertionError(f"SHA-256 mismatch for {relative}: {observed}")
        checked += 1
    return checked


def close(value: float, target: float, tolerance: float = 1e-9) -> None:
    if not np.isclose(float(value), float(target), atol=tolerance, rtol=0):
        raise AssertionError(f"Expected {target}, observed {value}")


def audit_headline_results() -> dict[str, object]:
    subtype = pd.read_csv(DATA / "subtype_replication_meta_analysis.csv")
    subtype = subtype[subtype["replication_unit"].eq("KC subtype")]
    if len(subtype) != 9 or not bool(subtype["right_minus_left_fraction"].gt(0).all()):
        raise AssertionError("Expected a positive 5-HT-predicted shift in all nine KC subtypes")

    specifications = pd.read_csv(DATA / "kc_5ht_specification_curve.csv")
    fraction_like = specifications[specifications["display_unit"].eq("percentage points")]
    positive_specs = int(fraction_like["estimate_pp"].gt(0).sum())
    if (positive_specs, len(fraction_like)) != (46, 52):
        raise AssertionError("Expected 46/52 positive fraction-like specifications")

    paired = pd.read_csv(DATA / "dpm_29fly_paired_slopes.csv")
    if len(paired) != 29 or int(paired["right_minus_left"].gt(0).sum()) != 22:
        raise AssertionError("Expected 22/29 positive fly-level DPM imaging effects")
    close(paired["right_minus_left"].mean(), 0.11706949285991511)

    point = pd.read_csv(DATA / "model_causal_point_dose_response.csv").sort_values("gate_strength")
    if not bool(np.all(np.diff(point["mean_delta_lateral_code_vs_sym"]) > 0)):
        raise AssertionError("Point-model dose response is not strictly monotonic")

    arbor = pd.read_csv(DATA / "model_causal_arbor_case_concordance.csv")
    cohort_counts = arbor.groupby("cohort")["case_id"].nunique().to_dict()
    if cohort_counts != {"effective_dimension_selected_15": 15, "lateral_support_selected_72": 72}:
        raise AssertionError(f"Unexpected Arbor cohorts: {cohort_counts}")

    steering = pd.read_csv(DATA / "lateralized_steering_contrast_summary.csv")
    steering_real = steering[
        steering["split"].eq("validation") & steering["contrast"].eq("real_minus_symmetrized")
    ].iloc[0]
    if (int(steering_real["positive_odors"]), int(steering_real["n_odors"])) != (118, 120):
        raise AssertionError("Expected 118/120 held-out steering shifts")

    controls = pd.read_csv(DATA / "associative_steering_structural_controls.csv").set_index("control")
    structural_ratio = float(controls.loc["registered", "left_right_ratio_of_means"])
    close(structural_ratio, 6.984183964682402)
    close(controls.loc["input_and_output_equalized", "left_right_ratio_of_means"], 1.0300499504959448)
    close(controls.loc["output_sides_swapped", "left_right_ratio_of_means"], 0.43651149779265014)

    stages = pd.read_csv(DATA / "associative_steering_stage_contrasts.csv")
    validation = stages[stages["split"].eq("validation")].set_index("contrast")
    retrieval = validation.loc["real_retrieval_only_minus_symmetrized"]
    both = validation.loc["real_both_minus_symmetrized"]
    if int(retrieval["positive_odors"]) != 118 or int(both["positive_odors"]) != 120:
        raise AssertionError("Unexpected retrieval-stage chemical-gate direction counts")

    return {
        "verified_source_files": verify_hashes(),
        "kc_subtypes_positive": "9/9",
        "fraction_like_specifications_positive": "46/52",
        "dpm_imaging_flies_positive": "22/29",
        "dpm_imaging_mean_right_minus_left": float(paired["right_minus_left"].mean()),
        "arbor_case_counts": cohort_counts,
        "held_out_dna02_shifts_positive": "118/120",
        "structural_left_right_command_ratio_under_symmetrized_gate": structural_ratio,
        "chemical_gate_retrieval_shifts_positive": "118/120",
        "boundary": "model panels and odors are deterministic robustness units, not animals",
    }


def main() -> None:
    print(json.dumps(audit_headline_results(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
