#!/usr/bin/env python
"""Build compact model-intervention evidence tables for the active manuscript."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA = ROOT / "data" / "source_data"
OUTPUTS = ROOT / "outputs"

POINT_DOSE_RAW = OUTPUTS / "lateralization_representation_memory" / "lateralization_representation_raw.csv"
POINT_MECHANISM_RAW = OUTPUTS / "lateralization_mechanism_suite" / "lateralization_mechanism_representation_raw.csv"
ARBOR_SUPPORT_DIR = OUTPUTS / "arbor_slide16_17_glomerulus_combo_compare_gate025_baseline_weights"
ARBOR_ORTHOGONAL_DIR = OUTPUTS / "arbor_slide16_17_norm_eff_dim_fastcheck_gate025"
BEHAVIOR_RAW = OUTPUTS / "lateralization_behavior_closure" / "behavior_closure_raw.csv"
STEERING_DIR = OUTPUTS / "lateralized_steering"


def bootstrap_mean_ci(values: np.ndarray, *, seed: int, repeats: int = 20_000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(repeats, values.size), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def two_sided_all_same_sign_p(n: int) -> float:
    return float(2.0 / (2.0**int(n))) if n > 0 else float("nan")


def observed_all_same_sign_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    nonzero = values[values != 0]
    if nonzero.size and (np.all(nonzero > 0) or np.all(nonzero < 0)):
        return two_sided_all_same_sign_p(int(nonzero.size))
    return float("nan")


def add_delta_from_seed_baseline(
    frame: pd.DataFrame,
    *,
    baseline_condition: str,
    value_column: str,
    output_column: str,
) -> pd.DataFrame:
    baseline = frame.loc[frame["condition_id"].eq(baseline_condition), ["seed", value_column]].set_index("seed")[value_column]
    out = frame.copy()
    out[output_column] = out[value_column] - out["seed"].map(baseline)
    if out[output_column].isna().any():
        raise ValueError(f"Missing seed baseline while computing {output_column}")
    return out


def build_point_dose() -> tuple[pd.DataFrame, dict[str, float | int]]:
    raw = pd.read_csv(POINT_DOSE_RAW)
    required = {"seed", "condition_id", "condition_class", "gate_strength", "lateral_code_index"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Point-dose input is missing columns: {sorted(missing)}")
    main = raw[raw["condition_class"].isin(["symmetrized", "mirror_reversed", "real_lateralized"])].copy()
    main = add_delta_from_seed_baseline(
        main,
        baseline_condition="symmetrized",
        value_column="lateral_code_index",
        output_column="delta_lateral_code_vs_sym",
    )
    rows: list[dict[str, object]] = []
    for index, (strength, group) in enumerate(main.groupby("gate_strength", sort=True)):
        values = group["delta_lateral_code_vs_sym"].to_numpy(float)
        lo, hi = bootstrap_mean_ci(values, seed=101 + index)
        rows.append(
            {
                "gate_strength": float(strength),
                "condition_class": str(group["condition_class"].iloc[0]),
                "n_seed_panels": int(group["seed"].nunique()),
                "mean_delta_lateral_code_vs_sym": float(values.mean()),
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
                "n_positive": int((values > 0).sum()),
                "n_negative": int((values < 0).sum()),
            }
        )
    summary = pd.DataFrame.from_records(rows).sort_values("gate_strength")

    slopes: list[float] = []
    r_squared: list[float] = []
    monotonic: list[bool] = []
    for _seed, group in main.groupby("seed"):
        group = group.sort_values("gate_strength")
        x = group["gate_strength"].to_numpy(float)
        y = group["delta_lateral_code_vs_sym"].to_numpy(float)
        slope, intercept = np.polyfit(x, y, 1)
        fitted = intercept + slope * x
        denominator = float(np.square(y - y.mean()).sum())
        slopes.append(float(slope))
        r_squared.append(float(1.0 - np.square(y - fitted).sum() / denominator) if denominator else 1.0)
        monotonic.append(bool(np.all(np.diff(y) > 0)))
    stats: dict[str, float | int] = {
        "n_seed_panels": int(len(slopes)),
        "mean_seed_slope": float(np.mean(slopes)),
        "min_seed_slope": float(np.min(slopes)),
        "max_seed_slope": float(np.max(slopes)),
        "n_positive_seed_slopes": int(np.sum(np.asarray(slopes) > 0)),
        "n_strictly_monotonic_seed_curves": int(np.sum(monotonic)),
        "mean_seed_linear_r_squared": float(np.mean(r_squared)),
        "all_positive_slope_sign_test_two_sided_p": two_sided_all_same_sign_p(len(slopes)),
    }
    return summary, stats


def build_point_interventions() -> pd.DataFrame:
    raw = pd.read_csv(POINT_MECHANISM_RAW)
    raw = add_delta_from_seed_baseline(
        raw,
        baseline_condition="symmetrized",
        value_column="lateral_code_index",
        output_column="delta_lateral_code_vs_sym",
    )
    labels = {
        "mirror_reversed": "mirror reversed",
        "real_lateralized": "real gate",
        "serotonin_only": "5-HT component",
        "glutamate_only": "Glu component",
        "real_no_gamma_kc": "without gamma KC",
        "real_no_ab_kc": "without alpha/beta KC",
        "real_no_apbp_kc": "without alpha-prime/beta-prime KC",
        "left_only_gate": "left half",
        "right_only_gate": "right half",
    }
    rows: list[dict[str, object]] = []
    for index, (condition_id, label) in enumerate(labels.items()):
        group = raw[raw["condition_id"].eq(condition_id)]
        if group.empty:
            raise ValueError(f"Missing point intervention condition: {condition_id}")
        values = group["delta_lateral_code_vs_sym"].to_numpy(float)
        lo, hi = bootstrap_mean_ci(values, seed=301 + index)
        rows.append(
            {
                "condition_order": index,
                "condition_id": condition_id,
                "display_label": label,
                "condition_class": str(group["condition_class"].iloc[0]),
                "n_seed_panels": int(group["seed"].nunique()),
                "mean_delta_lateral_code_vs_sym": float(values.mean()),
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
                "n_positive": int((values > 0).sum()),
                "n_negative": int((values < 0).sum()),
                "all_same_direction_sign_test_two_sided_p": observed_all_same_sign_p(values),
            }
        )
    return pd.DataFrame.from_records(rows)


def summarize_arbor_cohort(
    directory: Path,
    *,
    cohort: str,
    selection_rule: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    wide = pd.read_csv(directory / "combo_case_contrasts_wide.csv")
    raw = pd.read_csv(directory / "combo_case_raw_model_comparison.csv")
    case = wide.copy()
    case.insert(0, "cohort", cohort)
    case.insert(1, "selection_rule", selection_rule)

    arbor = raw[raw["model"].eq("arbor")].copy()
    dose = (
        arbor[
            arbor["condition_class"].isin(
                ["symmetrized", "mirror_reversed", "real_lateralized", "shuffled_lateralized"]
            )
        ]
        .groupby(["condition_class", "gate_strength"], as_index=False)
        .agg(
            n_case_conditions=("case_id", "size"),
            n_cases=("case_id", "nunique"),
            mean_delta_lateral_code_vs_sym=("lateral_code_index_delta_vs_symmetrized", "mean"),
            sd_delta_lateral_code_vs_sym=("lateral_code_index_delta_vs_symmetrized", "std"),
        )
    )
    dose.insert(0, "cohort", cohort)

    pivot = arbor.pivot_table(
        index="case_id",
        columns="condition_id",
        values="lateral_code_index_delta_vs_symmetrized",
        aggfunc="mean",
    )
    mirror_col = "mirror_reversed_strength_m1p00"
    real1_col = "real_lateralized_strength_p1p00"
    real3_col = "real_lateralized_strength_p3p00"
    exact_orientation = (pivot[mirror_col] < 0) & (pivot[real1_col] > 0) & (pivot[real3_col] > pivot[real1_col])

    arbor_delta = wide["lateral_code_index_real3_delta_vs_sym_arbor"].to_numpy(float)
    baseline_delta = wide["lateral_code_index_real3_delta_vs_sym_baseline"].to_numpy(float)
    pearson = float(pd.Series(arbor_delta).corr(pd.Series(baseline_delta)))
    spearman = float(pd.Series(arbor_delta).rank().corr(pd.Series(baseline_delta).rank()))
    stats: dict[str, object] = {
        "cohort": cohort,
        "selection_rule": selection_rule,
        "n_cases": int(len(wide)),
        "real3_mean_delta_lateral_code_vs_sym_arbor": float(arbor_delta.mean()),
        "real3_positive_cases": int((arbor_delta > 0).sum()),
        "real3_minus_mirror_positive_cases": int((wide["lateral_code_index_real3_minus_mirror_arbor"] > 0).sum()),
        "real3_minus_shuffle_positive_cases": int((wide["lateral_code_index_real3_minus_shuffle_arbor"] > 0).sum()),
        "mirror_negative_real1_positive_real3_gt_real1_cases": int(exact_orientation.sum()),
        "spike_count_decrease_cases": int((wide["mean_spike_count_real3_delta_vs_sym_arbor"] < 0).sum()),
        "effective_dimension_increase_cases": int((wide["normalized_effective_dimension_real3_delta_vs_sym_arbor"] > 0).sum()),
        "point_arbor_pearson_r": pearson,
        "point_arbor_spearman_rho": spearman,
    }
    return pd.concat([case], ignore_index=True), {"stats": stats, "dose": dose}


def build_behavior_interventions() -> pd.DataFrame:
    raw = pd.read_csv(BEHAVIOR_RAW)
    raw = add_delta_from_seed_baseline(
        raw,
        baseline_condition="delayed_conflict_wt",
        value_column="choice_index",
        output_column="delta_choice_index_vs_delayed_conflict_wt",
    )
    conditions = [
        ("bilateral_dpm_25pct_delayed_conflict", "bilateral DPM 25%", True),
        ("ppl1_25pct_delayed_conflict", "PPL1-DAN 25%", True),
        ("ppl1_dpm_25pct_delayed_conflict", "PPL1-DAN + DPM 25%", True),
        ("left_dpm_25pct_delayed_conflict", "left DPM 25%", False),
        ("right_dpm_25pct_delayed_conflict", "right DPM 25%", False),
        ("symmetrized_delayed_conflict", "symmetrized gate", False),
        ("mirror_reversed_delayed_conflict", "mirror gate", False),
    ]
    rows: list[dict[str, object]] = []
    for index, (condition_id, label, main_display) in enumerate(conditions):
        group = raw[raw["condition_id"].eq(condition_id)]
        if group.empty:
            raise ValueError(f"Missing behavior intervention condition: {condition_id}")
        values = group["delta_choice_index_vs_delayed_conflict_wt"].to_numpy(float)
        lo, hi = bootstrap_mean_ci(values, seed=701 + index)
        rows.append(
            {
                "condition_order": index,
                "condition_id": condition_id,
                "display_label": label,
                "main_display": bool(main_display),
                "n_seed_panels": int(group["seed"].nunique()),
                "mean_delta_choice_index": float(values.mean()),
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
                "n_negative": int((values < 0).sum()),
                "n_positive": int((values > 0).sum()),
                "all_same_direction_sign_test_two_sided_p": observed_all_same_sign_p(values),
                "boundary": "paired in-silico seed panels; model choice-index surrogate, not animal behavior",
            }
        )
    return pd.DataFrame.from_records(rows)


def build_evidence_ledger(
    point_dose_stats: dict[str, float | int],
    point_interventions: pd.DataFrame,
    arbor_stats: list[dict[str, object]],
    behavior: pd.DataFrame,
    steering: pd.DataFrame,
    steering_mediation: pd.DataFrame,
) -> pd.DataFrame:
    p = point_interventions.set_index("condition_id")
    support, orthogonal = arbor_stats
    behavior_index = behavior.set_index("condition_id")
    steering_index = steering[steering["split"].eq("validation")].set_index("contrast")
    mediation_index = steering_mediation.set_index("silenced_group")
    rows = [
        {
            "evidence_id": "point_orientation",
            "intervention": "symmetrize / real / mirror gate",
            "result": f"real {int(p.loc['real_lateralized', 'n_positive'])}/12 positive; mirror {int(p.loc['mirror_reversed', 'n_negative'])}/12 negative",
            "supports": "gate orientation controls signed lateral redistribution in the point model",
            "boundary": "model-seed consistency, not biological replication",
        },
        {
            "evidence_id": "point_dose",
            "intervention": "gate strength -1 to +3",
            "result": f"{int(point_dose_stats['n_strictly_monotonic_seed_curves'])}/{int(point_dose_stats['n_seed_panels'])} strictly monotonic; mean R2={float(point_dose_stats['mean_seed_linear_r_squared']):.5f}",
            "supports": "graded dose-response rather than a threshold artifact",
            "boundary": "gate amplitude is a model parameter",
        },
        {
            "evidence_id": "point_components",
            "intervention": "5-HT-only / Glu-only / subtype and side ablations",
            "result": "both transmitter components and both sides retain signed effects in 12/12 seed panels",
            "supports": "distributed, multi-component gate rather than one isolated model feature",
            "boundary": "components are rescaled counterfactual gates and are not additive biochemical effects",
        },
        {
            "evidence_id": "arbor_support_selected",
            "intervention": "symmetrize / mirror / real +1 / real +3 / shuffle",
            "result": f"{support['mirror_negative_real1_positive_real3_gt_real1_cases']}/{support['n_cases']} orientation-and-dose consistent",
            "supports": "same intervention survives morphology, placed conductances, threshold spikes and dynamic APL",
            "boundary": "panels partly selected for lateral support",
        },
        {
            "evidence_id": "arbor_orthogonal",
            "intervention": "same gate interventions on panels selected by effective dimension",
            "result": f"{orthogonal['mirror_negative_real1_positive_real3_gt_real1_cases']}/{orthogonal['n_cases']} orientation-and-dose consistent",
            "supports": "lateral result generalizes to an independently selected odor-panel cohort",
            "boundary": "deterministic model cases, not animals",
        },
        {
            "evidence_id": "cross_backend",
            "intervention": "matched point and Arbor runs on identical odor panels",
            "result": f"Spearman rho={support['point_arbor_spearman_rho']:.3f} (72-panel cohort), {orthogonal['point_arbor_spearman_rho']:.3f} (15-panel cohort)",
            "supports": "effect ranking is not specific to one neural backend",
            "boundary": "both backends share the same connectome-derived gate",
        },
        {
            "evidence_id": "mechanism_boundary",
            "intervention": "real +3 versus symmetrized in Arbor",
            "result": f"lateral code {support['real3_positive_cases']}/{support['n_cases']}; spike decrease {support['spike_count_decrease_cases']}/{support['n_cases']}; dimension increase {support['effective_dimension_increase_cases']}/{support['n_cases']}",
            "supports": "selective redistribution rather than global excitation",
            "boundary": "does not support a general dimensionality gain",
        },
        {
            "evidence_id": "dna02_steering",
            "intervention": "real / symmetrized / mirror KC chemical gate on the Shiu v783 signed graph",
            "result": (
                f"validation real {int(steering_index.loc['real_minus_symmetrized', 'positive_odors'])}/"
                f"{int(steering_index.loc['real_minus_symmetrized', 'n_odors'])} rightward; mirror "
                f"{int(steering_index.loc['mirror_minus_symmetrized', 'negative_odors'])}/"
                f"{int(steering_index.loc['mirror_minus_symmetrized', 'n_odors'])} leftward"
            ),
            "supports": "registered gate predicts a relative shift in a literature-anchored adult steering command",
            "boundary": "signed DNa02 graph drive is not measured firing, angular velocity or animal turning",
        },
        {
            "evidence_id": "mbon32_mediation",
            "intervention": "all-MBON or MBON32-pair model silencing",
            "result": (
                f"all MBON retained {100 * float(mediation_index.loc['all_MBON', 'retained_fraction']):.2f}%; "
                f"MBON32 retained {100 * float(mediation_index.loc['MBON32', 'retained_fraction']):.1f}% and reversed sign"
            ),
            "supports": "the DNa02 steering shift is mediated by the MBON layer and depends on MBON32",
            "boundary": "model-internal mediation on one signed connectome",
        },
        {
            "evidence_id": "memory_axis_interventions",
            "intervention": "bilateral DPM, PPL1-DAN and joint gain reduction",
            "result": "; ".join(
                f"{behavior_index.loc[c, 'display_label']} {int(behavior_index.loc[c, 'n_negative'])}/20 negative"
                for c in [
                    "bilateral_dpm_25pct_delayed_conflict",
                    "ppl1_25pct_delayed_conflict",
                    "ppl1_dpm_25pct_delayed_conflict",
                ]
            ),
            "supports": "model memory surrogate responds coherently to teaching and persistence interventions",
            "boundary": "does not show that the lateral gate improves animal memory",
        },
    ]
    return pd.DataFrame.from_records(rows)


def main() -> None:
    point_dose, point_dose_stats = build_point_dose()
    point_interventions = build_point_interventions()
    support_cases, support_bundle = summarize_arbor_cohort(
        ARBOR_SUPPORT_DIR,
        cohort="lateral_support_selected_72",
        selection_rule="partly selected by single-glomerulus lateral-code support",
    )
    orthogonal_cases, orthogonal_bundle = summarize_arbor_cohort(
        ARBOR_ORTHOGONAL_DIR,
        cohort="effective_dimension_selected_15",
        selection_rule="selected by effective-dimension criterion, independent of lateral-code direction",
    )
    arbor_cases = pd.concat([support_cases, orthogonal_cases], ignore_index=True)
    arbor_dose = pd.concat([support_bundle["dose"], orthogonal_bundle["dose"]], ignore_index=True)
    arbor_stats = [support_bundle["stats"], orthogonal_bundle["stats"]]
    behavior = build_behavior_interventions()
    steering = pd.read_csv(STEERING_DIR / "lateralized_steering_contrast_summary.csv")
    steering_mediation = pd.read_csv(STEERING_DIR / "lateralized_steering_mediation.csv")
    steering_null = pd.read_csv(STEERING_DIR / "lateralized_steering_subtype_null.csv")
    steering_mbon32 = pd.read_csv(STEERING_DIR / "lateralized_steering_mbon32_response.csv")
    steering_edges = pd.read_csv(STEERING_DIR / "mbon32_to_dna02_direct_edges.csv")
    steering_steps = pd.read_csv(STEERING_DIR / "lateralized_steering_step_sensitivity.csv")
    steering_propagation = pd.read_csv(
        STEERING_DIR / "lateralized_steering_propagation_sensitivity.csv"
    )
    steering_associative = pd.read_csv(STEERING_DIR / "lateralized_steering_associative_contrast_summary.csv")
    ledger = build_evidence_ledger(
        point_dose_stats,
        point_interventions,
        arbor_stats,
        behavior,
        steering,
        steering_mediation,
    )

    paths = {
        "point_dose": SOURCE_DATA / "model_causal_point_dose_response.csv",
        "point_interventions": SOURCE_DATA / "model_causal_point_interventions.csv",
        "arbor_cases": SOURCE_DATA / "model_causal_arbor_case_concordance.csv",
        "arbor_dose": SOURCE_DATA / "model_causal_arbor_dose_response.csv",
        "arbor_cohorts": SOURCE_DATA / "model_causal_arbor_cohort_summary.csv",
        "behavior": SOURCE_DATA / "model_causal_behavior_interventions.csv",
        "steering": SOURCE_DATA / "lateralized_steering_contrast_summary.csv",
        "steering_mediation": SOURCE_DATA / "lateralized_steering_mediation.csv",
        "steering_null": SOURCE_DATA / "lateralized_steering_subtype_null.csv",
        "steering_mbon32": SOURCE_DATA / "lateralized_steering_mbon32_response.csv",
        "steering_edges": SOURCE_DATA / "mbon32_to_dna02_direct_edges.csv",
        "steering_steps": SOURCE_DATA / "lateralized_steering_step_sensitivity.csv",
        "steering_propagation": SOURCE_DATA / "lateralized_steering_propagation_sensitivity.csv",
        "steering_associative": SOURCE_DATA / "lateralized_steering_associative_contrast_summary.csv",
        "ledger": SOURCE_DATA / "model_causal_evidence_ledger.csv",
        "statistics": SOURCE_DATA / "model_causal_triangulation_statistics.json",
    }
    point_dose.to_csv(paths["point_dose"], index=False)
    point_interventions.to_csv(paths["point_interventions"], index=False)
    arbor_cases.to_csv(paths["arbor_cases"], index=False)
    arbor_dose.to_csv(paths["arbor_dose"], index=False)
    pd.DataFrame.from_records(arbor_stats).to_csv(paths["arbor_cohorts"], index=False)
    behavior.to_csv(paths["behavior"], index=False)
    steering.to_csv(paths["steering"], index=False)
    steering_mediation.to_csv(paths["steering_mediation"], index=False)
    steering_null.to_csv(paths["steering_null"], index=False)
    steering_mbon32.to_csv(paths["steering_mbon32"], index=False)
    steering_edges.to_csv(paths["steering_edges"], index=False)
    steering_steps.to_csv(paths["steering_steps"], index=False)
    steering_propagation.to_csv(paths["steering_propagation"], index=False)
    steering_associative.to_csv(paths["steering_associative"], index=False)
    ledger.to_csv(paths["ledger"], index=False)
    paths["statistics"].write_text(
        json.dumps(
            {
                "point_dose": point_dose_stats,
                "arbor_cohorts": arbor_stats,
                "boundaries": {
                    "causal_language": "intervention-causal within specified models only",
                    "biological_causality": "not established by simulation",
                    "primary_model_endpoint": "signed lateral activity redistribution and relative DNa02 steering-command shift",
                    "negative_boundary": "no measured animal turning, general effective-dimension or animal-memory gain",
                },
                "source_files": {
                    "point_dose_raw": str(POINT_DOSE_RAW.relative_to(ROOT)),
                    "point_mechanism_raw": str(POINT_MECHANISM_RAW.relative_to(ROOT)),
                    "arbor_support_directory": str(ARBOR_SUPPORT_DIR.relative_to(ROOT)),
                    "arbor_orthogonal_directory": str(ARBOR_ORTHOGONAL_DIR.relative_to(ROOT)),
                    "behavior_raw": str(BEHAVIOR_RAW.relative_to(ROOT)),
                    "steering_directory": str(STEERING_DIR.relative_to(ROOT)),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for key, path in paths.items():
        print(f"{key}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
