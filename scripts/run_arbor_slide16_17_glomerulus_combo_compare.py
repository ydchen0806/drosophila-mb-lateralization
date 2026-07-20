#!/usr/bin/env python
"""Run selected glomerulus-combination cases in Arbor and graph baseline.

The case selector uses the completed target-only single-glomerulus screen to
prioritize glomeruli that strongly increase Arbor lateral-code directionality.
Each generated odor panel is then run through:

1. Arbor multicompartment KC spike simulation.
2. The PPT-style graph/top-k baseline on the exact same odor panel.

The glomerulus set is deliberately selected per case, but the within-odor
mixture weights follow the original PPT/baseline generator:
Dirichlet(1, ..., 1) over the selected glomeruli.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from bio_fly.kc_flywire_ratio_experiment import (
    LateralizationRepresentationMemoryConfig,
    _annotate_connectome_groups,
    _build_flywire_glomerulus_kc_matrix_from_frames,
    _conditioned_gate_vectors,
    _load_annotations,
    _load_kc_lateralization_gate,
    _mean_binary_cosine,
    _mean_jaccard,
    _normalize_rows,
    _representation_space_metrics,
    _sparsify,
)
from bio_fly.paths import DEFAULT_CONNECTIVITY_PATH, DEFAULT_OUTPUT_ROOT, PROJECT_ROOT, REPO_ROOT


DEFAULT_SINGLE_SCREEN = (
    DEFAULT_OUTPUT_ROOT
    / "arbor_slide16_17_glomerulus_target_only_gate025"
    / "target_only_interim_direction_scores.csv"
)
DEFAULT_CONTEXT_SCREEN = (
    DEFAULT_OUTPUT_ROOT
    / "arbor_slide16_17_glomerulus_all_screen_gate025"
    / "glomerulus_repeat_stability.csv"
)
DEFAULT_NORM_EFF_DIM_GLOMERULUS_SCORES = (
    DEFAULT_OUTPUT_ROOT
    / "arbor_slide16_17_glomerulus_all_screen_gate025"
    / "norm_eff_dim_glomerulus_direct_support_ranking.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "arbor_slide16_17_glomerulus_combo_compare_gate025",
    )
    parser.add_argument("--single-screen-csv", type=Path, default=DEFAULT_SINGLE_SCREEN)
    parser.add_argument("--context-screen-csv", type=Path, default=DEFAULT_CONTEXT_SCREEN)
    parser.add_argument(
        "--selection-mode",
        choices=["lateral_code", "norm_eff_dim"],
        default="lateral_code",
        help="Which prior screen to use when selecting glomerulus combinations.",
    )
    parser.add_argument("--norm-eff-dim-glomerulus-scores-csv", type=Path, default=DEFAULT_NORM_EFF_DIM_GLOMERULUS_SCORES)
    parser.add_argument("--n-cases", type=int, default=72)
    parser.add_argument("--n-odors", type=int, default=24)
    parser.add_argument("--shuffle-repeats", type=int, default=4)
    parser.add_argument("--gate-strengths", type=float, nargs="+", default=[-1.0, 1.0, 3.0])
    parser.add_argument("--gate-amplitude", type=float, default=0.25)
    parser.add_argument("--ratio", type=float, default=0.10)
    parser.add_argument(
        "--kc-nt-inputs-path",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "kc_nt_lateralization" / "kc_neuron_nt_inputs.parquet",
    )
    parser.add_argument("--case-workers", type=int, default=3)
    parser.add_argument("--threads", default="8")
    parser.add_argument("--condition-workers", type=int, default=8)
    parser.add_argument("--duration-ms", type=float, default=200.0)
    parser.add_argument("--dt-ms", type=float, default=0.1)
    parser.add_argument("--global-conductance-scale", type=float, default=1.0)
    parser.add_argument("--alpn-to-kc-conductance-scale", type=float, default=5.0)
    parser.add_argument("--connectome-weight-mode", choices=["raw", "attenuated"], default="raw")
    parser.add_argument("--alpn-nt-mode", choices=["ach_only", "all"], default="ach_only")
    parser.add_argument("--alpn-weight-normalization", choices=["glomerulus", "none"], default="glomerulus")
    parser.add_argument("--input-event-interval-ms", type=float, default=5.0)
    parser.add_argument("--kc-detector-threshold-mv", type=float, default=-55.0)
    parser.add_argument("--vm-mv", type=float, default=-65.0)
    parser.add_argument("--max-kc", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    return parser.parse_args()


def _clean_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")


def _family(glomerulus: str) -> str:
    match = re.match(r"[A-Za-z]+", str(glomerulus))
    return match.group(0) if match else str(glomerulus)


def _load_single_scores(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["repeat"].eq(0)].copy()
    frame["direction_score"] = frame["lateral_code_index_real3_delta_vs_sym"].astype(float)
    frame["spike_delta"] = frame["mean_spike_count_real3_delta_vs_sym"].astype(float)
    frame["active_delta"] = frame["mean_active_fraction_real3_delta_vs_sym"].astype(float)
    return frame.sort_values("direction_score", ascending=False).reset_index(drop=True)


def _diverse_pick(pool: list[str], count: int, offset: int = 0) -> list[str]:
    selected: list[str] = []
    family_counts: dict[str, int] = {}
    candidates = pool[offset:] + pool[:offset]
    for glomerulus in candidates:
        family = _family(glomerulus)
        if family_counts.get(family, 0) > 0 and len(selected) < min(count, 4):
            continue
        selected.append(glomerulus)
        family_counts[family] = family_counts.get(family, 0) + 1
        if len(selected) == count:
            return selected
    for glomerulus in candidates:
        if glomerulus not in selected:
            selected.append(glomerulus)
            if len(selected) == count:
                return selected
    return selected


def _add_case(
    rows: list[dict[str, object]],
    seen: set[tuple[str, ...]],
    *,
    case_class: str,
    glomeruli: list[str],
    rationale: str,
) -> None:
    glomeruli = [str(item) for item in glomeruli]
    key = tuple(sorted(glomeruli))
    if len(key) < 2 or key in seen:
        return
    seen.add(key)
    rows.append(
        {
            "case_index": len(rows),
            "case_id": f"{case_class}_{len(rows):03d}_{'_'.join(_clean_token(g) for g in glomeruli[:4])}",
            "case_class": case_class,
            "glomeruli": ";".join(glomeruli),
            "n_glomeruli": len(glomeruli),
            "rationale": rationale,
        }
    )


def select_cases(args: argparse.Namespace) -> pd.DataFrame:
    if str(args.selection_mode) == "norm_eff_dim":
        return select_norm_eff_dim_cases(args)

    scores = _load_single_scores(args.single_screen_csv)
    ordered = list(scores["target_glomerulus"].astype(str))
    strong = ordered[:18]
    mid = ordered[18:36]
    weak = ordered[-12:]
    neutral = ordered[36:46]
    score_by_glomerulus = scores.set_index("target_glomerulus")["direction_score"].to_dict()

    rows: list[dict[str, object]] = []
    seen: set[tuple[str, ...]] = set()

    for index in range(24):
        combo = _diverse_pick(strong, 2, offset=index)
        _add_case(
            rows,
            seen,
            case_class="support_pair",
            glomeruli=combo,
            rationale="two high target-only lateral-code glomeruli",
        )

    for index in range(18):
        combo = _diverse_pick(strong + mid, 3, offset=index * 2)
        _add_case(
            rows,
            seen,
            case_class="support_triple",
            glomeruli=combo,
            rationale="three high/mid target-only lateral-code glomeruli",
        )

    for index in range(10):
        combo = _diverse_pick(strong + mid, 4, offset=index * 3)
        _add_case(
            rows,
            seen,
            case_class="support_quad",
            glomeruli=combo,
            rationale="diverse four-glomerulus support pool",
        )

    for index in range(8):
        combo = _diverse_pick(strong, 2, offset=index * 2) + _diverse_pick(neutral, 1, offset=index)
        _add_case(
            rows,
            seen,
            case_class="support_with_neutral",
            glomeruli=combo,
            rationale="support glomeruli with one neutral/context-control glomerulus",
        )

    for index in range(6):
        combo = _diverse_pick(neutral, 3, offset=index)
        _add_case(
            rows,
            seen,
            case_class="neutral_control",
            glomeruli=combo,
            rationale="middle/low lateral-code control pool",
        )

    for index in range(6):
        combo = _diverse_pick(weak, 3, offset=index)
        _add_case(
            rows,
            seen,
            case_class="weak_control",
            glomeruli=combo,
            rationale="lowest target-only lateral-code control pool",
        )

    rng = np.random.default_rng(20260628)
    support_pool = strong + mid
    attempts = 0
    while len(rows) < int(args.n_cases) and attempts < 1000:
        attempts += 1
        size = int(rng.choice([3, 4], p=[0.65, 0.35]))
        combo = list(rng.choice(support_pool, size=size, replace=False))
        _add_case(
            rows,
            seen,
            case_class="support_extra",
            glomeruli=combo,
            rationale="additional support-biased diverse combination to reach runtime budget",
        )

    case_df = pd.DataFrame.from_records(rows).head(int(args.n_cases)).copy()
    case_df["mean_single_direction_score"] = case_df["glomeruli"].map(
        lambda value: float(np.mean([score_by_glomerulus.get(item, np.nan) for item in str(value).split(";")]))
    )
    return case_df


def _load_norm_eff_dim_scores(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).copy()
    if "target_glomerulus" in frame.columns:
        frame = frame.rename(
            columns={
                "target_glomerulus": "glomerulus",
                "real3_delta_norm_eff_dim_mean": "real3_mean",
            }
        )
    required = {"glomerulus", "real3_mean", "support_score_mean", "support_class"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Norm-effective-dimension glomerulus score table is missing columns: {sorted(missing)}")
    frame["selection_score"] = frame["support_score_mean"].astype(float)
    return frame.sort_values(["selection_score", "real3_mean"], ascending=False).reset_index(drop=True)


def select_norm_eff_dim_cases(args: argparse.Namespace) -> pd.DataFrame:
    scores = _load_norm_eff_dim_scores(args.norm_eff_dim_glomerulus_scores_csv)
    strong = scores[scores["support_class"].eq("strong_support")]["glomerulus"].astype(str).tolist()
    if len(strong) < 8:
        strong = scores[scores["real3_mean"].astype(float).gt(0)]["glomerulus"].astype(str).tolist()
    negative = (
        scores[scores["support_class"].eq("negative")]
        .sort_values("real3_mean", ascending=True)["glomerulus"]
        .astype(str)
        .tolist()
    )
    unstable = scores[scores["support_class"].eq("unstable")]["glomerulus"].astype(str).tolist()
    score_by_glomerulus = scores.set_index("glomerulus")["real3_mean"].astype(float).to_dict()

    rows: list[dict[str, object]] = []
    seen: set[tuple[str, ...]] = set()

    for index in range(4):
        combo = _diverse_pick(strong, 2, offset=index)
        _add_case(
            rows,
            seen,
            case_class="norm_eff_pair",
            glomeruli=combo,
            rationale="two glomeruli with positive combo-derived normalized-effective-dimension contribution",
        )

    for index in range(4):
        combo = _diverse_pick(strong, 3, offset=index * 2)
        _add_case(
            rows,
            seen,
            case_class="norm_eff_triple",
            glomeruli=combo,
            rationale="three stable positive normalized-effective-dimension glomeruli",
        )

    for index in range(2):
        combo = _diverse_pick(strong, 4, offset=index * 3)
        _add_case(
            rows,
            seen,
            case_class="norm_eff_quad",
            glomeruli=combo,
            rationale="four stable positive normalized-effective-dimension glomeruli",
        )

    for index in range(2):
        combo = _diverse_pick(strong, 2, offset=index) + _diverse_pick(unstable + negative, 1, offset=index)
        _add_case(
            rows,
            seen,
            case_class="norm_eff_context_control",
            glomeruli=combo,
            rationale="positive normalized-effective-dimension glomeruli with one negative/context glomerulus",
        )

    for index in range(3):
        combo = _diverse_pick(negative, 3, offset=index)
        _add_case(
            rows,
            seen,
            case_class="norm_eff_negative_control",
            glomeruli=combo,
            rationale="negative normalized-effective-dimension control glomeruli",
        )

    case_df = pd.DataFrame.from_records(rows).head(int(args.n_cases)).copy()
    case_df["mean_single_direction_score"] = case_df["glomeruli"].map(
        lambda value: float(np.mean([score_by_glomerulus.get(item, np.nan) for item in str(value).split(";")]))
    )
    case_df["selection_metric"] = "normalized_effective_dimension_delta_vs_symmetrized"
    return case_df


def _panel_for_case(row: pd.Series, *, n_odors: int, output_dir: Path) -> Path:
    glomeruli = [item for item in str(row["glomeruli"]).split(";") if item]
    rng = np.random.default_rng(20260628 + int(row["case_index"]) * 9973)
    records: list[dict[str, object]] = []
    for odor_index in range(int(n_odors)):
        weights = rng.dirichlet(np.ones(len(glomeruli), dtype=np.float64))
        records.append(
            {
                "odor_identity": f"{row['case_id']}_odor{odor_index + 1:02d}",
                "seed": -1,
                "n_glomeruli": len(glomeruli),
                "glomeruli": ";".join(glomeruli),
                "glomerulus_weights": ";".join(f"{float(weight):.6f}" for weight in weights),
                "case_id": str(row["case_id"]),
                "case_class": str(row["case_class"]),
            }
        )
    panel_dir = output_dir / "odor_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    path = panel_dir / f"{row['case_id']}.csv"
    pd.DataFrame.from_records(records).to_csv(path, index=False)
    return path


def generate_case_panels(args: argparse.Namespace, case_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in case_df.iterrows():
        panel_path = _panel_for_case(row, n_odors=int(args.n_odors), output_dir=args.output_dir)
        item = row.to_dict()
        item["odor_panel_csv"] = str(panel_path)
        rows.append(item)
    manifest = pd.DataFrame.from_records(rows)
    manifest.to_csv(args.output_dir / "combo_case_manifest.csv", index=False)
    return manifest


def _custom_activity(
    odor_panel: pd.DataFrame,
    glomerulus_names: list[str],
    glomerulus_matrix: np.ndarray,
) -> np.ndarray:
    index = {str(name): i for i, name in enumerate(glomerulus_names)}
    rows: list[np.ndarray] = []
    for row in odor_panel.itertuples(index=False):
        selected = [item for item in str(row.glomeruli).split(";") if item]
        weights = [float(item) for item in str(row.glomerulus_weights).split(";") if item]
        activity = np.zeros(glomerulus_matrix.shape[1], dtype=np.float64)
        for glomerulus, weight in zip(selected, weights, strict=False):
            glom_index = index.get(str(glomerulus))
            if glom_index is not None:
                activity += float(weight) * glomerulus_matrix[int(glom_index)]
        rows.append(activity)
    return _normalize_rows(np.vstack(rows))


def _baseline_inputs(args: argparse.Namespace) -> dict[str, object]:
    cfg = LateralizationRepresentationMemoryConfig(
        kc_nt_inputs_path=args.kc_nt_inputs_path,
        gate_amplitude=float(args.gate_amplitude),
        gate_strengths=tuple(float(value) for value in args.gate_strengths),
        shuffle_repeats=int(args.shuffle_repeats),
        ratio=float(args.ratio),
    )
    annotations = _load_annotations(cfg.annotation_path)
    annotated = _annotate_connectome_groups(annotations)
    edges = pd.read_parquet(
        DEFAULT_CONNECTIVITY_PATH,
        columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity", "Excitatory x Connectivity"],
    )
    glomerulus_names, glomerulus_matrix, kc_ids, channel_table = _build_flywire_glomerulus_kc_matrix_from_frames(
        annotated,
        edges,
    )
    gate_by_subtype_side, gate, side_by_kc = _load_kc_lateralization_gate(cfg.kc_nt_inputs_path, kc_ids)
    rng = np.random.default_rng(20260603)
    conditions = _conditioned_gate_vectors(gate, cfg=cfg, rng=rng)
    return {
        "cfg": cfg,
        "glomerulus_names": glomerulus_names,
        "glomerulus_matrix": glomerulus_matrix,
        "kc_ids": kc_ids,
        "channel_table": channel_table,
        "gate": gate,
        "side_by_kc": side_by_kc,
        "conditions": conditions,
    }


def _run_baseline_panel(args: argparse.Namespace, row: dict[str, object], inputs: dict[str, object]) -> dict[str, str]:
    case_id = str(row["case_id"])
    out_dir = args.output_dir / "panels" / case_id / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = out_dir / "baseline_lateralization_comparison.csv"
    metadata_path = out_dir / "baseline_lateralization_metadata.json"
    if bool(args.resume) and comparison_path.exists() and metadata_path.exists():
        return {"status": "reused", "output_dir": str(out_dir)}

    odor_panel = pd.read_csv(row["odor_panel_csv"])
    activity = _custom_activity(
        odor_panel,
        list(inputs["glomerulus_names"]),
        np.asarray(inputs["glomerulus_matrix"], dtype=np.float64),
    )
    raw_rows: list[dict[str, object]] = []
    for condition in inputs["conditions"]:
        gate_vector = np.asarray(condition["gate_vector"], dtype=np.float64)
        gains = np.clip(1.0 + float(args.gate_amplitude) * gate_vector, 0.05, 5.0)
        gated_activity = _normalize_rows(np.maximum(activity * gains[None, :], 0.0))
        binary, graded, active_k = _sparsify(gated_activity, float(args.ratio))
        metrics = _representation_space_metrics(binary, graded, np.asarray(inputs["side_by_kc"]))
        metrics["mean_jaccard_overlap"] = _mean_jaccard(binary)
        metrics["mean_binary_cosine"] = _mean_binary_cosine(binary)
        raw_rows.append(
            {
                "case_id": case_id,
                "condition_order": int(condition["condition_order"]),
                "condition_id": str(condition["condition_id"]).replace("_repeat_", "_"),
                "condition_label": str(condition["condition_label"]),
                "condition_class": str(condition["condition_class"]),
                "gate_strength": float(condition["gate_strength"]),
                "shuffle_repeat": int(condition["shuffle_repeat"]),
                "active_k": int(active_k),
                "mean_active_fraction": float(binary.mean()),
                "mean_spike_count": float(gated_activity.sum()),
                **metrics,
            }
        )
    comparison = pd.DataFrame.from_records(raw_rows).sort_values(["condition_order", "condition_id"])
    sym = comparison[comparison["condition_class"].eq("symmetrized")]
    if not sym.empty:
        sym_row = sym.iloc[0]
        for metric in [
            "mean_active_fraction",
            "mean_spike_count",
            "normalized_effective_dimension",
            "effective_dimension",
            "mean_pairwise_l2_distance",
            "mean_jaccard_overlap",
            "mean_binary_cosine",
            "lateral_code_index",
        ]:
            comparison[f"{metric}_delta_vs_symmetrized"] = comparison[metric] - float(sym_row[metric])
    comparison.to_csv(comparison_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "model": "ppt_graph_topk_baseline_custom_odor_panel",
                "case_id": case_id,
                "odor_panel_csv": str(row["odor_panel_csv"]),
                "ratio": float(args.ratio),
                "gate_amplitude": float(args.gate_amplitude),
                "gate_strengths": [float(value) for value in args.gate_strengths],
                "shuffle_repeats": int(args.shuffle_repeats),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"status": "completed", "output_dir": str(out_dir)}


def _run_arbor_panel(args: argparse.Namespace, row: dict[str, object]) -> dict[str, str]:
    case_id = str(row["case_id"])
    out_dir = args.output_dir / "panels" / case_id / "arbor"
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = out_dir / "arbor_lateralization_representation_comparison.csv"
    metadata_path = out_dir / "arbor_lateralization_metadata.json"
    if bool(args.resume) and comparison_path.exists() and metadata_path.exists():
        return {"status": "reused", "output_dir": str(out_dir)}
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_arbor_multicompartment_lateralization_representation.py"),
        "--output-dir",
        str(out_dir),
        "--odor-panel-csv",
        str(row["odor_panel_csv"]),
        "--threads",
        str(args.threads),
        "--condition-workers",
        str(args.condition_workers),
        "--seed",
        "0",
        "--n-odors",
        str(args.n_odors),
        "--shuffle-repeats",
        str(args.shuffle_repeats),
        "--gate-amplitude",
        str(args.gate_amplitude),
        "--gate-strengths",
        *[str(value) for value in args.gate_strengths],
        "--max-kc",
        str(args.max_kc),
        "--duration-ms",
        str(args.duration_ms),
        "--dt-ms",
        str(args.dt_ms),
        "--global-conductance-scale",
        str(args.global_conductance_scale),
        "--alpn-to-kc-conductance-scale",
        str(args.alpn_to_kc_conductance_scale),
        "--connectome-weight-mode",
        str(args.connectome_weight_mode),
        "--alpn-nt-mode",
        str(args.alpn_nt_mode),
        "--alpn-weight-normalization",
        str(args.alpn_weight_normalization),
        "--input-event-interval-ms",
        str(args.input_event_interval_ms),
        "--kc-detector-threshold-mv",
        str(args.kc_detector_threshold_mv),
        "--vm-mv",
        str(args.vm_mv),
        "--kc-nt-inputs-path",
        str(args.kc_nt_inputs_path),
    ]
    env = os.environ.copy()
    pythonpath = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = f"{pythonpath}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else pythonpath
    with (out_dir / "run.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return {"status": "completed", "output_dir": str(out_dir)}


def _run_case(args: argparse.Namespace, row: dict[str, object], baseline_inputs: dict[str, object]) -> dict[str, object]:
    start = time.perf_counter()
    baseline = _run_baseline_panel(args, row, baseline_inputs)
    arbor = _run_arbor_panel(args, row)
    return {
        "case_id": str(row["case_id"]),
        "case_class": str(row["case_class"]),
        "baseline_status": baseline["status"],
        "arbor_status": arbor["status"],
        "elapsed_s": float(time.perf_counter() - start),
    }


def _condition_key(frame: pd.DataFrame) -> pd.Series:
    key = frame["condition_class"].astype(str) + "_s" + frame["gate_strength"].astype(float).round(6).astype(str)
    key = key.where(~frame["condition_class"].eq("shuffled_lateralized"), "shuffled_lateralized")
    return key


def aggregate(args: argparse.Namespace, manifest: pd.DataFrame, case_metadata: list[dict[str, object]]) -> dict[str, Path]:
    raw_rows: list[pd.DataFrame] = []
    contrast_rows: list[dict[str, object]] = []
    for row in manifest.itertuples(index=False):
        case_id = str(row.case_id)
        arbor_path = args.output_dir / "panels" / case_id / "arbor" / "arbor_lateralization_representation_comparison.csv"
        base_path = args.output_dir / "panels" / case_id / "baseline" / "baseline_lateralization_comparison.csv"
        arbor = pd.read_csv(arbor_path)
        baseline = pd.read_csv(base_path)
        for model, frame in [("arbor", arbor), ("baseline", baseline)]:
            enriched = frame.copy().drop(columns=["model", "case_id", "case_class", "glomeruli"], errors="ignore")
            enriched.insert(0, "model", model)
            enriched.insert(0, "case_id", case_id)
            enriched.insert(1, "case_class", str(row.case_class))
            enriched.insert(2, "glomeruli", str(row.glomeruli))
            raw_rows.append(enriched)
        for model, frame in [("arbor", arbor), ("baseline", baseline)]:
            real3 = frame[frame["condition_id"].eq("real_lateralized_strength_p3p00")]
            mirror = frame[frame["condition_id"].eq("mirror_reversed_strength_m1p00")]
            shuffle = frame[frame["condition_class"].eq("shuffled_lateralized")]
            if real3.empty or mirror.empty or shuffle.empty:
                continue
            rec: dict[str, object] = {
                "case_id": case_id,
                "case_class": str(row.case_class),
                "glomeruli": str(row.glomeruli),
                "model": model,
            }
            for metric in [
                "lateral_code_index",
                "normalized_effective_dimension",
                "mean_active_fraction",
                "mean_spike_count",
                "mean_pairwise_l2_distance",
            ]:
                delta_col = f"{metric}_delta_vs_symmetrized"
                rec[f"{metric}_real3_delta_vs_sym"] = float(real3[delta_col].iloc[0]) if delta_col in real3 else np.nan
                rec[f"{metric}_real3_minus_mirror"] = (
                    float(real3[delta_col].iloc[0]) - float(mirror[delta_col].iloc[0])
                    if delta_col in real3 and delta_col in mirror
                    else np.nan
                )
                rec[f"{metric}_real3_minus_shuffle"] = (
                    float(real3[delta_col].iloc[0]) - float(shuffle[delta_col].mean())
                    if delta_col in real3 and delta_col in shuffle
                    else np.nan
                )
            contrast_rows.append(rec)
    raw = pd.concat(raw_rows, ignore_index=True, sort=False)
    contrasts = pd.DataFrame.from_records(contrast_rows)
    wide = contrasts.pivot_table(
        index=["case_id", "case_class", "glomeruli"],
        columns="model",
        values=[
            "lateral_code_index_real3_delta_vs_sym",
            "lateral_code_index_real3_minus_mirror",
            "lateral_code_index_real3_minus_shuffle",
            "normalized_effective_dimension_real3_delta_vs_sym",
            "normalized_effective_dimension_real3_minus_mirror",
            "normalized_effective_dimension_real3_minus_shuffle",
            "mean_spike_count_real3_delta_vs_sym",
        ],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{model}" for metric, model in wide.columns]
    wide = wide.reset_index()
    paths = {
        "raw": args.output_dir / "combo_case_raw_model_comparison.csv",
        "contrasts": args.output_dir / "combo_case_contrasts_long.csv",
        "wide": args.output_dir / "combo_case_contrasts_wide.csv",
        "metadata": args.output_dir / "combo_case_batch_metadata.json",
    }
    raw.to_csv(paths["raw"], index=False)
    contrasts.to_csv(paths["contrasts"], index=False)
    wide.to_csv(paths["wide"], index=False)
    paths["metadata"].write_text(
        json.dumps(
            {
                "config": vars(args),
                "n_cases": int(len(manifest)),
                "case_metadata": case_metadata,
                "paths": {key: str(value) for key, value in paths.items()},
                "interpretation": {
                    "case_selection": "support-biased glomerulus combinations selected from target-only lateral-code screen",
                    "baseline": "PPT-style graph/top-k proxy run on the same custom odor panels",
                    "arbor": "calibrated multicompartment KC spike model run on the same custom odor panels",
                },
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return paths


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    case_df = select_cases(args)
    manifest = generate_case_panels(args, case_df)
    print(f"Wrote {len(manifest)} combo odor panels to {args.output_dir / 'odor_panels'}", flush=True)
    print(manifest.groupby("case_class").size().to_string(), flush=True)
    if bool(args.generate_only):
        print(args.output_dir / "combo_case_manifest.csv")
        return

    baseline_inputs = _baseline_inputs(args)
    worker_count = max(1, min(int(args.case_workers), len(manifest)))
    metadata: list[dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_case = {
            executor.submit(_run_case, args, row._asdict(), baseline_inputs): str(row.case_id)
            for row in manifest.itertuples(index=False)
        }
        for index, future in enumerate(concurrent.futures.as_completed(future_to_case), start=1):
            case_id = future_to_case[future]
            result = future.result()
            metadata.append(result)
            print(
                f"[{index}/{len(manifest)}] case={case_id} "
                f"arbor={result.get('arbor_status')} baseline={result.get('baseline_status')} "
                f"elapsed_s={float(result.get('elapsed_s', 0.0)):.1f}",
                flush=True,
            )
    paths = aggregate(args, manifest, metadata)
    elapsed_s = time.perf_counter() - start
    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False, indent=2), flush=True)
    print(f"elapsed_s={elapsed_s:.3f}", flush=True)
    wide = pd.read_csv(paths["wide"])
    sort_col = "lateral_code_index_real3_delta_vs_sym_arbor"
    if sort_col in wide:
        print(wide.sort_values(sort_col, ascending=False).head(30).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
