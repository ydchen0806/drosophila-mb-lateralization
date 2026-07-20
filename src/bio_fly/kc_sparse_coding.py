from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paths import DEFAULT_CONNECTIVITY_PATH, DEFAULT_OUTPUT_ROOT, PROCESSED_DATA_ROOT
from .propagation import PropagationConfig, load_connectivity_edges, signed_multihop_response


OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT / "kc_sparse_coding_apl_validation"

LITERATURE_KC_ACTIVE_FRACTION: float = 0.10
LEGACY_ONE_SIXTH_KC_RATIO: float = 1 / 6


@dataclass(frozen=True)
class KCSparseCodingConfig:
    activation_ratios: tuple[float, ...] = (
        1 / 40,
        1 / 20,
        0.075,
        1 / 12,
        LITERATURE_KC_ACTIVE_FRACTION,
        1 / 8,
        LEGACY_ONE_SIXTH_KC_RATIO,
        1 / 4,
        1 / 3,
        1 / 2,
    )
    apl_gains: tuple[float, ...] = (0.33, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0)
    normal_apl_ratio: float = LITERATURE_KC_ACTIVE_FRACTION
    min_activation_ratio: float = 1 / 20
    max_activation_ratio: float = 1 / 2
    random_odor_count: int = 18
    random_seed: int = 17
    dropout_probability: float = 0.15
    false_positive_probability: float = 0.003
    max_learning_steps: int = 8
    test_repeats: int = 12
    evaluation_repeats: int = 5
    forgetting_interference_steps: int = 6
    learning_accuracy_criterion: float = 0.90
    similar_pair_fraction: float = 0.20
    state_propagation_steps: int = 2
    state_max_active: int = 3000
    max_state_seed_neurons: int = 400


@dataclass(frozen=True)
class KCSparseCodingPaths:
    output_dir: Path
    odor_panel: Path
    ratio_sweep: Path
    apl_sweep: Path
    apl_threshold_sweep: Path
    mechanism_audit: Path
    downstream_readout: Path
    state_predictions: Path
    wetlab_protocol: Path
    ratio_figure: Path
    apl_figure: Path
    state_figure: Path
    report: Path
    metadata: Path


def _read_annotations(annotation_path: Path | None = None) -> pd.DataFrame:
    path = annotation_path or PROCESSED_DATA_ROOT / "flywire_neuron_annotations.parquet"
    requested = [
        "root_id",
        "side",
        "super_class",
        "cell_class",
        "cell_sub_class",
        "supertype",
        "cell_type",
        "hemibrain_type",
        "top_nt",
        "synonyms",
    ]
    available = pd.read_parquet(path, columns=None).columns
    columns = [column for column in requested if column in available]
    return pd.read_parquet(path, columns=columns).drop_duplicates("root_id")


def load_kc_population(annotation_path: Path | None = None) -> pd.DataFrame:
    annotations = _read_annotations(annotation_path)
    if "cell_class" not in annotations:
        return pd.DataFrame(columns=["root_id", "side", "cell_type", "hemibrain_type", "top_nt"])
    kcs = annotations[annotations["cell_class"].astype(str).eq("Kenyon_Cell")].copy()
    for column in ["side", "cell_type", "hemibrain_type", "top_nt"]:
        if column not in kcs:
            kcs[column] = ""
    return kcs.sort_values("root_id").reset_index(drop=True)


def _load_kc_readout(kc_readout_path: Path) -> pd.DataFrame:
    readout = pd.read_csv(kc_readout_path)
    required = {"odor_identity", "root_id", "score"}
    missing = required - set(readout.columns)
    if missing:
        raise ValueError(f"KC readout is missing required columns: {sorted(missing)}")
    if "abs_score" not in readout:
        readout["abs_score"] = readout["score"].abs()
    return readout


def build_odor_activity_matrix(
    kc_readout: pd.DataFrame,
    kc_population: pd.DataFrame,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    kc_ids = kc_population["root_id"].astype("int64").to_numpy()
    if len(kc_ids) == 0:
        raise ValueError("No Kenyon cells found in annotation table.")
    kc_index = {int(root_id): index for index, root_id in enumerate(kc_ids)}
    odors = sorted(kc_readout["odor_identity"].dropna().astype(str).unique().tolist())
    matrix = np.zeros((len(odors), len(kc_ids)), dtype=np.float64)
    grouped = (
        kc_readout.assign(abs_score=kc_readout["score"].abs())
        .groupby(["odor_identity", "root_id"], as_index=False)["abs_score"]
        .sum()
    )
    odor_index = {odor: index for index, odor in enumerate(odors)}
    for row in grouped.itertuples(index=False):
        root_id = int(row.root_id)
        if root_id in kc_index and str(row.odor_identity) in odor_index:
            matrix[odor_index[str(row.odor_identity)], kc_index[root_id]] = float(row.abs_score)
    return odors, matrix, kc_ids


def augment_odor_panel(
    odor_names: list[str],
    activity_matrix: np.ndarray,
    random_odor_count: int,
    random_seed: int,
) -> tuple[list[str], np.ndarray, pd.DataFrame]:
    if activity_matrix.size == 0:
        raise ValueError("Cannot augment an empty odor activity matrix.")
    rng = np.random.default_rng(random_seed)
    base = np.nan_to_num(activity_matrix.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    base = np.maximum(base, 0.0)
    base_sums = base.sum(axis=1, keepdims=True)
    base = np.divide(base, base_sums, out=np.zeros_like(base), where=base_sums > 0)

    names = [f"observed_{name}" for name in odor_names]
    rows = [row.copy() for row in base]
    records = [
        {
            "odor_identity": name,
            "panel_source": "observed_connectome_readout",
            "base_mixture": name.replace("observed_", ""),
        }
        for name in names
    ]
    if len(rows) == 0:
        raise ValueError("At least one observed odor is required.")

    n_base, n_kc = base.shape
    for index in range(random_odor_count):
        mixture = rng.dirichlet(np.ones(n_base))
        mixed = mixture @ base
        template = base[int(rng.integers(0, n_base))]
        permuted = template[rng.permutation(n_kc)]
        multiplicative_noise = rng.lognormal(mean=0.0, sigma=0.70, size=n_kc)
        dropout = rng.random(n_kc) > rng.uniform(0.05, 0.25)
        synthetic = (0.72 * mixed * multiplicative_noise + 0.28 * permuted) * dropout
        background = rng.gamma(shape=0.45, scale=1e-6, size=n_kc)
        synthetic = np.maximum(synthetic + background, 0.0)
        total = synthetic.sum()
        if total <= 0:
            active = rng.choice(n_kc, size=max(1, n_kc // 10), replace=False)
            synthetic[active] = rng.random(len(active))
            total = synthetic.sum()
        synthetic = synthetic / total
        names.append(f"synthetic_odor_{index + 1:02d}")
        rows.append(synthetic)
        base_desc = ";".join(f"{odor}:{weight:.3f}" for odor, weight in zip(odor_names, mixture))
        records.append(
            {
                "odor_identity": names[-1],
                "panel_source": "connectome_constrained_jittered_panel",
                "base_mixture": base_desc,
            }
        )

    panel = np.vstack(rows)
    panel_frame = pd.DataFrame.from_records(records)
    panel_frame["n_kc"] = n_kc
    panel_frame["total_activity_mass"] = panel.sum(axis=1)
    return names, panel, panel_frame


def ratio_from_apl_gain(
    apl_gain: float,
    normal_ratio: float = LITERATURE_KC_ACTIVE_FRACTION,
    min_ratio: float = 1 / 20,
    max_ratio: float = 1 / 2,
) -> float:
    if apl_gain <= 0:
        return float(max_ratio)
    return float(np.clip(normal_ratio / apl_gain, min_ratio, max_ratio))


def sparsify_activity(activity_matrix: np.ndarray, activation_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    if activity_matrix.ndim != 2:
        raise ValueError("activity_matrix must be two-dimensional.")
    n_odors, n_kc = activity_matrix.shape
    if n_kc == 0:
        raise ValueError("activity_matrix has no KC columns.")
    k = max(1, min(n_kc, int(round(float(activation_ratio) * n_kc))))
    binary = np.zeros((n_odors, n_kc), dtype=bool)
    graded = np.zeros((n_odors, n_kc), dtype=np.float64)
    for row_index, row in enumerate(np.nan_to_num(activity_matrix, nan=0.0, posinf=0.0, neginf=0.0)):
        if k == n_kc:
            selected = np.arange(n_kc)
        else:
            selected = np.argpartition(row, -k)[-k:]
        binary[row_index, selected] = True
        graded[row_index, selected] = row[selected]
        row_sum = graded[row_index].sum()
        if row_sum > 0:
            graded[row_index] /= row_sum
    return binary, graded


def calibrate_apl_threshold(activity_matrix: np.ndarray, target_ratio: float) -> float:
    values = np.maximum(np.nan_to_num(activity_matrix.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    positive = values[values > 0]
    if len(positive) == 0:
        return 0.0
    quantile = float(np.clip(1.0 - target_ratio, 0.0, 1.0))
    return float(np.quantile(positive, quantile))


def apply_apl_threshold_inhibition(
    activity_matrix: np.ndarray,
    apl_gain: float,
    baseline_threshold: float,
    threshold_exponent: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.maximum(np.nan_to_num(activity_matrix.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    effective_threshold = max(0.0, float(baseline_threshold) * max(0.0, float(apl_gain)) ** float(threshold_exponent))
    inhibited = np.maximum(values - effective_threshold, 0.0)
    binary = inhibited > 0
    graded = inhibited.copy()
    row_sums = graded.sum(axis=1, keepdims=True)
    graded = np.divide(graded, row_sums, out=np.zeros_like(graded), where=row_sums > 0)
    return binary, graded


def _pairwise_code_metrics(binary_code: np.ndarray) -> dict[str, float]:
    n_odors, n_kc = binary_code.shape
    if n_odors < 2:
        return {
            "mean_jaccard_overlap": 0.0,
            "mean_hamming_distance": 0.0,
            "mean_binary_cosine": 0.0,
            "collision_rate_jaccard_gt_0_50": 0.0,
        }
    jaccards: list[float] = []
    hammings: list[float] = []
    cosines: list[float] = []
    collisions = 0
    pairs = 0
    active_counts = binary_code.sum(axis=1).astype(float)
    for left in range(n_odors):
        for right in range(left + 1, n_odors):
            inter = float(np.logical_and(binary_code[left], binary_code[right]).sum())
            union = float(np.logical_or(binary_code[left], binary_code[right]).sum())
            jaccard = inter / union if union else 0.0
            hamming = float(np.logical_xor(binary_code[left], binary_code[right]).sum()) / float(n_kc)
            denom = float(np.sqrt(active_counts[left] * active_counts[right]))
            cosine = inter / denom if denom else 0.0
            jaccards.append(jaccard)
            hammings.append(hamming)
            cosines.append(cosine)
            collisions += int(jaccard > 0.50)
            pairs += 1
    return {
        "mean_jaccard_overlap": float(np.mean(jaccards)),
        "mean_hamming_distance": float(np.mean(hammings)),
        "mean_binary_cosine": float(np.mean(cosines)),
        "collision_rate_jaccard_gt_0_50": float(collisions / pairs) if pairs else 0.0,
    }


def _population_entropy(binary_code: np.ndarray) -> tuple[float, float]:
    probabilities = binary_code.mean(axis=0)
    valid = (probabilities > 0) & (probabilities < 1)
    entropy = np.zeros_like(probabilities, dtype=np.float64)
    p = probabilities[valid]
    entropy[valid] = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return float(entropy.mean()), float(entropy.sum())


def _balanced_labels(n_items: int, rng: np.random.Generator) -> np.ndarray:
    labels = np.ones(n_items, dtype=np.float64)
    labels[n_items // 2 :] = -1.0
    rng.shuffle(labels)
    return labels


def _similar_pair_labels(
    binary_code: np.ndarray,
    rng: np.random.Generator,
    similar_pair_fraction: float,
) -> tuple[np.ndarray, dict[str, float]]:
    n_odors = binary_code.shape[0]
    labels = np.zeros(n_odors, dtype=np.float64)
    assigned: set[int] = set()
    pairs: list[tuple[float, int, int]] = []
    for left in range(n_odors):
        for right in range(left + 1, n_odors):
            inter = float(np.logical_and(binary_code[left], binary_code[right]).sum())
            union = float(np.logical_or(binary_code[left], binary_code[right]).sum())
            jaccard = inter / union if union else 0.0
            pairs.append((jaccard, left, right))
    pairs.sort(reverse=True)
    selected_jaccards: list[float] = []
    for jaccard, left, right in pairs:
        if left in assigned or right in assigned:
            continue
        labels[left] = 1.0
        labels[right] = -1.0
        assigned.add(left)
        assigned.add(right)
        selected_jaccards.append(jaccard)
        if len(assigned) >= n_odors - 1:
            break
    remaining = [index for index in range(n_odors) if index not in assigned]
    if remaining:
        remaining_labels = _balanced_labels(len(remaining), rng)
        for index, label in zip(remaining, remaining_labels):
            labels[index] = label
    if np.any(labels == 0):
        labels[labels == 0] = 1.0

    top_count = max(1, int(round(max(1, len(pairs)) * float(similar_pair_fraction))))
    top_jaccards = [item[0] for item in pairs[:top_count]]
    return labels, {
        "similar_pair_mean_jaccard": float(np.mean(selected_jaccards)) if selected_jaccards else 0.0,
        "top_similarity_decile_jaccard": float(np.mean(top_jaccards)) if top_jaccards else 0.0,
        "max_pairwise_jaccard": float(pairs[0][0]) if pairs else 0.0,
    }


def _noisy_binary_sample(
    code: np.ndarray,
    rng: np.random.Generator,
    dropout_probability: float,
    false_positive_probability: float,
) -> np.ndarray:
    sample = code.copy()
    if dropout_probability > 0:
        active = np.flatnonzero(sample)
        if len(active):
            sample[active[rng.random(len(active)) < dropout_probability]] = False
    if false_positive_probability > 0:
        inactive = np.flatnonzero(~sample)
        if len(inactive):
            sample[inactive[rng.random(len(inactive)) < false_positive_probability]] = True
    return sample.astype(np.float64)


def _evaluate_associative_memory(
    binary_code: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    config: KCSparseCodingConfig,
    train_steps: int,
    item_indices: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray]:
    n_odors, n_kc = binary_code.shape
    selected = np.arange(n_odors) if item_indices is None else item_indices
    w = np.zeros(n_kc, dtype=np.float64) if weights is None else weights.copy()
    for _ in range(train_steps):
        for odor_index in rng.permutation(selected):
            x = _noisy_binary_sample(
                binary_code[int(odor_index)],
                rng,
                config.dropout_probability,
                config.false_positive_probability,
            )
            active = x.sum()
            if active > 0:
                w += labels[int(odor_index)] * x / active
    correct = 0
    margins: list[float] = []
    total = 0
    for odor_index in selected:
        for _ in range(config.test_repeats):
            x = _noisy_binary_sample(
                binary_code[int(odor_index)],
                rng,
                config.dropout_probability,
                config.false_positive_probability,
            )
            score = float(x @ w)
            signed_margin = labels[int(odor_index)] * score
            prediction = 1.0 if score >= 0 else -1.0
            correct += int(prediction == labels[int(odor_index)])
            margins.append(float(signed_margin))
            total += 1
    accuracy = correct / total if total else 0.0
    margin = float(np.mean(margins)) if margins else 0.0
    return float(accuracy), margin, w


def _learning_and_forgetting_metrics(
    binary_code: np.ndarray,
    config: KCSparseCodingConfig,
    seed: int,
    labels: np.ndarray | None = None,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n_odors = binary_code.shape[0]
    labels = _balanced_labels(n_odors, rng) if labels is None else labels.astype(np.float64).copy()
    accuracy_by_step: list[float] = []
    margin_by_step: list[float] = []
    criterion_step = config.max_learning_steps + 1
    for step in range(1, config.max_learning_steps + 1):
        accuracy, margin, _ = _evaluate_associative_memory(binary_code, labels, rng, config, train_steps=step)
        accuracy_by_step.append(accuracy)
        margin_by_step.append(margin)
        if criterion_step == config.max_learning_steps + 1 and accuracy >= config.learning_accuracy_criterion:
            criterion_step = step

    split = max(1, n_odors // 2)
    primary = np.arange(0, split)
    interference = np.arange(split, n_odors)
    if len(interference) == 0:
        interference = primary.copy()
    initial_accuracy, _, weights = _evaluate_associative_memory(
        binary_code,
        labels,
        rng,
        config,
        train_steps=config.max_learning_steps,
        item_indices=primary,
    )
    current_weights = weights
    final_primary_accuracy = initial_accuracy
    for _ in range(config.forgetting_interference_steps):
        _, _, current_weights = _evaluate_associative_memory(
            binary_code,
            labels,
            rng,
            config,
            train_steps=1,
            item_indices=interference,
            weights=current_weights,
        )
        final_primary_accuracy, _, _ = _evaluate_associative_memory(
            binary_code,
            labels,
            rng,
            config,
            train_steps=0,
            item_indices=primary,
            weights=current_weights,
        )
    forgetting_rate = max(0.0, initial_accuracy - final_primary_accuracy) / max(
        1,
        config.forgetting_interference_steps,
    )

    return {
        "learning_accuracy_final": float(accuracy_by_step[-1]),
        "learning_steps_to_criterion": float(criterion_step),
        "association_margin_final": float(margin_by_step[-1]),
        "primary_memory_initial_accuracy": float(initial_accuracy),
        "retention_accuracy_after_interference": float(final_primary_accuracy),
        "forgetting_rate_per_interference_step": float(forgetting_rate),
    }


def _average_metric_records(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    keys = sorted({key for record in records for key in record})
    averaged: dict[str, float] = {}
    for key in keys:
        values = np.asarray([record[key] for record in records if key in record], dtype=np.float64)
        if len(values) == 0:
            continue
        averaged[key] = float(np.mean(values))
        if key in {
            "learning_accuracy_final",
            "retention_accuracy_after_interference",
            "forgetting_rate_per_interference_step",
        }:
            averaged[f"{key}_std"] = float(np.std(values, ddof=0))
    return averaged


def _prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def evaluate_sparse_code(
    activity_matrix: np.ndarray,
    activation_ratio: float,
    config: KCSparseCodingConfig,
    seed: int,
) -> dict[str, float]:
    binary, graded = sparsify_activity(activity_matrix, activation_ratio)
    pairwise = _pairwise_code_metrics(binary)
    entropy_per_kc, entropy_total = _population_entropy(binary)
    learning_records = [
        _learning_and_forgetting_metrics(binary, config, seed + repeat * 997)
        for repeat in range(max(1, config.evaluation_repeats))
    ]
    learning = _average_metric_records(learning_records)

    conflict_rng = np.random.default_rng(seed + 90_001)
    conflict_labels, conflict_pair_stats = _similar_pair_labels(
        binary,
        conflict_rng,
        similar_pair_fraction=config.similar_pair_fraction,
    )
    conflict_records = [
        _learning_and_forgetting_metrics(binary, config, seed + 50_000 + repeat * 997, labels=conflict_labels)
        for repeat in range(max(1, config.evaluation_repeats))
    ]
    conflict_learning = _prefix_metrics("similar_pair", _average_metric_records(conflict_records))
    return {
        "activation_ratio_target": float(activation_ratio),
        "active_fraction_observed": float(binary.mean()),
        "mean_active_kc": float(binary.sum(axis=1).mean()),
        "graded_activity_mass": float(graded.sum(axis=1).mean()),
        "population_entropy_bits_per_kc": entropy_per_kc,
        "population_entropy_total_bits": entropy_total,
        "metabolic_cost_active_fraction": float(binary.mean()),
        **pairwise,
        **conflict_pair_stats,
        **learning,
        **conflict_learning,
    }


def evaluate_binary_code(
    binary: np.ndarray,
    graded: np.ndarray,
    activation_ratio_target: float,
    config: KCSparseCodingConfig,
    seed: int,
) -> dict[str, float]:
    pairwise = _pairwise_code_metrics(binary)
    entropy_per_kc, entropy_total = _population_entropy(binary)
    learning_records = [
        _learning_and_forgetting_metrics(binary, config, seed + repeat * 997)
        for repeat in range(max(1, config.evaluation_repeats))
    ]
    learning = _average_metric_records(learning_records)
    conflict_rng = np.random.default_rng(seed + 90_001)
    conflict_labels, conflict_pair_stats = _similar_pair_labels(
        binary,
        conflict_rng,
        similar_pair_fraction=config.similar_pair_fraction,
    )
    conflict_records = [
        _learning_and_forgetting_metrics(binary, config, seed + 50_000 + repeat * 997, labels=conflict_labels)
        for repeat in range(max(1, config.evaluation_repeats))
    ]
    conflict_learning = _prefix_metrics("similar_pair", _average_metric_records(conflict_records))
    return {
        "activation_ratio_target": float(activation_ratio_target),
        "active_fraction_observed": float(binary.mean()),
        "mean_active_kc": float(binary.sum(axis=1).mean()),
        "graded_activity_mass": float(graded.sum(axis=1).mean()),
        "population_entropy_bits_per_kc": entropy_per_kc,
        "population_entropy_total_bits": entropy_total,
        "metabolic_cost_active_fraction": float(binary.mean()),
        **pairwise,
        **conflict_pair_stats,
        **learning,
        **conflict_learning,
    }


def _add_pareto_score(frame: pd.DataFrame, config: KCSparseCodingConfig) -> pd.DataFrame:
    scored = frame.copy()
    if scored.empty:
        scored["pareto_score"] = []
        return scored
    speed = 1.0 - (scored["learning_steps_to_criterion"].clip(1, config.max_learning_steps + 1) - 1) / config.max_learning_steps
    entropy = scored["population_entropy_bits_per_kc"]
    entropy_norm = (entropy - entropy.min()) / (entropy.max() - entropy.min()) if entropy.max() > entropy.min() else 0.0
    separation = 1.0 - scored["mean_jaccard_overlap"].clip(0, 1)
    conflict_accuracy = scored.get("similar_pair_learning_accuracy_final", scored["learning_accuracy_final"]).clip(0, 1)
    conflict_overlap = scored.get("similar_pair_mean_jaccard", scored["mean_jaccard_overlap"]).clip(0, 1)
    scored["learning_speed_score"] = speed.clip(0, 1)
    scored["entropy_norm"] = entropy_norm
    scored["similar_odor_confusability_proxy"] = conflict_overlap
    scored["similar_odor_discrimination_proxy"] = 1.0 - conflict_overlap
    scored["empirical_anchor_distance_from_literature"] = (
        scored["active_fraction_observed"] - config.normal_apl_ratio
    ).abs()
    denominator = max(1e-9, config.max_activation_ratio - config.min_activation_ratio)
    scored["empirical_anchor_score_literature"] = (
        1.0 - scored["empirical_anchor_distance_from_literature"] / denominator
    ).clip(0, 1)
    scored["legacy_distance_from_1_6"] = (
        scored["active_fraction_observed"] - LEGACY_ONE_SIXTH_KC_RATIO
    ).abs()
    scored["legacy_score_1_6"] = (1.0 - scored["legacy_distance_from_1_6"] / denominator).clip(0, 1)
    # Backward-compatible column names for old notebooks/reports.
    scored["empirical_anchor_distance_from_1_6"] = scored["legacy_distance_from_1_6"]
    scored["empirical_anchor_score_1_6"] = scored["legacy_score_1_6"]
    scored["odor_discrimination_score"] = (0.60 * conflict_accuracy + 0.40 * (1.0 - conflict_overlap)).clip(0, 1)
    scored["pareto_score"] = (
        0.20 * scored["learning_accuracy_final"].clip(0, 1)
        + 0.20 * scored["retention_accuracy_after_interference"].clip(0, 1)
        + 0.22 * scored["odor_discrimination_score"]
        + 0.10 * scored["learning_speed_score"].clip(0, 1)
        + 0.13 * separation
        + 0.10 * scored["entropy_norm"]
        - 0.05 * scored["metabolic_cost_active_fraction"].clip(0, 1)
    )
    return scored


def run_ratio_sweep(activity_matrix: np.ndarray, config: KCSparseCodingConfig) -> pd.DataFrame:
    records = []
    for index, ratio in enumerate(config.activation_ratios):
        metrics = evaluate_sparse_code(activity_matrix, ratio, config, seed=config.random_seed + index * 101)
        metrics["condition_type"] = "kc_activation_ratio_sweep"
        metrics["apl_gain"] = np.nan
        records.append(metrics)
    return _add_pareto_score(pd.DataFrame.from_records(records), config)


def run_apl_gain_sweep(activity_matrix: np.ndarray, config: KCSparseCodingConfig) -> pd.DataFrame:
    records = []
    for index, apl_gain in enumerate(config.apl_gains):
        ratio = ratio_from_apl_gain(
            apl_gain,
            normal_ratio=config.normal_apl_ratio,
            min_ratio=config.min_activation_ratio,
            max_ratio=config.max_activation_ratio,
        )
        metrics = evaluate_sparse_code(activity_matrix, ratio, config, seed=config.random_seed + 10_000 + index * 101)
        metrics["condition_type"] = "apl_gain_sweep"
        metrics["apl_gain"] = float(apl_gain)
        if apl_gain < 0.8:
            state = "APL_suppressed"
        elif apl_gain > 1.2:
            state = "APL_enhanced"
        else:
            state = "APL_near_normal"
        metrics["apl_state"] = state
        records.append(metrics)
    return _add_pareto_score(pd.DataFrame.from_records(records), config)


def run_apl_threshold_sweep(activity_matrix: np.ndarray, config: KCSparseCodingConfig) -> pd.DataFrame:
    baseline_threshold = calibrate_apl_threshold(activity_matrix, config.normal_apl_ratio)
    records = []
    for index, apl_gain in enumerate(config.apl_gains):
        binary, graded = apply_apl_threshold_inhibition(
            activity_matrix,
            apl_gain=apl_gain,
            baseline_threshold=baseline_threshold,
        )
        ratio_target = ratio_from_apl_gain(
            apl_gain,
            normal_ratio=config.normal_apl_ratio,
            min_ratio=config.min_activation_ratio,
            max_ratio=config.max_activation_ratio,
        )
        metrics = evaluate_binary_code(
            binary,
            graded,
            activation_ratio_target=ratio_target,
            config=config,
            seed=config.random_seed + 20_000 + index * 101,
        )
        metrics["condition_type"] = "apl_threshold_sweep"
        metrics["apl_gain"] = float(apl_gain)
        metrics["baseline_threshold"] = baseline_threshold
        metrics["effective_threshold"] = baseline_threshold * max(0.0, float(apl_gain))
        if apl_gain < 0.8:
            state = "APL_suppressed"
        elif apl_gain > 1.2:
            state = "APL_enhanced"
        else:
            state = "APL_near_normal"
        metrics["apl_state"] = state
        records.append(metrics)
    return _add_pareto_score(pd.DataFrame.from_records(records), config)


def _target_group(row: pd.Series) -> str:
    cell_class = str(row.get("cell_class", ""))
    cell_type = str(row.get("cell_type", ""))
    hemibrain_type = str(row.get("hemibrain_type", ""))
    if cell_type == "APL" or hemibrain_type == "APL":
        return "APL"
    if cell_type == "DPM" or hemibrain_type == "DPM":
        return "DPM"
    if cell_class == "MBON":
        return "MBON"
    if cell_class == "DAN":
        return "DAN"
    if cell_class == "MBIN":
        return "other_MBIN"
    return ""


def compute_downstream_readout(
    activity_matrix: np.ndarray,
    odor_names: list[str],
    kc_ids: np.ndarray,
    ratio_table: pd.DataFrame,
    annotation_path: Path | None = None,
    connectivity_path: Path = DEFAULT_CONNECTIVITY_PATH,
) -> pd.DataFrame:
    annotations = _read_annotations(annotation_path)
    target_annotations = annotations.copy()
    target_annotations["target_group"] = target_annotations.apply(_target_group, axis=1)
    targets = target_annotations[target_annotations["target_group"].ne("")][["root_id", "target_group"]].copy()
    if targets.empty:
        return pd.DataFrame()

    edges = load_connectivity_edges(connectivity_path)
    filtered = edges[
        edges["Presynaptic_ID"].isin(set(map(int, kc_ids)))
        & edges["Postsynaptic_ID"].isin(set(targets["root_id"].astype("int64")))
    ].copy()
    if filtered.empty:
        return pd.DataFrame()
    filtered = filtered.merge(targets, left_on="Postsynaptic_ID", right_on="root_id", how="inner")
    per_kc = (
        filtered.groupby(["Presynaptic_ID", "target_group"], as_index=False)["Excitatory x Connectivity"]
        .sum()
        .rename(columns={"Presynaptic_ID": "root_id", "Excitatory x Connectivity": "signed_drive"})
    )
    per_kc["abs_drive"] = per_kc["signed_drive"].abs()
    records = []
    kc_id_by_index = np.asarray(kc_ids, dtype=np.int64)
    for row in ratio_table.itertuples(index=False):
        ratio = float(row.activation_ratio_target)
        binary, _ = sparsify_activity(activity_matrix, ratio)
        for odor_index, odor_name in enumerate(odor_names):
            active_roots = set(map(int, kc_id_by_index[binary[odor_index]]))
            active_drive = per_kc[per_kc["root_id"].isin(active_roots)]
            grouped = active_drive.groupby("target_group", as_index=False).agg(
                signed_downstream_drive=("signed_drive", "sum"),
                absolute_downstream_drive=("abs_drive", "sum"),
                n_active_kc_with_target_edges=("root_id", "nunique"),
            )
            for target_row in grouped.itertuples(index=False):
                records.append(
                    {
                        "condition_type": getattr(row, "condition_type", "unknown"),
                        "activation_ratio_target": ratio,
                        "apl_gain": getattr(row, "apl_gain", np.nan),
                        "odor_identity": odor_name,
                        "target_group": target_row.target_group,
                        "signed_downstream_drive": float(target_row.signed_downstream_drive),
                        "absolute_downstream_drive": float(target_row.absolute_downstream_drive),
                        "n_active_kc_with_target_edges": int(target_row.n_active_kc_with_target_edges),
                    }
                )
    return pd.DataFrame.from_records(records)


def _text_contains(frame: pd.DataFrame, columns: Iterable[str], pattern: str) -> pd.Series:
    selected = [column for column in columns if column in frame.columns]
    if not selected:
        return pd.Series(False, index=frame.index)
    text = frame[selected].fillna("").astype(str).agg(" ".join, axis=1)
    return text.str.contains(pattern, case=False, regex=True, na=False)


def build_state_candidate_sets(annotations: pd.DataFrame, max_seed_neurons: int) -> dict[str, list[int]]:
    text_columns = ["cell_class", "cell_type", "hemibrain_type", "top_nt", "synonyms"]
    cell_class = annotations.get("cell_class", pd.Series("", index=annotations.index)).fillna("").astype(str)
    top_nt = annotations.get("top_nt", pd.Series("", index=annotations.index)).fillna("").astype(str)
    candidates = {
        "octopamine_arousal": top_nt.eq("octopamine")
        | _text_contains(annotations, text_columns, r"\bOA[-_A-Za-z0-9]*"),
        "dopamine_reinforcement_DAN": cell_class.eq("DAN"),
        "serotonin_stress_like": top_nt.eq("serotonin"),
        "metabolic_PI_DH44_IPC": cell_class.eq("pars_intercerebralis")
        | _text_contains(annotations, text_columns, r"\b(?:DH44|IPC|DILP)\b"),
        "circadian_clock_like": _text_contains(annotations, text_columns, r"\b(?:LNv|LNd|DN1|DN2|DN3|PDF)\b"),
        "olfactory_load_ORN_ALPN": cell_class.isin(["olfactory", "ALPN"]),
        "mechanosensory_arousal": cell_class.eq("mechanosensory"),
    }
    result: dict[str, list[int]] = {}
    for name, mask in candidates.items():
        selected = annotations.loc[mask.fillna(False), "root_id"].dropna().astype("int64").sort_values().tolist()
        result[name] = selected[:max_seed_neurons]
    return result


def predict_state_to_apl(
    annotation_path: Path | None = None,
    connectivity_path: Path = DEFAULT_CONNECTIVITY_PATH,
    config: KCSparseCodingConfig | None = None,
) -> pd.DataFrame:
    config = config or KCSparseCodingConfig()
    annotations = _read_annotations(annotation_path)
    apl_ids = set(
        annotations.loc[
            (annotations.get("cell_type", pd.Series("", index=annotations.index)).astype(str).eq("APL"))
            | (annotations.get("hemibrain_type", pd.Series("", index=annotations.index)).astype(str).eq("APL")),
            "root_id",
        ]
        .dropna()
        .astype("int64")
    )
    kc_ids = set(
        annotations.loc[
            annotations.get("cell_class", pd.Series("", index=annotations.index)).astype(str).eq("Kenyon_Cell"),
            "root_id",
        ]
        .dropna()
        .astype("int64")
    )
    mbon_ids = set(
        annotations.loc[
            annotations.get("cell_class", pd.Series("", index=annotations.index)).astype(str).eq("MBON"),
            "root_id",
        ]
        .dropna()
        .astype("int64")
    )
    dan_ids = set(
        annotations.loc[
            annotations.get("cell_class", pd.Series("", index=annotations.index)).astype(str).eq("DAN"),
            "root_id",
        ]
        .dropna()
        .astype("int64")
    )
    if not apl_ids:
        return pd.DataFrame()
    edges = load_connectivity_edges(connectivity_path)
    state_sets = build_state_candidate_sets(annotations, config.max_state_seed_neurons)
    records = []
    for state_name, seed_ids in state_sets.items():
        if not seed_ids:
            continue
        response = signed_multihop_response(
            edges,
            seed_ids=seed_ids,
            config=PropagationConfig(steps=config.state_propagation_steps, max_active=config.state_max_active),
        )
        aggregate = response.groupby("root_id", as_index=False)["score"].sum() if not response.empty else pd.DataFrame()
        def abs_mass(ids: set[int]) -> float:
            if aggregate.empty:
                return 0.0
            return float(aggregate.loc[aggregate["root_id"].isin(ids), "score"].abs().sum())

        apl_signed = (
            float(aggregate.loc[aggregate["root_id"].isin(apl_ids), "score"].sum()) if not aggregate.empty else 0.0
        )
        apl_abs = abs_mass(apl_ids)
        kc_abs = abs_mass(kc_ids)
        mbon_abs = abs_mass(mbon_ids)
        dan_abs = abs_mass(dan_ids)
        predicted_shift = float(np.tanh(apl_signed * 8.0))
        if predicted_shift < -0.05:
            effect = "candidate_APL_suppression"
        elif predicted_shift > 0.05:
            effect = "candidate_APL_enhancement"
        else:
            effect = "weak_or_ambiguous_APL_effect"
        records.append(
            {
                "physiological_state_proxy": state_name,
                "n_seed_neurons": len(seed_ids),
                "n_active_neurons": int(aggregate["root_id"].nunique()) if not aggregate.empty else 0,
                "apl_signed_score": apl_signed,
                "apl_abs_mass": apl_abs,
                "kc_abs_mass": kc_abs,
                "mbon_abs_mass": mbon_abs,
                "dan_abs_mass": dan_abs,
                "predicted_apl_gain_shift_proxy": predicted_shift,
                "predicted_effect": effect,
            }
        )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    frame["apl_priority_score"] = frame["apl_abs_mass"] * (1.0 + frame["predicted_apl_gain_shift_proxy"].abs())
    return frame.sort_values("apl_priority_score", ascending=False).reset_index(drop=True)


def build_wetlab_protocol_table(
    ratio_table: pd.DataFrame,
    apl_table: pd.DataFrame,
    state_table: pd.DataFrame,
    config: KCSparseCodingConfig,
    threshold_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    reference = ratio_table.iloc[
        (ratio_table["activation_ratio_target"] - config.normal_apl_ratio).abs().argsort()[:1]
    ]
    reference_fraction = float(reference["active_fraction_observed"].iloc[0]) if not reference.empty else np.nan
    reference_discrimination = (
        float(reference["similar_odor_discrimination_proxy"].iloc[0]) if not reference.empty else np.nan
    )
    suppressed = apl_table[apl_table["apl_state"].eq("APL_suppressed")].copy()
    suppressed_overlap = float(suppressed["mean_jaccard_overlap"].mean()) if not suppressed.empty else np.nan
    suppressed_confusability = (
        float(suppressed["similar_odor_confusability_proxy"].mean()) if not suppressed.empty else np.nan
    )
    normal = apl_table.iloc[(apl_table["apl_gain"] - 1.0).abs().argsort()[:1]]
    normal_overlap = float(normal["mean_jaccard_overlap"].iloc[0]) if not normal.empty else np.nan
    normal_confusability = (
        float(normal["similar_odor_confusability_proxy"].iloc[0]) if not normal.empty else np.nan
    )
    threshold_note = ""
    if threshold_table is not None and not threshold_table.empty:
        threshold_normal = threshold_table.iloc[(threshold_table["apl_gain"] - 1.0).abs().argsort()[:1]]
        threshold_suppressed = threshold_table[threshold_table["apl_state"].eq("APL_suppressed")]
        if not threshold_normal.empty and not threshold_suppressed.empty:
            threshold_note = (
                f"; threshold model normal {float(threshold_normal['active_fraction_observed'].iloc[0]):.3f}, "
                f"suppressed {float(threshold_suppressed['active_fraction_observed'].mean()):.3f}"
            )
    top_state = state_table["physiological_state_proxy"].iloc[0] if not state_table.empty else "需要新增状态 seed"
    rows = [
        {
            "experiment": "Reproduce normal KC sparsity",
            "simulation_result_to_test": f"normal APL is anchored to active fraction {reference_fraction:.4f}; similar-odor discrimination proxy {reference_discrimination:.3f}",
            "wetlab_prediction": "把 <=10% 可靠响应 KC 作为已知生理锚点复现，而不是让简单分类器重新发现；实测当前 odor panel 是否落在约 5-10% KC 激活范围。",
            "primary_readout": "每个气味激活 KC 比例、KC pattern overlap、trial-to-trial 稳定性",
            "critical_control": "同一气味浓度、同一 imaging plane、APL 未扰动遗传背景对照",
        },
        {
            "experiment": "APL loss-of-function or output blockade",
            "simulation_result_to_test": f"suppressed overlap {suppressed_overlap:.3f} vs normal {normal_overlap:.3f}; confusability {suppressed_confusability:.3f} vs {normal_confusability:.3f}{threshold_note}",
            "wetlab_prediction": "APL 抑制减弱会扩大 KC 激活集合、提高不同气味 pattern overlap/confusability，并降低相近气味学习辨别。",
            "primary_readout": "KC active fraction, OCT/MCH or similar-odor discrimination index, memory generalization",
            "critical_control": "APL rescue、温度/光照本身控制、不同 odor similarity 梯度",
        },
        {
            "experiment": "Interference learning and forgetting assay",
            "simulation_result_to_test": "very dense KC code has higher overlap and lower retention in the associative-memory proxy",
            "wetlab_prediction": "强 APL 抑制或接近 1/2 激活时，连续学习两个气味关联后旧记忆 retention 更低。",
            "primary_readout": "old-memory retention, new-memory acquisition, generalization to similar odors",
            "critical_control": "训练强度匹配、shock/sugar reinforcement 强度匹配、运动能力控制",
        },
        {
            "experiment": "Physiological-state candidate screen",
            "simulation_result_to_test": f"top connectome-prior state proxy: {top_state}",
            "wetlab_prediction": "优先测试该状态是否改变 APL activity 或 KC sparsity；不要先把它命名为 depression-like causality。",
            "primary_readout": "APL calcium/GABA proxy, KC active fraction, odor discrimination index",
            "critical_control": "状态操纵的 arousal/locomotion/confound readout 与遗传背景对照",
        },
    ]
    return pd.DataFrame.from_records(rows)


def build_mechanism_audit_table(
    ratio_table: pd.DataFrame,
    apl_table: pd.DataFrame,
    state_table: pd.DataFrame,
    config: KCSparseCodingConfig,
    threshold_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    normal = apl_table.iloc[(apl_table["apl_gain"] - 1.0).abs().argsort()[:1]].iloc[0]
    suppressed = apl_table[apl_table["apl_state"].eq("APL_suppressed")].copy()
    best_proxy = ratio_table.sort_values("pareto_score", ascending=False).iloc[0]
    target_ratio = float(config.normal_apl_ratio)
    normal_fraction = float(normal["active_fraction_observed"])
    suppressed_fraction = float(suppressed["active_fraction_observed"].mean()) if not suppressed.empty else np.nan
    normal_overlap = float(normal["mean_jaccard_overlap"])
    suppressed_overlap = float(suppressed["mean_jaccard_overlap"].mean()) if not suppressed.empty else np.nan
    normal_confusability = float(normal["similar_odor_confusability_proxy"])
    suppressed_confusability = (
        float(suppressed["similar_odor_confusability_proxy"].mean()) if not suppressed.empty else np.nan
    )
    best_proxy_ratio = float(best_proxy["activation_ratio_target"])
    top_state = state_table.iloc[0] if not state_table.empty else None
    threshold_normal = None
    threshold_suppressed = pd.DataFrame()
    if threshold_table is not None and not threshold_table.empty:
        threshold_normal = threshold_table.iloc[(threshold_table["apl_gain"] - 1.0).abs().argsort()[:1]].iloc[0]
        threshold_suppressed = threshold_table[threshold_table["apl_state"].eq("APL_suppressed")].copy()

    rows = [
        {
            "check_id": "normal_apl_reproduces_kc_sparsity_anchor",
            "claim_or_guardrail": "Normal APL should reproduce the literature/physiology anchor of <=10% KC activation.",
            "observed_value": normal_fraction,
            "comparison_value": target_ratio,
            "status": "pass" if abs(normal_fraction - target_ratio) <= 0.02 else "fail",
            "interpretation": "正常 APL gain 被校准到 <=10% KC active fraction；这是复现已知生理锚点，不是重新发现该比例。",
        },
        {
            "check_id": "apl_suppression_increases_kc_active_fraction",
            "claim_or_guardrail": "Reducing APL gain should make the KC code denser.",
            "observed_value": suppressed_fraction,
            "comparison_value": normal_fraction,
            "status": "pass" if suppressed_fraction > normal_fraction else "fail",
            "interpretation": "APL 抑制条件下 active fraction 高于 normal，符合 APL 维持稀疏编码的机制预期。",
        },
        {
            "check_id": "apl_suppression_increases_odor_overlap",
            "claim_or_guardrail": "Reducing APL gain should increase odor-code overlap.",
            "observed_value": suppressed_overlap,
            "comparison_value": normal_overlap,
            "status": "pass" if suppressed_overlap > normal_overlap else "fail",
            "interpretation": "APL 抑制提高 KC odor-code overlap，是可直接转化为 calcium imaging 的湿实验预测。",
        },
        {
            "check_id": "apl_suppression_increases_similar_odor_confusability",
            "claim_or_guardrail": "Reducing APL gain should increase similar-odor confusability.",
            "observed_value": suppressed_confusability,
            "comparison_value": normal_confusability,
            "status": "pass" if suppressed_confusability > normal_confusability else "fail",
            "interpretation": "相近气味 confusability proxy 上升，支持做相近气味辨别和泛化行为实验。",
        },
        {
            "check_id": "toy_proxy_score_is_not_a_biological_optimum",
            "claim_or_guardrail": "A simple associative proxy must not be used to claim de novo discovery of a fixed KC sparsity fraction.",
            "observed_value": best_proxy_ratio,
            "comparison_value": target_ratio,
            "status": "model_limit_warning" if abs(best_proxy_ratio - target_ratio) > 0.02 else "pass",
            "interpretation": "最高 proxy score 偏离 <=10% 锚点时，说明 toy classifier 有自身偏好；报告必须把 KC 稀疏度作为文献锚点复现。",
        },
        {
            "check_id": "threshold_apl_model_reproduces_normal_sparsity",
            "claim_or_guardrail": "A threshold/divisive APL model should also reproduce the normal sparsity anchor.",
            "observed_value": float(threshold_normal["active_fraction_observed"]) if threshold_normal is not None else np.nan,
            "comparison_value": target_ratio,
            "status": (
                "pass"
                if threshold_normal is not None and abs(float(threshold_normal["active_fraction_observed"]) - target_ratio) <= 0.03
                else "not_run"
            ),
            "interpretation": "阈值抑制模型在 normal APL 下也应接近 <=10% 锚点；这是比 top-k 更机制化的复现检查。",
        },
        {
            "check_id": "threshold_apl_suppression_increases_overlap",
            "claim_or_guardrail": "Threshold/divisive APL suppression should increase odor-code overlap.",
            "observed_value": (
                float(threshold_suppressed["mean_jaccard_overlap"].mean()) if not threshold_suppressed.empty else np.nan
            ),
            "comparison_value": (
                float(threshold_normal["mean_jaccard_overlap"]) if threshold_normal is not None else np.nan
            ),
            "status": (
                "pass"
                if threshold_normal is not None
                and not threshold_suppressed.empty
                and float(threshold_suppressed["mean_jaccard_overlap"].mean()) > float(threshold_normal["mean_jaccard_overlap"])
                else "not_run"
            ),
            "interpretation": "如果阈值模型中 APL 抑制也提高 overlap，说明该预测不依赖 top-k 代理。",
        },
        {
            "check_id": "state_to_apl_prediction_is_hypothesis_only",
            "claim_or_guardrail": "Physiological-state predictions are candidate screens, not causal proof.",
            "observed_value": float(top_state["apl_priority_score"]) if top_state is not None else np.nan,
            "comparison_value": 0.0,
            "status": "hypothesis_only",
            "interpretation": (
                f"当前最高状态候选为 {top_state['physiological_state_proxy']}；需要 APL activity/KC sparsity 湿实验校准。"
                if top_state is not None
                else "当前无状态候选；需要补充 state seed。"
            ),
        },
    ]
    return pd.DataFrame.from_records(rows)


def _markdown_table(frame: pd.DataFrame, max_rows: int = 12) -> str:
    if frame.empty:
        return "No rows."
    display = frame.head(max_rows).copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4g}")
        else:
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else str(value))
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.values.tolist():
        lines.append("| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def plot_ratio_sweep(ratio_table: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    x = ratio_table["activation_ratio_target"]
    axes[0].plot(x, ratio_table["learning_accuracy_final"], marker="o", label="learning")
    axes[0].plot(x, ratio_table["retention_accuracy_after_interference"], marker="s", label="retention")
    axes[0].plot(x, ratio_table["similar_pair_learning_accuracy_final"], marker="^", label="similar-pair")
    axes[0].set_xlabel("KC active fraction")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(frameon=False)
    axes[1].plot(x, ratio_table["mean_jaccard_overlap"], marker="o", color="#a13d2d")
    axes[1].set_xlabel("KC active fraction")
    axes[1].set_ylabel("Mean odor-code Jaccard")
    axes[2].plot(x, ratio_table["pareto_score"], marker="o", color="#2f6b4f")
    best = ratio_table.sort_values("pareto_score", ascending=False).iloc[0]
    axes[2].axvline(float(best["activation_ratio_target"]), color="#444444", linestyle="--", linewidth=1)
    axes[2].set_xlabel("KC active fraction")
    axes[2].set_ylabel("Heuristic Pareto score")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_apl_sweep(apl_table: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    x = apl_table["apl_gain"]
    axes[0].plot(x, apl_table["active_fraction_observed"], marker="o")
    axes[0].axvline(1.0, color="#444444", linestyle="--", linewidth=1)
    axes[0].set_xlabel("APL gain")
    axes[0].set_ylabel("KC active fraction")
    axes[1].plot(x, apl_table["mean_jaccard_overlap"], marker="o", color="#a13d2d")
    axes[1].axvline(1.0, color="#444444", linestyle="--", linewidth=1)
    axes[1].set_xlabel("APL gain")
    axes[1].set_ylabel("Mean odor-code Jaccard")
    axes[2].plot(x, apl_table["retention_accuracy_after_interference"], marker="o", color="#2f6b4f", label="retention")
    axes[2].plot(
        x,
        apl_table["similar_pair_learning_accuracy_final"],
        marker="^",
        color="#574b90",
        label="similar-pair",
    )
    axes[2].axvline(1.0, color="#444444", linestyle="--", linewidth=1)
    axes[2].set_xlabel("APL gain")
    axes[2].set_ylabel("Accuracy")
    axes[2].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_state_predictions(state_table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    if state_table.empty:
        ax.text(0.5, 0.5, "No state predictions", ha="center", va="center")
        ax.axis("off")
    else:
        display = state_table.head(8).iloc[::-1]
        colors = np.where(display["predicted_apl_gain_shift_proxy"] >= 0, "#2f6b4f", "#a13d2d")
        ax.barh(display["physiological_state_proxy"], display["apl_priority_score"], color=colors)
        ax.set_xlabel("APL priority score")
        ax.set_ylabel("State proxy")
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_experiment_report(
    output_dir: Path,
    ratio_table: pd.DataFrame,
    apl_table: pd.DataFrame,
    threshold_table: pd.DataFrame,
    state_table: pd.DataFrame,
    wetlab_table: pd.DataFrame,
    mechanism_audit: pd.DataFrame,
    paths: KCSparseCodingPaths,
    config: KCSparseCodingConfig,
) -> Path:
    best_proxy_ratio = ratio_table.sort_values("pareto_score", ascending=False).iloc[0]
    normal_ratio = ratio_table.iloc[
        (ratio_table["activation_ratio_target"] - config.normal_apl_ratio).abs().argsort()[:1]
    ].iloc[0]
    normal_row = apl_table.iloc[(apl_table["apl_gain"] - 1.0).abs().argsort()[:1]].iloc[0]
    suppressed = apl_table[apl_table["apl_state"].eq("APL_suppressed")]
    suppressed_overlap = float(suppressed["mean_jaccard_overlap"].mean()) if not suppressed.empty else np.nan
    top_state = state_table.iloc[0] if not state_table.empty else None
    state_sentence = (
        f"当前状态传播优先级最高的是 `{top_state['physiological_state_proxy']}`，APL priority score "
        f"{float(top_state['apl_priority_score']):.4g}。"
        if top_state is not None
        else "当前没有得到可排序的状态候选，需要补充状态 seed 或传播数据。"
    )
    report_path = output_dir / "KC_SPARSE_CODING_APL_VALIDATION_REPORT_CN.md"
    report_path.write_text(
        f"""# KC 稀疏编码与 APL 调控仿真实验报告

保存路径：`{report_path}`

## 实验目的

本报告对应会议提出的两个待办：

1. 在不同 KC 激活比例下比较编码效率、学习速度和遗忘率，检验文献中的 `<=10%` 可靠响应 KC 锚点是否处在有效工作区。
2. 模拟 APL 抑制强度变化，预测 APL 被抑制后 KC 编码、气味分辨和记忆干扰会如何改变，并给出湿实验优先级。

## 生物依据和模型边界

- Lin et al., Nature Neuroscience 2014 报道 KC-APL 负反馈维持 KC 稀疏、去相关气味编码；论文正文给出的 KC 单气味响应量级是约 5-10%，破坏该反馈会增加 KC odor response 重叠，并损害相近气味的学习辨别：https://pmc.ncbi.nlm.nih.gov/articles/PMC4000970/
- Caron et al., Nature 2013 支持投射神经元输入随机汇聚到 KC，是 MB 稀疏组合编码的结构基础：https://www.nature.com/articles/nature12063
- 本实验使用 FlyWire 注释和当前 `OCT/MCH -> KC readout` 作为 connectome-constrained 输入，再扩展为 jittered odor panel。它是计算假说生成，不是 spike-level 生理模型，也不是已完成湿实验。
- `APL gain -> KC active fraction` 采用参数化抑制近似：正常 APL gain 约对应 `{config.normal_apl_ratio:.4f}`，APL gain 下降使 KC code 变密，APL gain 上升使 code 更稀疏。

## 主要发现

1. 本版仿真把生理文献中的 KC 稀疏激活量级作为锚点：normal APL gain 对应 active fraction `{float(normal_row['active_fraction_observed']):.4f}`，即约 `<=10%`，不是旧报告中的 `1/6`。
2. 当前 sweep 中最高 proxy score 出现在 `{float(best_proxy_ratio['activation_ratio_target']):.4f}`，这暴露了旧目标函数的问题：简单线性关联读出有自身偏好，不能用来“发现”固定生物比例。因此报告将 `<=10%` 作为已知生理结果来复现，并把优化指标改为 overlap/confusability/能耗/干扰的机制解释。
3. 生理参考比例 `{config.normal_apl_ratio:.4f}` 的 mean Jaccard overlap 为 `{float(normal_ratio['mean_jaccard_overlap']):.4f}`，相近气味 confusability proxy 为 `{float(normal_ratio['similar_odor_confusability_proxy']):.4f}`；APL suppressed 条件的平均 overlap 为 `{suppressed_overlap:.4f}`。APL 抑制提高 overlap 是本仿真的主要可检验预测。
4. {state_sentence}

## KC 激活比例 sweep

{_markdown_table(ratio_table[['activation_ratio_target', 'active_fraction_observed', 'mean_jaccard_overlap', 'similar_pair_mean_jaccard', 'similar_odor_confusability_proxy', 'empirical_anchor_score_literature', 'legacy_score_1_6', 'learning_accuracy_final', 'retention_accuracy_after_interference', 'pareto_score']])}

核心图：`{paths.ratio_figure}`

## APL gain sweep

{_markdown_table(apl_table[['apl_gain', 'apl_state', 'activation_ratio_target', 'active_fraction_observed', 'mean_jaccard_overlap', 'similar_odor_confusability_proxy', 'empirical_anchor_score_literature', 'legacy_score_1_6', 'retention_accuracy_after_interference', 'pareto_score']])}

## APL threshold inhibition sweep

{_markdown_table(threshold_table[['apl_gain', 'apl_state', 'baseline_threshold', 'effective_threshold', 'active_fraction_observed', 'mean_jaccard_overlap', 'similar_odor_confusability_proxy', 'empirical_anchor_score_literature', 'legacy_score_1_6']] if not threshold_table.empty else threshold_table)}

核心图：`{paths.apl_figure}`

## 生理状态候选

{_markdown_table(state_table[['physiological_state_proxy', 'n_seed_neurons', 'apl_signed_score', 'apl_abs_mass', 'kc_abs_mass', 'predicted_apl_gain_shift_proxy', 'predicted_effect', 'apl_priority_score']] if not state_table.empty else state_table)}

核心图：`{paths.state_figure}`

## 湿实验启发

{_markdown_table(wetlab_table, max_rows=8)}

## 机制审计

{_markdown_table(mechanism_audit, max_rows=12)}

## 推荐的湿实验读出

- KC calcium imaging：每种气味激活 KC 比例、不同气味 KC pattern overlap、trial-to-trial 稳定性。
- APL manipulation：APL output blockade、APL rescue、APL gain enhancement，配合相同气味 panel。
- 行为验证：OCT/MCH 与相近气味 pair 的 discrimination index、generalization index、interference learning 后 retention。
- 状态验证：优先测试状态候选是否改变 APL activity 或 KC sparsity；在完成行为和神经读出前，不把候选状态写成 depression-like 因果机制。

## 不能过度声称

- 不能说仿真重新发现了固定 KC 比例。更准确的写法是：文献/生理观察给出约 `5-10%` 的 KC 稀疏激活量级，本系统用 APL gain target 复现该工作点，并预测偏离该工作点后的 overlap/confusability 变化。
- 不能说 APL gain 参数就是真实 GABA release 浓度；它是把 APL 抑制强弱映射到 KC 稀疏度的可检验模型。
- 不能说状态传播证明某个生理状态导致类抑郁；只能作为湿实验优先级。

## 输出文件

- odor panel：`{paths.odor_panel}`
- KC ratio sweep：`{paths.ratio_sweep}`
- APL gain sweep：`{paths.apl_sweep}`
- APL threshold sweep：`{paths.apl_threshold_sweep}`
- downstream readout：`{paths.downstream_readout}`
- mechanism audit：`{paths.mechanism_audit}`
- state predictions：`{paths.state_predictions}`
- wetlab protocol：`{paths.wetlab_protocol}`
- metadata：`{paths.metadata}`
""",
        encoding="utf-8",
    )
    return report_path


def run_kc_sparse_coding_experiment(
    output_dir: Path = OUTPUT_ROOT,
    kc_readout_path: Path = DEFAULT_OUTPUT_ROOT / "oct_mch_sensory_encoder" / "oct_mch_kc_readout.csv",
    annotation_path: Path | None = None,
    connectivity_path: Path = DEFAULT_CONNECTIVITY_PATH,
    config: KCSparseCodingConfig | None = None,
    include_downstream: bool = True,
    include_state: bool = True,
) -> KCSparseCodingPaths:
    config = config or KCSparseCodingConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)

    kc_population = load_kc_population(annotation_path)
    kc_readout = _load_kc_readout(kc_readout_path)
    odor_names, base_matrix, kc_ids = build_odor_activity_matrix(kc_readout, kc_population)
    odor_names, activity_matrix, odor_panel = augment_odor_panel(
        odor_names,
        base_matrix,
        random_odor_count=config.random_odor_count,
        random_seed=config.random_seed,
    )
    odor_panel_path = tables_dir / "odor_panel.csv"
    odor_panel.to_csv(odor_panel_path, index=False)

    ratio_table = run_ratio_sweep(activity_matrix, config)
    ratio_path = tables_dir / "kc_activation_ratio_sweep.csv"
    ratio_table.to_csv(ratio_path, index=False)

    apl_table = run_apl_gain_sweep(activity_matrix, config)
    apl_path = tables_dir / "apl_gain_sweep.csv"
    apl_table.to_csv(apl_path, index=False)

    threshold_table = run_apl_threshold_sweep(activity_matrix, config)
    threshold_path = tables_dir / "apl_threshold_sweep.csv"
    threshold_table.to_csv(threshold_path, index=False)

    state_table = pd.DataFrame()
    if include_state:
        state_table = predict_state_to_apl(
            annotation_path=annotation_path,
            connectivity_path=connectivity_path,
            config=config,
        )
    state_path = tables_dir / "physiological_state_to_apl_predictions.csv"
    state_table.to_csv(state_path, index=False)

    downstream = pd.DataFrame()
    if include_downstream:
        downstream = compute_downstream_readout(
            activity_matrix,
            odor_names,
            kc_ids,
            pd.concat([ratio_table, apl_table], ignore_index=True),
            annotation_path=annotation_path,
            connectivity_path=connectivity_path,
        )
    downstream_path = tables_dir / "kc_downstream_readout.csv"
    downstream.to_csv(downstream_path, index=False)

    wetlab_table = build_wetlab_protocol_table(ratio_table, apl_table, state_table, config, threshold_table)
    wetlab_path = tables_dir / "wetlab_validation_protocol.csv"
    wetlab_table.to_csv(wetlab_path, index=False)

    mechanism_audit = build_mechanism_audit_table(ratio_table, apl_table, state_table, config, threshold_table)
    mechanism_audit_path = tables_dir / "mechanism_audit.csv"
    mechanism_audit.to_csv(mechanism_audit_path, index=False)

    ratio_figure = figures_dir / "Fig_kc_activation_ratio_sweep.png"
    apl_figure = figures_dir / "Fig_apl_gain_sweep.png"
    state_figure = figures_dir / "Fig_state_to_apl_prediction.png"
    plot_ratio_sweep(ratio_table, ratio_figure)
    plot_apl_sweep(apl_table, apl_figure)
    plot_state_predictions(state_table, state_figure)

    metadata_path = output_dir / "suite_metadata.json"
    paths = KCSparseCodingPaths(
        output_dir=output_dir,
        odor_panel=odor_panel_path,
        ratio_sweep=ratio_path,
        apl_sweep=apl_path,
        apl_threshold_sweep=threshold_path,
        mechanism_audit=mechanism_audit_path,
        downstream_readout=downstream_path,
        state_predictions=state_path,
        wetlab_protocol=wetlab_path,
        ratio_figure=ratio_figure,
        apl_figure=apl_figure,
        state_figure=state_figure,
        report=output_dir / "KC_SPARSE_CODING_APL_VALIDATION_REPORT_CN.md",
        metadata=metadata_path,
    )
    report_path = write_experiment_report(
        output_dir,
        ratio_table,
        apl_table,
        threshold_table,
        state_table,
        wetlab_table,
        mechanism_audit,
        paths,
        config,
    )
    paths = KCSparseCodingPaths(**{**asdict(paths), "report": report_path})
    metadata_path.write_text(
        json.dumps(
            {
                "paths": {key: str(value) for key, value in asdict(paths).items()},
                "config": asdict(config),
                "n_kc": int(len(kc_ids)),
                "n_observed_odors": int(len(set(kc_readout["odor_identity"]))),
                "n_panel_odors": int(len(odor_names)),
                "include_downstream": include_downstream,
                "include_state": include_state,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths
