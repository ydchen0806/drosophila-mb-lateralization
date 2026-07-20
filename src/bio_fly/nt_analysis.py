from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .paths import PROCESSED_DATA_ROOT, RAW_DATA_ROOT, DEFAULT_OUTPUT_ROOT


NT_COLUMNS = ["gaba", "ach", "glut", "oct", "ser", "da"]
NT_AVG_COLUMNS = [f"{nt}_avg" for nt in NT_COLUMNS]
NT_INPUT_COLUMNS = [f"{nt}_input" for nt in NT_COLUMNS]
NT_FRACTION_COLUMNS = [f"{nt}_fraction" for nt in NT_COLUMNS]


def load_kc_annotations(
    mushroom_body_path: Path = PROCESSED_DATA_ROOT / "flywire_mushroom_body_annotations.parquet",
) -> pd.DataFrame:
    annotations = pd.read_parquet(mushroom_body_path)
    label_text = annotations[["cell_type", "hemibrain_type"]].fillna("").astype(str).agg(" ".join, axis=1)
    kc = annotations[label_text.str.lower().str.contains("kc|kenyon", regex=True)].copy()
    kc = kc[kc["side"].isin(["left", "right"])].copy()
    return kc[["root_id", "side", "cell_type", "hemibrain_type", "super_class", "cell_class", "top_nt"]].drop_duplicates()


def compute_nt_input_by_neuron(
    connections_path: Path = RAW_DATA_ROOT / "zenodo_10676866" / "proofread_connections_783.feather",
    kc_annotations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    kc_annotations = kc_annotations if kc_annotations is not None else load_kc_annotations()
    kc_ids = set(kc_annotations["root_id"].astype("int64").tolist())
    columns = ["pre_pt_root_id", "post_pt_root_id", "syn_count", *NT_AVG_COLUMNS]
    connections = pd.read_feather(connections_path, columns=columns)
    kc_inputs = connections[connections["post_pt_root_id"].isin(kc_ids)].copy()
    for nt, avg_column, input_column in zip(NT_COLUMNS, NT_AVG_COLUMNS, NT_INPUT_COLUMNS):
        kc_inputs[input_column] = kc_inputs["syn_count"].astype("float64") * kc_inputs[avg_column].astype("float64")
    neuron_inputs = (
        kc_inputs.groupby("post_pt_root_id", as_index=False)[["syn_count", *NT_INPUT_COLUMNS]]
        .sum()
        .rename(columns={"post_pt_root_id": "root_id", "syn_count": "total_input_synapses"})
    )
    merged = kc_annotations.merge(neuron_inputs, on="root_id", how="left").fillna(
        {column: 0.0 for column in ["total_input_synapses", *NT_INPUT_COLUMNS]}
    )
    denominator = merged["total_input_synapses"].replace(0, np.nan)
    for nt, input_column, fraction_column in zip(NT_COLUMNS, NT_INPUT_COLUMNS, NT_FRACTION_COLUMNS):
        merged[fraction_column] = (merged[input_column] / denominator).fillna(0.0)
    return merged


def summarize_nt_by_subtype(neuron_inputs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        neuron_inputs.groupby(["hemibrain_type", "cell_type", "side"], dropna=False)
        .agg(
            n_neurons=("root_id", "nunique"),
            mean_total_input_synapses=("total_input_synapses", "mean"),
            sum_total_input_synapses=("total_input_synapses", "sum"),
            **{f"mean_{column}": (column, "mean") for column in NT_INPUT_COLUMNS},
            **{f"sum_{column}": (column, "sum") for column in NT_INPUT_COLUMNS},
        )
        .reset_index()
    )

    effect_records: list[dict[str, object]] = []
    for (hemibrain_type, cell_type), group in summary.groupby(["hemibrain_type", "cell_type"], dropna=False):
        sides = {side: row for side, row in group.set_index("side").iterrows()}
        if "left" not in sides or "right" not in sides:
            continue
        for nt, input_column in zip(NT_COLUMNS, NT_INPUT_COLUMNS):
            left_value = float(sides["left"][f"mean_{input_column}"])
            right_value = float(sides["right"][f"mean_{input_column}"])
            total = left_value + right_value
            laterality = 0.0 if total == 0 else (right_value - left_value) / total
            effect_records.append(
                {
                    "hemibrain_type": hemibrain_type,
                    "cell_type": cell_type,
                    "nt": nt,
                    "left_mean_input": left_value,
                    "right_mean_input": right_value,
                    "right_minus_left": right_value - left_value,
                    "right_laterality_index": laterality,
                    "log2_right_left_ratio": float(np.log2((right_value + 1e-9) / (left_value + 1e-9))),
                    "left_n": int(sides["left"]["n_neurons"]),
                    "right_n": int(sides["right"]["n_neurons"]),
                }
            )
    effects = pd.DataFrame.from_records(effect_records)
    if not effects.empty:
        effects = effects.sort_values("right_laterality_index", ascending=False)
    return summary, effects


def _benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    n_tests = len(values)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.empty(n_tests, dtype=float)
    running_min = 1.0
    for rank in range(n_tests, 0, -1):
        idx = rank - 1
        running_min = min(running_min, ranked[idx] * n_tests / rank)
        adjusted[order[idx]] = running_min
    return pd.Series(np.clip(adjusted, 0, 1), index=p_values.index)


def _cohens_d(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    pooled = np.sqrt(((len(left) - 1) * np.var(left, ddof=1) + (len(right) - 1) * np.var(right, ddof=1)) / (len(left) + len(right) - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(right) - np.mean(left)) / pooled)


def _cliffs_delta(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) == 0 or len(right) == 0:
        return float("nan")
    u_stat, _ = stats.mannwhitneyu(right, left, alternative="two-sided")
    return float((2 * u_stat) / (len(left) * len(right)) - 1)


def _bootstrap_mean_diff_ci(left: np.ndarray, right: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    if len(left) == 0 or len(right) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    left_idx = rng.integers(0, len(left), size=(n_boot, len(left)))
    right_idx = rng.integers(0, len(right), size=(n_boot, len(right)))
    diffs = right[right_idx].mean(axis=1) - left[left_idx].mean(axis=1)
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def _safe_welch_p(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or len(right) < 2:
        return 1.0
    left_constant = np.allclose(left, left[0])
    right_constant = np.allclose(right, right[0])
    if left_constant and right_constant:
        return 1.0 if np.isclose(left[0], right[0]) else 0.0
    try:
        return float(stats.ttest_ind(right, left, equal_var=False, nan_policy="omit").pvalue)
    except ValueError:
        return 1.0


def compute_fraction_statistics(neuron_inputs: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    groups = neuron_inputs.groupby(["hemibrain_type", "cell_type"], dropna=False)
    for (hemibrain_type, cell_type), group in groups:
        left_group = group[group["side"] == "left"]
        right_group = group[group["side"] == "right"]
        if left_group.empty or right_group.empty:
            continue
        for nt, fraction_column in zip(NT_COLUMNS, NT_FRACTION_COLUMNS):
            left = left_group[fraction_column].to_numpy(dtype=float)
            right = right_group[fraction_column].to_numpy(dtype=float)
            mean_diff = float(np.mean(right) - np.mean(left))
            total_mean = float(np.mean(right) + np.mean(left))
            laterality = 0.0 if total_mean == 0 else mean_diff / total_mean
            try:
                mannwhitney_p = float(stats.mannwhitneyu(right, left, alternative="two-sided").pvalue)
            except ValueError:
                mannwhitney_p = 1.0
            welch_p = _safe_welch_p(left, right)
            ci_low, ci_high = _bootstrap_mean_diff_ci(left, right, n_boot=n_boot, seed=hash((str(hemibrain_type), str(cell_type), nt)) % (2**32))
            records.append(
                {
                    "hemibrain_type": hemibrain_type,
                    "cell_type": cell_type,
                    "nt": nt,
                    "left_n": int(len(left)),
                    "right_n": int(len(right)),
                    "left_mean_fraction": float(np.mean(left)),
                    "right_mean_fraction": float(np.mean(right)),
                    "right_minus_left_fraction": mean_diff,
                    "right_laterality_index": laterality,
                    "cohens_d": _cohens_d(left, right),
                    "cliffs_delta": _cliffs_delta(left, right),
                    "mannwhitney_p": mannwhitney_p,
                    "welch_p": welch_p,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "left_median_fraction": float(np.median(left)),
                    "right_median_fraction": float(np.median(right)),
                }
            )
    result = pd.DataFrame.from_records(records)
    if result.empty:
        return result
    result["fdr_q"] = _benjamini_hochberg(result["mannwhitney_p"])
    result["bonferroni_p"] = np.minimum(result["mannwhitney_p"] * len(result), 1.0)
    result["significant_fdr_0_05"] = result["fdr_q"] < 0.05
    result["significant_bonferroni_0_05"] = result["bonferroni_p"] < 0.05
    return result.sort_values(["nt", "hemibrain_type", "cell_type"]).reset_index(drop=True)


def compute_direction_tests(stats_frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    filtered = stats_frame[~stats_frame["hemibrain_type"].astype(str).str.contains(r"KCg-s[123]", regex=True, na=False)]
    for nt, direction in [("ser", "right"), ("glut", "left"), ("gaba", "left"), ("da", "right")]:
        sub = filtered[filtered["nt"] == nt]
        if sub.empty:
            continue
        if direction == "right":
            successes = int((sub["right_laterality_index"] > 0).sum())
        else:
            successes = int((sub["right_laterality_index"] < 0).sum())
        p_value = float(stats.binomtest(successes, n=len(sub), p=0.5, alternative="greater").pvalue)
        records.append(
            {
                "nt": nt,
                "expected_direction": direction,
                "successes": successes,
                "n_subtypes": int(len(sub)),
                "binomial_p": p_value,
            }
        )
    return pd.DataFrame.from_records(records)


def _well_sampled_stats(stats_frame: pd.DataFrame, min_cells_per_side: int = 50) -> pd.DataFrame:
    return stats_frame[
        ~stats_frame["hemibrain_type"].astype(str).str.contains(r"KCg-s[123]", regex=True, na=False)
        & (stats_frame["left_n"].astype(int) >= int(min_cells_per_side))
        & (stats_frame["right_n"].astype(int) >= int(min_cells_per_side))
    ].copy()


def _subtype_key(row: pd.Series) -> tuple[str, str]:
    return str(row["hemibrain_type"]), str(row["cell_type"])


def _ols_side_coefficient(
    frame: pd.DataFrame,
    *,
    fraction_column: str,
    covariate_columns: list[str],
) -> dict[str, float]:
    data = frame[["side", "hemibrain_type", "cell_type", "total_input_synapses", fraction_column, *covariate_columns]].copy()
    data = data[data["side"].isin(["left", "right"])].dropna(subset=[fraction_column])
    if data.empty:
        return {"n_neurons": 0, "side_right_coefficient": np.nan, "side_right_t": np.nan, "side_right_p": np.nan}

    y = data[fraction_column].to_numpy(dtype=float)
    subtype = data[["hemibrain_type", "cell_type"]].fillna("").astype(str).agg("|".join, axis=1)
    design_parts = [
        np.ones((len(data), 1), dtype=float),
        data["side"].eq("right").astype(float).to_numpy()[:, None],
        np.log1p(data["total_input_synapses"].astype(float).to_numpy())[:, None],
    ]
    subtype_dummies = pd.get_dummies(subtype, dtype=float)
    if subtype_dummies.shape[1] > 1:
        design_parts.append(subtype_dummies.iloc[:, 1:].to_numpy(dtype=float))
    for column in covariate_columns:
        values = pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        if finite.sum() < max(20, len(values) // 5):
            continue
        fill_value = float(np.nanmedian(values[finite]))
        values = np.where(finite, values, fill_value)
        scale = float(np.std(values))
        if scale > 1e-12:
            values = (values - float(np.mean(values))) / scale
            design_parts.append(values[:, None])
    x = np.hstack(design_parts)
    if x.shape[0] <= x.shape[1] + 2:
        return {"n_neurons": int(len(data)), "side_right_coefficient": np.nan, "side_right_t": np.nan, "side_right_p": np.nan}
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    dof = max(1, x.shape[0] - x.shape[1])
    sigma2 = float(np.dot(residual, residual) / dof)
    try:
        cov = sigma2 * np.linalg.pinv(x.T @ x)
        se = float(np.sqrt(max(cov[1, 1], 0.0)))
    except np.linalg.LinAlgError:
        se = np.nan
    coef = float(beta[1])
    t_value = float(coef / se) if np.isfinite(se) and se > 0 else np.nan
    p_value = float(2 * stats.t.sf(abs(t_value), dof)) if np.isfinite(t_value) else np.nan
    return {
        "n_neurons": int(len(data)),
        "side_right_coefficient": coef,
        "side_right_t": t_value,
        "side_right_p": p_value,
    }


def compute_kc_nt_artifact_controls(
    neuron_inputs: pd.DataFrame,
    stats_frame: pd.DataFrame,
    *,
    neuron_annotations_path: Path = PROCESSED_DATA_ROOT / "flywire_neuron_annotations.parquet",
    n_permutations: int = 2000,
    min_cells_per_side: int = 50,
    seed: int = 20260610,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run computable artifact controls for the KC NT lateralization result.

    These controls deliberately avoid claiming unavailable fields such as EM
    imaging-tile ID or true homologous-pair IDs.  The matched-neuron proxy pairs
    left/right KCs within subtype by total input size and available soma/position
    coordinates.
    """

    rng = np.random.default_rng(seed)
    sampled = _well_sampled_stats(stats_frame, min_cells_per_side=min_cells_per_side)
    sampled_keys = set(sampled[["hemibrain_type", "cell_type"]].fillna("").astype(str).agg("|".join, axis=1))
    filtered = neuron_inputs.copy()
    filtered["_subtype_key"] = filtered[["hemibrain_type", "cell_type"]].fillna("").astype(str).agg("|".join, axis=1)
    filtered = filtered[filtered["_subtype_key"].isin(sampled_keys) & filtered["side"].isin(["left", "right"])].copy()

    coordinate_columns: list[str] = []
    annotation_columns = ["root_id", "pos_x", "pos_y", "pos_z", "soma_x", "soma_y", "soma_z", "top_nt", "top_nt_conf", "known_nt"]
    try:
        annotations = pd.read_parquet(neuron_annotations_path, columns=annotation_columns)
        filtered = filtered.merge(annotations, on="root_id", how="left", suffixes=("", "_annotation"))
        coordinate_columns = [
            column
            for column in ["soma_x", "soma_y", "soma_z", "pos_x", "pos_y", "pos_z"]
            if column in filtered.columns and pd.to_numeric(filtered[column], errors="coerce").notna().sum() > 0
        ]
    except Exception:
        coordinate_columns = []

    records: list[dict[str, object]] = []
    matched_rows: list[dict[str, object]] = []
    direction_plan = [("ser", "right"), ("glut", "left")]
    for nt, expected_direction in direction_plan:
        fraction_column = f"{nt}_fraction"
        subtype_arrays: list[tuple[str, np.ndarray, np.ndarray, int, int]] = []
        for subtype_key, group in filtered.groupby("_subtype_key"):
            left = group.loc[group["side"].eq("left"), fraction_column].to_numpy(dtype=float)
            right = group.loc[group["side"].eq("right"), fraction_column].to_numpy(dtype=float)
            if len(left) < min_cells_per_side or len(right) < min_cells_per_side:
                continue
            subtype_arrays.append((subtype_key, left, right, len(left), len(right)))
        signed = 1.0 if expected_direction == "right" else -1.0
        observed_deltas = np.array([signed * (float(right.mean()) - float(left.mean())) for _, left, right, _, _ in subtype_arrays])
        observed_successes = int(np.sum(observed_deltas > 0))
        observed_mean_delta = float(observed_deltas.mean()) if observed_deltas.size else np.nan
        null_successes = np.zeros(int(n_permutations), dtype=int)
        null_means = np.zeros(int(n_permutations), dtype=float)
        for i in range(int(n_permutations)):
            perm_deltas: list[float] = []
            for _, left, right, n_left, n_right in subtype_arrays:
                values = np.concatenate([left, right])
                perm = rng.permutation(values)
                perm_left = perm[:n_left]
                perm_right = perm[n_left : n_left + n_right]
                perm_deltas.append(signed * (float(perm_right.mean()) - float(perm_left.mean())))
            perm_array = np.asarray(perm_deltas, dtype=float)
            null_successes[i] = int(np.sum(perm_array > 0))
            null_means[i] = float(perm_array.mean()) if perm_array.size else np.nan
        records.append(
            {
                "control": "within_subtype_side_label_permutation",
                "nt": nt,
                "expected_direction": expected_direction,
                "n_subtypes": int(len(subtype_arrays)),
                "observed_successes": observed_successes,
                "empirical_p_successes": float((1 + np.sum(null_successes >= observed_successes)) / (len(null_successes) + 1)),
                "observed_mean_signed_delta_fraction": observed_mean_delta,
                "null_mean_signed_delta_fraction": float(np.nanmean(null_means)),
                "empirical_p_mean_delta": float((1 + np.sum(null_means >= observed_mean_delta)) / (len(null_means) + 1))
                if np.isfinite(observed_mean_delta)
                else np.nan,
                "n_permutations": int(n_permutations),
            }
        )
        for covariates, label in [([], "subtype_total_input_ols"), (coordinate_columns, "subtype_total_input_spatial_ols")]:
            model = _ols_side_coefficient(filtered, fraction_column=fraction_column, covariate_columns=covariates)
            records.append(
                {
                    "control": label,
                    "nt": nt,
                    "expected_direction": expected_direction,
                    "n_subtypes": int(len(subtype_arrays)),
                    "observed_successes": observed_successes,
                    "empirical_p_successes": np.nan,
                    "observed_mean_signed_delta_fraction": observed_mean_delta,
                    "null_mean_signed_delta_fraction": np.nan,
                    "empirical_p_mean_delta": np.nan,
                    "n_permutations": 0,
                    "n_neurons": model["n_neurons"],
                    "side_right_coefficient": model["side_right_coefficient"],
                    "side_right_t": model["side_right_t"],
                    "side_right_p": model["side_right_p"],
                    "covariates": ",".join(["log_total_input", *covariates]),
                }
            )

        for subtype_key, group in filtered.groupby("_subtype_key"):
            left = group[group["side"].eq("left")].copy()
            right = group[group["side"].eq("right")].copy()
            if len(left) < min_cells_per_side or len(right) < min_cells_per_side:
                continue
            sort_columns = ["total_input_synapses"]
            if "soma_z" in group.columns and pd.to_numeric(group["soma_z"], errors="coerce").notna().sum() > 0:
                sort_columns.append("soma_z")
            left = left.sort_values(sort_columns).reset_index(drop=True)
            right = right.sort_values(sort_columns).reset_index(drop=True)
            n_pair = min(len(left), len(right))
            if n_pair == 0:
                continue
            diff = right.loc[: n_pair - 1, fraction_column].to_numpy(float) - left.loc[: n_pair - 1, fraction_column].to_numpy(float)
            signed_diff = signed * diff
            ci_low, ci_high = _bootstrap_mean_diff_ci(np.zeros_like(signed_diff), signed_diff, n_boot=1000, seed=seed + len(matched_rows) + len(nt))
            positives = int(np.sum(signed_diff > 0))
            sign_n = int(np.sum(signed_diff != 0))
            hemibrain_type, cell_type = subtype_key.split("|", 1)
            matched_rows.append(
                {
                    "nt": nt,
                    "expected_direction": expected_direction,
                    "hemibrain_type": hemibrain_type,
                    "cell_type": cell_type,
                    "n_matched_pairs": int(n_pair),
                    "mean_right_minus_left_fraction": float(diff.mean()),
                    "mean_signed_expected_delta_fraction": float(signed_diff.mean()),
                    "bootstrap_ci_low_signed_delta": ci_low,
                    "bootstrap_ci_high_signed_delta": ci_high,
                    "positive_expected_direction_pairs": positives,
                    "sign_test_p_expected_direction": float(stats.binomtest(positives, sign_n, 0.5, alternative="greater").pvalue)
                    if sign_n
                    else np.nan,
                }
            )

    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(matched_rows)


def compute_high_confidence_nt_sensitivity(
    connections_path: Path,
    kc_annotations: pd.DataFrame,
    base_neuron_inputs: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = (0.2, 0.3, 0.5, 0.7),
    min_cells_per_side: int = 50,
) -> pd.DataFrame:
    kc_ids = set(kc_annotations["root_id"].astype("int64").tolist())
    columns = ["pre_pt_root_id", "post_pt_root_id", "syn_count", *NT_AVG_COLUMNS]
    connections = pd.read_feather(connections_path, columns=columns)
    kc_inputs = connections[connections["post_pt_root_id"].isin(kc_ids)].copy()
    base = base_neuron_inputs[
        ["root_id", "side", "cell_type", "hemibrain_type", "total_input_synapses"]
    ].copy()
    records: list[dict[str, object]] = []
    for threshold in thresholds:
        threshold_inputs = base.copy()
        for nt in NT_COLUMNS:
            threshold_inputs[f"{nt}_input"] = 0.0
        for nt in ["ser", "glut"]:
            avg_col = f"{nt}_avg"
            selected = kc_inputs[kc_inputs[avg_col].astype(float) >= float(threshold)].copy()
            if selected.empty:
                per_neuron = pd.DataFrame({"root_id": [], f"{nt}_input": []})
            else:
                selected[f"{nt}_input"] = selected["syn_count"].astype(float) * selected[avg_col].astype(float)
                per_neuron = (
                    selected.groupby("post_pt_root_id", as_index=False)[f"{nt}_input"]
                    .sum()
                    .rename(columns={"post_pt_root_id": "root_id"})
                )
            threshold_inputs = threshold_inputs.drop(columns=[f"{nt}_input"], errors="ignore").merge(
                per_neuron, on="root_id", how="left"
            )
            threshold_inputs[f"{nt}_input"] = threshold_inputs[f"{nt}_input"].fillna(0.0)
        denominator = threshold_inputs["total_input_synapses"].replace(0, np.nan)
        for nt in NT_COLUMNS:
            input_col = f"{nt}_input"
            threshold_inputs[f"{nt}_fraction"] = (threshold_inputs[input_col] / denominator).fillna(0.0)
        stats_frame = compute_fraction_statistics(threshold_inputs, n_boot=300)
        sampled = _well_sampled_stats(stats_frame, min_cells_per_side=min_cells_per_side)
        for nt, expected_direction in [("ser", "right"), ("glut", "left")]:
            sub = sampled[sampled["nt"].eq(nt)].copy()
            if expected_direction == "right":
                successes = int((sub["right_minus_left_fraction"] > 0).sum())
                signed_delta = sub["right_minus_left_fraction"]
            else:
                successes = int((sub["right_minus_left_fraction"] < 0).sum())
                signed_delta = -sub["right_minus_left_fraction"]
            selected_edges = kc_inputs[kc_inputs[f"{nt}_avg"].astype(float) >= float(threshold)]
            records.append(
                {
                    "threshold": float(threshold),
                    "nt": nt,
                    "expected_direction": expected_direction,
                    "n_subtypes": int(len(sub)),
                    "successes": successes,
                    "binomial_p": float(stats.binomtest(successes, len(sub), 0.5, alternative="greater").pvalue)
                    if len(sub)
                    else np.nan,
                    "fdr_significant_subtypes": int((sub["fdr_q"] < 0.05).sum()),
                    "bonferroni_significant_subtypes": int((sub["bonferroni_p"] < 0.05).sum()),
                    "mean_signed_delta_fraction": float(signed_delta.mean()) if not signed_delta.empty else np.nan,
                    "median_signed_delta_fraction": float(signed_delta.median()) if not signed_delta.empty else np.nan,
                    "selected_edges": int(len(selected_edges)),
                    "selected_synapses": int(selected_edges["syn_count"].sum()) if not selected_edges.empty else 0,
                }
            )
    return pd.DataFrame.from_records(records)


def summarize_high_confidence_upstream_validation(
    connections_path: Path,
    kc_annotations: pd.DataFrame,
    *,
    neuron_annotations_path: Path = PROCESSED_DATA_ROOT / "flywire_neuron_annotations.parquet",
    threshold: float = 0.5,
) -> pd.DataFrame:
    kc_ids = set(kc_annotations["root_id"].astype("int64").tolist())
    annotation_columns = [
        "root_id",
        "side",
        "cell_type",
        "hemibrain_type",
        "super_class",
        "cell_class",
        "top_nt",
        "top_nt_conf",
        "known_nt",
        "known_nt_source",
    ]
    annotations = pd.read_parquet(neuron_annotations_path, columns=annotation_columns)
    columns = ["pre_pt_root_id", "post_pt_root_id", "syn_count", "ser_avg", "glut_avg"]
    connections = pd.read_feather(connections_path, columns=columns)
    kc_inputs = connections[connections["post_pt_root_id"].isin(kc_ids)].copy()
    rows: list[dict[str, object]] = []
    for nt, avg_col, expected_top_nt in [("ser", "ser_avg", "serotonin"), ("glut", "glut_avg", "glutamate")]:
        selected = kc_inputs[kc_inputs[avg_col].astype(float) >= float(threshold)].copy()
        if selected.empty:
            continue
        selected = selected.merge(
            annotations.add_prefix("pre_").rename(columns={"pre_root_id": "pre_pt_root_id"}),
            on="pre_pt_root_id",
            how="left",
        )
        selected = selected.merge(
            kc_annotations[["root_id", "side", "hemibrain_type", "cell_type"]].rename(
                columns={
                    "root_id": "post_pt_root_id",
                    "side": "kc_side",
                    "hemibrain_type": "kc_hemibrain_type",
                    "cell_type": "kc_cell_type",
                }
            ),
            on="post_pt_root_id",
            how="left",
        )
        for (cell_class, cell_type, top_nt, known_nt, kc_side), group in selected.groupby(
            ["pre_cell_class", "pre_cell_type", "pre_top_nt", "pre_known_nt", "kc_side"],
            dropna=False,
        ):
            rows.append(
                {
                    "nt_edge_class": nt,
                    "edge_threshold": float(threshold),
                    "kc_side": kc_side,
                    "pre_cell_class": cell_class,
                    "pre_cell_type": cell_type,
                    "pre_top_nt": top_nt,
                    "pre_known_nt": known_nt,
                    "n_edges": int(len(group)),
                    "syn_count": int(group["syn_count"].sum()),
                    "weighted_nt_mass": float((group["syn_count"].astype(float) * group[avg_col].astype(float)).sum()),
                    "expected_top_nt_match": bool(expected_top_nt in str(top_nt).lower()),
                    "known_nt_match": bool(expected_top_nt in str(known_nt).lower()),
                }
            )
    result = pd.DataFrame.from_records(rows)
    if result.empty:
        return result
    return result.sort_values(["nt_edge_class", "kc_side", "syn_count"], ascending=[True, True, False])


def make_kc_nt_distribution_control_figure(
    neuron_inputs: pd.DataFrame,
    stats_frame: pd.DataFrame,
    controls: pd.DataFrame,
    threshold_sensitivity: pd.DataFrame,
    output_dir: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / "Fig_kc_nt_distribution_controls.png"
    sampled = _well_sampled_stats(stats_frame).copy()
    subtype_order = (
        sampled[sampled["nt"].eq("ser")]
        .sort_values("right_minus_left_fraction")["hemibrain_type"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    plot_neurons = neuron_inputs[
        neuron_inputs["hemibrain_type"].astype(str).isin(subtype_order)
        & neuron_inputs["side"].isin(["left", "right"])
    ].copy()
    plot_neurons["subtype"] = pd.Categorical(plot_neurons["hemibrain_type"].astype(str), categories=subtype_order, ordered=True)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), constrained_layout=True)
    for ax, nt, title in [
        (axes[0, 0], "ser", "5-HT-predicted input fraction per KC"),
        (axes[0, 1], "glut", "Glu-predicted input fraction per KC"),
    ]:
        value_col = f"{nt}_fraction"
        sns.boxplot(
            data=plot_neurons,
            x="subtype",
            y=value_col,
            hue="side",
            ax=ax,
            showfliers=False,
            width=0.72,
            palette={"left": "#7DA7D9", "right": "#E58B7E"},
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("absolute fraction of KC input")
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.legend(frameon=False, title="")

    forest = sampled[sampled["nt"].isin(["ser", "glut"])].copy()
    forest["label"] = forest["hemibrain_type"].astype(str) + " / " + forest["nt"].map({"ser": "5-HT", "glut": "Glu"})
    forest = forest.sort_values(["nt", "right_minus_left_fraction"])
    y = np.arange(len(forest))
    colors = forest["nt"].map({"ser": "#C94C4C", "glut": "#4C78A8"}).to_numpy()
    axes[1, 0].axvline(0, color="#333333", lw=0.8)
    axes[1, 0].errorbar(
        forest["right_minus_left_fraction"],
        y,
        xerr=[
            forest["right_minus_left_fraction"] - forest["bootstrap_ci_low"],
            forest["bootstrap_ci_high"] - forest["right_minus_left_fraction"],
        ],
        fmt="none",
        ecolor="#666666",
        elinewidth=1,
        capsize=2,
    )
    axes[1, 0].scatter(forest["right_minus_left_fraction"], y, c=colors, s=28, zorder=3)
    axes[1, 0].set_yticks(y)
    axes[1, 0].set_yticklabels(forest["label"], fontsize=8)
    axes[1, 0].set_xlabel("right - left absolute input fraction")
    axes[1, 0].set_title("Absolute effect sizes with bootstrap CI")

    control_rows = controls[controls["control"].eq("within_subtype_side_label_permutation")].copy()
    threshold_rows = threshold_sensitivity[threshold_sensitivity["threshold"].eq(0.5)].copy()
    labels: list[str] = []
    values: list[float] = []
    colors2: list[str] = []
    for _, row in control_rows.iterrows():
        labels.append(f"{row['nt']} permutation\np={row['empirical_p_successes']:.3g}")
        values.append(float(row["observed_successes"]) / max(1, int(row["n_subtypes"])))
        colors2.append("#C94C4C" if row["nt"] == "ser" else "#4C78A8")
    for _, row in threshold_rows.iterrows():
        labels.append(f"{row['nt']} high-conf >=0.5\n{int(row['successes'])}/{int(row['n_subtypes'])}")
        values.append(float(row["successes"]) / max(1, int(row["n_subtypes"])))
        colors2.append("#E2A84B" if row["nt"] == "ser" else "#73A66B")
    axes[1, 1].bar(np.arange(len(values)), values, color=colors2)
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_ylabel("fraction of subtypes in expected direction")
    axes[1, 1].set_xticks(np.arange(len(values)), labels, rotation=20, ha="right")
    axes[1, 1].set_title("Permutation and high-confidence NT controls")
    axes[1, 1].spines[["top", "right"]].set_visible(False)

    fig.suptitle("KC neurotransmitter lateralization: magnitude, distribution and artifact controls", fontsize=13)
    fig.savefig(figure_path, dpi=240)
    plt.close(fig)
    return figure_path


def make_nt_figures(stats_frame: pd.DataFrame, upstream: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    filtered = stats_frame[~stats_frame["hemibrain_type"].astype(str).str.contains(r"KCg-s[123]", regex=True, na=False)].copy()

    heatmap_table = filtered.pivot_table(index="hemibrain_type", columns="nt", values="right_laterality_index", aggfunc="mean")
    heatmap_table = heatmap_table[[column for column in ["ser", "glut", "gaba", "da", "ach", "oct"] if column in heatmap_table.columns]]
    heatmap_path = figure_dir / "Fig_NT_lateralization_heatmap.png"
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    sns.heatmap(heatmap_table, center=0, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("KC neurotransmitter input lateralization\npositive = right enriched")
    ax.set_xlabel("Neurotransmitter")
    ax.set_ylabel("KC subtype")
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=240)
    plt.close(fig)

    forest_path = figure_dir / "Fig_serotonin_glutamate_forest.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), sharey=True)
    for ax, nt, title in zip(axes, ["ser", "glut"], ["Serotonin: right enrichment", "Glutamate: left bias"]):
        sub = filtered[filtered["nt"] == nt].sort_values("right_laterality_index")
        y = np.arange(len(sub))
        ax.axvline(0, color="black", lw=0.8)
        ax.errorbar(
            sub["right_minus_left_fraction"],
            y,
            xerr=[
                sub["right_minus_left_fraction"] - sub["bootstrap_ci_low"],
                sub["bootstrap_ci_high"] - sub["right_minus_left_fraction"],
            ],
            fmt="o",
            color="tab:red" if nt == "ser" else "tab:blue",
            ecolor="0.5",
            capsize=2,
        )
        ax.set_title(title)
        ax.set_xlabel("Right - left NT fraction")
        ax.set_yticks(y)
        ax.set_yticklabels(sub["hemibrain_type"])
    fig.tight_layout()
    fig.savefig(forest_path, dpi=240)
    plt.close(fig)

    volcano_path = figure_dir / "Fig_nt_effect_volcano.png"
    plot_data = filtered.copy()
    plot_data["neg_log10_q"] = -np.log10(plot_data["fdr_q"].clip(lower=1e-300))
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=plot_data,
        x="cohens_d",
        y="neg_log10_q",
        hue="nt",
        style="significant_fdr_0_05",
        ax=ax,
        s=60,
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.axhline(-np.log10(0.05), color="0.4", lw=0.8, ls="--")
    ax.set_xlabel("Cohen's d (right - left)")
    ax.set_ylabel("-log10(FDR q)")
    ax.set_title("KC NT input lateralization statistics")
    fig.tight_layout()
    fig.savefig(volcano_path, dpi=240)
    plt.close(fig)

    upstream_path = figure_dir / "Fig_serotonin_upstream_alpha_prime_beta_prime.png"
    upstream_alpha = upstream[
        upstream["kc_hemibrain_type"].astype(str).str.contains("KCa'b'", regex=False)
    ].copy()
    if not upstream_alpha.empty:
        class_summary = (
            upstream_alpha.groupby(["kc_hemibrain_type", "kc_side", "pre_cell_class"], dropna=False)["syn_count"]
            .sum()
            .reset_index()
        )
        class_summary["pre_cell_class"] = class_summary["pre_cell_class"].fillna("unannotated")
        top_classes = class_summary.groupby("pre_cell_class")["syn_count"].sum().nlargest(6).index
        class_summary = class_summary[class_summary["pre_cell_class"].isin(top_classes)]
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=class_summary, x="kc_hemibrain_type", y="syn_count", hue="kc_side", ax=ax)
        ax.set_title("Serotonin-dominant upstream synapses to α′β′ KCs")
        ax.set_xlabel("KC subtype")
        ax.set_ylabel("Synapse count")
        fig.tight_layout()
        fig.savefig(upstream_path, dpi=240)
        plt.close(fig)
    else:
        upstream_path.write_text("No upstream alpha-prime-beta-prime serotonin records found.")

    return {
        "heatmap": heatmap_path,
        "forest": forest_path,
        "volcano": volcano_path,
        "upstream_figure": upstream_path,
    }


def summarize_serotonin_upstream(
    connections_path: Path = RAW_DATA_ROOT / "zenodo_10676866" / "proofread_connections_783.feather",
    neuron_annotations_path: Path = PROCESSED_DATA_ROOT / "flywire_neuron_annotations.parquet",
    kc_annotations: pd.DataFrame | None = None,
    ser_threshold: float = 0.5,
) -> pd.DataFrame:
    kc_annotations = kc_annotations if kc_annotations is not None else load_kc_annotations()
    neuron_annotations = pd.read_parquet(
        neuron_annotations_path,
        columns=["root_id", "side", "cell_type", "hemibrain_type", "super_class", "cell_class", "top_nt"],
    )
    kc_ids = set(kc_annotations["root_id"].astype("int64").tolist())
    columns = ["pre_pt_root_id", "post_pt_root_id", "syn_count", "ser_avg", "da_avg", "gaba_avg", "glut_avg", "ach_avg"]
    connections = pd.read_feather(connections_path, columns=columns)
    ser_inputs = connections[
        connections["post_pt_root_id"].isin(kc_ids) & (connections["ser_avg"] >= ser_threshold)
    ].copy()
    ser_inputs = ser_inputs.merge(
        kc_annotations[["root_id", "side", "cell_type", "hemibrain_type"]].rename(
            columns={
                "root_id": "post_pt_root_id",
                "side": "kc_side",
                "cell_type": "kc_cell_type",
                "hemibrain_type": "kc_hemibrain_type",
            }
        ),
        on="post_pt_root_id",
        how="left",
    )
    ser_inputs = ser_inputs.merge(
        neuron_annotations.rename(
            columns={
                "root_id": "pre_pt_root_id",
                "side": "pre_side",
                "cell_type": "pre_cell_type",
                "hemibrain_type": "pre_hemibrain_type",
                "super_class": "pre_super_class",
                "cell_class": "pre_cell_class",
                "top_nt": "pre_top_nt",
            }
        ),
        on="pre_pt_root_id",
        how="left",
    )
    upstream = (
        ser_inputs.groupby(
            ["kc_hemibrain_type", "kc_cell_type", "kc_side", "pre_cell_class", "pre_cell_type", "pre_top_nt"],
            dropna=False,
        )
        .agg(n_edges=("syn_count", "size"), syn_count=("syn_count", "sum"), mean_ser_avg=("ser_avg", "mean"))
        .reset_index()
        .sort_values(["kc_hemibrain_type", "kc_cell_type", "kc_side", "syn_count"], ascending=[True, True, True, False])
    )
    return upstream


def run_kc_nt_analysis(
    output_dir: Path = DEFAULT_OUTPUT_ROOT / "kc_nt_lateralization",
    connections_path: Path = RAW_DATA_ROOT / "zenodo_10676866" / "proofread_connections_783.feather",
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    kc_annotations = load_kc_annotations()
    neuron_inputs = compute_nt_input_by_neuron(connections_path=connections_path, kc_annotations=kc_annotations)
    summary, effects = summarize_nt_by_subtype(neuron_inputs)
    fraction_stats = compute_fraction_statistics(neuron_inputs)
    direction_tests = compute_direction_tests(fraction_stats)
    upstream = summarize_serotonin_upstream(connections_path=connections_path, kc_annotations=kc_annotations)
    artifact_controls, matched_proxy = compute_kc_nt_artifact_controls(neuron_inputs, fraction_stats)
    threshold_sensitivity = compute_high_confidence_nt_sensitivity(
        connections_path,
        kc_annotations,
        neuron_inputs,
    )
    upstream_validation = summarize_high_confidence_upstream_validation(
        connections_path,
        kc_annotations,
    )

    neuron_inputs_path = output_dir / "kc_neuron_nt_inputs.parquet"
    summary_path = output_dir / "kc_nt_input_by_subtype_side.csv"
    effects_path = output_dir / "kc_nt_lateralization_effects.csv"
    upstream_path = output_dir / "serotonin_dominant_upstream_by_class.csv"
    fraction_stats_path = output_dir / "kc_nt_fraction_stats.csv"
    direction_tests_path = output_dir / "nt_direction_binomial_tests.csv"
    artifact_controls_path = output_dir / "kc_nt_artifact_controls.csv"
    matched_proxy_path = output_dir / "kc_nt_matched_neuron_proxy.csv"
    threshold_sensitivity_path = output_dir / "kc_nt_high_confidence_threshold_sensitivity.csv"
    upstream_validation_path = output_dir / "kc_nt_high_confidence_upstream_validation.csv"

    neuron_inputs.to_parquet(neuron_inputs_path, index=False)
    summary.to_csv(summary_path, index=False)
    effects.to_csv(effects_path, index=False)
    fraction_stats.to_csv(fraction_stats_path, index=False)
    direction_tests.to_csv(direction_tests_path, index=False)
    upstream.to_csv(upstream_path, index=False)
    artifact_controls.to_csv(artifact_controls_path, index=False)
    matched_proxy.to_csv(matched_proxy_path, index=False)
    threshold_sensitivity.to_csv(threshold_sensitivity_path, index=False)
    upstream_validation.to_csv(upstream_validation_path, index=False)
    figure_paths = make_nt_figures(fraction_stats, upstream, output_dir)
    figure_paths["distribution_controls"] = make_kc_nt_distribution_control_figure(
        neuron_inputs,
        fraction_stats,
        artifact_controls,
        threshold_sensitivity,
        output_dir,
    )
    return {
        "neuron_inputs": neuron_inputs_path,
        "summary": summary_path,
        "effects": effects_path,
        "fraction_stats": fraction_stats_path,
        "direction_tests": direction_tests_path,
        "upstream": upstream_path,
        "artifact_controls": artifact_controls_path,
        "matched_proxy": matched_proxy_path,
        "threshold_sensitivity": threshold_sensitivity_path,
        "upstream_validation": upstream_validation_path,
        **figure_paths,
    }
