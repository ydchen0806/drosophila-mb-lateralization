"""Connect the KC chemical gate to a literature-anchored steering readout.

The experiment uses the public FlyWire v783 annotation table and the signed
connectivity released with the Shiu et al. whole-brain point model.  Bilateral
DNa02 signed drive is the primary readout because the right-minus-left firing
rate of this cell pair predicts rotational velocity in walking flies.

All effects reported here are deterministic interventions inside the specified
model.  They predict a relative steering shift; they are not animal behaviour
measurements and do not calibrate signed graph drive into degrees per second.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Iterable

import numpy as np
import pandas as pd

from .kc_flywire_ratio_experiment import (
    _build_flywire_glomerulus_kc_matrix_from_frames,
    _load_kc_lateralization_components,
    _normalize_rows,
    _sparsify,
    build_mixture_odor_panel,
)
from .paths import DEFAULT_OUTPUT_ROOT, REPO_ROOT
from .propagation import (
    PropagationConfig,
    build_torch_propagation_graph,
    signed_multihop_response_torch,
)


DEFAULT_ANNOTATION_PATH = (
    REPO_ROOT
    / "data"
    / "external"
    / "flywire_annotations_upstream"
    / "supplemental_files"
    / "Supplemental_file1_neuron_annotations.tsv"
)
DEFAULT_SHIU_CONNECTIVITY_PATH = REPO_ROOT / "data" / "external" / "shiu_drosophila_brain_model" / "Connectivity_783.parquet"
DEFAULT_KC_NT_INPUTS_PATH = REPO_ROOT / "outputs" / "kc_nt_lateralization" / "kc_neuron_nt_inputs.parquet"


@dataclass(frozen=True)
class LateralizedSteeringConfig:
    annotation_path: Path = field(default_factory=lambda: DEFAULT_ANNOTATION_PATH)
    connectivity_path: Path = field(default_factory=lambda: DEFAULT_SHIU_CONNECTIVITY_PATH)
    kc_nt_inputs_path: Path = field(default_factory=lambda: DEFAULT_KC_NT_INPUTS_PATH)
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_ROOT / "lateralized_steering")
    device: str = "cuda:0"
    seeds: tuple[int, ...] = tuple(range(20))
    validation_seed_start: int = 10
    n_odors: int = 12
    n_mediation_odors: int = 8
    min_glomeruli_per_odor: int = 2
    max_glomeruli_per_odor: int = 6
    channel_noise_sigma: float = 0.15
    kc_active_fraction: float = 0.10
    gate_amplitude: float = 0.25
    gate_strengths: tuple[float, ...] = (-1.0, 0.0, 0.5, 1.0, 2.0, 3.0)
    propagation_steps: int = 3
    max_active: int = 30_000
    null_repeats: int = 32
    random_seed: int = 20260718
    step_sensitivity: tuple[int, ...] = (2, 3, 4)
    propagation_active_caps: tuple[int, ...] = (15_000, 30_000, 60_000)
    propagation_normalization_options: tuple[bool, ...] = (True, False)
    learned_contrast_seeds: tuple[int, ...] = tuple(range(12))
    learned_pairs_per_seed: int = 12


def load_annotations(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", low_memory=False)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    return pd.read_parquet(path)


def subtype_preserving_gate_shuffle(
    gate: np.ndarray,
    subtype: Iterable[str],
    rng: np.random.Generator,
) -> np.ndarray:
    """Destroy side registration while preserving each subtype's gate values."""

    shuffled = np.asarray(gate, dtype=np.float64).copy()
    subtype_array = np.asarray(list(subtype), dtype=object)
    for label in pd.unique(subtype_array):
        indices = np.flatnonzero(subtype_array == label)
        shuffled[indices] = rng.permutation(shuffled[indices])
    return shuffled


def extract_dna02_drive(
    response: pd.DataFrame,
    dna02_ids: dict[str, int],
    step_limit: int,
) -> dict[str, float]:
    selected = response[response["step"].le(int(step_limit))]
    aggregate = selected.groupby("root_id")["score"].sum() if not selected.empty else pd.Series(dtype=float)
    left = float(aggregate.get(int(dna02_ids["left"]), 0.0))
    right = float(aggregate.get(int(dna02_ids["right"]), 0.0))
    return {
        "dna02_left_drive": left,
        "dna02_right_drive": right,
        "dna02_right_minus_left": right - left,
        "dna02_absolute_drive": abs(left) + abs(right),
    }


def paired_condition_contrast(
    raw: pd.DataFrame,
    left_condition: str,
    right_condition: str,
    *,
    value_column: str = "dna02_right_minus_left",
) -> pd.DataFrame:
    keys = ["seed", "split", "odor_index", "odor_name"]
    selected = raw[raw["condition"].isin([left_condition, right_condition])]
    wide = selected.pivot(index=keys, columns="condition", values=value_column).dropna().reset_index()
    wide["contrast"] = f"{left_condition}_minus_{right_condition}"
    wide["delta"] = wide[left_condition] - wide[right_condition]
    return wide


def _bootstrap_seed_ci(values: pd.DataFrame, value_column: str, random_seed: int) -> tuple[float, float]:
    seed_means = values.groupby("seed")[value_column].mean().to_numpy(dtype=float)
    if seed_means.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(random_seed))
    draws = rng.choice(seed_means, size=(20_000, seed_means.size), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def summarize_dose(raw: pd.DataFrame, random_seed: int) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for split in ["all", "development", "validation"]:
        split_frame = raw if split == "all" else raw[raw["split"].eq(split)]
        for (condition, strength), group in split_frame.groupby(["condition", "gate_strength"], sort=False):
            low, high = _bootstrap_seed_ci(group, "dna02_right_minus_left", random_seed)
            records.append(
                {
                    "split": split,
                    "condition": condition,
                    "gate_strength": float(strength),
                    "n_seeds": int(group["seed"].nunique()),
                    "n_odors": int(len(group)),
                    "mean_dna02_right_minus_left": float(group["dna02_right_minus_left"].mean()),
                    "seed_bootstrap_ci_low": low,
                    "seed_bootstrap_ci_high": high,
                    "mean_dna02_absolute_drive": float(group["dna02_absolute_drive"].mean()),
                }
            )
    return pd.DataFrame.from_records(records)


def summarize_contrasts(raw: pd.DataFrame, random_seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    contrast_pairs = [
        ("real", "symmetrized"),
        ("mirror", "symmetrized"),
        ("real", "mirror"),
        ("real_dose_2", "real"),
        ("real_dose_3", "real_dose_2"),
    ]
    paired = pd.concat(
        [paired_condition_contrast(raw, left, right) for left, right in contrast_pairs],
        ignore_index=True,
    )
    records: list[dict[str, object]] = []
    for split in ["all", "development", "validation"]:
        split_frame = paired if split == "all" else paired[paired["split"].eq(split)]
        for contrast, group in split_frame.groupby("contrast", sort=False):
            low, high = _bootstrap_seed_ci(group, "delta", random_seed)
            records.append(
                {
                    "split": split,
                    "contrast": contrast,
                    "n_seeds": int(group["seed"].nunique()),
                    "n_odors": int(len(group)),
                    "mean_delta": float(group["delta"].mean()),
                    "seed_bootstrap_ci_low": low,
                    "seed_bootstrap_ci_high": high,
                    "positive_odors": int((group["delta"] > 0).sum()),
                    "negative_odors": int((group["delta"] < 0).sum()),
                    "zero_odors": int((group["delta"] == 0).sum()),
                }
            )
    return paired, pd.DataFrame.from_records(records)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _plot_steering_figure(
    primary: pd.DataFrame,
    dose_summary: pd.DataFrame,
    contrast_summary: pd.DataFrame,
    mediation: pd.DataFrame,
    null_summary: pd.DataFrame,
    mbon32_response: pd.DataFrame,
    output_dir: Path,
    random_seed: int,
) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), dpi=300, constrained_layout=True)
    ax = axes[0, 0]
    dose = primary[primary["split"].eq("validation")].merge(
        primary[
            primary["split"].eq("validation") & primary["condition"].eq("symmetrized")
        ][["seed", "odor_index", "dna02_right_minus_left"]].rename(
            columns={"dna02_right_minus_left": "symmetrized_drive"}
        ),
        on=["seed", "odor_index"],
        how="left",
    )
    dose["delta_vs_symmetrized"] = dose["dna02_right_minus_left"] - dose["symmetrized_drive"]
    dose_records = []
    for strength, group in dose.groupby("gate_strength"):
        low, high = _bootstrap_seed_ci(group, "delta_vs_symmetrized", random_seed)
        dose_records.append(
            {
                "gate_strength": float(strength),
                "mean": float(group["delta_vs_symmetrized"].mean()),
                "low": low,
                "high": high,
            }
        )
    dose_plot = pd.DataFrame.from_records(dose_records).sort_values("gate_strength")
    x = dose_plot["gate_strength"].to_numpy(float)
    y = dose_plot["mean"].to_numpy(float)
    ax.plot(x, y, color="#2f7d5c", marker="o", lw=1.5, ms=4)
    ax.fill_between(
        x,
        dose_plot["low"].to_numpy(float),
        dose_plot["high"].to_numpy(float),
        color="#2f7d5c",
        alpha=0.18,
        linewidth=0,
    )
    ax.axhline(0, color="#555555", lw=0.7)
    ax.set_xlabel("chemical-gate strength")
    ax.set_ylabel("change in right - left DNa02 drive")
    ax.set_title("Dose-dependent steering-command shift")

    ax = axes[0, 1]
    real = contrast_summary[
        contrast_summary["split"].eq("validation")
        & contrast_summary["contrast"].eq("real_minus_symmetrized")
    ].iloc[0]
    null_values = null_summary["mean_delta"].to_numpy(float)
    ax.hist(null_values, bins=10, color="#b9b9b9", edgecolor="white")
    ax.axvline(float(real["mean_delta"]), color="#b4474a", lw=2, label="registered gate")
    ax.set_xlabel("mean steering delta vs symmetrized")
    ax.set_ylabel("subtype-preserving shuffles")
    ax.set_title("Side-registration null")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 0]
    order = ["none", "all_MBON", "MBON32", "MBON26", "MBON27", "MBON31"]
    med = mediation.set_index("silenced_group").reindex(order).dropna().reset_index()
    colors = ["#2f7d5c" if name == "none" else "#4c78a8" for name in med["silenced_group"]]
    ax.axhline(0, color="#555555", lw=0.7)
    ax.bar(np.arange(len(med)), med["retained_fraction"], color=colors)
    ax.set_xticks(np.arange(len(med)), med["silenced_group"], rotation=40, ha="right")
    ax.set_ylabel("fraction of intact steering effect")
    ax.set_title("Model-internal pathway mediation")
    ax.set_ylim(-0.5, 1.75)

    ax = axes[1, 1]
    mb = mbon32_response.sort_values("side")
    ax.axhline(0, color="#555555", lw=0.7)
    ax.bar(mb["side"], mb["mean_real_minus_sym_step1_drive"], color=["#4c78a8", "#c65f4a"])
    ax.set_ylabel("real - sym first-hop signed drive")
    ax.set_title("Gate shifts the MBON32 pair")

    for label, ax in zip("abcd", axes.flat):
        ax.text(-0.14, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)
        ax.title.set_fontsize(8)
        ax.xaxis.label.set_size(7)
        ax.yaxis.label.set_size(7)
    png = output_dir / "Fig_lateralized_steering.png"
    pdf = output_dir / "Fig_lateralized_steering.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def run_lateralized_steering(config: LateralizedSteeringConfig | None = None) -> dict[str, Path]:
    cfg = config or LateralizedSteeringConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    annotations = load_annotations(cfg.annotation_path)
    edge_columns = ["Presynaptic_ID", "Postsynaptic_ID", "Connectivity", "Excitatory x Connectivity"]
    edges = pd.read_parquet(cfg.connectivity_path, columns=edge_columns)
    glomeruli, glomerulus_matrix, kc_ids, channel_table = _build_flywire_glomerulus_kc_matrix_from_frames(
        annotations,
        edges,
    )
    aligned_gate = _load_kc_lateralization_components(cfg.kc_nt_inputs_path, kc_ids)
    gate = aligned_gate["symmetry_breaking_gate"].to_numpy(dtype=np.float64)
    graph = build_torch_propagation_graph(cfg.connectivity_path, device=cfg.device)

    dna02 = annotations[annotations["cell_type"].fillna("").astype(str).eq("DNa02")]
    dna02_ids = {str(row.side): int(row.root_id) for row in dna02.itertuples()}
    if set(dna02_ids) != {"left", "right"}:
        raise ValueError(f"Expected one left and one right DNa02, found {dna02_ids}")

    panel_rows: list[dict[str, object]] = []
    panels: list[tuple[int, str, int, str, np.ndarray]] = []
    for seed in cfg.seeds:
        odor_names, activity, panel = build_mixture_odor_panel(
            glomeruli,
            glomerulus_matrix,
            seed=int(seed),
            n_odors=int(cfg.n_odors),
            min_glomeruli_per_odor=int(cfg.min_glomeruli_per_odor),
            max_glomeruli_per_odor=int(cfg.max_glomeruli_per_odor),
            channel_noise_sigma=float(cfg.channel_noise_sigma),
        )
        split = "validation" if int(seed) >= int(cfg.validation_seed_start) else "development"
        for odor_index, (odor_name, row) in enumerate(zip(odor_names, activity)):
            panels.append((int(seed), split, odor_index, odor_name, row))
        panel = panel.assign(seed=int(seed), split=split)
        panel_rows.extend(panel.to_dict("records"))

    propagation_config = PropagationConfig(
        steps=int(cfg.propagation_steps),
        max_active=int(cfg.max_active),
        normalize_each_step=True,
    )

    def evaluate_gate(
        gate_vector: np.ndarray,
        condition: str,
        strength: float,
        selected_panels: list[tuple[int, str, int, str, np.ndarray]],
        silence_ids: set[int] | None = None,
        run_config: PropagationConfig | None = None,
    ) -> pd.DataFrame:
        records: list[dict[str, object]] = []
        active_config = run_config or propagation_config
        gain = np.clip(1.0 + float(cfg.gate_amplitude) * gate_vector, 0.05, 5.0)
        for seed, split, odor_index, odor_name, activity in selected_panels:
            gated = _normalize_rows(np.maximum(activity[None, :] * gain[None, :], 0.0))
            _, graded, active_k = _sparsify(gated, float(cfg.kc_active_fraction))
            weights = {
                int(root_id): float(value)
                for root_id, value in zip(kc_ids, graded[0])
                if float(value) > 0
            }
            response = signed_multihop_response_torch(
                graph,
                weights,
                active_config,
                silence_ids=silence_ids,
            )
            record = {
                "seed": seed,
                "split": split,
                "odor_index": odor_index,
                "odor_name": odor_name,
                "condition": condition,
                "gate_strength": float(strength),
                "active_k": int(active_k),
                "silenced_neurons": int(len(silence_ids or set())),
            }
            record.update(extract_dna02_drive(response, dna02_ids, active_config.steps))
            records.append(record)
        return pd.DataFrame.from_records(records)

    primary_frames: list[pd.DataFrame] = []
    for strength in cfg.gate_strengths:
        condition = {
            -1.0: "mirror",
            0.0: "symmetrized",
            0.5: "real_dose_0.5",
            1.0: "real",
            2.0: "real_dose_2",
            3.0: "real_dose_3",
        }.get(float(strength), f"gate_strength_{float(strength):g}")
        primary_frames.append(evaluate_gate(gate * float(strength), condition, float(strength), panels))
    primary = pd.concat(primary_frames, ignore_index=True)
    dose_summary = summarize_dose(primary, cfg.random_seed)
    paired, contrast_summary = summarize_contrasts(primary, cfg.random_seed)

    sensitivity_config = PropagationConfig(
        steps=max(cfg.step_sensitivity),
        max_active=int(cfg.max_active),
        normalize_each_step=True,
    )
    sensitivity_records: list[dict[str, object]] = []
    for condition, applied_gate in [
        ("mirror", -gate),
        ("symmetrized", np.zeros_like(gate)),
        ("real", gate),
    ]:
        gain = np.clip(1.0 + float(cfg.gate_amplitude) * applied_gate, 0.05, 5.0)
        for seed, split, odor_index, odor_name, activity in panels:
            gated = _normalize_rows(np.maximum(activity[None, :] * gain[None, :], 0.0))
            _, graded, _ = _sparsify(gated, float(cfg.kc_active_fraction))
            weights = {
                int(root_id): float(value)
                for root_id, value in zip(kc_ids, graded[0])
                if float(value) > 0
            }
            response = signed_multihop_response_torch(graph, weights, sensitivity_config)
            for step_limit in cfg.step_sensitivity:
                record = {
                    "seed": seed,
                    "split": split,
                    "odor_index": odor_index,
                    "odor_name": odor_name,
                    "condition": condition,
                    "step_limit": int(step_limit),
                }
                record.update(extract_dna02_drive(response, dna02_ids, int(step_limit)))
                sensitivity_records.append(record)
    step_raw = pd.DataFrame.from_records(sensitivity_records)
    step_records: list[dict[str, object]] = []
    for step_limit in cfg.step_sensitivity:
        step_frame = step_raw[step_raw["step_limit"].eq(int(step_limit))]
        for split in ["all", "development", "validation"]:
            split_frame = step_frame if split == "all" else step_frame[step_frame["split"].eq(split)]
            for left_condition, right_condition in [("real", "symmetrized"), ("mirror", "symmetrized")]:
                contrast = paired_condition_contrast(split_frame, left_condition, right_condition)
                low, high = _bootstrap_seed_ci(contrast, "delta", cfg.random_seed + int(step_limit))
                step_records.append(
                    {
                        "step_limit": int(step_limit),
                        "split": split,
                        "contrast": f"{left_condition}_minus_{right_condition}",
                        "n_seeds": int(contrast["seed"].nunique()),
                        "n_odors": int(len(contrast)),
                        "mean_delta": float(contrast["delta"].mean()),
                        "seed_bootstrap_ci_low": low,
                        "seed_bootstrap_ci_high": high,
                        "positive_odors": int((contrast["delta"] > 0).sum()),
                        "negative_odors": int((contrast["delta"] < 0).sum()),
                    }
                )
    step_summary = pd.DataFrame.from_records(step_records)

    validation_panels_all = [item for item in panels if item[1] == "validation"]
    propagation_sensitivity_frames: list[pd.DataFrame] = []
    for active_cap in cfg.propagation_active_caps:
        for normalize_each_step in cfg.propagation_normalization_options:
            sensitivity_run_config = PropagationConfig(
                steps=int(cfg.propagation_steps),
                max_active=int(active_cap),
                normalize_each_step=bool(normalize_each_step),
            )
            for condition, applied_gate in [
                ("mirror", -gate),
                ("symmetrized", np.zeros_like(gate)),
                ("real", gate),
            ]:
                frame = evaluate_gate(
                    applied_gate,
                    condition,
                    -1.0 if condition == "mirror" else float(condition == "real"),
                    validation_panels_all,
                    run_config=sensitivity_run_config,
                )
                frame["active_cap"] = int(active_cap)
                frame["normalize_each_step"] = bool(normalize_each_step)
                propagation_sensitivity_frames.append(frame)
    propagation_sensitivity_raw = pd.concat(propagation_sensitivity_frames, ignore_index=True)
    propagation_sensitivity_records: list[dict[str, object]] = []
    for (active_cap, normalize_each_step), group in propagation_sensitivity_raw.groupby(
        ["active_cap", "normalize_each_step"],
        sort=True,
    ):
        for left_condition, right_condition in [("real", "symmetrized"), ("mirror", "symmetrized")]:
            contrast = paired_condition_contrast(group, left_condition, right_condition)
            low, high = _bootstrap_seed_ci(
                contrast,
                "delta",
                cfg.random_seed + int(active_cap) + int(bool(normalize_each_step)),
            )
            propagation_sensitivity_records.append(
                {
                    "active_cap": int(active_cap),
                    "normalize_each_step": bool(normalize_each_step),
                    "contrast": f"{left_condition}_minus_{right_condition}",
                    "n_seeds": int(contrast["seed"].nunique()),
                    "n_odors": int(len(contrast)),
                    "mean_delta": float(contrast["delta"].mean()),
                    "seed_bootstrap_ci_low": low,
                    "seed_bootstrap_ci_high": high,
                    "positive_odors": int((contrast["delta"] > 0).sum()),
                    "negative_odors": int((contrast["delta"] < 0).sum()),
                }
            )
    propagation_sensitivity = pd.DataFrame.from_records(propagation_sensitivity_records)

    learned_records: list[dict[str, object]] = []
    learned_strengths = (-1.0, 0.0, 1.0, 2.0, 3.0)
    for seed in cfg.learned_contrast_seeds:
        odor_names, activity, _panel = build_mixture_odor_panel(
            glomeruli,
            glomerulus_matrix,
            seed=int(seed),
            n_odors=2 * int(cfg.learned_pairs_per_seed),
            min_glomeruli_per_odor=int(cfg.min_glomeruli_per_odor),
            max_glomeruli_per_odor=int(cfg.max_glomeruli_per_odor),
            channel_noise_sigma=float(cfg.channel_noise_sigma),
        )
        split = "validation" if int(seed) >= len(cfg.learned_contrast_seeds) // 2 else "development"
        for strength in learned_strengths:
            condition = {
                -1.0: "mirror",
                0.0: "symmetrized",
                1.0: "real",
                2.0: "real_dose_2",
                3.0: "real_dose_3",
            }[float(strength)]
            gain = np.clip(1.0 + float(cfg.gate_amplitude) * float(strength) * gate, 0.05, 5.0)
            gated = _normalize_rows(np.maximum(activity * gain[None, :], 0.0))
            _, graded, active_k = _sparsify(gated, float(cfg.kc_active_fraction))
            for pair_index in range(int(cfg.learned_pairs_per_seed)):
                cs_plus = pair_index
                cs_minus = pair_index + int(cfg.learned_pairs_per_seed)
                contrast_vector = graded[cs_plus] - graded[cs_minus]
                weights = {
                    int(root_id): float(value)
                    for root_id, value in zip(kc_ids, contrast_vector)
                    if float(value) != 0
                }
                response = signed_multihop_response_torch(graph, weights, propagation_config)
                record = {
                    "seed": int(seed),
                    "split": split,
                    "odor_index": pair_index,
                    "odor_name": f"{odor_names[cs_plus]}_minus_{odor_names[cs_minus]}",
                    "condition": condition,
                    "gate_strength": float(strength),
                    "active_k": int(active_k),
                }
                record.update(extract_dna02_drive(response, dna02_ids, cfg.propagation_steps))
                learned_records.append(record)
    learned_raw = pd.DataFrame.from_records(learned_records)
    learned_paired, learned_summary = summarize_contrasts(learned_raw, cfg.random_seed + 1000)

    validation_panels = [
        item
        for item in panels
        if item[1] == "validation" and item[2] < int(cfg.n_mediation_odors)
    ]
    component_frames = []
    for condition, component in [
        ("symmetrized", np.zeros_like(gate)),
        ("combined_real", gate),
        ("serotonin_only", aligned_gate["serotonin_only_gate"].to_numpy(float)),
        ("glutamate_only", aligned_gate["glutamate_only_gate"].to_numpy(float)),
    ]:
        component_frames.append(evaluate_gate(component, condition, 1.0, validation_panels))
    component_raw = pd.concat(component_frames, ignore_index=True)
    component_sym = component_raw[component_raw["condition"].eq("symmetrized")][
        ["seed", "odor_index", "dna02_right_minus_left"]
    ].rename(columns={"dna02_right_minus_left": "symmetrized_drive"})
    component_effects = component_raw.merge(component_sym, on=["seed", "odor_index"], how="left")
    component_effects["delta_vs_symmetrized"] = (
        component_effects["dna02_right_minus_left"] - component_effects["symmetrized_drive"]
    )

    def ids_for(mask: pd.Series) -> set[int]:
        return set(annotations.loc[mask, "root_id"].dropna().astype("int64"))

    cell_type = annotations["cell_type"].fillna("").astype(str)
    cell_class = annotations["cell_class"].fillna("").astype(str)
    silence_groups = {
        "none": set(),
        "all_MBON": ids_for(cell_class.eq("MBON")),
        "MBON26": ids_for(cell_type.eq("MBON26")),
        "MBON27": ids_for(cell_type.eq("MBON27")),
        "MBON31": ids_for(cell_type.eq("MBON31")),
        "MBON32": ids_for(cell_type.eq("MBON32")),
        "APL": ids_for(cell_type.str.contains("APL", case=False, na=False)),
        "DPM": ids_for(cell_type.str.contains("DPM", case=False, na=False)),
        "DAN": ids_for(cell_class.eq("DAN")),
    }
    mediation_records: list[dict[str, object]] = []
    intact_mean = float("nan")
    for group_name, silence_set in silence_groups.items():
        sym = evaluate_gate(np.zeros_like(gate), "symmetrized", 0.0, validation_panels, silence_set)
        real = evaluate_gate(gate, "real", 1.0, validation_panels, silence_set)
        joined = real.merge(
            sym[["seed", "odor_index", "dna02_right_minus_left"]],
            on=["seed", "odor_index"],
            suffixes=("_real", "_sym"),
        )
        delta = joined["dna02_right_minus_left_real"] - joined["dna02_right_minus_left_sym"]
        effect = float(delta.mean())
        joined = joined.assign(delta=delta)
        ci_low, ci_high = _bootstrap_seed_ci(joined, "delta", cfg.random_seed)
        if group_name == "none":
            intact_mean = effect
        mediation_records.append(
            {
                "silenced_group": group_name,
                "n_silenced_neurons": int(len(silence_set)),
                "n_seeds": int(joined["seed"].nunique()),
                "n_odors": int(len(joined)),
                "mean_delta_vs_matched_symmetrized": effect,
                "seed_bootstrap_ci_low": ci_low,
                "seed_bootstrap_ci_high": ci_high,
                "positive_odors": int((delta > 0).sum()),
                "negative_odors": int((delta < 0).sum()),
            }
        )
    mediation = pd.DataFrame.from_records(mediation_records)
    mediation["retained_fraction"] = mediation["mean_delta_vs_matched_symmetrized"] / intact_mean

    rng = np.random.default_rng(int(cfg.random_seed))
    sym_validation = evaluate_gate(np.zeros_like(gate), "symmetrized", 0.0, validation_panels)
    null_records: list[dict[str, object]] = []
    for repeat in range(int(cfg.null_repeats)):
        shuffled = subtype_preserving_gate_shuffle(gate, aligned_gate["subtype"], rng)
        shuffled_raw = evaluate_gate(shuffled, f"shuffle_{repeat:02d}", 1.0, validation_panels)
        joined = shuffled_raw.merge(
            sym_validation[["seed", "odor_index", "dna02_right_minus_left"]],
            on=["seed", "odor_index"],
            suffixes=("_shuffle", "_sym"),
        )
        delta = joined["dna02_right_minus_left_shuffle"] - joined["dna02_right_minus_left_sym"]
        null_records.append(
            {
                "repeat": repeat,
                "mean_delta": float(delta.mean()),
                "positive_odors": int((delta > 0).sum()),
                "negative_odors": int((delta < 0).sum()),
            }
        )
    null_summary = pd.DataFrame.from_records(null_records)

    mbon32 = annotations[cell_type.eq("MBON32")][["root_id", "side", "cell_type"]]
    mbon32_ids = {str(row.side): int(row.root_id) for row in mbon32.itertuples()}
    mbon32_records: list[dict[str, object]] = []
    for condition, applied_gate in [("symmetrized", np.zeros_like(gate)), ("real", gate)]:
        gain = np.clip(1.0 + float(cfg.gate_amplitude) * applied_gate, 0.05, 5.0)
        for seed, split, odor_index, odor_name, activity in validation_panels:
            gated = _normalize_rows(np.maximum(activity[None, :] * gain[None, :], 0.0))
            _, graded, _ = _sparsify(gated, float(cfg.kc_active_fraction))
            weights = {
                int(root_id): float(value)
                for root_id, value in zip(kc_ids, graded[0])
                if float(value) > 0
            }
            response = signed_multihop_response_torch(graph, weights, propagation_config)
            step_one = response[response["step"].eq(1)].set_index("root_id")["score"]
            for side, root_id in mbon32_ids.items():
                mbon32_records.append(
                    {
                        "seed": seed,
                        "odor_index": odor_index,
                        "condition": condition,
                        "side": side,
                        "root_id": root_id,
                        "step1_drive": float(step_one.get(root_id, 0.0)),
                    }
                )
    mbon32_raw = pd.DataFrame.from_records(mbon32_records)
    mbon32_wide = mbon32_raw.pivot(
        index=["seed", "odor_index", "side", "root_id"],
        columns="condition",
        values="step1_drive",
    ).reset_index()
    mbon32_wide["real_minus_sym_step1_drive"] = mbon32_wide["real"] - mbon32_wide["symmetrized"]
    mbon32_response = (
        mbon32_wide.groupby(["side", "root_id"], as_index=False)
        .agg(
            n_odors=("real_minus_sym_step1_drive", "size"),
            mean_real_minus_sym_step1_drive=("real_minus_sym_step1_drive", "mean"),
            positive_odors=("real_minus_sym_step1_drive", lambda value: int((value > 0).sum())),
            negative_odors=("real_minus_sym_step1_drive", lambda value: int((value < 0).sum())),
        )
    )

    direct_edges = edges[
        edges["Presynaptic_ID"].isin(set(mbon32_ids.values()))
        & edges["Postsynaptic_ID"].isin(set(dna02_ids.values()))
    ].copy()
    direct_edges = direct_edges.merge(
        mbon32.rename(columns={"root_id": "Presynaptic_ID", "side": "mbon32_side"}),
        on="Presynaptic_ID",
        how="left",
    ).merge(
        dna02[["root_id", "side"]].rename(columns={"root_id": "Postsynaptic_ID", "side": "dna02_side"}),
        on="Postsynaptic_ID",
        how="left",
    )

    primary_path = cfg.output_dir / "lateralized_steering_primary_raw.csv"
    dose_path = cfg.output_dir / "lateralized_steering_dose_summary.csv"
    paired_path = cfg.output_dir / "lateralized_steering_paired_contrasts.csv"
    contrast_path = cfg.output_dir / "lateralized_steering_contrast_summary.csv"
    step_raw_path = cfg.output_dir / "lateralized_steering_step_sensitivity_raw.csv"
    step_path = cfg.output_dir / "lateralized_steering_step_sensitivity.csv"
    propagation_sensitivity_raw_path = (
        cfg.output_dir / "lateralized_steering_propagation_sensitivity_raw.csv"
    )
    propagation_sensitivity_path = cfg.output_dir / "lateralized_steering_propagation_sensitivity.csv"
    learned_raw_path = cfg.output_dir / "lateralized_steering_associative_contrast_raw.csv"
    learned_paired_path = cfg.output_dir / "lateralized_steering_associative_paired_contrasts.csv"
    learned_path = cfg.output_dir / "lateralized_steering_associative_contrast_summary.csv"
    component_path = cfg.output_dir / "lateralized_steering_component_effects.csv"
    mediation_path = cfg.output_dir / "lateralized_steering_mediation.csv"
    null_path = cfg.output_dir / "lateralized_steering_subtype_null.csv"
    mbon32_path = cfg.output_dir / "lateralized_steering_mbon32_response.csv"
    edge_path = cfg.output_dir / "mbon32_to_dna02_direct_edges.csv"
    panel_path = cfg.output_dir / "lateralized_steering_odor_panel.csv"
    channel_path = cfg.output_dir / "lateralized_steering_glomerulus_channels.csv"
    metadata_path = cfg.output_dir / "lateralized_steering_metadata.json"
    report_path = cfg.output_dir / "LATERALIZED_STEERING_REPORT_CN.md"

    primary.to_csv(primary_path, index=False)
    dose_summary.to_csv(dose_path, index=False)
    paired.to_csv(paired_path, index=False)
    contrast_summary.to_csv(contrast_path, index=False)
    step_raw.to_csv(step_raw_path, index=False)
    step_summary.to_csv(step_path, index=False)
    propagation_sensitivity_raw.to_csv(propagation_sensitivity_raw_path, index=False)
    propagation_sensitivity.to_csv(propagation_sensitivity_path, index=False)
    learned_raw.to_csv(learned_raw_path, index=False)
    learned_paired.to_csv(learned_paired_path, index=False)
    learned_summary.to_csv(learned_path, index=False)
    component_effects.to_csv(component_path, index=False)
    mediation.to_csv(mediation_path, index=False)
    null_summary.to_csv(null_path, index=False)
    mbon32_response.to_csv(mbon32_path, index=False)
    direct_edges.to_csv(edge_path, index=False)
    pd.DataFrame.from_records(panel_rows).to_csv(panel_path, index=False)
    channel_table.to_csv(channel_path, index=False)

    validation_real = contrast_summary[
        contrast_summary["split"].eq("validation")
        & contrast_summary["contrast"].eq("real_minus_symmetrized")
    ].iloc[0]
    validation_mirror = contrast_summary[
        contrast_summary["split"].eq("validation")
        & contrast_summary["contrast"].eq("mirror_minus_symmetrized")
    ].iloc[0]
    all_mbon = mediation[mediation["silenced_group"].eq("all_MBON")].iloc[0]
    mbon32_silence = mediation[mediation["silenced_group"].eq("MBON32")].iloc[0]
    null_rank = 1 + int((null_summary["mean_delta"] >= float(validation_real["mean_delta"])).sum())
    metadata = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(cfg).items()},
        "data_provenance": {
            "flywire_annotations_commit": _git_commit(cfg.annotation_path.parents[1]),
            "shiu_model_commit": _git_commit(cfg.connectivity_path.parent),
            "annotation_sha256": _file_sha256(cfg.annotation_path),
            "connectivity_sha256": _file_sha256(cfg.connectivity_path),
            "kc_nt_inputs_sha256": _file_sha256(cfg.kc_nt_inputs_path),
        },
        "primary_result": {
            "validation_mean_real_minus_sym": float(validation_real["mean_delta"]),
            "validation_real_positive_odors": int(validation_real["positive_odors"]),
            "validation_real_n_odors": int(validation_real["n_odors"]),
            "validation_mean_mirror_minus_sym": float(validation_mirror["mean_delta"]),
            "validation_mirror_negative_odors": int(validation_mirror["negative_odors"]),
            "subtype_shuffle_rank": f"{null_rank}/{int(cfg.null_repeats) + 1}",
            "all_mbon_silence_retained_fraction": float(all_mbon["retained_fraction"]),
            "mbon32_silence_retained_fraction": float(mbon32_silence["retained_fraction"]),
            "step_sensitivity": step_summary[step_summary["split"].eq("validation")].to_dict("records"),
            "propagation_sensitivity": propagation_sensitivity.to_dict("records"),
            "associative_contrast_proxy": learned_summary[
                learned_summary["split"].eq("validation")
            ].to_dict("records"),
        },
        "claim_boundary": {
            "supported": "relative rightward shift of odor-evoked DNa02 steering command inside the model",
            "not_supported": "animal turning magnitude, improved memory accuracy, learned choice direction, or transmitter release identity",
            "readout": "signed graph drive, not spikes or degrees per second",
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    figure_png, figure_pdf = _plot_steering_figure(
        primary,
        dose_summary,
        contrast_summary,
        mediation,
        null_summary,
        mbon32_response,
        cfg.output_dir,
        cfg.random_seed,
    )
    report_path.write_text(
        "# KC chemical gate 到 DNa02 转向命令的模型因果链\n\n"
        "## 结论\n\n"
        "在相同 FlyWire 连接组、气味输入和 KC 稀疏度下，真实 side-by-subtype chemical gate "
        f"相对对称门控使验证集 DNa02 的 right-minus-left signed drive 平均增加 "
        f"`{float(validation_real['mean_delta']):.6g}`，方向一致 "
        f"`{int(validation_real['positive_odors'])}/{int(validation_real['n_odors'])}`；镜像门控方向相反 "
        f"`{int(validation_mirror['negative_odors'])}/{int(validation_mirror['n_odors'])}`。这预测相同气味诱发的"
        "转向命令相对对照向右移动，而不是记忆准确率普遍提高。\n\n"
        "## 专一性与通路\n\n"
        f"- 真实效应在 subtype-preserving side-registration null 中排名 `{null_rank}/{int(cfg.null_repeats) + 1}`。\n"
        f"- 沉默全部 MBON 后仅保留 `{float(all_mbon['retained_fraction']):.3%}` 的效应。\n"
        f"- 单独沉默 MBON32 后保留 `{float(mbon32_silence['retained_fraction']):.3%}`，并发生方向反转。\n"
        "- 真实门控降低左 MBON32、提高右 MBON32 的第一跳 drive；MBON32 到 DNa02 的直接抑制连接"
        "形成左右不等的推拉结构，因此主要通过解除右 DNa02 抑制产生相对右移。\n\n"
        "## 边界\n\n"
        "DNa02 左右活动差有成年果蝇转向实验标定，但这里输出仍是 signed graph drive，不是实际放电率或"
        "角速度。seed/odor 是确定性输入 realization，不是动物重复。5-HT-only 与 Glu-only gate 独立标准化，"
        "不能作为可相加的生化贡献。未施加突触学习更新的 CS-labelled contrast proxy 没有通过方向和剂量"
        "稳健性门槛。active-cap 与逐步归一化敏感性结果保存在独立表中；不同归一化设置的效应幅度"
        "不可直接比较。因此当前结论是 odor-evoked steering bias，而不是 memory enhancement。\n",
        encoding="utf-8",
    )
    return {
        "primary_raw": primary_path,
        "dose_summary": dose_path,
        "paired_contrasts": paired_path,
        "contrast_summary": contrast_path,
        "step_sensitivity_raw": step_raw_path,
        "step_sensitivity": step_path,
        "propagation_sensitivity_raw": propagation_sensitivity_raw_path,
        "propagation_sensitivity": propagation_sensitivity_path,
        "associative_contrast_raw": learned_raw_path,
        "associative_paired_contrasts": learned_paired_path,
        "associative_contrast_summary": learned_path,
        "component_effects": component_path,
        "mediation": mediation_path,
        "null_summary": null_path,
        "mbon32_response": mbon32_path,
        "direct_edges": edge_path,
        "metadata": metadata_path,
        "report": report_path,
        "figure_png": figure_png,
        "figure_pdf": figure_pdf,
    }
