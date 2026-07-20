"""Replay an odor-specific KC-to-MBON32 memory trace into steering output.

The model implements the literature-anchored depression of active KC-to-MBON
synapses during aversive learning. It asks how much that learned synaptic
change alters the contraversive DNa02 command through the direct inhibitory
MBON32 pathway. The result is a memory-expression prediction inside the
specified connectome model, not measured learning or animal choice.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .kc_flywire_ratio_experiment import (
    _build_flywire_glomerulus_kc_matrix_from_frames,
    _load_kc_lateralization_components,
    _normalize_rows,
    _sparsify,
    build_mixture_odor_panel,
)
from .lateralized_steering import (
    DEFAULT_ANNOTATION_PATH,
    DEFAULT_KC_NT_INPUTS_PATH,
    DEFAULT_SHIU_CONNECTIVITY_PATH,
    extract_dna02_drive,
    load_annotations,
    subtype_preserving_gate_shuffle,
)
from .paths import DEFAULT_OUTPUT_ROOT, REPO_ROOT
from .propagation import (
    PropagationConfig,
    build_torch_propagation_graph,
    signed_multihop_response_torch,
)


DEFAULT_MALE_EDGE_COMPONENT_PATH = (
    REPO_ROOT
    / "data"
    / "external"
    / "male_cns_2025"
    / "supplemental_data"
    / "mcns_fw_edge_comp.feather"
)

STAGE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("symmetrized", "symmetrized", "symmetrized"),
    ("real_training_only", "real", "symmetrized"),
    ("real_retrieval_only", "symmetrized", "real"),
    ("real_both", "real", "real"),
    ("mirror_both", "mirror", "mirror"),
)


@dataclass(frozen=True)
class AssociativeSteeringConfig:
    annotation_path: Path = field(default_factory=lambda: DEFAULT_ANNOTATION_PATH)
    connectivity_path: Path = field(default_factory=lambda: DEFAULT_SHIU_CONNECTIVITY_PATH)
    kc_nt_inputs_path: Path = field(default_factory=lambda: DEFAULT_KC_NT_INPUTS_PATH)
    male_edge_component_path: Path = field(default_factory=lambda: DEFAULT_MALE_EDGE_COMPONENT_PATH)
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_ROOT / "associative_steering")
    device: str = "cuda:0"
    seeds: tuple[int, ...] = tuple(range(20))
    validation_seed_start: int = 10
    n_odors: int = 12
    min_glomeruli_per_odor: int = 2
    max_glomeruli_per_odor: int = 6
    channel_noise_sigma: float = 0.15
    kc_active_fraction_per_side: float = 0.10
    gate_amplitude: float = 0.25
    depression_fraction: float = 0.75
    null_repeats: int = 32
    random_seed: int = 20260718
    transfer_steps: int = 4
    max_active: int = 60_000


def build_side_restricted_code(
    activity: np.ndarray,
    gate: np.ndarray,
    side_labels: np.ndarray,
    side: str,
    *,
    gate_amplitude: float,
    active_fraction: float,
) -> np.ndarray:
    """Apply a gate and sparsify one KC hemisphere with matched intensity."""

    side_mask = np.asarray(side_labels, dtype=object) == str(side)
    if not bool(np.any(side_mask)):
        raise ValueError(f"No KCs found for side {side!r}.")
    gain = np.clip(1.0 + float(gate_amplitude) * np.asarray(gate, dtype=float), 0.05, 5.0)
    selected = _normalize_rows(np.maximum(activity[:, side_mask] * gain[side_mask][None, :], 0.0))
    _, graded, _ = _sparsify(selected, float(active_fraction))
    full = np.zeros_like(activity, dtype=float)
    full[:, side_mask] = graded
    return full


def learned_mbon32_depression(
    training_code: np.ndarray,
    retrieval_code: np.ndarray,
    kc_to_mbon32_weights: np.ndarray,
    depression_fraction: float,
) -> float:
    """Return the lost MBON32 excitation at odor-tagged KC synapses."""

    training_active = np.asarray(training_code, dtype=float) > 0
    retrieval = np.asarray(retrieval_code, dtype=float)
    weights = np.asarray(kc_to_mbon32_weights, dtype=float)
    if training_active.shape != retrieval.shape or retrieval.shape != weights.shape:
        raise ValueError("training, retrieval and KC-to-MBON32 weight vectors must align")
    return float(float(depression_fraction) * np.sum(retrieval * training_active * weights))


def signed_dna02_learning_delta(side: str, mbon32_depression: float, transfer_weight: float) -> float:
    """Map reduced inhibitory MBON32 drive to right-minus-left DNa02."""

    command = float(mbon32_depression) * abs(float(transfer_weight))
    if side == "left":
        return command
    if side == "right":
        return -command
    raise ValueError(f"Unknown side {side!r}")


def _bootstrap_seed_ci(frame: pd.DataFrame, column: str, random_seed: int) -> tuple[float, float]:
    values = frame.groupby("seed")[column].mean().to_numpy(dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(random_seed))
    draws = rng.choice(values, size=(20_000, values.size), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


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


def _direct_path_data(
    annotations: pd.DataFrame,
    edges: pd.DataFrame,
    kc_ids: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float], pd.DataFrame, pd.DataFrame, dict[str, int], dict[str, int]]:
    mbon32 = annotations[annotations["cell_type"].fillna("").astype(str).eq("MBON32")]
    dna02 = annotations[annotations["cell_type"].fillna("").astype(str).eq("DNa02")]
    mbon32_ids = {str(row.side): int(row.root_id) for row in mbon32.itertuples()}
    dna02_ids = {str(row.side): int(row.root_id) for row in dna02.itertuples()}
    if set(mbon32_ids) != {"left", "right"} or set(dna02_ids) != {"left", "right"}:
        raise ValueError("Expected one bilateral MBON32 and DNa02 pair.")

    kc_set = set(map(int, kc_ids))
    kc_weights: dict[str, np.ndarray] = {}
    transfer_weights: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for side in ("left", "right"):
        kc_edges = edges[
            edges["Presynaptic_ID"].isin(kc_set)
            & edges["Postsynaptic_ID"].eq(mbon32_ids[side])
        ]
        grouped = kc_edges.groupby("Presynaptic_ID")["Connectivity"].sum()
        kc_weights[side] = grouped.reindex(kc_ids, fill_value=0.0).to_numpy(dtype=float)
        rows.append(
            {
                "stage": "KC_to_MBON32",
                "side": side,
                "n_edges": int(len(kc_edges)),
                "n_source_neurons": int(kc_edges["Presynaptic_ID"].nunique()),
                "synapse_weight": float(kc_edges["Connectivity"].sum()),
                "signed_weight": float(kc_edges["Excitatory x Connectivity"].sum()),
            }
        )

        contralateral = "right" if side == "left" else "left"
        output_edges = edges[
            edges["Presynaptic_ID"].eq(mbon32_ids[side])
            & edges["Postsynaptic_ID"].eq(dna02_ids[contralateral])
        ]
        transfer_weights[side] = float(output_edges["Excitatory x Connectivity"].sum())
        rows.append(
            {
                "stage": "MBON32_to_contralateral_DNa02",
                "side": side,
                "n_edges": int(len(output_edges)),
                "n_source_neurons": 1,
                "synapse_weight": float(output_edges["Connectivity"].sum()),
                "signed_weight": float(output_edges["Excitatory x Connectivity"].sum()),
            }
        )

    ppl103 = annotations[annotations["cell_type"].fillna("").astype(str).eq("PPL103")]
    ppl_ids = set(ppl103["root_id"].dropna().astype("int64"))
    for side in ("left", "right"):
        teaching_edges = edges[
            edges["Presynaptic_ID"].isin(ppl_ids)
            & edges["Postsynaptic_ID"].eq(mbon32_ids[side])
        ]
        rows.append(
            {
                "stage": "PPL103_to_MBON32",
                "side": side,
                "n_edges": int(len(teaching_edges)),
                "n_source_neurons": int(teaching_edges["Presynaptic_ID"].nunique()),
                "synapse_weight": float(teaching_edges["Connectivity"].sum()),
                "signed_weight": float(teaching_edges["Excitatory x Connectivity"].sum()),
            }
        )
    inventory = pd.DataFrame.from_records(rows)
    wide = inventory.pivot(index="stage", columns="side", values="synapse_weight").reset_index()
    wide["left_to_right_ratio"] = wide["left"] / wide["right"].replace(0.0, np.nan)

    mbon = annotations[annotations["cell_class"].fillna("").astype(str).eq("MBON")][
        ["root_id", "side", "cell_type"]
    ].dropna(subset=["cell_type"])
    kc_edge_all = edges[edges["Presynaptic_ID"].isin(kc_set)].merge(
        mbon.rename(columns={"root_id": "Postsynaptic_ID", "side": "target_side", "cell_type": "mbon_type"}),
        on="Postsynaptic_ID",
        how="inner",
    )
    pair = (
        kc_edge_all.groupby(["mbon_type", "target_side"], as_index=False)
        .agg(
            synapse_weight=("Connectivity", "sum"),
            target_neurons=("Postsynaptic_ID", "nunique"),
        )
    )
    pair["weight_per_target"] = pair["synapse_weight"] / pair["target_neurons"]
    pair_wide = pair.pivot(index="mbon_type", columns="target_side", values="weight_per_target").dropna()
    pair_wide["right_minus_left_index"] = (
        (pair_wide["right"] - pair_wide["left"]) / (pair_wide["right"] + pair_wide["left"])
    )
    pair_wide["absolute_laterality"] = pair_wide["right_minus_left_index"].abs()
    pair_wide = pair_wide.sort_values("absolute_laterality", ascending=False).reset_index()
    pair_wide["absolute_laterality_rank"] = np.arange(1, len(pair_wide) + 1)
    return kc_weights, transfer_weights, inventory.merge(wide[["stage", "left_to_right_ratio"]], on="stage"), pair_wide, mbon32_ids, dna02_ids


def _male_route_validation(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    edges = pd.read_feather(path, columns=["pre", "post", "weight_m", "weight_f", "verdict_corr"])
    specs = (
        ("KC_to_MBON32", edges["pre"].astype(str).str.startswith("KC") & edges["post"].eq("MBON32")),
        ("PPL103_to_MBON32", edges["pre"].eq("PPL103") & edges["post"].eq("MBON32")),
        ("MBON32_to_DNa02", edges["pre"].eq("MBON32") & edges["post"].eq("DNa02")),
    )
    rows: list[dict[str, object]] = []
    for route, mask in specs:
        selected = edges[mask]
        rows.append(
            {
                "route": route,
                "n_type_components": int(len(selected)),
                "male_weight": float(selected["weight_m"].sum()),
                "female_weight": float(selected["weight_f"].sum()),
                "isomorphic_components": int(selected["verdict_corr"].astype(str).eq("isomorphic").sum()),
                "all_nonzero_components_isomorphic": bool(
                    selected.loc[
                        selected[["weight_m", "weight_f"]].sum(axis=1).gt(0),
                        "verdict_corr",
                    ].astype(str).eq("isomorphic").all()
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def _summarize_self_test(self_test: pd.DataFrame, random_seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    for split in ("all", "development", "validation"):
        frame = self_test if split == "all" else self_test[self_test["split"].eq(split)]
        for stage, group in frame.groupby("stage", sort=False):
            low, high = _bootstrap_seed_ci(group, "left_minus_right_command", random_seed)
            records.append(
                {
                    "split": split,
                    "stage": stage,
                    "n_seeds": int(group["seed"].nunique()),
                    "n_odors": int(len(group)),
                    "mean_left_command": float(group["left"].mean()),
                    "mean_right_command": float(group["right"].mean()),
                    "left_right_ratio_of_means": float(group["left"].mean() / group["right"].mean()),
                    "mean_left_minus_right_command": float(group["left_minus_right_command"].mean()),
                    "seed_bootstrap_ci_low": low,
                    "seed_bootstrap_ci_high": high,
                    "left_stronger_odors": int((group["left_minus_right_command"] > 0).sum()),
                    "right_stronger_odors": int((group["left_minus_right_command"] < 0).sum()),
                }
            )
    summary = pd.DataFrame.from_records(records)

    contrasts: list[dict[str, object]] = []
    wide = self_test.pivot(
        index=["seed", "split", "odor_index", "odor_name"],
        columns="stage",
        values="left_minus_right_command",
    ).reset_index()
    for split in ("all", "development", "validation"):
        frame = wide if split == "all" else wide[wide["split"].eq(split)]
        for stage in ("real_training_only", "real_retrieval_only", "real_both", "mirror_both"):
            delta = frame[stage] - frame["symmetrized"]
            paired = frame[["seed"]].assign(delta=delta)
            low, high = _bootstrap_seed_ci(paired, "delta", random_seed + len(stage))
            contrasts.append(
                {
                    "split": split,
                    "contrast": f"{stage}_minus_symmetrized",
                    "n_seeds": int(frame["seed"].nunique()),
                    "n_odors": int(len(frame)),
                    "mean_delta": float(delta.mean()),
                    "seed_bootstrap_ci_low": low,
                    "seed_bootstrap_ci_high": high,
                    "positive_odors": int((delta > 0).sum()),
                    "negative_odors": int((delta < 0).sum()),
                    "zero_odors": int((delta == 0).sum()),
                }
            )
    return summary, pd.DataFrame.from_records(contrasts)


def _structural_controls(
    self_test: pd.DataFrame,
    inventory: pd.DataFrame,
    depression_fraction: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sym = self_test[
        self_test["split"].eq("validation") & self_test["stage"].eq("symmetrized")
    ].copy()
    kc_rows = inventory[inventory["stage"].eq("KC_to_MBON32")].set_index("side")
    output_rows = inventory[inventory["stage"].eq("MBON32_to_contralateral_DNa02")].set_index("side")
    input_mean = float(kc_rows["synapse_weight"].mean())
    output_mean = float(output_rows["synapse_weight"].mean())
    specs = {
        "registered": (1.0, 1.0, float(output_rows.loc["left", "synapse_weight"]), float(output_rows.loc["right", "synapse_weight"])),
        "input_mass_equalized": (
            input_mean / float(kc_rows.loc["left", "synapse_weight"]),
            input_mean / float(kc_rows.loc["right", "synapse_weight"]),
            float(output_rows.loc["left", "synapse_weight"]),
            float(output_rows.loc["right", "synapse_weight"]),
        ),
        "output_equalized": (1.0, 1.0, output_mean, output_mean),
        "input_and_output_equalized": (
            input_mean / float(kc_rows.loc["left", "synapse_weight"]),
            input_mean / float(kc_rows.loc["right", "synapse_weight"]),
            output_mean,
            output_mean,
        ),
        "output_sides_swapped": (
            1.0,
            1.0,
            float(output_rows.loc["right", "synapse_weight"]),
            float(output_rows.loc["left", "synapse_weight"]),
        ),
    }
    raw_rows: list[dict[str, object]] = []
    for row in sym.itertuples(index=False):
        left_depression = float(row.left_mbon32_depression) / float(depression_fraction)
        right_depression = float(row.right_mbon32_depression) / float(depression_fraction)
        for control, (left_scale, right_scale, left_output, right_output) in specs.items():
            left = float(depression_fraction) * left_depression * left_scale * left_output
            right = float(depression_fraction) * right_depression * right_scale * right_output
            raw_rows.append(
                {
                    "seed": int(row.seed),
                    "odor_index": int(row.odor_index),
                    "control": control,
                    "left_command": left,
                    "right_command": right,
                    "left_minus_right_command": left - right,
                }
            )
    raw = pd.DataFrame.from_records(raw_rows)
    summary_rows: list[dict[str, object]] = []
    for control, group in raw.groupby("control", sort=False):
        low, high = _bootstrap_seed_ci(group, "left_minus_right_command", random_seed)
        summary_rows.append(
            {
                "control": control,
                "n_seeds": int(group["seed"].nunique()),
                "n_odors": int(len(group)),
                "mean_left_command": float(group["left_command"].mean()),
                "mean_right_command": float(group["right_command"].mean()),
                "left_right_ratio_of_means": float(group["left_command"].mean() / group["right_command"].mean()),
                "mean_left_minus_right_command": float(group["left_minus_right_command"].mean()),
                "seed_bootstrap_ci_low": low,
                "seed_bootstrap_ci_high": high,
                "left_stronger_odors": int((group["left_minus_right_command"] > 0).sum()),
                "right_stronger_odors": int((group["left_minus_right_command"] < 0).sum()),
            }
        )
    return raw, pd.DataFrame.from_records(summary_rows)


def _generalization_summary(raw: pd.DataFrame, random_seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = raw[
        raw["split"].eq("validation")
        & raw["stage"].eq("real_both")
        & ~raw["is_self_test"]
    ].copy()
    rows: list[dict[str, object]] = []
    for (seed, side), group in selected.groupby(["seed", "side"]):
        slope = float(np.polyfit(group["kc_jaccard"], group["contraversive_command"], 1)[0])
        rho = float(spearmanr(group["kc_jaccard"], group["contraversive_command"]).statistic)
        rows.append({"seed": int(seed), "side": side, "slope": slope, "spearman_rho": rho, "n_pairs": len(group)})
    by_seed = pd.DataFrame.from_records(rows)
    summary_rows: list[dict[str, object]] = []
    for side, group in by_seed.groupby("side"):
        slope_low, slope_high = _bootstrap_seed_ci(group, "slope", random_seed)
        rho_low, rho_high = _bootstrap_seed_ci(group, "spearman_rho", random_seed + 1)
        summary_rows.append(
            {
                "side": side,
                "n_seeds": int(group["seed"].nunique()),
                "mean_slope": float(group["slope"].mean()),
                "slope_seed_bootstrap_ci_low": slope_low,
                "slope_seed_bootstrap_ci_high": slope_high,
                "mean_spearman_rho": float(group["spearman_rho"].mean()),
                "rho_seed_bootstrap_ci_low": rho_low,
                "rho_seed_bootstrap_ci_high": rho_high,
            }
        )
    bins = [-1e-12, 0.05, 0.10, 0.20, 0.35, 1.0]
    selected["overlap_bin"] = pd.cut(selected["kc_jaccard"], bins=bins, include_lowest=True)
    profile = (
        selected.groupby(["side", "overlap_bin"], observed=True, as_index=False)
        .agg(
            n_pairs=("contraversive_command", "size"),
            mean_kc_jaccard=("kc_jaccard", "mean"),
            mean_contraversive_command=("contraversive_command", "mean"),
        )
    )
    profile["overlap_bin"] = profile["overlap_bin"].astype(str)
    return by_seed, pd.DataFrame.from_records(summary_rows), profile


def _plot_figure(
    inventory: pd.DataFrame,
    stage_contrasts: pd.DataFrame,
    structural_summary: pd.DataFrame,
    generalization_profile: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), dpi=300, constrained_layout=True)
    ax = axes[0, 0]
    stages = ["KC_to_MBON32", "PPL103_to_MBON32", "MBON32_to_contralateral_DNa02"]
    data = inventory.pivot(index="stage", columns="side", values="synapse_weight").reindex(stages)
    x = np.arange(len(data))
    ax.bar(x - 0.18, data["left"], width=0.36, color="#4c78a8", label="left arm")
    ax.bar(x + 0.18, data["right"], width=0.36, color="#e07b62", label="right arm")
    ax.set_xticks(x, ["KC→MBON32", "PPL103→MBON32", "MBON32→DNa02"])
    ax.set_yscale("log")
    ax.set_ylabel("connectivity weight")
    ax.set_title("Aligned memory-to-steering asymmetry")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[0, 1]
    val = stage_contrasts[stage_contrasts["split"].eq("validation")].copy()
    order = [
        "real_training_only_minus_symmetrized",
        "real_retrieval_only_minus_symmetrized",
        "real_both_minus_symmetrized",
        "mirror_both_minus_symmetrized",
    ]
    val = val.set_index("contrast").reindex(order).reset_index()
    y = np.arange(len(val))
    ax.errorbar(
        val["mean_delta"],
        y,
        xerr=np.vstack(
            [
                val["mean_delta"] - val["seed_bootstrap_ci_low"],
                val["seed_bootstrap_ci_high"] - val["mean_delta"],
            ]
        ),
        fmt="o",
        color="#2f7d5c",
        capsize=2,
    )
    ax.axvline(0, color="#555555", lw=0.7)
    ax.set_yticks(y, ["training only", "retrieval only", "both", "mirror both"])
    ax.set_xlabel("change in left-right learned command")
    ax.set_title("Chemical gate acts mainly at retrieval")

    ax = axes[1, 0]
    order = ["registered", "input_mass_equalized", "output_equalized", "input_and_output_equalized", "output_sides_swapped"]
    control = structural_summary.set_index("control").reindex(order).reset_index()
    ax.barh(np.arange(len(control)), control["left_right_ratio_of_means"], color="#6f8f72")
    ax.axvline(1, color="#555555", lw=0.7)
    ax.set_yticks(np.arange(len(control)), ["registered", "input equal", "output equal", "both equal", "output swapped"])
    ax.invert_yaxis()
    ax.set_xlabel("left/right learned-command ratio")
    ax.set_title("Input and output asymmetries both contribute")

    ax = axes[1, 1]
    for side, color in [("left", "#4c78a8"), ("right", "#e07b62")]:
        group = generalization_profile[generalization_profile["side"].eq(side)]
        ax.plot(group["mean_kc_jaccard"], group["mean_contraversive_command"], "o-", color=color, label=side)
    ax.set_xlabel("train-test KC overlap")
    ax.set_ylabel("learned contraversive command")
    ax.set_title("Stimulus-specific generalization")
    ax.legend(frameon=False, fontsize=7)

    for label, ax in zip("abcd", axes.flat):
        ax.text(-0.14, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
        ax.title.set_fontsize(8)
        ax.xaxis.label.set_size(7)
        ax.yaxis.label.set_size(7)
    png = output_dir / "Fig_associative_steering.png"
    pdf = output_dir / "Fig_associative_steering.pdf"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def run_associative_steering(config: AssociativeSteeringConfig | None = None) -> dict[str, Path]:
    cfg = config or AssociativeSteeringConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    annotations = load_annotations(cfg.annotation_path)
    edge_columns = ["Presynaptic_ID", "Postsynaptic_ID", "Connectivity", "Excitatory x Connectivity"]
    edges = pd.read_parquet(cfg.connectivity_path, columns=edge_columns)
    glomeruli, glomerulus_matrix, kc_ids, channel_table = _build_flywire_glomerulus_kc_matrix_from_frames(
        annotations,
        edges,
    )
    gate_table = _load_kc_lateralization_components(cfg.kc_nt_inputs_path, kc_ids)
    real_gate = gate_table["symmetry_breaking_gate"].to_numpy(dtype=float)
    side_labels = annotations.set_index("root_id").reindex(kc_ids)["side"].fillna("").astype(str).to_numpy()
    kc_weights, transfer_weights, inventory, pair_rank, mbon32_ids, dna02_ids = _direct_path_data(
        annotations,
        edges,
        kc_ids,
    )
    male_routes = _male_route_validation(cfg.male_edge_component_path)

    panels: list[tuple[int, str, list[str], np.ndarray]] = []
    panel_rows: list[pd.DataFrame] = []
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
        panels.append((int(seed), split, odor_names, activity))
        panel_rows.append(panel.assign(seed=int(seed), split=split))

    raw_rows: list[dict[str, object]] = []
    cached_codes: dict[tuple[int, str, str], np.ndarray] = {}
    gate_vectors = {
        "symmetrized": np.zeros_like(real_gate),
        "real": real_gate,
        "mirror": -real_gate,
    }
    for seed, split, odor_names, activity in panels:
        for gate_name, gate_vector in gate_vectors.items():
            for side in ("left", "right"):
                cached_codes[(seed, gate_name, side)] = build_side_restricted_code(
                    activity,
                    gate_vector,
                    side_labels,
                    side,
                    gate_amplitude=cfg.gate_amplitude,
                    active_fraction=cfg.kc_active_fraction_per_side,
                )
        for stage, training_gate, retrieval_gate in STAGE_SPECS:
            for training_index, training_name in enumerate(odor_names):
                for test_index, test_name in enumerate(odor_names):
                    for side in ("left", "right"):
                        training_code = cached_codes[(seed, training_gate, side)][training_index]
                        retrieval_code = cached_codes[(seed, retrieval_gate, side)][test_index]
                        depression = learned_mbon32_depression(
                            training_code,
                            retrieval_code,
                            kc_weights[side],
                            cfg.depression_fraction,
                        )
                        signed_delta = signed_dna02_learning_delta(side, depression, transfer_weights[side])
                        training_active = training_code > 0
                        retrieval_active = retrieval_code > 0
                        union = int(np.logical_or(training_active, retrieval_active).sum())
                        overlap = int(np.logical_and(training_active, retrieval_active).sum()) / max(1, union)
                        raw_rows.append(
                            {
                                "seed": seed,
                                "split": split,
                                "stage": stage,
                                "training_gate": training_gate,
                                "retrieval_gate": retrieval_gate,
                                "training_odor_index": training_index,
                                "training_odor_name": training_name,
                                "test_odor_index": test_index,
                                "test_odor_name": test_name,
                                "is_self_test": bool(training_index == test_index),
                                "side": side,
                                "kc_jaccard": overlap,
                                "mbon32_depression": depression,
                                "signed_dna02_right_minus_left_delta": signed_delta,
                                "contraversive_command": abs(signed_delta),
                            }
                        )
    raw = pd.DataFrame.from_records(raw_rows)
    self_long = raw[raw["is_self_test"]].copy()
    self_test = self_long.pivot(
        index=["seed", "split", "stage", "training_odor_index", "training_odor_name"],
        columns="side",
        values=["contraversive_command", "mbon32_depression"],
    ).reset_index()
    self_test.columns = [
        "_".join(str(value) for value in column if str(value)) if isinstance(column, tuple) else str(column)
        for column in self_test.columns
    ]
    self_test = self_test.rename(
        columns={
            "training_odor_index": "odor_index",
            "training_odor_name": "odor_name",
            "contraversive_command_left": "left",
            "contraversive_command_right": "right",
            "mbon32_depression_left": "left_mbon32_depression",
            "mbon32_depression_right": "right_mbon32_depression",
        }
    )
    self_test["left_minus_right_command"] = self_test["left"] - self_test["right"]
    self_by_seed = (
        self_test.groupby(["seed", "split", "stage"], as_index=False)
        .agg(
            n_odors=("odor_index", "size"),
            mean_left_command=("left", "mean"),
            mean_right_command=("right", "mean"),
            mean_left_minus_right_command=("left_minus_right_command", "mean"),
        )
    )
    self_by_seed["left_right_ratio_of_means"] = (
        self_by_seed["mean_left_command"] / self_by_seed["mean_right_command"]
    )
    stage_summary, stage_contrasts = _summarize_self_test(self_test, cfg.random_seed)
    structural_raw, structural_summary = _structural_controls(
        self_test,
        inventory,
        cfg.depression_fraction,
        cfg.random_seed,
    )
    generalization_by_seed, generalization_summary, generalization_profile = _generalization_summary(
        raw,
        cfg.random_seed,
    )

    sym_validation = self_test[
        self_test["split"].eq("validation") & self_test["stage"].eq("symmetrized")
    ][["seed", "odor_index", "left_minus_right_command"]].rename(
        columns={"left_minus_right_command": "symmetrized_lateral_command"}
    )
    validation_panels = [panel for panel in panels if panel[1] == "validation"]
    null_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(int(cfg.random_seed))
    for repeat in range(int(cfg.null_repeats)):
        shuffled_gate = subtype_preserving_gate_shuffle(real_gate, gate_table["subtype"], rng)
        repeat_rows: list[dict[str, object]] = []
        for seed, _split, odor_names, activity in validation_panels:
            commands: dict[str, np.ndarray] = {}
            for side in ("left", "right"):
                code = build_side_restricted_code(
                    activity,
                    shuffled_gate,
                    side_labels,
                    side,
                    gate_amplitude=cfg.gate_amplitude,
                    active_fraction=cfg.kc_active_fraction_per_side,
                )
                values = []
                for odor_index in range(len(odor_names)):
                    depression = learned_mbon32_depression(
                        code[odor_index],
                        code[odor_index],
                        kc_weights[side],
                        cfg.depression_fraction,
                    )
                    values.append(abs(signed_dna02_learning_delta(side, depression, transfer_weights[side])))
                commands[side] = np.asarray(values, dtype=float)
            for odor_index in range(len(odor_names)):
                repeat_rows.append(
                    {
                        "seed": seed,
                        "odor_index": odor_index,
                        "lateral_command": commands["left"][odor_index] - commands["right"][odor_index],
                    }
                )
        repeat_frame = pd.DataFrame.from_records(repeat_rows).merge(
            sym_validation,
            on=["seed", "odor_index"],
            how="inner",
        )
        delta = repeat_frame["lateral_command"] - repeat_frame["symmetrized_lateral_command"]
        null_rows.append(
            {
                "repeat": repeat,
                "mean_delta_vs_symmetrized": float(delta.mean()),
                "positive_odors": int((delta > 0).sum()),
                "negative_odors": int((delta < 0).sum()),
            }
        )
    gate_null = pd.DataFrame.from_records(null_rows)

    graph = build_torch_propagation_graph(cfg.connectivity_path, device=cfg.device)
    transfer_rows: list[dict[str, object]] = []
    for normalize_each_step in (False, True):
        propagation_config = PropagationConfig(
            steps=int(cfg.transfer_steps),
            max_active=int(cfg.max_active),
            normalize_each_step=normalize_each_step,
        )
        for side in ("left", "right"):
            response = signed_multihop_response_torch(
                graph,
                {mbon32_ids[side]: -1.0},
                propagation_config,
            )
            for step_limit in range(1, int(cfg.transfer_steps) + 1):
                signed_delta = extract_dna02_drive(response, dna02_ids, step_limit)["dna02_right_minus_left"]
                transfer_rows.append(
                    {
                        "side": side,
                        "step_limit": step_limit,
                        "normalize_each_step": normalize_each_step,
                        "signed_dna02_right_minus_left_for_unit_mbon32_depression": signed_delta,
                        "contraversive_transfer": signed_delta if side == "left" else -signed_delta,
                    }
                )
    transfer_sensitivity = pd.DataFrame.from_records(transfer_rows)

    paths = {
        "raw": cfg.output_dir / "associative_steering_raw.csv",
        "self_test": cfg.output_dir / "associative_steering_self_test.csv",
        "self_by_seed": cfg.output_dir / "associative_steering_self_by_seed.csv",
        "stage_summary": cfg.output_dir / "associative_steering_stage_summary.csv",
        "stage_contrasts": cfg.output_dir / "associative_steering_stage_contrasts.csv",
        "structural_inventory": cfg.output_dir / "associative_steering_structural_inventory.csv",
        "mbon_pair_rank": cfg.output_dir / "associative_steering_mbon_pair_rank.csv",
        "structural_controls_raw": cfg.output_dir / "associative_steering_structural_controls_raw.csv",
        "structural_controls": cfg.output_dir / "associative_steering_structural_controls.csv",
        "generalization_by_seed": cfg.output_dir / "associative_steering_generalization_by_seed.csv",
        "generalization_summary": cfg.output_dir / "associative_steering_generalization_summary.csv",
        "generalization_profile": cfg.output_dir / "associative_steering_generalization_profile.csv",
        "gate_null": cfg.output_dir / "associative_steering_gate_null.csv",
        "transfer_sensitivity": cfg.output_dir / "associative_steering_transfer_sensitivity.csv",
        "male_routes": cfg.output_dir / "associative_steering_male_route_validation.csv",
        "odor_panel": cfg.output_dir / "associative_steering_odor_panel.csv",
        "channel_table": cfg.output_dir / "associative_steering_glomerulus_channels.csv",
        "metadata": cfg.output_dir / "associative_steering_metadata.json",
        "report": cfg.output_dir / "ASSOCIATIVE_STEERING_REPORT_CN.md",
    }
    raw.to_csv(paths["raw"], index=False)
    self_test.to_csv(paths["self_test"], index=False)
    self_by_seed.to_csv(paths["self_by_seed"], index=False)
    stage_summary.to_csv(paths["stage_summary"], index=False)
    stage_contrasts.to_csv(paths["stage_contrasts"], index=False)
    inventory.to_csv(paths["structural_inventory"], index=False)
    pair_rank.to_csv(paths["mbon_pair_rank"], index=False)
    structural_raw.to_csv(paths["structural_controls_raw"], index=False)
    structural_summary.to_csv(paths["structural_controls"], index=False)
    generalization_by_seed.to_csv(paths["generalization_by_seed"], index=False)
    generalization_summary.to_csv(paths["generalization_summary"], index=False)
    generalization_profile.to_csv(paths["generalization_profile"], index=False)
    gate_null.to_csv(paths["gate_null"], index=False)
    transfer_sensitivity.to_csv(paths["transfer_sensitivity"], index=False)
    male_routes.to_csv(paths["male_routes"], index=False)
    pd.concat(panel_rows, ignore_index=True).to_csv(paths["odor_panel"], index=False)
    channel_table.to_csv(paths["channel_table"], index=False)

    validation_contrasts = stage_contrasts[stage_contrasts["split"].eq("validation")].set_index("contrast")
    registered = structural_summary.set_index("control").loc["registered"]
    both_equal = structural_summary.set_index("control").loc["input_and_output_equalized"]
    output_swap = structural_summary.set_index("control").loc["output_sides_swapped"]
    registered_gate_effect = float(
        validation_contrasts.loc["real_both_minus_symmetrized", "mean_delta"]
    )
    null_rank = 1 + int((gate_null["mean_delta_vs_symmetrized"] >= registered_gate_effect).sum())
    metadata = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(cfg).items()},
        "provenance": {
            "flywire_annotations_commit": _git_commit(cfg.annotation_path.parents[1]),
            "shiu_model_commit": _git_commit(cfg.connectivity_path.parent),
            "annotation_sha256": _file_sha256(cfg.annotation_path),
            "connectivity_sha256": _file_sha256(cfg.connectivity_path),
            "kc_nt_inputs_sha256": _file_sha256(cfg.kc_nt_inputs_path),
            "male_edge_component_sha256": _file_sha256(cfg.male_edge_component_path)
            if cfg.male_edge_component_path.exists()
            else "unavailable",
        },
        "primary_result": {
            "registered_left_right_ratio": float(registered["left_right_ratio_of_means"]),
            "registered_left_stronger_odors": int(registered["left_stronger_odors"]),
            "registered_n_odors": int(registered["n_odors"]),
            "input_output_equalized_ratio": float(both_equal["left_right_ratio_of_means"]),
            "output_swap_ratio": float(output_swap["left_right_ratio_of_means"]),
            "real_both_minus_symmetrized": validation_contrasts.loc[
                "real_both_minus_symmetrized"
            ].to_dict(),
            "real_training_only_minus_symmetrized": validation_contrasts.loc[
                "real_training_only_minus_symmetrized"
            ].to_dict(),
            "real_retrieval_only_minus_symmetrized": validation_contrasts.loc[
                "real_retrieval_only_minus_symmetrized"
            ].to_dict(),
            "mirror_both_minus_symmetrized": validation_contrasts.loc[
                "mirror_both_minus_symmetrized"
            ].to_dict(),
            "registered_gate_null_rank": f"{null_rank}/{int(cfg.null_repeats) + 1}",
            "generalization": {
                row.side: {
                    "mean_slope": float(row.mean_slope),
                    "slope_seed_bootstrap_ci_low": float(row.slope_seed_bootstrap_ci_low),
                    "slope_seed_bootstrap_ci_high": float(row.slope_seed_bootstrap_ci_high),
                    "mean_spearman_rho": float(row.mean_spearman_rho),
                }
                for row in generalization_summary.itertuples(index=False)
            },
        },
        "claim_boundary": {
            "supported": (
                "odor-specific KC-to-MBON32 depression predicts a stronger left-arm contraversive DNa02 change, "
                "and the registered chemical gate acts mainly during retrieval/expression"
            ),
            "not_supported": (
                "measured memory accuracy, absolute approach-to-avoidance reversal, calibrated turning, or animal causality"
            ),
        },
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    paths["report"].write_text(
        "# 侧化记忆到转向表达的模型分析\n\n"
        "## 结论\n\n"
        f"在验证集配对气味中，同样的 odor-tagged KC->MBON32 突触抑制经真实左右结构产生 "
        f"`{float(registered['left_right_ratio_of_means']):.2f}x` 的左/右 contraversive DNa02 指令比，"
        f"左侧更强为 `{int(registered['left_stronger_odors'])}/{int(registered['n_odors'])}`。"
        "同时等化 KC 输入质量和 MBON32 输出后，该比值接近 1；交换输出侧别会翻转。\n\n"
        "真实 chemical gate 在训练和检索期同时存在时进一步增大左右差，而 mirror gate 减小；"
        "只在训练期施加时平均效应很小、略向反方向且逐气味方向混合，只在检索期施加时则在 "
        f"`{int(validation_contrasts.loc['real_retrieval_only_minus_symmetrized', 'positive_odors'])}/"
        f"{int(validation_contrasts.loc['real_retrieval_only_minus_symmetrized', 'n_odors'])}` 个留出气味中"
        "几乎都保留预期方向。因此当前模型把侧化定位为"
        "记忆检索/行为表达增益，而不是学习准确率提高。\n\n"
        "## 生物学边界\n\n"
        "训练规则是文献锚定的活跃 KC->MBON32 突触抑制，输出读取真实 MBON32->DNa02 抑制边。"
        "模型没有显式重建固定抑制支路的膜电位平衡，因此报告的是学习导致的 contraversive command "
        "increment，不是绝对趋近/回避翻转，也不是实测果蝇选择。\n",
        encoding="utf-8",
    )
    figure_png, figure_pdf = _plot_figure(
        inventory,
        stage_contrasts,
        structural_summary,
        generalization_profile,
        cfg.output_dir,
    )
    paths["figure_png"] = figure_png
    paths["figure_pdf"] = figure_pdf
    return paths


__all__ = [
    "AssociativeSteeringConfig",
    "build_side_restricted_code",
    "learned_mbon32_depression",
    "run_associative_steering",
    "signed_dna02_learning_delta",
]
