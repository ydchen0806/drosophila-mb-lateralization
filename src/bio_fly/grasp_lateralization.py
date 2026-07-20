"""Direction-agnostic analysis of paired left/right GRASP measurements."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class GraspAnalysisConfig:
    experimental_group: str = "experimental"
    control_group: str | None = "control"
    bootstrap_repeats: int = 20_000
    permutation_repeats: int = 20_000
    random_seed: int = 20260720


def _bootstrap_mean_ci(values: np.ndarray, repeats: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(repeats, values.size), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _sign_flip_p(values: np.ndarray, repeats: int, seed: int) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    chunk = 2_000
    completed = 0
    while completed < repeats:
        current = min(chunk, repeats - completed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(current, values.size))
        exceed += int((np.abs((signs * values).mean(axis=1)) >= observed).sum())
        completed += current
    return float((1 + exceed) / (1 + repeats))


def _absolute_group_permutation_p(
    experimental: np.ndarray,
    control: np.ndarray,
    repeats: int,
    seed: int,
) -> float:
    experimental = np.asarray(experimental, dtype=float)
    control = np.asarray(control, dtype=float)
    observed = float(experimental.mean() - control.mean())
    pooled = np.concatenate([experimental, control])
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(repeats):
        shuffled = rng.permutation(pooled)
        delta = float(shuffled[: experimental.size].mean() - shuffled[experimental.size :].mean())
        exceed += int(delta >= observed)
    return float((1 + exceed) / (1 + repeats))


def prepare_fly_level(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"fly_id", "left_signal", "right_signal"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"GRASP input is missing columns: {sorted(missing)}")

    data = frame.copy()
    if "group" not in data:
        data["group"] = "experimental"
    if "batch" not in data:
        data["batch"] = "all"
    if "exclude" not in data:
        data["exclude"] = False
    data["group"] = data["group"].fillna("experimental").astype(str)
    data["batch"] = data["batch"].fillna("all").astype(str)
    if data["exclude"].dtype == object:
        data["exclude"] = data["exclude"].fillna(False).astype(str).str.lower().isin({"1", "true", "yes"})
    else:
        data["exclude"] = data["exclude"].fillna(False).astype(bool)
    data = data.loc[~data["exclude"]].copy()
    for column in ["left_signal", "right_signal"]:
        data[column] = pd.to_numeric(data[column], errors="raise")

    left_background = (
        pd.to_numeric(data["left_background"], errors="raise")
        if "left_background" in data
        else pd.Series(0.0, index=data.index)
    )
    right_background = (
        pd.to_numeric(data["right_background"], errors="raise")
        if "right_background" in data
        else pd.Series(0.0, index=data.index)
    )
    data["left_corrected"] = data["left_signal"] - left_background
    data["right_corrected"] = data["right_signal"] - right_background
    if data[["left_corrected", "right_corrected"]].le(0).any().any():
        raise ValueError("Background-corrected GRASP signals must be positive")

    fly = (
        data.groupby(["group", "batch", "fly_id"], as_index=False)
        .agg(
            left_signal=("left_corrected", "mean"),
            right_signal=("right_corrected", "mean"),
            n_technical_repeats=("fly_id", "size"),
        )
    )
    fly["right_minus_left"] = fly["right_signal"] - fly["left_signal"]
    fly["laterality_index"] = fly["right_minus_left"] / (fly["right_signal"] + fly["left_signal"])
    fly["absolute_laterality_index"] = fly["laterality_index"].abs()
    fly["direction"] = np.select(
        [fly["right_minus_left"].gt(0), fly["right_minus_left"].lt(0)],
        ["right", "left"],
        default="tie",
    )
    return fly.sort_values(["group", "batch", "fly_id"]).reset_index(drop=True)


def analyze_fly_level(fly: pd.DataFrame, config: GraspAnalysisConfig) -> dict[str, object]:
    experimental = fly[fly["group"].eq(config.experimental_group)].copy()
    if experimental.empty:
        raise ValueError(f"No rows found for experimental group {config.experimental_group!r}")

    signed = experimental["laterality_index"].to_numpy(float)
    absolute = np.abs(signed)
    signed_ci = _bootstrap_mean_ci(signed, config.bootstrap_repeats, config.random_seed)
    absolute_ci = _bootstrap_mean_ci(absolute, config.bootstrap_repeats, config.random_seed + 1)
    non_ties = experimental[~experimental["direction"].eq("tie")]
    right_count = int(experimental["direction"].eq("right").sum())
    left_count = int(experimental["direction"].eq("left").sum())
    sign_p = float(
        stats.binomtest(right_count, right_count + left_count, p=0.5, alternative="two-sided").pvalue
    ) if right_count + left_count else float("nan")

    control = pd.DataFrame()
    absolute_control_p = float("nan")
    absolute_control_u_p = float("nan")
    control_threshold = float("nan")
    n_above_control = None
    if config.control_group is not None:
        control = fly[fly["group"].eq(config.control_group)].copy()
    if not control.empty:
        control_absolute = control["absolute_laterality_index"].to_numpy(float)
        absolute_control_p = _absolute_group_permutation_p(
            absolute,
            control_absolute,
            config.permutation_repeats,
            config.random_seed + 2,
        )
        absolute_control_u_p = float(
            stats.mannwhitneyu(absolute, control_absolute, alternative="greater").pvalue
        )
        control_threshold = float(np.quantile(control_absolute, 0.95))
        n_above_control = int((absolute > control_threshold).sum())

    if signed_ci[0] > 0:
        interpretation = "population_right_shifted_with_individual_variability"
    elif signed_ci[1] < 0:
        interpretation = "population_left_shifted_with_individual_variability"
    elif not control.empty and absolute_control_p < 0.05:
        interpretation = "direction_variable_lateralization_above_control"
    elif control.empty:
        interpretation = "direction_variable_descriptive_control_required"
    else:
        interpretation = "no_group_level_lateralization_above_control"

    return {
        "experimental_group": config.experimental_group,
        "control_group": config.control_group if not control.empty else None,
        "n_experimental_flies": int(len(experimental)),
        "n_control_flies": int(len(control)),
        "right_lateralized_flies": right_count,
        "left_lateralized_flies": left_count,
        "tied_flies": int(len(experimental) - len(non_ties)),
        "mean_signed_laterality_index": float(signed.mean()),
        "signed_laterality_bootstrap_ci_low": signed_ci[0],
        "signed_laterality_bootstrap_ci_high": signed_ci[1],
        "signed_mean_sign_flip_p_two_sided": _sign_flip_p(
            signed,
            config.permutation_repeats,
            config.random_seed + 3,
        ),
        "right_left_exact_sign_p_two_sided": sign_p,
        "mean_absolute_laterality_index": float(absolute.mean()),
        "absolute_laterality_bootstrap_ci_low": absolute_ci[0],
        "absolute_laterality_bootstrap_ci_high": absolute_ci[1],
        "absolute_laterality_vs_control_permutation_p_one_sided": absolute_control_p,
        "absolute_laterality_vs_control_mannwhitney_p_one_sided": absolute_control_u_p,
        "control_absolute_laterality_95th_percentile": control_threshold,
        "experimental_flies_above_control_95th_percentile": n_above_control,
        "interpretation": interpretation,
        "boundary": (
            "a non-zero absolute left-right difference alone is insufficient evidence for biological "
            "lateralization without negative-control or repeatability information"
        ),
    }


def summarize_batches(fly: pd.DataFrame, config: GraspAnalysisConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected = fly[fly["group"].eq(config.experimental_group)]
    for index, (batch, group) in enumerate(selected.groupby("batch", sort=True)):
        signed = group["laterality_index"].to_numpy(float)
        absolute = np.abs(signed)
        signed_ci = _bootstrap_mean_ci(signed, config.bootstrap_repeats, config.random_seed + 10 + index)
        absolute_ci = _bootstrap_mean_ci(absolute, config.bootstrap_repeats, config.random_seed + 100 + index)
        rows.append(
            {
                "batch": batch,
                "n_flies": int(len(group)),
                "right_flies": int((signed > 0).sum()),
                "left_flies": int((signed < 0).sum()),
                "mean_signed_laterality_index": float(signed.mean()),
                "signed_ci_low": signed_ci[0],
                "signed_ci_high": signed_ci[1],
                "mean_absolute_laterality_index": float(absolute.mean()),
                "absolute_ci_low": absolute_ci[0],
                "absolute_ci_high": absolute_ci[1],
            }
        )
    return pd.DataFrame.from_records(rows)


def build_figure(fly: pd.DataFrame, summary: dict[str, object], output_stem: Path) -> None:
    experimental = fly[fly["group"].eq(summary["experimental_group"])].copy()
    experimental = experimental.sort_values("laterality_index").reset_index(drop=True)
    control_group = summary.get("control_group")
    control = fly[fly["group"].eq(control_group)].copy() if control_group else pd.DataFrame()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.45), gridspec_kw={"width_ratios": [1.0, 1.15, 0.9]})
    red, blue, gray, teal = "#B84C5C", "#3F78A8", "#858585", "#2F8178"

    ax = axes[0]
    for row in experimental.itertuples():
        color = red if row.laterality_index > 0 else blue
        ax.plot([0, 1], [row.left_signal, row.right_signal], color=color, alpha=0.35, lw=0.7)
        ax.scatter([0, 1], [row.left_signal, row.right_signal], color=color, s=8)
    ax.set_xticks([0, 1], ["left", "right"])
    ax.set_ylabel("background-corrected GRASP signal")
    ax.set_title("Paired fly-level signal")

    ax = axes[1]
    x = np.arange(1, len(experimental) + 1)
    colors = np.where(experimental["laterality_index"].gt(0), red, blue)
    ax.axhline(0, color="#222222", lw=0.7)
    ax.vlines(x, 0, experimental["laterality_index"], color=colors, lw=0.9)
    ax.scatter(x, experimental["laterality_index"], color=colors, s=11, zorder=3)
    ax.set_xlabel("flies sorted by signed effect")
    ax.set_ylabel("laterality index, (R-L)/(R+L)")
    ax.set_title("Direction varies across flies")

    ax = axes[2]
    rng = np.random.default_rng(20260720)
    groups = [("GRASP", experimental["absolute_laterality_index"].to_numpy(float), teal)]
    if not control.empty:
        groups.append(("control", control["absolute_laterality_index"].to_numpy(float), gray))
    for index, (label, values, color) in enumerate(groups):
        jitter = rng.uniform(-0.10, 0.10, size=len(values))
        ax.scatter(np.full(len(values), index) + jitter, values, color=color, alpha=0.75, s=13)
        ax.plot([index - 0.18, index + 0.18], [values.mean(), values.mean()], color="#222222", lw=1.2)
    ax.set_xticks(range(len(groups)), [item[0] for item in groups])
    ax.set_ylabel("absolute laterality index")
    ax.set_title("Lateralization magnitude")

    for letter, ax in zip("abc", axes):
        ax.text(-0.18, 1.07, letter, transform=ax.transAxes, fontweight="bold", fontsize=9, va="top")
        ax.grid(axis="y", color="#E6E6E6", lw=0.5)
        ax.set_axisbelow(True)
    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def run_grasp_analysis(input_path: Path, output_dir: Path, config: GraspAnalysisConfig) -> dict[str, Path]:
    frame = pd.read_csv(input_path)
    fly = prepare_fly_level(frame)
    summary = analyze_fly_level(fly, config)
    batches = summarize_batches(fly, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "fly_level": output_dir / "grasp_fly_level.csv",
        "batch_summary": output_dir / "grasp_batch_summary.csv",
        "statistics": output_dir / "grasp_statistics.json",
        "figure_pdf": output_dir / "Fig_grasp_lateralization.pdf",
        "figure_png": output_dir / "Fig_grasp_lateralization.png",
    }
    fly.to_csv(paths["fly_level"], index=False)
    batches.to_csv(paths["batch_summary"], index=False)
    paths["statistics"].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    build_figure(fly, summary, output_dir / "Fig_grasp_lateralization")
    return paths


__all__ = [
    "GraspAnalysisConfig",
    "analyze_fly_level",
    "build_figure",
    "prepare_fly_level",
    "run_grasp_analysis",
    "summarize_batches",
]
