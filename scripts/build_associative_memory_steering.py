#!/usr/bin/env python3
"""Build the compact evidence package and main figure for learned steering."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA = ROOT / "data" / "source_data"
FIGURES = ROOT / "figures"
SOURCE = ROOT / "outputs" / "associative_steering"

LEFT = "#86A9C3"
RIGHT = "#E88A98"
TEAL = "#72BDB5"
GOLD = "#E6B464"
DARK = "#263238"
GRAY = "#AEB8BE"
LIGHT = "#E8ECEF"
BG = "#FCFCFA"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "axes.titlesize": 8.1,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "legend.fontsize": 5.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


COMPACT_FILES = [
    "associative_steering_self_by_seed.csv",
    "associative_steering_stage_summary.csv",
    "associative_steering_stage_contrasts.csv",
    "associative_steering_structural_inventory.csv",
    "associative_steering_structural_controls.csv",
    "associative_steering_generalization_by_seed.csv",
    "associative_steering_generalization_summary.csv",
    "associative_steering_generalization_profile.csv",
    "associative_steering_gate_null.csv",
    "associative_steering_transfer_sensitivity.csv",
    "associative_steering_male_route_validation.csv",
    "associative_steering_metadata.json",
]


def panel(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.13, 1.17, letter, transform=ax.transAxes, fontsize=8.8, fontweight="bold", va="top")


def style(ax: plt.Axes, axis: str = "y") -> None:
    ax.set_facecolor(BG)
    ax.grid(True, axis=axis, color=LIGHT, lw=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4C5459")
    ax.spines["bottom"].set_color("#4C5459")


def copy_compact_evidence() -> None:
    for name in COMPACT_FILES:
        source = SOURCE / name
        if not source.exists():
            raise FileNotFoundError(f"Run associative steering first: missing {source}")
        shutil.copy2(source, SOURCE_DATA / name)


def build_figure() -> None:
    inventory = pd.read_csv(SOURCE_DATA / "associative_steering_structural_inventory.csv")
    seeds = pd.read_csv(SOURCE_DATA / "associative_steering_self_by_seed.csv")
    contrasts = pd.read_csv(SOURCE_DATA / "associative_steering_stage_contrasts.csv")
    controls = pd.read_csv(SOURCE_DATA / "associative_steering_structural_controls.csv")
    profile = pd.read_csv(SOURCE_DATA / "associative_steering_generalization_profile.csv")
    generalization = pd.read_csv(SOURCE_DATA / "associative_steering_generalization_summary.csv")
    male = pd.read_csv(SOURCE_DATA / "associative_steering_male_route_validation.csv")
    metadata = json.loads((SOURCE_DATA / "associative_steering_metadata.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(2, 3, figsize=(7.35, 5.35), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.ravel()

    # a: three registered anatomical stages align in the same direction.
    stage_order = ["KC_to_MBON32", "PPL103_to_MBON32", "MBON32_to_contralateral_DNa02"]
    stage_labels = ["KC→MBON32", "PPL103→MBON32", "MBON32→DNa02"]
    piv = inventory.pivot(index="stage", columns="side", values="synapse_weight").reindex(stage_order)
    x = np.arange(3)
    width = 0.34
    style(ax_a, "y")
    ax_a.bar(x - width / 2, piv["left"], width, color=LEFT, label="left arm", zorder=3)
    ax_a.bar(x + width / 2, piv["right"], width, color=RIGHT, label="right arm", zorder=3)
    ax_a.set_yscale("log")
    ax_a.set_xticks(x, stage_labels, rotation=24, ha="right")
    ax_a.set_ylabel("registered edge weight (log)")
    ax_a.set_title("Aligned memory-to-steering anatomy")
    ax_a.legend(frameon=False, loc="upper right")
    panel(ax_a, "a")

    # b: paired validation-seed commands from the fully registered replay.
    paired = seeds[
        seeds["split"].eq("validation") & seeds["stage"].eq("real_both")
    ].sort_values("seed")
    style(ax_b, "y")
    for row in paired.itertuples(index=False):
        ax_b.plot([0, 1], [row.mean_left_command, row.mean_right_command], color=GRAY, lw=0.75, alpha=0.75, zorder=2)
    ax_b.scatter(np.zeros(len(paired)), paired["mean_left_command"], s=22, color=LEFT, edgecolor="white", lw=0.4, zorder=4)
    ax_b.scatter(np.ones(len(paired)), paired["mean_right_command"], s=22, color=RIGHT, edgecolor="white", lw=0.4, zorder=4)
    ax_b.set_xticks([0, 1], ["left cue arm", "right cue arm"])
    ax_b.set_ylabel("learned contraversive\nDNa02 command increment")
    real_ratio = float(paired["mean_left_command"].mean() / paired["mean_right_command"].mean())
    ax_b.text(0.5, 0.96, f"{real_ratio:.2f}× left/right; 120/120", transform=ax_b.transAxes, ha="center", va="top", fontsize=6.2)
    ax_b.set_title("Paired held-out odor replays")
    panel(ax_b, "b")

    # c: acquisition versus retrieval expression.
    val = contrasts[contrasts["split"].eq("validation")].set_index("contrast")
    contrast_order = [
        "real_training_only_minus_symmetrized",
        "real_retrieval_only_minus_symmetrized",
        "real_both_minus_symmetrized",
        "mirror_both_minus_symmetrized",
    ]
    contrast_labels = ["real at training", "real at retrieval", "real at both", "mirror at both"]
    colors = [GRAY, TEAL, RIGHT, LEFT]
    rows = val.reindex(contrast_order)
    y = np.arange(4)
    style(ax_c, "x")
    ax_c.axvline(0, color="#596166", lw=0.8)
    for i, (row, color) in enumerate(zip(rows.itertuples(), colors)):
        ax_c.errorbar(
            row.mean_delta,
            i,
            xerr=[[row.mean_delta - row.seed_bootstrap_ci_low], [row.seed_bootstrap_ci_high - row.mean_delta]],
            fmt="o",
            color=color,
            ecolor=color,
            ms=4.8,
            capsize=2,
            lw=1.2,
            zorder=4,
        )
    ax_c.set_yticks(y, contrast_labels)
    ax_c.set_xlabel("change in lateral learned command")
    ax_c.set_title("Chemical gate acts mainly at retrieval")
    panel(ax_c, "c")

    # d: structural counterfactuals isolate input and output contributions.
    control_order = [
        "registered",
        "input_mass_equalized",
        "output_equalized",
        "input_and_output_equalized",
        "output_sides_swapped",
    ]
    control_labels = ["registered", "input equal", "output equal", "both equal", "output swapped"]
    ctrl = controls.set_index("control").reindex(control_order)
    y = np.arange(len(ctrl))
    style(ax_d, "x")
    ax_d.axvline(1, color="#596166", lw=0.8, ls="--")
    ax_d.barh(y, ctrl["left_right_ratio_of_means"], color=[RIGHT, GOLD, TEAL, GRAY, LEFT], height=0.62, zorder=3)
    for yi, value in enumerate(ctrl["left_right_ratio_of_means"]):
        ax_d.text(value + 0.10, yi, f"{value:.2f}×", va="center", fontsize=5.8)
    ax_d.set_yticks(y, control_labels)
    ax_d.invert_yaxis()
    ax_d.set_xlim(0, 7.8)
    ax_d.set_xlabel("left/right learned-command ratio")
    ax_d.set_title("Input and output both contribute")
    panel(ax_d, "d")

    # e: memory is stimulus-specific and generalizes with cue overlap.
    style(ax_e, "y")
    side_colors = {"left": LEFT, "right": RIGHT}
    for side, group in profile.groupby("side"):
        group = group.sort_values("mean_kc_jaccard")
        ax_e.plot(
            group["mean_kc_jaccard"],
            group["mean_contraversive_command"],
            marker="o",
            ms=3.8,
            lw=1.4,
            color=side_colors[side],
            label=f"{side} arm",
            zorder=3,
        )
    slopes = generalization.set_index("side")["mean_slope"]
    ax_e.text(0.03, 0.96, f"slope L/R = {slopes['left'] / slopes['right']:.2f}×", transform=ax_e.transAxes, va="top", fontsize=6.1)
    ax_e.set_xlabel("training–test KC-code overlap")
    ax_e.set_ylabel("learned contraversive command")
    ax_e.set_title("Cue-specific memory generalization")
    ax_e.legend(frameon=False, loc="lower right")
    panel(ax_e, "e")

    # f: the same route families are independently retained in male CNS.
    route_order = ["KC_to_MBON32", "PPL103_to_MBON32", "MBON32_to_DNa02"]
    route_labels = ["KC→MBON32", "PPL103→MBON32", "MBON32→DNa02"]
    m = male.set_index("route").reindex(route_order)
    x = np.arange(3)
    style(ax_f, "y")
    ax_f.bar(x - width / 2, m["female_weight"], width, color=RIGHT, label="female FlyWire", zorder=3)
    ax_f.bar(x + width / 2, m["male_weight"], width, color=TEAL, label="male CNS", zorder=3)
    ax_f.set_yscale("log")
    ax_f.set_xticks(x, route_labels, rotation=24, ha="right")
    ax_f.set_ylabel("type-component weight (log)")
    ax_f.text(
        0.98,
        0.96,
        "3/3 route families retained",
        transform=ax_f.transAxes,
        ha="right",
        va="top",
        fontsize=6.1,
    )
    ax_f.set_title("Independent route conservation")
    ax_f.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.86))
    panel(ax_f, "f")

    # Fail loudly if the figure was built from stale or unexpected primary results.
    primary = metadata["primary_result"]
    if not (6.9 < float(primary["registered_left_right_ratio"]) < 7.1):
        raise ValueError("Unexpected registered learned-command ratio")
    if primary["registered_gate_null_rank"] != "1/33":
        raise ValueError("Unexpected registered-gate null rank")

    for suffix in ("pdf", "png"):
        fig.savefig(
            FIGURES / f"Fig6_associative_memory_steering.{suffix}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.035,
        )
    plt.close(fig)


def main() -> None:
    copy_compact_evidence()
    build_figure()


if __name__ == "__main__":
    main()
