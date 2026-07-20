#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA = ROOT / "data" / "source_data"
FIGURES = ROOT / "figures"

BLUE = "#3F78A8"
RED = "#B84C5C"
TEAL = "#2F8178"
ORANGE = "#C4872E"
GRAY = "#858585"
LIGHT_GRAY = "#D9D9D9"
DARK = "#222222"
GRID = "#E6E6E6"


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.0,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3,
        "legend.fontsize": 6.0,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def style_axis(ax: plt.Axes, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(DARK)
    ax.spines["bottom"].set_color(DARK)
    ax.tick_params(direction="out", width=0.7, length=3, color=DARK)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.55, zorder=0)
        ax.set_axisbelow(True)


def label_panel(ax: plt.Axes, letter: str, x: float = -0.18, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=9.0,
        fontweight="bold",
        ha="left",
        va="top",
        clip_on=False,
    )


def errorbar_h(
    ax: plt.Axes,
    y: np.ndarray,
    mean: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    color: str | list[str],
    marker: str = "o",
    size: float = 4.2,
) -> None:
    colors = [color] * len(mean) if isinstance(color, str) else color
    for yi, mi, lo, hi, ci in zip(y, mean, low, high, colors):
        ax.plot([lo, hi], [yi, yi], color=ci, lw=1.0, zorder=2)
        ax.plot(mi, yi, marker=marker, ms=size, color=ci, mec="white", mew=0.45, zorder=3)


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    # Nature Neuroscience caps submitted panels at 180 mm. The fixed 7.05-inch
    # canvas is 179.1 mm; avoiding a tight bounding box prevents outer labels
    # from silently expanding the physical page beyond that limit.
    fig.savefig(FIGURES / f"{stem}.pdf")
    fig.savefig(FIGURES / f"{stem}.png", dpi=300)
    plt.close(fig)


def build_statistics() -> None:
    subtype = pd.read_csv(SOURCE_DATA / "subtype_replication_meta_analysis.csv")
    spec = pd.read_csv(SOURCE_DATA / "kc_5ht_specification_curve.csv")
    stress = pd.read_csv(SOURCE_DATA / "kc_5ht_classifier_stress_test.csv")
    source = pd.read_csv(SOURCE_DATA / "kc_5ht_source_filter_sensitivity.csv")
    placebo = pd.read_csv(SOURCE_DATA / "whole_brain_matched_placebo_null.csv")
    placebo_summary = pd.read_csv(SOURCE_DATA / "whole_brain_matched_placebo_summary.csv").iloc[0]

    fig = plt.figure(figsize=(7.05, 4.75))
    gs = fig.add_gridspec(
        2,
        6,
        left=0.13,
        right=0.985,
        top=0.95,
        bottom=0.10,
        hspace=0.58,
        wspace=0.82,
    )
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[0, 4:6])
    ax_d = fig.add_subplot(gs[1, 0:3])
    ax_e = fig.add_subplot(gs[1, 3:6])

    # a, subtype forest plot.
    subtype = subtype[subtype["replication_unit"].eq("KC subtype")].copy()
    subtype = subtype.sort_values("right_minus_left_fraction", ascending=True).reset_index(drop=True)
    y = np.arange(len(subtype))
    mean = subtype["right_minus_left_fraction"].to_numpy() * 100
    low = subtype["bootstrap_ci_low"].to_numpy() * 100
    high = subtype["bootstrap_ci_high"].to_numpy() * 100
    errorbar_h(ax_a, y, mean, low, high, RED)
    labels = [
        str(x).replace("KCa'b'", "KC a'/b'").replace("KCab", "KC a/b").replace("KCg", "KC g")
        for x in subtype["hemibrain_type"]
    ]
    ax_a.set_yticks(y, labels)
    ax_a.axvline(0, color=DARK, lw=0.7)
    ax_a.axvline(0.42, color=GRAY, lw=0.8, ls="--")
    ax_a.text(0.44, len(y) - 0.55, "pooled 0.42 pp", fontsize=5.7, color=GRAY, va="top")
    ax_a.set_xlabel("5-HT-predicted input, right - left (pp)")
    ax_a.set_title("KC subtype effects")
    style_axis(ax_a, "x")
    label_panel(ax_a, "a", -0.16)

    # b, specification curve without compressing the sparse source-identity estimand.
    frac = spec[spec["display_unit"].eq("percentage points")].copy()
    sparse = frac["source_filter"].astype(str).str.contains("source_top_5HT", case=False, na=False)
    main = frac.loc[~sparse].sort_values("estimate_pp").reset_index(drop=True)
    x = np.arange(len(main))
    colors = np.where(main["estimate_pp"].to_numpy() > 0, RED, GRAY)
    ax_b.scatter(x, main["estimate_pp"], s=13, c=colors, edgecolors="white", linewidths=0.3, zorder=3)
    ax_b.axhline(0, color=DARK, lw=0.7)
    ax_b.axhline(float(main["estimate_pp"].median()), color=GRAY, ls="--", lw=0.8)
    ax_b.set_xlabel("fraction-like specifications")
    ax_b.set_ylabel("right - left (pp)")
    ax_b.set_title(f"Specification curve ({int((frac['estimate_pp'] > 0).sum())}/{len(frac)} positive)")
    ax_b.text(0.02, 0.96, "sparse source-top-5-HT estimand excluded from axis", transform=ax_b.transAxes, fontsize=5.1, color=GRAY, va="top")
    style_axis(ax_b, "y")
    label_panel(ax_b, "b")

    # c, uniform classifier-overcall stress.
    x = stress["right_side_ser_probability_downshift_pp"].to_numpy()
    ax_c.plot(x, stress["population_right_minus_left_ser_fraction_pp"], color=RED, marker="o", ms=2.6, label="population")
    ax_c.plot(x, stress["mean_subtype_shift_pp"], color=TEAL, marker="o", ms=2.6, label="subtype mean")
    first_break = stress.loc[stress["n_positive_subtypes"] < stress["n_subtypes"], "right_side_ser_probability_downshift_pp"]
    if len(first_break):
        ax_c.axvline(float(first_break.iloc[0]), color=GRAY, lw=0.8, ls="--")
        ax_c.text(float(first_break.iloc[0]) + 0.02, 0.47, "9/9 first breaks", fontsize=5.4, color=GRAY, rotation=90, va="top")
    ax_c.axhline(0, color=DARK, lw=0.7)
    ax_c.set_xlabel("uniform right overcall removed (pp)")
    ax_c.set_ylabel("residual right - left (pp)")
    ax_c.set_title("Classifier overcall stress")
    ax_c.legend(frameon=False, loc="upper right")
    style_axis(ax_c, "y")
    label_panel(ax_c, "c")

    # d, source-filter sensitivity.
    order = [
        "all",
        "exclude_DAN",
        "exclude_DPM_MBIN",
        "exclude_source_top_dopamine",
        "exclude_low_source_confidence",
        "exclude_top20_influential_sources",
    ]
    labels_map = {
        "all": "all inputs",
        "exclude_DAN": "no DAN",
        "exclude_DPM_MBIN": "no DPM/MBIN",
        "exclude_source_top_dopamine": "no source-top DA",
        "exclude_low_source_confidence": "confidence >= 0.7",
        "exclude_top20_influential_sources": "top 20 out",
    }
    sub = source.set_index("analysis").reindex(order).dropna().reset_index()
    y = np.arange(len(sub))
    vals = sub["right_minus_left_ser_fraction_pp"].to_numpy()
    ax_d.axvline(0, color=DARK, lw=0.7)
    for yi, val, name in zip(y, vals, sub["analysis"]):
        ci = RED if name == "all" else (ORANGE if name == "exclude_DAN" else GRAY)
        ax_d.plot([0, val], [yi, yi], color=ci, lw=1.1)
        ax_d.plot(val, yi, "o", ms=4.2, color=ci, mec="white", mew=0.4)
    ax_d.set_yticks(y, [labels_map[x] for x in sub["analysis"]])
    ax_d.invert_yaxis()
    ax_d.set_xlabel("5-HT-predicted input, right - left (pp)")
    ax_d.set_title("Source-filter sensitivity")
    style_axis(ax_d, "x")
    label_panel(ax_d, "d", -0.14)

    # e, matched non-KC placebo null.
    null = placebo["matched_placebo_right_minus_left"].to_numpy() * 100
    obs = float(placebo_summary["observed_kc_right_minus_left_pp"])
    ax_e.hist(null, bins=22, color=LIGHT_GRAY, edgecolor="white", linewidth=0.4)
    ax_e.axvline(float(null.mean()), color=GRAY, lw=1.0, label="placebo mean")
    ax_e.axvline(obs, color=RED, lw=1.5, label="KC observed")
    ax_e.text(obs - 0.01, ax_e.get_ylim()[1] * 0.88, "99.7th percentile\np = 0.0066", color=RED, fontsize=5.8, ha="right", va="top")
    ax_e.set_xlabel("matched non-KC right - left (pp)")
    ax_e.set_ylabel("resamples")
    ax_e.set_title("Matched non-KC placebo")
    ax_e.legend(frameon=False, loc="upper left")
    style_axis(ax_e, "y")
    label_panel(ax_e, "e")

    save_figure(fig, "Fig1_brief_statistics")


def build_biology() -> None:
    dpm_switch = pd.read_csv(SOURCE_DATA / "dpm_5ht_compartment_switch.csv")
    paired = pd.read_csv(SOURCE_DATA / "dpm_29fly_paired_slopes.csv")
    timecourse = pd.read_csv(SOURCE_DATA / "dpm_timecourse_right_left_summary.csv")
    batches = pd.read_csv(SOURCE_DATA / "dpm_batch_aware_statistics.csv")
    with (SOURCE_DATA / "dpm_5ht_compartment_switch_statistics.json").open() as f:
        switch_stats = json.load(f)

    fig = plt.figure(figsize=(7.05, 4.80))
    gs = fig.add_gridspec(
        2,
        6,
        left=0.13,
        right=0.985,
        top=0.95,
        bottom=0.10,
        hspace=0.58,
        wspace=0.82,
    )
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[0, 4:6])
    ax_d = fig.add_subplot(gs[1, 0:4])
    ax_e = fig.add_subplot(gs[1, 4:6])

    # a, DPM compartment switch.
    dpm_switch = dpm_switch.sort_values("right_minus_left_synapses").reset_index(drop=True)
    y = np.arange(len(dpm_switch))
    vals = dpm_switch["right_minus_left_synapses"].to_numpy()
    cols = np.where(dpm_switch["compartment"].eq("alpha_prime_beta_prime"), RED, BLUE)
    ax_a.axvline(0, color=DARK, lw=0.7)
    for yi, val, ci in zip(y, vals, cols):
        ax_a.plot([0, val], [yi, yi], color=ci, lw=1.0)
        ax_a.plot(val, yi, "o", ms=4.0, color=ci, mec="white", mew=0.4)
    labels = [x.replace("KCa'b'", "KC a'/b' ").replace("KCab", "KC a/b ").replace("KCg", "KC g ") for x in dpm_switch["kc_subtype"]]
    ax_a.set_yticks(y, labels)
    ax_a.set_xlabel("high-confidence DPM-labelled synapses, R - L")
    ax_a.set_title("DPM-labelled inputs by KC subtype")
    ax_a.text(0.02, 0.96, f"exact P = {switch_stats['subtype_label_permutation_p_ge_observed']:.3f}", transform=ax_a.transAxes, fontsize=5.6, va="top")
    style_axis(ax_a, "x")
    label_panel(ax_a, "a", -0.16)

    # b, paired fly-level slopes.
    for row in paired.itertuples():
        ax_b.plot([0, 1], [row.left, row.right], color=LIGHT_GRAY, lw=0.65, zorder=1)
    ax_b.scatter(np.zeros(len(paired)), paired["left"], color=BLUE, s=14, edgecolor="white", linewidth=0.35, zorder=2)
    ax_b.scatter(np.ones(len(paired)), paired["right"], color=RED, s=14, edgecolor="white", linewidth=0.35, zorder=2)
    ax_b.plot([0, 1], [paired["left"].mean(), paired["right"].mean()], color=DARK, marker="D", ms=4.0, lw=1.3, zorder=3)
    ax_b.set_xlim(-0.28, 1.28)
    ax_b.set_xticks([0, 1], ["left", "right"])
    ax_b.set_ylabel("KC 5-HT peak dose slope")
    ax_b.set_title("Paired fly-level dose slopes")
    style_axis(ax_b, "y")
    label_panel(ax_b, "b")

    # c, sorted individual paired differences.
    ordered = paired.sort_values("right_minus_left").reset_index(drop=True)
    x = np.arange(1, len(ordered) + 1)
    pos = ordered["right_minus_left"].to_numpy() > 0
    markers = np.where(ordered["batch"].eq("new"), "s", "o")
    for xi, yi, is_pos, marker in zip(x, ordered["right_minus_left"], pos, markers):
        ci = RED if is_pos else BLUE
        ax_c.plot([xi, xi], [0, yi], color=ci, lw=0.8)
        ax_c.plot(xi, yi, marker=marker, color=ci, ms=3.6, mec="white", mew=0.35)
    ax_c.axhline(0, color=DARK, lw=0.7)
    ax_c.set_xlabel("flies sorted by paired effect")
    ax_c.set_ylabel("right - left slope")
    ax_c.set_title("Fly-level right-left effects")
    ax_c.text(0.04, 0.96, "circle: batch 1\nsquare: batch 2", transform=ax_c.transAxes, fontsize=5.2, va="top", color=GRAY)
    style_axis(ax_c, "y")
    label_panel(ax_c, "c")

    # d, timecourse.
    t = timecourse["time_s"].to_numpy()
    mean = timecourse["mean_right_minus_left"].to_numpy()
    sem = timecourse["sem_right_minus_left"].to_numpy()
    ax_d.axvspan(20, 140, color="#F1F1F1", zorder=0)
    ax_d.fill_between(t, mean - sem, mean + sem, color=RED, alpha=0.18, lw=0)
    ax_d.plot(t, mean, color=RED, lw=1.2)
    ax_d.axhline(0, color=DARK, lw=0.7)
    cluster = timecourse["is_positive_cluster_point"].astype(bool).to_numpy()
    if cluster.any():
        ax_d.plot(t[cluster], np.full(cluster.sum(), ax_d.get_ylim()[0] * 0.70), color=RED, lw=2.0, solid_capstyle="butt")
    ax_d.text(0.02, 0.96, f"cluster permutation P = {timecourse['cluster_permutation_p'].iloc[0]:.4f}", transform=ax_d.transAxes, fontsize=5.6, va="top")
    ax_d.set_xlabel("time (s)")
    ax_d.set_ylabel("KC 5-HT response, right - left")
    ax_d.set_title("DPM-evoked response time course")
    style_axis(ax_d, "y")
    label_panel(ax_d, "d", -0.10)

    # e, batch estimates.
    batch = batches[batches["analysis"].eq("peak_log_dose_slope")].copy()
    order = ["new", "old", "combined"]
    batch = batch.set_index("batch").reindex(order).reset_index()
    y = np.arange(len(batch))
    colors = [GRAY, GRAY, RED]
    errorbar_h(
        ax_e,
        y,
        batch["mean_right_minus_left"].to_numpy(),
        batch["bootstrap_ci_low"].to_numpy(),
        batch["bootstrap_ci_high"].to_numpy(),
        colors,
        size=4.5,
    )
    ax_e.axvline(0, color=DARK, lw=0.7)
    ax_e.set_yticks(y, [f"batch 2 (n={int(batch.loc[0, 'n_flies'])})", f"batch 1 (n={int(batch.loc[1, 'n_flies'])})", "combined (n=29)"])
    ax_e.invert_yaxis()
    ax_e.set_xlabel("mean paired right - left slope")
    ax_e.set_title("Batch estimates")
    ax_e.text(0.98, 0.06, "mean and bootstrap 95% CI", transform=ax_e.transAxes, fontsize=5.2, ha="right", color=GRAY)
    style_axis(ax_e, "x")
    label_panel(ax_e, "e", -0.22)

    save_figure(fig, "Fig2_brief_biology")


def build_simulation() -> None:
    point = pd.read_csv(SOURCE_DATA / "model_causal_point_dose_response.csv")
    arbor = pd.read_csv(SOURCE_DATA / "model_causal_arbor_dose_response.csv")
    concordance = pd.read_csv(SOURCE_DATA / "model_causal_arbor_case_concordance.csv")
    steering = pd.read_csv(SOURCE_DATA / "lateralized_steering_contrast_summary.csv")
    mediation = pd.read_csv(SOURCE_DATA / "lateralized_steering_mediation.csv")
    inventory = pd.read_csv(SOURCE_DATA / "associative_steering_structural_inventory.csv")
    by_seed = pd.read_csv(SOURCE_DATA / "associative_steering_self_by_seed.csv")
    controls = pd.read_csv(SOURCE_DATA / "associative_steering_structural_controls.csv")
    stages = pd.read_csv(SOURCE_DATA / "associative_steering_stage_contrasts.csv")

    fig = plt.figure(figsize=(7.05, 7.15))
    gs = fig.add_gridspec(
        3,
        6,
        left=0.11,
        right=0.99,
        top=0.965,
        bottom=0.075,
        hspace=0.72,
        wspace=0.95,
    )
    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 0:3]),
        fig.add_subplot(gs[1, 3:6]),
        fig.add_subplot(gs[2, 0:2]),
        fig.add_subplot(gs[2, 2:4]),
        fig.add_subplot(gs[2, 4:6]),
    ]
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h = axes

    # a, point-model dose response.
    ax_a.errorbar(
        point["gate_strength"],
        point["mean_delta_lateral_code_vs_sym"],
        yerr=[
            point["mean_delta_lateral_code_vs_sym"] - point["bootstrap_ci_low"],
            point["bootstrap_ci_high"] - point["mean_delta_lateral_code_vs_sym"],
        ],
        color=TEAL,
        marker="o",
        ms=3.8,
        capsize=2,
        lw=1.2,
    )
    ax_a.axhline(0, color=DARK, lw=0.7)
    ax_a.axvline(0, color=GRAY, lw=0.6)
    ax_a.set_xlabel("signed gate strength")
    ax_a.set_ylabel("lateral-code change vs symmetric")
    ax_a.set_title("Point model")
    ax_a.text(0.03, 0.96, "10/10 monotonic", transform=ax_a.transAxes, fontsize=5.4, va="top")
    style_axis(ax_a, "y")
    label_panel(ax_a, "a", -0.14)

    # b, cable-cell dose response.
    cohort_labels = {
        "lateral_support_selected_72": "support-selected (72)",
        "effective_dimension_selected_15": "orthogonal (15)",
    }
    for cohort, ci, marker in [
        ("lateral_support_selected_72", TEAL, "o"),
        ("effective_dimension_selected_15", RED, "s"),
    ]:
        sub = arbor[(arbor["cohort"].eq(cohort)) & (~arbor["condition_class"].eq("shuffled_lateralized"))].sort_values("gate_strength")
        err = sub["sd_delta_lateral_code_vs_sym"]
        ax_b.errorbar(sub["gate_strength"], sub["mean_delta_lateral_code_vs_sym"], yerr=err, color=ci, marker=marker, ms=3.6, capsize=2, lw=1.1, label=cohort_labels[cohort])
    ax_b.axhline(0, color=DARK, lw=0.7)
    ax_b.axvline(0, color=GRAY, lw=0.6)
    ax_b.set_xlabel("signed gate strength")
    ax_b.set_ylabel("cable-cell lateral-code change")
    ax_b.set_title("Cable-cell validation")
    ax_b.legend(frameon=False, loc="upper left", handlelength=1.3)
    style_axis(ax_b, "y")
    label_panel(ax_b, "b")

    # c, panelwise backend concordance.
    for cohort, ci, marker in [
        ("lateral_support_selected_72", TEAL, "o"),
        ("effective_dimension_selected_15", RED, "s"),
    ]:
        sub = concordance[concordance["cohort"].eq(cohort)]
        ax_c.scatter(
            sub["lateral_code_index_real3_delta_vs_sym_baseline"],
            sub["lateral_code_index_real3_delta_vs_sym_arbor"],
            s=11,
            color=ci,
            marker=marker,
            alpha=0.70,
            edgecolors="white",
            linewidths=0.25,
        )
    lim = [0, max(concordance["lateral_code_index_real3_delta_vs_sym_baseline"].max(), concordance["lateral_code_index_real3_delta_vs_sym_arbor"].max()) * 1.04]
    ax_c.plot(lim, lim, color=GRAY, ls="--", lw=0.8)
    ax_c.set_xlim(lim)
    ax_c.set_ylim(lim)
    ax_c.set_xlabel("point-model change")
    ax_c.set_ylabel("cable-cell change")
    ax_c.set_title("Backend concordance")
    ax_c.text(0.04, 0.95, "Spearman rho = 0.972 / 0.968", transform=ax_c.transAxes, fontsize=5.2, va="top")
    style_axis(ax_c, None)
    label_panel(ax_c, "c")

    # d, DNa02 direction and MBON mediation.
    val = steering[(steering["split"].eq("validation")) & steering["contrast"].isin(["mirror_minus_symmetrized", "real_minus_symmetrized"])].copy()
    rows = []
    for contrast, label in [("mirror_minus_symmetrized", "mirror gate"), ("real_minus_symmetrized", "registered gate")]:
        r = val[val["contrast"].eq(contrast)].iloc[0]
        rows.append((label, r.mean_delta * 1e6, r.seed_bootstrap_ci_low * 1e6, r.seed_bootstrap_ci_high * 1e6, RED if "mirror" in label else TEAL))
    for silenced, label in [("all_MBON", "all MBON off"), ("MBON32", "MBON32 off")]:
        r = mediation[mediation["silenced_group"].eq(silenced)].iloc[0]
        rows.append((label, r.mean_delta_vs_matched_symmetrized * 1e6, r.seed_bootstrap_ci_low * 1e6, r.seed_bootstrap_ci_high * 1e6, GRAY if silenced == "all_MBON" else ORANGE))
    y = np.arange(len(rows))
    errorbar_h(ax_d, y, np.array([r[1] for r in rows]), np.array([r[2] for r in rows]), np.array([r[3] for r in rows]), [r[4] for r in rows], size=4.0)
    ax_d.axvline(0, color=DARK, lw=0.7)
    ax_d.set_yticks(y, [r[0] for r in rows])
    ax_d.invert_yaxis()
    ax_d.set_xlabel("DNa02 right - left drive change (x 1e-6)")
    ax_d.set_title("MBON32-dependent DNa02 steering readout")
    style_axis(ax_d, "x")
    label_panel(ax_d, "d", -0.15)

    # e, registered structural route inventory.
    stage_order = ["KC_to_MBON32", "PPL103_to_MBON32", "MBON32_to_contralateral_DNa02"]
    stage_labels = ["KC to MBON32", "PPL103 to MBON32", "MBON32 to contra-DNa02"]
    pivot = inventory.pivot(index="stage", columns="side", values="synapse_weight").reindex(stage_order)
    y = np.arange(len(pivot))
    for yi, (_, row) in enumerate(pivot.iterrows()):
        ax_e.plot([row["right"], row["left"]], [yi, yi], color=LIGHT_GRAY, lw=1.2)
        ax_e.plot(row["left"], yi, "o", color=BLUE, ms=4.2, mec="white", mew=0.4)
        ax_e.plot(row["right"], yi, "o", color=RED, ms=4.2, mec="white", mew=0.4)
    ax_e.set_xscale("log")
    ax_e.set_yticks(y, stage_labels)
    ax_e.invert_yaxis()
    ax_e.set_xlabel("registered edge weight (log scale)")
    ax_e.set_title("Aligned memory-to-steering anatomy")
    ax_e.legend(
        [plt.Line2D([], [], marker="o", color=BLUE, lw=0), plt.Line2D([], [], marker="o", color=RED, lw=0)],
        ["left arm", "right arm"],
        frameon=False,
        loc="lower right",
    )
    style_axis(ax_e, "x")
    label_panel(ax_e, "e", -0.30)

    # f, held-out paired associative commands.
    seed = by_seed[(by_seed["split"].eq("validation")) & (by_seed["stage"].eq("real_both"))].sort_values("seed")
    for row in seed.itertuples():
        ax_f.plot([0, 1], [row.mean_left_command, row.mean_right_command], color=LIGHT_GRAY, lw=0.75)
    ax_f.scatter(np.zeros(len(seed)), seed["mean_left_command"], color=BLUE, s=15, edgecolor="white", linewidth=0.35, zorder=3)
    ax_f.scatter(np.ones(len(seed)), seed["mean_right_command"], color=RED, s=15, edgecolor="white", linewidth=0.35, zorder=3)
    ax_f.set_xlim(-0.25, 1.25)
    ax_f.set_xticks([0, 1], ["left cue arm", "right cue arm"])
    ax_f.set_ylabel("learned contraversive command")
    ax_f.set_title("Held-out associative replay")
    ax_f.text(0.04, 0.96, "10 validation seeds; 120 cues", transform=ax_f.transAxes, fontsize=5.2, va="top")
    style_axis(ax_f, "y")
    label_panel(ax_f, "f", -0.14)

    # g, structural causal controls.
    order = ["registered", "input_mass_equalized", "output_equalized", "input_and_output_equalized", "output_sides_swapped"]
    labels = ["registered", "input equal", "output equal", "both equal", "output swapped"]
    ctrl = controls.set_index("control").reindex(order).reset_index()
    y = np.arange(len(ctrl))
    vals = ctrl["left_right_ratio_of_means"].to_numpy()
    colors = [RED, ORANGE, TEAL, GRAY, BLUE]
    ax_g.axvline(1, color=GRAY, lw=0.8, ls="--")
    for yi, val, ci in zip(y, vals, colors):
        ax_g.plot([1, val], [yi, yi], color=ci, lw=1.0)
        ax_g.plot(val, yi, "o", color=ci, ms=4.2, mec="white", mew=0.4)
    ax_g.set_yticks(y, labels)
    ax_g.invert_yaxis()
    ax_g.set_xlabel("left/right learned-command ratio")
    ax_g.set_title("Structural controls")
    style_axis(ax_g, "x")
    label_panel(ax_g, "g", -0.26)

    # h, acquisition-versus-retrieval stage contrasts.
    order = [
        "mirror_both_minus_symmetrized",
        "real_training_only_minus_symmetrized",
        "real_retrieval_only_minus_symmetrized",
        "real_both_minus_symmetrized",
    ]
    labels = ["mirror, both", "training only", "retrieval only", "registered, both"]
    st = stages[(stages["split"].eq("validation"))].set_index("contrast").reindex(order).reset_index()
    y = np.arange(len(st))
    colors = [BLUE, GRAY, TEAL, RED]
    errorbar_h(
        ax_h,
        y,
        st["mean_delta"].to_numpy(),
        st["seed_bootstrap_ci_low"].to_numpy(),
        st["seed_bootstrap_ci_high"].to_numpy(),
        colors,
        size=4.0,
    )
    ax_h.axvline(0, color=DARK, lw=0.7)
    ax_h.set_yticks(y, labels)
    ax_h.invert_yaxis()
    ax_h.set_xlabel("change in lateral learned command")
    ax_h.set_title("Retrieval-stage effect")
    style_axis(ax_h, "x")
    label_panel(ax_h, "h", -0.30)

    save_figure(fig, "Fig3_brief_simulation")


def main() -> None:
    build_statistics()
    build_biology()
    build_simulation()
    print("Generated Fig1_brief_statistics, Fig2_brief_biology and Fig3_brief_simulation (PDF + PNG).")


if __name__ == "__main__":
    main()
