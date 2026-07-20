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
MALE_SUMMARY = SOURCE_DATA / "male_cns_mb_route_summary.csv"
MANC_BRIDGE = SOURCE_DATA / "manc_downstream_bridge_validation.csv"
BANC_SUMMARY = SOURCE_DATA / "banc_external_validation_summary.csv"


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")


def main() -> None:
    male = pd.read_csv(MALE_SUMMARY)
    manc = pd.read_csv(MANC_BRIDGE)
    banc = pd.read_csv(BANC_SUMMARY)
    matched = manc[manc["n_manc_bodyids_matched"].gt(0)].copy()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(10.8, 4.2), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.25])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])

    ordered = male.sort_values("female_route_share").reset_index(drop=True)
    y = np.arange(len(ordered))
    for idx, row in ordered.iterrows():
        ax_a.plot(
            [row["female_route_share"], row["male_route_share"]],
            [idx, idx],
            color="#c8d0d4",
            lw=1.2,
            zorder=1,
        )
    ax_a.scatter(ordered["female_route_share"], y, color="#7f8c8d", s=30, label="female FlyWire", zorder=2)
    ax_a.scatter(ordered["male_route_share"], y, color="#168aad", s=30, label="male CNS", zorder=3)
    ax_a.set_yticks(y, ordered["route"].str.replace("_", " "))
    ax_a.set_xlabel("share of selected MB-route weight")
    ax_a.set_title("MB route composition across sex")
    ax_a.legend(frameon=False, fontsize=7, loc="lower right")
    _panel_label(ax_a, "a")

    x = np.arange(len(ordered))
    for idx, row in ordered.iterrows():
        ax_b.plot(
            [idx, idx],
            [row["log_weight_spearman"], row["present_both_edge_fraction"]],
            color="#c8d0d4",
            lw=1.2,
            zorder=1,
        )
    ax_b.scatter(
        x,
        ordered["present_both_edge_fraction"],
        color="#3aa99f",
        s=28,
        label="edge present in both",
        zorder=3,
    )
    ax_b.scatter(
        x,
        ordered["log_weight_spearman"],
        color="#d09a3e",
        s=28,
        label="log-weight Spearman",
        zorder=2,
    )
    ax_b.set_xticks(x, ordered["route"].str.replace("_", " "), rotation=58, ha="right")
    ax_b.set_ylim(0, 1.05)
    ax_b.set_ylabel("fraction or correlation")
    ax_b.set_title("Matched edge conservation")
    ax_b.legend(frameon=False, fontsize=7, loc="lower left")
    _panel_label(ax_b, "b")

    top = matched.sort_values("manc_motor_output_fraction", ascending=True).tail(13).reset_index(drop=True)
    colors = np.where(top["manc_is_bilateral"].astype(bool), "#3aa99f", "#8d99a6")
    y_c = np.arange(len(top))
    ax_c.barh(y_c, top["manc_motor_output_fraction"], color=colors, height=0.68)
    ax_c.set_yticks(y_c, top["manc_cell_type"])
    ax_c.set_xlabel("fraction of MANC output onto motor neurons")
    ax_c.set_title("BANC candidates cross-checked in MANC")
    ax_c.text(
        0.99,
        0.02,
        "teal: LHS+RHS soma\ngray: side not registered",
        transform=ax_c.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#45535b",
    )
    _panel_label(ax_c, "c")

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "Fig6_external_connectome_validation"
    fig.savefig(out.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)

    metric = dict(zip(banc["metric"], banc["value"]))
    stats = {
        "male_cns_routes": int(len(male)),
        "male_cns_core_route_min_present_both": float(
            male.loc[~male["route"].isin(["5HT_to_MB", "MB_to_5HT"]), "present_both_edge_fraction"].min()
        ),
        "male_cns_core_route_min_log_weight_spearman": float(
            male.loc[~male["route"].isin(["5HT_to_MB", "MB_to_5HT"]), "log_weight_spearman"].min()
        ),
        "male_cns_all_isomorphic_mass_fraction": float(
            np.average(male["isomorphic_mass_fraction"], weights=male["weight_m"] + male["weight_f"])
        ),
        "manc_candidate_types_matched": int(len(matched)),
        "manc_matched_types_bilateral": int(matched["manc_is_bilateral"].astype(bool).sum()),
        "manc_max_motor_output_fraction": float(matched["manc_motor_output_fraction"].max()),
        "banc_metadata_neurons": int(float(metric.get("n_nodes_metadata", 0))),
        "banc_loaded_edges": int(float(metric.get("n_edges_loaded", 0))),
        "boundary": (
            "external structural conservation and downstream actionability; not replication of KC left-right "
            "5-HT-predicted input-label asymmetry"
        ),
    }
    (FIGURES / "external_connectome_validation_statistics.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
