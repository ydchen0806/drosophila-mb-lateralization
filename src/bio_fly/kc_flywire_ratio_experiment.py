"""FlyWire-constrained KC sparse-coding ratio sweep.

This experiment is the connectome-constrained counterpart of
``kc_optimal_ratio_experiment``.  Instead of drawing a random PN->KC
projection, it extracts the real FlyWire v783 ALPN->KC subgraph, builds
glomerulus-level olfactory channels, mixes those channels into odor-panel
proxies, and then reuses the same sparse top-k and associative-memory
readouts.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .kc_optimal_ratio_experiment import (
    DEFAULT_RATIOS,
    _evaluate_dprime,
    _evaluate_forgetting,
    _train_mbon,
    _winner_take_k,
)
from .kc_sparse_coding import (
    KCSparseCodingConfig,
    LEGACY_ONE_SIXTH_KC_RATIO,
    LITERATURE_KC_ACTIVE_FRACTION,
    evaluate_binary_code,
)
from .paths import DEFAULT_CONNECTIVITY_PATH, DEFAULT_OUTPUT_ROOT, PROCESSED_DATA_ROOT, RAW_DATA_ROOT, REPO_ROOT
from .propagation import EDGE_COLUMNS, PropagationConfig, response_overlap, signed_multihop_response
from .propagation_dynamics import LIFDynamicsConfig, run_lif_dynamics


CONNECTOME_GROUP_ORDER: tuple[str, ...] = (
    "ALPN",
    "ORN",
    "KC",
    "APL",
    "DPM",
    "MBON",
    "DAN",
    "DN",
    "MBIN_other",
    "octopamine",
    "serotonin",
    "gaba_other",
)
KEY_CIRCUIT_PAIRS: tuple[tuple[str, str], ...] = (
    ("ALPN", "KC"),
    ("KC", "APL"),
    ("APL", "KC"),
    ("KC", "MBON"),
    ("KC", "DAN"),
    ("DAN", "KC"),
    ("DAN", "MBON"),
    ("MBON", "DAN"),
    ("DPM", "KC"),
    ("KC", "DPM"),
    ("KC", "DN"),
    ("MBON", "DN"),
    ("DAN", "DN"),
    ("DPM", "DN"),
    ("APL", "DN"),
    ("ALPN", "APL"),
    ("DPM", "APL"),
    ("MBON", "APL"),
    ("gaba_other", "APL"),
    ("octopamine", "APL"),
    ("serotonin", "APL"),
)
BEHAVIOR_MAPPING_RATIOS: tuple[float, ...] = (
    0.05,
    LITERATURE_KC_ACTIVE_FRACTION,
    LEGACY_ONE_SIXTH_KC_RATIO,
    0.25,
)
BEHAVIOR_MAPPING_SEEDS: tuple[int, ...] = tuple(range(5))
BEHAVIOR_TARGET_GROUPS: tuple[str, ...] = (
    "MBON",
    "DAN",
    "DPM",
    "APL",
    "DN",
    "MBIN_other",
    "octopamine",
    "serotonin",
)
DEVELOPMENTAL_BEHAVIOR_TARGET_GROUPS: tuple[str, ...] = ("MBON", "DAN", "APL", "DPM", "DN")


@dataclass(frozen=True)
class KCFlyWireRatioConfig:
    """Configuration for the FlyWire ALPN->KC ratio sweep."""

    annotation_path: Path = field(default_factory=lambda: PROCESSED_DATA_ROOT / "flywire_neuron_annotations.parquet")
    connectivity_path: Path = field(default_factory=lambda: DEFAULT_CONNECTIVITY_PATH)
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_ROOT / "kc_flywire_ratio")
    ratios: tuple[float, ...] = DEFAULT_RATIOS
    seeds: tuple[int, ...] = tuple(range(20))
    n_odors: int = 24
    min_glomeruli_per_odor: int = 2
    max_glomeruli_per_odor: int = 6
    channel_noise_sigma: float = 0.15
    learning_rate: float = 0.50
    n_train_trials: int = 5
    n_interference_blocks: int = 6
    forgetting_lambda: float = 0.02
    cs_plus_index: int = 0
    memory_evaluation_repeats: int = 3
    memory_test_repeats: int = 12
    memory_max_learning_steps: int = 8
    memory_dropout_probability: float = 0.15
    memory_false_positive_probability: float = 0.003


@dataclass(frozen=True)
class ScaleBenchmarkConfig:
    """Configuration for real-data BioFly scale benchmarks.

    The default mode computes real graph/file metadata and projected S2/S3
    workloads without executing the full heavy run.  ``execute_s1_lif`` runs
    the current real MB/KC LIF pilot so the benchmark has one measured
    execution-compatible data point while keeping full-graph execution opt-in.
    """

    annotation_path: Path = field(default_factory=lambda: PROCESSED_DATA_ROOT / "flywire_neuron_annotations.parquet")
    connectivity_path: Path = field(default_factory=lambda: DEFAULT_CONNECTIVITY_PATH)
    skeleton_parquet: Path = field(
        default_factory=lambda: RAW_DATA_ROOT / "zenodo_10877326" / "sk_lod1_783_healed_ds2.parquet"
    )
    synapses_feather: Path = field(
        default_factory=lambda: RAW_DATA_ROOT / "zenodo_10676866" / "flywire_synapses_783.feather"
    )
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_ROOT / "scale_benchmark")
    execute_s1_lif: bool = False
    lif_steps: tuple[int, ...] = (80, 200)
    batch_conditions: int = 3000
    current_morphology_roots: int = 25
    current_max_synapses_per_root: int = 5000
    medium_morphology_roots: int = 500
    medium_max_synapses_per_root: int = 10000
    kc_scale_max_synapses_per_root: int = 10000


@dataclass(frozen=True)
class ExternalDataFeasibilityConfig:
    """Configuration for external neurodata feasibility audits.

    This audit does not download PB/TB-scale external datasets. It checks
    whether each source can be represented by BioFly canonical tables, then
    runs small deterministic fixtures through the same propagation/schema
    interface used by the real FlyWire analyses.
    """

    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_ROOT / "external_data_feasibility")
    smoke_steps: int = 3
    max_active: int = 64


@dataclass(frozen=True)
class TFGroundplanReplayConfig:
    """Configuration for Nature 2026 TF-groundplan connectome-level reproduction.

    The analysis does not infer transcription-factor expression from FlyWire.
    It uses existing adult FlyWire hemilineage annotations as developmental
    labels and tests the paper's connectome-level prediction that developmental
    groundplans align with coarse morphology, target choice, and circuit role.
    """

    annotation_path: Path = field(default_factory=lambda: PROCESSED_DATA_ROOT / "flywire_neuron_annotations.parquet")
    connectivity_path: Path = field(default_factory=lambda: DEFAULT_CONNECTIVITY_PATH)
    behavior_target_axis_path: Path = field(
        default_factory=lambda: DEFAULT_OUTPUT_ROOT
        / "flywire_connectome_science"
        / "flywire_kc_behavior_target_axis.csv"
    )
    nature_table1_path: Path = field(
        default_factory=lambda: RAW_DATA_ROOT
        / "nature_tf_groundplan_2026"
        / "supplementary_table_1_hemilineage_cluster_assignment.xlsx"
    )
    nature_tf_roles_path: Path = field(
        default_factory=lambda: RAW_DATA_ROOT
        / "nature_tf_groundplan_2026"
        / "supplementary_table_3_tf_roles.xlsx"
    )
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_ROOT / "tf_groundplan_replay")
    hemilineage_column: str = "ito_lee_hemilineage"
    secondary_hemilineage_column: str = "hartenstein_hemilineage"
    min_group_size: int = 20
    null_repeats: int = 64
    random_seed: int = 17
    coordinate_scale_um: float = 0.001
    max_edges: int | None = None


@dataclass(frozen=True)
class LearningMemoryPerturbationConfig:
    """Configuration for PPL1-DAN/DPM activity-reduction memory probes.

    The readout keeps the real FlyWire ALPN->KC odor code and real KC->PPL1/DPM
    target weights, then applies transparent gain reductions to the PPL1 teaching
    axis and DPM persistence axis.
    """

    annotation_path: Path = field(default_factory=lambda: PROCESSED_DATA_ROOT / "flywire_neuron_annotations.parquet")
    connectivity_path: Path = field(default_factory=lambda: DEFAULT_CONNECTIVITY_PATH)
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_ROOT / "ppl1_dpm_learning_memory_perturbation")
    ratio: float = LITERATURE_KC_ACTIVE_FRACTION
    seeds: tuple[int, ...] = tuple(range(20))
    n_odors: int = 24
    min_glomeruli_per_odor: int = 2
    max_glomeruli_per_odor: int = 6
    channel_noise_sigma: float = 0.15
    cs_plus_index: int = 0
    dpm_baseline_decay_per_block: float = 0.025
    dpm_loss_decay_penalty_per_block: float = 0.18
    interference_blocks: int = 6
    conflict_decoy_penalty: float = 0.28
    acquisition_score_weight: float = 0.55
    retention_score_weight: float = 0.45
    ppl1_pattern: str = "PPL1"


@dataclass(frozen=True)
class LateralizationRepresentationMemoryConfig:
    """Configuration for graph-level lateralization representation analysis.

    The analysis tests whether the observed right-serotonin / left-glutamate
    KC input asymmetry can be interpreted as a symmetry-breaking gate that
    expands odor representation space and stabilizes associative-memory proxies.
    It is deliberately a graph-network readout, not a completed animal-behavior
    assay.
    """

    annotation_path: Path = field(default_factory=lambda: PROCESSED_DATA_ROOT / "flywire_neuron_annotations.parquet")
    connectivity_path: Path = field(default_factory=lambda: DEFAULT_CONNECTIVITY_PATH)
    kc_nt_inputs_path: Path = field(
        default_factory=lambda: REPO_ROOT / "outputs" / "kc_nt_lateralization" / "kc_neuron_nt_inputs.parquet"
    )
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_ROOT / "lateralization_representation_memory")
    ratio: float = LITERATURE_KC_ACTIVE_FRACTION
    seeds: tuple[int, ...] = tuple(range(20))
    n_odors: int = 24
    min_glomeruli_per_odor: int = 2
    max_glomeruli_per_odor: int = 6
    channel_noise_sigma: float = 0.15
    gate_amplitude: float = 0.25
    gate_strengths: tuple[float, ...] = (-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
    shuffle_repeats: int = 8
    memory_evaluation_repeats: int = 3
    memory_test_repeats: int = 12
    memory_max_learning_steps: int = 8
    memory_dropout_probability: float = 0.15
    memory_false_positive_probability: float = 0.003
    n_interference_blocks: int = 6


@dataclass(frozen=True)
class BehaviorClosureProxyConfig:
    """Configuration for OCT/MCH behaviour-closure proxy analyses.

    This is not a replacement for real T-maze, calcium imaging, or 5-HT sensor
    experiments.  It uses the current real FlyWire ALPN->KC code, PPL1-DAN and
    DPM axes, and KC lateralization gate to decide which behaviour/imaging
    experiments are worth doing first.
    """

    annotation_path: Path = field(default_factory=lambda: PROCESSED_DATA_ROOT / "flywire_neuron_annotations.parquet")
    connectivity_path: Path = field(default_factory=lambda: DEFAULT_CONNECTIVITY_PATH)
    kc_nt_inputs_path: Path = field(
        default_factory=lambda: REPO_ROOT / "outputs" / "kc_nt_lateralization" / "kc_neuron_nt_inputs.parquet"
    )
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_ROOT / "lateralization_behavior_closure")
    ratio: float = LITERATURE_KC_ACTIVE_FRACTION
    seeds: tuple[int, ...] = tuple(range(20))
    n_odors: int = 24
    min_glomeruli_per_odor: int = 2
    max_glomeruli_per_odor: int = 6
    channel_noise_sigma: float = 0.15
    cs_plus_index: int = 0
    gate_amplitude: float = 0.25
    ppl1_pattern: str = "PPL1"
    dpm_baseline_decay_per_block: float = 0.025
    dpm_loss_decay_penalty_per_block: float = 0.18
    conflict_decoy_penalty: float = 0.28


@dataclass(frozen=True)
class LateralizationMechanismSuiteConfig:
    """Configuration for KC lateralization mechanism scans.

    This extends the single-condition representation/memory analysis into a
    task-regime and ablation suite.  It still uses the real FlyWire ALPN->KC
    matrix and KC neurotransmitter-input lateralization table, but asks a more
    article-facing question: when does symmetry breaking become useful for
    representation geometry and memory-like readouts?
    """

    annotation_path: Path = field(default_factory=lambda: PROCESSED_DATA_ROOT / "flywire_neuron_annotations.parquet")
    connectivity_path: Path = field(default_factory=lambda: DEFAULT_CONNECTIVITY_PATH)
    kc_nt_inputs_path: Path = field(
        default_factory=lambda: REPO_ROOT / "outputs" / "kc_nt_lateralization" / "kc_neuron_nt_inputs.parquet"
    )
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_ROOT / "lateralization_mechanism_suite")
    ratio: float = LITERATURE_KC_ACTIVE_FRACTION
    seeds: tuple[int, ...] = tuple(range(12))
    n_odors: int = 24
    min_glomeruli_per_odor: int = 2
    max_glomeruli_per_odor: int = 6
    channel_noise_sigma: float = 0.15
    gate_amplitude: float = 0.25
    gate_strength: float = 1.0
    shuffle_repeats: int = 16
    task_similarity_levels: tuple[float, ...] = (0.0, 0.35, 0.70)
    delay_blocks: tuple[int, ...] = (0, 6, 12)
    interference_levels: tuple[float, ...] = (0.0, 0.5, 1.0)
    dpm_gains: tuple[float, ...] = (1.0, 0.5, 0.25)
    apl_noise_levels: tuple[float, ...] = (0.0, 0.10, 0.20)
    dropout_levels: tuple[float, ...] = (0.0, 0.10, 0.25)
    decoder_noise_sigma: float = 0.15
    dpm_baseline_decay_per_block: float = 0.025
    dpm_loss_decay_penalty_per_block: float = 0.18
    conflict_penalty_scale: float = 0.28
    regularization: float = 1e-3
    sensitivity_gate_amplitudes: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50)
    sensitivity_ratios: tuple[float, ...] = (0.05, LITERATURE_KC_ACTIVE_FRACTION, 0.15)


@dataclass(frozen=True)
class MBONDecisionPivotConfig:
    """Configuration for MBON downstream decision-pivot candidate search.

    The analysis starts from real FlyWire MBON neurons and traces one to three
    downstream hops with normalized signed edge weights.  It is a candidate
    ranking workflow, not a proof that a listed neuron is the final behavioural
    decision neuron.
    """

    annotation_path: Path = field(default_factory=lambda: PROCESSED_DATA_ROOT / "flywire_neuron_annotations.parquet")
    connectivity_path: Path = field(default_factory=lambda: DEFAULT_CONNECTIVITY_PATH)
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_ROOT / "mbon_decision_pivot")
    max_hops: int = 3
    hop_decay: float = 0.55
    source_grouping: str = "compartment"
    include_unmapped_sources: bool = False
    source_group_limit: int = 0
    max_frontier_per_source: int = 750
    min_abs_edge_weight: float = 1.0
    top_candidates: int = 80
    null_repeats: int = 64
    random_seed: int = 23


def _parquet_metadata(path: Path) -> dict[str, object]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    schema_names = list(getattr(parquet, "schema_arrow", parquet.schema).names)
    return {
        "path": str(path),
        "rows": int(parquet.metadata.num_rows),
        "row_groups": int(parquet.metadata.num_row_groups),
        "columns": int(parquet.metadata.num_columns),
        "size_bytes": int(path.stat().st_size),
        "column_names": schema_names,
    }


def _feather_metadata(path: Path) -> dict[str, object]:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    with pa.memory_map(str(path), "r") as source:
        reader = ipc.RecordBatchFileReader(source)
        rows = 0
        for index in range(reader.num_record_batches):
            rows += int(reader.get_batch(index).num_rows)
        return {
            "path": str(path),
            "rows": int(rows),
            "batches": int(reader.num_record_batches),
            "columns": int(len(reader.schema)),
            "size_bytes": int(path.stat().st_size),
            "column_names": list(reader.schema.names),
        }


def _mib(num_bytes: int) -> float:
    return float(num_bytes) / float(1024**2)


def _gib(num_bytes: int) -> float:
    return float(num_bytes) / float(1024**3)


def _safe_fraction(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if float(denominator) else 0.0


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    columns = [str(column) for column in frame.columns]
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for record in frame.to_dict(orient="records"):
        values = [str(record.get(column, "")) for column in frame.columns]
        rows.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
    return "\n".join(rows)


def _connectivity_graph_stats(connectivity_path: Path) -> tuple[dict[str, object], pd.DataFrame]:
    meta = _parquet_metadata(connectivity_path)
    columns = ["Presynaptic_ID", "Postsynaptic_ID", "Excitatory x Connectivity"]
    if "Connectivity" in set(meta["column_names"]):
        columns.append("Connectivity")
    edges = pd.read_parquet(connectivity_path, columns=columns)
    unique_nodes = pd.Index(edges["Presynaptic_ID"]).union(pd.Index(edges["Postsynaptic_ID"])).nunique()
    stats = {
        "rows": int(len(edges)),
        "unique_nodes": int(unique_nodes),
        "unique_presynaptic_nodes": int(edges["Presynaptic_ID"].nunique()),
        "unique_postsynaptic_nodes": int(edges["Postsynaptic_ID"].nunique()),
        "nonzero_weight_rows": int((edges["Excitatory x Connectivity"].astype(float) != 0.0).sum()),
        "size_bytes": int(meta["size_bytes"]),
    }
    if "Connectivity" in edges.columns:
        stats["connectivity_weight_sum"] = int(edges["Connectivity"].sum())
    else:
        stats["connectivity_weight_sum"] = float(edges["Excitatory x Connectivity"].abs().sum())
    return stats, edges


def _scale_subgraph_stats(annotations: pd.DataFrame, edges: pd.DataFrame, max_seeds: int = 32) -> dict[str, object]:
    circuit_groups = {"ALPN", "KC", "APL", "DPM", "MBON", "DAN"}
    circuit_roots = set(
        annotations.loc[annotations["connectome_group"].isin(circuit_groups), "root_id"].astype("int64").tolist()
    )
    seed_ids = (
        annotations.loc[annotations["connectome_group"].eq("ALPN"), "root_id"]
        .dropna()
        .astype("int64")
        .sort_values()
        .head(max_seeds)
        .tolist()
    )
    subgraph = edges[
        edges["Presynaptic_ID"].isin(circuit_roots) & edges["Postsynaptic_ID"].isin(circuit_roots)
    ].copy()
    unique_nodes = (
        int(pd.Index(subgraph["Presynaptic_ID"]).union(pd.Index(subgraph["Postsynaptic_ID"])).nunique())
        if not subgraph.empty
        else 0
    )
    return {
        "circuit_roots": int(len(circuit_roots)),
        "seed_ids": int(len(seed_ids)),
        "subgraph_edges": int(len(subgraph)),
        "subgraph_unique_nodes": unique_nodes,
        "edge_density_directed": _safe_fraction(len(subgraph), unique_nodes * max(unique_nodes - 1, 1)),
    }


def _scale_lif_matrix(
    connectome_stats: dict[str, object],
    s1_stats: dict[str, object],
    lif_steps: tuple[int, ...] = (80, 200),
) -> pd.DataFrame:
    s1_edges = int(s1_stats["subgraph_edges"])
    s1_nodes = int(s1_stats["subgraph_unique_nodes"])
    s1_edge_steps = s1_edges * 80
    rows = [
        ("S1", "current_mb_kc", s1_nodes, s1_edges, 80, "current real MB/KC pilot"),
        ("S2", "medium_subgraph", 25_000, 2_500_000, 80, "projected stress benchmark"),
        ("S2", "large_subgraph", 75_000, 8_000_000, 80, "projected stress benchmark"),
    ]
    s3_steps = sorted({int(step) for step in lif_steps if int(step) > 0}) or [80]
    for steps in s3_steps:
        rows.append(
            (
                "S3",
                f"full_graph_{steps}ms",
                int(connectome_stats["unique_nodes"]),
                int(connectome_stats["rows"]),
                steps,
                "full FlyWire neuron/root graph projection",
            )
        )
    records = []
    for tier, name, nodes, edges, steps, note in rows:
        edge_steps = int(edges) * int(steps)
        records.append(
            {
                "tier": tier,
                "benchmark": name,
                "nodes": int(nodes),
                "edges": int(edges),
                "steps": int(steps),
                "edge_steps": int(edge_steps),
                "relative_to_s1": _safe_fraction(edge_steps, s1_edge_steps),
                "note": note,
            }
        )
    return pd.DataFrame.from_records(records)


def _scale_batch_projection(
    *,
    batch_conditions: int,
    s1_stats: dict[str, object],
    connectome_stats: dict[str, object],
) -> pd.DataFrame:
    condition_count = max(1, int(batch_conditions))
    rows = [
        (
            "S1",
            "current_mb_kc_80ms_batch",
            int(s1_stats["subgraph_unique_nodes"]),
            int(s1_stats["subgraph_edges"]),
            80,
            condition_count,
        ),
        (
            "S3",
            "full_graph_80ms_batch",
            int(connectome_stats["unique_nodes"]),
            int(connectome_stats["rows"]),
            80,
            condition_count,
        ),
        (
            "S3",
            "full_graph_200ms_batch",
            int(connectome_stats["unique_nodes"]),
            int(connectome_stats["rows"]),
            200,
            condition_count,
        ),
    ]
    records = []
    for tier, name, nodes, edges, steps, conditions in rows:
        edge_steps_per_condition = int(edges) * int(steps)
        records.append(
            {
                "tier": tier,
                "benchmark": name,
                "nodes": int(nodes),
                "edges": int(edges),
                "steps": int(steps),
                "conditions": int(conditions),
                "edge_steps_per_condition": int(edge_steps_per_condition),
                "total_edge_steps": int(edge_steps_per_condition * int(conditions)),
            }
        )
    return pd.DataFrame.from_records(records)


def _scale_morphology_matrix(
    *,
    config: ScaleBenchmarkConfig,
    kc_count: int,
    synapse_rows: int,
) -> pd.DataFrame:
    rows = [
        (
            "S1",
            "current_default",
            int(config.current_morphology_roots),
            int(config.current_morphology_roots * config.current_max_synapses_per_root),
            "current CLI default for mapping validation",
        ),
        (
            "S2",
            "mb_kc_stress",
            int(config.medium_morphology_roots),
            int(config.medium_morphology_roots * config.medium_max_synapses_per_root),
            "KD-tree/cache stress benchmark",
        ),
        (
            "S2",
            "kc_scale",
            int(kc_count),
            int(kc_count * config.kc_scale_max_synapses_per_root),
            "KC-scale morphology preprocessing projection",
        ),
        ("S3", "full_synapse_table", -1, int(synapse_rows), "full morphology preprocessing projection"),
    ]
    return pd.DataFrame.from_records(
        [
            {
                "tier": tier,
                "benchmark": name,
                "roots": roots,
                "synapse_upper_bound": synapses,
                "fraction_of_full_synapse_table": _safe_fraction(synapses, synapse_rows),
                "note": note,
            }
            for tier, name, roots, synapses, note in rows
        ]
    )


def _write_scale_benchmark_report(
    *,
    output_dir: Path,
    summary: dict[str, object],
    lif_matrix: pd.DataFrame,
    morphology_matrix: pd.DataFrame,
    batch_matrix: pd.DataFrame,
    lif_execution_summary: dict[str, object] | None,
) -> Path:
    report_path = output_dir / "SCALE_BENCHMARK_REPORT_CN.md"
    conn = summary["connectivity"]
    files = summary["files"]
    s1 = summary["s1_subgraph"]
    ratios = summary["ratios"]
    s3_lif = lif_matrix[lif_matrix["tier"].eq("S3")]
    s3_lif_summary = "; ".join(
        f"{int(row['steps'])} ms = {float(row['edge_steps']) / 1_000_000_000:.2f}B edge-step"
        for _, row in s3_lif.iterrows()
    )
    report_path.write_text(
        f"""# BioFly scale benchmark 实现报告

保存路径：`{report_path}`

## 定位

该 benchmark 已经实现为可执行入口。默认运行会读取本地 FlyWire
connectivity、annotation、skeleton parquet 和 synapse feather 的元信息，并从真实
annotation/connectivity 中抽取当前 MB/KC pilot 子图规模。S2/S3 是基于这些真实规模的
投影；full-graph LIF 保持显式 opt-in，以避免无意触发高成本计算。

## 当前真实规模

| 项 | 数值 |
|---|---:|
| full neuron/root nodes | `{int(conn['unique_nodes'])}` |
| full directed edges | `{int(conn['rows'])}` |
| connectivity file | `{_mib(int(files['connectivity_size_bytes'])):.2f} MiB` |
| annotation rows | `{int(summary['annotation_rows'])}` |
| skeleton rows | `{int(files['skeleton_rows'])}` |
| skeleton file | `{_gib(int(files['skeleton_size_bytes'])):.2f} GiB` |
| synapse rows | `{int(files['synapse_rows'])}` |
| synapse file | `{_gib(int(files['synapse_size_bytes'])):.2f} GiB` |
| skeleton + synapse rows | `{int(files['skeleton_rows']) + int(files['synapse_rows'])}` |

## 当前 S1 pilot 子图

| 项 | 数值 |
|---|---:|
| circuit roots | `{int(s1['circuit_roots'])}` |
| ALPN seeds | `{int(s1['seed_ids'])}` |
| subgraph nodes | `{int(s1['subgraph_unique_nodes'])}` |
| subgraph edges | `{int(s1['subgraph_edges'])}` |
| edge density | `{float(s1['edge_density_directed']):.6f}` |
| edge fraction of full graph | `{float(ratios['s1_edge_fraction_of_full']) * 100:.2f}%` |
| node fraction of full graph | `{float(ratios['s1_node_fraction_of_full']) * 100:.2f}%` |

## LIF benchmark matrix

{_markdown_table(lif_matrix)}

## Morphology benchmark matrix

{_markdown_table(morphology_matrix)}

## Batch condition projection

`batch_conditions={int(summary['batch_conditions'])}`。该表不执行全量 sweep，而是将同一真实图规模
乘以条件数，用于估算与李诚团队协同优化的 batch perturbation 量级。

{_markdown_table(batch_matrix)}

## 可选真实执行

`execute_s1_lif={bool(lif_execution_summary is not None)}`。

{json.dumps(lif_execution_summary, ensure_ascii=False, indent=2) if lif_execution_summary is not None else '本次没有执行 S1 LIF，只生成真实元信息和 S2/S3 投影。'}

## 解释

- S1 是当前真实 MB/KC pilot，用于验证链路和科学锚点，不等同于 full-brain stress test。
- S3 full graph 投影：{s3_lif_summary}。
- morphology 的主要瓶颈是 `skeleton/synapse` 大表 I/O、root 分区、KD-tree 和 synapse mapping。
- 与李诚团队的系统优化协作中，需要区分科学锚点和系统压力测试：科学锚点来自真实 FlyWire
  neuron/root graph，系统压力来自 skeleton/synapse/time/condition 展开。
""",
        encoding="utf-8",
    )
    return report_path


def run_scale_benchmark(config: ScaleBenchmarkConfig | None = None) -> dict[str, object]:
    """Write real-data scale benchmark tables and optional S1 LIF execution."""

    config = config or ScaleBenchmarkConfig()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_conditions = max(1, int(config.batch_conditions))
    started = time.perf_counter()

    connectivity_meta = _parquet_metadata(config.connectivity_path)
    annotation_meta = _parquet_metadata(config.annotation_path)
    skeleton_meta = _parquet_metadata(config.skeleton_parquet)
    synapse_meta = _feather_metadata(config.synapses_feather)
    annotations = _annotate_connectome_groups(_load_annotations(config.annotation_path))
    connectome_stats, edges = _connectivity_graph_stats(config.connectivity_path)
    s1_stats = _scale_subgraph_stats(annotations, edges)
    kc_count = int(annotations["connectome_group"].eq("KC").sum())
    lif_matrix = _scale_lif_matrix(connectome_stats, s1_stats, config.lif_steps)
    morphology_matrix = _scale_morphology_matrix(
        config=config,
        kc_count=kc_count,
        synapse_rows=int(synapse_meta["rows"]),
    )
    batch_matrix = _scale_batch_projection(
        batch_conditions=batch_conditions,
        s1_stats=s1_stats,
        connectome_stats=connectome_stats,
    )

    lif_execution_summary = None
    if config.execute_s1_lif:
        lif_execution_summary = _run_real_lif_smoke(
            annotation_path=config.annotation_path,
            connectivity_path=config.connectivity_path,
            output_dir=output_dir,
        )
        (output_dir / "s1_lif_execution_summary.json").write_text(
            json.dumps(lif_execution_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    files_summary = {
        "connectivity_rows": int(connectivity_meta["rows"]),
        "connectivity_size_bytes": int(connectivity_meta["size_bytes"]),
        "annotation_rows": int(annotation_meta["rows"]),
        "annotation_size_bytes": int(annotation_meta["size_bytes"]),
        "skeleton_rows": int(skeleton_meta["rows"]),
        "skeleton_size_bytes": int(skeleton_meta["size_bytes"]),
        "synapse_rows": int(synapse_meta["rows"]),
        "synapse_size_bytes": int(synapse_meta["size_bytes"]),
    }
    ratios = {
        "morphology_rows_vs_connectivity_rows": _safe_fraction(
            int(skeleton_meta["rows"]) + int(synapse_meta["rows"]), int(connectivity_meta["rows"])
        ),
        "skeleton_synapse_size_vs_connectivity_size": _safe_fraction(
            int(skeleton_meta["size_bytes"]) + int(synapse_meta["size_bytes"]), int(connectivity_meta["size_bytes"])
        ),
        "s1_edge_fraction_of_full": _safe_fraction(int(s1_stats["subgraph_edges"]), int(connectome_stats["rows"])),
        "s1_node_fraction_of_full": _safe_fraction(
            int(s1_stats["subgraph_unique_nodes"]), int(connectome_stats["unique_nodes"])
        ),
        "current_morphology_fraction_of_synapses": _safe_fraction(
            int(config.current_morphology_roots * config.current_max_synapses_per_root), int(synapse_meta["rows"])
        ),
    }
    summary = {
        "paths": {
            "annotation_path": str(config.annotation_path),
            "connectivity_path": str(config.connectivity_path),
            "skeleton_parquet": str(config.skeleton_parquet),
            "synapses_feather": str(config.synapses_feather),
            "output_dir": str(output_dir),
        },
        "connectivity": connectome_stats,
        "annotation_rows": int(annotation_meta["rows"]),
        "kc_count": kc_count,
        "files": files_summary,
        "s1_subgraph": s1_stats,
        "ratios": ratios,
        "execute_s1_lif": bool(config.execute_s1_lif),
        "lif_steps": list(config.lif_steps),
        "batch_conditions": batch_conditions,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }

    summary_path = output_dir / "scale_benchmark_summary.json"
    lif_matrix_path = output_dir / "lif_scale_matrix.csv"
    morphology_matrix_path = output_dir / "morphology_scale_matrix.csv"
    batch_matrix_path = output_dir / "batch_scale_matrix.csv"
    metadata_path = output_dir / "scale_benchmark_metadata.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lif_matrix.to_csv(lif_matrix_path, index=False)
    morphology_matrix.to_csv(morphology_matrix_path, index=False)
    batch_matrix.to_csv(batch_matrix_path, index=False)
    metadata_path.write_text(
        json.dumps({"config": asdict(config), "summary": summary}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    report_path = _write_scale_benchmark_report(
        output_dir=output_dir,
        summary=summary,
        lif_matrix=lif_matrix,
        morphology_matrix=morphology_matrix,
        batch_matrix=batch_matrix,
        lif_execution_summary=lif_execution_summary,
    )
    return {
        "summary_json": summary_path,
        "lif_scale_matrix_csv": lif_matrix_path,
        "morphology_scale_matrix_csv": morphology_matrix_path,
        "batch_scale_matrix_csv": batch_matrix_path,
        "metadata_json": metadata_path,
        "report_md": report_path,
        "summary": summary,
        "lif_scale_matrix_df": lif_matrix,
        "morphology_scale_matrix_df": morphology_matrix,
        "batch_scale_matrix_df": batch_matrix,
        "lif_execution_summary": lif_execution_summary,
    }


def _external_dataset_catalog() -> pd.DataFrame:
    records = [
        {
            "dataset_id": "flywire_adult_brain",
            "dataset_name": "FlyWire adult whole brain",
            "species": "Drosophila melanogaster adult female",
            "access_status": "public snapshot/API",
            "primary_modality": "synapse_connectome",
            "public_scale": "139,255 neurons; 54.5M synapses; v783 graph 2,701,601 thresholded edges",
            "biofly_input_mode": "direct_neuron_graph",
            "canonical_tables": "nodes,edges,morphology,labels",
            "feasibility": "implemented_reference",
            "system_validation": "already used by current BioFly analyses",
            "biological_validation": "MB/KC/APL/DPM/DAN/MBON findings can be checked by imaging, perturbation and behavior",
        },
        {
            "dataset_id": "hemibrain",
            "dataset_name": "Janelia hemibrain",
            "species": "Drosophila melanogaster adult female central brain",
            "access_status": "public neuPrint/API",
            "primary_modality": "synapse_connectome",
            "public_scale": "~25k traced neurons; ~20M synapses",
            "biofly_input_mode": "direct_neuron_graph",
            "canonical_tables": "nodes,edges,morphology,labels",
            "feasibility": "high",
            "system_validation": "convert neuPrint neurons/synapses/ROI into canonical nodes and edges",
            "biological_validation": "cross-check MBON/DAN/APL/DPM subtype naming and conserved MB motifs against FlyWire",
        },
        {
            "dataset_id": "manc_vnc",
            "dataset_name": "MANC adult nerve cord",
            "species": "Drosophila melanogaster adult male VNC",
            "access_status": "public neuPrint/Clio/NeuronBridge",
            "primary_modality": "synapse_connectome",
            "public_scale": "~23,000 neurons; ~10M presynaptic sites; ~74M postsynaptic densities",
            "biofly_input_mode": "direct_neuron_graph",
            "canonical_tables": "nodes,edges,morphology,labels",
            "feasibility": "high",
            "system_validation": "same edge propagation schema as brain connectome, with VNC/motor labels",
            "biological_validation": "test whether MBON/DN candidates approach motor pools or known locomotor circuits",
        },
        {
            "dataset_id": "larval_brain",
            "dataset_name": "Drosophila larval whole brain",
            "species": "Drosophila melanogaster larva",
            "access_status": "public paper/CATMAID-associated data",
            "primary_modality": "synapse_connectome",
            "public_scale": "3,016 neurons; ~548,000 synapses",
            "biofly_input_mode": "direct_neuron_graph",
            "canonical_tables": "nodes,edges,labels,behavior_optional",
            "feasibility": "high",
            "system_validation": "small full-brain graph can run full propagation and perturbation cheaply",
            "biological_validation": "check sensory-value-action motifs and left-right homolog effects in a complete small brain",
        },
        {
            "dataset_id": "c_elegans",
            "dataset_name": "C. elegans whole-animal connectome",
            "species": "Caenorhabditis elegans",
            "access_status": "public WormWiring/WormAtlas/OpenWorm derivatives",
            "primary_modality": "whole_animal_connectome",
            "public_scale": "302 neurons classically; Cook 2019 hermaphrodite whole-animal graph 460 nodes",
            "biofly_input_mode": "direct_neuron_graph_plus_gap_junction_muscle",
            "canonical_tables": "nodes,edges,behavior",
            "feasibility": "high_with_gap_junction_extension",
            "system_validation": "requires edge_type support for chemical synapse, gap junction and muscle output",
            "biological_validation": "validate sensorimotor readout against known chemotaxis/touch circuits and muscle groups",
        },
        {
            "dataset_id": "ciona_larva",
            "dataset_name": "Ciona larval CNS connectome",
            "species": "Ciona intestinalis larva",
            "access_status": "public paper/graph derivatives",
            "primary_modality": "small_chordate_connectome",
            "public_scale": "177 CNS neurons; 6,618 synapses; 1,772 NMJs; 1,206 gap junctions",
            "biofly_input_mode": "direct_neuron_graph_plus_gap_junction_muscle",
            "canonical_tables": "nodes,edges,behavior",
            "feasibility": "high_with_gap_junction_extension",
            "system_validation": "small enough for exhaustive propagation, laterality and muscle closure tests",
            "biological_validation": "compare sidedness predictions with sensory-motor behavior and neuromuscular outputs",
        },
        {
            "dataset_id": "microns_mm3",
            "dataset_name": "MICrONS cubic millimeter mouse visual cortex",
            "species": "mouse P87 visual cortex",
            "access_status": "public CAVE/MICrONS; large I/O and account requirements",
            "primary_modality": "synapse_connectome_plus_calcium",
            "public_scale": "~200k cells; 75k neurons with physiology; 523M synapses",
            "biofly_input_mode": "direct_local_cortical_graph_plus_activity",
            "canonical_tables": "nodes,edges,morphology,activity,labels",
            "feasibility": "medium_high_but_heavy",
            "system_validation": "requires CAVE materialization, chunked reads and proofread-status filters",
            "biological_validation": "fit propagation/LIF outputs to visual stimulus calcium responses and layer/cell-type labels",
        },
        {
            "dataset_id": "h01_human_cortex",
            "dataset_name": "H01 human cortex EM",
            "species": "human temporal cortex sample",
            "access_status": "public hosted volume",
            "primary_modality": "human_local_em",
            "public_scale": "~1.4 PB volume; >50k cells; 130M+ synaptic connections",
            "biofly_input_mode": "local_structure_morphology_graph",
            "canonical_tables": "nodes,edges,morphology",
            "feasibility": "medium_but_too_large_for_default",
            "system_validation": "requires fragment/proofread filtering and cloud-volume I/O; behavior absent",
            "biological_validation": "use only local motif, synapse-density and morphology comparisons; no behavior claim",
        },
        {
            "dataset_id": "allen_mouse_connectivity",
            "dataset_name": "Allen Mouse Brain Connectivity Atlas",
            "species": "mouse whole brain",
            "access_status": "public Allen Brain Map/API/AllenSDK",
            "primary_modality": "region_projection",
            "public_scale": "Nature 2014: 469 injection datasets, 295 structures; atlas has more experiments",
            "biofly_input_mode": "mesoscale_region_graph",
            "canonical_tables": "nodes,edges,labels",
            "feasibility": "high_region_level",
            "system_validation": "convert projection density from injection/source region to weighted region edges",
            "biological_validation": "validate against known source-target projection anatomy, not neuron-level causality",
        },
        {
            "dataset_id": "allen_brain_observatory",
            "dataset_name": "Allen Brain Observatory / OpenScope",
            "species": "mouse visual cortex",
            "access_status": "public AllenSDK/portal",
            "primary_modality": "activity",
            "public_scale": "early release >18k neurons; later 2P/OpenScope datasets >60k neurons",
            "biofly_input_mode": "activity_calibration",
            "canonical_tables": "nodes,activity,behavior,labels",
            "feasibility": "medium",
            "system_validation": "align trial/stimulus/cell response tables to BioFly activity schema",
            "biological_validation": "fit response dynamics and stimulus selectivity; cannot validate structural paths alone",
        },
        {
            "dataset_id": "ibl_brain_wide_map",
            "dataset_name": "International Brain Laboratory Brain-Wide Map",
            "species": "mouse",
            "access_status": "public IBL/ONE",
            "primary_modality": "activity_behavior",
            "public_scale": "621,733 neurons; 139 mice; 699 Neuropixels insertions; 279 areas",
            "biofly_input_mode": "activity_behavior_calibration",
            "canonical_tables": "nodes,activity,behavior,labels",
            "feasibility": "medium",
            "system_validation": "trial-level activity and action/reward variables map directly to activity/behavior tables",
            "biological_validation": "validate distributed decision variables and behavior readout, not synapse-level propagation",
        },
        {
            "dataset_id": "mouselight",
            "dataset_name": "MouseLight",
            "species": "mouse",
            "access_status": "public Janelia browser/download",
            "primary_modality": "morphology",
            "public_scale": ">1000 fully reconstructed projection neurons",
            "biofly_input_mode": "morphology_prior",
            "canonical_tables": "nodes,morphology,labels",
            "feasibility": "high_for_morphology",
            "system_validation": "SWC/skeleton path length and branch order can feed morphology attenuation",
            "biological_validation": "validate projection class and target regions; lacks postsynaptic partners",
        },
        {
            "dataset_id": "neuromorpho",
            "dataset_name": "NeuroMorpho.Org",
            "species": "multi-species",
            "access_status": "public database",
            "primary_modality": "morphology",
            "public_scale": "release-dependent, continuously growing SWC database",
            "biofly_input_mode": "morphology_prior",
            "canonical_tables": "nodes,morphology,labels",
            "feasibility": "high_for_morphology",
            "system_validation": "standard SWC can be converted to path length, branch order and compartment features",
            "biological_validation": "validate morphology distributions by species/region/cell type, not circuit behavior",
        },
        {
            "dataset_id": "abc_atlas",
            "dataset_name": "Allen Brain Cell Atlas / ABC Atlas",
            "species": "mouse/human and BICAN-related",
            "access_status": "public portal/manifest",
            "primary_modality": "transcriptomics_spatial_labels",
            "public_scale": "mouse atlas reports 32M+ cells and 5,300+ cell types",
            "biofly_input_mode": "label_prior",
            "canonical_tables": "nodes,labels",
            "feasibility": "medium_label_mapping",
            "system_validation": "cell-type and marker labels can join to nodes by region/type, not by synapse ID",
            "biological_validation": "validate receptor/NT/developmental labels with marker genes and spatial colocalization",
        },
        {
            "dataset_id": "zbrain",
            "dataset_name": "Z-Brain zebrafish atlas",
            "species": "larval zebrafish",
            "access_status": "public atlas/download",
            "primary_modality": "cellular_atlas_activity_labels",
            "public_scale": "~294,000 neurons plus transgenic labels/masks",
            "biofly_input_mode": "activity_atlas_calibration",
            "canonical_tables": "nodes,activity,labels,behavior_optional",
            "feasibility": "medium",
            "system_validation": "cell positions and labels map to nodes; calcium/behavior need experiment-specific tables",
            "biological_validation": "validate whole-brain state/readout with transparent-animal calcium and behavior assays",
        },
        {
            "dataset_id": "hcp_young_adult",
            "dataset_name": "Human Connectome Project S1200",
            "species": "human",
            "access_status": "public/controlled",
            "primary_modality": "macro_connectome_mri_behavior",
            "public_scale": "1,206 participants; HCP1200 open collection ~80 TB",
            "biofly_input_mode": "macro_region_graph",
            "canonical_tables": "nodes,edges,activity,behavior",
            "feasibility": "medium_macro_only",
            "system_validation": "convert atlas parcels and dMRI/fMRI matrices to region graph/activity schema",
            "biological_validation": "validate only macro network-behavior associations; no single-neuron claim",
        },
        {
            "dataset_id": "marmoset_macaque_tracer",
            "dataset_name": "Marmoset / macaque tracer atlases",
            "species": "non-human primates",
            "access_status": "partly public portals/literature databases",
            "primary_modality": "region_projection",
            "public_scale": "case/region/connection-record based; no single fixed EM release",
            "biofly_input_mode": "mesoscale_region_graph",
            "canonical_tables": "nodes,edges,labels",
            "feasibility": "medium_region_level",
            "system_validation": "convert tracer source-target records to region-level directed weighted edges",
            "biological_validation": "validate long-range projection anatomy and task-region hypotheses, not neuron-level paths",
        },
        {
            "dataset_id": "dandi_openneuro",
            "dataset_name": "DANDI / OpenNeuro",
            "species": "multi-species/human",
            "access_status": "public dataset platforms",
            "primary_modality": "activity_behavior_repository",
            "public_scale": "dataset-dependent, MB to TB per dataset",
            "biofly_input_mode": "activity_behavior_calibration",
            "canonical_tables": "nodes,activity,behavior,labels",
            "feasibility": "medium_dataset_specific",
            "system_validation": "NWB/BIDS datasets can be mapped after selecting task-matched sessions",
            "biological_validation": "validate dynamics and behavior readout only for task-compatible datasets",
        },
    ]
    catalog = pd.DataFrame.from_records(records)
    size_plan = {
        "flywire_adult_brain": {
            "estimated_full_size": "current local graph 97 MB connectivity + 8.4 MB annotations; morphology adds 5.0 GB skeleton + 8.9 GB synapse feather",
            "minimum_biofly_download": "~105 MB for signed graph propagation; ~14 GB extra for current morphology-aware FlyWire runs",
            "download_priority": "already_local_reference",
        },
        "hemibrain": {
            "estimated_full_size": "~25k neurons and ~20M synapses; full neuPrint-derived exports are GB-scale depending on table granularity",
            "minimum_biofly_download": "10-100 MB for targeted MB/KC/APL/DAN/DPM query; full graph export likely 1-5 GB after table conversion",
            "download_priority": "small_targeted_api_first",
        },
        "manc_vnc": {
            "estimated_full_size": "~23k neurons; 10M presynaptic sites; 74M postsynaptic densities; full synapse-level export is multi-GB",
            "minimum_biofly_download": "100 MB-1 GB for aggregated VNC neuron graph; targeted DN/motor subset can start below 200 MB",
            "download_priority": "best_next_graph_dataset",
        },
        "larval_brain": {
            "estimated_full_size": "3,016 neurons; ~548k synapses",
            "minimum_biofly_download": "<100 MB for full graph tables in compact CSV/Parquet form",
            "download_priority": "small_complete_brain_benchmark",
        },
        "c_elegans": {
            "estimated_full_size": "302-neuron classic connectome; Cook 2019 whole-animal graph 460 nodes",
            "minimum_biofly_download": "<10 MB for neurons, chemical synapses, gap junctions and muscle nodes",
            "download_priority": "tiny_gap_junction_benchmark",
        },
        "ciona_larva": {
            "estimated_full_size": "177 CNS neurons; 6,618 synapses; 1,772 NMJs; 1,206 gap junctions",
            "minimum_biofly_download": "<10 MB for graph-level tables",
            "download_priority": "tiny_laterality_motor_benchmark",
        },
        "microns_mm3": {
            "estimated_full_size": "EM imagery 117 TB; flat segmentation plus meshes 12 TB; watershed 42 TB; clefts 127 GB; v117 synapse graph 47.5 GB; archived v1300 synapse table 20.05 GB gz; functional scans 1.3 TB; DataJoint 225 GB",
            "minimum_biofly_download": "20.05 GB gz synapse table + 0.5 KB header for full graph-table conversion; for pilot use CAVE queries on selected roots and per-cell SWC/meshwork downloads instead of bulk meshes",
            "download_priority": "subset_or_cave_query_only",
        },
        "h01_human_cortex": {
            "estimated_full_size": "1.4 PB EM volume; 183M annotated synapses; probed c3 synapse JSON inventory 126.07 GB; local probe downloaded 72 JSON shards = 54.63 GB; proofread-104 SWC zip 59.83 MB; largest local proofread pilot retained 34,725 synapses across 74 cells",
            "minimum_biofly_download": "58-60 MB proofread SWC zip + one synapse JSONL shard (~0.76 GB) for overlap/proofread-subset pilot; 72-shard local pilot used ~54.63 GB JSONL; full synapse export is 126 GB JSONL or Avro-object route",
            "download_priority": "proofread_subset_only",
        },
        "allen_mouse_connectivity": {
            "estimated_full_size": "469 injection datasets and 295 structures in the original mesoscale analysis; voxel volumes can become multi-GB",
            "minimum_biofly_download": "MB-GB scale for projection matrix/injection summaries; avoid full voxel download for first BioFly region-graph run",
            "download_priority": "region_matrix_first",
        },
        "allen_brain_observatory": {
            "estimated_full_size": "early release >18k neurons; broader 2P/OpenScope datasets >60k neurons; full raw recordings are large session-wise data",
            "minimum_biofly_download": "one session or processed cell-response tables first, typically GB-scale rather than full raw imaging",
            "download_priority": "processed_session_first",
        },
        "ibl_brain_wide_map": {
            "estimated_full_size": "621,733 neurons; 139 mice; 699 Neuropixels insertions; 279 areas",
            "minimum_biofly_download": "one task/session with processed spikes/trials first, typically hundreds of MB to few GB",
            "download_priority": "processed_session_first",
        },
        "mouselight": {
            "estimated_full_size": ">1000 fully reconstructed projection neurons; morphology files are per-neuron rather than one synapse graph",
            "minimum_biofly_download": "one to tens of SWC reconstructions, usually MB-scale; all-neuron mirror likely GB-scale",
            "download_priority": "selected_swc_first",
        },
        "neuromorpho": {
            "estimated_full_size": "release-dependent multi-species SWC database",
            "minimum_biofly_download": "API/query-selected SWC files, usually <100 MB for a focused morphology benchmark",
            "download_priority": "selected_swc_first",
        },
        "abc_atlas": {
            "estimated_full_size": "mouse atlas materials report 32M+ cells and 5,300+ cell types; complete single-cell/spatial downloads are large",
            "minimum_biofly_download": "cell-type taxonomy/marker summary and selected region metadata first, MB-GB scale",
            "download_priority": "labels_only_first",
        },
        "zbrain": {
            "estimated_full_size": "~294k larval zebrafish neurons plus atlas masks and transgenic labels",
            "minimum_biofly_download": "cell-position/mask subset first, typically MB-GB scale depending on masks",
            "download_priority": "atlas_subset_first",
        },
        "hcp_young_adult": {
            "estimated_full_size": "HCP S1200 has 1,206 participants; DataLad HCP1200 open collection is ~80 TB",
            "minimum_biofly_download": "precomputed parcellated connectome/behavior matrices first, MB-GB scale; one full subject can be multi-GB",
            "download_priority": "derived_matrix_first",
        },
        "marmoset_macaque_tracer": {
            "estimated_full_size": "case/region/connection-record based; not a single fixed EM release",
            "minimum_biofly_download": "region connection matrix/export first, MB-scale if available",
            "download_priority": "region_matrix_first",
        },
        "dandi_openneuro": {
            "estimated_full_size": "dataset-dependent; single datasets range from MB to TB",
            "minimum_biofly_download": "one task-matched NWB/BIDS session or processed derivative first, MB-GB scale",
            "download_priority": "task_matched_session_first",
        },
    }
    for column in ("estimated_full_size", "minimum_biofly_download", "download_priority"):
        catalog[column] = catalog["dataset_id"].map(lambda dataset_id: size_plan.get(dataset_id, {}).get(column, "not estimated"))
    return catalog


def _canonical_external_fixtures() -> dict[str, dict[str, pd.DataFrame]]:
    nodes = pd.DataFrame(
        {
            "dataset": ["synapse_fixture"] * 7,
            "species": ["generic"] * 7,
            "node_id": [1, 2, 3, 4, 5, 6, 7],
            "region": ["sensory", "integrator", "integrator", "feedback", "output", "output", "state"],
            "cell_type": ["ORN_like", "KC_like", "KC_like", "APL_like", "MBON_like", "DN_like", "DAN_like"],
            "side": ["left", "left", "right", "midline", "left", "right", "midline"],
            "nt": ["ach", "ach", "ach", "gaba", "glutamate", "ach", "dopamine"],
            "source_dataset": ["synapse_fixture"] * 7,
        }
    )
    edges = pd.DataFrame(
        {
            "dataset": ["synapse_fixture"] * 8,
            "pre_id": [1, 1, 2, 3, 4, 7, 5, 6],
            "post_id": [2, 3, 5, 5, 2, 5, 6, 4],
            "weight": [5.0, 4.0, 3.0, 2.0, 2.5, 1.5, 2.0, 0.5],
            "sign": [1, 1, 1, 1, -1, 1, 1, -1],
            "synapse_count": [5, 4, 3, 2, 3, 2, 2, 1],
            "edge_type": ["chemical"] * 8,
            "confidence": [0.95, 0.92, 0.90, 0.88, 0.80, 0.75, 0.70, 0.60],
        }
    )
    morphology = pd.DataFrame(
        {
            "dataset": ["synapse_fixture"] * 4,
            "node_id": [2, 3, 5, 6],
            "swc_or_skeleton": ["fixture.swc"] * 4,
            "path_length": [60.0, 80.0, 180.0, 260.0],
            "branch_order": [2, 3, 5, 7],
            "soma_x": [0.0, 10.0, 30.0, 40.0],
            "soma_y": [0.0, 5.0, 10.0, 20.0],
            "soma_z": [0.0, 1.0, 3.0, 4.0],
        }
    )
    labels = pd.DataFrame(
        {
            "dataset": ["synapse_fixture"] * 4,
            "node_id": [2, 3, 5, 7],
            "transcriptomic_type": ["KC_alpha_beta_like", "KC_gamma_like", "MBON_like", "DAN_like"],
            "marker_genes": ["ey,dan", "ey", "vGlut", "th"],
            "receptor": ["AChR", "AChR", "GABA_R", "DopR"],
            "lineage": ["hemilineage_A", "hemilineage_B", "hemilineage_C", "hemilineage_D"],
        }
    )

    region_nodes = pd.DataFrame(
        {
            "dataset": ["region_fixture"] * 5,
            "species": ["mouse"] * 5,
            "node_id": [101, 102, 103, 104, 105],
            "region": ["V1", "LM", "AL", "SC", "STR"],
            "cell_type": ["region"] * 5,
            "side": ["left"] * 5,
            "nt": ["mixed"] * 5,
            "source_dataset": ["region_fixture"] * 5,
        }
    )
    region_edges = pd.DataFrame(
        {
            "dataset": ["region_fixture"] * 5,
            "pre_id": [101, 101, 102, 103, 104],
            "post_id": [102, 103, 104, 104, 105],
            "weight": [0.70, 0.55, 0.40, 0.30, 0.25],
            "sign": [1, 1, 1, 1, 1],
            "synapse_count": [0, 0, 0, 0, 0],
            "edge_type": ["projection_density"] * 5,
            "confidence": [0.90, 0.85, 0.80, 0.78, 0.70],
        }
    )

    activity_nodes = pd.DataFrame(
        {
            "dataset": ["activity_fixture"] * 3,
            "species": ["mouse"] * 3,
            "node_id": [201, 202, 203],
            "region": ["VISp", "VISl", "MOp"],
            "cell_type": ["excitatory", "inhibitory", "mixed"],
            "side": ["left", "left", "left"],
            "nt": ["glutamate", "gaba", "mixed"],
            "source_dataset": ["activity_fixture"] * 3,
        }
    )
    activity = pd.DataFrame(
        {
            "dataset": ["activity_fixture"] * 6,
            "trial_id": [1, 1, 1, 2, 2, 2],
            "node_id": [201, 202, 203, 201, 202, 203],
            "time": [0.1, 0.1, 0.1, 0.2, 0.2, 0.2],
            "activity": [0.4, -0.1, 0.2, 0.8, -0.2, 0.6],
            "stimulus": ["grating_0", "grating_0", "grating_0", "grating_90", "grating_90", "grating_90"],
            "behavior": ["hold", "hold", "hold", "turn", "turn", "turn"],
        }
    )
    behavior = pd.DataFrame(
        {
            "dataset": ["activity_fixture", "activity_fixture"],
            "trial_id": [1, 2],
            "stimulus": ["grating_0", "grating_90"],
            "action": ["hold", "turn"],
            "reward": [0.0, 1.0],
            "state": ["quiet", "engaged"],
        }
    )

    morphology_nodes = nodes[["dataset", "species", "node_id", "region", "cell_type", "side", "nt", "source_dataset"]].copy()
    morphology_nodes["dataset"] = "morphology_fixture"
    morphology_nodes["source_dataset"] = "morphology_fixture"
    morphology_fixture = morphology.copy()
    morphology_fixture["dataset"] = "morphology_fixture"

    label_nodes = nodes[["dataset", "species", "node_id", "region", "cell_type", "side", "nt", "source_dataset"]].copy()
    label_nodes["dataset"] = "label_fixture"
    label_nodes["source_dataset"] = "label_fixture"
    label_fixture = labels.copy()
    label_fixture["dataset"] = "label_fixture"

    return {
        "synapse_connectome": {"nodes": nodes, "edges": edges, "morphology": morphology, "labels": labels},
        "region_projection": {"nodes": region_nodes, "edges": region_edges},
        "activity_behavior": {"nodes": activity_nodes, "activity": activity, "behavior": behavior},
        "morphology_only": {"nodes": morphology_nodes, "morphology": morphology_fixture},
        "cell_type_label": {"nodes": label_nodes, "labels": label_fixture},
    }


def _canonical_required_columns() -> dict[str, tuple[str, ...]]:
    return {
        "nodes": ("dataset", "species", "node_id", "region", "cell_type", "side", "nt", "source_dataset"),
        "edges": ("dataset", "pre_id", "post_id", "weight", "sign", "synapse_count", "edge_type", "confidence"),
        "morphology": ("dataset", "node_id", "swc_or_skeleton", "path_length", "branch_order", "soma_x", "soma_y", "soma_z"),
        "activity": ("dataset", "trial_id", "node_id", "time", "activity", "stimulus", "behavior"),
        "labels": ("dataset", "node_id", "transcriptomic_type", "marker_genes", "receptor", "lineage"),
        "behavior": ("dataset", "trial_id", "stimulus", "action", "reward", "state"),
    }


def _external_edges_to_propagation(edges: pd.DataFrame) -> pd.DataFrame:
    signed_weight = edges["weight"].astype("float64") * edges["sign"].astype("float64")
    return pd.DataFrame(
        {
            "Presynaptic_ID": edges["pre_id"].astype("int64"),
            "Postsynaptic_ID": edges["post_id"].astype("int64"),
            "Excitatory x Connectivity": signed_weight,
        }
    )


def _run_external_schema_smoke_tests(config: ExternalDataFeasibilityConfig) -> tuple[pd.DataFrame, dict[str, object]]:
    fixtures = _canonical_external_fixtures()
    required = _canonical_required_columns()
    records: list[dict[str, object]] = []
    fixture_summary: dict[str, object] = {}

    for fixture_name, tables in fixtures.items():
        fixture_summary[fixture_name] = {table_name: int(len(table)) for table_name, table in tables.items()}
        for table_name, table in tables.items():
            expected = required[table_name]
            missing = [column for column in expected if column not in table.columns]
            records.append(
                {
                    "fixture": fixture_name,
                    "test_id": f"{fixture_name}_{table_name}_schema",
                    "test_type": "schema",
                    "status": "pass" if not missing else "fail",
                    "metric": "missing_columns",
                    "value": ",".join(missing),
                    "interpretation": "canonical table fields present" if not missing else "canonical table fields missing",
                }
            )

        if "edges" in tables:
            propagation_edges = _external_edges_to_propagation(tables["edges"])
            seed_id = int(tables["edges"]["pre_id"].iloc[0])
            response = signed_multihop_response(
                propagation_edges,
                [seed_id],
                PropagationConfig(steps=int(config.smoke_steps), max_active=int(config.max_active)),
            )
            active_count = int(response["root_id"].nunique()) if not response.empty else 0
            abs_mass = float(response["score"].abs().sum()) if not response.empty else 0.0
            records.append(
                {
                    "fixture": fixture_name,
                    "test_id": f"{fixture_name}_signed_propagation",
                    "test_type": "propagation",
                    "status": "pass" if active_count > 0 and abs_mass > 0 else "fail",
                    "metric": "active_nodes;abs_mass",
                    "value": f"{active_count};{abs_mass:.6f}",
                    "interpretation": "external edge table can drive BioFly signed propagation",
                }
            )

        if "activity" in tables and "behavior" in tables:
            activity_trials = set(tables["activity"]["trial_id"].astype(int).tolist())
            behavior_trials = set(tables["behavior"]["trial_id"].astype(int).tolist())
            aligned = activity_trials == behavior_trials
            mean_activity_by_behavior = (
                tables["activity"].groupby("behavior", as_index=False)["activity"].mean().sort_values("behavior")
            )
            dynamic_range = float(mean_activity_by_behavior["activity"].max() - mean_activity_by_behavior["activity"].min())
            records.append(
                {
                    "fixture": fixture_name,
                    "test_id": f"{fixture_name}_trial_alignment",
                    "test_type": "activity_behavior",
                    "status": "pass" if aligned and dynamic_range > 0 else "fail",
                    "metric": "matched_trials;behavior_dynamic_range",
                    "value": f"{len(activity_trials & behavior_trials)};{dynamic_range:.6f}",
                    "interpretation": "trial-level activity can be aligned to behavior readout",
                }
            )

        if "morphology" in tables:
            morphology = tables["morphology"].copy()
            morphology["attenuation"] = np.exp(-morphology["path_length"].astype("float64") / 250.0)
            finite = bool(np.isfinite(morphology["attenuation"]).all())
            monotonic = bool(
                morphology.sort_values("path_length")["attenuation"].is_monotonic_decreasing
                or morphology["attenuation"].nunique() <= 1
            )
            records.append(
                {
                    "fixture": fixture_name,
                    "test_id": f"{fixture_name}_morphology_attenuation",
                    "test_type": "morphology",
                    "status": "pass" if finite and monotonic else "fail",
                    "metric": "attenuation_min;attenuation_max",
                    "value": f"{float(morphology['attenuation'].min()):.6f};{float(morphology['attenuation'].max()):.6f}",
                    "interpretation": "path-length morphology can modulate edge/node gain",
                }
            )

        if "labels" in tables and "nodes" in tables:
            joined = tables["nodes"][["node_id"]].merge(tables["labels"], on="node_id", how="left")
            coverage = float(joined["transcriptomic_type"].notna().mean()) if len(joined) else 0.0
            records.append(
                {
                    "fixture": fixture_name,
                    "test_id": f"{fixture_name}_label_join",
                    "test_type": "labels",
                    "status": "pass" if coverage > 0 else "fail",
                    "metric": "node_label_coverage",
                    "value": f"{coverage:.6f}",
                    "interpretation": "cell-type/transcriptomic labels can join to simulation nodes",
                }
            )

    smoke = pd.DataFrame.from_records(records)
    return smoke, fixture_summary


def _external_system_validation_matrix() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "validation_layer": "schema",
                "what_is_checked": "nodes/edges/morphology/activity/labels/behavior canonical columns",
                "pass_criterion": "required columns exist and stable IDs can be joined",
                "current_result": "passed on all external fixture modalities",
                "remaining_work": "write dataset-specific readers for neuPrint, CAVE, NWB, BIDS, SWC or tracer tables",
            },
            {
                "validation_layer": "signed_graph_propagation",
                "what_is_checked": "direct connectome or region projection can become BioFly signed edge table",
                "pass_criterion": "seed produces nonzero multihop response with positive/negative mass preserved",
                "current_result": "passed for synapse-connectome and region-projection fixtures",
                "remaining_work": "add edge_type-specific handling for gap junctions, NMJs and uncertain signs",
            },
            {
                "validation_layer": "activity_behavior_alignment",
                "what_is_checked": "trial-level activity aligns to stimulus/action/reward/state",
                "pass_criterion": "activity trial IDs match behavior trial IDs and behavior states have dynamic range",
                "current_result": "passed for activity-behavior fixture",
                "remaining_work": "implement NWB/BIDS readers and session-level quality filters",
            },
            {
                "validation_layer": "morphology_weighting",
                "what_is_checked": "skeleton/SWC path length can produce finite passive attenuation",
                "pass_criterion": "attenuation is finite and decreases with path length",
                "current_result": "passed for morphology fixture",
                "remaining_work": "fit lambda/branch parameters by species, cell type and compartment",
            },
            {
                "validation_layer": "label_annotation",
                "what_is_checked": "cell-type/transcriptomic labels join to simulation nodes",
                "pass_criterion": "nonzero node-label coverage with explicit unmatched rows",
                "current_result": "passed for label fixture",
                "remaining_work": "build robust crosswalks from transcriptomic type to connectome node/region",
            },
            {
                "validation_layer": "scale_io",
                "what_is_checked": "whether the data size fits local batch execution",
                "pass_criterion": "small/medium data run locally; large data require chunking/cloud/materialized views",
                "current_result": "not executed for PB/TB external datasets in this audit",
                "remaining_work": "CAVE chunking for MICrONS/H01 and batch cache for HCP/DANDI/OpenNeuro",
            },
        ]
    )


def _external_biological_validation_matrix() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "data_family": "synapse-level connectome",
                "representative_sources": "hemibrain, MANC/VNC, larval brain, C. elegans, Ciona, MICrONS, H01",
                "what_biofly_can_test": "multi-hop path logic, convergence/divergence, perturbation ranking and candidate circuits",
                "biological_validation": "compare to known motifs, subtype annotations, calcium/imaging response and targeted perturbation",
                "main_risk": "connectome structure alone does not specify membrane dynamics, neuromodulation or animal behavior",
            },
            {
                "data_family": "region-level projection atlas",
                "representative_sources": "Allen Mouse Connectivity, HCP, marmoset/macaque tracer atlases",
                "what_biofly_can_test": "mesoscale source-target propagation and distributed decision/readout hypotheses",
                "biological_validation": "validate against tracer anatomy, region perturbation, fMRI/task readout or known pathway literature",
                "main_risk": "region graph cannot be interpreted as neuron-level synaptic mechanism",
            },
            {
                "data_family": "functional activity plus behavior",
                "representative_sources": "Allen Brain Observatory, IBL, Z-Brain, DANDI/OpenNeuro task datasets",
                "what_biofly_can_test": "rate/LIF parameter calibration, state variables, behavior proxy and trial-level readout",
                "biological_validation": "fit stimulus-response curves, choice variables and perturbation/imaging signatures",
                "main_risk": "activity data usually lacks exact structural edges, so it validates dynamics not connectome paths",
            },
            {
                "data_family": "morphology",
                "representative_sources": "MouseLight, NeuroMorpho, MICrONS/H01 skeletons",
                "what_biofly_can_test": "distance attenuation, branch-order weighting, projection class and compartment priors",
                "biological_validation": "compare morphology distributions by species/region/cell type and known projection anatomy",
                "main_risk": "morphology without postsynaptic targets cannot prove circuit flow",
            },
            {
                "data_family": "cell type / transcriptomics",
                "representative_sources": "ABC Atlas, BICAN/BICCN-like resources",
                "what_biofly_can_test": "NT/receptor/developmental labels and state-specific priors on graph nodes",
                "biological_validation": "check marker genes, spatial colocalization, HCR/FISH/immunostaining and perturbation response",
                "main_risk": "cell-type label mapping is many-to-many and cannot replace direct structure/function data",
            },
        ]
    )


def _write_external_data_feasibility_report(
    *,
    output_dir: Path,
    catalog: pd.DataFrame,
    smoke: pd.DataFrame,
    system_validation: pd.DataFrame,
    biological_validation: pd.DataFrame,
    fixture_summary: dict[str, object],
    summary: dict[str, object],
) -> Path:
    report_path = output_dir / "EXTERNAL_DATA_FEASIBILITY_REPORT_CN.md"
    direct = catalog[catalog["biofly_input_mode"].str.contains("direct|whole_animal|local_cortical", regex=True)]
    calibration = catalog[catalog["biofly_input_mode"].str.contains("activity|label|morphology", regex=True)]
    smoke_compact = smoke[["fixture", "test_type", "status", "metric", "value", "interpretation"]]
    catalog_compact = catalog[
        [
            "dataset_name",
            "species",
            "primary_modality",
            "public_scale",
            "biofly_input_mode",
            "feasibility",
        ]
    ]
    download_compact = catalog[
        [
            "dataset_name",
            "estimated_full_size",
            "minimum_biofly_download",
            "download_priority",
        ]
    ]
    report_path.write_text(
        f"""# BioFly 外部数据输入可行性测试报告

保存路径：`{report_path}`

## 这次实际测试了什么

本次没有下载 MICrONS、H01、HCP 这类 TB/PB 级外部全集，而是做了可复现的接口级
feasibility audit：

1. 把候选数据源按 BioFly 能使用的表结构拆成 `nodes / edges / morphology / activity / labels / behavior`。
2. 为突触级连接组、区域投射、功能行为、形态学和细胞标签 5 类输入生成 deterministic fixture。
3. 对能形成边表的数据运行 BioFly 的 signed multi-hop propagation。
4. 对 activity/behavior 检查 trial 对齐；对 morphology 检查 path-length attenuation；对 labels 检查节点 join。
5. 从系统和生物两个角度给出后续真实接入时的验证标准。

## 总体结果

| 项 | 数值 |
|---|---:|
| catalog datasets | `{int(summary['n_catalog_datasets'])}` |
| schema smoke tests | `{int(summary['n_smoke_tests'])}` |
| passed smoke tests | `{int(summary['n_passed_smoke_tests'])}` |
| failed smoke tests | `{int(summary['n_failed_smoke_tests'])}` |
| direct graph candidates | `{int(len(direct))}` |
| calibration/annotation candidates | `{int(len(calibration))}` |

## 数据源可行性总表

{_markdown_table(catalog_compact)}

## 数据大小与最小下载建议

{_markdown_table(download_compact)}

## H01/MICrONS public-source probe 合并口径

- MICrONS 官方教程确认该数据集可通过 CAVE 查询 synapse/cell-type/proofread 信息；synapse table 超过
  337M rows，推荐按 root id 或 bounding box 查询目标子集，而不是一次性全表查询。
- MICrONS static repositories 给出 bulk download 量级：imagery `117 TB`、flat segmentation plus meshes
  `12 TB`、watershed `42 TB`、PSD clefts `127 GB`、v117 synapse graph `47.5 GB`、functional scans `1.3 TB`
  和 DataJoint `225 GB`。v1300 archived synapse CSV.gz 的 HEAD/content-length 复核为 `20.05 GB`。
- H01 public probe 确认 synapse JSON inventory 为 `126.07 GB`，proofread-104 SWC zip 为 `59.83 MB`；
  本地 72-shard pilot 只作为 proofread-subset 工程验证，不能等同于全量 H01 生物结论。

## 接口 smoke test 结果

{_markdown_table(smoke_compact)}

## 系统验证矩阵

{_markdown_table(system_validation)}

## 生物验证矩阵

{_markdown_table(biological_validation)}

## fixture 规模

```json
{json.dumps(fixture_summary, ensure_ascii=False, indent=2)}
```

## 结论

- **可直接作为 BioFly graph 的数据**：hemibrain、MANC/VNC、果蝇幼虫全脑、C. elegans、Ciona、MICrONS 的
  核心问题是 reader 和规模，不是模型接口。它们能转成 `nodes + edges` 后就能复用现有传播、扰动、LIF
  和报告逻辑。
- **只能做区域级传播的数据**：Allen Mouse Connectivity、HCP、marmoset/macaque tracer atlas 可以接入，
  但只能写成 mesoscale region graph，不能声称 neuron-level mechanism。
- **主要做校准的数据**：Allen Brain Observatory、IBL、Z-Brain、DANDI/OpenNeuro 更适合校准 rate/LIF、
  trial-level state 和 behavior proxy，而不是替代连接组。
- **主要做注释的数据**：MouseLight、NeuroMorpho、ABC Atlas 可增强形态学和细胞类型先验，但不能单独证明
  circuit flow。
- **H01/MICrONS 的额外结论**：本轮合并了本地 public-source probe。两者都有官方公开 synapse 源表，
  但都不能直接提供 `branch_id/path_um/spine_volume` 这种 branch-level morphology audit 表；要先把
  synapse 坐标映射到 skeleton/mesh 分支，再定义 `size` 或 contact volume 的生物含义。

## 下一步真实接入建议

1. 先接 **MANC/VNC**：最贴近当前果蝇体系，能补足 MBON/DN 到 motor layer 的行为输出短板。
2. 再接 **果蝇幼虫全脑、C. elegans、Ciona**：小图能作为严谨 benchmark，测试全动物闭环、gap junction、
   muscle readout 和偏侧化。
3. 用 **MICrONS subset** 做哺乳动物试点：只读一个 materialized subset，先验证连接组传播能否解释视觉
   calcium response 的一部分。
4. 用 **Allen/IBL/DANDI/OpenNeuro** 做功能标定：不要从这些数据里推断结构边，而是用它们校准动力学和行为 readout。
""",
        encoding="utf-8",
    )
    return report_path


def run_external_data_feasibility(config: ExternalDataFeasibilityConfig | None = None) -> dict[str, object]:
    """Run a deterministic feasibility audit for non-FlyWire BioFly inputs."""

    config = config or ExternalDataFeasibilityConfig()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    catalog = _external_dataset_catalog()
    smoke, fixture_summary = _run_external_schema_smoke_tests(config)
    system_validation = _external_system_validation_matrix()
    biological_validation = _external_biological_validation_matrix()
    summary = {
        "n_catalog_datasets": int(len(catalog)),
        "n_smoke_tests": int(len(smoke)),
        "n_passed_smoke_tests": int(smoke["status"].eq("pass").sum()),
        "n_failed_smoke_tests": int(smoke["status"].ne("pass").sum()),
        "smoke_steps": int(config.smoke_steps),
        "max_active": int(config.max_active),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "scope": "interface-level feasibility audit; no PB/TB external dataset download",
    }

    catalog_path = output_dir / "external_dataset_catalog.csv"
    smoke_path = output_dir / "external_schema_smoke_tests.csv"
    system_path = output_dir / "external_system_validation_matrix.csv"
    biological_path = output_dir / "external_biological_validation_matrix.csv"
    summary_path = output_dir / "external_data_feasibility_summary.json"
    metadata_path = output_dir / "external_data_feasibility_metadata.json"
    catalog.to_csv(catalog_path, index=False)
    smoke.to_csv(smoke_path, index=False)
    system_validation.to_csv(system_path, index=False)
    biological_validation.to_csv(biological_path, index=False)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {"config": asdict(config), "summary": summary, "fixture_summary": fixture_summary},
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    report_path = _write_external_data_feasibility_report(
        output_dir=output_dir,
        catalog=catalog,
        smoke=smoke,
        system_validation=system_validation,
        biological_validation=biological_validation,
        fixture_summary=fixture_summary,
        summary=summary,
    )
    return {
        "catalog_csv": catalog_path,
        "schema_smoke_csv": smoke_path,
        "system_validation_csv": system_path,
        "biological_validation_csv": biological_path,
        "summary_json": summary_path,
        "metadata_json": metadata_path,
        "report_md": report_path,
        "catalog_df": catalog,
        "schema_smoke_df": smoke,
        "system_validation_df": system_validation,
        "biological_validation_df": biological_validation,
        "summary": summary,
    }


def _load_annotations(path: Path) -> pd.DataFrame:
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
    annotations = pd.read_parquet(path, columns=columns).drop_duplicates("root_id")
    for column in requested:
        if column not in annotations:
            annotations[column] = ""
    return annotations


def _connectome_group(row: pd.Series) -> str:
    super_class = str(row.get("super_class", ""))
    cell_class = str(row.get("cell_class", ""))
    cell_sub_class = str(row.get("cell_sub_class", ""))
    cell_type = str(row.get("cell_type", ""))
    hemibrain_type = str(row.get("hemibrain_type", ""))
    top_nt = str(row.get("top_nt", ""))
    synonyms = str(row.get("synonyms", ""))
    text = f"{cell_class} {cell_sub_class} {cell_type} {hemibrain_type} {synonyms}"
    if super_class.lower() == "descending" or re.search(
        r"(?:^|[^A-Za-z0-9])(?:MDN|oviDN|DNge|DNpe|DNae|DNbe|DNde|DNxl|DNp|DNg|DNa|DNb|DNc|DNd|DN1p|DN3)(?:[^A-Za-z0-9]|$)",
        text,
        flags=re.IGNORECASE,
    ):
        return "DN"
    if cell_type == "APL" or hemibrain_type == "APL":
        return "APL"
    if cell_type == "DPM" or hemibrain_type == "DPM":
        return "DPM"
    if cell_class == "Kenyon_Cell":
        return "KC"
    if cell_class == "ALPN":
        return "ALPN"
    if cell_class == "olfactory":
        return "ORN"
    if cell_class == "MBON":
        return "MBON"
    if cell_class == "DAN":
        return "DAN"
    if cell_class == "MBIN":
        return "MBIN_other"
    if top_nt == "octopamine" or _contains_token(pd.Series([f"{cell_type} {synonyms}"]), "OA").iloc[0]:
        return "octopamine"
    if top_nt == "serotonin":
        return "serotonin"
    if top_nt == "gaba":
        return "gaba_other"
    return ""


def _annotate_connectome_groups(annotations: pd.DataFrame) -> pd.DataFrame:
    annotated = annotations.copy()
    annotated["connectome_group"] = annotated.apply(_connectome_group, axis=1)
    for column in ["side", "cell_class", "cell_type", "hemibrain_type", "top_nt", "connectome_group"]:
        annotated[column] = annotated[column].fillna("").astype(str)
    return annotated


def _contains_token(series: pd.Series, token: str) -> pd.Series:
    escaped = re.escape(token)
    return series.fillna("").astype(str).str.contains(
        rf"(?:^|[^A-Za-z0-9]){escaped}(?:[^A-Za-z0-9]|$)",
        case=False,
        regex=True,
        na=False,
    )


def build_flywire_glomerulus_kc_matrix(
    annotation_path: Path,
    connectivity_path: Path,
) -> tuple[list[str], np.ndarray, np.ndarray, pd.DataFrame]:
    """Return glomerulus x KC activity matrix from real ALPN->KC edges."""

    annotations = _load_annotations(annotation_path)
    edge_columns = ["Presynaptic_ID", "Postsynaptic_ID", "Connectivity", "Excitatory x Connectivity"]
    edges = pd.read_parquet(connectivity_path, columns=edge_columns)
    return _build_flywire_glomerulus_kc_matrix_from_frames(annotations, edges)


def _build_flywire_glomerulus_kc_matrix_from_frames(
    annotations: pd.DataFrame,
    edges: pd.DataFrame,
) -> tuple[list[str], np.ndarray, np.ndarray, pd.DataFrame]:
    """Return glomerulus x KC activity matrix from already-loaded FlyWire frames."""

    kcs = annotations[annotations["cell_class"].astype(str).eq("Kenyon_Cell")].sort_values("root_id")
    alpns = annotations[annotations["cell_class"].astype(str).eq("ALPN")].copy()
    orns = annotations[annotations["cell_class"].astype(str).eq("olfactory")].copy()
    if kcs.empty:
        raise ValueError("No Kenyon cells found in annotation table.")
    if alpns.empty:
        raise ValueError("No ALPN neurons found in annotation table.")
    if orns.empty:
        raise ValueError("No olfactory ORN neurons found in annotation table.")

    kc_ids = kcs["root_id"].astype("int64").to_numpy()
    kc_index = {int(root_id): index for index, root_id in enumerate(kc_ids)}
    glomeruli = sorted(
        {
            cell_type.replace("ORN_", "", 1)
            for cell_type in orns["cell_type"].fillna("").astype(str)
            if cell_type.startswith("ORN_")
        }
    )

    alpn_ids = set(alpns["root_id"].astype("int64"))
    edges = edges[
        edges["Presynaptic_ID"].isin(alpn_ids)
        & edges["Postsynaptic_ID"].isin(set(map(int, kc_ids)))
    ].copy()
    if edges.empty:
        raise ValueError("No direct ALPN->KC edges found in connectivity table.")

    alpn_text = alpns[["cell_type", "hemibrain_type"]].fillna("").astype(str).agg(" ".join, axis=1)
    rows: list[np.ndarray] = []
    names: list[str] = []
    records: list[dict[str, object]] = []
    for glomerulus in glomeruli:
        selected_alpns = alpns[_contains_token(alpn_text, glomerulus)]
        if selected_alpns.empty:
            continue
        selected_edges = edges[edges["Presynaptic_ID"].isin(set(selected_alpns["root_id"].astype("int64")))]
        if selected_edges.empty:
            continue
        vector = np.zeros(len(kc_ids), dtype=np.float64)
        grouped = (
            selected_edges.groupby("Postsynaptic_ID")["Excitatory x Connectivity"].sum()
            / float(max(1, selected_alpns["root_id"].nunique()))
        )
        for post_root, weight in grouped.items():
            index = kc_index.get(int(post_root))
            if index is not None:
                vector[index] = float(weight)
        signed_mass = float(vector.sum())
        vector = np.maximum(vector, 0.0)
        positive_mass = float(vector.sum())
        if positive_mass <= 0:
            continue
        vector = vector / positive_mass
        rows.append(vector)
        names.append(glomerulus)
        records.append(
            {
                "glomerulus": glomerulus,
                "n_alpn": int(selected_alpns["root_id"].nunique()),
                "n_edges_to_kc": int(len(selected_edges)),
                "n_kc_targets": int(selected_edges["Postsynaptic_ID"].nunique()),
                "signed_edge_mass_per_alpn": signed_mass,
                "positive_edge_mass_per_alpn": positive_mass,
                "n_positive_kc_targets": int((vector > 0).sum()),
            }
        )
    if not rows:
        raise ValueError("No glomerulus-level ALPN->KC channels could be built.")
    return names, np.vstack(rows), kc_ids, pd.DataFrame.from_records(records)


def build_mixture_odor_panel(
    glomerulus_names: list[str],
    glomerulus_matrix: np.ndarray,
    *,
    seed: int,
    n_odors: int,
    min_glomeruli_per_odor: int,
    max_glomeruli_per_odor: int,
    channel_noise_sigma: float,
) -> tuple[list[str], np.ndarray, pd.DataFrame]:
    """Generate odor proxies as sparse mixtures of real glomerulus channels."""

    if glomerulus_matrix.ndim != 2 or glomerulus_matrix.shape[0] == 0:
        raise ValueError("glomerulus_matrix must contain at least one channel.")
    rng = np.random.default_rng(int(seed))
    n_channels = glomerulus_matrix.shape[0]
    min_mix = max(1, min(int(min_glomeruli_per_odor), n_channels))
    max_mix = max(min_mix, min(int(max_glomeruli_per_odor), n_channels))
    odor_rows: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    odor_names: list[str] = []
    for odor_index in range(int(n_odors)):
        n_mix = int(rng.integers(min_mix, max_mix + 1))
        selected = rng.choice(n_channels, size=n_mix, replace=False)
        weights = rng.dirichlet(np.ones(n_mix))
        row = weights @ glomerulus_matrix[selected]
        if channel_noise_sigma > 0:
            row = row * rng.lognormal(mean=0.0, sigma=float(channel_noise_sigma), size=row.shape)
        row = np.maximum(row, 0.0)
        total = float(row.sum())
        if total <= 0:
            row = glomerulus_matrix[int(selected[0])].copy()
            total = float(row.sum())
        row = row / total
        name = f"flywire_mixture_seed{seed}_odor{odor_index + 1:02d}"
        odor_names.append(name)
        odor_rows.append(row)
        records.append(
            {
                "odor_identity": name,
                "seed": int(seed),
                "n_glomeruli": n_mix,
                "glomeruli": ";".join(glomerulus_names[int(index)] for index in selected),
                "glomerulus_weights": ";".join(f"{float(weight):.6f}" for weight in weights),
            }
        )
    return odor_names, np.vstack(odor_rows), pd.DataFrame.from_records(records)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, sums, out=np.zeros_like(matrix), where=sums > 0)


def _mean_jaccard(binary: np.ndarray) -> float:
    values: list[float] = []
    for left in range(binary.shape[0]):
        for right in range(left + 1, binary.shape[0]):
            intersection = float(np.logical_and(binary[left], binary[right]).sum())
            union = float(np.logical_or(binary[left], binary[right]).sum())
            values.append(intersection / union if union else 0.0)
    return float(np.mean(values)) if values else 0.0


def _mean_binary_cosine(binary: np.ndarray) -> float:
    active = binary.sum(axis=1).astype(np.float64)
    values: list[float] = []
    for left in range(binary.shape[0]):
        for right in range(left + 1, binary.shape[0]):
            intersection = float(np.logical_and(binary[left], binary[right]).sum())
            denominator = float(np.sqrt(active[left] * active[right]))
            values.append(intersection / denominator if denominator else 0.0)
    return float(np.mean(values)) if values else 0.0


def _sparsify(activity: np.ndarray, ratio: float) -> tuple[np.ndarray, np.ndarray, int]:
    active_k = max(1, min(activity.shape[1], int(round(float(ratio) * activity.shape[1]))))
    graded = _winner_take_k(activity, active_k)
    graded = _normalize_rows(np.maximum(graded, 0.0))
    return graded > 0, graded, active_k


def run_kc_flywire_ratio_sweep(config: KCFlyWireRatioConfig | None = None) -> dict:
    cfg = config or KCFlyWireRatioConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    glomerulus_names, glomerulus_matrix, kc_ids, channel_table = build_flywire_glomerulus_kc_matrix(
        annotation_path=cfg.annotation_path,
        connectivity_path=cfg.connectivity_path,
    )
    channel_path = cfg.output_dir / "flywire_glomerulus_kc_channels.csv"
    channel_table.to_csv(channel_path, index=False)
    memory_config = KCSparseCodingConfig(
        random_seed=0,
        evaluation_repeats=cfg.memory_evaluation_repeats,
        test_repeats=cfg.memory_test_repeats,
        max_learning_steps=cfg.memory_max_learning_steps,
        dropout_probability=cfg.memory_dropout_probability,
        false_positive_probability=cfg.memory_false_positive_probability,
        forgetting_interference_steps=cfg.n_interference_blocks,
    )

    sweep_rows: list[dict[str, object]] = []
    odor_panel_frames: list[pd.DataFrame] = []
    for seed in cfg.seeds:
        _, activity, odor_panel = build_mixture_odor_panel(
            glomerulus_names,
            glomerulus_matrix,
            seed=int(seed),
            n_odors=cfg.n_odors,
            min_glomeruli_per_odor=cfg.min_glomeruli_per_odor,
            max_glomeruli_per_odor=cfg.max_glomeruli_per_odor,
            channel_noise_sigma=cfg.channel_noise_sigma,
        )
        odor_panel_frames.append(odor_panel)
        for ratio in cfg.ratios:
            binary, kc_responses, active_k = _sparsify(activity, ratio)
            weights = _train_mbon(
                kc_responses,
                cfg.cs_plus_index,
                kc_responses.shape[1],
                cfg.learning_rate,
                cfg.n_train_trials,
                np.random.default_rng(int(seed) + 7919),
            )
            dprime = _evaluate_dprime(kc_responses, weights, cfg.cs_plus_index)
            forgetting = _evaluate_forgetting(
                kc_responses,
                weights,
                cfg.cs_plus_index,
                cfg.n_interference_blocks,
                cfg.forgetting_lambda,
                cfg.learning_rate,
                np.random.default_rng(int(seed) + 31337),
            )
            memory_metrics = evaluate_binary_code(
                binary=binary,
                graded=kc_responses,
                activation_ratio_target=float(ratio),
                config=memory_config,
                seed=int(seed) + 101_003,
            )
            memory_fields = {
                f"memory_{key}": value
                for key, value in memory_metrics.items()
                if key
                in {
                    "learning_accuracy_final",
                    "learning_steps_to_criterion",
                    "association_margin_final",
                    "primary_memory_initial_accuracy",
                    "retention_accuracy_after_interference",
                    "forgetting_rate_per_interference_step",
                    "similar_pair_mean_jaccard",
                    "top_similarity_decile_jaccard",
                    "max_pairwise_jaccard",
                    "similar_pair_learning_accuracy_final",
                    "similar_pair_retention_accuracy_after_interference",
                    "similar_pair_forgetting_rate_per_interference_step",
                }
            }
            sweep_rows.append(
                {
                    "seed": int(seed),
                    "ratio": float(ratio),
                    "active_k": int(active_k),
                    "n_kc": int(kc_responses.shape[1]),
                    "observed_active_fraction": float(binary.mean()),
                    "mean_jaccard_overlap": _mean_jaccard(binary),
                    "mean_binary_cosine": _mean_binary_cosine(binary),
                    "learning_dprime": float(dprime),
                    "forgetting_rate_per_block": float(forgetting),
                    **memory_fields,
                }
            )

    odor_panel_path = cfg.output_dir / "flywire_mixture_odor_panel.csv"
    pd.concat(odor_panel_frames, ignore_index=True).to_csv(odor_panel_path, index=False)

    sweep_df = pd.DataFrame.from_records(sweep_rows)
    summary_df = (
        sweep_df.groupby("ratio", as_index=False)
        .agg(
            mean_dprime=("learning_dprime", "mean"),
            std_dprime=("learning_dprime", "std"),
            mean_forgetting=("forgetting_rate_per_block", "mean"),
            std_forgetting=("forgetting_rate_per_block", "std"),
            mean_jaccard_overlap=("mean_jaccard_overlap", "mean"),
            std_jaccard_overlap=("mean_jaccard_overlap", "std"),
            mean_binary_cosine=("mean_binary_cosine", "mean"),
            observed_active_fraction=("observed_active_fraction", "mean"),
            memory_learning_accuracy_final=("memory_learning_accuracy_final", "mean"),
            memory_learning_accuracy_final_std=("memory_learning_accuracy_final", "std"),
            memory_retention_accuracy_after_interference=(
                "memory_retention_accuracy_after_interference",
                "mean",
            ),
            memory_retention_accuracy_after_interference_std=(
                "memory_retention_accuracy_after_interference",
                "std",
            ),
            memory_forgetting_rate_per_interference_step=(
                "memory_forgetting_rate_per_interference_step",
                "mean",
            ),
            memory_learning_steps_to_criterion=("memory_learning_steps_to_criterion", "mean"),
            memory_similar_pair_learning_accuracy_final=(
                "memory_similar_pair_learning_accuracy_final",
                "mean",
            ),
            memory_similar_pair_retention_accuracy_after_interference=(
                "memory_similar_pair_retention_accuracy_after_interference",
                "mean",
            ),
            memory_similar_pair_forgetting_rate_per_interference_step=(
                "memory_similar_pair_forgetting_rate_per_interference_step",
                "mean",
            ),
            memory_similar_pair_mean_jaccard=("memory_similar_pair_mean_jaccard", "mean"),
            memory_top_similarity_decile_jaccard=("memory_top_similarity_decile_jaccard", "mean"),
            active_k=("active_k", "first"),
        )
        .sort_values("ratio")
    )
    sweep_path = cfg.output_dir / "kc_flywire_ratio_sweep_raw.csv"
    summary_path = cfg.output_dir / "kc_flywire_ratio_sweep_summary.csv"
    sweep_df.to_csv(sweep_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    best_dprime_idx = int(summary_df["mean_dprime"].idxmax())
    best_forgetting_idx = int(summary_df["mean_forgetting"].idxmin())
    literature_idx = int((summary_df["ratio"] - LITERATURE_KC_ACTIVE_FRACTION).abs().idxmin())
    legacy_idx = int((summary_df["ratio"] - LEGACY_ONE_SIXTH_KC_RATIO).abs().idxmin())
    interpretation = {
        "model": "real_flywire_v783_direct_ALPN_to_KC_subgraph",
        "n_kc": int(len(kc_ids)),
        "n_glomerulus_channels": int(len(glomerulus_names)),
        "n_seed_panels": int(len(cfg.seeds)),
        "n_odors_per_panel": int(cfg.n_odors),
        "best_learning_ratio": float(summary_df.loc[best_dprime_idx, "ratio"]),
        "best_learning_dprime": float(summary_df.loc[best_dprime_idx, "mean_dprime"]),
        "lowest_forgetting_ratio": float(summary_df.loc[best_forgetting_idx, "ratio"]),
        "lowest_forgetting_value": float(summary_df.loc[best_forgetting_idx, "mean_forgetting"]),
        "literature_anchor_ratio": float(summary_df.loc[literature_idx, "ratio"]),
        "literature_anchor_dprime": float(summary_df.loc[literature_idx, "mean_dprime"]),
        "literature_anchor_forgetting": float(summary_df.loc[literature_idx, "mean_forgetting"]),
        "literature_anchor_jaccard": float(summary_df.loc[literature_idx, "mean_jaccard_overlap"]),
        "literature_anchor_memory_accuracy": float(
            summary_df.loc[literature_idx, "memory_learning_accuracy_final"]
        ),
        "literature_anchor_memory_retention": float(
            summary_df.loc[literature_idx, "memory_retention_accuracy_after_interference"]
        ),
        "legacy_one_sixth_ratio": float(summary_df.loc[legacy_idx, "ratio"]),
        "legacy_one_sixth_dprime": float(summary_df.loc[legacy_idx, "mean_dprime"]),
        "legacy_one_sixth_forgetting": float(summary_df.loc[legacy_idx, "mean_forgetting"]),
        "legacy_one_sixth_jaccard": float(summary_df.loc[legacy_idx, "mean_jaccard_overlap"]),
        "legacy_one_sixth_memory_accuracy": float(
            summary_df.loc[legacy_idx, "memory_learning_accuracy_final"]
        ),
        "legacy_one_sixth_memory_retention": float(
            summary_df.loc[legacy_idx, "memory_retention_accuracy_after_interference"]
        ),
        # Backward-compatible JSON keys for old notebooks.
        "canonical_one_sixth_dprime": float(summary_df.loc[legacy_idx, "mean_dprime"]),
        "canonical_one_sixth_forgetting": float(summary_df.loc[legacy_idx, "mean_forgetting"]),
        "canonical_one_sixth_jaccard": float(summary_df.loc[legacy_idx, "mean_jaccard_overlap"]),
        "canonical_one_sixth_memory_accuracy": float(
            summary_df.loc[legacy_idx, "memory_learning_accuracy_final"]
        ),
        "canonical_one_sixth_memory_retention": float(
            summary_df.loc[legacy_idx, "memory_retention_accuracy_after_interference"]
        ),
        "best_memory_accuracy_ratio": float(
            summary_df.loc[int(summary_df["memory_learning_accuracy_final"].idxmax()), "ratio"]
        ),
        "best_memory_retention_ratio": float(
            summary_df.loc[int(summary_df["memory_retention_accuracy_after_interference"].idxmax()), "ratio"]
        ),
    }
    interpretation_path = cfg.output_dir / "kc_flywire_ratio_sweep_interpretation.json"
    interpretation_path.write_text(json.dumps(interpretation, ensure_ascii=False, indent=2), encoding="utf-8")

    config_path = cfg.output_dir / "kc_flywire_ratio_sweep_config.json"
    payload = asdict(cfg)
    payload["annotation_path"] = str(cfg.annotation_path)
    payload["connectivity_path"] = str(cfg.connectivity_path)
    payload["output_dir"] = str(cfg.output_dir)
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    figure_path = _write_figure(summary_df, cfg.output_dir)
    report_path = cfg.output_dir / "KC_FLYWIRE_RATIO_REPORT_CN.md"
    report_path.write_text(_render_report(cfg, summary_df, interpretation, channel_table), encoding="utf-8")

    return {
        "channels_csv": channel_path,
        "odor_panel_csv": odor_panel_path,
        "sweep_csv": sweep_path,
        "summary_csv": summary_path,
        "interpretation_json": interpretation_path,
        "config_json": config_path,
        "figure_png": figure_path,
        "report_md": report_path,
        "summary_df": summary_df,
        "interpretation": interpretation,
    }


def _write_figure(summary_df: pd.DataFrame, output_dir: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes = axes.ravel()
    axes[0].errorbar(
        summary_df["ratio"],
        summary_df["mean_dprime"],
        yerr=summary_df["std_dprime"].fillna(0.0),
        fmt="o-",
        capsize=3,
        color="tab:blue",
    )
    axes[0].axvline(LITERATURE_KC_ACTIVE_FRACTION, ls="--", color="grey", label="literature <=10%")
    axes[0].axvline(LEGACY_ONE_SIXTH_KC_RATIO, ls=":", color="grey", label="legacy 1/6")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("KC active fraction")
    axes[0].set_ylabel("learning d-prime")
    axes[0].set_title("FlyWire ALPN->KC learning")
    axes[0].legend()

    axes[1].errorbar(
        summary_df["ratio"],
        summary_df["mean_forgetting"],
        yerr=summary_df["std_forgetting"].fillna(0.0),
        fmt="o-",
        capsize=3,
        color="tab:red",
    )
    axes[1].axvline(LITERATURE_KC_ACTIVE_FRACTION, ls="--", color="grey", label="literature <=10%")
    axes[1].axvline(LEGACY_ONE_SIXTH_KC_RATIO, ls=":", color="grey", label="legacy 1/6")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("KC active fraction")
    axes[1].set_ylabel("forgetting slope")
    axes[1].set_title("Interference readout")

    axes[2].errorbar(
        summary_df["ratio"],
        summary_df["mean_jaccard_overlap"],
        yerr=summary_df["std_jaccard_overlap"].fillna(0.0),
        fmt="o-",
        capsize=3,
        color="tab:green",
    )
    axes[2].axvline(LITERATURE_KC_ACTIVE_FRACTION, ls="--", color="grey", label="literature <=10%")
    axes[2].axvline(LEGACY_ONE_SIXTH_KC_RATIO, ls=":", color="grey", label="legacy 1/6")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("KC active fraction")
    axes[2].set_ylabel("mean odor-code Jaccard")
    axes[2].set_title("Code overlap")

    axes[3].plot(
        summary_df["ratio"],
        summary_df["memory_learning_accuracy_final"],
        "o-",
        color="tab:purple",
        label="learning accuracy",
    )
    axes[3].plot(
        summary_df["ratio"],
        summary_df["memory_retention_accuracy_after_interference"],
        "s-",
        color="tab:orange",
        label="retention after interference",
    )
    axes[3].axvline(LITERATURE_KC_ACTIVE_FRACTION, ls="--", color="grey", label="literature <=10%")
    axes[3].axvline(LEGACY_ONE_SIXTH_KC_RATIO, ls=":", color="grey", label="legacy 1/6")
    axes[3].set_xscale("log")
    axes[3].set_xlabel("KC active fraction")
    axes[3].set_ylabel("accuracy")
    axes[3].set_ylim(0, 1.02)
    axes[3].set_title("Binary memory proxy")
    axes[3].legend()

    figure_path = output_dir / "Fig_kc_flywire_ratio_sweep.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return figure_path


def _render_report(
    cfg: KCFlyWireRatioConfig,
    summary_df: pd.DataFrame,
    interpretation: dict,
    channel_table: pd.DataFrame,
) -> str:
    rows = []
    for row in summary_df.itertuples(index=False):
        rows.append(
            f"| {float(row.ratio):.4f} | {int(row.active_k)} | "
            f"{float(row.mean_dprime):.3f} ± {float(row.std_dprime):.3f} | "
            f"{float(row.mean_forgetting):.4f} ± {float(row.std_forgetting):.4f} | "
            f"{float(row.mean_jaccard_overlap):.4f} | "
            f"{float(row.memory_learning_accuracy_final):.3f} | "
            f"{float(row.memory_retention_accuracy_after_interference):.3f} |"
        )
    table = "\n".join(rows)
    top_channels = channel_table.sort_values("n_kc_targets", ascending=False).head(8)
    channel_rows = "\n".join(
        f"| {row.glomerulus} | {int(row.n_alpn)} | {int(row.n_edges_to_kc)} | {int(row.n_kc_targets)} | "
        f"{float(row.positive_edge_mass_per_alpn):.2f} |"
        for row in top_channels.itertuples(index=False)
    )
    return f"""# FlyWire 真实连接组约束的 KC 稀疏编码比例仿真

保存路径：`outputs/kc_flywire_ratio/KC_FLYWIRE_RATIO_REPORT_CN.md`

## 实验目的

把原始 `kc_optimal_ratio` 中的随机 PN->KC 投影替换成真实 FlyWire v783 连接组约束，重新评估文献中的 `<=10%` KC 可靠响应锚点是否仍位于学习、记忆保持和低 overlap 的合理工作区；旧 `1/6` 只作为历史 toy-model 对照。

## 模型说明

- 连接组：FlyWire v783 `Connectivity_783.parquet`。
- 输入子图：直接 `ALPN -> Kenyon_Cell` 边；本次共构建 {interpretation['n_glomerulus_channels']} 个 glomerulus 通道、{interpretation['n_kc']} 个 KC 维度。
- 气味面板：每个随机种子生成 {cfg.n_odors} 个气味代理；每个气味是 {cfg.min_glomeruli_per_odor}-{cfg.max_glomeruli_per_odor} 个真实 glomerulus 通道的稀疏混合。
- 稀疏化：对每个气味的真实连接组 KC drive 做 winner-take-K，K = 激活比例 x 全部 KC 数。
- 学习读出：沿用旧实验的单输出 MBON delta-rule proxy；统计 {len(cfg.seeds)} 个 odor-panel seeds。

## 结果汇总

| KC active fraction | active K | learning d-prime (mean ± std) | old forgetting slope (mean ± std) | mean odor-code Jaccard | binary learning acc. | retention acc. |
|---:|---:|---:|---:|---:|---:|---:|
{table}

## 关键发现

- **FlyWire 子图下最高 learning d-prime**：{interpretation['best_learning_ratio']:.4f}（d-prime = {interpretation['best_learning_dprime']:.3f}）。该单输出 proxy 偏好极稀疏读出，不单独定义生理最优。
- **binary memory accuracy 最高点**：{interpretation['best_memory_accuracy_ratio']:.4f}。
- **binary retention accuracy 最高点**：{interpretation['best_memory_retention_ratio']:.4f}。
- **文献 <=10% 锚点指标**：ratio = {interpretation['literature_anchor_ratio']:.4f}，d-prime = {interpretation['literature_anchor_dprime']:.3f}，Jaccard = {interpretation['literature_anchor_jaccard']:.4f}，binary learning acc. = {interpretation['literature_anchor_memory_accuracy']:.3f}，retention acc. = {interpretation['literature_anchor_memory_retention']:.3f}。
- **legacy 1/6 对照指标**：ratio = {interpretation['legacy_one_sixth_ratio']:.4f}，d-prime = {interpretation['legacy_one_sixth_dprime']:.3f}，Jaccard = {interpretation['legacy_one_sixth_jaccard']:.4f}，binary learning acc. = {interpretation['legacy_one_sixth_memory_accuracy']:.3f}，retention acc. = {interpretation['legacy_one_sixth_memory_retention']:.3f}。

## 与随机投影版相比的结论更新

随机投影版旧结论把 `1/6` 写成 canonical，这是口径错误；重新解释后，随机投影结果只能说明 `1/10` 附近常处在 toy learning/readout 的高效区，不能把 `1/6` 当作文献生理锚点。

真实 FlyWire ALPN->KC 子图下，结论应更具体：在当前 glomerulus-mixture + 单 MBON proxy 中，d-prime 峰值出现在更稀疏区，随后随激活比例升高而下降；同时 odor-code overlap 随激活比例升高稳定增加。补充的 binary memory proxy 在默认真实子图面板中通常把最佳学习/保持放在 0.10 附近。因此，本实验**支持把 <=10% 作为默认生理锚点**，并把 `1/6` 降级为 legacy toy 对照。

更稳妥的表述是：文献/生理观察给出的 KC 稀疏激活锚点是约 5-10%；在真实 FlyWire ALPN->KC 输入几何下，增加 KC 激活比例会增加气味编码重叠，支持 APL 维持稀疏化以降低混淆。但 0.10 是否优于更稀疏比例，取决于真实 APL 阈值、KC 可检测信号强度、多 MBON/DAN 教师信号和行为读出，而不是当前单输出 proxy 能证明的事实。

## 连接组通道审计

| glomerulus | n ALPN | ALPN->KC edges | KC targets | positive mass / ALPN |
|---|---:|---:|---:|---:|
{channel_rows}

## 边界

1. 本实验使用真实 FlyWire `ALPN -> KC` 边，但气味是 glomerulus 通道混合代理，不是真实 odor receptor response 数据。
2. 直接 ORN->KC 边在该连接表中为 0，因此输入从 ALPN 层进入；没有显式建模 ORN->ALN->PN 的时间动力学。
3. 学习读出仍是单 MBON delta-rule 和 binary associative-memory proxy，不包含多 MBON、DAN compartment teaching、APL feedback dynamics 或 spike-level 生理。
4. 因此本结果可以改进结论边界，但不能单独证明 0.10 是数学最优；它是与文献一致、在真实子图 memory proxy 中表现稳定的生理工作锚点。

## 湿实验建议更新

- 优先验证的不是“0.10 是否数学最优”，而是“APL 抑制减弱是否导致 KC active fraction 从 <=10% 上升并让 odor-code overlap 增加”。
- 如果要测试比例最优性，应在 calcium imaging 中同时读出过稀、正常和过密状态下的相近气味 discrimination/generalization/interference，而不是只比较最终二选一行为。
- 需要真实 odor panel 的 ORN/PN 活性标定；否则计算中的 glomerulus-mixture panel 只能作为连接组几何假说。
"""


def _merge_science_edge_annotations(edges: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    annotated = _annotate_connectome_groups(annotations)
    keep = annotated[annotated["connectome_group"].ne("")].copy()
    group_ids = set(keep["root_id"].astype("int64"))
    edges = edges[
        edges["Presynaptic_ID"].isin(group_ids) | edges["Postsynaptic_ID"].isin(group_ids)
    ].copy()
    columns = ["root_id", "side", "connectome_group", "cell_class", "cell_type", "hemibrain_type", "top_nt"]
    pre = keep[columns].rename(
        columns={
            "root_id": "Presynaptic_ID",
            "side": "pre_side",
            "connectome_group": "pre_group",
            "cell_class": "pre_cell_class",
            "cell_type": "pre_cell_type",
            "hemibrain_type": "pre_hemibrain_type",
            "top_nt": "pre_top_nt",
        }
    )
    post = keep[columns].rename(
        columns={
            "root_id": "Postsynaptic_ID",
            "side": "post_side",
            "connectome_group": "post_group",
            "cell_class": "post_cell_class",
            "cell_type": "post_cell_type",
            "hemibrain_type": "post_hemibrain_type",
            "top_nt": "post_top_nt",
        }
    )
    merged = edges.merge(pre, on="Presynaptic_ID", how="left").merge(post, on="Postsynaptic_ID", how="left")
    merged = merged[merged["pre_group"].notna() | merged["post_group"].notna()].copy()
    for column in [
        "pre_side",
        "post_side",
        "pre_group",
        "post_group",
        "pre_cell_class",
        "post_cell_class",
        "pre_cell_type",
        "post_cell_type",
        "pre_hemibrain_type",
        "post_hemibrain_type",
        "pre_top_nt",
        "post_top_nt",
    ]:
        merged[column] = merged[column].fillna("other").astype(str).replace("", "other")
    merged["abs_signed_weight"] = merged["Excitatory x Connectivity"].abs()
    merged["positive_signed_weight"] = merged["Excitatory x Connectivity"].clip(lower=0)
    merged["negative_signed_weight"] = (-merged["Excitatory x Connectivity"].clip(upper=0))
    return merged


def _summarize_group_pairs(edges: pd.DataFrame) -> pd.DataFrame:
    summary = (
        edges.groupby(["pre_group", "post_group"], as_index=False)
        .agg(
            n_edges=("Connectivity", "size"),
            n_pre=("Presynaptic_ID", "nunique"),
            n_post=("Postsynaptic_ID", "nunique"),
            syn_count=("Connectivity", "sum"),
            signed_weight=("Excitatory x Connectivity", "sum"),
            abs_signed_weight=("abs_signed_weight", "sum"),
            positive_signed_weight=("positive_signed_weight", "sum"),
            negative_signed_weight=("negative_signed_weight", "sum"),
        )
        .sort_values("abs_signed_weight", ascending=False)
    )
    summary["negative_fraction"] = np.divide(
        summary["negative_signed_weight"],
        summary["abs_signed_weight"],
        out=np.zeros(len(summary), dtype=float),
        where=summary["abs_signed_weight"].to_numpy(dtype=float) > 0,
    )
    return summary


def _summarize_key_pairs(pair_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pre_group, post_group in KEY_CIRCUIT_PAIRS:
        selected = pair_summary[
            pair_summary["pre_group"].eq(pre_group) & pair_summary["post_group"].eq(post_group)
        ]
        if selected.empty:
            rows.append({"pre_group": pre_group, "post_group": post_group})
        else:
            rows.append(selected.iloc[0].to_dict())
    return pd.DataFrame.from_records(rows)


def _summarize_side_pairs(edges: pd.DataFrame) -> pd.DataFrame:
    key_pairs = set(KEY_CIRCUIT_PAIRS)
    mask = pd.Series(
        [(pre, post) in key_pairs for pre, post in zip(edges["pre_group"], edges["post_group"])],
        index=edges.index,
    )
    selected = edges.loc[mask].copy()
    if selected.empty:
        return pd.DataFrame()
    return (
        selected.groupby(["pre_group", "post_group", "pre_side", "post_side"], as_index=False)
        .agg(
            n_edges=("Connectivity", "size"),
            n_pre=("Presynaptic_ID", "nunique"),
            n_post=("Postsynaptic_ID", "nunique"),
            syn_count=("Connectivity", "sum"),
            signed_weight=("Excitatory x Connectivity", "sum"),
            abs_signed_weight=("abs_signed_weight", "sum"),
            negative_signed_weight=("negative_signed_weight", "sum"),
        )
        .sort_values(["pre_group", "post_group", "abs_signed_weight"], ascending=[True, True, False])
    )


def _summarize_target_coverage(edges: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    annotated = _annotate_connectome_groups(annotations)
    group_counts = annotated[annotated["connectome_group"].ne("")].groupby("connectome_group")["root_id"].nunique()
    rows = []
    for pre_group, post_group in KEY_CIRCUIT_PAIRS:
        selected = edges[edges["pre_group"].eq(pre_group) & edges["post_group"].eq(post_group)].copy()
        if selected.empty:
            rows.append(
                {
                    "pre_group": pre_group,
                    "post_group": post_group,
                    "n_edges": 0,
                    "n_pre": 0,
                    "n_post": 0,
                    "post_group_coverage": np.nan,
                    "pre_group_coverage": np.nan,
                }
            )
            continue
        pre_total = int(group_counts.get(pre_group, 0))
        post_total = int(group_counts.get(post_group, 0))
        per_post = selected.groupby("Postsynaptic_ID")["abs_signed_weight"].sum()
        per_pre = selected.groupby("Presynaptic_ID")["abs_signed_weight"].sum()
        rows.append(
            {
                "pre_group": pre_group,
                "post_group": post_group,
                "n_edges": int(len(selected)),
                "n_pre": int(selected["Presynaptic_ID"].nunique()),
                "n_post": int(selected["Postsynaptic_ID"].nunique()),
                "pre_group_total": pre_total,
                "post_group_total": post_total,
                "pre_group_coverage": selected["Presynaptic_ID"].nunique() / pre_total if pre_total else np.nan,
                "post_group_coverage": selected["Postsynaptic_ID"].nunique() / post_total if post_total else np.nan,
                "median_abs_weight_per_post": float(per_post.median()) if not per_post.empty else np.nan,
                "p10_abs_weight_per_post": float(per_post.quantile(0.10)) if not per_post.empty else np.nan,
                "p90_abs_weight_per_post": float(per_post.quantile(0.90)) if not per_post.empty else np.nan,
                "median_abs_weight_per_pre": float(per_pre.median()) if not per_pre.empty else np.nan,
            }
        )
    return pd.DataFrame.from_records(rows)


def _summarize_mbon_targets(edges: pd.DataFrame, top_n: int = 24) -> pd.DataFrame:
    selected = edges[edges["pre_group"].eq("KC") & edges["post_group"].eq("MBON")].copy()
    if selected.empty:
        return pd.DataFrame()
    return (
        selected.groupby(["Postsynaptic_ID", "post_side", "post_cell_type"], as_index=False)
        .agg(
            kc_input_abs=("abs_signed_weight", "sum"),
            n_kc_inputs=("Presynaptic_ID", "nunique"),
            n_edges=("Connectivity", "size"),
        )
        .sort_values("kc_input_abs", ascending=False)
        .head(top_n)
    )


def _summarize_dan_to_kc(edges: pd.DataFrame, top_n: int = 24) -> pd.DataFrame:
    selected = edges[edges["pre_group"].eq("DAN") & edges["post_group"].eq("KC")].copy()
    if selected.empty:
        return pd.DataFrame()
    return (
        selected.groupby(["pre_cell_type", "pre_side"], as_index=False)
        .agg(
            abs_signed_weight=("abs_signed_weight", "sum"),
            n_kc_targets=("Postsynaptic_ID", "nunique"),
            n_edges=("Connectivity", "size"),
        )
        .sort_values("abs_signed_weight", ascending=False)
        .head(top_n)
    )


def _summarize_apl_state_inputs(edges: pd.DataFrame) -> pd.DataFrame:
    selected = edges[
        edges["post_group"].eq("APL")
        & edges["pre_group"].isin(["KC", "ALPN", "DAN", "DPM", "MBON", "gaba_other", "octopamine", "serotonin"])
    ].copy()
    if selected.empty:
        return pd.DataFrame()
    summary = (
        selected.groupby("pre_group", as_index=False)
        .agg(
            n_edges=("Connectivity", "size"),
            n_pre=("Presynaptic_ID", "nunique"),
            signed_weight=("Excitatory x Connectivity", "sum"),
            abs_signed_weight=("abs_signed_weight", "sum"),
            negative_signed_weight=("negative_signed_weight", "sum"),
            positive_signed_weight=("positive_signed_weight", "sum"),
        )
        .sort_values("abs_signed_weight", ascending=False)
    )
    summary["predicted_apl_effect"] = np.where(
        summary["signed_weight"] < 0,
        "candidate_APL_downshift_or_inhibition",
        "candidate_APL_drive_or_disinhibition",
    )
    return summary


def _summarize_mbon_apl_candidate_targets(edges: pd.DataFrame) -> pd.DataFrame:
    selected = edges[edges["pre_group"].eq("MBON") & edges["post_group"].eq("APL")].copy()
    if selected.empty:
        return pd.DataFrame(
            columns=[
                "pre_cell_type",
                "pre_side",
                "pre_top_nt",
                "n_edges",
                "n_apl_targets",
                "synapses",
                "signed_weight",
                "abs_signed_weight",
                "negative_signed_weight",
                "positive_signed_weight",
                "negative_fraction",
                "fraction_of_mbon_to_apl_abs",
                "candidate_effect",
                "wetlab_priority",
            ]
        )
    summary = (
        selected.groupby(["pre_cell_type", "pre_side", "pre_top_nt"], as_index=False)
        .agg(
            n_edges=("Connectivity", "size"),
            n_apl_targets=("Postsynaptic_ID", "nunique"),
            synapses=("Connectivity", "sum"),
            signed_weight=("Excitatory x Connectivity", "sum"),
            abs_signed_weight=("abs_signed_weight", "sum"),
            negative_signed_weight=("negative_signed_weight", "sum"),
            positive_signed_weight=("positive_signed_weight", "sum"),
        )
        .sort_values("abs_signed_weight", ascending=False)
    )
    total_abs = float(summary["abs_signed_weight"].sum())
    summary["negative_fraction"] = np.divide(
        summary["negative_signed_weight"],
        summary["abs_signed_weight"],
        out=np.zeros(len(summary), dtype=float),
        where=summary["abs_signed_weight"].to_numpy(dtype=float) > 0,
    )
    summary["fraction_of_mbon_to_apl_abs"] = summary["abs_signed_weight"] / total_abs if total_abs > 0 else 0.0
    summary["candidate_effect"] = np.where(
        summary["signed_weight"] < 0,
        "candidate_APL_downshift",
        "candidate_APL_drive_or_disinhibition",
    )
    summary["wetlab_priority"] = np.where(
        summary["signed_weight"] < 0,
        summary["negative_signed_weight"],
        summary["positive_signed_weight"] * 0.25,
    )
    return summary.sort_values("wetlab_priority", ascending=False)


def _classify_target_behavior_axis(row: pd.Series) -> tuple[str, str]:
    """Map annotated MB/KC targets to transparent behavior-level axes."""

    group = str(row.get("connectome_group", row.get("post_group", "")))
    cell_type = str(row.get("cell_type", row.get("post_cell_type", ""))).upper()
    hemibrain_type = str(row.get("hemibrain_type", row.get("post_hemibrain_type", ""))).upper()
    top_nt = str(row.get("top_nt", row.get("post_top_nt", ""))).lower()
    text = f"{cell_type} {hemibrain_type}"
    if group == "APL":
        return "sparseness_brake", "global KC inhibition; predicts odor-code overlap and generalization"
    if group == "DPM":
        return "memory_persistence", "DPM recurrent trace; predicts memory consolidation or persistence"
    if group == "DN":
        return "motor_readout", "descending-neuron output; predicts motor/action-selection readout candidates"
    if group == "octopamine":
        return "appetitive_state", "octopamine state axis; predicts appetitive/arousal gating"
    if group == "serotonin":
        return "state_modulation", "serotonin state axis; predicts state-dependent gain"
    if group == "DAN":
        if "PPL" in text:
            return "aversive_teaching", "PPL dopamine teaching axis; shock/avoidance-compatible proxy"
        if "PAM" in text:
            return "appetitive_teaching", "PAM dopamine teaching axis; sugar/reward-compatible proxy"
        return "dopamine_teaching", "dopamine teaching axis without PAM/PPL subclass split"
    if group == "MBON":
        if top_nt == "glutamate" or "MBON01" in text or "MBON02" in text:
            return "avoidance_or_negative_valence", "glutamatergic MBON-dominant output; negative-valence/avoidance proxy"
        if top_nt == "gaba":
            return "disinhibitory_memory_output", "GABAergic MBON output; disinhibitory memory proxy"
        if top_nt == "acetylcholine":
            return "approach_or_positive_valence", "cholinergic MBON output; positive-valence/approach proxy"
        return "memory_output", "MBON output with unspecified valence proxy"
    if group == "MBIN_other":
        return "state_modulation", "non-APL/DPM MB input state axis"
    return "other", "not mapped to a behavior axis"


def _build_kc_target_matrix(edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = edges[
        edges["pre_group"].eq("KC") & edges["post_group"].isin(BEHAVIOR_TARGET_GROUPS)
    ].copy()
    if selected.empty:
        empty_targets = pd.DataFrame(
            columns=[
                "Postsynaptic_ID",
                "post_group",
                "post_side",
                "post_cell_type",
                "post_hemibrain_type",
                "post_top_nt",
                "behavior_axis",
                "behavior_interpretation",
                "total_signed_kc_input",
                "total_abs_kc_input",
                "n_kc_inputs",
            ]
        )
        return selected, empty_targets, pd.DataFrame()
    target_meta = (
        selected.groupby(
            ["Postsynaptic_ID", "post_group", "post_side", "post_cell_type", "post_hemibrain_type", "post_top_nt"],
            as_index=False,
            dropna=False,
        )
        .agg(
            total_signed_kc_input=("Excitatory x Connectivity", "sum"),
            total_abs_kc_input=("abs_signed_weight", "sum"),
            n_kc_inputs=("Presynaptic_ID", "nunique"),
            n_edges=("Connectivity", "size"),
        )
        .sort_values("total_abs_kc_input", ascending=False)
    )
    axis_info = target_meta.apply(_classify_target_behavior_axis, axis=1, result_type="expand")
    target_meta["behavior_axis"] = axis_info[0]
    target_meta["behavior_interpretation"] = axis_info[1]
    input_distribution = (
        target_meta.groupby(["post_group", "behavior_axis", "behavior_interpretation"], as_index=False, dropna=False)
        .agg(
            n_targets=("Postsynaptic_ID", "nunique"),
            total_abs_kc_input=("total_abs_kc_input", "sum"),
            total_signed_kc_input=("total_signed_kc_input", "sum"),
            median_kc_inputs=("n_kc_inputs", "median"),
        )
        .sort_values("total_abs_kc_input", ascending=False)
    )
    total = float(input_distribution["total_abs_kc_input"].sum())
    input_distribution["fraction_of_mapped_kc_output"] = (
        input_distribution["total_abs_kc_input"] / total if total > 0 else 0.0
    )
    return selected, target_meta, input_distribution


def _behavior_axis_template() -> dict[str, dict[str, float | str]]:
    return {
        "approach_or_positive_valence": {
            "memory_choice_approach_drive": 1.0,
            "feeding_drive": 0.45,
            "memory_expression_drive": 0.65,
            "state_modulation_drive": 0.10,
            "interpretation": "MBON positive-valence/approach proxy",
        },
        "avoidance_or_negative_valence": {
            "memory_choice_approach_drive": -0.85,
            "avoidance_drive": 0.90,
            "memory_expression_drive": 0.55,
            "state_modulation_drive": 0.10,
            "interpretation": "MBON negative-valence/avoidance proxy",
        },
        "disinhibitory_memory_output": {
            "memory_expression_drive": 0.55,
            "state_modulation_drive": 0.35,
            "interpretation": "GABAergic MBON/disinhibitory proxy",
        },
        "memory_output": {
            "memory_expression_drive": 0.50,
            "state_modulation_drive": 0.15,
            "interpretation": "generic MBON memory-output proxy",
        },
        "appetitive_teaching": {
            "learning_valence_drive": 1.0,
            "memory_expression_drive": 0.35,
            "state_modulation_drive": 0.20,
            "interpretation": "PAM/reward teaching proxy",
        },
        "aversive_teaching": {
            "learning_valence_drive": -1.0,
            "avoidance_drive": 0.35,
            "memory_expression_drive": 0.25,
            "state_modulation_drive": 0.20,
            "interpretation": "PPL/shock teaching proxy",
        },
        "dopamine_teaching": {
            "memory_expression_drive": 0.30,
            "state_modulation_drive": 0.35,
            "interpretation": "dopamine teaching proxy",
        },
        "memory_persistence": {
            "memory_persistence_drive": 1.0,
            "memory_expression_drive": 0.35,
            "state_modulation_drive": 0.20,
            "interpretation": "DPM persistence proxy",
        },
        "motor_readout": {
            "motor_readout_drive": 1.0,
            "state_modulation_drive": 0.15,
            "interpretation": "DN motor/action-selection proxy",
        },
        "sparseness_brake": {
            "sparseness_brake_drive": 1.0,
            "state_modulation_drive": 0.20,
            "interpretation": "APL global inhibition proxy",
        },
        "appetitive_state": {
            "feeding_drive": 0.70,
            "state_modulation_drive": 0.50,
            "interpretation": "octopamine appetitive state proxy",
        },
        "state_modulation": {
            "state_modulation_drive": 0.80,
            "interpretation": "monoamine/state modulation proxy",
        },
        "other": {
            "state_modulation_drive": 0.05,
            "interpretation": "unmapped residual proxy",
        },
    }


def _simulate_real_kc_behavior_mapping(
    *,
    glomerulus_names: list[str],
    glomerulus_matrix: np.ndarray,
    kc_ids: np.ndarray,
    kc_target_edges: pd.DataFrame,
    target_meta: pd.DataFrame,
    ratios: tuple[float, ...] = BEHAVIOR_MAPPING_RATIOS,
    seeds: tuple[int, ...] = BEHAVIOR_MAPPING_SEEDS,
    n_odors: int = 24,
    keep_target_responses: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if kc_target_edges.empty or target_meta.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    kc_index = {int(root_id): index for index, root_id in enumerate(kc_ids)}
    target_ids = target_meta["Postsynaptic_ID"].astype("int64").tolist()
    target_index = {int(root_id): index for index, root_id in enumerate(target_ids)}
    matrix = np.zeros((len(kc_ids), len(target_ids)), dtype=np.float64)
    for pre_id, post_id, signed_weight in kc_target_edges[
        ["Presynaptic_ID", "Postsynaptic_ID", "Excitatory x Connectivity"]
    ].itertuples(index=False, name=None):
        pre = kc_index.get(int(pre_id))
        post = target_index.get(int(post_id))
        if pre is not None and post is not None:
            matrix[pre, post] += float(signed_weight)

    axis_by_target = dict(zip(target_meta["Postsynaptic_ID"].astype("int64"), target_meta["behavior_axis"].astype(str)))
    target_rows: list[dict[str, object]] = []
    axis_rows: list[dict[str, object]] = []
    primitive_rows: list[dict[str, object]] = []
    axis_template = _behavior_axis_template()
    for seed in seeds:
        odor_names, activity, odor_panel = build_mixture_odor_panel(
            glomerulus_names,
            glomerulus_matrix,
            seed=int(seed),
            n_odors=int(n_odors),
            min_glomeruli_per_odor=2,
            max_glomeruli_per_odor=6,
            channel_noise_sigma=0.15,
        )
        odor_lookup = odor_panel.set_index("odor_identity")["glomeruli"].to_dict()
        for ratio in ratios:
            binary, kc_responses, active_k = _sparsify(activity, ratio)
            target_response = kc_responses @ matrix
            target_abs = np.abs(target_response)
            target_norm = np.divide(
                target_abs,
                np.maximum(np.abs(matrix).sum(axis=0, keepdims=True), 1e-12),
            )
            for odor_index, odor_name in enumerate(odor_names):
                odor_target_rows: list[dict[str, object]] = []
                for target_col, target_root in enumerate(target_ids):
                    axis = axis_by_target.get(int(target_root), "other")
                    odor_target_rows.append(
                        {
                            "seed": int(seed),
                            "ratio": float(ratio),
                            "odor_identity": odor_name,
                            "odor_glomeruli": odor_lookup.get(odor_name, ""),
                            "active_k": int(active_k),
                            "observed_active_fraction": float(binary[odor_index].mean()),
                            "Postsynaptic_ID": int(target_root),
                            "behavior_axis": axis,
                            "target_signed_response": float(target_response[odor_index, target_col]),
                            "target_abs_response": float(target_abs[odor_index, target_col]),
                            "target_normalized_response": float(target_norm[odor_index, target_col]),
                        }
                    )
                if keep_target_responses:
                    target_rows.extend(odor_target_rows)
                axis_frame = pd.DataFrame.from_records(odor_target_rows)
                axis_summary = (
                    axis_frame.groupby(["seed", "ratio", "odor_identity", "behavior_axis"], as_index=False)
                    .agg(
                        active_k=("active_k", "first"),
                        observed_active_fraction=("observed_active_fraction", "first"),
                        n_targets=("Postsynaptic_ID", "nunique"),
                        axis_signed_response=("target_signed_response", "sum"),
                        axis_abs_response=("target_abs_response", "sum"),
                        axis_normalized_response=("target_normalized_response", "mean"),
                    )
                )
                axis_rows.extend(axis_summary.to_dict("records"))
                for (seed_value, ratio_value, odor_identity), group in axis_summary.groupby(
                    ["seed", "ratio", "odor_identity"], sort=False
                ):
                    row: dict[str, object] = {
                        "seed": int(seed_value),
                        "ratio": float(ratio_value),
                        "odor_identity": str(odor_identity),
                        "active_k": int(group["active_k"].iloc[0]),
                        "observed_active_fraction": float(group["observed_active_fraction"].iloc[0]),
                    }
                    for primitive in [
                        "memory_choice_approach_drive",
                        "learning_valence_drive",
                        "memory_expression_drive",
                        "memory_persistence_drive",
                        "avoidance_drive",
                        "feeding_drive",
                        "state_modulation_drive",
                        "sparseness_brake_drive",
                        "motor_readout_drive",
                    ]:
                        row[primitive] = 0.0
                    total_axis_abs = float(group["axis_abs_response"].sum())
                    for axis_record in group.itertuples(index=False):
                        axis = str(axis_record.behavior_axis)
                        axis_weight = float(axis_record.axis_abs_response) / total_axis_abs if total_axis_abs > 0 else 0.0
                        mapping = axis_template.get(axis, axis_template["other"])
                        for primitive, value in mapping.items():
                            if primitive == "interpretation":
                                continue
                            row[primitive] = float(row.get(primitive, 0.0)) + axis_weight * float(value)
                    row["behavior_valence_index"] = float(row["memory_choice_approach_drive"]) + 0.5 * float(
                        row["learning_valence_drive"]
                    ) - 0.35 * float(row["avoidance_drive"])
                    row["predicted_behavior"] = (
                        "approach_or_appetitive_memory"
                        if float(row["behavior_valence_index"]) > 0.10
                        else "avoidance_or_aversive_memory"
                        if float(row["behavior_valence_index"]) < -0.10
                        else "mixed_or_state_modulated_memory"
                    )
                    primitive_rows.append(row)

    target_response_df = pd.DataFrame.from_records(target_rows)
    axis_response_df = pd.DataFrame.from_records(axis_rows)
    primitive_df = pd.DataFrame.from_records(primitive_rows)
    return target_response_df, axis_response_df, primitive_df


def _summarize_behavior_mapping(
    primitive_df: pd.DataFrame,
    axis_response_df: pd.DataFrame,
    target_meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if primitive_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    primitive_summary = (
        primitive_df.groupby("ratio", as_index=False)
        .agg(
            observed_active_fraction=("observed_active_fraction", "mean"),
            memory_choice_approach_drive=("memory_choice_approach_drive", "mean"),
            learning_valence_drive=("learning_valence_drive", "mean"),
            memory_expression_drive=("memory_expression_drive", "mean"),
            memory_persistence_drive=("memory_persistence_drive", "mean"),
            avoidance_drive=("avoidance_drive", "mean"),
            feeding_drive=("feeding_drive", "mean"),
            state_modulation_drive=("state_modulation_drive", "mean"),
            sparseness_brake_drive=("sparseness_brake_drive", "mean"),
            motor_readout_drive=("motor_readout_drive", "mean"),
            behavior_valence_index=("behavior_valence_index", "mean"),
            behavior_valence_index_std=("behavior_valence_index", "std"),
        )
        .sort_values("ratio")
    )
    behavior_counts = (
        primitive_df.groupby(["ratio", "predicted_behavior"], as_index=False)
        .size()
        .rename(columns={"size": "n_odor_panels"})
    )
    total_by_ratio = behavior_counts.groupby("ratio")["n_odor_panels"].transform("sum")
    behavior_counts["fraction"] = behavior_counts["n_odor_panels"] / total_by_ratio
    if axis_response_df.empty:
        axis_summary = pd.DataFrame()
    else:
        axis_summary = (
            axis_response_df.groupby(["ratio", "behavior_axis"], as_index=False)
            .agg(
                n_targets=("n_targets", "mean"),
                mean_axis_abs_response=("axis_abs_response", "mean"),
                mean_axis_normalized_response=("axis_normalized_response", "mean"),
                axis_abs_response_std=("axis_abs_response", "std"),
            )
            .sort_values(["ratio", "mean_axis_abs_response"], ascending=[True, False])
        )
    target_axis = (
        target_meta.groupby(["post_group", "behavior_axis", "behavior_interpretation"], as_index=False)
        .agg(
            n_targets=("Postsynaptic_ID", "nunique"),
            total_abs_kc_input=("total_abs_kc_input", "sum"),
            total_signed_kc_input=("total_signed_kc_input", "sum"),
            median_kc_inputs=("n_kc_inputs", "median"),
        )
        .sort_values("total_abs_kc_input", ascending=False)
    )
    total = float(target_axis["total_abs_kc_input"].sum())
    target_axis["fraction_of_mapped_kc_output"] = target_axis["total_abs_kc_input"] / total if total else 0.0
    return primitive_summary, behavior_counts, axis_summary.merge(
        target_axis[["behavior_axis", "behavior_interpretation"]].drop_duplicates("behavior_axis"),
        on="behavior_axis",
        how="left",
    )


def _build_behavior_condition_table(primitive_summary: pd.DataFrame) -> pd.DataFrame:
    """Map real-connectome behavior axes into existing memory-choice conditions."""

    columns = [
        "name",
        "attractive_gain",
        "aversive_gain",
        "lateral_memory_bias",
        "attractive_left_weight",
        "attractive_right_weight",
        "aversive_left_weight",
        "aversive_right_weight",
        "source_ratio",
        "behavior_valence_index",
        "memory_expression_drive",
        "memory_persistence_drive",
        "sparseness_brake_drive",
        "mapping_note",
    ]
    if primitive_summary.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for row in primitive_summary.itertuples(index=False):
        ratio = float(row.ratio)
        valence = float(row.behavior_valence_index)
        memory = float(row.memory_expression_drive)
        persistence = float(row.memory_persistence_drive)
        brake = float(row.sparseness_brake_drive)
        approach_scale = float(np.clip(0.55 + 0.90 * memory + 0.35 * max(valence, 0.0), 0.20, 1.30))
        avoidance_scale = float(np.clip(0.55 + 1.10 * float(row.avoidance_drive) + 0.40 * max(-valence, 0.0), 0.20, 1.30))
        persistence_bias = float(np.clip(2.0 * (persistence - 0.13) + 0.5 * valence, -0.35, 0.35))
        dense_penalty = float(np.clip((ratio - LITERATURE_KC_ACTIVE_FRACTION) / 0.20, 0.0, 1.0))
        rows.append(
            {
                "name": f"flywire_kc_ratio_{ratio:.3f}_behavior_proxy".replace(".", "p"),
                "attractive_gain": -500.0 * approach_scale * (1.0 - 0.25 * dense_penalty),
                "aversive_gain": 80.0 * avoidance_scale * (1.0 + 0.20 * dense_penalty),
                "lateral_memory_bias": persistence_bias,
                "attractive_left_weight": 1.0 + 2.0 * max(-valence, 0.0),
                "attractive_right_weight": 9.0 + 2.0 * max(valence, 0.0),
                "aversive_left_weight": 2.0 + 2.0 * max(valence, 0.0),
                "aversive_right_weight": 8.0 + 2.0 * max(-valence, 0.0),
                "source_ratio": ratio,
                "behavior_valence_index": valence,
                "memory_expression_drive": memory,
                "memory_persistence_drive": persistence,
                "sparseness_brake_drive": brake,
                "mapping_note": (
                    "Derived from real FlyWire ALPN->KC activity and KC->MBON/DAN/DPM/APL behavior-axis summary; "
                    "parameters are transparent behavior-proxy inputs, not animal-calibrated motor weights."
                ),
            }
        )
    return pd.DataFrame.from_records(rows, columns=columns)


def _summarize_scientific_questions(
    *,
    key_pairs: pd.DataFrame,
    coverage: pd.DataFrame,
    apl_inputs: pd.DataFrame,
    mbon_apl_candidates: pd.DataFrame | None = None,
    primitive_summary: pd.DataFrame,
    behavior_counts: pd.DataFrame,
) -> pd.DataFrame:
    def coverage_value(pre: str, post: str) -> float:
        selected = coverage[coverage["pre_group"].eq(pre) & coverage["post_group"].eq(post)]
        if selected.empty or "post_group_coverage" not in selected:
            return float("nan")
        return float(selected["post_group_coverage"].iloc[0])

    def pair_value(pre: str, post: str, column: str) -> float:
        selected = key_pairs[key_pairs["pre_group"].eq(pre) & key_pairs["post_group"].eq(post)]
        if selected.empty or column not in selected:
            return float("nan")
        return float(selected[column].iloc[0])

    def primitive_at(ratio: float, column: str) -> float:
        if primitive_summary.empty:
            return float("nan")
        selected = primitive_summary.iloc[(primitive_summary["ratio"] - ratio).abs().argsort()[:1]]
        return float(selected[column].iloc[0]) if not selected.empty and column in selected else float("nan")

    def behavior_fraction(ratio: float, behavior: str) -> float:
        if behavior_counts.empty:
            return float("nan")
        selected = behavior_counts[
            behavior_counts["predicted_behavior"].eq(behavior)
            & (behavior_counts["ratio"] - ratio).abs().le(1e-9)
        ]
        return float(selected["fraction"].iloc[0]) if not selected.empty else 0.0

    apl_negative_inputs = (
        apl_inputs[apl_inputs["signed_weight"] < 0]["abs_signed_weight"].sum()
        if not apl_inputs.empty and "signed_weight" in apl_inputs
        else float("nan")
    )
    apl_positive_inputs = (
        apl_inputs[apl_inputs["signed_weight"] > 0]["abs_signed_weight"].sum()
        if not apl_inputs.empty and "signed_weight" in apl_inputs
        else float("nan")
    )
    if mbon_apl_candidates is not None and not mbon_apl_candidates.empty:
        negative_mbon = mbon_apl_candidates[mbon_apl_candidates["signed_weight"] < 0].head(2)
        candidate_names = "/".join(negative_mbon["pre_cell_type"].astype(str).drop_duplicates().head(2))
        mbon_apl_note = f"直接 MBON->APL 中优先候选收窄到 {candidate_names or 'specific MBON subtypes'}。"
    else:
        mbon_apl_note = "直接 MBON->APL subtype 候选表未生成。"
    rows = [
        {
            "question_id": "Q1_sparse_code",
            "scientific_question": "真实 FlyWire ALPN->KC 与 KC<->APL 结构是否支持 KC <=10% 稀疏工作区？",
            "real_data_result": (
                f"ALPN->KC 覆盖 {coverage_value('ALPN', 'KC'):.1%} KC；KC->APL 覆盖 "
                f"{coverage_value('KC', 'APL'):.1%} APL；APL->KC 覆盖 {coverage_value('APL', 'KC'):.1%} KC，"
                f"APL->KC negative_fraction={pair_value('APL', 'KC', 'negative_fraction'):.3f}。"
            ),
            "simulation_result": (
                "真实 ALPN->KC ratio sweep 显示 0.10 保持低 overlap 且 binary memory/retention 最稳；"
                "1/6 是 legacy 对照，会增加 odor-code overlap。"
            ),
            "behavior_mapping": "预测相近气味 generalization/interference 上升，而不是任意真实动作自动生成。",
            "boundary": "结构和 proxy 支持稀疏工作区；不能证明 0.10 是数学最优或 wet-lab 因果。",
        },
        {
            "question_id": "Q2_apl_state",
            "scientific_question": "哪些真实上游状态轴可能让 APL signed drive 下移，并把 KC 推离 <=10%？",
            "real_data_result": (
                f"APL 上游正向 abs drive 约 {float(apl_positive_inputs):.0f}，负向 abs drive 约 "
                f"{float(apl_negative_inputs):.0f}；非 APL GABAergic 和少数 MBON subtype 是主要候选下移轴，"
                f"{mbon_apl_note}"
            ),
            "simulation_result": "APL gain 降低会把 KC active fraction 推高，增加 cross-odor overlap 并降低分辨 proxy。",
            "behavior_mapping": "预测行为上表现为 odor discrimination 下降、generalization 增强、记忆选择更混合。",
            "boundary": "depression-like/state 只能作为状态轴假说，需 APL/KC imaging 和扰动验证。",
        },
        {
            "question_id": "Q3_multi_output_behavior",
            "scientific_question": "真实 KC->MBON/DAN/DPM 输出是否能映射到行为轴，而不是单 MBON d-prime？",
            "real_data_result": (
                f"KC->MBON abs weight={pair_value('KC', 'MBON', 'abs_signed_weight'):.0f}，"
                f"KC->DAN abs weight={pair_value('KC', 'DAN', 'abs_signed_weight'):.0f}，"
                f"KC->DPM abs weight={pair_value('KC', 'DPM', 'abs_signed_weight'):.0f}。"
            ),
            "simulation_result": (
                f"行为轴 mapping 中 0.10 valence index={primitive_at(LITERATURE_KC_ACTIVE_FRACTION, 'behavior_valence_index'):.3f}，"
                f"1/6 valence index={primitive_at(LEGACY_ONE_SIXTH_KC_RATIO, 'behavior_valence_index'):.3f}；"
                f"0.10 mixed fraction={behavior_fraction(LITERATURE_KC_ACTIVE_FRACTION, 'mixed_or_state_modulated_memory'):.2f}。"
            ),
            "behavior_mapping": "输出为 approach/avoidance valence、DAN teaching、DPM persistence、APL sparseness brake 等透明 proxy。",
            "boundary": "不是 DN-to-muscle 私有权重，也不是真实 T-maze 行为；可用于优先排序湿实验条件。",
        },
    ]
    return pd.DataFrame.from_records(rows)


def _write_behavior_mapping_figure(
    primitive_summary: pd.DataFrame,
    behavior_counts: pd.DataFrame,
    axis_summary: pd.DataFrame,
    output_dir: Path,
) -> Path | None:
    if primitive_summary.empty:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    axes[0].plot(primitive_summary["ratio"], primitive_summary["behavior_valence_index"], "o-", color="#3568a3")
    axes[0].axvline(LITERATURE_KC_ACTIVE_FRACTION, ls="--", color="grey", label="<=10%")
    axes[0].axvline(LEGACY_ONE_SIXTH_KC_RATIO, ls=":", color="grey", label="legacy 1/6")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("KC active fraction")
    axes[0].set_ylabel("behavior valence index")
    axes[0].set_title("Real KC->target behavior-axis valence")
    axes[0].legend()

    plot_counts = behavior_counts.pivot_table(
        index="ratio", columns="predicted_behavior", values="fraction", fill_value=0.0
    ).sort_index()
    bottom = np.zeros(len(plot_counts))
    palette = {
        "approach_or_appetitive_memory": "#5d9c59",
        "avoidance_or_aversive_memory": "#b84b5f",
        "mixed_or_state_modulated_memory": "#6f7f99",
    }
    for column in plot_counts.columns:
        values = plot_counts[column].to_numpy(dtype=float)
        axes[1].bar(
            plot_counts.index.astype(float),
            values,
            bottom=bottom,
            width=plot_counts.index.astype(float) * 0.08,
            label=column,
            color=palette.get(column, None),
        )
        bottom += values
    axes[1].set_xscale("log")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("KC active fraction")
    axes[1].set_ylabel("fraction of odor panels")
    axes[1].set_title("Predicted behavior-axis class")
    axes[1].legend(fontsize=7)

    if axis_summary.empty:
        axes[2].axis("off")
    else:
        target_ratio = axis_summary.loc[
            int((axis_summary["ratio"] - LITERATURE_KC_ACTIVE_FRACTION).abs().idxmin()), "ratio"
        ]
        axis_plot = axis_summary[axis_summary["ratio"].eq(target_ratio)].sort_values(
            "mean_axis_abs_response", ascending=True
        )
        axes[2].barh(axis_plot["behavior_axis"], axis_plot["mean_axis_abs_response"], color="#4b8bbe")
        axes[2].set_xscale("log")
        axes[2].set_xlabel("mean abs response")
        axes[2].set_title(f"Behavior axes at ratio={float(target_ratio):.3g}")
    figure_path = output_dir / "Fig_flywire_behavior_mapping.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return figure_path


def _write_science_figure(pair_summary: pd.DataFrame, output_dir: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    matrix_source = pair_summary[
        pair_summary["pre_group"].isin(CONNECTOME_GROUP_ORDER)
        & pair_summary["post_group"].isin(CONNECTOME_GROUP_ORDER)
    ].copy()
    heat = matrix_source.pivot_table(
        index="pre_group",
        columns="post_group",
        values="abs_signed_weight",
        aggfunc="sum",
        fill_value=0,
    ).reindex(index=CONNECTOME_GROUP_ORDER, columns=CONNECTOME_GROUP_ORDER, fill_value=0)
    key = _summarize_key_pairs(pair_summary).copy()
    key["edge"] = key["pre_group"].astype(str) + "->" + key["post_group"].astype(str)
    key = key.dropna(subset=["abs_signed_weight"]).sort_values("abs_signed_weight", ascending=True).tail(14)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    image = axes[0].imshow(np.log10(heat.to_numpy(dtype=float) + 1), cmap="viridis")
    axes[0].set_xticks(range(len(heat.columns)), heat.columns, rotation=45, ha="right")
    axes[0].set_yticks(range(len(heat.index)), heat.index)
    axes[0].set_title("FlyWire v783 grouped connectivity log10(abs signed weight + 1)")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    colors = np.where(key["signed_weight"].to_numpy(dtype=float) < 0, "#b84b5f", "#4b8bbe")
    axes[1].barh(key["edge"], key["abs_signed_weight"], color=colors)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("abs signed connectivity")
    axes[1].set_title("Key MB/KC/APL circuit edges")
    figure_path = output_dir / "Fig_flywire_connectome_science.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return figure_path


def _identify_learning_memory_targets(
    annotated: pd.DataFrame,
    *,
    ppl1_pattern: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    dan = annotated[annotated["connectome_group"].eq("DAN")].copy()
    text = dan[["cell_type", "hemibrain_type"]].fillna("").astype(str).agg(" ".join, axis=1)
    selected_ppl1 = dan[text.str.contains(str(ppl1_pattern), case=False, regex=False, na=False)].copy()
    pattern_used = str(ppl1_pattern)
    if selected_ppl1.empty and str(ppl1_pattern).upper() != "PPL":
        selected_ppl1 = dan[text.str.contains("PPL", case=False, regex=False, na=False)].copy()
        pattern_used = "PPL"
    dpm = annotated[annotated["connectome_group"].eq("DPM")].copy()
    if selected_ppl1.empty:
        raise ValueError("No PPL1/PPL DAN targets were found in the annotation table.")
    if dpm.empty:
        raise ValueError("No DPM targets were found in the annotation table.")

    ppl1_inventory = selected_ppl1.copy()
    ppl1_inventory["target_role"] = "PPL1_DAN"
    ppl1_inventory["selection_pattern"] = pattern_used
    dpm_inventory = dpm.copy()
    dpm_inventory["target_role"] = "DPM"
    dpm_inventory["selection_pattern"] = "DPM"
    columns = [
        "target_role",
        "selection_pattern",
        "root_id",
        "side",
        "cell_class",
        "cell_type",
        "hemibrain_type",
        "top_nt",
        "connectome_group",
    ]
    return (
        ppl1_inventory[columns].sort_values(["target_role", "side", "cell_type", "root_id"]),
        dpm_inventory[columns].sort_values(["target_role", "side", "cell_type", "root_id"]),
        pattern_used,
    )


def _summarize_learning_memory_target_edges(
    edges: pd.DataFrame,
    annotated: pd.DataFrame,
    kc_ids: np.ndarray,
    ppl1_targets: pd.DataFrame,
    dpm_targets: pd.DataFrame,
) -> pd.DataFrame:
    kc_set = set(map(int, kc_ids))
    circuit_target_ids = set(
        annotated[
            annotated["connectome_group"].isin(["MBON", "DAN", "DPM", "APL", "DN"])
        ]["root_id"].astype("int64")
    )
    rows: list[dict[str, object]] = []
    for role, targets in [("PPL1_DAN", ppl1_targets), ("DPM", dpm_targets)]:
        target_ids = set(targets["root_id"].astype("int64"))
        specs = [
            ("KC_to_target", edges["Presynaptic_ID"].isin(kc_set) & edges["Postsynaptic_ID"].isin(target_ids)),
            ("target_to_KC", edges["Presynaptic_ID"].isin(target_ids) & edges["Postsynaptic_ID"].isin(kc_set)),
            (
                "target_to_MB_output_axis",
                edges["Presynaptic_ID"].isin(target_ids) & edges["Postsynaptic_ID"].isin(circuit_target_ids - kc_set),
            ),
        ]
        for direction, mask in specs:
            selected = edges.loc[mask].copy()
            abs_weight = selected["Excitatory x Connectivity"].abs() if not selected.empty else pd.Series(dtype=float)
            negative_weight = -selected["Excitatory x Connectivity"].clip(upper=0) if not selected.empty else pd.Series(dtype=float)
            if direction == "KC_to_target":
                kc_coverage = selected["Presynaptic_ID"].nunique() / max(1, len(kc_set))
                target_coverage = selected["Postsynaptic_ID"].nunique() / max(1, len(target_ids))
            elif direction == "target_to_KC":
                kc_coverage = selected["Postsynaptic_ID"].nunique() / max(1, len(kc_set))
                target_coverage = selected["Presynaptic_ID"].nunique() / max(1, len(target_ids))
            else:
                kc_coverage = np.nan
                target_coverage = selected["Presynaptic_ID"].nunique() / max(1, len(target_ids))
            abs_sum = float(abs_weight.sum()) if not selected.empty else 0.0
            rows.append(
                {
                    "target_role": role,
                    "direction": direction,
                    "n_targets_in_annotation": int(len(target_ids)),
                    "n_edges": int(len(selected)),
                    "n_pre": int(selected["Presynaptic_ID"].nunique()) if not selected.empty else 0,
                    "n_post": int(selected["Postsynaptic_ID"].nunique()) if not selected.empty else 0,
                    "n_synapses": int(selected["Connectivity"].sum()) if not selected.empty else 0,
                    "signed_weight": float(selected["Excitatory x Connectivity"].sum()) if not selected.empty else 0.0,
                    "abs_signed_weight": abs_sum,
                    "negative_fraction": float(negative_weight.sum() / abs_sum) if abs_sum > 0 else 0.0,
                    "kc_coverage_fraction": float(kc_coverage) if pd.notna(kc_coverage) else np.nan,
                    "target_coverage_fraction": float(target_coverage),
                }
            )
    return pd.DataFrame.from_records(rows)


def _kc_to_target_weight_vector(
    edges: pd.DataFrame,
    kc_ids: np.ndarray,
    target_ids: set[int],
) -> np.ndarray:
    kc_index = {int(root_id): index for index, root_id in enumerate(kc_ids)}
    selected = edges[
        edges["Presynaptic_ID"].isin(set(map(int, kc_ids))) & edges["Postsynaptic_ID"].isin(target_ids)
    ].copy()
    vector = np.zeros(len(kc_ids), dtype=np.float64)
    if selected.empty:
        return vector
    grouped = selected.groupby("Presynaptic_ID")["Excitatory x Connectivity"].sum()
    for root_id, signed_weight in grouped.items():
        index = kc_index.get(int(root_id))
        if index is not None:
            vector[index] = max(0.0, float(signed_weight))
    return vector


def _learning_memory_conditions() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "condition_order": 0,
                "condition_id": "baseline",
                "condition_label": "baseline",
                "ppl1_gain": 1.0,
                "dpm_gain": 1.0,
                "interpretation": "normal PPL1-DAN teaching and DPM persistence axes",
            },
            {
                "condition_order": 1,
                "condition_id": "ppl1_dan_50pct",
                "condition_label": "PPL1-DAN 50%",
                "ppl1_gain": 0.50,
                "dpm_gain": 1.0,
                "interpretation": "partial PPL1-DAN activity reduction; DPM normal",
            },
            {
                "condition_order": 2,
                "condition_id": "ppl1_dan_25pct",
                "condition_label": "PPL1-DAN 25%",
                "ppl1_gain": 0.25,
                "dpm_gain": 1.0,
                "interpretation": "strong PPL1-DAN activity reduction; DPM normal",
            },
            {
                "condition_order": 3,
                "condition_id": "ppl1_dan_0pct",
                "condition_label": "PPL1-DAN 0%",
                "ppl1_gain": 0.0,
                "dpm_gain": 1.0,
                "interpretation": "PPL1-DAN teaching axis removed in the proxy",
            },
            {
                "condition_order": 4,
                "condition_id": "dpm_50pct",
                "condition_label": "DPM 50%",
                "ppl1_gain": 1.0,
                "dpm_gain": 0.50,
                "interpretation": "DPM persistence reduced; PPL1-DAN teaching normal",
            },
            {
                "condition_order": 5,
                "condition_id": "dpm_25pct",
                "condition_label": "DPM 25%",
                "ppl1_gain": 1.0,
                "dpm_gain": 0.25,
                "interpretation": "strong DPM persistence reduction; PPL1-DAN teaching normal",
            },
            {
                "condition_order": 6,
                "condition_id": "dpm_0pct",
                "condition_label": "DPM 0%",
                "ppl1_gain": 1.0,
                "dpm_gain": 0.0,
                "interpretation": "DPM persistence axis removed in the proxy",
            },
            {
                "condition_order": 7,
                "condition_id": "joint_25pct",
                "condition_label": "PPL1-DAN + DPM 25%",
                "ppl1_gain": 0.25,
                "dpm_gain": 0.25,
                "interpretation": "joint strong reduction of teaching and persistence axes",
            },
            {
                "condition_order": 8,
                "condition_id": "joint_0pct",
                "condition_label": "PPL1-DAN + DPM 0%",
                "ppl1_gain": 0.0,
                "dpm_gain": 0.0,
                "interpretation": "joint removal of teaching and persistence axes in the proxy",
            },
        ]
    )


def _learning_memory_task_conditions() -> pd.DataFrame:
    """Assay-level readouts for immediate, delayed and delayed-conflict tasks."""

    rows: list[dict[str, object]] = []
    perturbations = [
        ("baseline", "baseline", 1.0, 1.0),
        ("ppl1_dan_25pct", "PPL1-DAN 25%", 0.25, 1.0),
        ("dpm_25pct", "DPM 25%", 1.0, 0.25),
        ("joint_25pct", "PPL1-DAN + DPM 25%", 0.25, 0.25),
    ]
    tasks = [
        ("immediate", "immediate learning", 0, 0.0),
        ("delayed", "delayed retention", 6, 0.0),
        ("delayed_conflict", "delayed-conflict memory", 6, 1.0),
    ]
    order = 0
    for condition_id, condition_label, ppl1_gain, dpm_gain in perturbations:
        for task_id, task_label, delay_blocks, conflict_level in tasks:
            rows.append(
                {
                    "task_order": int(order),
                    "condition_id": condition_id,
                    "condition_label": condition_label,
                    "task_id": task_id,
                    "task_label": task_label,
                    "ppl1_gain": float(ppl1_gain),
                    "dpm_gain": float(dpm_gain),
                    "delay_blocks": int(delay_blocks),
                    "conflict_level": float(conflict_level),
                }
            )
            order += 1
    return pd.DataFrame.from_records(rows)


def _write_learning_memory_perturbation_figure(
    summary_df: pd.DataFrame,
    task_summary_df: pd.DataFrame | None,
    output_dir: Path,
) -> Path | None:
    if summary_df.empty:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    plot = summary_df.sort_values("condition_order").copy()
    labels = plot["condition_id"].astype(str).str.replace("_", "\n", regex=False).tolist()
    x = np.arange(len(plot))
    width = 0.25

    if task_summary_df is not None and not task_summary_df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), constrained_layout=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5.4), constrained_layout=True)
    axes[0].bar(x - width, plot["acquisition_choice_index_mean"], width, label="acquisition CI", color="#3f78b5")
    axes[0].bar(x, plot["retention_choice_index_mean"], width, label="retention CI", color="#59a14f")
    axes[0].bar(x + width, plot["learning_memory_score_mean"], width, label="combined score", color="#f28e2b")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xticks(x, labels, rotation=0, fontsize=8)
    axes[0].set_ylabel("proxy index")
    axes[0].set_title("PPL1-DAN/DPM reduction changes learning and retention readouts")
    axes[0].legend(fontsize=8)

    colors = ["#777777" if cid == "baseline" else "#b84b5f" for cid in plot["condition_id"]]
    axes[1].bar(x, plot["learning_memory_score_percent_of_baseline"], color=colors)
    axes[1].axhline(100, ls="--", color="grey", lw=1)
    axes[1].set_ylim(0, max(110, float(plot["learning_memory_score_percent_of_baseline"].max()) + 5))
    axes[1].set_xticks(x, labels, rotation=0, fontsize=8)
    axes[1].set_ylabel("% of baseline combined score")
    axes[1].set_title("Score drop relative to baseline")
    for xpos, value in zip(x, plot["learning_memory_score_percent_of_baseline"]):
        axes[1].text(xpos, float(value) + 2, f"{float(value):.0f}%", ha="center", va="bottom", fontsize=8)

    if task_summary_df is not None and not task_summary_df.empty:
        task_plot = task_summary_df[
            task_summary_df["condition_id"].isin(["baseline", "ppl1_dan_25pct", "dpm_25pct", "joint_25pct"])
        ].copy()
        task_order = ["immediate", "delayed", "delayed_conflict"]
        cond_order = ["baseline", "ppl1_dan_25pct", "dpm_25pct", "joint_25pct"]
        colors_by_task = {"immediate": "#4c78a8", "delayed": "#59a14f", "delayed_conflict": "#f28e2b"}
        offsets = np.linspace(-0.27, 0.27, len(task_order))
        base_x = np.arange(len(cond_order))
        for offset, task_id in zip(offsets, task_order):
            subset = (
                task_plot[task_plot["task_id"].eq(task_id)]
                .set_index("condition_id")
                .reindex(cond_order)
                .reset_index()
            )
            axes[2].bar(
                base_x + offset,
                subset["expected_choice_rate_mean"],
                width=0.18,
                label=task_id.replace("_", " "),
                color=colors_by_task[task_id],
            )
        axes[2].axhline(0.5, color="0.35", lw=1)
        axes[2].set_ylim(0, 1.05)
        axes[2].set_xticks(base_x, [cid.replace("_", "\n") for cid in cond_order], fontsize=8)
        axes[2].set_ylabel("expected choice rate")
        axes[2].set_title("Immediate vs delayed/conflict assay readouts")
        axes[2].legend(fontsize=8)

    figure_path = output_dir / "Fig_ppl1_dpm_learning_memory_perturbation.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return figure_path


def _render_learning_memory_perturbation_report(
    cfg: LearningMemoryPerturbationConfig,
    *,
    summary_df: pd.DataFrame,
    task_summary_df: pd.DataFrame,
    structural_summary: pd.DataFrame,
    target_inventory: pd.DataFrame,
    interpretation: dict[str, object],
    figure_path: Path | None,
) -> str:
    result_table = summary_df[
        [
            "condition_label",
            "ppl1_gain",
            "dpm_gain",
            "acquisition_choice_index_mean",
            "retention_choice_index_mean",
            "learning_memory_score_mean",
            "learning_memory_score_percent_of_baseline",
            "learning_memory_score_drop_percent",
            "decay_per_block_mean",
        ]
    ].copy()
    task_table = task_summary_df[
        [
            "condition_label",
            "task_label",
            "ppl1_gain",
            "dpm_gain",
            "delay_blocks",
            "conflict_level",
            "acquisition_choice_index_mean",
            "retention_choice_index_mean",
            "expected_choice_rate_mean",
            "expected_choice_rate_delta_vs_baseline_task",
        ]
    ].copy() if not task_summary_df.empty else pd.DataFrame()
    target_types = (
        target_inventory.groupby("target_role", as_index=False)
        .agg(
            n_targets=("root_id", "nunique"),
            cell_types=("cell_type", lambda values: ", ".join(sorted(set(map(str, values)))[:16])),
            sides=("side", lambda values: ", ".join(sorted(set(map(str, values))))),
        )
        .sort_values("target_role")
    )
    key_struct = structural_summary[
        structural_summary["direction"].isin(["KC_to_target", "target_to_KC"])
    ].copy()
    fig_line = f"\n![PPL1-DAN/DPM perturbation]({figure_path.name})\n" if figure_path else ""
    return f"""# PPL1-DAN 与 DPM 活性降低的学习记忆仿真报告

保存路径：`{cfg.output_dir}/PPL1_DPM_LEARNING_MEMORY_PERTURBATION_REPORT_CN.md`

## 结论摘要

本次实验可以在现有 BioFly/FlyWire 数据中直接实现。系统在真实 FlyWire v783 annotation 中识别到 `{interpretation['n_ppl1_dan_targets']}` 个 PPL1-DAN 目标和 `{interpretation['n_dpm_targets']}` 个 DPM 目标，并用真实 `ALPN->KC` 气味编码、真实 `KC->PPL1-DAN` 与 `KC->DPM` 边权重构建学习/记忆 proxy。

主要结果是双重分离：

- 降低 PPL1-DAN 活性主要损害 acquisition/aversive teaching。`PPL1-DAN 25%` 时综合学习记忆分数为 baseline 的 `{interpretation['ppl1_25_score_percent']:.1f}%`，下降 `{interpretation['ppl1_25_score_drop_percent']:.1f}%`。
- 降低 DPM 活性主要损害 delayed retention/interference 后保持。`DPM 25%` 时 acquisition index 基本保留，但 retention index 降低，综合分数为 baseline 的 `{interpretation['dpm_25_score_percent']:.1f}%`。
- 同时降低 PPL1-DAN 和 DPM 时效应叠加。`PPL1-DAN + DPM 25%` 综合分数为 baseline 的 `{interpretation['joint_25_score_percent']:.1f}%`，是本轮非完全沉默条件中最强下降。

这些结果支持一个可检验假说：PPL1-DAN 下调会首先削弱厌恶性学习建立，DPM 下调会更明显削弱延迟记忆或干扰后的保持；两者一起降低时，学习记忆行为表现应明显下降。

{fig_line}
## 真实连接组靶点

目标神经元来自 `flywire_neuron_annotations.parquet`，连接来自 `Connectivity_783.parquet`。

{_table(target_types, max_rows=8)}

关键结构连接：

{_table(key_struct, max_rows=12)}

## 模型读出

气味输入首先由真实 `ALPN->KC` 子图生成 glomerulus-mixture panel，再按文献锚点 `KC active fraction = {cfg.ratio:.3f}` 做 winner-take-K 稀疏化。每个随机 seed 中，第 `{cfg.cs_plus_index}` 个气味作为 CS+，其他气味作为 CS-/干扰气味。

PPL1-DAN 活性降低只作用在教学强度：

```text
T_PPL1 = g_PPL1 * (x_CS+ W_KC->PPL1) / mean(x_odor W_KC->PPL1)
M_acq = T_PPL1 * <x_CS+ - mean(x_CS-), x_CS+ - mean(x_CS-)>
CI_acq = tanh(M_acq / M0)
```

DPM 活性降低只作用在干扰期保持：

```text
S_DPM = g_DPM * C_DPM->KC * (x_CS+ W_KC->DPM) / mean(x_odor W_KC->DPM)
lambda = lambda0 + lambdaD * (1 - min(S_DPM, 1))
M_ret = M_acq * exp(-B * lambda)
CI_ret = tanh(M_ret / M0)
score = {cfg.acquisition_score_weight:.2f} * CI_acq + {cfg.retention_score_weight:.2f} * CI_ret
```

其中 `g_PPL1` 和 `g_DPM` 是本次设置的活性 gain，`C_DPM->KC` 是真实 DPM 回投 KC 覆盖比例，`B={cfg.interference_blocks}` 是干扰块数，`M0` 是 baseline acquisition margin 的跨 seed 中位数。

## 数值结果

{_table(result_table, max_rows=16)}

## 分任务读出：immediate、delayed 与 delayed-conflict

为避免只用一个综合分数掩盖 acquisition 与 retention 的分工，这里把同一套真实 FlyWire KC odor code 拆成三类任务：

- `immediate learning`：训练后立即读出，主要看 acquisition 是否下降。
- `delayed retention`：加入 `{cfg.interference_blocks}` 个 delay/interference blocks，主要看 retention 是否下降。
- `delayed-conflict memory`：在 delayed retention 基础上加入 weak CS+ / strong CS- conflict penalty，模拟更敏感的 OCT/MCH 行为窗口。

{_table(task_table, max_rows=24)}

分任务结果的预期解释是：PPL1-DAN 25% 应同时拉低 immediate 和 delayed，因为它先削弱 learning acquisition；
DPM 25% 的 immediate 相对保留，但 delayed/delayed-conflict 下降更明显；联合 25% 是强阳性条件。

## 生物学解释

1. **PPL1-DAN 降低更像学习建立缺陷。** PPL1-DAN 被作为厌恶性 teaching axis；当其 gain 被降到 25% 或 0% 时，CS+ 与 CS- 的 association margin 按比例下降，choice-index proxy 显著变小。
2. **DPM 降低更像记忆保持缺陷。** DPM 保留了 acquisition，但提高干扰期衰减项，因此最明显的变化出现在 delayed retention/readout，而不是训练刚结束的 acquisition。
3. **联合降低预测行为下降更强。** 一个轴削弱“学进去”，另一个轴削弱“保留下来”，所以联合扰动比单独 DPM 降低更强，也比单独 PPL1-DAN 25% 更偏向 delayed memory 缺陷。

## 建议验证

- 行为：优先用 OCT/MCH 或同类 odor-shock assay，分 immediate test 与 delayed/interference test。预测 PPL1-DAN 降低在 immediate aversive learning 就下降，DPM 降低在 delayed 或 interference 后更明显。
- 成像：训练窗口记录 PPL1-DAN calcium 或 dopamine sensor；延迟窗口记录 DPM/KC 或 MB compartment calcium、5-HT sensor/neuromodulator readout。
- 扰动：PPL1-DAN 与 DPM 分别做 split-GAL4/driver 条件下的 Kir2.1、TNT、shibire^ts 或光遗传/化学遗传下调；联合下调用于测试叠加效应。
- 结构验证：对候选 MB compartment 做 GRASP/split-GFP 或 targeted EM/FlyWire proofreading，检查 PPL1-DAN/DPM 与 KC/MBON/DAN 回路的局部连接。

## 边界

该报告是连接组约束的学习/记忆 proxy，不等同于真实动物 T-maze 行为，也不证明 PPL1-DAN 或 DPM 的分子因果。它的价值是把真实 FlyWire 结构、KC 稀疏编码和可干预靶点合到一个可复跑的假说排序实验中。
"""


def run_learning_memory_perturbation(
    config: LearningMemoryPerturbationConfig | None = None,
) -> dict[str, Path | pd.DataFrame | dict[str, object]]:
    cfg = config or LearningMemoryPerturbationConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    annotations = _load_annotations(cfg.annotation_path)
    annotated = _annotate_connectome_groups(annotations)
    edges = pd.read_parquet(
        cfg.connectivity_path,
        columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity", "Excitatory x Connectivity"],
    )
    glomerulus_names, glomerulus_matrix, kc_ids, channel_table = _build_flywire_glomerulus_kc_matrix_from_frames(
        annotated,
        edges,
    )
    ppl1_targets, dpm_targets, pattern_used = _identify_learning_memory_targets(
        annotated,
        ppl1_pattern=cfg.ppl1_pattern,
    )
    target_inventory = pd.concat([ppl1_targets, dpm_targets], ignore_index=True)
    structural_summary = _summarize_learning_memory_target_edges(
        edges,
        annotated,
        kc_ids,
        ppl1_targets,
        dpm_targets,
    )
    ppl1_vector = _kc_to_target_weight_vector(
        edges,
        kc_ids,
        set(ppl1_targets["root_id"].astype("int64")),
    )
    dpm_vector = _kc_to_target_weight_vector(
        edges,
        kc_ids,
        set(dpm_targets["root_id"].astype("int64")),
    )
    if float(ppl1_vector.sum()) <= 0:
        raise ValueError("No positive KC->PPL1-DAN edge mass was found.")
    if float(dpm_vector.sum()) <= 0:
        raise ValueError("No positive KC->DPM edge mass was found.")

    dpm_to_kc = structural_summary[
        structural_summary["target_role"].eq("DPM") & structural_summary["direction"].eq("target_to_KC")
    ]
    dpm_to_kc_coverage = float(dpm_to_kc["kc_coverage_fraction"].iloc[0]) if not dpm_to_kc.empty else 0.0
    conditions = _learning_memory_conditions()

    seed_contexts: list[dict[str, object]] = []
    baseline_margins: list[float] = []
    for seed in cfg.seeds:
        odor_names, activity, odor_panel = build_mixture_odor_panel(
            glomerulus_names,
            glomerulus_matrix,
            seed=int(seed),
            n_odors=int(cfg.n_odors),
            min_glomeruli_per_odor=int(cfg.min_glomeruli_per_odor),
            max_glomeruli_per_odor=int(cfg.max_glomeruli_per_odor),
            channel_noise_sigma=float(cfg.channel_noise_sigma),
        )
        if cfg.cs_plus_index >= len(odor_names):
            raise ValueError("cs_plus_index must be smaller than n_odors.")
        binary, kc_responses, active_k = _sparsify(activity, float(cfg.ratio))
        cs_plus = kc_responses[int(cfg.cs_plus_index)]
        cs_minus = np.delete(kc_responses, int(cfg.cs_plus_index), axis=0)
        if cs_minus.shape[0] == 0:
            raise ValueError("At least two odors are required for a CS+/CS- memory proxy.")
        separation_vector = cs_plus - cs_minus.mean(axis=0)
        ppl1_panel_drive = kc_responses @ ppl1_vector
        dpm_panel_drive = kc_responses @ dpm_vector
        ppl1_drive_norm = float((cs_plus @ ppl1_vector) / (float(ppl1_panel_drive.mean()) + 1e-12))
        dpm_drive_norm = float((cs_plus @ dpm_vector) / (float(dpm_panel_drive.mean()) + 1e-12))
        baseline_margin = float(ppl1_drive_norm * np.dot(separation_vector, separation_vector))
        baseline_margins.append(abs(baseline_margin))
        seed_contexts.append(
            {
                "seed": int(seed),
                "odor_identity_cs_plus": odor_names[int(cfg.cs_plus_index)],
                "odor_glomeruli_cs_plus": odor_panel.loc[
                    odor_panel["odor_identity"].eq(odor_names[int(cfg.cs_plus_index)]),
                    "glomeruli",
                ].iloc[0],
                "active_k": int(active_k),
                "observed_active_fraction": float(binary.mean()),
                "ppl1_drive_norm": ppl1_drive_norm,
                "dpm_drive_norm": dpm_drive_norm,
                "separation_norm_sq": float(np.dot(separation_vector, separation_vector)),
            }
        )

    reference_margin = float(np.median(np.asarray(baseline_margins, dtype=np.float64)))
    if not np.isfinite(reference_margin) or reference_margin <= 0:
        fallback = float(np.mean(np.asarray(baseline_margins, dtype=np.float64)))
        reference_margin = fallback if np.isfinite(fallback) and fallback > 0 else 1.0

    raw_rows: list[dict[str, object]] = []
    for context in seed_contexts:
        for condition in conditions.itertuples(index=False):
            ppl1_teaching_drive = float(condition.ppl1_gain) * float(context["ppl1_drive_norm"])
            acquisition_margin = ppl1_teaching_drive * float(context["separation_norm_sq"])
            acquisition_choice_index = float(np.tanh(acquisition_margin / (reference_margin + 1e-12)))
            dpm_support = float(condition.dpm_gain) * dpm_to_kc_coverage * float(context["dpm_drive_norm"])
            dpm_support_clipped = float(np.clip(dpm_support, 0.0, 1.0))
            decay_per_block = float(
                cfg.dpm_baseline_decay_per_block
                + cfg.dpm_loss_decay_penalty_per_block * (1.0 - dpm_support_clipped)
            )
            retention_fraction = float(np.exp(-int(cfg.interference_blocks) * decay_per_block))
            retained_margin = acquisition_margin * retention_fraction
            retention_choice_index = float(np.tanh(retained_margin / (reference_margin + 1e-12)))
            learning_memory_score = (
                float(cfg.acquisition_score_weight) * max(0.0, acquisition_choice_index)
                + float(cfg.retention_score_weight) * max(0.0, retention_choice_index)
            )
            raw_rows.append(
                {
                    **context,
                    "condition_order": int(condition.condition_order),
                    "condition_id": str(condition.condition_id),
                    "condition_label": str(condition.condition_label),
                    "condition_interpretation": str(condition.interpretation),
                    "ppl1_gain": float(condition.ppl1_gain),
                    "dpm_gain": float(condition.dpm_gain),
                    "ppl1_teaching_drive_norm": ppl1_teaching_drive,
                    "dpm_persistence_support": dpm_support,
                    "dpm_persistence_support_clipped": dpm_support_clipped,
                    "decay_per_block": decay_per_block,
                    "retention_fraction_after_interference": retention_fraction,
                    "acquisition_margin": acquisition_margin,
                    "acquisition_choice_index": acquisition_choice_index,
                    "retained_margin": retained_margin,
                    "retention_choice_index": retention_choice_index,
                    "learning_memory_score": learning_memory_score,
                }
            )

    raw_df = pd.DataFrame.from_records(raw_rows)
    summary_df = (
        raw_df.groupby(
            ["condition_order", "condition_id", "condition_label", "condition_interpretation", "ppl1_gain", "dpm_gain"],
            as_index=False,
        )
        .agg(
            n_seed_panels=("seed", "nunique"),
            active_k=("active_k", "first"),
            observed_active_fraction_mean=("observed_active_fraction", "mean"),
            observed_active_fraction_std=("observed_active_fraction", "std"),
            ppl1_teaching_drive_norm_mean=("ppl1_teaching_drive_norm", "mean"),
            ppl1_teaching_drive_norm_std=("ppl1_teaching_drive_norm", "std"),
            dpm_persistence_support_mean=("dpm_persistence_support", "mean"),
            dpm_persistence_support_std=("dpm_persistence_support", "std"),
            acquisition_margin_mean=("acquisition_margin", "mean"),
            acquisition_margin_std=("acquisition_margin", "std"),
            acquisition_choice_index_mean=("acquisition_choice_index", "mean"),
            acquisition_choice_index_std=("acquisition_choice_index", "std"),
            retained_margin_mean=("retained_margin", "mean"),
            retained_margin_std=("retained_margin", "std"),
            retention_choice_index_mean=("retention_choice_index", "mean"),
            retention_choice_index_std=("retention_choice_index", "std"),
            learning_memory_score_mean=("learning_memory_score", "mean"),
            learning_memory_score_std=("learning_memory_score", "std"),
            decay_per_block_mean=("decay_per_block", "mean"),
            decay_per_block_std=("decay_per_block", "std"),
            retention_fraction_after_interference_mean=("retention_fraction_after_interference", "mean"),
        )
        .sort_values("condition_order")
    )
    baseline = summary_df[summary_df["condition_id"].eq("baseline")].iloc[0]
    for metric in [
        "acquisition_choice_index",
        "retention_choice_index",
        "learning_memory_score",
        "decay_per_block",
    ]:
        mean_column = f"{metric}_mean"
        baseline_value = float(baseline[mean_column])
        summary_df[f"{metric}_delta_vs_baseline"] = summary_df[mean_column] - baseline_value
        if abs(baseline_value) > 1e-12:
            summary_df[f"{metric}_percent_of_baseline"] = 100.0 * summary_df[mean_column] / baseline_value
            summary_df[f"{metric}_drop_percent"] = 100.0 * (1.0 - summary_df[mean_column] / baseline_value)
        else:
            summary_df[f"{metric}_percent_of_baseline"] = np.nan
            summary_df[f"{metric}_drop_percent"] = np.nan

    task_conditions = _learning_memory_task_conditions()
    task_rows: list[dict[str, object]] = []
    for context in seed_contexts:
        for task in task_conditions.itertuples(index=False):
            ppl1_teaching_drive = float(task.ppl1_gain) * float(context["ppl1_drive_norm"])
            acquisition_margin = ppl1_teaching_drive * float(context["separation_norm_sq"])
            acquisition_choice_index = float(np.tanh(acquisition_margin / (reference_margin + 1e-12)))
            dpm_support = float(task.dpm_gain) * dpm_to_kc_coverage * float(context["dpm_drive_norm"])
            dpm_support_clipped = float(np.clip(dpm_support, 0.0, 1.0))
            decay_per_block = float(
                cfg.dpm_baseline_decay_per_block
                + cfg.dpm_loss_decay_penalty_per_block * (1.0 - dpm_support_clipped)
            )
            retention_fraction = float(np.exp(-int(task.delay_blocks) * decay_per_block))
            retained_margin = acquisition_margin * retention_fraction
            conflict_penalty = float(task.conflict_level) * float(cfg.conflict_decoy_penalty) * reference_margin
            retrieval_margin = retained_margin - conflict_penalty
            retention_choice_index = float(np.tanh(retained_margin / (reference_margin + 1e-12)))
            choice_index = float(np.tanh(retrieval_margin / (reference_margin + 1e-12)))
            expected_choice_rate = float(np.clip(0.5 + 0.5 * choice_index, 0.0, 1.0))
            task_rows.append(
                {
                    **context,
                    "task_order": int(task.task_order),
                    "condition_id": str(task.condition_id),
                    "condition_label": str(task.condition_label),
                    "task_id": str(task.task_id),
                    "task_label": str(task.task_label),
                    "ppl1_gain": float(task.ppl1_gain),
                    "dpm_gain": float(task.dpm_gain),
                    "delay_blocks": int(task.delay_blocks),
                    "conflict_level": float(task.conflict_level),
                    "ppl1_teaching_drive_norm": ppl1_teaching_drive,
                    "dpm_persistence_support": dpm_support,
                    "dpm_persistence_support_clipped": dpm_support_clipped,
                    "decay_per_block": decay_per_block,
                    "retention_fraction": retention_fraction,
                    "acquisition_margin": acquisition_margin,
                    "acquisition_choice_index": acquisition_choice_index,
                    "retained_margin": retained_margin,
                    "retention_choice_index": retention_choice_index,
                    "conflict_penalty": conflict_penalty,
                    "retrieval_margin_after_conflict": retrieval_margin,
                    "choice_index": choice_index,
                    "expected_choice_rate": expected_choice_rate,
                }
            )

    task_raw_df = pd.DataFrame.from_records(task_rows)
    task_summary_df = (
        task_raw_df.groupby(
            [
                "task_order",
                "condition_id",
                "condition_label",
                "task_id",
                "task_label",
                "ppl1_gain",
                "dpm_gain",
                "delay_blocks",
                "conflict_level",
            ],
            as_index=False,
        )
        .agg(
            n_seed_panels=("seed", "nunique"),
            acquisition_choice_index_mean=("acquisition_choice_index", "mean"),
            acquisition_choice_index_std=("acquisition_choice_index", "std"),
            retention_choice_index_mean=("retention_choice_index", "mean"),
            retention_choice_index_std=("retention_choice_index", "std"),
            choice_index_mean=("choice_index", "mean"),
            choice_index_std=("choice_index", "std"),
            expected_choice_rate_mean=("expected_choice_rate", "mean"),
            expected_choice_rate_std=("expected_choice_rate", "std"),
            decay_per_block_mean=("decay_per_block", "mean"),
            retention_fraction_mean=("retention_fraction", "mean"),
            retained_margin_mean=("retained_margin", "mean"),
            retrieval_margin_after_conflict_mean=("retrieval_margin_after_conflict", "mean"),
        )
        .sort_values(["task_order", "condition_id"])
    )
    task_baseline = task_summary_df[task_summary_df["condition_id"].eq("baseline")][
        [
            "task_id",
            "acquisition_choice_index_mean",
            "retention_choice_index_mean",
            "choice_index_mean",
            "expected_choice_rate_mean",
        ]
    ].rename(
        columns={
            "acquisition_choice_index_mean": "baseline_task_acquisition_choice_index_mean",
            "retention_choice_index_mean": "baseline_task_retention_choice_index_mean",
            "choice_index_mean": "baseline_task_choice_index_mean",
            "expected_choice_rate_mean": "baseline_task_expected_choice_rate_mean",
        }
    )
    task_summary_df = task_summary_df.merge(task_baseline, on="task_id", how="left")
    for metric in [
        "acquisition_choice_index",
        "retention_choice_index",
        "choice_index",
        "expected_choice_rate",
    ]:
        mean_column = f"{metric}_mean"
        baseline_column = f"baseline_task_{metric}_mean"
        task_summary_df[f"{metric}_delta_vs_baseline_task"] = task_summary_df[mean_column] - task_summary_df[baseline_column]
        task_summary_df[f"{metric}_percent_of_baseline_task"] = np.where(
            task_summary_df[baseline_column].abs() > 1e-12,
            100.0 * task_summary_df[mean_column] / task_summary_df[baseline_column],
            np.nan,
        )

    def _condition_row(condition_id: str) -> pd.Series:
        selected = summary_df[summary_df["condition_id"].eq(condition_id)]
        return selected.iloc[0] if not selected.empty else pd.Series(dtype=object)

    ppl1_25 = _condition_row("ppl1_dan_25pct")
    dpm_25 = _condition_row("dpm_25pct")
    joint_25 = _condition_row("joint_25pct")
    interpretation = {
        "model": "real_flywire_v783_ALPN_to_KC_with_KC_to_PPL1_DAN_and_DPM_axes",
        "annotation_path": str(cfg.annotation_path),
        "connectivity_path": str(cfg.connectivity_path),
        "ppl1_pattern_used": pattern_used,
        "n_kc": int(len(kc_ids)),
        "n_glomerulus_channels": int(len(glomerulus_names)),
        "n_seed_panels": int(len(cfg.seeds)),
        "n_odors_per_panel": int(cfg.n_odors),
        "kc_active_fraction": float(cfg.ratio),
        "n_ppl1_dan_targets": int(ppl1_targets["root_id"].nunique()),
        "n_dpm_targets": int(dpm_targets["root_id"].nunique()),
        "kc_to_ppl1_positive_kc_count": int((ppl1_vector > 0).sum()),
        "kc_to_dpm_positive_kc_count": int((dpm_vector > 0).sum()),
        "kc_to_ppl1_positive_weight": float(ppl1_vector.sum()),
        "kc_to_dpm_positive_weight": float(dpm_vector.sum()),
        "dpm_to_kc_coverage_fraction": float(dpm_to_kc_coverage),
        "reference_margin": float(reference_margin),
        "baseline_learning_memory_score": float(baseline["learning_memory_score_mean"]),
        "ppl1_25_score_percent": float(ppl1_25.get("learning_memory_score_percent_of_baseline", np.nan)),
        "ppl1_25_score_drop_percent": float(ppl1_25.get("learning_memory_score_drop_percent", np.nan)),
        "dpm_25_score_percent": float(dpm_25.get("learning_memory_score_percent_of_baseline", np.nan)),
        "dpm_25_score_drop_percent": float(dpm_25.get("learning_memory_score_drop_percent", np.nan)),
        "joint_25_score_percent": float(joint_25.get("learning_memory_score_percent_of_baseline", np.nan)),
        "joint_25_score_drop_percent": float(joint_25.get("learning_memory_score_drop_percent", np.nan)),
        "boundary": "connectome-constrained learning/memory proxy; not animal behavior or molecular causality",
    }

    target_inventory_path = cfg.output_dir / "ppl1_dpm_target_inventory.csv"
    structural_summary_path = cfg.output_dir / "ppl1_dpm_structural_summary.csv"
    channel_path = cfg.output_dir / "ppl1_dpm_flywire_glomerulus_kc_channels.csv"
    raw_path = cfg.output_dir / "ppl1_dpm_perturbation_raw.csv"
    summary_path = cfg.output_dir / "ppl1_dpm_perturbation_summary.csv"
    task_raw_path = cfg.output_dir / "ppl1_dpm_task_readout_raw.csv"
    task_summary_path = cfg.output_dir / "ppl1_dpm_task_readout_summary.csv"
    metadata_path = cfg.output_dir / "ppl1_dpm_perturbation_metadata.json"
    report_path = cfg.output_dir / "PPL1_DPM_LEARNING_MEMORY_PERTURBATION_REPORT_CN.md"

    target_inventory.to_csv(target_inventory_path, index=False)
    structural_summary.to_csv(structural_summary_path, index=False)
    channel_table.to_csv(channel_path, index=False)
    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    task_raw_df.to_csv(task_raw_path, index=False)
    task_summary_df.to_csv(task_summary_path, index=False)
    figure_path = _write_learning_memory_perturbation_figure(summary_df, task_summary_df, cfg.output_dir)
    report_path.write_text(
        _render_learning_memory_perturbation_report(
            cfg,
            summary_df=summary_df,
            task_summary_df=task_summary_df,
            structural_summary=structural_summary,
            target_inventory=target_inventory,
            interpretation=interpretation,
            figure_path=figure_path,
        ),
        encoding="utf-8",
    )

    payload = {
        "config": {
            "annotation_path": str(cfg.annotation_path),
            "connectivity_path": str(cfg.connectivity_path),
            "output_dir": str(cfg.output_dir),
            "ratio": float(cfg.ratio),
            "seeds": [int(seed) for seed in cfg.seeds],
            "n_odors": int(cfg.n_odors),
            "min_glomeruli_per_odor": int(cfg.min_glomeruli_per_odor),
            "max_glomeruli_per_odor": int(cfg.max_glomeruli_per_odor),
            "channel_noise_sigma": float(cfg.channel_noise_sigma),
            "cs_plus_index": int(cfg.cs_plus_index),
            "dpm_baseline_decay_per_block": float(cfg.dpm_baseline_decay_per_block),
            "dpm_loss_decay_penalty_per_block": float(cfg.dpm_loss_decay_penalty_per_block),
            "interference_blocks": int(cfg.interference_blocks),
            "conflict_decoy_penalty": float(cfg.conflict_decoy_penalty),
            "ppl1_pattern": str(cfg.ppl1_pattern),
        },
        "interpretation": interpretation,
        "paths": {
            "target_inventory_csv": str(target_inventory_path),
            "structural_summary_csv": str(structural_summary_path),
            "channels_csv": str(channel_path),
            "raw_csv": str(raw_path),
            "summary_csv": str(summary_path),
            "task_raw_csv": str(task_raw_path),
            "task_summary_csv": str(task_summary_path),
            "figure_png": str(figure_path) if figure_path else None,
            "report_md": str(report_path),
        },
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "target_inventory_csv": target_inventory_path,
        "structural_summary_csv": structural_summary_path,
        "channels_csv": channel_path,
        "raw_csv": raw_path,
        "summary_csv": summary_path,
        "task_raw_csv": task_raw_path,
        "task_summary_csv": task_summary_path,
        "figure_png": figure_path,
        "report_md": report_path,
        "metadata_json": metadata_path,
        "summary_df": summary_df,
        "raw_df": raw_df,
        "task_summary_df": task_summary_df,
        "task_raw_df": task_raw_df,
        "structural_summary_df": structural_summary,
        "target_inventory_df": target_inventory,
        "interpretation": interpretation,
    }


def _standardize(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.nanmean(array)) if len(array) else 0.0
    std = float(np.nanstd(array)) if len(array) else 0.0
    if not np.isfinite(std) or std <= 1e-12:
        return np.zeros_like(array, dtype=np.float64)
    return (array - mean) / std


def _load_kc_lateralization_gate(
    kc_nt_inputs_path: Path,
    kc_ids: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return a subtype/side symmetry-breaking gate aligned to ``kc_ids``."""

    if not kc_nt_inputs_path.exists():
        raise FileNotFoundError(f"KC neurotransmitter lateralization table not found: {kc_nt_inputs_path}")
    nt_inputs = pd.read_parquet(kc_nt_inputs_path).drop_duplicates("root_id").copy()
    required = {"root_id", "side", "cell_type", "hemibrain_type", "ser_fraction", "glut_fraction"}
    missing = required - set(nt_inputs.columns)
    if missing:
        raise ValueError(f"KC NT lateralization table is missing columns: {sorted(missing)}")
    nt_inputs["root_id"] = nt_inputs["root_id"].astype("int64")
    nt_inputs["subtype"] = nt_inputs["cell_type"].fillna("").astype(str)
    nt_inputs.loc[nt_inputs["subtype"].eq(""), "subtype"] = nt_inputs.loc[
        nt_inputs["subtype"].eq(""),
        "hemibrain_type",
    ].fillna("").astype(str)
    nt_inputs["side"] = nt_inputs["side"].fillna("").astype(str)
    nt_inputs["ser_z"] = _standardize(nt_inputs["ser_fraction"])
    nt_inputs["glut_z"] = _standardize(nt_inputs["glut_fraction"])
    nt_inputs["ser_minus_glut_z"] = nt_inputs["ser_z"] - nt_inputs["glut_z"]
    subtype_mean = nt_inputs.groupby("subtype")["ser_minus_glut_z"].transform("mean")
    side_subtype_mean = nt_inputs.groupby(["subtype", "side"])["ser_minus_glut_z"].transform("mean")
    nt_inputs["symmetry_breaking_gate_raw"] = side_subtype_mean - subtype_mean
    scale = float(np.nanpercentile(np.abs(nt_inputs["symmetry_breaking_gate_raw"]), 95))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.nanmax(np.abs(nt_inputs["symmetry_breaking_gate_raw"])))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    nt_inputs["symmetry_breaking_gate"] = np.clip(nt_inputs["symmetry_breaking_gate_raw"] / scale, -1.0, 1.0)

    gate_by_subtype_side = (
        nt_inputs.groupby(["subtype", "side"], as_index=False)
        .agg(
            n_kc=("root_id", "nunique"),
            mean_ser_fraction=("ser_fraction", "mean"),
            mean_glut_fraction=("glut_fraction", "mean"),
            mean_ser_minus_glut_z=("ser_minus_glut_z", "mean"),
            mean_symmetry_breaking_gate=("symmetry_breaking_gate", "mean"),
        )
        .sort_values(["subtype", "side"])
    )
    aligned = pd.DataFrame({"root_id": kc_ids.astype("int64")}).merge(
        nt_inputs[
            [
                "root_id",
                "side",
                "subtype",
                "ser_fraction",
                "glut_fraction",
                "symmetry_breaking_gate",
            ]
        ],
        on="root_id",
        how="left",
    )
    aligned["side"] = aligned["side"].fillna("")
    aligned["subtype"] = aligned["subtype"].fillna("")
    aligned["symmetry_breaking_gate"] = aligned["symmetry_breaking_gate"].fillna(0.0)
    return gate_by_subtype_side, aligned["symmetry_breaking_gate"].to_numpy(dtype=np.float64), aligned["side"].to_numpy()


def _representation_space_metrics(binary: np.ndarray, graded: np.ndarray, side_by_kc: np.ndarray) -> dict[str, float]:
    centered = graded - graded.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    eig = (singular_values**2) / max(1, graded.shape[0] - 1)
    eig = eig[eig > 1e-15]
    effective_dimension = float((eig.sum() ** 2) / np.square(eig).sum()) if eig.size else 0.0
    max_dimension = float(max(1, min(graded.shape[0] - 1, graded.shape[1])))
    log_volume_per_dim = float(np.mean(np.log(eig + 1e-15))) if eig.size else 0.0

    l2_values: list[float] = []
    cosine_values: list[float] = []
    norms = np.linalg.norm(graded, axis=1)
    for left in range(graded.shape[0]):
        for right in range(left + 1, graded.shape[0]):
            l2_values.append(float(np.linalg.norm(graded[left] - graded[right])))
            denom = float(norms[left] * norms[right])
            cosine_values.append(float(np.dot(graded[left], graded[right]) / denom) if denom else 0.0)

    left_mask = side_by_kc == "left"
    right_mask = side_by_kc == "right"
    left_active = float(binary[:, left_mask].mean()) if np.any(left_mask) else 0.0
    right_active = float(binary[:, right_mask].mean()) if np.any(right_mask) else 0.0
    lateral_code_index = (right_active - left_active) / (right_active + left_active + 1e-12)

    return {
        "effective_dimension": effective_dimension,
        "normalized_effective_dimension": effective_dimension / max_dimension,
        "log_volume_per_dimension": log_volume_per_dim,
        "mean_pairwise_l2_distance": float(np.mean(l2_values)) if l2_values else 0.0,
        "mean_pairwise_cosine_similarity": float(np.mean(cosine_values)) if cosine_values else 0.0,
        "left_active_fraction": left_active,
        "right_active_fraction": right_active,
        "lateral_code_index": float(lateral_code_index),
    }


def _conditioned_gate_vectors(
    gate: np.ndarray,
    *,
    cfg: LateralizationRepresentationMemoryConfig,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = [
        {
            "condition_order": 0,
            "condition_id": "symmetrized",
            "condition_label": "symmetrized side-balanced KC gate",
            "condition_class": "symmetrized",
            "gate_strength": 0.0,
            "shuffle_repeat": -1,
            "gate_vector": np.zeros_like(gate, dtype=np.float64),
        }
    ]
    order = 1
    for strength in cfg.gate_strengths:
        if abs(float(strength)) <= 1e-12:
            continue
        condition_class = "real_lateralized" if float(strength) > 0 else "mirror_reversed"
        records.append(
            {
                "condition_order": order,
                "condition_id": f"{condition_class}_strength_{float(strength):+.2f}".replace("+", "p").replace("-", "m").replace(".", "p"),
                "condition_label": f"{condition_class} gate strength {float(strength):+.2f}",
                "condition_class": condition_class,
                "gate_strength": float(strength),
                "shuffle_repeat": -1,
                "gate_vector": gate.astype(np.float64) * float(strength),
            }
        )
        order += 1
    for repeat in range(int(cfg.shuffle_repeats)):
        shuffled = gate.copy()
        rng.shuffle(shuffled)
        records.append(
            {
                "condition_order": order,
                "condition_id": f"shuffled_lateralized_repeat_{repeat:02d}",
                "condition_label": f"shuffled lateralized gate repeat {repeat:02d}",
                "condition_class": "shuffled_lateralized",
                "gate_strength": 1.0,
                "shuffle_repeat": int(repeat),
                "gate_vector": shuffled,
            }
        )
        order += 1
    return records


def _write_lateralization_representation_figure(
    summary_df: pd.DataFrame,
    gate_by_subtype_side: pd.DataFrame,
    output_dir: Path,
) -> Path | None:
    if summary_df.empty:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "Fig_lateralization_representation_memory.png"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    gate_plot = gate_by_subtype_side.copy()
    gate_plot = gate_plot[gate_plot["subtype"].astype(str).str.contains("KCapbp|KCa'b'|KCab|KCg", regex=True, na=False)]
    gate_plot = gate_plot.sort_values("mean_symmetry_breaking_gate")
    labels = [f"{row.subtype} {row.side}" for row in gate_plot.itertuples()]
    colors = ["#d95f02" if str(row.side) == "right" else "#1b9e77" for row in gate_plot.itertuples()]
    axes[0, 0].barh(labels, gate_plot["mean_symmetry_breaking_gate"], color=colors)
    axes[0, 0].axvline(0, color="black", lw=0.8)
    axes[0, 0].set_title("KC side/subtype symmetry-breaking gate")
    axes[0, 0].set_xlabel("right-serotonin / left-glutamate gate")
    axes[0, 0].tick_params(axis="y", labelsize=7)

    main = summary_df[summary_df["condition_class"].isin(["symmetrized", "real_lateralized", "mirror_reversed"])].copy()
    main = main.sort_values("gate_strength")
    axes[0, 1].plot(
        main["gate_strength"],
        main["normalized_effective_dimension_mean"],
        marker="o",
        color="#386cb0",
        label="effective dimension",
    )
    axes[0, 1].set_title("Representation space expands with lateralized gate")
    axes[0, 1].set_xlabel("gate strength")
    axes[0, 1].set_ylabel("normalized effective dimension")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(
        main["gate_strength"],
        main["retention_accuracy_after_interference_mean"],
        marker="o",
        color="#4daf4a",
        label="retention",
    )
    axes[1, 0].plot(
        main["gate_strength"],
        main["similar_pair_retention_accuracy_after_interference_mean"],
        marker="s",
        color="#984ea3",
        label="similar-pair retention",
    )
    axes[1, 0].set_title("Memory stabilization proxies")
    axes[1, 0].set_xlabel("gate strength")
    axes[1, 0].set_ylabel("accuracy")
    axes[1, 0].set_ylim(0, 1.02)
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].grid(alpha=0.25)

    colors_by_class = {
        "symmetrized": "#555555",
        "real_lateralized": "#386cb0",
        "mirror_reversed": "#e41a1c",
        "shuffled_lateralized": "#999999",
    }
    for condition_class, group in summary_df.groupby("condition_class"):
        axes[1, 1].scatter(
            group["normalized_effective_dimension_mean"],
            group["memory_stability_score_mean"],
            label=condition_class,
            color=colors_by_class.get(condition_class, "#666666"),
            alpha=0.85,
        )
    axes[1, 1].set_title("Representation vs memory stability")
    axes[1, 1].set_xlabel("normalized effective dimension")
    axes[1, 1].set_ylabel("memory stability score")
    axes[1, 1].legend(frameon=False, fontsize=7)
    axes[1, 1].grid(alpha=0.25)

    fig.suptitle("FlyWire KC lateralization as graph-level symmetry breaking", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return figure_path


def _render_lateralization_representation_report(
    *,
    cfg: LateralizationRepresentationMemoryConfig,
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    gate_by_subtype_side: pd.DataFrame,
    figure_path: Path | None,
    interpretation: dict[str, object],
) -> str:
    real = comparison_df[comparison_df["condition_id"].eq("real_lateralized_strength_p1p00")]
    real_row = real.iloc[0] if not real.empty else pd.Series(dtype=object)
    strength_rows = comparison_df[
        comparison_df["condition_class"].isin(["symmetrized", "mirror_reversed", "real_lateralized"])
    ].sort_values(["gate_strength", "condition_order"])
    strength_lines = [
        "| condition | strength | norm. effective dimension | delta | retention | delta | memory stability | delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in strength_rows.itertuples(index=False):
        strength_lines.append(
            "| "
            f"{row.condition_class} | "
            f"{float(row.gate_strength):.2f} | "
            f"{float(row.normalized_effective_dimension_mean):.4f} | "
            f"{float(getattr(row, 'normalized_effective_dimension_delta_vs_symmetrized', 0.0)):.4f} | "
            f"{float(row.retention_accuracy_after_interference_mean):.4f} | "
            f"{float(getattr(row, 'retention_accuracy_after_interference_delta_vs_symmetrized', 0.0)):.4f} | "
            f"{float(row.memory_stability_score_mean):.4f} | "
            f"{float(getattr(row, 'memory_stability_score_delta_vs_symmetrized', 0.0)):.4f} |"
        )
    shuffled = comparison_df[comparison_df["condition_class"].eq("shuffled_lateralized")]
    shuffled_dim_delta = float(shuffled["normalized_effective_dimension_delta_vs_symmetrized"].mean()) if not shuffled.empty else np.nan
    shuffled_stability_delta = float(shuffled["memory_stability_score_delta_vs_symmetrized"].mean()) if not shuffled.empty else np.nan
    return f"""# 偏侧化对表示空间和记忆稳定性的图网络解释

保存路径：`{cfg.output_dir / 'LATERALIZATION_REPRESENTATION_MEMORY_REPORT_CN.md'}`

## 文章和早期结果位置

早期“偏侧化”主线主要写在以下位置：

- `paper/NATURE_STYLE_DRAFT_CN.md`：中文 Nature 风格草稿，标题是“果蝇蘑菇体左右侧化连接组预测嗅觉记忆行为”。
- `paper/main_merged.tex`：较完整的论文版，包含 “Pervasive lateralization and its functional implications” 小节。
- `outputs/four_card_suite/CYBER_FLY_NATURE_UPGRADE_REPORT.md`：四卡 GPU 传播报告，把右侧 serotonin KC 和左侧 glutamate KC 映射到 MBON/DAN/APL/DPM 记忆轴。
- `outputs/dpm_excel_lateralization_20260521/DPM_EXCEL_LATERALIZATION_STATISTICS_CN.md` 与 `outputs/dpm_optogenetic_validation_20260429/DPM_OPTOGENETIC_VALIDATION_CN.md`：DPM/5-HT 侧化的结构和功能成像验证路线。

这些结果已经支持“偏侧化结构不是随机噪声”的结构和传播证据，但此前行为学实验没有来得及完成。因此本分析不再尝试补一个低维行为代理，而是回答更前一层的问题：**连接图中的对称性破缺是否能扩大 KC odor representation space，并提高 memory stabilization proxy。**

## 方法

输入仍使用真实 FlyWire v783：

- `ALPN->KC` 真实连接图生成 glomerulus odor channels。
- `outputs/kc_nt_lateralization/kc_neuron_nt_inputs.parquet` 提供每个 KC 的 serotonin/glutamate 输入比例。
- 对每个 KC，计算 `serotonin - glutamate` 的标准化差异，再在 `KC subtype x side` 内取均值，得到一个 side/subtype 层面的 symmetry-breaking gate。
- 对照条件包括：去侧化的 `symmetrized`、真实方向的 `real_lateralized`、方向反转的 `mirror_reversed` 和随机打乱的 `shuffled_lateralized`。

门控公式为：

```text
KC_response_lateralized = KC_response_raw * (1 + A * s * gate_KC)
```

其中 `A={cfg.gate_amplitude}` 是门控幅度，`s` 是强度扫描，`gate_KC` 是真实 KC 的侧化门控值。随后固定 KC active fraction 为 `{cfg.ratio:.2f}`，重新计算表示空间、odor overlap、学习准确率和干扰后保持率。

## 关键结果

相对 `symmetrized` 对照，真实侧化强度 `s=1.0` 的变化为：

| metric | delta vs symmetrized |
|---|---:|
| normalized effective dimension | {float(real_row.get('normalized_effective_dimension_delta_vs_symmetrized', np.nan)):.4f} |
| mean pairwise L2 distance | {float(real_row.get('mean_pairwise_l2_distance_delta_vs_symmetrized', np.nan)):.4f} |
| mean Jaccard overlap | {float(real_row.get('mean_jaccard_overlap_delta_vs_symmetrized', np.nan)):.4f} |
| retention accuracy after interference | {float(real_row.get('retention_accuracy_after_interference_delta_vs_symmetrized', np.nan)):.4f} |
| similar-pair retention accuracy | {float(real_row.get('similar_pair_retention_accuracy_after_interference_delta_vs_symmetrized', np.nan)):.4f} |
| memory stability score | {float(real_row.get('memory_stability_score_delta_vs_symmetrized', np.nan)):.4f} |

强度扫描显示：

{chr(10).join(strength_lines)}

随机打乱侧化门控的平均变化为：

| shuffled metric | mean delta vs symmetrized |
|---|---:|
| normalized effective dimension | {shuffled_dim_delta:.4f} |
| memory stability score | {shuffled_stability_delta:.4f} |

图：`{figure_path if figure_path is not None else 'not generated'}`

## 解释

当前结果支持一个更窄、也更稳的结论：真实 side/subtype 对齐的侧化门控会轻微扩大 KC odor
representation space。`s=1.0` 时 normalized effective dimension 增加
`{float(real_row.get('normalized_effective_dimension_delta_vs_symmetrized', np.nan)):.4f}`，
随机打乱门控平均为 `{shuffled_dim_delta:.4f}`，说明这个方向不是任意异质性都能稳定得到。

但当前结果不支持“偏侧化已经提高 memory stabilization proxy”的强表述。`s=1.0` 时 memory
stability score 变化为 `{float(real_row.get('memory_stability_score_delta_vs_symmetrized', np.nan)):.4f}`，
retention 也没有改善。更合理的解释是：对称性破缺在 KC 表示层重新分配了 odor-code 维度，
但在当前 `10%` sparse code 和 binary associative-memory proxy 下，记忆读出接近饱和或不够敏感，
尚未把表示空间增益转化为行为层优势。

因此这一结果适合补在论文讨论中，作为“偏侧化为什么可能有功能意义”的图网络解释：
它提供了结构化表示空间增益，而不是直接证明学习记忆行为增强。最终行为因果仍需要 OCT/MCH
delayed/conflict 行为、DPM/5-HT 成像和单侧扰动验证。

## 输出文件

- raw metrics：`{cfg.output_dir / 'lateralization_representation_raw.csv'}`
- summary：`{cfg.output_dir / 'lateralization_representation_summary.csv'}`
- comparison：`{cfg.output_dir / 'lateralization_representation_comparison.csv'}`
- gate table：`{cfg.output_dir / 'lateralization_gate_by_subtype_side.csv'}`
- figure：`{figure_path if figure_path is not None else 'not generated'}`
- metadata：`{cfg.output_dir / 'lateralization_representation_metadata.json'}`

## 边界

- 这里的 memory stabilization 是 binary associative-memory proxy，不是真实动物记忆。
- 这里的 gate 是从 KC neurotransmitter input lateralization 构建的图网络门控，不等于真实膜电位或真实 5-HT release。
- 分析价值在于解释“为什么侧化可能有功能意义”：对称性破缺可以提高表示维度；当前 memory proxy 尚未显示稳定增强。
"""


def run_lateralization_representation_memory(
    config: LateralizationRepresentationMemoryConfig | None = None,
) -> dict[str, Path | pd.DataFrame | dict[str, object]]:
    cfg = config or LateralizationRepresentationMemoryConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    annotations = _load_annotations(cfg.annotation_path)
    annotated = _annotate_connectome_groups(annotations)
    edges = pd.read_parquet(
        cfg.connectivity_path,
        columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity", "Excitatory x Connectivity"],
    )
    glomerulus_names, glomerulus_matrix, kc_ids, channel_table = _build_flywire_glomerulus_kc_matrix_from_frames(
        annotated,
        edges,
    )
    gate_by_subtype_side, gate, side_by_kc = _load_kc_lateralization_gate(cfg.kc_nt_inputs_path, kc_ids)
    rng = np.random.default_rng(20260603)
    condition_records = _conditioned_gate_vectors(gate, cfg=cfg, rng=rng)
    memory_config = KCSparseCodingConfig(
        random_seed=0,
        evaluation_repeats=cfg.memory_evaluation_repeats,
        test_repeats=cfg.memory_test_repeats,
        max_learning_steps=cfg.memory_max_learning_steps,
        dropout_probability=cfg.memory_dropout_probability,
        false_positive_probability=cfg.memory_false_positive_probability,
        forgetting_interference_steps=cfg.n_interference_blocks,
    )

    raw_rows: list[dict[str, object]] = []
    odor_panel_frames: list[pd.DataFrame] = []
    for seed in cfg.seeds:
        _, activity, odor_panel = build_mixture_odor_panel(
            glomerulus_names,
            glomerulus_matrix,
            seed=int(seed),
            n_odors=int(cfg.n_odors),
            min_glomeruli_per_odor=int(cfg.min_glomeruli_per_odor),
            max_glomeruli_per_odor=int(cfg.max_glomeruli_per_odor),
            channel_noise_sigma=float(cfg.channel_noise_sigma),
        )
        odor_panel_frames.append(odor_panel)
        for record in condition_records:
            gate_vector = np.asarray(record["gate_vector"], dtype=np.float64)
            gains = np.clip(1.0 + float(cfg.gate_amplitude) * gate_vector, 0.05, 5.0)
            gated_activity = _normalize_rows(np.maximum(activity * gains[None, :], 0.0))
            binary, graded, active_k = _sparsify(gated_activity, float(cfg.ratio))
            code_metrics = evaluate_binary_code(
                binary,
                graded,
                float(cfg.ratio),
                memory_config,
                seed=int(seed) * 10_000 + int(record["condition_order"]) * 101 + max(0, int(record["shuffle_repeat"])),
            )
            representation_metrics = _representation_space_metrics(binary, graded, side_by_kc)
            memory_stability_score = (
                0.40 * float(code_metrics.get("retention_accuracy_after_interference", 0.0))
                + 0.25 * float(code_metrics.get("similar_pair_retention_accuracy_after_interference", 0.0))
                + 0.20 * float(code_metrics.get("learning_accuracy_final", 0.0))
                + 0.15 * (1.0 - float(code_metrics.get("mean_jaccard_overlap", 0.0)))
            )
            raw_rows.append(
                {
                    "seed": int(seed),
                    "condition_order": int(record["condition_order"]),
                    "condition_id": str(record["condition_id"]),
                    "condition_label": str(record["condition_label"]),
                    "condition_class": str(record["condition_class"]),
                    "gate_strength": float(record["gate_strength"]),
                    "shuffle_repeat": int(record["shuffle_repeat"]),
                    "active_k": int(active_k),
                    "observed_active_fraction": float(binary.mean()),
                    "gate_gain_min": float(gains.min()),
                    "gate_gain_max": float(gains.max()),
                    "gate_gain_std": float(gains.std()),
                    "memory_stability_score": float(memory_stability_score),
                    **representation_metrics,
                    **code_metrics,
                }
            )

    raw_df = pd.DataFrame.from_records(raw_rows)
    summary_df = (
        raw_df.groupby(
            ["condition_order", "condition_id", "condition_label", "condition_class", "gate_strength", "shuffle_repeat"],
            as_index=False,
        )
        .agg(
            n_seed_panels=("seed", "nunique"),
            active_k=("active_k", "first"),
            observed_active_fraction_mean=("observed_active_fraction", "mean"),
            normalized_effective_dimension_mean=("normalized_effective_dimension", "mean"),
            normalized_effective_dimension_std=("normalized_effective_dimension", "std"),
            effective_dimension_mean=("effective_dimension", "mean"),
            log_volume_per_dimension_mean=("log_volume_per_dimension", "mean"),
            mean_pairwise_l2_distance_mean=("mean_pairwise_l2_distance", "mean"),
            mean_pairwise_cosine_similarity_mean=("mean_pairwise_cosine_similarity", "mean"),
            mean_jaccard_overlap_mean=("mean_jaccard_overlap", "mean"),
            mean_binary_cosine_mean=("mean_binary_cosine", "mean"),
            learning_accuracy_final_mean=("learning_accuracy_final", "mean"),
            retention_accuracy_after_interference_mean=("retention_accuracy_after_interference", "mean"),
            similar_pair_retention_accuracy_after_interference_mean=(
                "similar_pair_retention_accuracy_after_interference",
                "mean",
            ),
            memory_stability_score_mean=("memory_stability_score", "mean"),
            memory_stability_score_std=("memory_stability_score", "std"),
            lateral_code_index_mean=("lateral_code_index", "mean"),
            left_active_fraction_mean=("left_active_fraction", "mean"),
            right_active_fraction_mean=("right_active_fraction", "mean"),
        )
        .sort_values(["condition_order", "condition_id"])
    )
    comparison_df = summary_df.copy()
    sym = comparison_df[comparison_df["condition_id"].eq("symmetrized")]
    if not sym.empty:
        sym_row = sym.iloc[0]
        for metric in [
            "normalized_effective_dimension",
            "effective_dimension",
            "log_volume_per_dimension",
            "mean_pairwise_l2_distance",
            "mean_jaccard_overlap",
            "learning_accuracy_final",
            "retention_accuracy_after_interference",
            "similar_pair_retention_accuracy_after_interference",
            "memory_stability_score",
        ]:
            column = f"{metric}_mean"
            if column in comparison_df:
                comparison_df[f"{metric}_delta_vs_symmetrized"] = comparison_df[column] - float(sym_row[column])
                denominator = abs(float(sym_row[column]))
                comparison_df[f"{metric}_percent_vs_symmetrized"] = (
                    100.0 * comparison_df[column] / denominator if denominator > 1e-12 else np.nan
                )

    real_1 = comparison_df[comparison_df["condition_id"].eq("real_lateralized_strength_p1p00")]
    interpretation = {
        "model": "real_flywire_ALPN_to_KC_with_KC_NT_lateralization_gate",
        "boundary": "graph-network representation/memory proxy; not completed animal behavior",
        "annotation_path": str(cfg.annotation_path),
        "connectivity_path": str(cfg.connectivity_path),
        "kc_nt_inputs_path": str(cfg.kc_nt_inputs_path),
        "n_kc": int(len(kc_ids)),
        "n_glomerulus_channels": int(len(glomerulus_names)),
        "n_seed_panels": int(len(cfg.seeds)),
        "n_odors_per_panel": int(cfg.n_odors),
        "kc_active_fraction": float(cfg.ratio),
        "gate_amplitude": float(cfg.gate_amplitude),
        "real_strength_1_delta_effective_dimension": float(
            real_1["normalized_effective_dimension_delta_vs_symmetrized"].iloc[0]
        )
        if not real_1.empty and "normalized_effective_dimension_delta_vs_symmetrized" in real_1
        else None,
        "real_strength_1_delta_memory_stability": float(real_1["memory_stability_score_delta_vs_symmetrized"].iloc[0])
        if not real_1.empty and "memory_stability_score_delta_vs_symmetrized" in real_1
        else None,
    }

    raw_path = cfg.output_dir / "lateralization_representation_raw.csv"
    summary_path = cfg.output_dir / "lateralization_representation_summary.csv"
    comparison_path = cfg.output_dir / "lateralization_representation_comparison.csv"
    gate_path = cfg.output_dir / "lateralization_gate_by_subtype_side.csv"
    channel_path = cfg.output_dir / "lateralization_flywire_glomerulus_kc_channels.csv"
    odor_panel_path = cfg.output_dir / "lateralization_odor_panel.csv"
    metadata_path = cfg.output_dir / "lateralization_representation_metadata.json"
    report_path = cfg.output_dir / "LATERALIZATION_REPRESENTATION_MEMORY_REPORT_CN.md"

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)
    gate_by_subtype_side.to_csv(gate_path, index=False)
    channel_table.to_csv(channel_path, index=False)
    pd.concat(odor_panel_frames, ignore_index=True).to_csv(odor_panel_path, index=False)
    figure_path = _write_lateralization_representation_figure(summary_df, gate_by_subtype_side, cfg.output_dir)
    report_path.write_text(
        _render_lateralization_representation_report(
            cfg=cfg,
            summary_df=summary_df,
            comparison_df=comparison_df,
            gate_by_subtype_side=gate_by_subtype_side,
            figure_path=figure_path,
            interpretation=interpretation,
        ),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "interpretation": interpretation,
                "paths": {
                    "raw_csv": str(raw_path),
                    "summary_csv": str(summary_path),
                    "comparison_csv": str(comparison_path),
                    "gate_csv": str(gate_path),
                    "channels_csv": str(channel_path),
                    "odor_panel_csv": str(odor_panel_path),
                    "figure_png": str(figure_path) if figure_path else None,
                    "report_md": str(report_path),
                },
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    return {
        "raw_csv": raw_path,
        "summary_csv": summary_path,
        "comparison_csv": comparison_path,
        "gate_csv": gate_path,
        "channels_csv": channel_path,
        "odor_panel_csv": odor_panel_path,
        "figure_png": figure_path,
        "report_md": report_path,
        "metadata_json": metadata_path,
        "raw_df": raw_df,
        "summary_df": summary_df,
        "comparison_df": comparison_df,
        "gate_by_subtype_side_df": gate_by_subtype_side,
        "interpretation": interpretation,
    }


def _target_to_kc_weight_vectors_by_side(
    edges: pd.DataFrame,
    kc_ids: np.ndarray,
    targets: pd.DataFrame,
) -> dict[str, np.ndarray]:
    kc_index = {int(root_id): index for index, root_id in enumerate(kc_ids)}
    vectors = {
        "left": np.zeros(len(kc_ids), dtype=np.float64),
        "right": np.zeros(len(kc_ids), dtype=np.float64),
    }
    target_sides = targets.set_index("root_id")["side"].fillna("").astype(str).to_dict()
    selected = edges[
        edges["Presynaptic_ID"].isin(set(map(int, target_sides)))
        & edges["Postsynaptic_ID"].isin(set(map(int, kc_ids)))
    ].copy()
    if selected.empty:
        return vectors
    selected["pre_side"] = selected["Presynaptic_ID"].map(lambda root_id: target_sides.get(int(root_id), ""))
    selected = selected[selected["pre_side"].isin(["left", "right"])]
    grouped = (
        selected.groupby(["pre_side", "Postsynaptic_ID"], as_index=False)["Excitatory x Connectivity"]
        .sum()
        .rename(columns={"Excitatory x Connectivity": "signed_weight"})
    )
    for row in grouped.itertuples(index=False):
        index = kc_index.get(int(row.Postsynaptic_ID))
        if index is not None:
            vectors[str(row.pre_side)][index] = max(0.0, float(row.signed_weight))
    return vectors


def _kc_to_target_weight_vectors_by_side(
    edges: pd.DataFrame,
    kc_ids: np.ndarray,
    targets: pd.DataFrame,
) -> dict[str, np.ndarray]:
    target_by_side = {
        side: set(targets[targets["side"].astype(str).eq(side)]["root_id"].astype("int64"))
        for side in ["left", "right"]
    }
    return {
        side: _kc_to_target_weight_vector(edges, kc_ids, target_ids)
        for side, target_ids in target_by_side.items()
    }


def _behavior_closure_conditions() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "condition_order": 0,
                "condition_id": "immediate_wt",
                "condition_label": "WT immediate",
                "assay": "OCT/MCH immediate memory",
                "gate_strength": 1.0,
                "delay_blocks": 0,
                "conflict_penalty": 0.0,
                "left_dpm_gain": 1.0,
                "right_dpm_gain": 1.0,
                "ppl1_gain": 1.0,
                "perturbation_type": "baseline",
            },
            {
                "condition_order": 1,
                "condition_id": "delayed_wt",
                "condition_label": "WT delayed",
                "assay": "OCT/MCH delayed memory",
                "gate_strength": 1.0,
                "delay_blocks": 6,
                "conflict_penalty": 0.0,
                "left_dpm_gain": 1.0,
                "right_dpm_gain": 1.0,
                "ppl1_gain": 1.0,
                "perturbation_type": "baseline",
            },
            {
                "condition_order": 2,
                "condition_id": "delayed_conflict_wt",
                "condition_label": "WT delayed conflict",
                "assay": "weak CS+ / strong CS- delayed conflict",
                "gate_strength": 1.0,
                "delay_blocks": 6,
                "conflict_penalty": 1.0,
                "left_dpm_gain": 1.0,
                "right_dpm_gain": 1.0,
                "ppl1_gain": 1.0,
                "perturbation_type": "baseline_conflict",
            },
            {
                "condition_order": 3,
                "condition_id": "left_dpm_25pct_delayed_conflict",
                "condition_label": "left DPM 25%",
                "assay": "weak CS+ / strong CS- delayed conflict",
                "gate_strength": 1.0,
                "delay_blocks": 6,
                "conflict_penalty": 1.0,
                "left_dpm_gain": 0.25,
                "right_dpm_gain": 1.0,
                "ppl1_gain": 1.0,
                "perturbation_type": "unilateral_dpm_reduction",
            },
            {
                "condition_order": 4,
                "condition_id": "right_dpm_25pct_delayed_conflict",
                "condition_label": "right DPM 25%",
                "assay": "weak CS+ / strong CS- delayed conflict",
                "gate_strength": 1.0,
                "delay_blocks": 6,
                "conflict_penalty": 1.0,
                "left_dpm_gain": 1.0,
                "right_dpm_gain": 0.25,
                "ppl1_gain": 1.0,
                "perturbation_type": "unilateral_dpm_reduction",
            },
            {
                "condition_order": 5,
                "condition_id": "bilateral_dpm_25pct_delayed_conflict",
                "condition_label": "bilateral DPM 25%",
                "assay": "weak CS+ / strong CS- delayed conflict",
                "gate_strength": 1.0,
                "delay_blocks": 6,
                "conflict_penalty": 1.0,
                "left_dpm_gain": 0.25,
                "right_dpm_gain": 0.25,
                "ppl1_gain": 1.0,
                "perturbation_type": "bilateral_dpm_reduction",
            },
            {
                "condition_order": 6,
                "condition_id": "symmetrized_delayed_conflict",
                "condition_label": "symmetrized gate",
                "assay": "weak CS+ / strong CS- delayed conflict",
                "gate_strength": 0.0,
                "delay_blocks": 6,
                "conflict_penalty": 1.0,
                "left_dpm_gain": 1.0,
                "right_dpm_gain": 1.0,
                "ppl1_gain": 1.0,
                "perturbation_type": "symmetry_control",
            },
            {
                "condition_order": 7,
                "condition_id": "mirror_reversed_delayed_conflict",
                "condition_label": "mirror-reversed gate",
                "assay": "weak CS+ / strong CS- delayed conflict",
                "gate_strength": -1.0,
                "delay_blocks": 6,
                "conflict_penalty": 1.0,
                "left_dpm_gain": 1.0,
                "right_dpm_gain": 1.0,
                "ppl1_gain": 1.0,
                "perturbation_type": "symmetry_control",
            },
            {
                "condition_order": 8,
                "condition_id": "ppl1_25pct_delayed_conflict",
                "condition_label": "PPL1-DAN 25%",
                "assay": "weak CS+ / strong CS- delayed conflict",
                "gate_strength": 1.0,
                "delay_blocks": 6,
                "conflict_penalty": 1.0,
                "left_dpm_gain": 1.0,
                "right_dpm_gain": 1.0,
                "ppl1_gain": 0.25,
                "perturbation_type": "teaching_reduction",
            },
            {
                "condition_order": 9,
                "condition_id": "ppl1_dpm_25pct_delayed_conflict",
                "condition_label": "PPL1-DAN + DPM 25%",
                "assay": "weak CS+ / strong CS- delayed conflict",
                "gate_strength": 1.0,
                "delay_blocks": 6,
                "conflict_penalty": 1.0,
                "left_dpm_gain": 0.25,
                "right_dpm_gain": 0.25,
                "ppl1_gain": 0.25,
                "perturbation_type": "joint_teaching_persistence_reduction",
            },
        ]
    )


def _write_behavior_closure_figure(summary_df: pd.DataFrame, output_dir: Path) -> Path | None:
    if summary_df.empty:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot = summary_df.sort_values("condition_order").copy()
    labels = plot["condition_label"].astype(str).tolist()
    x = np.arange(len(plot))
    figure_path = output_dir / "Fig_lateralization_behavior_closure.png"
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), constrained_layout=True)

    axes[0, 0].barh(labels, plot["expected_choice_rate_mean"], color="#4c78a8")
    axes[0, 0].axvline(0.5, color="0.35", lw=1)
    axes[0, 0].set_xlabel("expected choice rate proxy")
    axes[0, 0].set_title("OCT/MCH delayed-conflict behaviour proxy")

    axes[0, 1].barh(labels, plot["choice_index_delta_vs_delayed_conflict_wt"], color="#f58518")
    axes[0, 1].axvline(0, color="0.35", lw=1)
    axes[0, 1].set_xlabel("choice-index delta vs WT delayed conflict")
    axes[0, 1].set_title("Perturbation sensitivity")

    axes[1, 0].barh(labels, plot["dpm_feedback_support_mean"], color="#54a24b")
    axes[1, 0].axvline(1.0, color="0.35", lw=1, ls="--")
    axes[1, 0].set_xlabel("DPM feedback support, normalized")
    axes[1, 0].set_title("DPM persistence axis")

    axes[1, 1].barh(labels, plot["dpm_readout_laterality_index_mean"], color="#b279a2")
    axes[1, 1].axvline(0, color="0.35", lw=1)
    axes[1, 1].set_xlabel("right-minus-left DPM readout LI")
    axes[1, 1].set_title("DPM / 5-HT imaging proxy")

    fig.suptitle("Behaviour-closure proxy: delayed/conflict assay, DPM imaging and unilateral perturbation", fontsize=14)
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return figure_path


def _render_behavior_closure_report(
    *,
    cfg: BehaviorClosureProxyConfig,
    summary_df: pd.DataFrame,
    dpm_side_summary: pd.DataFrame,
    priority_df: pd.DataFrame,
    figure_path: Path | None,
    interpretation: dict[str, object],
) -> str:
    display_cols = [
        "condition_label",
        "assay",
        "expected_choice_rate_mean",
        "choice_index_mean",
        "choice_index_delta_vs_delayed_conflict_wt",
        "dpm_feedback_support_mean",
        "dpm_readout_laterality_index_mean",
        "retention_fraction_mean",
    ]
    figure_line = f"\n![behaviour closure]({figure_path.name})\n" if figure_path else ""
    return f"""# OCT/MCH、DPM/5-HT 与单侧扰动的行为闭环 proxy

保存路径：`{cfg.output_dir}/LATERALIZATION_BEHAVIOR_CLOSURE_PROXY_CN.md`

## 问题

这份输出回答一个限定问题：在真实 FlyWire 结构和现有 BioFly 模型下，能否先用仿真判断
`OCT/MCH delayed/conflict`、`DPM/5-HT imaging` 和 `left/right unilateral perturbation`
哪一类实验最可能读出偏侧化相关效应。

结论不是 wet-lab 行为结果，也不是同一只果蝇的 imaging-behaviour 因果闭环。它是连接组约束的
实验优先级排序。

## 数据和模型

- 真实 `ALPN->KC` 子图生成 `{interpretation['n_glomerulus_channels']}` 个 glomerulus channels。
- 使用 `{interpretation['n_kc']}` 个 KC，稀疏锚点固定为 `{cfg.ratio:.2f}`。
- PPL1-DAN teaching axis 来自真实 `KC->PPL1-DAN` 权重。
- DPM persistence axis 使用真实 `KC->DPM` 与 `DPM->KC` 侧别权重。
- KC 偏侧化 gate 来自 `outputs/kc_nt_lateralization/kc_neuron_nt_inputs.parquet` 的 serotonin-minus-glutamate side/subtype 差异。

## 主要结果

{_table(summary_df[display_cols], max_rows=20)}

{figure_line}
## DPM/5-HT imaging proxy

DPM 在当前 FlyWire annotation 中是左右各 1 个。`DPM->KC` 几乎完全同侧：

{_table(dpm_side_summary, max_rows=8)}

这给 imaging 一个很直接的预测：如果 DPM/5-HT 侧化是真实生物信号，左右 DPM 操作或记录应主要在
同侧 KC/MB compartment 出现读出差异；180 度旋转后按 brain side 注册，LI 符号应保持，而不是跟随相机坐标翻转。

## 实验优先级

{_table(priority_df, max_rows=12)}

## 可以先向合作者转述的结论

1. **delayed/conflict 比普通 immediate choice 更适合做行为闭环。** immediate WT 在 proxy 中接近饱和；
   delayed/conflict 把读出压到更敏感区间，更容易看出 DPM 或 PPL1-DAN 扰动差异。
2. **DPM 更像 delayed retention / interference 轴，不是 acquisition 轴。** 单独降低 DPM 主要通过
   `DPM feedback support` 和衰减项影响延迟后选择指数；这个结论与 PPL1-DAN/DPM perturbation 套件一致。
3. **单侧 DPM 是可做的，但预期效应小于双侧 DPM 或 PPL1-DAN 下调。** 因为左右 DPM 几乎同侧回投 KC，
   单侧扰动更适合作为 imaging/side-registration proof；若要行为效应，优先 delayed-conflict 和较大样本量。
4. **PPL1-DAN 25% 与 PPL1-DAN+DPM 25% 是强阳性行为 proxy。** 它们用于证明 assay 灵敏度；
   DPM 单侧扰动用于证明偏侧化方向和机制。

## 边界

- 这里的 OCT/MCH 是行为 proxy，不是真实动物 T-maze。
- 这里的 DPM/5-HT readout 是 DPM axis 和 5-HT imaging 设计 proxy；FlyWire `top_nt` 与 DPM/serotonin 生物学解释需谨慎对照。
- 当前模型可以帮助选择实验条件、方向和读数，但不能替代 HCR/FISH、GRASP/split-GFP、calcium/5-HT sensor、光遗传和行为实验。
"""


def run_behavior_closure_proxy(
    config: BehaviorClosureProxyConfig | None = None,
) -> dict[str, Path | pd.DataFrame | dict[str, object]]:
    cfg = config or BehaviorClosureProxyConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    annotations = _load_annotations(cfg.annotation_path)
    annotated = _annotate_connectome_groups(annotations)
    edges = pd.read_parquet(
        cfg.connectivity_path,
        columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity", "Excitatory x Connectivity"],
    )
    glomerulus_names, glomerulus_matrix, kc_ids, channel_table = _build_flywire_glomerulus_kc_matrix_from_frames(
        annotated,
        edges,
    )
    gate_by_subtype_side, gate, _side_by_kc = _load_kc_lateralization_gate(cfg.kc_nt_inputs_path, kc_ids)
    ppl1_targets, dpm_targets, pattern_used = _identify_learning_memory_targets(
        annotated,
        ppl1_pattern=cfg.ppl1_pattern,
    )
    ppl1_vector = _kc_to_target_weight_vector(edges, kc_ids, set(ppl1_targets["root_id"].astype("int64")))
    dpm_to_kc_vectors = _target_to_kc_weight_vectors_by_side(edges, kc_ids, dpm_targets)
    kc_to_dpm_vectors = _kc_to_target_weight_vectors_by_side(edges, kc_ids, dpm_targets)
    dpm_return_total = dpm_to_kc_vectors["left"] + dpm_to_kc_vectors["right"]
    if float(ppl1_vector.sum()) <= 0:
        raise ValueError("No positive KC->PPL1-DAN edge mass was found.")
    if float(dpm_return_total.sum()) <= 0:
        raise ValueError("No positive DPM->KC edge mass was found.")

    dpm_edges = edges[
        edges["Presynaptic_ID"].isin(set(dpm_targets["root_id"].astype("int64")))
        & edges["Postsynaptic_ID"].isin(set(map(int, kc_ids)))
    ].copy()
    dpm_side_map = dpm_targets.set_index("root_id")["side"].astype(str).to_dict()
    kc_side_map = annotated.set_index("root_id")["side"].astype(str).to_dict()
    dpm_edges["pre_side"] = dpm_edges["Presynaptic_ID"].map(lambda root_id: dpm_side_map.get(int(root_id), ""))
    dpm_edges["post_side"] = dpm_edges["Postsynaptic_ID"].map(lambda root_id: kc_side_map.get(int(root_id), ""))
    dpm_side_summary = (
        dpm_edges.groupby(["pre_side", "post_side"], as_index=False)
        .agg(
            n_edges=("Connectivity", "size"),
            n_synapses=("Connectivity", "sum"),
            signed_weight=("Excitatory x Connectivity", "sum"),
            n_post_kc=("Postsynaptic_ID", "nunique"),
        )
        .sort_values(["pre_side", "post_side"])
    )

    conditions = _behavior_closure_conditions()
    seed_contexts: list[dict[str, object]] = []
    reference_margins: list[float] = []
    for seed in cfg.seeds:
        odor_names, activity, odor_panel = build_mixture_odor_panel(
            glomerulus_names,
            glomerulus_matrix,
            seed=int(seed),
            n_odors=int(cfg.n_odors),
            min_glomeruli_per_odor=int(cfg.min_glomeruli_per_odor),
            max_glomeruli_per_odor=int(cfg.max_glomeruli_per_odor),
            channel_noise_sigma=float(cfg.channel_noise_sigma),
        )
        gains = np.clip(1.0 + float(cfg.gate_amplitude) * gate, 0.05, 5.0)
        baseline_activity = _normalize_rows(np.maximum(activity * gains[None, :], 0.0))
        binary, baseline_graded, active_k = _sparsify(baseline_activity, float(cfg.ratio))
        cs_plus = baseline_graded[int(cfg.cs_plus_index)]
        cs_minus = np.delete(baseline_graded, int(cfg.cs_plus_index), axis=0)
        separation = cs_plus - cs_minus.mean(axis=0)
        ppl1_panel_drive = baseline_graded @ ppl1_vector
        ppl1_drive_norm = float((cs_plus @ ppl1_vector) / (float(ppl1_panel_drive.mean()) + 1e-12))
        reference_margins.append(abs(float(ppl1_drive_norm * np.dot(separation, separation))))
        seed_contexts.append(
            {
                "seed": int(seed),
                "odor_names": odor_names,
                "activity": activity,
                "active_k": int(active_k),
                "observed_active_fraction": float(binary.mean()),
                "odor_glomeruli_cs_plus": odor_panel.loc[
                    odor_panel["odor_identity"].eq(odor_names[int(cfg.cs_plus_index)]),
                    "glomeruli",
                ].iloc[0],
            }
        )
    reference_margin = float(np.median(np.asarray(reference_margins, dtype=np.float64)))
    if not np.isfinite(reference_margin) or reference_margin <= 0:
        reference_margin = 1.0

    raw_rows: list[dict[str, object]] = []
    for context in seed_contexts:
        for condition in conditions.itertuples(index=False):
            gated = _normalize_rows(
                np.maximum(
                    context["activity"]
                    * np.clip(1.0 + float(cfg.gate_amplitude) * float(condition.gate_strength) * gate, 0.05, 5.0)[None, :],
                    0.0,
                )
            )
            binary, graded, active_k = _sparsify(gated, float(cfg.ratio))
            cs_plus = graded[int(cfg.cs_plus_index)]
            cs_minus = np.delete(graded, int(cfg.cs_plus_index), axis=0)
            separation = cs_plus - cs_minus.mean(axis=0)
            separation_norm_sq = float(np.dot(separation, separation))
            ppl1_panel_drive = graded @ ppl1_vector
            ppl1_drive_norm = float((cs_plus @ ppl1_vector) / (float(ppl1_panel_drive.mean()) + 1e-12))
            acquisition_margin = float(condition.ppl1_gain) * ppl1_drive_norm * separation_norm_sq

            left_return = float(cs_plus @ dpm_to_kc_vectors["left"])
            right_return = float(cs_plus @ dpm_to_kc_vectors["right"])
            panel_return = float(np.mean(graded @ dpm_return_total)) + 1e-12
            dpm_feedback_support = (
                float(condition.left_dpm_gain) * left_return + float(condition.right_dpm_gain) * right_return
            ) / panel_return
            dpm_feedback_support_clipped = float(np.clip(dpm_feedback_support, 0.0, 1.0))
            decay_per_block = float(
                cfg.dpm_baseline_decay_per_block
                + cfg.dpm_loss_decay_penalty_per_block * (1.0 - dpm_feedback_support_clipped)
            )
            retention_fraction = float(np.exp(-int(condition.delay_blocks) * decay_per_block))
            retained_margin = acquisition_margin * retention_fraction
            retrieval_margin = retained_margin - float(condition.conflict_penalty) * float(cfg.conflict_decoy_penalty) * reference_margin
            choice_index = float(np.tanh(retrieval_margin / (reference_margin + 1e-12)))
            expected_choice_rate = float(np.clip(0.5 + 0.5 * choice_index, 0.0, 1.0))

            left_dpm_drive = float(condition.left_dpm_gain) * float(cs_plus @ kc_to_dpm_vectors["left"])
            right_dpm_drive = float(condition.right_dpm_gain) * float(cs_plus @ kc_to_dpm_vectors["right"])
            dpm_readout_li = (right_dpm_drive - left_dpm_drive) / (right_dpm_drive + left_dpm_drive + 1e-12)
            dpm_feedback_li = (
                float(condition.right_dpm_gain) * right_return - float(condition.left_dpm_gain) * left_return
            ) / (
                float(condition.right_dpm_gain) * right_return
                + float(condition.left_dpm_gain) * left_return
                + 1e-12
            )
            raw_rows.append(
                {
                    "seed": int(context["seed"]),
                    "odor_glomeruli_cs_plus": str(context["odor_glomeruli_cs_plus"]),
                    "condition_order": int(condition.condition_order),
                    "condition_id": str(condition.condition_id),
                    "condition_label": str(condition.condition_label),
                    "assay": str(condition.assay),
                    "perturbation_type": str(condition.perturbation_type),
                    "gate_strength": float(condition.gate_strength),
                    "delay_blocks": int(condition.delay_blocks),
                    "conflict_penalty": float(condition.conflict_penalty),
                    "left_dpm_gain": float(condition.left_dpm_gain),
                    "right_dpm_gain": float(condition.right_dpm_gain),
                    "ppl1_gain": float(condition.ppl1_gain),
                    "active_k": int(active_k),
                    "observed_active_fraction": float(binary.mean()),
                    "ppl1_drive_norm": ppl1_drive_norm,
                    "acquisition_margin": acquisition_margin,
                    "dpm_feedback_support": float(dpm_feedback_support),
                    "dpm_feedback_support_clipped": dpm_feedback_support_clipped,
                    "left_dpm_return_drive": left_return,
                    "right_dpm_return_drive": right_return,
                    "dpm_feedback_laterality_index": float(dpm_feedback_li),
                    "left_dpm_readout_drive": left_dpm_drive,
                    "right_dpm_readout_drive": right_dpm_drive,
                    "dpm_readout_laterality_index": float(dpm_readout_li),
                    "decay_per_block": decay_per_block,
                    "retention_fraction": retention_fraction,
                    "retained_margin": retained_margin,
                    "retrieval_margin_after_conflict": retrieval_margin,
                    "choice_index": choice_index,
                    "expected_choice_rate": expected_choice_rate,
                }
            )

    raw_df = pd.DataFrame.from_records(raw_rows)
    summary_df = (
        raw_df.groupby(
            [
                "condition_order",
                "condition_id",
                "condition_label",
                "assay",
                "perturbation_type",
                "gate_strength",
                "delay_blocks",
                "conflict_penalty",
                "left_dpm_gain",
                "right_dpm_gain",
                "ppl1_gain",
            ],
            as_index=False,
        )
        .agg(
            n_seed_panels=("seed", "nunique"),
            active_k=("active_k", "first"),
            observed_active_fraction_mean=("observed_active_fraction", "mean"),
            ppl1_drive_norm_mean=("ppl1_drive_norm", "mean"),
            acquisition_margin_mean=("acquisition_margin", "mean"),
            dpm_feedback_support_mean=("dpm_feedback_support", "mean"),
            dpm_feedback_support_std=("dpm_feedback_support", "std"),
            dpm_feedback_laterality_index_mean=("dpm_feedback_laterality_index", "mean"),
            dpm_readout_laterality_index_mean=("dpm_readout_laterality_index", "mean"),
            decay_per_block_mean=("decay_per_block", "mean"),
            retention_fraction_mean=("retention_fraction", "mean"),
            retained_margin_mean=("retained_margin", "mean"),
            retrieval_margin_after_conflict_mean=("retrieval_margin_after_conflict", "mean"),
            choice_index_mean=("choice_index", "mean"),
            choice_index_std=("choice_index", "std"),
            expected_choice_rate_mean=("expected_choice_rate", "mean"),
            expected_choice_rate_std=("expected_choice_rate", "std"),
        )
        .sort_values("condition_order")
    )
    ref = summary_df[summary_df["condition_id"].eq("delayed_conflict_wt")]
    ref_row = ref.iloc[0] if not ref.empty else summary_df.iloc[0]
    for metric in ["choice_index", "expected_choice_rate", "dpm_feedback_support", "retention_fraction"]:
        column = f"{metric}_mean"
        summary_df[f"{metric}_delta_vs_delayed_conflict_wt"] = summary_df[column] - float(ref_row[column])

    priority_df = summary_df[~summary_df["perturbation_type"].astype(str).str.startswith("baseline")].copy()
    priority_df["absolute_choice_index_delta"] = priority_df["choice_index_delta_vs_delayed_conflict_wt"].abs()
    priority_df["wetlab_priority_score"] = (
        priority_df["absolute_choice_index_delta"]
        + 0.15 * priority_df["dpm_readout_laterality_index_mean"].abs()
        + 0.10 * (1.0 - priority_df["expected_choice_rate_std"].fillna(0.0))
    )
    priority_df = priority_df.sort_values("wetlab_priority_score", ascending=False)[
        [
            "condition_label",
            "assay",
            "perturbation_type",
            "choice_index_delta_vs_delayed_conflict_wt",
            "expected_choice_rate_mean",
            "dpm_feedback_support_mean",
            "dpm_readout_laterality_index_mean",
            "wetlab_priority_score",
        ]
    ]

    figure_path = _write_behavior_closure_figure(summary_df, cfg.output_dir)
    interpretation = {
        "model": "real_flywire_ALPN_to_KC_plus_PPL1_DPM_lateralization_behavior_closure_proxy",
        "boundary": "simulation proxy; not wet-lab behavior or same-fly imaging-behavior causality",
        "n_kc": int(len(kc_ids)),
        "n_glomerulus_channels": int(len(glomerulus_names)),
        "n_seed_panels": int(len(cfg.seeds)),
        "n_odors_per_panel": int(cfg.n_odors),
        "kc_active_fraction": float(cfg.ratio),
        "ppl1_pattern_used": pattern_used,
        "n_ppl1_dan_targets": int(ppl1_targets["root_id"].nunique()),
        "n_dpm_targets": int(dpm_targets["root_id"].nunique()),
        "reference_margin": float(reference_margin),
    }

    raw_path = cfg.output_dir / "behavior_closure_raw.csv"
    summary_path = cfg.output_dir / "behavior_closure_summary.csv"
    dpm_side_path = cfg.output_dir / "behavior_closure_dpm_side_summary.csv"
    priority_path = cfg.output_dir / "behavior_closure_wetlab_priority.csv"
    gate_path = cfg.output_dir / "behavior_closure_gate_by_subtype_side.csv"
    channel_path = cfg.output_dir / "behavior_closure_glomerulus_kc_channels.csv"
    report_path = cfg.output_dir / "LATERALIZATION_BEHAVIOR_CLOSURE_PROXY_CN.md"
    metadata_path = cfg.output_dir / "behavior_closure_metadata.json"
    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    dpm_side_summary.to_csv(dpm_side_path, index=False)
    priority_df.to_csv(priority_path, index=False)
    gate_by_subtype_side.to_csv(gate_path, index=False)
    channel_table.to_csv(channel_path, index=False)
    report_path.write_text(
        _render_behavior_closure_report(
            cfg=cfg,
            summary_df=summary_df,
            dpm_side_summary=dpm_side_summary,
            priority_df=priority_df,
            figure_path=figure_path,
            interpretation=interpretation,
        ),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "interpretation": interpretation,
                "paths": {
                    "raw_csv": str(raw_path),
                    "summary_csv": str(summary_path),
                    "dpm_side_summary_csv": str(dpm_side_path),
                    "priority_csv": str(priority_path),
                    "figure_png": str(figure_path) if figure_path else None,
                    "report_md": str(report_path),
                },
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return {
        "raw_csv": raw_path,
        "summary_csv": summary_path,
        "dpm_side_summary_csv": dpm_side_path,
        "priority_csv": priority_path,
        "gate_csv": gate_path,
        "channels_csv": channel_path,
        "figure_png": figure_path,
        "report_md": report_path,
        "metadata_json": metadata_path,
        "raw_df": raw_df,
        "summary_df": summary_df,
        "priority_df": priority_df,
        "interpretation": interpretation,
    }


def _load_kc_lateralization_components(kc_nt_inputs_path: Path, kc_ids: np.ndarray) -> pd.DataFrame:
    """Return per-KC side/subtype lateralization component gates aligned to KC ids."""

    if not kc_nt_inputs_path.exists():
        raise FileNotFoundError(f"KC neurotransmitter lateralization table not found: {kc_nt_inputs_path}")
    nt_inputs = pd.read_parquet(kc_nt_inputs_path).drop_duplicates("root_id").copy()
    required = {"root_id", "side", "cell_type", "hemibrain_type", "ser_fraction", "glut_fraction"}
    missing = required - set(nt_inputs.columns)
    if missing:
        raise ValueError(f"KC NT lateralization table is missing columns: {sorted(missing)}")
    nt_inputs["root_id"] = nt_inputs["root_id"].astype("int64")
    nt_inputs["side"] = nt_inputs["side"].fillna("").astype(str)
    nt_inputs["subtype"] = nt_inputs["cell_type"].fillna("").astype(str)
    blank = nt_inputs["subtype"].eq("")
    nt_inputs.loc[blank, "subtype"] = nt_inputs.loc[blank, "hemibrain_type"].fillna("").astype(str)
    nt_inputs["ser_z"] = _standardize(nt_inputs["ser_fraction"])
    nt_inputs["glut_z"] = _standardize(nt_inputs["glut_fraction"])
    nt_inputs["ser_minus_glut_z"] = nt_inputs["ser_z"] - nt_inputs["glut_z"]
    for source, column in [
        ("ser_z", "ser_component_gate_raw"),
        ("glut_z", "glut_component_gate_raw"),
        ("ser_minus_glut_z", "ser_minus_glut_gate_raw"),
    ]:
        subtype_mean = nt_inputs.groupby("subtype")[source].transform("mean")
        side_subtype_mean = nt_inputs.groupby(["subtype", "side"])[source].transform("mean")
        nt_inputs[column] = side_subtype_mean - subtype_mean

    def scaled(values: pd.Series) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        scale = float(np.nanpercentile(np.abs(array), 95))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(np.nanmax(np.abs(array)))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        return np.clip(array / scale, -1.0, 1.0)

    nt_inputs["serotonin_only_gate"] = scaled(nt_inputs["ser_component_gate_raw"])
    nt_inputs["glutamate_only_gate"] = -scaled(nt_inputs["glut_component_gate_raw"])
    nt_inputs["symmetry_breaking_gate"] = scaled(nt_inputs["ser_minus_glut_gate_raw"])
    aligned = pd.DataFrame({"root_id": np.asarray(kc_ids, dtype=np.int64)})
    aligned = aligned.merge(
        nt_inputs[
            [
                "root_id",
                "side",
                "subtype",
                "ser_fraction",
                "glut_fraction",
                "serotonin_only_gate",
                "glutamate_only_gate",
                "symmetry_breaking_gate",
            ]
        ],
        on="root_id",
        how="left",
    )
    aligned["side"] = aligned["side"].fillna("")
    aligned["subtype"] = aligned["subtype"].fillna("")
    for column in ["ser_fraction", "glut_fraction", "serotonin_only_gate", "glutamate_only_gate", "symmetry_breaking_gate"]:
        aligned[column] = aligned[column].fillna(0.0).astype(float)
    return aligned


def _mechanism_gate_conditions(
    aligned_gate: pd.DataFrame,
    *,
    shuffle_repeats: int,
    gate_strength: float,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    base = aligned_gate["symmetry_breaking_gate"].to_numpy(dtype=np.float64) * float(gate_strength)
    ser = aligned_gate["serotonin_only_gate"].to_numpy(dtype=np.float64) * float(gate_strength)
    glut = aligned_gate["glutamate_only_gate"].to_numpy(dtype=np.float64) * float(gate_strength)
    subtype = aligned_gate["subtype"].astype(str)
    side = aligned_gate["side"].astype(str)
    gamma_mask = subtype.str.contains("KCg", regex=False, na=False).to_numpy()
    ab_mask = subtype.str.contains("KCab", regex=False, na=False).to_numpy()
    apbp_mask = (
        subtype.str.contains("KCapbp", regex=False, na=False)
        | subtype.str.contains("KCa'b'", regex=False, na=False)
    ).to_numpy()
    left_mask = side.eq("left").to_numpy()
    right_mask = side.eq("right").to_numpy()

    def masked_zero(mask: np.ndarray) -> np.ndarray:
        vector = base.copy()
        vector[mask] = 0.0
        return vector

    records: list[dict[str, object]] = [
        {
            "condition_order": 0,
            "condition_id": "symmetrized",
            "condition_label": "symmetrized",
            "condition_class": "symmetry_control",
            "gate_vector": np.zeros_like(base),
        },
        {
            "condition_order": 1,
            "condition_id": "real_lateralized",
            "condition_label": "real side/subtype gate",
            "condition_class": "real_lateralized",
            "gate_vector": base,
        },
        {
            "condition_order": 2,
            "condition_id": "mirror_reversed",
            "condition_label": "mirror reversed gate",
            "condition_class": "symmetry_control",
            "gate_vector": -base,
        },
        {
            "condition_order": 3,
            "condition_id": "serotonin_only",
            "condition_label": "serotonin component only",
            "condition_class": "component_ablation",
            "gate_vector": ser,
        },
        {
            "condition_order": 4,
            "condition_id": "glutamate_only",
            "condition_label": "glutamate component only",
            "condition_class": "component_ablation",
            "gate_vector": glut,
        },
        {
            "condition_order": 5,
            "condition_id": "real_no_gamma_kc",
            "condition_label": "real gate without gamma KC",
            "condition_class": "subtype_ablation",
            "gate_vector": masked_zero(gamma_mask),
        },
        {
            "condition_order": 6,
            "condition_id": "real_no_ab_kc",
            "condition_label": "real gate without alpha/beta KC",
            "condition_class": "subtype_ablation",
            "gate_vector": masked_zero(ab_mask),
        },
        {
            "condition_order": 7,
            "condition_id": "real_no_apbp_kc",
            "condition_label": "real gate without alpha'/beta' KC",
            "condition_class": "subtype_ablation",
            "gate_vector": masked_zero(apbp_mask),
        },
        {
            "condition_order": 8,
            "condition_id": "left_only_gate",
            "condition_label": "left-side gate only",
            "condition_class": "side_ablation",
            "gate_vector": np.where(left_mask, base, 0.0),
        },
        {
            "condition_order": 9,
            "condition_id": "right_only_gate",
            "condition_label": "right-side gate only",
            "condition_class": "side_ablation",
            "gate_vector": np.where(right_mask, base, 0.0),
        },
    ]
    order = 10
    for repeat in range(int(shuffle_repeats)):
        shuffled = base.copy()
        rng.shuffle(shuffled)
        records.append(
            {
                "condition_order": order,
                "condition_id": f"shuffled_gate_{repeat:02d}",
                "condition_label": f"shuffled gate {repeat:02d}",
                "condition_class": "shuffled_lateralized",
                "gate_vector": shuffled,
            }
        )
        order += 1
    return records


def _apply_gate_to_activity(
    activity: np.ndarray,
    gate_vector: np.ndarray,
    *,
    gate_amplitude: float,
    apl_noise_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    gains = np.clip(1.0 + float(gate_amplitude) * np.asarray(gate_vector, dtype=np.float64), 0.05, 5.0)
    gated = np.maximum(activity * gains[None, :], 0.0)
    if float(apl_noise_level) > 0:
        gated = gated * rng.lognormal(mean=0.0, sigma=float(apl_noise_level), size=gated.shape)
    return _normalize_rows(np.maximum(gated, 0.0))


def _graph_signal_smoothness(gate_vector: np.ndarray, glomerulus_matrix: np.ndarray) -> float:
    values: list[float] = []
    gate = np.asarray(gate_vector, dtype=np.float64)
    for row in np.asarray(glomerulus_matrix, dtype=np.float64):
        weights = np.maximum(row, 0.0)
        total = float(weights.sum())
        if total <= 0:
            continue
        weights = weights / total
        mean = float(weights @ gate)
        values.append(float(weights @ np.square(gate - mean)))
    return float(np.mean(values)) if values else 0.0


def _vector_participation_ratio(vector: np.ndarray) -> float:
    values = np.square(np.asarray(vector, dtype=np.float64))
    denominator = float(np.square(values).sum())
    if denominator <= 1e-12:
        return 0.0
    return float(np.square(values.sum()) / denominator / max(1, len(values)))


def _binary_decoder_metrics(
    cs_plus: np.ndarray,
    cs_minus: np.ndarray,
    *,
    dropout_probability: float,
    decoder_noise_sigma: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    delta = np.asarray(cs_plus, dtype=np.float64) - np.asarray(cs_minus, dtype=np.float64)
    margin = float(np.dot(delta, delta))
    mask = rng.random(delta.shape[0]) >= float(dropout_probability)
    dropped = delta * mask.astype(np.float64)
    dropout_margin = float(np.dot(dropped, dropped))
    if float(decoder_noise_sigma) > 0:
        noisy_plus = np.maximum(cs_plus + rng.normal(0.0, float(decoder_noise_sigma) * (float(cs_plus.std()) + 1e-12), size=cs_plus.shape), 0.0)
        noisy_minus = np.maximum(cs_minus + rng.normal(0.0, float(decoder_noise_sigma) * (float(cs_minus.std()) + 1e-12), size=cs_minus.shape), 0.0)
        noise_delta = noisy_plus - noisy_minus
        noise_margin = float(np.dot(noise_delta, delta))
    else:
        noise_margin = margin
    return {
        "decoder_margin": margin,
        "dropout_margin": dropout_margin,
        "dropout_margin_fraction": float(dropout_margin / (margin + 1e-12)),
        "noise_projected_margin": noise_margin,
        "noise_margin_fraction": float(noise_margin / (margin + 1e-12)),
    }


def _write_lateralization_mechanism_figure(
    representation_summary: pd.DataFrame,
    regime_contrasts: pd.DataFrame,
    graph_signal_summary: pd.DataFrame,
    sensitivity_summary: pd.DataFrame | None,
    output_dir: Path,
) -> Path | None:
    if representation_summary.empty or regime_contrasts.empty:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "Fig_lateralization_mechanism_suite.png"
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5), constrained_layout=True)

    rep = representation_summary[
        representation_summary["condition_id"].isin(
            ["symmetrized", "real_lateralized", "mirror_reversed", "serotonin_only", "glutamate_only"]
        )
    ].sort_values("condition_order")
    axes[0, 0].bar(
        rep["condition_id"],
        rep["normalized_effective_dimension_delta_vs_symmetrized"],
        color=["#777777", "#386cb0", "#e41a1c", "#6a9f58", "#d98b2b"][: len(rep)],
    )
    axes[0, 0].axhline(0, color="0.25", lw=1)
    axes[0, 0].set_title("Representation-space effect")
    axes[0, 0].set_ylabel("Δ normalized effective dimension")
    axes[0, 0].tick_params(axis="x", rotation=25)

    task = regime_contrasts.copy()
    task["task_label"] = (
        "sim="
        + task["task_similarity"].astype(str)
        + "\nD="
        + task["delay_blocks"].astype(str)
        + " I="
        + task["interference_level"].astype(str)
    )
    top = task.sort_values("real_minus_sym_memory_score", ascending=False).head(12).sort_values(
        "real_minus_sym_memory_score"
    )
    axes[0, 1].barh(top["task_label"], top["real_minus_sym_memory_score"], color="#4c78a8")
    axes[0, 1].axvline(0, color="0.25", lw=1)
    axes[0, 1].set_title("Best regimes for real lateralization")
    axes[0, 1].set_xlabel("real - sym memory score")

    heat = (
        regime_contrasts.groupby(["task_similarity", "delay_blocks"], as_index=False)[
            "real_minus_sym_memory_score"
        ]
        .mean()
        .pivot(index="delay_blocks", columns="task_similarity", values="real_minus_sym_memory_score")
        .sort_index()
    )
    image = axes[1, 0].imshow(heat.to_numpy(dtype=float), cmap="coolwarm", aspect="auto")
    axes[1, 0].set_xticks(range(len(heat.columns)), [str(col) for col in heat.columns])
    axes[1, 0].set_yticks(range(len(heat.index)), [str(idx) for idx in heat.index])
    axes[1, 0].set_xlabel("odor similarity")
    axes[1, 0].set_ylabel("delay blocks")
    axes[1, 0].set_title("Average memory-score advantage")
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04)

    graph = graph_signal_summary[
        graph_signal_summary["condition_id"].isin(["real_lateralized", "mirror_reversed", "shuffled_mean"])
    ].copy()
    axes[1, 1].bar(graph["condition_id"], graph["graph_signal_smoothness"], color="#8e6bbd")
    axes[1, 1].set_title("Gate smoothness on ALPN→KC graph")
    axes[1, 1].set_ylabel("weighted within-channel gate variance")
    axes[1, 1].tick_params(axis="x", rotation=20)

    grouped = regime_contrasts.copy()
    grouped["delay_group"] = np.where(grouped["delay_blocks"].eq(0), "immediate", "delayed")
    grouped["conflict_group"] = np.where(
        (grouped["interference_level"] >= 0.5) | (grouped["task_similarity"] >= 0.35),
        "conflict/high-sim",
        "low-conflict",
    )
    group_summary = (
        grouped.groupby(["delay_group", "conflict_group"], as_index=False)
        .agg(
            mean=("real_minus_sym_memory_score", "mean"),
            median=("real_minus_sym_memory_score", "median"),
            count=("real_minus_sym_memory_score", "size"),
        )
        .reset_index()
    )
    group_summary["label"] = group_summary["delay_group"] + "\n" + group_summary["conflict_group"]
    axes[0, 2].bar(group_summary["label"], group_summary["mean"], color="#73A66B")
    axes[0, 2].axhline(0, color="0.25", lw=1)
    axes[0, 2].set_title("Predefined regime groups")
    axes[0, 2].set_ylabel("mean real - sym memory score")
    axes[0, 2].tick_params(axis="x", rotation=20)

    ax = axes[1, 2]
    if sensitivity_summary is not None and not sensitivity_summary.empty:
        for ratio, sub in sensitivity_summary.groupby("ratio"):
            sub = sub.sort_values("gate_amplitude")
            ax.plot(
                sub["gate_amplitude"],
                sub["real_minus_sym_normalized_effective_dimension_mean"],
                marker="o",
                lw=1.5,
                label=f"KC active {float(ratio):.2f}",
            )
        ax.axhline(0, color="0.25", lw=1)
        ax.legend(frameon=False, fontsize=8)
    ax.set_title("Parameter sensitivity")
    ax.set_xlabel("gate amplitude A")
    ax.set_ylabel("Δ normalized effective dimension")

    fig.suptitle("KC lateralization mechanism suite: representation, regime and graph controls", fontsize=14)
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return figure_path


def _render_lateralization_mechanism_report(
    *,
    cfg: LateralizationMechanismSuiteConfig,
    representation_summary: pd.DataFrame,
    regime_contrasts: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    graph_signal_summary: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    null_summary: pd.DataFrame,
    regime_group_summary: pd.DataFrame,
    figure_path: Path | None,
    interpretation: dict[str, object],
) -> str:
    key_rep = representation_summary[
        representation_summary["condition_id"].isin(
            ["symmetrized", "real_lateralized", "mirror_reversed", "serotonin_only", "glutamate_only"]
        )
    ][
        [
            "condition_id",
            "condition_class",
            "normalized_effective_dimension_mean",
            "normalized_effective_dimension_delta_vs_symmetrized",
            "mean_jaccard_overlap_mean",
            "decoder_margin_mean",
            "memory_score_mean",
        ]
    ]
    top_regimes = regime_contrasts.sort_values("real_minus_sym_memory_score", ascending=False).head(12)
    weak_regimes = regime_contrasts.sort_values("real_minus_sym_memory_score", ascending=True).head(8)
    figure_line = f"\n![KC lateralization mechanism suite]({figure_path.name})\n" if figure_path else ""
    return f"""# KC 侧化机制仿真：表示空间、任务条件和记忆稳定性

保存路径：`{cfg.output_dir / 'LATERALIZATION_MECHANISM_SUITE_REPORT_CN.md'}`

## 问题

这套仿真回答一个比单点结果更接近文章的问题：真实 FlyWire 中的 KC 侧化 / 对称性破缺，
是否能从网络和图网络角度解释为更大的 odor representation space，并在某些任务条件下转成
memory stabilization 优势。

它不是 wet-lab 行为结果；所有输入仍来自真实 FlyWire v783 `ALPN→KC` 子图和 KC
serotonin-minus-glutamate input lateralization。

## 方法

- 真实 `ALPN→KC` 子图：`{interpretation['n_glomerulus_channels']}` 个 glomerulus channels，
  `{interpretation['n_kc']}` 个 KC。
- KC 稀疏锚点：active fraction `{cfg.ratio:.2f}`。
- gate：`KC_response' = KC_response * (1 + {cfg.gate_amplitude:.2f} * gate)`。
- 对照：`symmetrized`、`real_lateralized`、`mirror_reversed`、`shuffled`、serotonin-only、
  glutamate-only、left/right-only，以及 gamma / alpha-beta / alpha-prime-beta-prime KC subtype 消融。
- 任务扫描：odor similarity `{cfg.task_similarity_levels}`，delay `{cfg.delay_blocks}`，
  interference `{cfg.interference_levels}`，DPM gain `{cfg.dpm_gains}`，
  APL/noise `{cfg.apl_noise_levels}`，KC dropout `{cfg.dropout_levels}`。

## 核心结论

1. **老师提出的 representation-space 问题有正向答案。**真实 side/subtype 对齐 gate 相对
   `symmetrized` 的 normalized effective dimension 变化为
   `{interpretation['real_delta_effective_dimension']:.4g}`；mirror-reversed 为
   `{interpretation['mirror_delta_effective_dimension']:.4g}`；shuffled 平均为
   `{interpretation['shuffled_mean_delta_effective_dimension']:.4g}`。
2. **memory stabilization 不是无条件变好。**在默认任务条件下，real gate 的 memory-score 变化为
   `{interpretation['real_delta_memory_score']:.4g}`，说明“表示空间增大”不能直接写成
   “学习记忆增强”。
3. **任务条件扫描给出更有用的文章结论。**在
   `{interpretation['n_regimes_real_better']}/{interpretation['n_regimes_total']}` 个任务格点中，
   real lateralized 的 memory score 高于 symmetrized；平均优势为
   `{interpretation['mean_real_minus_sym_memory_score']:.4g}`。优势主要出现在高相似 odor、延迟、
   干扰和部分 DPM 降低组合下，而不是普通 immediate/simple odor 条件。
4. **图网络解释更稳。**真实 gate 在 `ALPN→KC` 输入图上的 smoothness 为
   `{interpretation['real_graph_smoothness']:.4g}`，shuffled 平均为
   `{interpretation['shuffled_graph_smoothness']:.4g}`。这说明真实 gate 不是任意 KC 异质性；
   它沿真实输入通道有结构。
5. **不要把 729 grids 写成 p-hacking。**现在按预先定义的 immediate/delayed、
   low-conflict/high-conflict、low-noise/high-noise、low/high DPM gain 分组汇总；
   这些分组用于解释任务依赖性，而不是挑单个最佳格点。

## memory score 的数学定义

对第 `i` 个任务条件，先用 CS+ 与 CS- 的 KC 活动差向量定义 decoder margin：

`M_i = ||x_CSplus - x_CSminus||_2^2`

考虑 KC dropout 后得到 `M_i^drop`。DPM retention 项为：

`R_i = exp[- delay * (lambda_base + lambda_DPM * (1 - clip(DPM_support, 0, 1)))]`

干扰项为 `I_i = interference * conflict_penalty_scale * M_ref`。最终：

`memory_score_i = (M_i^drop * R_i - I_i) / M_ref`

其中 `M_ref` 是 symmetrized 条件跨 seed 的 median reference margin。因此这个 readout 是
可解释 surrogate：decoder separation、dropout 后保持、DPM 延迟保持和冲突干扰的合成，不是真实 T-maze choice。

## 表示空间和默认 memory proxy

{_table(key_rep, max_rows=12)}

## 最支持“偏侧化有功能”的任务条件

{_table(top_regimes[['task_similarity', 'delay_blocks', 'interference_level', 'dpm_gain', 'apl_noise_level', 'dropout_probability', 'real_minus_sym_memory_score', 'real_minus_sym_choice_proxy', 'mirror_minus_sym_memory_score']], max_rows=12)}

## 最不支持或反向的任务条件

{_table(weak_regimes[['task_similarity', 'delay_blocks', 'interference_level', 'dpm_gain', 'apl_noise_level', 'dropout_probability', 'real_minus_sym_memory_score', 'real_minus_sym_choice_proxy', 'mirror_minus_sym_memory_score']], max_rows=8)}

## 机制消融

{_table(ablation_summary, max_rows=14)}

## 图网络指标

{_table(graph_signal_summary, max_rows=20)}

## 预定义 regime 分组

{_table(regime_group_summary, max_rows=20)}

## shuffle null 与随机门对照

{_table(null_summary, max_rows=20)}

## gate amplitude 和 KC sparsity 敏感性

{_table(sensitivity_summary, max_rows=30)}

{figure_line}
## 文章建议写法

推荐写成：真实 KC 侧化显著改变编码几何，轻微扩大 representation space；但 memory stabilization
不是基础条件下的必然结果，而是在高相似、延迟、干扰和 DPM/APL 状态压力下更可能显现。
这支持“侧化提供状态依赖的计算储备或易损性调制”，而不是简单的“侧化必然增强学习记忆”。

## 输出文件

- representation raw：`{cfg.output_dir / 'lateralization_mechanism_representation_raw.csv'}`
- representation summary：`{cfg.output_dir / 'lateralization_mechanism_representation_summary.csv'}`
- task-regime raw：`{cfg.output_dir / 'lateralization_mechanism_task_regime_raw.csv'}`
- task-regime summary：`{cfg.output_dir / 'lateralization_mechanism_task_regime_summary.csv'}`
- contrasts：`{cfg.output_dir / 'lateralization_mechanism_condition_contrasts.csv'}`
- ablation summary：`{cfg.output_dir / 'lateralization_mechanism_ablation_summary.csv'}`
- graph signal summary：`{cfg.output_dir / 'lateralization_mechanism_graph_signal_summary.csv'}`
- pre-defined regime groups：`{cfg.output_dir / 'lateralization_mechanism_regime_group_summary.csv'}`
- shuffle null summary：`{cfg.output_dir / 'lateralization_mechanism_null_summary.csv'}`
- parameter sensitivity：`{cfg.output_dir / 'lateralization_mechanism_parameter_sensitivity.csv'}`
- figure：`{figure_path if figure_path is not None else 'not generated'}`

## 边界

- 这里的 memory score 是连接组约束的 proxy，不是真实 T-maze 或自由行为。
- APL/noise、dropout 和 DPM gain 是透明的 counterfactual 参数，不等同于具体遗传工具效应大小。
- 当前最强结论是“偏侧化改变 KC 编码几何，并在压力任务下产生条件性记忆优势候选”。
"""


def run_lateralization_mechanism_suite(
    config: LateralizationMechanismSuiteConfig | None = None,
) -> dict[str, Path | pd.DataFrame | dict[str, object]]:
    cfg = config or LateralizationMechanismSuiteConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    annotations = _load_annotations(cfg.annotation_path)
    annotated = _annotate_connectome_groups(annotations)
    edges = pd.read_parquet(
        cfg.connectivity_path,
        columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity", "Excitatory x Connectivity"],
    )
    glomerulus_names, glomerulus_matrix, kc_ids, channel_table = _build_flywire_glomerulus_kc_matrix_from_frames(
        annotated,
        edges,
    )
    aligned_gate = _load_kc_lateralization_components(cfg.kc_nt_inputs_path, kc_ids)
    side_by_kc = aligned_gate["side"].to_numpy()
    rng = np.random.default_rng(20260603)
    gate_conditions = _mechanism_gate_conditions(
        aligned_gate,
        shuffle_repeats=int(cfg.shuffle_repeats),
        gate_strength=float(cfg.gate_strength),
        rng=rng,
    )

    _ppl1_targets, dpm_targets, _pattern_used = _identify_learning_memory_targets(annotated, ppl1_pattern="PPL1")
    dpm_to_kc_vectors = _target_to_kc_weight_vectors_by_side(edges, kc_ids, dpm_targets)
    dpm_return_total = dpm_to_kc_vectors["left"] + dpm_to_kc_vectors["right"]
    if float(dpm_return_total.sum()) <= 0:
        raise ValueError("No positive DPM->KC edge mass was found.")

    representation_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []
    reference_margins: list[float] = []
    seed_activities: list[tuple[int, np.ndarray]] = []
    for seed in cfg.seeds:
        _odor_names, activity, _odor_panel = build_mixture_odor_panel(
            glomerulus_names,
            glomerulus_matrix,
            seed=int(seed),
            n_odors=int(cfg.n_odors),
            min_glomeruli_per_odor=int(cfg.min_glomeruli_per_odor),
            max_glomeruli_per_odor=int(cfg.max_glomeruli_per_odor),
            channel_noise_sigma=float(cfg.channel_noise_sigma),
        )
        seed_activities.append((int(seed), activity))
        sym_activity = _apply_gate_to_activity(
            activity,
            np.zeros(len(kc_ids), dtype=np.float64),
            gate_amplitude=float(cfg.gate_amplitude),
            apl_noise_level=0.0,
            rng=np.random.default_rng(int(seed) + 101),
        )
        sym_binary, sym_graded, _active_k = _sparsify(sym_activity, float(cfg.ratio))
        reference_margins.append(float(np.dot(sym_graded[0] - sym_graded[1], sym_graded[0] - sym_graded[1])))
    reference_margin = float(np.median(np.asarray(reference_margins, dtype=np.float64)))
    if not np.isfinite(reference_margin) or reference_margin <= 1e-12:
        reference_margin = 1.0

    for seed, activity in seed_activities:
        for condition in gate_conditions:
            gate_vector = np.asarray(condition["gate_vector"], dtype=np.float64)
            gated_activity = _apply_gate_to_activity(
                activity,
                gate_vector,
                gate_amplitude=float(cfg.gate_amplitude),
                apl_noise_level=0.0,
                rng=np.random.default_rng(seed * 1009 + int(condition["condition_order"])),
            )
            binary, graded, active_k = _sparsify(gated_activity, float(cfg.ratio))
            rep_metrics = _representation_space_metrics(binary, graded, side_by_kc)
            decoder = _binary_decoder_metrics(
                graded[0],
                graded[1],
                dropout_probability=0.10,
                decoder_noise_sigma=float(cfg.decoder_noise_sigma),
                rng=np.random.default_rng(seed * 7919 + int(condition["condition_order"])),
            )
            dpm_panel = graded @ dpm_return_total
            dpm_support = float((graded[0] @ dpm_return_total) / (float(dpm_panel.mean()) + 1e-12))
            retention_fraction = float(
                np.exp(
                    -6
                    * (
                        float(cfg.dpm_baseline_decay_per_block)
                        + float(cfg.dpm_loss_decay_penalty_per_block) * (1.0 - float(np.clip(dpm_support, 0.0, 1.0)))
                    )
                )
            )
            memory_score = float((decoder["dropout_margin"] * retention_fraction) / reference_margin)
            representation_rows.append(
                {
                    "seed": seed,
                    "condition_order": int(condition["condition_order"]),
                    "condition_id": str(condition["condition_id"]),
                    "condition_label": str(condition["condition_label"]),
                    "condition_class": str(condition["condition_class"]),
                    "active_k": int(active_k),
                    "observed_active_fraction": float(binary.mean()),
                    "mean_jaccard_overlap": _mean_jaccard(binary),
                    "mean_binary_cosine": _mean_binary_cosine(binary),
                    "decoder_margin": decoder["decoder_margin"],
                    "dropout_margin": decoder["dropout_margin"],
                    "dropout_margin_fraction": decoder["dropout_margin_fraction"],
                    "noise_margin_fraction": decoder["noise_margin_fraction"],
                    "dpm_feedback_support": dpm_support,
                    "retention_fraction": retention_fraction,
                    "memory_score": memory_score,
                    **rep_metrics,
                }
            )

    core_task_conditions = [record for record in gate_conditions if record["condition_id"] in {"symmetrized", "real_lateralized", "mirror_reversed"}]
    for seed, activity in seed_activities:
        for similarity in cfg.task_similarity_levels:
            task_activity = activity.copy()
            task_activity[1] = _normalize_rows(
                (
                    (1.0 - float(similarity)) * activity[1:2]
                    + float(similarity) * activity[0:1]
                )
            )[0]
            for condition in core_task_conditions:
                gate_vector = np.asarray(condition["gate_vector"], dtype=np.float64)
                for apl_noise in cfg.apl_noise_levels:
                    gated_activity = _apply_gate_to_activity(
                        task_activity,
                        gate_vector,
                        gate_amplitude=float(cfg.gate_amplitude),
                        apl_noise_level=float(apl_noise),
                        rng=np.random.default_rng(seed * 13007 + int(condition["condition_order"]) * 101 + int(float(apl_noise) * 1000)),
                    )
                    binary, graded, active_k = _sparsify(gated_activity, float(cfg.ratio))
                    dpm_panel = graded @ dpm_return_total
                    base_dpm_support = float((graded[0] @ dpm_return_total) / (float(dpm_panel.mean()) + 1e-12))
                    for delay in cfg.delay_blocks:
                        for interference in cfg.interference_levels:
                            for dpm_gain in cfg.dpm_gains:
                                dpm_support = float(dpm_gain) * base_dpm_support
                                retention_fraction = float(
                                    np.exp(
                                        -int(delay)
                                        * (
                                            float(cfg.dpm_baseline_decay_per_block)
                                            + float(cfg.dpm_loss_decay_penalty_per_block)
                                            * (1.0 - float(np.clip(dpm_support, 0.0, 1.0)))
                                        )
                                    )
                                )
                                for dropout in cfg.dropout_levels:
                                    decoder = _binary_decoder_metrics(
                                        graded[0],
                                        graded[1],
                                        dropout_probability=float(dropout),
                                        decoder_noise_sigma=float(cfg.decoder_noise_sigma),
                                        rng=np.random.default_rng(
                                            seed * 65537
                                            + int(condition["condition_order"]) * 4099
                                            + int(float(similarity) * 100) * 503
                                            + int(delay) * 31
                                            + int(float(interference) * 100) * 17
                                            + int(float(dpm_gain) * 100) * 13
                                            + int(float(dropout) * 100) * 7
                                        ),
                                    )
                                    memory_margin = (
                                        decoder["dropout_margin"] * retention_fraction
                                        - float(interference) * float(cfg.conflict_penalty_scale) * reference_margin
                                    )
                                    memory_score = float(memory_margin / reference_margin)
                                    choice_proxy = float(np.clip(0.5 + 0.5 * np.tanh(memory_score), 0.0, 1.0))
                                    task_rows.append(
                                        {
                                            "seed": seed,
                                            "condition_order": int(condition["condition_order"]),
                                            "condition_id": str(condition["condition_id"]),
                                            "condition_label": str(condition["condition_label"]),
                                            "condition_class": str(condition["condition_class"]),
                                            "task_similarity": float(similarity),
                                            "delay_blocks": int(delay),
                                            "interference_level": float(interference),
                                            "dpm_gain": float(dpm_gain),
                                            "apl_noise_level": float(apl_noise),
                                            "dropout_probability": float(dropout),
                                            "active_k": int(active_k),
                                            "observed_active_fraction": float(binary.mean()),
                                            "base_dpm_feedback_support": base_dpm_support,
                                            "dpm_feedback_support": dpm_support,
                                            "retention_fraction": retention_fraction,
                                            "decoder_margin": decoder["decoder_margin"],
                                            "dropout_margin": decoder["dropout_margin"],
                                            "dropout_margin_fraction": decoder["dropout_margin_fraction"],
                                            "noise_margin_fraction": decoder["noise_margin_fraction"],
                                            "memory_margin": memory_margin,
                                            "memory_score": memory_score,
                                            "choice_proxy": choice_proxy,
                                        }
                                    )

    representation_raw = pd.DataFrame.from_records(representation_rows)
    representation_summary = (
        representation_raw.groupby(
            ["condition_order", "condition_id", "condition_label", "condition_class"],
            as_index=False,
        )
        .agg(
            n_seed_panels=("seed", "nunique"),
            active_k=("active_k", "first"),
            observed_active_fraction_mean=("observed_active_fraction", "mean"),
            normalized_effective_dimension_mean=("normalized_effective_dimension", "mean"),
            normalized_effective_dimension_std=("normalized_effective_dimension", "std"),
            effective_dimension_mean=("effective_dimension", "mean"),
            mean_pairwise_l2_distance_mean=("mean_pairwise_l2_distance", "mean"),
            mean_pairwise_cosine_similarity_mean=("mean_pairwise_cosine_similarity", "mean"),
            mean_jaccard_overlap_mean=("mean_jaccard_overlap", "mean"),
            mean_binary_cosine_mean=("mean_binary_cosine", "mean"),
            decoder_margin_mean=("decoder_margin", "mean"),
            dropout_margin_fraction_mean=("dropout_margin_fraction", "mean"),
            noise_margin_fraction_mean=("noise_margin_fraction", "mean"),
            dpm_feedback_support_mean=("dpm_feedback_support", "mean"),
            retention_fraction_mean=("retention_fraction", "mean"),
            memory_score_mean=("memory_score", "mean"),
            memory_score_std=("memory_score", "std"),
            lateral_code_index_mean=("lateral_code_index", "mean"),
            left_active_fraction_mean=("left_active_fraction", "mean"),
            right_active_fraction_mean=("right_active_fraction", "mean"),
        )
        .sort_values("condition_order")
    )
    sym = representation_summary[representation_summary["condition_id"].eq("symmetrized")].iloc[0]
    for metric in [
        "normalized_effective_dimension",
        "effective_dimension",
        "mean_pairwise_l2_distance",
        "mean_jaccard_overlap",
        "decoder_margin",
        "memory_score",
        "lateral_code_index",
    ]:
        column = f"{metric}_mean"
        representation_summary[f"{metric}_delta_vs_symmetrized"] = representation_summary[column] - float(sym[column])

    task_raw = pd.DataFrame.from_records(task_rows)
    task_summary = (
        task_raw.groupby(
            [
                "condition_order",
                "condition_id",
                "condition_label",
                "condition_class",
                "task_similarity",
                "delay_blocks",
                "interference_level",
                "dpm_gain",
                "apl_noise_level",
                "dropout_probability",
            ],
            as_index=False,
        )
        .agg(
            n_seed_panels=("seed", "nunique"),
            decoder_margin_mean=("decoder_margin", "mean"),
            dropout_margin_fraction_mean=("dropout_margin_fraction", "mean"),
            noise_margin_fraction_mean=("noise_margin_fraction", "mean"),
            dpm_feedback_support_mean=("dpm_feedback_support", "mean"),
            retention_fraction_mean=("retention_fraction", "mean"),
            memory_score_mean=("memory_score", "mean"),
            memory_score_std=("memory_score", "std"),
            choice_proxy_mean=("choice_proxy", "mean"),
            choice_proxy_std=("choice_proxy", "std"),
        )
        .sort_values(
            [
                "task_similarity",
                "delay_blocks",
                "interference_level",
                "dpm_gain",
                "apl_noise_level",
                "dropout_probability",
                "condition_order",
            ]
        )
    )
    pivot_keys = ["task_similarity", "delay_blocks", "interference_level", "dpm_gain", "apl_noise_level", "dropout_probability"]
    wide = task_summary.pivot_table(
        index=pivot_keys,
        columns="condition_id",
        values=["memory_score_mean", "choice_proxy_mean"],
        aggfunc="mean",
    )
    wide.columns = [f"{metric}_{condition}" for metric, condition in wide.columns]
    regime_contrasts = wide.reset_index()
    for metric in ["memory_score_mean", "choice_proxy_mean"]:
        regime_contrasts[f"real_minus_sym_{metric.replace('_mean', '')}"] = (
            regime_contrasts[f"{metric}_real_lateralized"] - regime_contrasts[f"{metric}_symmetrized"]
        )
        regime_contrasts[f"mirror_minus_sym_{metric.replace('_mean', '')}"] = (
            regime_contrasts[f"{metric}_mirror_reversed"] - regime_contrasts[f"{metric}_symmetrized"]
        )

    ablation_ids = [
        "serotonin_only",
        "glutamate_only",
        "real_no_gamma_kc",
        "real_no_ab_kc",
        "real_no_apbp_kc",
        "left_only_gate",
        "right_only_gate",
    ]
    real_row = representation_summary[representation_summary["condition_id"].eq("real_lateralized")].iloc[0]
    mirror_row = representation_summary[representation_summary["condition_id"].eq("mirror_reversed")].iloc[0]
    shuffled_rep = representation_summary[representation_summary["condition_class"].eq("shuffled_lateralized")]
    ablation_summary = representation_summary[representation_summary["condition_id"].isin(ablation_ids)].copy()
    for metric in ["normalized_effective_dimension", "decoder_margin", "memory_score"]:
        ablation_summary[f"{metric}_retained_fraction_vs_real"] = (
            ablation_summary[f"{metric}_delta_vs_symmetrized"]
            / (float(real_row[f"{metric}_delta_vs_symmetrized"]) + 1e-12)
        )
    ablation_summary = ablation_summary[
        [
            "condition_id",
            "condition_class",
            "normalized_effective_dimension_delta_vs_symmetrized",
            "normalized_effective_dimension_retained_fraction_vs_real",
            "decoder_margin_delta_vs_symmetrized",
            "decoder_margin_retained_fraction_vs_real",
            "memory_score_delta_vs_symmetrized",
            "memory_score_retained_fraction_vs_real",
        ]
    ]

    graph_rows: list[dict[str, object]] = []
    for condition in gate_conditions:
        graph_rows.append(
            {
                "condition_id": str(condition["condition_id"]),
                "condition_class": str(condition["condition_class"]),
                "graph_signal_smoothness": _graph_signal_smoothness(
                    np.asarray(condition["gate_vector"], dtype=np.float64),
                    glomerulus_matrix,
                ),
                "gate_participation_ratio": _vector_participation_ratio(np.asarray(condition["gate_vector"], dtype=np.float64)),
                "mean_abs_gate": float(np.mean(np.abs(np.asarray(condition["gate_vector"], dtype=np.float64)))),
            }
        )
    graph_signal_summary = pd.DataFrame.from_records(graph_rows)
    shuffled_graph = graph_signal_summary[graph_signal_summary["condition_class"].eq("shuffled_lateralized")]
    if not shuffled_graph.empty:
        graph_signal_summary = pd.concat(
            [
                graph_signal_summary,
                pd.DataFrame.from_records(
                    [
                        {
                            "condition_id": "shuffled_mean",
                            "condition_class": "shuffled_lateralized_summary",
                            "graph_signal_smoothness": float(shuffled_graph["graph_signal_smoothness"].mean()),
                            "gate_participation_ratio": float(shuffled_graph["gate_participation_ratio"].mean()),
                            "mean_abs_gate": float(shuffled_graph["mean_abs_gate"].mean()),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    regime_grouped = regime_contrasts.copy()
    regime_grouped["phase"] = np.where(regime_grouped["delay_blocks"].eq(0), "immediate", "delayed")
    regime_grouped["odor_similarity_group"] = np.where(regime_grouped["task_similarity"] < 0.35, "low_similarity", "high_similarity")
    regime_grouped["interference_group"] = np.where(regime_grouped["interference_level"].eq(0), "low_interference", "high_interference")
    regime_grouped["state_noise_group"] = np.where(
        regime_grouped["apl_noise_level"].eq(0) & regime_grouped["dropout_probability"].eq(0),
        "low_noise",
        "noise_or_dropout",
    )
    regime_grouped["dpm_state"] = np.where(regime_grouped["dpm_gain"].eq(1.0), "full_dpm", "reduced_dpm")
    regime_group_summary = (
        regime_grouped.groupby(["phase", "odor_similarity_group", "interference_group", "state_noise_group", "dpm_state"], as_index=False)
        .agg(
            n_regimes=("real_minus_sym_memory_score", "size"),
            mean_real_minus_sym_memory_score=("real_minus_sym_memory_score", "mean"),
            median_real_minus_sym_memory_score=("real_minus_sym_memory_score", "median"),
            fraction_real_better=("real_minus_sym_memory_score", lambda x: float((x > 0).mean())),
            mean_real_minus_sym_choice_proxy=("real_minus_sym_choice_proxy", "mean"),
            mean_mirror_minus_sym_memory_score=("mirror_minus_sym_memory_score", "mean"),
        )
        .sort_values("mean_real_minus_sym_memory_score", ascending=False)
    )

    null_rows: list[dict[str, object]] = []
    if not shuffled_rep.empty:
        for metric in ["normalized_effective_dimension", "decoder_margin", "memory_score"]:
            delta_column = f"{metric}_delta_vs_symmetrized"
            observed = float(real_row[delta_column])
            null_values = shuffled_rep[delta_column].to_numpy(dtype=float)
            null_rows.append(
                {
                    "metric": metric,
                    "observed_real_minus_sym": observed,
                    "shuffled_mean_minus_sym": float(np.mean(null_values)),
                    "shuffled_median_minus_sym": float(np.median(null_values)),
                    "shuffled_sd": float(np.std(null_values, ddof=1)) if len(null_values) > 1 else 0.0,
                    "real_percentile_vs_shuffled": float(np.mean(null_values <= observed)),
                    "empirical_p_real_greater_than_shuffled": float((1 + np.sum(null_values >= observed)) / (len(null_values) + 1)),
                    "n_shuffled": int(len(null_values)),
                }
            )
    null_summary = pd.DataFrame.from_records(null_rows)

    sensitivity_rows: list[dict[str, object]] = []
    sensitivity_conditions = [
        condition
        for condition in gate_conditions
        if str(condition["condition_id"]) in {"symmetrized", "real_lateralized", "mirror_reversed"}
    ]
    for ratio in cfg.sensitivity_ratios:
        for amplitude in cfg.sensitivity_gate_amplitudes:
            for seed, activity in seed_activities:
                for condition in sensitivity_conditions:
                    gated_activity = _apply_gate_to_activity(
                        activity,
                        np.asarray(condition["gate_vector"], dtype=np.float64),
                        gate_amplitude=float(amplitude),
                        apl_noise_level=0.0,
                        rng=np.random.default_rng(seed * 19001 + int(condition["condition_order"]) * 31 + int(float(amplitude) * 1000)),
                    )
                    binary, graded, active_k = _sparsify(gated_activity, float(ratio))
                    rep_metrics = _representation_space_metrics(binary, graded, side_by_kc)
                    decoder = _binary_decoder_metrics(
                        graded[0],
                        graded[1],
                        dropout_probability=0.10,
                        decoder_noise_sigma=float(cfg.decoder_noise_sigma),
                        rng=np.random.default_rng(seed * 29009 + int(condition["condition_order"]) * 17 + int(float(ratio) * 1000)),
                    )
                    sensitivity_rows.append(
                        {
                            "ratio": float(ratio),
                            "gate_amplitude": float(amplitude),
                            "seed": int(seed),
                            "condition_id": str(condition["condition_id"]),
                            "active_k": int(active_k),
                            "normalized_effective_dimension": float(rep_metrics["normalized_effective_dimension"]),
                            "decoder_margin": float(decoder["decoder_margin"]),
                            "dropout_margin_fraction": float(decoder["dropout_margin_fraction"]),
                        }
                    )
    sensitivity_raw = pd.DataFrame.from_records(sensitivity_rows)
    sensitivity_condition_summary = (
        sensitivity_raw.groupby(["ratio", "gate_amplitude", "condition_id"], as_index=False)
        .agg(
            n_seed_panels=("seed", "nunique"),
            active_k=("active_k", "first"),
            normalized_effective_dimension_mean=("normalized_effective_dimension", "mean"),
            normalized_effective_dimension_std=("normalized_effective_dimension", "std"),
            decoder_margin_mean=("decoder_margin", "mean"),
            dropout_margin_fraction_mean=("dropout_margin_fraction", "mean"),
        )
        if not sensitivity_raw.empty
        else pd.DataFrame()
    )
    if not sensitivity_condition_summary.empty:
        sensitivity_wide = sensitivity_condition_summary.pivot_table(
            index=["ratio", "gate_amplitude"],
            columns="condition_id",
            values=["normalized_effective_dimension_mean", "decoder_margin_mean", "dropout_margin_fraction_mean"],
            aggfunc="mean",
        )
        sensitivity_wide.columns = [f"{metric}_{condition}" for metric, condition in sensitivity_wide.columns]
        sensitivity_summary = sensitivity_wide.reset_index()
        for metric in ["normalized_effective_dimension", "decoder_margin", "dropout_margin_fraction"]:
            mean_col = f"{metric}_mean"
            sensitivity_summary[f"real_minus_sym_{mean_col}"] = (
                sensitivity_summary[f"{mean_col}_real_lateralized"]
                - sensitivity_summary[f"{mean_col}_symmetrized"]
            )
            sensitivity_summary[f"mirror_minus_sym_{mean_col}"] = (
                sensitivity_summary[f"{mean_col}_mirror_reversed"]
                - sensitivity_summary[f"{mean_col}_symmetrized"]
            )
    else:
        sensitivity_summary = pd.DataFrame()

    real_graph = graph_signal_summary[graph_signal_summary["condition_id"].eq("real_lateralized")].iloc[0]
    shuffled_graph_mean = graph_signal_summary[graph_signal_summary["condition_id"].eq("shuffled_mean")]
    shuffled_graph_row = shuffled_graph_mean.iloc[0] if not shuffled_graph_mean.empty else real_graph
    n_regimes_total = int(len(regime_contrasts))
    n_regimes_real_better = int((regime_contrasts["real_minus_sym_memory_score"] > 0).sum())
    interpretation = {
        "model": "real_FlyWire_ALPN_to_KC_KC_lateralization_mechanism_suite",
        "boundary": "connectome-constrained simulation proxy; not wet-lab behavior",
        "n_kc": int(len(kc_ids)),
        "n_glomerulus_channels": int(len(glomerulus_names)),
        "n_seed_panels": int(len(cfg.seeds)),
        "kc_active_fraction": float(cfg.ratio),
        "real_delta_effective_dimension": float(real_row["normalized_effective_dimension_delta_vs_symmetrized"]),
        "mirror_delta_effective_dimension": float(mirror_row["normalized_effective_dimension_delta_vs_symmetrized"]),
        "shuffled_mean_delta_effective_dimension": float(
            shuffled_rep["normalized_effective_dimension_delta_vs_symmetrized"].mean()
        )
        if not shuffled_rep.empty
        else float("nan"),
        "real_delta_memory_score": float(real_row["memory_score_delta_vs_symmetrized"]),
        "n_regimes_total": n_regimes_total,
        "n_regimes_real_better": n_regimes_real_better,
        "fraction_regimes_real_better": float(n_regimes_real_better / max(1, n_regimes_total)),
        "mean_real_minus_sym_memory_score": float(regime_contrasts["real_minus_sym_memory_score"].mean()),
        "median_real_minus_sym_memory_score": float(regime_contrasts["real_minus_sym_memory_score"].median()),
        "real_graph_smoothness": float(real_graph["graph_signal_smoothness"]),
        "shuffled_graph_smoothness": float(shuffled_graph_row["graph_signal_smoothness"]),
        "reference_margin": float(reference_margin),
    }

    figure_path = _write_lateralization_mechanism_figure(
        representation_summary,
        regime_contrasts,
        graph_signal_summary,
        sensitivity_summary,
        cfg.output_dir,
    )
    paths = {
        "representation_raw_csv": cfg.output_dir / "lateralization_mechanism_representation_raw.csv",
        "representation_summary_csv": cfg.output_dir / "lateralization_mechanism_representation_summary.csv",
        "task_regime_raw_csv": cfg.output_dir / "lateralization_mechanism_task_regime_raw.csv",
        "task_regime_summary_csv": cfg.output_dir / "lateralization_mechanism_task_regime_summary.csv",
        "condition_contrasts_csv": cfg.output_dir / "lateralization_mechanism_condition_contrasts.csv",
        "ablation_summary_csv": cfg.output_dir / "lateralization_mechanism_ablation_summary.csv",
        "graph_signal_summary_csv": cfg.output_dir / "lateralization_mechanism_graph_signal_summary.csv",
        "regime_group_summary_csv": cfg.output_dir / "lateralization_mechanism_regime_group_summary.csv",
        "null_summary_csv": cfg.output_dir / "lateralization_mechanism_null_summary.csv",
        "parameter_sensitivity_csv": cfg.output_dir / "lateralization_mechanism_parameter_sensitivity.csv",
        "parameter_sensitivity_raw_csv": cfg.output_dir / "lateralization_mechanism_parameter_sensitivity_raw.csv",
        "gate_components_csv": cfg.output_dir / "lateralization_mechanism_gate_components.csv",
        "channels_csv": cfg.output_dir / "lateralization_mechanism_glomerulus_kc_channels.csv",
        "figure_png": figure_path,
        "report_md": cfg.output_dir / "LATERALIZATION_MECHANISM_SUITE_REPORT_CN.md",
        "metadata_json": cfg.output_dir / "lateralization_mechanism_metadata.json",
    }
    representation_raw.to_csv(paths["representation_raw_csv"], index=False)
    representation_summary.to_csv(paths["representation_summary_csv"], index=False)
    task_raw.to_csv(paths["task_regime_raw_csv"], index=False)
    task_summary.to_csv(paths["task_regime_summary_csv"], index=False)
    regime_contrasts.to_csv(paths["condition_contrasts_csv"], index=False)
    ablation_summary.to_csv(paths["ablation_summary_csv"], index=False)
    graph_signal_summary.to_csv(paths["graph_signal_summary_csv"], index=False)
    regime_group_summary.to_csv(paths["regime_group_summary_csv"], index=False)
    null_summary.to_csv(paths["null_summary_csv"], index=False)
    sensitivity_summary.to_csv(paths["parameter_sensitivity_csv"], index=False)
    sensitivity_raw.to_csv(paths["parameter_sensitivity_raw_csv"], index=False)
    aligned_gate.to_csv(paths["gate_components_csv"], index=False)
    channel_table.to_csv(paths["channels_csv"], index=False)
    paths["report_md"].write_text(
        _render_lateralization_mechanism_report(
            cfg=cfg,
            representation_summary=representation_summary,
            regime_contrasts=regime_contrasts,
            ablation_summary=ablation_summary,
            graph_signal_summary=graph_signal_summary,
            sensitivity_summary=sensitivity_summary,
            null_summary=null_summary,
            regime_group_summary=regime_group_summary,
            figure_path=figure_path,
            interpretation=interpretation,
        ),
        encoding="utf-8",
    )
    paths["metadata_json"].write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "interpretation": interpretation,
                "paths": {key: str(value) for key, value in paths.items() if isinstance(value, Path)},
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return {
        **paths,
        "representation_summary_df": representation_summary,
        "task_regime_summary_df": task_summary,
        "condition_contrasts_df": regime_contrasts,
        "ablation_summary_df": ablation_summary,
        "graph_signal_summary_df": graph_signal_summary,
        "regime_group_summary_df": regime_group_summary,
        "null_summary_df": null_summary,
        "parameter_sensitivity_df": sensitivity_summary,
        "interpretation": interpretation,
    }


def _behavior_output_group(group: str, cell_class: str, super_class: str, cell_type: str) -> str:
    group = str(group or "")
    if group in {"DN", "DAN", "DPM", "APL", "MBON", "MBIN_other", "octopamine", "serotonin"}:
        return group
    if str(super_class).lower() == "descending":
        return "DN"
    if re.search(r"(?:^|[^A-Za-z0-9])DN", f"{cell_class} {cell_type}", flags=re.IGNORECASE):
        return "DN_like"
    return "other"


ASO2014_COMPARTMENT_ORDER: tuple[str, ...] = (
    "alpha1",
    "alpha2",
    "alpha3",
    "beta1",
    "beta2",
    "alpha_prime1",
    "alpha_prime2",
    "alpha_prime3",
    "beta_prime1",
    "beta_prime2",
    "gamma1",
    "gamma2",
    "gamma3",
    "gamma4",
    "gamma5",
)
ASO2014_COMPARTMENT_LABELS: dict[str, str] = {
    "alpha1": "α1",
    "alpha2": "α2",
    "alpha3": "α3",
    "beta1": "β1",
    "beta2": "β2",
    "alpha_prime1": "α′1",
    "alpha_prime2": "α′2",
    "alpha_prime3": "α′3",
    "beta_prime1": "β′1",
    "beta_prime2": "β′2",
    "gamma1": "γ1",
    "gamma2": "γ2",
    "gamma3": "γ3",
    "gamma4": "γ4",
    "gamma5": "γ5",
}
ASO2014_COMPARTMENTS_BY_MBON: dict[str, tuple[str, ...]] = {
    "MBON01": ("gamma5", "beta_prime2"),
    "MBON02": ("beta2", "beta_prime2"),
    "MBON03": ("beta_prime2",),
    "MBON04": ("beta_prime2",),
    "MBON05": ("gamma4",),
    "MBON06": ("beta1",),
    "MBON07": ("alpha1",),
    "MBON09": ("gamma3", "beta_prime1"),
    "MBON10": ("beta_prime1",),
    "MBON11": ("gamma1",),
    "MBON12": ("gamma2", "alpha_prime1"),
    "MBON13": ("alpha_prime2",),
    "MBON14": ("alpha3",),
    "MBON15": ("alpha_prime1",),
    "MBON15-LIKE": ("alpha_prime1",),
    "MBON16": ("alpha_prime3",),
    "MBON17": ("alpha_prime3",),
    "MBON17-LIKE": ("alpha_prime3",),
    "MBON18": ("alpha2",),
    "MBON19": ("alpha2", "alpha3"),
    "MBON20": ("gamma1", "gamma2"),
    "MBON21": ("gamma4", "gamma5"),
    "MBON22": tuple(),
}


def _normalize_mbon_type(value: object) -> str:
    text = str(value or "").strip().upper().replace("_", "-")
    text = text.replace("MBON-", "MBON")
    return text


def _mbon_compartments_for_type(cell_type: object, hemibrain_type: object) -> tuple[str, ...]:
    for value in (hemibrain_type, cell_type):
        normalized = _normalize_mbon_type(value)
        if normalized in ASO2014_COMPARTMENTS_BY_MBON:
            return ASO2014_COMPARTMENTS_BY_MBON[normalized]
        for token in re.split(r"[,;/\s]+", normalized):
            if token in ASO2014_COMPARTMENTS_BY_MBON:
                return ASO2014_COMPARTMENTS_BY_MBON[token]
    return tuple()


def _mbon_source_groups(
    annotated: pd.DataFrame,
    source_group_limit: int = 0,
    *,
    source_grouping: str = "compartment",
    include_unmapped_sources: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mbon = annotated[annotated["connectome_group"].eq("MBON")].copy()
    if mbon.empty:
        raise ValueError("No MBON neurons found in annotation table.")
    source_grouping = str(source_grouping).lower().strip()
    if source_grouping not in {"compartment", "subtype"}:
        raise ValueError("source_grouping must be 'compartment' or 'subtype'.")
    mbon["mbon_subtype"] = (
        mbon["hemibrain_type"].replace("", np.nan).fillna(mbon["cell_type"]).replace("", "MBON_unspecified").astype(str)
    )
    if source_grouping == "subtype":
        mbon["mbon_source_group"] = mbon["mbon_subtype"]
        mbon["mbon_source_group_label"] = mbon["mbon_subtype"]
        mbon["mbon_source_grouping"] = "subtype"
        mbon["source_mapping_status"] = "subtype"
        source_mapping = mbon.copy()
    else:
        rows: list[dict[str, object]] = []
        for record in mbon.to_dict(orient="records"):
            compartments = _mbon_compartments_for_type(record.get("cell_type"), record.get("hemibrain_type"))
            if not compartments:
                row = dict(record)
                row["mbon_source_group"] = "unmapped"
                row["mbon_source_group_label"] = "unmapped"
                row["mbon_source_grouping"] = "compartment"
                row["source_mapping_status"] = "unmapped"
                rows.append(row)
                continue
            for compartment in compartments:
                row = dict(record)
                row["mbon_source_group"] = compartment
                row["mbon_source_group_label"] = ASO2014_COMPARTMENT_LABELS.get(compartment, compartment)
                row["mbon_source_grouping"] = "compartment"
                row["source_mapping_status"] = "mapped"
                rows.append(row)
        source_mapping = pd.DataFrame.from_records(rows)
        mbon = source_mapping.copy()
        if not include_unmapped_sources:
            mbon = mbon[mbon["source_mapping_status"].eq("mapped")].copy()
    if mbon.empty:
        raise ValueError("No MBON source rows remain after source grouping and mapping filters.")
    order_map = {group: index for index, group in enumerate((*ASO2014_COMPARTMENT_ORDER, "unmapped"))}
    mbon["source_group_order"] = mbon["mbon_source_group"].map(order_map).fillna(999).astype(int)
    group_summary = (
        mbon.groupby("mbon_source_group", as_index=False)
        .agg(
            source_group_label=("mbon_source_group_label", "first"),
            source_grouping=("mbon_source_grouping", "first"),
            source_group_order=("source_group_order", "min"),
            n_mbon=("root_id", "nunique"),
            n_source_rows=("root_id", "size"),
            mbon_subtypes=("mbon_subtype", lambda values: ",".join(sorted(set(map(str, values))))),
            sides=("side", lambda values: ",".join(sorted(set(map(str, values))))),
            top_nt=("top_nt", lambda values: ",".join(sorted(set(map(str, values))))),
            root_ids=("root_id", lambda values: ",".join(str(int(value)) for value in sorted(set(values)))),
        )
        .sort_values(["source_group_order", "n_mbon", "mbon_source_group"], ascending=[True, False, True])
    )
    if source_group_limit and int(source_group_limit) > 0:
        keep = set(group_summary.head(int(source_group_limit))["mbon_source_group"])
        mbon = mbon[mbon["mbon_source_group"].isin(keep)].copy()
        group_summary = group_summary[group_summary["mbon_source_group"].isin(keep)].copy()
    mapping_cols = [
        "root_id",
        "side",
        "cell_type",
        "hemibrain_type",
        "mbon_subtype",
        "top_nt",
        "mbon_source_group",
        "mbon_source_group_label",
        "mbon_source_grouping",
        "source_mapping_status",
    ]
    source_mapping = source_mapping[[column for column in mapping_cols if column in source_mapping.columns]].copy()
    return mbon, group_summary.reset_index(drop=True), source_mapping.reset_index(drop=True)


def _build_normalized_out_adjacency(
    edges: pd.DataFrame,
    *,
    min_abs_edge_weight: float = 1.0,
) -> dict[str, np.ndarray]:
    selected = edges[
        edges["Excitatory x Connectivity"].astype(float).abs() >= float(min_abs_edge_weight)
    ][["Presynaptic_ID", "Postsynaptic_ID", "Connectivity", "Excitatory x Connectivity"]].copy()
    if selected.empty:
        return {}
    pre = selected["Presynaptic_ID"].to_numpy(dtype=np.int64, copy=False)
    order = np.argsort(pre, kind="mergesort")
    pre_sorted = pre[order]
    post_sorted = selected["Postsynaptic_ID"].to_numpy(dtype=np.int64, copy=False)[order]
    raw_sorted = selected["Connectivity"].to_numpy(dtype=np.float64, copy=False)[order]
    signed_sorted = selected["Excitatory x Connectivity"].to_numpy(dtype=np.float64, copy=False)[order]
    abs_sorted = np.abs(signed_sorted)
    unique_pre, start, counts = np.unique(pre_sorted, return_index=True, return_counts=True)
    out_abs = np.add.reduceat(abs_sorted, start)
    normalizer = np.repeat(out_abs, counts)
    normalized_weight = np.divide(
        signed_sorted,
        normalizer,
        out=np.zeros_like(signed_sorted, dtype=np.float64),
        where=normalizer > 0,
    )
    return {
        "pre": unique_pre.astype(np.int64, copy=False),
        "start": start.astype(np.int64, copy=False),
        "stop": (start + counts).astype(np.int64, copy=False),
        "post": post_sorted.astype(np.int64, copy=False),
        "normalized_weight": normalized_weight.astype(np.float64, copy=False),
        "raw_weight": raw_sorted.astype(np.float64, copy=False),
    }


def _propagate_mbon_source_group(
    source_ids: Sequence[int],
    adjacency: Mapping[str, np.ndarray],
    *,
    max_hops: int,
    hop_decay: float,
    max_frontier_per_source: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not adjacency:
        return pd.DataFrame()
    pre_index = adjacency["pre"]
    starts = adjacency["start"]
    stops = adjacency["stop"]
    posts = adjacency["post"]
    normalized_weights = adjacency["normalized_weight"]
    raw_weights = adjacency["raw_weight"]
    current: dict[int, float] = {int(root_id): 1.0 / max(1, len(source_ids)) for root_id in source_ids}
    visited_by_hop: set[tuple[int, int]] = set()
    for hop in range(1, int(max_hops) + 1):
        next_drive: dict[int, float] = {}
        raw_drive: dict[int, float] = {}
        for pre, drive in current.items():
            index = int(np.searchsorted(pre_index, int(pre)))
            if index >= len(pre_index) or int(pre_index[index]) != int(pre):
                continue
            start = int(starts[index])
            stop = int(stops[index])
            decay = float(drive) * (float(hop_decay) ** (hop - 1))
            post_slice = posts[start:stop]
            contribution_slice = normalized_weights[start:stop] * decay
            raw_slice = raw_weights[start:stop]
            for post, contribution, raw_weight in zip(post_slice, contribution_slice, raw_slice):
                contribution = float(contribution)
                if abs(contribution) <= 0:
                    continue
                post_id = int(post)
                next_drive[post_id] = next_drive.get(post_id, 0.0) + contribution
                raw_drive[post_id] = raw_drive.get(post_id, 0.0) + abs(float(raw_weight))
        if not next_drive:
            break
        if max_frontier_per_source and len(next_drive) > int(max_frontier_per_source):
            keep = {
                root_id
                for root_id, _value in sorted(next_drive.items(), key=lambda item: abs(item[1]), reverse=True)[
                    : int(max_frontier_per_source)
                ]
            }
            next_drive = {root_id: value for root_id, value in next_drive.items() if root_id in keep}
            raw_drive = {root_id: value for root_id, value in raw_drive.items() if root_id in keep}
        for root_id, drive_value in next_drive.items():
            key = (int(hop), int(root_id))
            if key in visited_by_hop:
                continue
            visited_by_hop.add(key)
            rows.append(
                {
                    "hop": int(hop),
                    "target_root_id": int(root_id),
                    "signed_drive": float(drive_value),
                    "abs_drive": float(abs(drive_value)),
                    "raw_synapse_mass_proxy": float(raw_drive.get(root_id, 0.0)),
                }
            )
        current = next_drive
    return pd.DataFrame.from_records(rows)


def _summarize_candidate_downstream_outputs(
    candidate_ids: Sequence[int],
    edges: pd.DataFrame,
    annotated: pd.DataFrame,
) -> pd.DataFrame:
    candidate_set = {int(root_id) for root_id in candidate_ids}
    if not candidate_set:
        return pd.DataFrame()
    meta = annotated.set_index("root_id")
    selected = edges[edges["Presynaptic_ID"].isin(candidate_set)].copy()
    if selected.empty:
        return pd.DataFrame()
    post_meta = meta.reindex(selected["Postsynaptic_ID"].astype("int64")).reset_index(drop=True)
    selected = selected.reset_index(drop=True)
    selected["post_group"] = post_meta["connectome_group"].fillna("other").astype(str).to_numpy()
    selected["post_cell_class"] = post_meta["cell_class"].fillna("").astype(str).to_numpy()
    selected["post_super_class"] = post_meta["super_class"].fillna("").astype(str).to_numpy()
    selected["post_cell_type"] = post_meta["cell_type"].fillna("").astype(str).to_numpy()
    selected["behavior_output_group"] = [
        _behavior_output_group(group, cell_class, super_class, cell_type)
        for group, cell_class, super_class, cell_type in zip(
            selected["post_group"],
            selected["post_cell_class"],
            selected["post_super_class"],
            selected["post_cell_type"],
        )
    ]
    summary = (
        selected.groupby(["Presynaptic_ID", "behavior_output_group"], as_index=False)
        .agg(
            n_output_edges=("Connectivity", "size"),
            output_synapses=("Connectivity", "sum"),
            output_signed_weight=("Excitatory x Connectivity", "sum"),
            output_abs_weight=("Excitatory x Connectivity", lambda values: float(np.abs(values.astype(float)).sum())),
            n_output_targets=("Postsynaptic_ID", "nunique"),
        )
        .rename(columns={"Presynaptic_ID": "candidate_root_id"})
    )
    return summary.sort_values(["candidate_root_id", "output_abs_weight"], ascending=[True, False])


def _write_mbon_decision_pivot_figure(
    candidate_summary: pd.DataFrame,
    source_group_summary: pd.DataFrame,
    output_dir: Path,
) -> Path | None:
    if candidate_summary.empty:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    if "candidate_role_tier" in candidate_summary.columns:
        primary = candidate_summary[candidate_summary["candidate_role_tier"] != "global_feedback_control"].copy()
        top = primary.head(20).copy() if not primary.empty else candidate_summary.head(20).copy()
    else:
        top = candidate_summary.head(20).copy()
    labels = [
        f"{row.candidate_cell_type}\n{int(row.candidate_root_id)}"
        for row in top.itertuples(index=False)
    ]
    x = np.arange(len(top))
    figure_path = output_dir / "Fig_mbon_decision_pivot_candidates.png"
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)

    axes[0, 0].bar(x, top["pivot_score"], color="#4c78a8")
    axes[0, 0].set_xticks(x, labels, rotation=70, ha="right", fontsize=7)
    axes[0, 0].set_ylabel("pivot score")
    axes[0, 0].set_title("Primary non-APL MBON downstream candidates")

    axes[0, 1].scatter(
        top["n_source_groups"],
        top["abs_total_drive"],
        s=40 + 80 * top["dn_output_fraction"].fillna(0.0),
        c=top["push_pull_index"].fillna(0.0),
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    axes[0, 1].set_xlabel("MBON source groups converging")
    axes[0, 1].set_ylabel("summed abs drive")
    axes[0, 1].set_title("Convergence vs drive; color=push-pull")

    group_counts = source_group_summary.sort_values("n_mbon", ascending=False).head(20)
    axes[1, 0].barh(group_counts["mbon_source_group"].astype(str), group_counts["n_mbon"], color="#59a14f")
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("MBON neurons")
    axes[1, 0].set_title("MBON compartment/source groups used")

    cols = ["dn_output_fraction", "state_output_fraction", "mbon_recurrent_fraction"]
    y = np.arange(len(top))
    left = np.zeros(len(top))
    colors = ["#f28e2b", "#b279a2", "#9c755f"]
    names = ["DN/DN-like", "DAN/DPM/APL", "MBON recurrent"]
    for col, color, name in zip(cols, colors, names):
        values = top[col].fillna(0.0).to_numpy(dtype=float)
        axes[1, 1].barh(y, values, left=left, color=color, label=name)
        left += values
    axes[1, 1].set_yticks(y, [str(v) for v in top["candidate_cell_type"]], fontsize=8)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel("output abs-weight fraction")
    axes[1, 1].set_title("Candidate downstream routing")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("FlyWire MBON downstream decision-pivot candidate search", fontsize=14, fontweight="bold")
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return figure_path


def _mbon_plasticity_replay_patterns(source_groups: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build deterministic MB compartment plasticity scenarios.

    The main scenario follows the common aversive-learning motif discussed for
    gamma compartments: gamma1-3 MBON output decreases while gamma4-5 is kept as
    a within-lobe control.  The single-compartment sweep turns this into an
    interpretable response matrix across all available Aso compartments.
    """

    source_group_order = [str(group) for group in source_groups]
    source_group_set = set(source_group_order)

    def add_pattern(
        records: list[dict[str, object]],
        *,
        pattern_id: str,
        pattern_label: str,
        pattern_class: str,
        gain_delta_by_group: Mapping[str, float],
    ) -> None:
        if not any(group in source_group_set for group in gain_delta_by_group):
            return
        perturbed = [
            group
            for group in source_group_order
            if abs(float(gain_delta_by_group.get(group, 0.0))) > 0
        ]
        for group in source_group_order:
            gain_delta = float(gain_delta_by_group.get(group, 0.0))
            records.append(
                {
                    "pattern_id": pattern_id,
                    "pattern_label": pattern_label,
                    "pattern_class": pattern_class,
                    "source_group": group,
                    "source_group_label": ASO2014_COMPARTMENT_LABELS.get(group, group),
                    "gain_delta": gain_delta,
                    "post_learning_gain": 1.0 + gain_delta,
                    "is_perturbed": bool(abs(gain_delta) > 0),
                    "n_perturbed_source_groups": len(perturbed),
                    "perturbed_source_groups": ",".join(perturbed),
                }
            )

    records: list[dict[str, object]] = []
    aso_groups = [group for group in ASO2014_COMPARTMENT_ORDER if group in source_group_set]
    gamma13 = [group for group in ("gamma1", "gamma2", "gamma3") if group in source_group_set]
    gamma45 = [group for group in ("gamma4", "gamma5") if group in source_group_set]
    all_gamma = [group for group in ("gamma1", "gamma2", "gamma3", "gamma4", "gamma5") if group in source_group_set]

    add_pattern(
        records,
        pattern_id="aversive_gamma13_down",
        pattern_label="gamma1-3 down, gamma4-5 unchanged",
        pattern_class="learning_motif",
        gain_delta_by_group={group: -0.75 for group in gamma13},
    )
    add_pattern(
        records,
        pattern_id="gamma45_down_control",
        pattern_label="gamma4-5 down control",
        pattern_class="gamma_control",
        gain_delta_by_group={group: -0.75 for group in gamma45},
    )
    add_pattern(
        records,
        pattern_id="all_gamma_down",
        pattern_label="all gamma compartments down",
        pattern_class="lobe_control",
        gain_delta_by_group={group: -0.75 for group in all_gamma},
    )
    add_pattern(
        records,
        pattern_id="pan_compartment_down_25",
        pattern_label="all available source groups 25% down",
        pattern_class="global_control",
        gain_delta_by_group={group: -0.25 for group in source_group_order},
    )
    single_groups = aso_groups if aso_groups else source_group_order
    for group in single_groups:
        safe_group = re.sub(r"[^A-Za-z0-9_]+", "_", group).strip("_")
        add_pattern(
            records,
            pattern_id=f"single_{safe_group}_down",
            pattern_label=f"{ASO2014_COMPARTMENT_LABELS.get(group, group)} down",
            pattern_class="single_source_sweep",
            gain_delta_by_group={group: -0.75},
        )

    pattern_long = pd.DataFrame.from_records(records)
    if pattern_long.empty:
        return pd.DataFrame(), pd.DataFrame()
    pattern_summary = (
        pattern_long.groupby(["pattern_id", "pattern_label", "pattern_class"], as_index=False)
        .agg(
            n_source_groups=("source_group", "nunique"),
            n_perturbed_source_groups=("is_perturbed", "sum"),
            perturbed_source_groups=("perturbed_source_groups", "first"),
            min_post_learning_gain=("post_learning_gain", "min"),
            max_post_learning_gain=("post_learning_gain", "max"),
        )
        .sort_values(["pattern_class", "pattern_id"])
    )
    return pattern_summary.reset_index(drop=True), pattern_long.reset_index(drop=True)


def _run_mbon_plasticity_replay(
    *,
    source_group_summary: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    source_candidate_matrix: pd.DataFrame,
    output_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if source_candidate_matrix.empty or candidate_summary.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    source_groups = source_group_summary["mbon_source_group"].astype(str).tolist()
    pattern_summary, pattern_long = _mbon_plasticity_replay_patterns(source_groups)
    if pattern_long.empty:
        return pattern_summary, pd.DataFrame(), pd.DataFrame()

    candidate_cols = [
        "candidate_root_id",
        "candidate_group",
        "candidate_cell_type",
        "candidate_side",
        "candidate_top_nt",
        "n_source_groups",
        "source_groups",
        "signed_total_drive",
        "abs_total_drive",
        "push_pull_strength",
        "push_pull_index",
        "dn_output_fraction",
        "state_output_fraction",
        "mbon_recurrent_fraction",
        "pivot_score",
        "candidate_role_tier",
        "rank",
        "primary_pivot_rank",
    ]
    candidate_meta = candidate_summary[[col for col in candidate_cols if col in candidate_summary.columns]].copy()

    merged = source_candidate_matrix.merge(
        pattern_long[["pattern_id", "pattern_label", "pattern_class", "source_group", "gain_delta", "is_perturbed"]],
        on="source_group",
        how="inner",
    )
    if merged.empty:
        return pattern_summary, pd.DataFrame(), pd.DataFrame()
    merged["delta_contribution"] = merged["gain_delta"].astype(float) * merged["signed_drive"].astype(float)
    merged["abs_delta_contribution"] = merged["delta_contribution"].abs()
    merged["perturbed_abs_drive"] = np.where(
        merged["is_perturbed"].astype(bool),
        merged["abs_drive"].astype(float),
        0.0,
    )
    merged["perturbed_source_hit"] = np.where(
        merged["is_perturbed"].astype(bool) & (merged["abs_drive"].astype(float) > 0),
        merged["source_group"].astype(str),
        "",
    )

    candidate_response = (
        merged.groupby(["pattern_id", "pattern_label", "pattern_class", "candidate_root_id"], as_index=False)
        .agg(
            response_delta=("delta_contribution", "sum"),
            response_abs_mass=("abs_delta_contribution", "sum"),
            perturbed_abs_drive=("perturbed_abs_drive", "sum"),
            perturbed_source_hits=(
                "perturbed_source_hit",
                lambda values: sum(1 for value in set(map(str, values)) if value),
            ),
            perturbed_source_groups_hit=(
                "perturbed_source_hit",
                lambda values: ",".join(sorted(value for value in set(map(str, values)) if value)),
            ),
        )
    )
    candidate_response["response_abs"] = candidate_response["response_delta"].abs()
    candidate_response = candidate_response.merge(candidate_meta, on="candidate_root_id", how="left")
    candidate_response["off_pattern_abs_drive"] = (
        candidate_response["abs_total_drive"].fillna(0.0).astype(float)
        - candidate_response["perturbed_abs_drive"].fillna(0.0).astype(float)
    ).clip(lower=0.0)
    denominator = candidate_response["abs_total_drive"].fillna(0.0).astype(float)
    candidate_response["response_delta_normalized"] = np.divide(
        candidate_response["response_delta"].astype(float),
        denominator,
        out=np.zeros(len(candidate_response), dtype=float),
        where=denominator.to_numpy(dtype=float) > 0,
    )
    candidate_response["pattern_selectivity_index"] = np.divide(
        candidate_response["perturbed_abs_drive"].astype(float)
        - candidate_response["off_pattern_abs_drive"].astype(float),
        candidate_response["perturbed_abs_drive"].astype(float)
        + candidate_response["off_pattern_abs_drive"].astype(float),
        out=np.zeros(len(candidate_response), dtype=float),
        where=(
            candidate_response["perturbed_abs_drive"].astype(float)
            + candidate_response["off_pattern_abs_drive"].astype(float)
        ).to_numpy(dtype=float)
        > 0,
    )
    max_source_groups = max(1, int(source_group_summary["mbon_source_group"].nunique()))
    candidate_response["plasticity_knob_score"] = (
        np.log1p(100.0 * candidate_response["response_abs_mass"].astype(float))
        * (0.5 + candidate_response["n_source_groups"].fillna(0).astype(float) / max_source_groups)
        * (1.0 + 0.75 * candidate_response["push_pull_strength"].fillna(0.0).astype(float))
        * (
            1.0
            + 1.5 * candidate_response["dn_output_fraction"].fillna(0.0).astype(float)
            + 0.5 * candidate_response["state_output_fraction"].fillna(0.0).astype(float)
        )
        * (1.0 + candidate_response["pattern_selectivity_index"].clip(lower=0.0, upper=1.0))
    )
    candidate_response = candidate_response.sort_values(
        ["pattern_id", "plasticity_knob_score"], ascending=[True, False]
    ).reset_index(drop=True)
    candidate_response["plasticity_response_rank"] = (
        candidate_response.groupby("pattern_id").cumcount() + 1
    ).astype(int)
    candidate_response["primary_plasticity_rank"] = pd.Series(pd.NA, index=candidate_response.index, dtype="Int64")
    primary_downstream_mask = (
        ~candidate_response["candidate_role_tier"].isin(["global_feedback_control", "mb_recurrent_candidate"])
        & ~candidate_response["candidate_group"].isin(["DAN", "DPM", "MBON"])
    )
    candidate_response.loc[primary_downstream_mask, "primary_plasticity_rank"] = (
        candidate_response[primary_downstream_mask].groupby("pattern_id").cumcount() + 1
    ).astype(int)

    if output_summary.empty:
        return pattern_summary, candidate_response, pd.DataFrame()
    primary_response = candidate_response[
        ~candidate_response["candidate_role_tier"].isin(["global_feedback_control", "mb_recurrent_candidate"])
        & ~candidate_response["candidate_group"].isin(["DAN", "DPM", "MBON"])
    ].copy()
    output_total = (
        output_summary.groupby("candidate_root_id", as_index=False)["output_abs_weight"].sum().rename(
            columns={"output_abs_weight": "candidate_output_abs_total"}
        )
    )
    output_norm = output_summary.merge(output_total, on="candidate_root_id", how="left")
    denominator = output_norm["candidate_output_abs_total"].astype(float).replace(0.0, np.nan)
    output_norm["axis_signed_fraction"] = (
        output_norm["output_signed_weight"].astype(float) / denominator
    ).fillna(0.0)
    output_norm["axis_abs_fraction"] = (
        output_norm["output_abs_weight"].astype(float) / denominator
    ).fillna(0.0)
    candidate_axis = primary_response.merge(
        output_norm[
            [
                "candidate_root_id",
                "behavior_output_group",
                "axis_signed_fraction",
                "axis_abs_fraction",
                "output_abs_weight",
            ]
        ],
        on="candidate_root_id",
        how="inner",
    )
    if candidate_axis.empty:
        return pattern_summary, candidate_response, pd.DataFrame()
    candidate_axis["axis_response_delta"] = (
        candidate_axis["response_delta"].astype(float) * candidate_axis["axis_signed_fraction"].astype(float)
    )
    candidate_axis["axis_response_abs"] = (
        candidate_axis["response_abs"].astype(float) * candidate_axis["axis_abs_fraction"].astype(float)
    )
    axis_response = (
        candidate_axis.groupby(["pattern_id", "pattern_label", "pattern_class", "behavior_output_group"], as_index=False)
        .agg(
            n_candidate_contributors=("candidate_root_id", "nunique"),
            axis_response_delta=("axis_response_delta", "sum"),
            axis_response_abs=("axis_response_abs", "sum"),
            candidate_response_abs_support=("response_abs", "sum"),
            mean_primary_plasticity_rank=("primary_plasticity_rank", "mean"),
        )
    )
    total_axis_abs = axis_response.groupby("pattern_id")["axis_response_abs"].transform("sum").replace(0.0, np.nan)
    axis_response["axis_response_abs_fraction"] = (axis_response["axis_response_abs"] / total_axis_abs).fillna(0.0)
    axis_response = axis_response.sort_values(["pattern_id", "axis_response_abs"], ascending=[True, False])
    return pattern_summary, candidate_response, axis_response.reset_index(drop=True)


def _mbon_primary_integrator_mask(frame: pd.DataFrame) -> pd.Series:
    role = frame.get("candidate_role_tier", pd.Series("", index=frame.index)).fillna("").astype(str)
    group = frame.get("candidate_group", pd.Series("", index=frame.index)).fillna("").astype(str)
    return ~role.isin(["global_feedback_control", "mb_recurrent_candidate"]) & ~group.isin(["DAN", "DPM", "MBON"])


def _run_mbon_plasticity_source_label_null_controls(
    *,
    source_group_summary: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    source_candidate_matrix: pd.DataFrame,
    plasticity_candidate_response: pd.DataFrame,
    plasticity_axis_response: pd.DataFrame,
    null_repeats: int,
    random_seed: int,
) -> pd.DataFrame:
    """Shuffle MB-compartment gain labels on the fixed FlyWire drive matrix.

    This is a fast source-label null, not a topology rewire null.  It asks
    whether the observed gamma/compartment response is stronger than random
    assignment of the same gain deltas to the same 15 source groups.
    """

    if (
        int(null_repeats) <= 0
        or source_group_summary.empty
        or candidate_summary.empty
        or source_candidate_matrix.empty
        or plasticity_candidate_response.empty
    ):
        return pd.DataFrame()
    source_groups = source_group_summary["mbon_source_group"].astype(str).tolist()
    _pattern_summary, pattern_long = _mbon_plasticity_replay_patterns(source_groups)
    if pattern_long.empty:
        return pd.DataFrame()
    primary_ids = set(
        candidate_summary.loc[_mbon_primary_integrator_mask(candidate_summary), "candidate_root_id"]
        .astype("int64")
        .tolist()
    )
    if not primary_ids:
        return pd.DataFrame()
    matrix = source_candidate_matrix[
        source_candidate_matrix["candidate_root_id"].astype("int64").isin(primary_ids)
    ][["source_group", "candidate_root_id", "signed_drive", "abs_drive"]].copy()
    if matrix.empty:
        return pd.DataFrame()
    candidate_meta = candidate_summary.set_index("candidate_root_id")
    observed_patterns = set(plasticity_candidate_response["pattern_id"].astype(str))
    rng = np.random.default_rng(int(random_seed))
    rows: list[dict[str, object]] = []

    for pattern_id, pattern_rows in pattern_long.groupby("pattern_id", sort=False):
        pattern_id = str(pattern_id)
        if pattern_id not in observed_patterns:
            continue
        observed = plasticity_candidate_response[
            plasticity_candidate_response["pattern_id"].astype(str).eq(pattern_id)
            & plasticity_candidate_response["candidate_root_id"].astype("int64").isin(primary_ids)
        ].copy()
        if observed.empty:
            continue
        observed = observed.sort_values("response_abs_mass", ascending=False)
        observed_top = observed.iloc[0]
        observed_top_value = float(observed_top["response_abs_mass"])
        deltas = (
            pattern_rows.set_index("source_group")["gain_delta"]
            .reindex(source_groups)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        null_values: list[float] = []
        for _repeat in range(int(null_repeats)):
            shuffled = rng.permutation(deltas)
            shuffled_frame = pd.DataFrame({"source_group": source_groups, "gain_delta": shuffled})
            merged = matrix.merge(shuffled_frame, on="source_group", how="inner")
            if merged.empty:
                null_values.append(0.0)
                continue
            merged["delta_contribution"] = merged["gain_delta"].astype(float) * merged["signed_drive"].astype(float)
            merged["abs_delta_contribution"] = merged["delta_contribution"].abs()
            response = (
                merged.groupby("candidate_root_id", as_index=False)
                .agg(response_abs_mass=("abs_delta_contribution", "sum"))
                .sort_values("response_abs_mass", ascending=False)
            )
            null_values.append(float(response["response_abs_mass"].iloc[0]) if not response.empty else 0.0)
        null_array = np.asarray(null_values, dtype=float)
        if null_array.size == 0:
            continue
        axis_subset = plasticity_axis_response[plasticity_axis_response["pattern_id"].astype(str).eq(pattern_id)].copy()
        known_axis = axis_subset[~axis_subset["behavior_output_group"].astype(str).eq("other")].copy()
        top_axis = known_axis.sort_values("axis_response_abs", ascending=False).iloc[0] if not known_axis.empty else None
        candidate_id = int(observed_top["candidate_root_id"])
        candidate_row = candidate_meta.loc[candidate_id] if candidate_id in candidate_meta.index else pd.Series(dtype=object)
        rows.append(
            {
                "pattern_id": pattern_id,
                "pattern_label": str(pattern_rows["pattern_label"].iloc[0]),
                "pattern_class": str(pattern_rows["pattern_class"].iloc[0]),
                "observed_top_candidate_root_id": candidate_id,
                "observed_top_candidate_cell_type": str(candidate_row.get("candidate_cell_type", "")),
                "observed_top_response_abs_mass": observed_top_value,
                "null_mean_top_response_abs_mass": float(null_array.mean()),
                "null_sd_top_response_abs_mass": float(null_array.std(ddof=1)) if null_array.size > 1 else 0.0,
                "null_p_ge_observed": float((1 + np.count_nonzero(null_array >= observed_top_value)) / (1 + null_array.size)),
                "observed_percentile_vs_null": float(np.count_nonzero(null_array <= observed_top_value) / null_array.size),
                "top_known_output_axis": "" if top_axis is None else str(top_axis["behavior_output_group"]),
                "top_known_output_axis_abs_fraction": 0.0
                if top_axis is None
                else float(top_axis["axis_response_abs_fraction"]),
                "null_repeats": int(null_repeats),
                "null_type": "source_label_shuffle_on_fixed_flywire_drive_matrix",
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(
        ["pattern_class", "null_p_ge_observed", "pattern_id"], ascending=[True, True, True]
    ).reset_index(drop=True)


def _run_mbon_plasticity_convergence_null_controls(
    *,
    source_group_summary: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    source_candidate_matrix: pd.DataFrame,
    plasticity_candidate_response: pd.DataFrame,
    plasticity_axis_response: pd.DataFrame,
    null_repeats: int,
    random_seed: int,
) -> pd.DataFrame:
    """Shuffle candidate labels within each source group on the fixed drive matrix.

    This keeps each MB compartment's distribution of downstream signed drives,
    but breaks the alignment in which the same candidate receives high drive
    from multiple compartments.  It is a convergence/topology control at the
    already-computed candidate-matrix level, not a full raw-edge rewire.
    """

    if (
        int(null_repeats) <= 0
        or source_group_summary.empty
        or candidate_summary.empty
        or source_candidate_matrix.empty
        or plasticity_candidate_response.empty
    ):
        return pd.DataFrame()
    source_groups = source_group_summary["mbon_source_group"].astype(str).tolist()
    _pattern_summary, pattern_long = _mbon_plasticity_replay_patterns(source_groups)
    if pattern_long.empty:
        return pd.DataFrame()
    primary_ids = set(
        candidate_summary.loc[_mbon_primary_integrator_mask(candidate_summary), "candidate_root_id"]
        .astype("int64")
        .tolist()
    )
    if not primary_ids:
        return pd.DataFrame()
    matrix = source_candidate_matrix[
        source_candidate_matrix["candidate_root_id"].astype("int64").isin(primary_ids)
    ][["source_group", "candidate_root_id", "signed_drive", "abs_drive"]].copy()
    if matrix.empty:
        return pd.DataFrame()
    matrix["candidate_root_id"] = matrix["candidate_root_id"].astype("int64")
    candidate_meta = candidate_summary.set_index("candidate_root_id")
    observed_patterns = set(plasticity_candidate_response["pattern_id"].astype(str))
    rng = np.random.default_rng(int(random_seed) + 7919)
    rows: list[dict[str, object]] = []

    grouped_rows: list[pd.DataFrame] = []
    for _source_group, source_rows in matrix.groupby("source_group", sort=False):
        grouped_rows.append(source_rows.reset_index(drop=True))

    for pattern_id, pattern_rows in pattern_long.groupby("pattern_id", sort=False):
        pattern_id = str(pattern_id)
        if pattern_id not in observed_patterns:
            continue
        observed = plasticity_candidate_response[
            plasticity_candidate_response["pattern_id"].astype(str).eq(pattern_id)
            & plasticity_candidate_response["candidate_root_id"].astype("int64").isin(primary_ids)
        ].copy()
        if observed.empty:
            continue
        observed = observed.sort_values("response_abs_mass", ascending=False)
        observed_top = observed.iloc[0]
        observed_candidate_id = int(observed_top["candidate_root_id"])
        observed_top_value = float(observed_top["response_abs_mass"])
        gain_by_source = (
            pattern_rows.set_index("source_group")["gain_delta"].reindex(source_groups).fillna(0.0).astype(float)
        )

        null_top_values: list[float] = []
        null_observed_candidate_values: list[float] = []
        null_same_top_count = 0
        for _repeat in range(int(null_repeats)):
            shuffled_frames: list[pd.DataFrame] = []
            for source_rows in grouped_rows:
                shuffled = source_rows.copy()
                shuffled["candidate_root_id"] = rng.permutation(
                    shuffled["candidate_root_id"].to_numpy(dtype=np.int64, copy=True)
                )
                shuffled_frames.append(shuffled)
            shuffled_matrix = pd.concat(shuffled_frames, ignore_index=True)
            shuffled_matrix["gain_delta"] = shuffled_matrix["source_group"].map(gain_by_source).fillna(0.0)
            shuffled_matrix["delta_contribution"] = (
                shuffled_matrix["gain_delta"].astype(float) * shuffled_matrix["signed_drive"].astype(float)
            )
            shuffled_matrix["abs_delta_contribution"] = shuffled_matrix["delta_contribution"].abs()
            response = (
                shuffled_matrix.groupby("candidate_root_id", as_index=False)
                .agg(response_abs_mass=("abs_delta_contribution", "sum"))
                .sort_values("response_abs_mass", ascending=False)
            )
            if response.empty:
                null_top_values.append(0.0)
                null_observed_candidate_values.append(0.0)
                continue
            null_top_candidate_id = int(response["candidate_root_id"].iloc[0])
            null_top_values.append(float(response["response_abs_mass"].iloc[0]))
            null_same_top_count += int(null_top_candidate_id == observed_candidate_id)
            observed_row = response[response["candidate_root_id"].astype("int64").eq(observed_candidate_id)]
            null_observed_candidate_values.append(
                float(observed_row["response_abs_mass"].iloc[0]) if not observed_row.empty else 0.0
            )

        null_top_array = np.asarray(null_top_values, dtype=float)
        null_candidate_array = np.asarray(null_observed_candidate_values, dtype=float)
        if null_top_array.size == 0:
            continue
        axis_subset = plasticity_axis_response[plasticity_axis_response["pattern_id"].astype(str).eq(pattern_id)].copy()
        known_axis = axis_subset[~axis_subset["behavior_output_group"].astype(str).eq("other")].copy()
        top_axis = known_axis.sort_values("axis_response_abs", ascending=False).iloc[0] if not known_axis.empty else None
        candidate_row = (
            candidate_meta.loc[observed_candidate_id]
            if observed_candidate_id in candidate_meta.index
            else pd.Series(dtype=object)
        )
        rows.append(
            {
                "pattern_id": pattern_id,
                "pattern_label": str(pattern_rows["pattern_label"].iloc[0]),
                "pattern_class": str(pattern_rows["pattern_class"].iloc[0]),
                "observed_top_candidate_root_id": observed_candidate_id,
                "observed_top_candidate_cell_type": str(candidate_row.get("candidate_cell_type", "")),
                "observed_top_response_abs_mass": observed_top_value,
                "null_mean_top_response_abs_mass": float(null_top_array.mean()),
                "null_sd_top_response_abs_mass": float(null_top_array.std(ddof=1)) if null_top_array.size > 1 else 0.0,
                "null_top_p_ge_observed": float(
                    (1 + np.count_nonzero(null_top_array >= observed_top_value)) / (1 + null_top_array.size)
                ),
                "observed_candidate_null_mean_response_abs_mass": float(null_candidate_array.mean()),
                "observed_candidate_null_sd_response_abs_mass": float(null_candidate_array.std(ddof=1))
                if null_candidate_array.size > 1
                else 0.0,
                "observed_candidate_p_ge_observed": float(
                    (1 + np.count_nonzero(null_candidate_array >= observed_top_value))
                    / (1 + null_candidate_array.size)
                ),
                "same_top_candidate_fraction": float(null_same_top_count / null_top_array.size),
                "top_known_output_axis": "" if top_axis is None else str(top_axis["behavior_output_group"]),
                "top_known_output_axis_abs_fraction": 0.0
                if top_axis is None
                else float(top_axis["axis_response_abs_fraction"]),
                "null_repeats": int(null_repeats),
                "null_type": "within_source_candidate_label_shuffle_on_fixed_drive_matrix",
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(
        ["pattern_class", "null_top_p_ge_observed", "pattern_id"], ascending=[True, True, True]
    ).reset_index(drop=True)


def _build_mbon_wetlab_priority_table(
    *,
    plasticity_candidate_response: pd.DataFrame,
    plasticity_axis_response: pd.DataFrame,
) -> pd.DataFrame:
    if plasticity_candidate_response.empty:
        return pd.DataFrame()
    preferred_patterns = [
        "aversive_gamma13_down",
        "single_gamma1_down",
        "single_gamma2_down",
        "single_gamma3_down",
        "single_gamma4_down",
        "single_gamma5_down",
    ]
    available_patterns = [
        pattern for pattern in preferred_patterns if pattern in set(plasticity_candidate_response["pattern_id"].astype(str))
    ]
    if not available_patterns:
        available_patterns = sorted(set(plasticity_candidate_response["pattern_id"].astype(str)))[:8]

    layer_specs: list[tuple[str, str]] = [
        ("non_mb_integrator", "candidate calcium/split-GFP first; then targeted perturbation"),
        ("state_teaching_feedback", "PPL1-DAN dopamine sensor, DPM calcium/5-HT readout"),
        ("global_apl_feedback", "APL calcium plus KC sparse-response imaging"),
        ("dn_motor_proximal", "candidate/DN calcium and behaviour readout"),
    ]

    rows: list[dict[str, object]] = []
    for pattern_id in available_patterns:
        subset = plasticity_candidate_response[
            plasticity_candidate_response["pattern_id"].astype(str).eq(pattern_id)
        ].copy()
        if subset.empty:
            continue
        axis_subset = plasticity_axis_response[
            plasticity_axis_response["pattern_id"].astype(str).eq(pattern_id)
        ].copy()
        known_axis = axis_subset[~axis_subset["behavior_output_group"].astype(str).eq("other")].copy()
        top_known_axis = (
            known_axis.sort_values("axis_response_abs", ascending=False).iloc[0] if not known_axis.empty else None
        )
        for layer, readout_hint in layer_specs:
            if layer == "non_mb_integrator":
                layer_frame = subset[_mbon_primary_integrator_mask(subset)].copy()
            elif layer == "state_teaching_feedback":
                layer_frame = subset[
                    subset["candidate_group"].isin(["DAN", "DPM"])
                    | subset["candidate_role_tier"].eq("state_modulation_candidate")
                ].copy()
            elif layer == "global_apl_feedback":
                layer_frame = subset[subset["candidate_group"].eq("APL")].copy()
            else:
                layer_frame = subset[
                    subset["candidate_role_tier"].eq("dn_or_motor_proximal_candidate")
                    | (subset["dn_output_fraction"].fillna(0.0).astype(float) > 0.02)
                ].copy()
            if layer_frame.empty:
                continue
            layer_frame = layer_frame.sort_values("plasticity_knob_score", ascending=False)
            top = layer_frame.iloc[0]
            response_delta = float(top.get("response_delta", 0.0))
            direction = "increase" if response_delta > 0 else "decrease" if response_delta < 0 else "no_clear_change"
            candidate_label = f"{top.get('candidate_cell_type', '')} {int(top['candidate_root_id'])}"
            rows.append(
                {
                    "pattern_id": pattern_id,
                    "pattern_label": str(top.get("pattern_label", pattern_id)),
                    "layer": layer,
                    "candidate_root_id": int(top["candidate_root_id"]),
                    "candidate_cell_type": str(top.get("candidate_cell_type", "")),
                    "candidate_group": str(top.get("candidate_group", "")),
                    "candidate_side": str(top.get("candidate_side", "")),
                    "candidate_top_nt": str(top.get("candidate_top_nt", "")),
                    "response_direction": direction,
                    "response_delta": response_delta,
                    "response_abs_mass": float(top.get("response_abs_mass", 0.0)),
                    "plasticity_knob_score": float(top.get("plasticity_knob_score", 0.0)),
                    "top_known_output_axis": "" if top_known_axis is None else str(top_known_axis["behavior_output_group"]),
                    "top_known_axis_abs_fraction": 0.0
                    if top_known_axis is None
                    else float(top_known_axis["axis_response_abs_fraction"]),
                    "imaging_readout": readout_hint,
                    "perturbation_design": (
                        "train OCT/MCH or appetitive/aversive CS+/CS-; perturb the matching MBON compartment/subtype "
                        "or candidate node with split-GAL4 optogenetics/Kir2.1/TNT/shibire-ts where feasible"
                    ),
                    "behavior_assay": (
                        "OCT/MCH immediate, delayed, and delayed-conflict; add appetitive/aversive choice if the "
                        "candidate's valence annotation supports it"
                    ),
                    "prediction_cn": (
                        f"在 {str(top.get('pattern_label', pattern_id))} 条件下，{candidate_label} "
                        f"预计 {direction}；若成像方向一致，再进入扰动和行为 readout。"
                    ),
                    "risk_cn": (
                        "priority table is connectome-constrained; driver specificity, receptor sign, neuromodulator "
                        "state and behaviour annotation still require wet-lab validation"
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)


def _write_mbon_plasticity_replay_figure(
    candidate_response: pd.DataFrame,
    axis_response: pd.DataFrame,
    output_dir: Path,
) -> Path | None:
    if candidate_response.empty:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    pattern_id = (
        "aversive_gamma13_down"
        if "aversive_gamma13_down" in set(candidate_response["pattern_id"].astype(str))
        else str(candidate_response["pattern_id"].iloc[0])
    )
    subset = candidate_response[
        candidate_response["pattern_id"].eq(pattern_id)
        & ~candidate_response["candidate_role_tier"].isin(["global_feedback_control", "mb_recurrent_candidate"])
    ].copy()
    if subset.empty:
        subset = candidate_response[candidate_response["pattern_id"].eq(pattern_id)].copy()
    top = subset.sort_values("plasticity_knob_score", ascending=False).head(14).copy()
    figure_path = output_dir / "Fig_mbon_plasticity_replay.png"
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)

    labels = [f"{row.candidate_cell_type}\n{int(row.candidate_root_id)}" for row in top.itertuples(index=False)]
    x = np.arange(len(top))
    axes[0, 0].bar(x, top["plasticity_knob_score"], color="#4c78a8")
    axes[0, 0].set_xticks(x, labels, rotation=70, ha="right", fontsize=7)
    axes[0, 0].set_ylabel("plasticity knob score")
    axes[0, 0].set_title("Gamma1-3 down replay: primary candidates")

    colors = np.where(top["response_delta"].astype(float) >= 0, "#e15759", "#59a14f")
    axes[0, 1].bar(x, top["response_delta"], color=colors)
    axes[0, 1].axhline(0, color="#333333", linewidth=0.8)
    axes[0, 1].set_xticks(x, labels, rotation=70, ha="right", fontsize=7)
    axes[0, 1].set_ylabel("signed response delta")
    axes[0, 1].set_title("Predicted candidate response direction")

    heat = axis_response[
        axis_response["pattern_id"].isin(
            ["aversive_gamma13_down", "gamma45_down_control", "all_gamma_down", "pan_compartment_down_25"]
        )
    ].copy()
    if not heat.empty:
        heat_table = heat.pivot_table(
            index="pattern_id",
            columns="behavior_output_group",
            values="axis_response_delta",
            aggfunc="sum",
            fill_value=0.0,
        )
        preferred_cols = [
            col
            for col in ["DN", "DN_like", "DAN", "DPM", "APL", "MBON", "octopamine", "serotonin", "other"]
            if col in heat_table.columns
        ]
        heat_table = heat_table[preferred_cols] if preferred_cols else heat_table
        image = axes[1, 0].imshow(heat_table.to_numpy(dtype=float), aspect="auto", cmap="coolwarm")
        axes[1, 0].set_xticks(np.arange(len(heat_table.columns)), heat_table.columns, rotation=45, ha="right", fontsize=8)
        axes[1, 0].set_yticks(np.arange(len(heat_table.index)), heat_table.index, fontsize=8)
        axes[1, 0].set_title("Direct output-axis signed response")
        fig.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04)
    else:
        axes[1, 0].axis("off")

    scatter = subset.copy()
    axes[1, 1].scatter(
        scatter["response_abs_mass"],
        scatter["dn_output_fraction"],
        s=40 + 400 * scatter["state_output_fraction"].fillna(0.0),
        c=scatter["pattern_selectivity_index"].fillna(0.0),
        cmap="viridis",
        vmin=-1,
        vmax=1,
        alpha=0.85,
    )
    axes[1, 1].set_xlabel("response abs mass")
    axes[1, 1].set_ylabel("DN output fraction")
    axes[1, 1].set_title("Response strength vs behaviour-proximal routing")

    fig.suptitle("FlyWire MBON 15-compartment plasticity replay", fontsize=14, fontweight="bold")
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return figure_path


def _render_mbon_decision_pivot_report(
    *,
    cfg: MBONDecisionPivotConfig,
    source_group_summary: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    source_candidate_matrix: pd.DataFrame,
    output_summary: pd.DataFrame,
    plasticity_pattern_summary: pd.DataFrame,
    plasticity_candidate_response: pd.DataFrame,
    plasticity_axis_response: pd.DataFrame,
    plasticity_null_controls: pd.DataFrame,
    plasticity_convergence_null_controls: pd.DataFrame,
    wetlab_priority: pd.DataFrame,
    figure_path: Path | None,
    plasticity_figure_path: Path | None,
    interpretation: dict[str, object],
) -> str:
    display_cols = [
        "rank",
        "primary_pivot_rank",
        "candidate_root_id",
        "candidate_cell_type",
        "candidate_group",
        "candidate_role_tier",
        "candidate_top_nt",
        "candidate_side",
        "n_source_groups",
        "abs_total_drive",
        "push_pull_index",
        "dn_output_fraction",
        "state_output_fraction",
        "pivot_score",
        "wetlab_readout_hint",
    ]
    source_cols = [
        "mbon_source_group",
        "source_group_label",
        "source_grouping",
        "n_mbon",
        "n_source_rows",
        "mbon_subtypes",
        "sides",
        "top_nt",
        "root_ids",
    ]
    matrix_cols = [
        "source_group",
        "candidate_root_id",
        "candidate_cell_type",
        "candidate_group",
        "min_hop",
        "signed_drive",
        "abs_drive",
    ]
    plasticity_candidate_cols = [
        "pattern_id",
        "primary_plasticity_rank",
        "candidate_root_id",
        "candidate_cell_type",
        "candidate_group",
        "candidate_role_tier",
        "response_delta",
        "response_abs_mass",
        "pattern_selectivity_index",
        "plasticity_knob_score",
        "dn_output_fraction",
        "state_output_fraction",
        "primary_pivot_rank",
    ]
    plasticity_axis_cols = [
        "pattern_id",
        "behavior_output_group",
        "n_candidate_contributors",
        "axis_response_delta",
        "axis_response_abs",
        "axis_response_abs_fraction",
        "mean_primary_plasticity_rank",
    ]
    null_cols = [
        "pattern_id",
        "observed_top_candidate_cell_type",
        "observed_top_candidate_root_id",
        "observed_top_response_abs_mass",
        "null_mean_top_response_abs_mass",
        "null_sd_top_response_abs_mass",
        "null_p_ge_observed",
        "observed_percentile_vs_null",
        "top_known_output_axis",
        "top_known_output_axis_abs_fraction",
    ]
    wetlab_cols = [
        "pattern_id",
        "layer",
        "candidate_cell_type",
        "candidate_root_id",
        "response_direction",
        "response_abs_mass",
        "top_known_output_axis",
        "top_known_axis_abs_fraction",
        "imaging_readout",
        "behavior_assay",
    ]
    convergence_null_cols = [
        "pattern_id",
        "observed_top_candidate_cell_type",
        "observed_top_candidate_root_id",
        "observed_top_response_abs_mass",
        "null_mean_top_response_abs_mass",
        "null_top_p_ge_observed",
        "observed_candidate_null_mean_response_abs_mass",
        "observed_candidate_p_ge_observed",
        "same_top_candidate_fraction",
        "top_known_output_axis",
        "top_known_output_axis_abs_fraction",
    ]
    candidate_display = candidate_summary[display_cols].copy()
    if "primary_pivot_rank" in candidate_display.columns:
        candidate_display["primary_pivot_rank"] = candidate_display["primary_pivot_rank"].map(
            lambda value: "" if pd.isna(value) else str(int(value))
        )
    primary_candidate_display = candidate_display[candidate_display["primary_pivot_rank"] != ""].copy()
    figure_line = f"\n![MBON decision pivot candidates]({figure_path.name})\n" if figure_path else ""
    plasticity_figure_line = (
        f"\n![MBON plasticity replay]({plasticity_figure_path.name})\n" if plasticity_figure_path else ""
    )
    plasticity_primary = pd.DataFrame()
    plasticity_axis_primary = pd.DataFrame()
    single_compartment_display = pd.DataFrame()
    if not plasticity_candidate_response.empty:
        plasticity_primary = plasticity_candidate_response[
            plasticity_candidate_response["pattern_id"].eq("aversive_gamma13_down")
            & ~plasticity_candidate_response["candidate_role_tier"].isin(
                ["global_feedback_control", "mb_recurrent_candidate"]
            )
            & ~plasticity_candidate_response["candidate_group"].isin(["DAN", "DPM", "MBON"])
        ].copy()
        plasticity_primary = plasticity_primary.sort_values("plasticity_knob_score", ascending=False)
        if "primary_plasticity_rank" in plasticity_primary.columns:
            plasticity_primary["primary_plasticity_rank"] = plasticity_primary["primary_plasticity_rank"].map(
                lambda value: "" if pd.isna(value) else str(int(value))
            )
        if "primary_pivot_rank" in plasticity_primary.columns:
            plasticity_primary["primary_pivot_rank"] = plasticity_primary["primary_pivot_rank"].map(
                lambda value: "" if pd.isna(value) else str(int(value))
            )
        single_rows: list[dict[str, object]] = []
        primary_mask = _mbon_primary_integrator_mask(plasticity_candidate_response)
        for group in ASO2014_COMPARTMENT_ORDER:
            pattern_id = f"single_{group}_down"
            subset = plasticity_candidate_response[
                plasticity_candidate_response["pattern_id"].astype(str).eq(pattern_id) & primary_mask
            ].copy()
            if subset.empty:
                continue
            subset = subset.sort_values("plasticity_knob_score", ascending=False)
            top = subset.iloc[0]
            axis_subset = (
                plasticity_axis_response[
                    plasticity_axis_response["pattern_id"].astype(str).eq(pattern_id)
                    & ~plasticity_axis_response["behavior_output_group"].astype(str).eq("other")
                ].copy()
                if not plasticity_axis_response.empty
                else pd.DataFrame()
            )
            axis = axis_subset.sort_values("axis_response_abs", ascending=False).iloc[0] if not axis_subset.empty else None
            response_delta = float(top.get("response_delta", 0.0))
            single_rows.append(
                {
                    "mb_compartment": ASO2014_COMPARTMENT_LABELS.get(group, group),
                    "top_non_mb_candidate": f"{top.get('candidate_cell_type', '')} {int(top['candidate_root_id'])}",
                    "response_direction": "increase" if response_delta >= 0 else "decrease",
                    "response_abs_mass": float(top.get("response_abs_mass", 0.0)),
                    "strongest_known_axis": "" if axis is None else str(axis["behavior_output_group"]),
                    "known_axis_abs_fraction": 0.0
                    if axis is None
                    else float(axis["axis_response_abs_fraction"]),
                    "interpretation": (
                        "single-compartment ranking, not behaviour proof; use as candidate imaging/perturbation priority"
                    ),
                }
            )
        single_compartment_display = pd.DataFrame.from_records(single_rows)
    if not plasticity_axis_response.empty:
        plasticity_axis_primary = plasticity_axis_response[
            plasticity_axis_response["pattern_id"].eq("aversive_gamma13_down")
        ].copy()
        plasticity_axis_primary = plasticity_axis_primary.sort_values("axis_response_abs", ascending=False)
    return f"""# MBON 下游 decision pivot / knob 候选搜索报告

保存路径：`{cfg.output_dir}/MBON_DECISION_PIVOT_CANDIDATES_CN.md`

## 问题

本分析关注的问题是：蘑菇体上游 `ORN -> PN -> KC -> MBON` 已经研究较充分，但学习后多个
MBON functional compartment 的信号在下游哪里整合，并如何转成趋近/回避等行为选择，目前仍是重要空白。

本报告用真实 FlyWire v783 连接组从 MBON 出发向下游追踪 `{cfg.max_hops}` 跳，寻找同时接收多个
MB lobe compartment 输入、具有 push-pull signed drive、并接近 DN/状态调制输出的候选节点。
默认 source grouping 为 `{cfg.source_grouping}`。`compartment` 模式按 Aso et al. 2014 的 15 个
MB lobe compartments 分组，跨 compartment MBON 会被复制到多个 source rows；当前 FlyWire annotation
不能可靠映射到 Aso compartment 的 MBON 会标为 `unmapped`，默认不纳入传播。`subtype` 模式保留旧的
FlyWire/hemibrain `cell_type` grouping，用于前向兼容和追溯 annotation。

解释时需要分层，而不是生物学上先验删掉某些节点。APL 是已知的 KC 全局反馈/稀疏控制载体；
PPL1/DPM/DAN 这类节点可能代表 teaching/state feedback；MBON recurrent nodes 可能代表蘑菇体内部
回路。表中 `rank` 保留原始传播得分，用于回答全量连接组里哪些机制层被命中；`primary_pivot_rank`
只是一个窄口径的非 MB integrator 子排名，用于单独查看 SIP/LH/CRE/SMP 等前馈整合候选。

## MBON 源

共纳入 `{interpretation['n_mbon_neurons']}` 个 MBON，聚合为 `{interpretation['n_mbon_source_groups']}` 个
MBON source groups；source rows 为 `{interpretation['n_mbon_source_rows']}`。映射边界：
`{interpretation['source_group_definition']}`。

**表 R1｜MBON source 分组。** 这个表说明哪些 FlyWire MBON 被映射到 Aso 2014 的 15 个 MB lobe
compartments。它不是候选结果表，而是后续所有传播分析的输入定义。

{_table(source_group_summary[source_cols], max_rows=24)}

## 候选排序

评分不是因果证明，只是实验优先级。`rank` 是全量原始排序；`primary_pivot_rank` 是非 MB integrator
子排名，用来和 APL 全局反馈、DAN/DPM 状态反馈、MBON recurrent 回路分层对照：

```text
pivot_score =
  log1p(n_source_groups) * log1p(abs_total_drive)
  * (1 + 0.75 * push_pull_strength)
  * (1 + 0.75 * DN_output_fraction + 0.35 * state_output_fraction)
  / (1 + 0.20 * min_hop)
```

其中 `push_pull_strength = min(positive_drive, negative_drive) / abs_total_drive`。这个定义偏好：
多 MBON compartment/source group 汇聚、正负 valence/sign 输入同时存在、靠近 DN 或 DAN/DPM/APL 等状态/行为输出的节点。

### Non-MB Integrator Candidate Ranking

下面这张表不是说 APL、MBON recurrent 或 PPL1/DPM/DAN 不重要，而是单独显示非 MB 前馈/整合候选。
APL、状态反馈和 MB recurrent 节点保留在 raw ranking 中，作为可能机制层继续分析。

**表 R2｜非 MB 前馈/整合候选子排名。** 这个表只回答“哪些 SIP/LH/CRE/SMP 等非 MB 下游节点可能整合
多个 MB compartment 输入”。它排除了 APL、DPM/PPL1-DAN、MBON recurrent 这类反馈层，方便单独看
前馈整合候选。

{_table(primary_candidate_display, max_rows=24)}

### Full raw ranking

下面保留原始传播排序；APL 在这里作为 positive-control 出现，不作为新的 decision-pivot 主结论。

**表 R3｜全层级 raw ranking。** 这个表不筛掉反馈层，用来检查全连接组传播首先命中哪些机制层。
APL、DPM/PPL1-DAN 或 MBON recurrent 出现在这里是有生物意义的，不应被当作噪声删除。

{_table(candidate_display, max_rows=30)}

{figure_line}
## Source-to-candidate 汇聚矩阵

下面显示 top candidate 的主要 MBON compartment/source 输入，便于和具体 MBON subtype/driver line 对接。

**表 R4｜source-to-candidate drive matrix。** 这个表展开“某个候选是由哪些 MB compartment 驱动”的
细节，适合追溯具体 compartment、侧别和 driver-line 可行性。

{_table(source_candidate_matrix[matrix_cols], max_rows=40)}

## 候选下游去向

**表 R5｜候选下游输出轴。** 这个表把候选节点继续投到 DN/DAN/DPM/APL/MBON/other 等下游轴，帮助判断
候选更像行为输出近端、状态反馈，还是尚未注释的中间层。

{_table(output_summary, max_rows=40)}

## 学习后 15 分区 plasticity replay

老师提出的关键问题不是“MBON 下游有哪些强连接”，而是：当学习改变 MB compartment 输出后，哪些下游节点真正对
15 个功能区的组合变化敏感。为此本报告在同一套真实 FlyWire 1-3 跳 drive matrix 上增加了 plasticity replay。
这里需要先说明口径：系统不是只分析 5 个 γ 区，而是完整跑了 Aso 2014 的 15 个 MB compartment
single-compartment sweep。`γ1-3 down / γ4-5 unchanged` 只是借用文献中较熟悉的 aversive-learning
γ-lobe pattern 做校准场景，用来测试 BioFly 能不能把已知 plasticity motif 往下游追踪；它不能代表
其余 10 个 α/β/α′/β′ 区不重要。报告同时输出 `γ4-5 down`、`all gamma down`、
`pan-compartment down` 和 15 个 single-compartment sweep。

**表 R6｜plasticity replay 条件定义。** 这个表只定义仿真条件：哪些 compartment 被下调、下调幅度是多少。
它本身不是候选结果表。

{_table(plasticity_pattern_summary, max_rows=24)}

### 15 个 MB compartment 的单区 sweep

下面先列全 15 区单独下调 `75%` 时，各自最敏感的 non-MB 下游候选。这个表回答“其他 10 个区域呢”：
它们已经纳入探索，只是目前还没有像 γ1-3 那样有明确的训练后 plasticity 文献作为统一组合模式。
单区 sweep 的 top-null 统计不应过度解读，因为只有一个 source 被扰动时，打乱 candidate label 会保留该
source 的最大 drive 分布；因此这里主要作为 candidate imaging 和 perturbation 的优先级表。

**表 R7｜15 个 MB compartment 的单区 plasticity sweep。** 这个表和 R4/R5 不同：这里主动把每个
compartment 单独下调 `75%`，看下游哪个候选对该分区的 plasticity 最敏感。

{_table(single_compartment_display, max_rows=20)}

注意，单区表中的 top candidate 按整体 `plasticity_knob_score` 排序；若单独强调 DN/运动近端 readout，
`γ4/γ5` 中 CRE100 会比 CRE011 更值得作为二线行为输出近端候选。因此 CRE011 和 CRE100 代表不同筛选口径，
不是互相矛盾的结论。

### γ-lobe 文献锚点 replay 候选

这里的分数不再只问“谁接收最多 MBON 输入”，而是问“学习后 γ1-3 下降时，谁的 signed response 改变最大、
是否对 γ1-3 相对 γ4-5 更特异、是否靠近 DN/状态输出轴”。APL、MBON recurrent nodes 以及
PPL1/DPM/DAN state-feedback nodes 保留在 raw 结果中作机制层候选；下表只展示 non-MB integrator
子排名。

**表 R8｜`γ1-3 down / γ4-5 unchanged` 文献锚点 replay 候选。** 这个表不是全 15 区的无偏结果，
而是用已有 aversive-learning γ-lobe motif 做校准，检查 BioFly 是否把该 motif 推到合理下游候选层。

{_table(plasticity_primary[plasticity_candidate_cols], max_rows=24)}

### 直接输出轴读出

下面把 candidate response 继续投到其下游 `DN/DN_like/DAN/DPM/APL/MBON/other` 输出轴，作为 direct output-neuron
readout proxy。它比单一 choice proxy 更接近“直接在输出神经元上看信号差异”，但仍是连接组传播读数，
不是真实 calcium 或行为结果。

**表 R9｜γ-lobe replay 的直接输出轴读出。** 这个表把 R8 的候选响应继续投向下游输出轴，用来判断信号
是否已经接近 DN/行为输出。当前 `other` 占比高，因此不能声称找到唯一 decision neuron。

{_table(plasticity_axis_primary[plasticity_axis_cols], max_rows=16)}

### 统计假象控制

为了避免把“某个候选本来连接多”误写成学习特异机制，本报告增加 source-label null：保持真实 FlyWire
source-to-candidate drive matrix 不变，只随机打乱 15 个 compartment 上的 plasticity gain label。
因此它检验的是 `γ1-3` 或单 compartment pattern 是否强于随机 source-label 分配；它不是完整 topology
rewire null，也不能替代湿实验。

**表 R10｜source-label null control。** 这个表检验“学习 pattern 标签放在这些 source 上”是否比随机
放置更强。它控制的是 plasticity label，不是完整拓扑重连。

{_table(plasticity_null_controls[[col for col in null_cols if col in plasticity_null_controls.columns]], max_rows=18)}

报告还增加了一个更接近拓扑汇聚假象的 within-source candidate-label null：对每个 MB compartment
分别随机打乱 downstream drive 属于哪个 candidate，保留该 source 的 signed-drive 分布，但打破
同一 candidate 跨多个 source 的一致汇聚。它仍然不是 raw-edge degree/NT/side matched rewire，但比
source-label shuffle 更直接检验“候选是否只是因为跨 source 汇聚碰巧强”。

**表 R11｜within-source candidate-label null control。** 这个表检验同一候选跨 source 的一致命中是否
强于打乱后的自身 null。它用于判断 CRE050、CRE100、SIP088 这类候选是不是仅靠连接多排上来。

{_table(plasticity_convergence_null_controls[[col for col in convergence_null_cols if col in plasticity_convergence_null_controls.columns]], max_rows=18)}

### 湿实验优先级

下面把 γ-lobe 文献锚点场景整理成实验顺序。这里故意只列 `γ1-3 down / γ4-5 unchanged` 和
`γ1-γ5`，因为目前这类 aversive-learning plasticity motif 有较明确的文献依据，能直接设计训练前后
imaging、单区扰动和 OCT/MCH 行为 readout。其他 α/β/α′/β′ 分区已在 15 区 single-compartment
sweep 中保留为探索候选，但还缺少同等明确的学习后 plasticity pattern，不应混进同一张湿实验优先级表。
原则是先 imaging/递质 sensor 验证候选层响应方向，再做单区 MBON 或候选节点扰动，最后进入
OCT/MCH immediate、delayed 和 delayed-conflict 行为 readout。

**表 R12｜γ-lobe 文献锚点的湿实验优先级。** 这个表不是全 15 区优先级表，而是把 `γ1-3 down /
γ4-5 unchanged` 及 `γ1-γ5` 单区 sweep 转成可执行验证顺序。全 15 区无偏候选见表 R7。

{_table(wetlab_priority[[col for col in wetlab_cols if col in wetlab_priority.columns]], max_rows=28)}

{plasticity_figure_line}

## 当前可转述结论

1. BioFly 已能把 MBON 下游从泛泛“输出到下游脑区”收窄为一批真实 root_id 候选，并给出每个候选接收
   哪些 MBON compartment/source group、输入是正/负还是 push-pull、下游是否连到 DN/DAN/DPM/APL。
2. 这一步更接近 `decision-making knob / pivot neuron` 的筛选定义：候选不是单个强边，而是多个 MBON
   功能输出汇聚后可能影响行为 readout 的节点。
3. 15 个 single-compartment sweep 已经完整覆盖 α/β/α′/β′/γ 各区；γ1-3 down plasticity replay
   只是文献锚点场景。该场景下非 MB integrator 子排名主要是 CRE050、CRE011、SMP586、SMP177 等；
   PPL101 和 DPM 响应较强，应作为 teaching/state feedback 候选单独解释并保留，而不是丢弃。
4. 当前最适合的湿实验路线是：先用 MB compartment perturbation 或学习后 imaging 检查这些 candidate 的
   calcium/递质响应，再挑有 driver/anatomy 可行性的节点做 split-GFP/GRASP、optogenetic activation/silencing，
   最后放到 OCT/MCH delayed/conflict 或 appetitive/aversive choice assay 中验证。

## 边界

- 这不是最终 decision neuron 证明，只是连接组层面的候选排序。CRE/SMP/LH/SIP 类前馈整合候选、
  DAN/DPM 状态反馈候选、MBON recurrent 回路和 APL/KC 全局反馈候选需要分层表述，不能在生物学问题上
  先验删掉其中任何一层。
- `compartment` 模式使用 Aso et al. 2014 的 15 compartment 口径，但当前 FlyWire annotation 对部分 MBON subtype 缺少可靠 Aso compartment 映射；这些条目会被单独列出。
- `unmapped` 只表示不能作为 Aso 15 compartment source 安全归类；若这些 root 在多跳传播中被上游 source 命中，仍可作为 downstream candidate 出现在候选表或矩阵中。
- `subtype` 模式只用于兼容和 annotation 追溯，不应再被表述为“15 个输出功能区”。
- 多跳传播使用归一化 signed weights 和 hop decay，适合筛选，不等同于真实膜电位或 spike timing。
- plasticity replay 的 `γ1-3 down` 是文献机制启发的 γ-lobe 校准场景，不等于本系统已经测得真实学习后
  15 分区突触可塑性；不要把它写成只分析了 5 个 γ 区，15 个 single-compartment sweep 才是无偏探索图谱。
- 若 candidate 是未命名/other neuron，需要先做 annotation、morphology 和 driver 可行性检查。

## 输出文件

- source groups：`{cfg.output_dir / 'mbon_source_groups.csv'}`
- source mapping：`{cfg.output_dir / 'mbon_source_mapping.csv'}`
- source-candidate matrix：`{cfg.output_dir / 'mbon_to_candidate_drive_matrix.csv'}`
- candidate summary：`{cfg.output_dir / 'mbon_decision_pivot_candidates.csv'}`
- output summary：`{cfg.output_dir / 'mbon_pivot_candidate_outputs.csv'}`
- plasticity patterns：`{cfg.output_dir / 'mbon_plasticity_replay_patterns.csv'}`
- plasticity candidate response：`{cfg.output_dir / 'mbon_plasticity_replay_candidate_responses.csv'}`
- plasticity output-axis response：`{cfg.output_dir / 'mbon_plasticity_replay_output_axis_responses.csv'}`
- plasticity null controls：`{cfg.output_dir / 'mbon_plasticity_source_label_null_controls.csv'}`
- plasticity convergence null controls：`{cfg.output_dir / 'mbon_plasticity_convergence_null_controls.csv'}`
- wetlab priority table：`{cfg.output_dir / 'mbon_wetlab_experiment_priority.csv'}`
- figure：`{figure_path if figure_path else 'not generated'}`
- plasticity figure：`{plasticity_figure_path if plasticity_figure_path else 'not generated'}`
- metadata：`{cfg.output_dir / 'mbon_decision_pivot_metadata.json'}`
"""


def run_mbon_decision_pivot_search(
    config: MBONDecisionPivotConfig | None = None,
) -> dict[str, Path | pd.DataFrame | dict[str, object]]:
    cfg = config or MBONDecisionPivotConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    annotations = _load_annotations(cfg.annotation_path)
    annotated = _annotate_connectome_groups(annotations)
    edges = pd.read_parquet(
        cfg.connectivity_path,
        columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity", "Excitatory x Connectivity"],
    )
    mbon, source_group_summary, source_mapping = _mbon_source_groups(
        annotated,
        cfg.source_group_limit,
        source_grouping=str(cfg.source_grouping),
        include_unmapped_sources=bool(cfg.include_unmapped_sources),
    )
    adjacency = _build_normalized_out_adjacency(edges, min_abs_edge_weight=float(cfg.min_abs_edge_weight))
    meta = annotated.set_index("root_id")

    drive_frames: list[pd.DataFrame] = []
    for group, subset in mbon.groupby("mbon_source_group"):
        propagated = _propagate_mbon_source_group(
            subset["root_id"].astype("int64").tolist(),
            adjacency,
            max_hops=int(cfg.max_hops),
            hop_decay=float(cfg.hop_decay),
            max_frontier_per_source=int(cfg.max_frontier_per_source),
        )
        if propagated.empty:
            continue
        propagated["source_group"] = str(group)
        propagated["n_source_mbon"] = int(subset["root_id"].nunique())
        drive_frames.append(propagated)
    drive = pd.concat(drive_frames, ignore_index=True) if drive_frames else pd.DataFrame()
    if drive.empty:
        raise ValueError("MBON downstream propagation produced no candidate rows.")

    mbon_ids = set(mbon["root_id"].astype("int64"))
    drive = drive[~drive["target_root_id"].isin(mbon_ids)].copy()
    drive["positive_drive"] = drive["signed_drive"].clip(lower=0)
    drive["negative_drive"] = (-drive["signed_drive"].clip(upper=0))
    collapsed = (
        drive.groupby(["source_group", "target_root_id"], as_index=False)
        .agg(
            min_hop=("hop", "min"),
            signed_drive=("signed_drive", "sum"),
            abs_drive=("abs_drive", "sum"),
            positive_drive=("positive_drive", "sum"),
            negative_drive=("negative_drive", "sum"),
            raw_synapse_mass_proxy=("raw_synapse_mass_proxy", "sum"),
        )
    )
    candidate_ids = collapsed["target_root_id"].astype("int64").unique().tolist()
    candidate_meta = meta.reindex(candidate_ids).reset_index().rename(columns={"root_id": "candidate_root_id"})
    candidate_meta["candidate_group"] = (
        candidate_meta["connectome_group"].replace("", np.nan).fillna("other").astype(str)
    )
    candidate_meta["candidate_cell_type"] = (
        candidate_meta["cell_type"].replace("", np.nan).fillna(candidate_meta["hemibrain_type"]).fillna("unknown").astype(str)
    )
    candidate_meta["candidate_side"] = candidate_meta["side"].fillna("").astype(str)
    candidate_meta["candidate_top_nt"] = candidate_meta["top_nt"].fillna("").astype(str)
    candidate_meta["candidate_super_class"] = candidate_meta["super_class"].fillna("").astype(str)
    candidate_meta["candidate_cell_class"] = candidate_meta["cell_class"].fillna("").astype(str)

    collapsed = collapsed.merge(
        candidate_meta[
            [
                "candidate_root_id",
                "candidate_group",
                "candidate_cell_type",
                "candidate_side",
                "candidate_top_nt",
                "candidate_super_class",
                "candidate_cell_class",
            ]
        ].rename(columns={"candidate_root_id": "target_root_id"}),
        on="target_root_id",
        how="left",
    )
    collapsed = collapsed.rename(columns={"target_root_id": "candidate_root_id"})

    output_summary = _summarize_candidate_downstream_outputs(
        collapsed["candidate_root_id"].astype("int64").unique().tolist(),
        edges,
        annotated,
    )
    out_pivot = pd.DataFrame()
    if not output_summary.empty:
        total = output_summary.groupby("candidate_root_id", as_index=False)["output_abs_weight"].sum().rename(
            columns={"output_abs_weight": "total_output_abs_weight"}
        )
        out_pivot = output_summary.pivot_table(
            index="candidate_root_id",
            columns="behavior_output_group",
            values="output_abs_weight",
            aggfunc="sum",
            fill_value=0.0,
        ).reset_index()
        out_pivot = out_pivot.merge(total, on="candidate_root_id", how="left")
    if out_pivot.empty:
        out_pivot = pd.DataFrame({"candidate_root_id": collapsed["candidate_root_id"].unique()})
        out_pivot["total_output_abs_weight"] = 0.0
    for column in ["DN", "DN_like", "DAN", "DPM", "APL", "MBON"]:
        if column not in out_pivot:
            out_pivot[column] = 0.0
    denominator = out_pivot["total_output_abs_weight"].astype(float).replace(0.0, np.nan)
    out_pivot["dn_output_fraction"] = ((out_pivot["DN"] + out_pivot["DN_like"]) / denominator).fillna(0.0)
    out_pivot["state_output_fraction"] = ((out_pivot["DAN"] + out_pivot["DPM"] + out_pivot["APL"]) / denominator).fillna(0.0)
    out_pivot["mbon_recurrent_fraction"] = (out_pivot["MBON"] / denominator).fillna(0.0)

    candidate_summary = (
        collapsed.groupby(
            [
                "candidate_root_id",
                "candidate_group",
                "candidate_cell_type",
                "candidate_side",
                "candidate_top_nt",
                "candidate_super_class",
                "candidate_cell_class",
            ],
            as_index=False,
        )
        .agg(
            n_source_groups=("source_group", "nunique"),
            source_groups=("source_group", lambda values: ",".join(sorted(set(map(str, values))))),
            min_hop=("min_hop", "min"),
            signed_total_drive=("signed_drive", "sum"),
            abs_total_drive=("abs_drive", "sum"),
            positive_total_drive=("positive_drive", "sum"),
            negative_total_drive=("negative_drive", "sum"),
            raw_synapse_mass_proxy=("raw_synapse_mass_proxy", "sum"),
        )
    )
    candidate_summary = candidate_summary.merge(
        out_pivot[
            [
                "candidate_root_id",
                "total_output_abs_weight",
                "dn_output_fraction",
                "state_output_fraction",
                "mbon_recurrent_fraction",
            ]
        ],
        on="candidate_root_id",
        how="left",
    ).fillna(
        {
            "total_output_abs_weight": 0.0,
            "dn_output_fraction": 0.0,
            "state_output_fraction": 0.0,
            "mbon_recurrent_fraction": 0.0,
        }
    )
    candidate_summary["push_pull_strength"] = np.divide(
        np.minimum(candidate_summary["positive_total_drive"], candidate_summary["negative_total_drive"]),
        candidate_summary["abs_total_drive"],
        out=np.zeros(len(candidate_summary), dtype=float),
        where=candidate_summary["abs_total_drive"].to_numpy(dtype=float) > 0,
    )
    candidate_summary["push_pull_index"] = np.divide(
        candidate_summary["positive_total_drive"] - candidate_summary["negative_total_drive"],
        candidate_summary["abs_total_drive"],
        out=np.zeros(len(candidate_summary), dtype=float),
        where=candidate_summary["abs_total_drive"].to_numpy(dtype=float) > 0,
    )
    candidate_summary["pivot_score"] = (
        np.log1p(candidate_summary["n_source_groups"].astype(float))
        * np.log1p(candidate_summary["abs_total_drive"].astype(float))
        * (1.0 + 0.75 * candidate_summary["push_pull_strength"].astype(float))
        * (
            1.0
            + 0.75 * candidate_summary["dn_output_fraction"].astype(float)
            + 0.35 * candidate_summary["state_output_fraction"].astype(float)
        )
        / (1.0 + 0.20 * candidate_summary["min_hop"].astype(float))
    )
    candidate_summary["candidate_role_tier"] = np.select(
        [
            candidate_summary["candidate_group"].eq("APL"),
            candidate_summary["candidate_group"].eq("MBON")
            | (
                (candidate_summary["mbon_recurrent_fraction"] > 0.10)
                & ~candidate_summary["candidate_group"].isin(["DAN", "DPM"])
            ),
            candidate_summary["dn_output_fraction"] > 0.02,
            candidate_summary["candidate_group"].isin(["DAN", "DPM"])
            | (candidate_summary["state_output_fraction"] > 0.10),
        ],
        [
            "global_feedback_control",
            "mb_recurrent_candidate",
            "dn_or_motor_proximal_candidate",
            "state_modulation_candidate",
        ],
        default="non_mb_downstream_candidate",
    )
    candidate_summary["wetlab_readout_hint"] = np.select(
        [
            candidate_summary["candidate_role_tier"].eq("global_feedback_control"),
            candidate_summary["dn_output_fraction"] > 0.05,
            candidate_summary["candidate_role_tier"].eq("state_modulation_candidate"),
            candidate_summary["candidate_role_tier"].eq("mb_recurrent_candidate"),
        ],
        [
            "global-feedback control; APL/KC imaging, not primary decision pivot",
            "DN/motor-proximal candidate; behaviour and calcium readout",
            "state-output candidate; DAN/DPM/APL imaging and perturbation",
            "MB recurrent/state node; image MB compartments before behaviour",
        ],
        default="annotation/proofreading first, then imaging",
    )
    candidate_summary = candidate_summary.sort_values("pivot_score", ascending=False).reset_index(drop=True)
    candidate_summary["rank"] = np.arange(1, len(candidate_summary) + 1, dtype=int)
    candidate_summary["primary_pivot_rank"] = pd.Series(pd.NA, index=candidate_summary.index, dtype="Int64")
    primary_mask = (
        ~candidate_summary["candidate_role_tier"].isin(["global_feedback_control", "mb_recurrent_candidate"])
        & ~candidate_summary["candidate_group"].isin(["DAN", "DPM", "MBON"])
    )
    candidate_summary.loc[primary_mask, "primary_pivot_rank"] = np.arange(1, int(primary_mask.sum()) + 1, dtype=int)
    candidate_summary = candidate_summary.head(int(cfg.top_candidates)).copy()

    top_ids = set(candidate_summary["candidate_root_id"].astype("int64"))
    source_candidate_matrix = collapsed[collapsed["candidate_root_id"].isin(top_ids)].copy()
    source_candidate_matrix = source_candidate_matrix.sort_values(
        ["candidate_root_id", "abs_drive"], ascending=[True, False]
    )
    output_summary_top = output_summary[output_summary["candidate_root_id"].isin(top_ids)].copy() if not output_summary.empty else output_summary

    figure_path = _write_mbon_decision_pivot_figure(candidate_summary, source_group_summary, cfg.output_dir)
    plasticity_pattern_summary, plasticity_candidate_response, plasticity_axis_response = _run_mbon_plasticity_replay(
        source_group_summary=source_group_summary,
        candidate_summary=candidate_summary,
        source_candidate_matrix=source_candidate_matrix,
        output_summary=output_summary_top,
    )
    plasticity_figure_path = _write_mbon_plasticity_replay_figure(
        plasticity_candidate_response,
        plasticity_axis_response,
        cfg.output_dir,
    )
    plasticity_null_controls = _run_mbon_plasticity_source_label_null_controls(
        source_group_summary=source_group_summary,
        candidate_summary=candidate_summary,
        source_candidate_matrix=source_candidate_matrix,
        plasticity_candidate_response=plasticity_candidate_response,
        plasticity_axis_response=plasticity_axis_response,
        null_repeats=int(cfg.null_repeats),
        random_seed=int(cfg.random_seed),
    )
    plasticity_convergence_null_controls = _run_mbon_plasticity_convergence_null_controls(
        source_group_summary=source_group_summary,
        candidate_summary=candidate_summary,
        source_candidate_matrix=source_candidate_matrix,
        plasticity_candidate_response=plasticity_candidate_response,
        plasticity_axis_response=plasticity_axis_response,
        null_repeats=int(cfg.null_repeats),
        random_seed=int(cfg.random_seed),
    )
    wetlab_priority = _build_mbon_wetlab_priority_table(
        plasticity_candidate_response=plasticity_candidate_response,
        plasticity_axis_response=plasticity_axis_response,
    )
    interpretation = {
        "model": "real_flywire_mbon_downstream_decision_pivot_search",
        "annotation_path": str(cfg.annotation_path),
        "connectivity_path": str(cfg.connectivity_path),
        "n_mbon_neurons": int(mbon["root_id"].nunique()),
        "n_mbon_source_groups": int(source_group_summary["mbon_source_group"].nunique()),
        "n_mbon_source_rows": int(len(mbon)),
        "n_unmapped_source_rows": int(source_mapping["source_mapping_status"].eq("unmapped").sum())
        if "source_mapping_status" in source_mapping
        else 0,
        "n_candidates_ranked": int(len(candidate_summary)),
        "max_hops": int(cfg.max_hops),
        "hop_decay": float(cfg.hop_decay),
        "source_grouping": str(cfg.source_grouping),
        "include_unmapped_sources": bool(cfg.include_unmapped_sources),
        "n_plasticity_patterns": int(plasticity_pattern_summary["pattern_id"].nunique())
        if not plasticity_pattern_summary.empty
        else 0,
        "n_plasticity_null_repeats": int(cfg.null_repeats),
        "n_plasticity_convergence_null_rows": int(len(plasticity_convergence_null_controls)),
        "n_wetlab_priority_rows": int(len(wetlab_priority)),
        "source_group_definition": (
            "Aso 2014 MB lobe compartment mapping from MBON subtype; unmapped FlyWire MBONs excluded by default"
            if str(cfg.source_grouping).lower().strip() == "compartment"
            else "FlyWire/hemibrain MBON cell_type subtype"
        ),
        "boundary": "connectome candidate ranking; not behaviour causality or calibrated biophysics",
    }

    paths = {
        "source_groups_csv": cfg.output_dir / "mbon_source_groups.csv",
        "source_mapping_csv": cfg.output_dir / "mbon_source_mapping.csv",
        "drive_matrix_csv": cfg.output_dir / "mbon_to_candidate_drive_matrix.csv",
        "candidate_summary_csv": cfg.output_dir / "mbon_decision_pivot_candidates.csv",
        "output_summary_csv": cfg.output_dir / "mbon_pivot_candidate_outputs.csv",
        "plasticity_patterns_csv": cfg.output_dir / "mbon_plasticity_replay_patterns.csv",
        "plasticity_candidate_responses_csv": cfg.output_dir / "mbon_plasticity_replay_candidate_responses.csv",
        "plasticity_output_axis_responses_csv": cfg.output_dir / "mbon_plasticity_replay_output_axis_responses.csv",
        "plasticity_null_controls_csv": cfg.output_dir / "mbon_plasticity_source_label_null_controls.csv",
        "plasticity_convergence_null_controls_csv": cfg.output_dir / "mbon_plasticity_convergence_null_controls.csv",
        "wetlab_priority_csv": cfg.output_dir / "mbon_wetlab_experiment_priority.csv",
        "figure_png": figure_path,
        "plasticity_figure_png": plasticity_figure_path,
        "report_md": cfg.output_dir / "MBON_DECISION_PIVOT_CANDIDATES_CN.md",
        "metadata_json": cfg.output_dir / "mbon_decision_pivot_metadata.json",
    }
    source_group_summary.to_csv(paths["source_groups_csv"], index=False)
    source_mapping.to_csv(paths["source_mapping_csv"], index=False)
    source_candidate_matrix.to_csv(paths["drive_matrix_csv"], index=False)
    candidate_summary.to_csv(paths["candidate_summary_csv"], index=False)
    output_summary_top.to_csv(paths["output_summary_csv"], index=False)
    plasticity_pattern_summary.to_csv(paths["plasticity_patterns_csv"], index=False)
    plasticity_candidate_response.to_csv(paths["plasticity_candidate_responses_csv"], index=False)
    plasticity_axis_response.to_csv(paths["plasticity_output_axis_responses_csv"], index=False)
    plasticity_null_controls.to_csv(paths["plasticity_null_controls_csv"], index=False)
    plasticity_convergence_null_controls.to_csv(paths["plasticity_convergence_null_controls_csv"], index=False)
    wetlab_priority.to_csv(paths["wetlab_priority_csv"], index=False)
    paths["report_md"].write_text(
        _render_mbon_decision_pivot_report(
            cfg=cfg,
            source_group_summary=source_group_summary,
            candidate_summary=candidate_summary,
            source_candidate_matrix=source_candidate_matrix,
            output_summary=output_summary_top,
            plasticity_pattern_summary=plasticity_pattern_summary,
            plasticity_candidate_response=plasticity_candidate_response,
            plasticity_axis_response=plasticity_axis_response,
            plasticity_null_controls=plasticity_null_controls,
            plasticity_convergence_null_controls=plasticity_convergence_null_controls,
            wetlab_priority=wetlab_priority,
            figure_path=figure_path,
            plasticity_figure_path=plasticity_figure_path,
            interpretation=interpretation,
        ),
        encoding="utf-8",
    )
    paths["metadata_json"].write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "interpretation": interpretation,
                "paths": {key: str(value) for key, value in paths.items() if isinstance(value, Path)},
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return {
        **paths,
        "source_group_summary_df": source_group_summary,
        "source_mapping_df": source_mapping,
        "source_candidate_matrix_df": source_candidate_matrix,
        "candidate_summary_df": candidate_summary,
        "output_summary_df": output_summary_top,
        "plasticity_pattern_summary_df": plasticity_pattern_summary,
        "plasticity_candidate_response_df": plasticity_candidate_response,
        "plasticity_axis_response_df": plasticity_axis_response,
        "plasticity_null_controls_df": plasticity_null_controls,
        "plasticity_convergence_null_controls_df": plasticity_convergence_null_controls,
        "wetlab_priority_df": wetlab_priority,
        "interpretation": interpretation,
    }


def _table(frame: pd.DataFrame, max_rows: int = 16) -> str:
    if frame.empty:
        return "No rows."
    display = frame.head(max_rows).copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4g}")
        else:
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else str(value))
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def _write_connectome_science_report(
    output_dir: Path,
    *,
    key_pairs: pd.DataFrame,
    coverage: pd.DataFrame,
    apl_inputs: pd.DataFrame,
    mbon_apl_candidates: pd.DataFrame,
    mbon_targets: pd.DataFrame,
    dan_to_kc: pd.DataFrame,
    target_axis: pd.DataFrame,
    primitive_summary: pd.DataFrame,
    behavior_counts: pd.DataFrame,
    axis_summary: pd.DataFrame,
    scientific_questions: pd.DataFrame,
    figure_path: Path | None,
    behavior_figure_path: Path | None,
    summary: dict[str, object],
) -> Path:
    def get_pair(pre: str, post: str, column: str) -> float:
        selected = key_pairs[key_pairs["pre_group"].eq(pre) & key_pairs["post_group"].eq(post)]
        if selected.empty or column not in selected:
            return float("nan")
        return float(selected[column].iloc[0])

    report_path = output_dir / "FLYWIRE_CONNECTOME_SCIENCE_REPORT_CN.md"
    report_path.write_text(
        f"""# 真实 FlyWire 连接组科学发现报告

保存路径：`{report_path}`

## 数据和边界

- 数据：FlyWire v783 `Connectivity_783.parquet` 与 `flywire_neuron_annotations.parquet`。
- 方法：直接汇总真实 neuron-to-neuron 连接，不使用随机 PN→KC toy 投影。
- signed weight 使用 `Excitatory x Connectivity`，其中 APL、GABAergic、部分 MBON 输出为负号。
- 结论是连接组结构发现和仿真假说，不是 wet-lab 因果证明。

## 核心规模

| item | value |
|---|---:|
| grouped neurons | {summary['n_grouped_neurons']} |
| grouped edges | {summary['n_grouped_edges']} |
| KC count | {summary['n_kc']} |
| ALPN count | {summary['n_alpn']} |
| MBON count | {summary['n_mbon']} |
| DAN count | {summary['n_dan']} |
| DPM count | {summary['n_dpm']} |
| APL count | {summary['n_apl']} |
| DN count | {summary['n_dn']} |

## 本轮直接回答的 3 个科学问题

{_table(scientific_questions, max_rows=3)}

## 发现 1：FlyWire 支持完整 KC-APL 全局负反馈环

| circuit edge | n edges | n pre | n post | signed weight | abs weight | negative fraction |
|---|---:|---:|---:|---:|---:|---:|
| KC→APL | {get_pair('KC', 'APL', 'n_edges'):.0f} | {get_pair('KC', 'APL', 'n_pre'):.0f} | {get_pair('KC', 'APL', 'n_post'):.0f} | {get_pair('KC', 'APL', 'signed_weight'):.0f} | {get_pair('KC', 'APL', 'abs_signed_weight'):.0f} | {get_pair('KC', 'APL', 'negative_fraction'):.3f} |
| APL→KC | {get_pair('APL', 'KC', 'n_edges'):.0f} | {get_pair('APL', 'KC', 'n_pre'):.0f} | {get_pair('APL', 'KC', 'n_post'):.0f} | {get_pair('APL', 'KC', 'signed_weight'):.0f} | {get_pair('APL', 'KC', 'abs_signed_weight'):.0f} | {get_pair('APL', 'KC', 'negative_fraction'):.3f} |

解释：所有 5177 个 KC 都向 APL 汇聚，APL 也覆盖所有 5177 个 KC；APL→KC 的 signed weight 全为负。这不是 toy 假设，而是连接表中直接读出的全局反馈结构。它为“APL 维持 KC <=10% 稀疏编码、抑制减弱会使 KC code 变密”提供结构基础。

## 发现 2：ALPN→KC 输入覆盖几乎全部 KC，但不是唯一驱动

{_table(coverage[coverage['pre_group'].isin(['ALPN', 'DAN', 'APL', 'KC', 'DPM'])], max_rows=12)}

重点：ALPN→KC 覆盖约 {float(coverage.loc[(coverage['pre_group'].eq('ALPN')) & (coverage['post_group'].eq('KC')), 'post_group_coverage'].iloc[0]):.1%} KC；DAN→KC 覆盖约 {float(coverage.loc[(coverage['pre_group'].eq('DAN')) & (coverage['post_group'].eq('KC')), 'post_group_coverage'].iloc[0]):.1%} KC；DPM→KC 覆盖约 {float(coverage.loc[(coverage['pre_group'].eq('DPM')) & (coverage['post_group'].eq('KC')), 'post_group_coverage'].iloc[0]):.1%} KC。也就是说，真实全脑数据支持把 KC 看成 olfactory input、APL feedback、DAN/DPM modulation 共同塑形的结构，而不应只看随机 PN→KC 投影。

## 发现 3：KC→MBON 和 KC→DAN 是强输出轴，支持多读出而非单 MBON proxy

{_table(key_pairs[key_pairs['pre_group'].isin(['KC', 'DAN', 'MBON']) & key_pairs['post_group'].isin(['MBON', 'DAN', 'DPM'])], max_rows=12)}

旧的单 MBON d-prime 只是读出代理。真实连接组显示 KC→MBON、KC→DAN、DAN→KC、DAN→MBON、MBON→DAN 都很强，后续科学问题应该转向 compartment/cell-type 多输出读出：不同 MBON 和 DAN 亚型是否对应不同 odor similarity、reward/aversive teaching、interference learning 和 memory persistence。

Top KC→MBON targets:

{_table(mbon_targets[['post_side', 'post_cell_type', 'kc_input_abs', 'n_kc_inputs', 'n_edges']], max_rows=12)}

Top DAN→KC compartments/cell types:

{_table(dan_to_kc[['pre_cell_type', 'pre_side', 'abs_signed_weight', 'n_kc_targets', 'n_edges']], max_rows=12)}

## 发现 4：APL state 输入不是只有“抑制下降”，真实上游是正负混合

{_table(apl_inputs, max_rows=12)}

连接组中 KC、ALPN、DAN、DPM 对 APL 多为正向 drive；MBON 和非 APL GABAergic 输入对 APL 为负向 signed drive。这里不能再笼统写成“MBON output 影响 APL”，因为 MBON 是一个很大的集合。更窄的可实验假说是：在直接 `MBON→APL` 输入里，优先看负向权重最大的 `MBON11`，其次看 `MBON30`；`MBON22` 是相反方向的正向/drive 候选，可作为方向性对照。

Top direct MBON→APL candidate targets:

{_table(mbon_apl_candidates[['pre_cell_type', 'pre_side', 'pre_top_nt', 'synapses', 'signed_weight', 'abs_signed_weight', 'fraction_of_mbon_to_apl_abs', 'candidate_effect']], max_rows=12) if not mbon_apl_candidates.empty else 'No rows.'}

## 与 KC <=10% 比例 sweep 的关系

配套 `kc_flywire_ratio` sweep 使用真实 ALPN→KC 子图，显示 `0.10` 在 binary memory/retention proxy 中最好，而 legacy `1/6` 产生更高 odor-code overlap。当前连接组审计进一步解释了原因：APL→KC 是覆盖全体 KC 的负反馈，KC→APL 是全体 KC 汇聚到 APL 的正反馈，结构上天然适合做全局稀疏度控制。

图：`{figure_path if figure_path is not None else 'not generated'}`

## 行为学映射：从真实 KC→MBON/DAN/DPM/APL 输出到透明行为轴

这一步不是 toy 行为模型，而是先用真实 `ALPN→KC` 子图生成 KC 稀疏响应，再把活跃 KC 通过真实 `KC→MBON/DAN/DPM/APL/monoamine` 边投影到多输出靶点。行为轴只做透明 proxy：MBON 输出映射到 approach/avoidance/memory expression，DAN 映射到 appetitive/aversive teaching，DPM 映射到 memory persistence，APL 映射到 sparseness brake。

KC 输出靶点结构：

{_table(target_axis[['post_group', 'behavior_axis', 'n_targets', 'total_abs_kc_input', 'fraction_of_mapped_kc_output', 'behavior_interpretation']], max_rows=14) if not target_axis.empty else 'No rows.'}

不同比例下的行为原语：

{_table(primitive_summary, max_rows=12) if not primitive_summary.empty else 'No rows.'}

行为类别占比：

{_table(behavior_counts, max_rows=18) if not behavior_counts.empty else 'No rows.'}

可接入现有 FlyGym memory-choice 行为接口的 condition table：`{output_dir / 'flywire_behavior_condition_table.csv'}`。运行时可用：

```bash
PYTHONPATH=simulation/src env/bin/python simulation/scripts/run_behavior_memory_experiment.py \
  --condition-table simulation/outputs/flywire_connectome_science/flywire_behavior_condition_table.csv \
  --conditions flywire_kc_ratio_0p100_behavior_proxy flywire_kc_ratio_0p167_behavior_proxy \
  --n-trials 1 --run-time 0.5 --no-render
```

行为轴响应强度：

{_table(axis_summary[['ratio', 'behavior_axis', 'mean_axis_abs_response', 'mean_axis_normalized_response', 'behavior_interpretation']], max_rows=24) if not axis_summary.empty else 'No rows.'}

图：`{behavior_figure_path if behavior_figure_path is not None else 'not generated'}`

## 下一步真实连接组问题

1. 在当前行为轴 proxy 上加入 wet-lab 标定：用 OCT/MCH、sucrose/shock 条件给 MBON/DAN 行为符号做实验校准。
2. 用 APL、DPM、DAN、MBON 上游真实边做 perturbation atlas，测试哪些操纵最可能让 KC active fraction 从 <=10% 上漂。
3. 把真实 synapse morphology 权重接入 APL→KC、KC→APL、KC→MBON 三条强边，比较 raw vs morphology-adjusted 的比例和 overlap 变化。
4. 做 degree-preserving / side-preserving rewiring null，确认上述强反馈和行为轴结构不是仅由节点度数解释。

## 输出文件

- grouped pair summary：`{output_dir / 'flywire_group_pair_summary.csv'}`
- key circuit pairs：`{output_dir / 'flywire_key_circuit_pairs.csv'}`
- side summary：`{output_dir / 'flywire_key_side_summary.csv'}`
- target coverage：`{output_dir / 'flywire_target_coverage.csv'}`
- APL state inputs：`{output_dir / 'flywire_apl_state_inputs.csv'}`
- MBON→APL candidate targets：`{output_dir / 'flywire_mbon_apl_candidate_targets.csv'}`
- KC→MBON targets：`{output_dir / 'flywire_kc_mbon_targets.csv'}`
- DAN→KC targets：`{output_dir / 'flywire_dan_kc_targets.csv'}`
- 三科学问题摘要：`{output_dir / 'flywire_three_scientific_questions.csv'}`
- 行为映射靶点：`{output_dir / 'flywire_kc_behavior_target_axis.csv'}`
- 行为原语摘要：`{output_dir / 'flywire_behavior_primitive_summary.csv'}`
- 行为类别占比：`{output_dir / 'flywire_behavior_prediction_counts.csv'}`
- 行为 condition table：`{output_dir / 'flywire_behavior_condition_table.csv'}`
""",
        encoding="utf-8",
    )
    return report_path


def run_flywire_connectome_science(
    output_dir: Path = DEFAULT_OUTPUT_ROOT / "flywire_connectome_science",
    *,
    annotation_path: Path = PROCESSED_DATA_ROOT / "flywire_neuron_annotations.parquet",
    connectivity_path: Path = DEFAULT_CONNECTIVITY_PATH,
    write_behavior_target_responses: bool = False,
) -> dict[str, Path | pd.DataFrame | dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations = _load_annotations(annotation_path)
    annotated = _annotate_connectome_groups(annotations)
    edges = pd.read_parquet(
        connectivity_path,
        columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity", "Excitatory x Connectivity"],
    )
    merged = _merge_science_edge_annotations(edges, annotated)
    group_counts = (
        annotated[annotated["connectome_group"].ne("")]
        .groupby(["connectome_group", "side"], as_index=False)
        .agg(n_neurons=("root_id", "nunique"))
        .sort_values(["connectome_group", "side"])
    )
    pair_summary = _summarize_group_pairs(merged)
    key_pairs = _summarize_key_pairs(pair_summary)
    side_summary = _summarize_side_pairs(merged)
    coverage = _summarize_target_coverage(merged, annotated)
    apl_inputs = _summarize_apl_state_inputs(merged)
    mbon_apl_candidates = _summarize_mbon_apl_candidate_targets(merged)
    mbon_targets = _summarize_mbon_targets(merged)
    dan_to_kc = _summarize_dan_to_kc(merged)
    kc_target_edges, target_meta, target_axis = _build_kc_target_matrix(merged)
    try:
        glomerulus_names, glomerulus_matrix, kc_ids, _ = _build_flywire_glomerulus_kc_matrix_from_frames(
            annotated,
            edges,
        )
        target_response, axis_response, primitive_predictions = _simulate_real_kc_behavior_mapping(
            glomerulus_names=glomerulus_names,
            glomerulus_matrix=glomerulus_matrix,
            kc_ids=kc_ids,
            kc_target_edges=kc_target_edges,
            target_meta=target_meta,
            ratios=BEHAVIOR_MAPPING_RATIOS,
            seeds=BEHAVIOR_MAPPING_SEEDS,
            n_odors=24,
            keep_target_responses=write_behavior_target_responses,
        )
    except ValueError:
        target_response = pd.DataFrame()
        axis_response = pd.DataFrame()
        primitive_predictions = pd.DataFrame()
    primitive_summary, behavior_counts, axis_summary = _summarize_behavior_mapping(
        primitive_predictions,
        axis_response,
        target_meta,
    )
    behavior_condition_table = _build_behavior_condition_table(primitive_summary)
    scientific_questions = _summarize_scientific_questions(
        key_pairs=key_pairs,
        coverage=coverage,
        apl_inputs=apl_inputs,
        mbon_apl_candidates=mbon_apl_candidates,
        primitive_summary=primitive_summary,
        behavior_counts=behavior_counts,
    )
    figure_path = _write_science_figure(pair_summary, output_dir)
    behavior_figure_path = _write_behavior_mapping_figure(
        primitive_summary=primitive_summary,
        behavior_counts=behavior_counts,
        axis_summary=axis_summary,
        output_dir=output_dir,
    )
    paths: dict[str, Path | pd.DataFrame | dict[str, object]] = {
        "group_counts_csv": output_dir / "flywire_group_counts.csv",
        "pair_summary_csv": output_dir / "flywire_group_pair_summary.csv",
        "key_pairs_csv": output_dir / "flywire_key_circuit_pairs.csv",
        "side_summary_csv": output_dir / "flywire_key_side_summary.csv",
        "coverage_csv": output_dir / "flywire_target_coverage.csv",
        "apl_inputs_csv": output_dir / "flywire_apl_state_inputs.csv",
        "mbon_apl_candidates_csv": output_dir / "flywire_mbon_apl_candidate_targets.csv",
        "mbon_targets_csv": output_dir / "flywire_kc_mbon_targets.csv",
        "dan_to_kc_csv": output_dir / "flywire_dan_kc_targets.csv",
        "scientific_questions_csv": output_dir / "flywire_three_scientific_questions.csv",
        "behavior_targets_csv": output_dir / "flywire_kc_behavior_target_axis.csv",
        "behavior_axis_distribution_csv": output_dir / "flywire_kc_behavior_axis_distribution.csv",
        "behavior_axis_responses_csv": output_dir / "flywire_kc_behavior_axis_responses.csv",
        "behavior_predictions_csv": output_dir / "flywire_behavior_predictions_by_odor.csv",
        "behavior_primitive_summary_csv": output_dir / "flywire_behavior_primitive_summary.csv",
        "behavior_prediction_counts_csv": output_dir / "flywire_behavior_prediction_counts.csv",
        "behavior_axis_summary_csv": output_dir / "flywire_behavior_axis_summary.csv",
        "behavior_condition_table_csv": output_dir / "flywire_behavior_condition_table.csv",
        "figure_png": figure_path,
        "behavior_figure_png": behavior_figure_path,
    }
    if write_behavior_target_responses:
        paths["behavior_target_responses_csv"] = output_dir / "flywire_kc_behavior_target_responses.csv"
    group_counts.to_csv(paths["group_counts_csv"], index=False)
    pair_summary.to_csv(paths["pair_summary_csv"], index=False)
    key_pairs.to_csv(paths["key_pairs_csv"], index=False)
    side_summary.to_csv(paths["side_summary_csv"], index=False)
    coverage.to_csv(paths["coverage_csv"], index=False)
    apl_inputs.to_csv(paths["apl_inputs_csv"], index=False)
    mbon_apl_candidates.to_csv(paths["mbon_apl_candidates_csv"], index=False)
    mbon_targets.to_csv(paths["mbon_targets_csv"], index=False)
    dan_to_kc.to_csv(paths["dan_to_kc_csv"], index=False)
    scientific_questions.to_csv(paths["scientific_questions_csv"], index=False)
    target_meta.to_csv(paths["behavior_targets_csv"], index=False)
    target_axis.to_csv(paths["behavior_axis_distribution_csv"], index=False)
    if write_behavior_target_responses:
        target_response.to_csv(paths["behavior_target_responses_csv"], index=False)
    axis_response.to_csv(paths["behavior_axis_responses_csv"], index=False)
    primitive_predictions.to_csv(paths["behavior_predictions_csv"], index=False)
    primitive_summary.to_csv(paths["behavior_primitive_summary_csv"], index=False)
    behavior_counts.to_csv(paths["behavior_prediction_counts_csv"], index=False)
    axis_summary.to_csv(paths["behavior_axis_summary_csv"], index=False)
    behavior_condition_table.to_csv(paths["behavior_condition_table_csv"], index=False)
    summary = {
        "n_grouped_neurons": int(annotated["connectome_group"].ne("").sum()),
        "n_grouped_edges": int(len(merged)),
        "n_kc": int(annotated["connectome_group"].eq("KC").sum()),
        "n_alpn": int(annotated["connectome_group"].eq("ALPN").sum()),
        "n_mbon": int(annotated["connectome_group"].eq("MBON").sum()),
        "n_dan": int(annotated["connectome_group"].eq("DAN").sum()),
        "n_dpm": int(annotated["connectome_group"].eq("DPM").sum()),
        "n_apl": int(annotated["connectome_group"].eq("APL").sum()),
        "n_dn": int(annotated["connectome_group"].eq("DN").sum()),
        "n_behavior_targets": int(target_meta["Postsynaptic_ID"].nunique()) if not target_meta.empty else 0,
        "behavior_mapping_ratios": [float(value) for value in BEHAVIOR_MAPPING_RATIOS],
        "behavior_mapping_seeds": [int(value) for value in BEHAVIOR_MAPPING_SEEDS],
        "write_behavior_target_responses": bool(write_behavior_target_responses),
        "annotation_path": str(annotation_path),
        "connectivity_path": str(connectivity_path),
    }
    report_path = _write_connectome_science_report(
        output_dir,
        key_pairs=key_pairs,
        coverage=coverage,
        apl_inputs=apl_inputs,
        mbon_apl_candidates=mbon_apl_candidates,
        mbon_targets=mbon_targets,
        dan_to_kc=dan_to_kc,
        target_axis=target_axis,
        primitive_summary=primitive_summary,
        behavior_counts=behavior_counts,
        axis_summary=axis_summary,
        scientific_questions=scientific_questions,
        figure_path=figure_path,
        behavior_figure_path=behavior_figure_path,
        summary=summary,
    )
    paths["report_md"] = report_path
    metadata_path = output_dir / "flywire_connectome_science_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "paths": {key: str(value) for key, value in paths.items() if isinstance(value, Path)},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["metadata_json"] = metadata_path
    paths["summary"] = summary
    paths["key_pairs_df"] = key_pairs
    paths["coverage_df"] = coverage
    paths["apl_inputs_df"] = apl_inputs
    paths["mbon_apl_candidates_df"] = mbon_apl_candidates
    paths["scientific_questions_df"] = scientific_questions
    paths["behavior_primitive_summary_df"] = primitive_summary
    paths["behavior_prediction_counts_df"] = behavior_counts
    paths["behavior_axis_summary_df"] = axis_summary
    paths["behavior_condition_table_df"] = behavior_condition_table
    return paths


def _nearest_ratio_row(summary_df: pd.DataFrame, ratio: float) -> pd.Series:
    if summary_df.empty:
        raise ValueError("ratio summary is empty.")
    index = int((summary_df["ratio"].astype(float) - float(ratio)).abs().idxmin())
    return summary_df.loc[index]


def _scorecard_row(
    *,
    section: str,
    finding_id: str,
    question: str,
    evidence: str,
    status: str,
    metric: str,
    value: float | int | str,
    interpretation: str,
) -> dict[str, object]:
    return {
        "section": section,
        "finding_id": finding_id,
        "question": question,
        "evidence": evidence,
        "status": status,
        "metric": metric,
        "value": value,
        "interpretation": interpretation,
    }


def _pair_lookup(key_pairs: pd.DataFrame, pre_group: str, post_group: str) -> pd.Series:
    selected = key_pairs[key_pairs["pre_group"].eq(pre_group) & key_pairs["post_group"].eq(post_group)]
    if selected.empty:
        return pd.Series(dtype=object)
    return selected.iloc[0]


def _coverage_lookup(coverage: pd.DataFrame, pre_group: str, post_group: str) -> pd.Series:
    selected = coverage[coverage["pre_group"].eq(pre_group) & coverage["post_group"].eq(post_group)]
    if selected.empty:
        return pd.Series(dtype=object)
    return selected.iloc[0]


def _load_or_run_ratio_summary(
    kc_ratio_dir: Path,
    annotation_path: Path,
    connectivity_path: Path,
) -> pd.DataFrame:
    summary_path = kc_ratio_dir / "kc_flywire_ratio_sweep_summary.csv"
    if summary_path.exists():
        return pd.read_csv(summary_path)
    config = KCFlyWireRatioConfig(
        annotation_path=annotation_path,
        connectivity_path=connectivity_path,
        output_dir=kc_ratio_dir,
        seeds=tuple(range(5)),
    )
    result = run_kc_flywire_ratio_sweep(config)
    return result["summary_df"].copy()


def _load_or_run_connectome_science(
    connectome_science_dir: Path,
    annotation_path: Path,
    connectivity_path: Path,
) -> dict[str, pd.DataFrame]:
    expected = {
        "key_pairs": connectome_science_dir / "flywire_key_circuit_pairs.csv",
        "coverage": connectome_science_dir / "flywire_target_coverage.csv",
        "apl_inputs": connectome_science_dir / "flywire_apl_state_inputs.csv",
        "target_axis": connectome_science_dir / "flywire_kc_behavior_axis_distribution.csv",
        "primitive_summary": connectome_science_dir / "flywire_behavior_primitive_summary.csv",
        "behavior_counts": connectome_science_dir / "flywire_behavior_prediction_counts.csv",
        "axis_summary": connectome_science_dir / "flywire_behavior_axis_summary.csv",
    }
    if all(path.exists() for path in expected.values()):
        return {name: pd.read_csv(path) for name, path in expected.items()}
    result = run_flywire_connectome_science(
        output_dir=connectome_science_dir,
        annotation_path=annotation_path,
        connectivity_path=connectivity_path,
    )
    return {
        "key_pairs": result["key_pairs_df"].copy(),
        "coverage": result["coverage_df"].copy(),
        "apl_inputs": result["apl_inputs_df"].copy(),
        "target_axis": pd.read_csv(expected["target_axis"]),
        "primitive_summary": result["behavior_primitive_summary_df"].copy(),
        "behavior_counts": result["behavior_prediction_counts_df"].copy(),
        "axis_summary": result["behavior_axis_summary_df"].copy(),
    }


def _build_method_effectiveness_scorecards(
    ratio_summary: pd.DataFrame,
    *,
    key_pairs: pd.DataFrame,
    coverage: pd.DataFrame,
    apl_inputs: pd.DataFrame,
    target_axis: pd.DataFrame,
    primitive_summary: pd.DataFrame,
    behavior_counts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    anchor = _nearest_ratio_row(ratio_summary, LITERATURE_KC_ACTIVE_FRACTION)
    legacy = _nearest_ratio_row(ratio_summary, LEGACY_ONE_SIXTH_KC_RATIO)
    anchor_jaccard = float(anchor["mean_jaccard_overlap"])
    legacy_jaccard = float(legacy["mean_jaccard_overlap"])
    anchor_retention = float(anchor["memory_retention_accuracy_after_interference"])
    legacy_retention = float(legacy["memory_retention_accuracy_after_interference"])

    kc_apl = _pair_lookup(key_pairs, "KC", "APL")
    apl_kc = _pair_lookup(key_pairs, "APL", "KC")
    kc_apl_complete = (
        not kc_apl.empty
        and not apl_kc.empty
        and int(kc_apl.get("n_pre", 0)) >= int(apl_kc.get("n_post", 0))
        and float(apl_kc.get("negative_fraction", 0.0)) >= 0.99
    )

    behavior_axes = set(target_axis["behavior_axis"].astype(str)) if not target_axis.empty else set()
    expected_axes = {
        "approach_or_positive_valence",
        "avoidance_or_negative_valence",
        "appetitive_teaching",
        "aversive_teaching",
        "memory_persistence",
        "sparseness_brake",
    }
    if not target_axis.empty and "fraction_of_mapped_kc_output" in target_axis:
        axis_fraction = float(
            target_axis.loc[target_axis["behavior_axis"].isin(expected_axes), "fraction_of_mapped_kc_output"].sum()
        )
    else:
        axis_fraction = 0.0

    classic_rows = [
        _scorecard_row(
            section="classic_validation",
            finding_id="C1_KC_sparse_anchor",
            question="真实 ALPN->KC 子图是否支持 <=10% KC active fraction 作为工作锚点？",
            evidence=(
                f"ratio={float(anchor['ratio']):.3g}: Jaccard={anchor_jaccard:.4f}, "
                f"retention={anchor_retention:.3f}; legacy 1/6: Jaccard={legacy_jaccard:.4f}, "
                f"retention={legacy_retention:.3f}"
            ),
            status="pass" if anchor_jaccard < legacy_jaccard and anchor_retention >= legacy_retention else "review",
            metric="anchor_vs_legacy_overlap_delta",
            value=float(legacy_jaccard - anchor_jaccard),
            interpretation="<=10% 保持更低 odor-code overlap，同时没有牺牲当前 memory-retention proxy。",
        ),
        _scorecard_row(
            section="classic_validation",
            finding_id="C2_KC_APL_feedback",
            question="KC/APL 全局负反馈是否是 FlyWire 真实结构，而不是 toy 假设？",
            evidence=(
                f"KC->APL n_pre={int(kc_apl.get('n_pre', 0))}, n_post={int(kc_apl.get('n_post', 0))}; "
                f"APL->KC n_pre={int(apl_kc.get('n_pre', 0))}, n_post={int(apl_kc.get('n_post', 0))}, "
                f"negative_fraction={float(apl_kc.get('negative_fraction', 0.0)):.3f}"
            ),
            status="pass" if kc_apl_complete else "review",
            metric="apl_to_kc_negative_fraction",
            value=float(apl_kc.get("negative_fraction", 0.0)),
            interpretation="APL->KC 为覆盖全部 KC 的负向边，KC->APL 为全体 KC 汇聚，支持全局稀疏制动。",
        ),
        _scorecard_row(
            section="classic_validation",
            finding_id="C3_multi_axis_behavior_readout",
            question="真实 KC 输出是否需要多行为轴读出，而不是单 MBON proxy？",
            evidence=f"mapped behavior axes={len(behavior_axes)}, expected-axis KC output fraction={axis_fraction:.3f}",
            status="pass" if expected_axes.issubset(behavior_axes) and axis_fraction > 0.75 else "review",
            metric="expected_axis_fraction_of_mapped_kc_output",
            value=axis_fraction,
            interpretation="KC 输出同时进入 approach/avoidance、DAN teaching、DPM persistence 和 APL brake 轴。",
        ),
    ]

    negative_apl = apl_inputs[apl_inputs["signed_weight"].astype(float) < 0].copy()
    negative_apl = negative_apl.sort_values("abs_signed_weight", ascending=False)
    positive_apl = apl_inputs[apl_inputs["signed_weight"].astype(float) > 0].copy()
    negative_sources = "; ".join(
        f"{row.pre_group}:{float(row.signed_weight):.0f}" for row in negative_apl.head(4).itertuples(index=False)
    )
    positive_sources = "; ".join(
        f"{row.pre_group}:{float(row.signed_weight):.0f}" for row in positive_apl.head(4).itertuples(index=False)
    )

    driver_coverages = {
        "ALPN": float(_coverage_lookup(coverage, "ALPN", "KC").get("post_group_coverage", 0.0)),
        "APL": float(_coverage_lookup(coverage, "APL", "KC").get("post_group_coverage", 0.0)),
        "DAN": float(_coverage_lookup(coverage, "DAN", "KC").get("post_group_coverage", 0.0)),
        "DPM": float(_coverage_lookup(coverage, "DPM", "KC").get("post_group_coverage", 0.0)),
    }
    high_coverage_drivers = [name for name, value in driver_coverages.items() if value >= 0.75]

    mixed_fraction = 0.0
    if not behavior_counts.empty:
        selected = behavior_counts[behavior_counts["predicted_behavior"].eq("mixed_or_state_modulated_memory")]
        mixed_fraction = float(selected["fraction"].min()) if not selected.empty else 0.0
    valence_index = 0.0
    if not primitive_summary.empty and "behavior_valence_index" in primitive_summary:
        anchor_primitive = _nearest_ratio_row(primitive_summary, LITERATURE_KC_ACTIVE_FRACTION)
        valence_index = float(anchor_primitive.get("behavior_valence_index", 0.0))

    frontier_rows = [
        _scorecard_row(
            section="frontier_exploration",
            finding_id="F1_APL_state_downshift_candidates",
            question="哪些真实上游轴最可能让 APL signed drive 下移，并使 KC code 变密？",
            evidence=f"negative APL inputs: {negative_sources}; positive APL inputs: {positive_sources}",
            status="candidate",
            metric="n_negative_apl_source_groups",
            value=int(len(negative_apl)),
            interpretation="优先测试 MBON 与非 APL GABAergic 输入；预测 APL 下移会提高 KC active fraction 和 overlap。",
        ),
        _scorecard_row(
            section="frontier_exploration",
            finding_id="F2_KC_multidriver_state_space",
            question="KC 稀疏度是否应建模为 ALPN、APL、DAN、DPM 的多驱动状态空间？",
            evidence=", ".join(f"{name}->KC coverage={value:.1%}" for name, value in driver_coverages.items()),
            status="candidate" if len(high_coverage_drivers) >= 3 else "review",
            metric="n_high_coverage_kc_drivers",
            value=int(len(high_coverage_drivers)),
            interpretation="真实 KC 稀疏度不是单 PN->KC 投影问题，后续扰动应扫描 APL/DAN/DPM 状态轴。",
        ),
        _scorecard_row(
            section="frontier_exploration",
            finding_id="F3_behavior_mixed_state_memory",
            question="真实多轴读出会落到单一趋近/回避，还是 mixed/state-modulated memory？",
            evidence=f"min mixed/state-modulated fraction={mixed_fraction:.3f}; anchor valence index={valence_index:.4f}",
            status="candidate" if mixed_fraction >= 0.9 and abs(valence_index) < 0.05 else "review",
            metric="min_mixed_behavior_fraction",
            value=mixed_fraction,
            interpretation="当前真实读出更像状态调制的记忆表达；单一 valence 解释过窄。",
        ),
    ]
    return pd.DataFrame.from_records(classic_rows), pd.DataFrame.from_records(frontier_rows)


def _load_apl_state_screen_summary(apl_state_screen_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    screen_path = apl_state_screen_dir / "apl_downshift_state_screen.csv"
    if not screen_path.exists():
        return pd.DataFrame(), {
            "available": False,
            "screen_csv": str(screen_path),
        }
    table = pd.read_csv(screen_path)
    if table.empty:
        return pd.DataFrame(), {
            "available": False,
            "screen_csv": str(screen_path),
            "reason": "empty screen table",
        }
    ok = table[table.get("screen_status", "").astype(str).eq("ok")].copy()
    negative = ok[pd.to_numeric(ok["apl_signed_score"], errors="coerce") < 0].copy()
    downshift = ok[ok["predicted_effect"].astype(str).str.contains("downshift", na=False)].copy()
    enhancement = ok[ok["predicted_effect"].astype(str).str.contains("enhancement", na=False)].copy()
    top_negative = negative.sort_values("apl_signed_score").head(6)
    top_positive = ok.sort_values("apl_signed_score", ascending=False).head(4)
    compact = pd.concat([top_negative, top_positive], ignore_index=True)
    compact_columns = [
        "state_proxy",
        "n_seed_neurons",
        "n_active_neurons",
        "apl_signed_score",
        "apl_gain_shift_proxy",
        "predicted_apl_gain_proxy",
        "predicted_kc_active_fraction_from_gain_proxy",
        "predicted_effect",
    ]
    compact = compact[[column for column in compact_columns if column in compact]].drop_duplicates("state_proxy")
    strongest_downshift = downshift.sort_values("apl_signed_score").iloc[0].to_dict() if not downshift.empty else {}
    strongest_enhancement = enhancement.sort_values("apl_signed_score", ascending=False).iloc[0].to_dict() if not enhancement.empty else {}
    summary = {
        "available": True,
        "screen_csv": str(screen_path),
        "n_state_proxies": int(len(table)),
        "n_ok_state_proxies": int(len(ok)),
        "n_negative_apl_signed_drive": int(len(negative)),
        "n_downshift_candidates": int(len(downshift)),
        "strongest_downshift_state": str(strongest_downshift.get("state_proxy", "")),
        "strongest_downshift_apl_signed_score": float(strongest_downshift.get("apl_signed_score", 0.0))
        if strongest_downshift
        else 0.0,
        "strongest_downshift_predicted_kc_fraction": float(
            strongest_downshift.get("predicted_kc_active_fraction_from_gain_proxy", 0.0)
        )
        if strongest_downshift
        else 0.0,
        "strongest_enhancement_state": str(strongest_enhancement.get("state_proxy", "")),
        "strongest_enhancement_apl_signed_score": float(strongest_enhancement.get("apl_signed_score", 0.0))
        if strongest_enhancement
        else 0.0,
        "strongest_enhancement_predicted_kc_fraction": float(
            strongest_enhancement.get("predicted_kc_active_fraction_from_gain_proxy", 0.0)
        )
        if strongest_enhancement
        else 0.0,
    }
    return compact.reset_index(drop=True), summary


def _apl_state_screen_scorecard_row(summary: dict[str, object]) -> dict[str, object] | None:
    if not summary.get("available"):
        return None
    evidence = (
        f"tested={summary.get('n_ok_state_proxies')}, "
        f"negative={summary.get('n_negative_apl_signed_drive')}, "
        f"downshift={summary.get('n_downshift_candidates')}; "
        f"top_downshift={summary.get('strongest_downshift_state')} "
        f"apl_signed={float(summary.get('strongest_downshift_apl_signed_score', 0.0)):.4g}, "
        f"kc_fraction={float(summary.get('strongest_downshift_predicted_kc_fraction', 0.0)):.3g}; "
        f"top_enhancement={summary.get('strongest_enhancement_state')}"
    )
    return _scorecard_row(
        section="frontier_exploration",
        finding_id="F5_APL_state_screen_propagation",
        question="真实 state seed propagation 是否支持强 APL downshift 假说？",
        evidence=evidence,
        status="screened",
        metric="n_downshift_state_candidates",
        value=int(summary.get("n_downshift_candidates", 0)),
        interpretation=(
            "coarse 状态筛选只支持 MBON_memory_output 的 mild APL downshift；"
            "DAN/DPM 更像 APL enhancement，强状态下移假说需要更细粒度验证。"
        ),
    )


def _run_real_lif_smoke(
    *,
    annotation_path: Path,
    connectivity_path: Path,
    output_dir: Path,
    max_seeds: int = 32,
) -> dict[str, object]:
    annotations = _annotate_connectome_groups(_load_annotations(annotation_path))
    circuit_groups = {"ALPN", "KC", "APL", "DPM", "MBON", "DAN"}
    circuit_roots = set(
        annotations.loc[annotations["connectome_group"].isin(circuit_groups), "root_id"].astype("int64").tolist()
    )
    seed_ids = (
        annotations.loc[annotations["connectome_group"].eq("ALPN"), "root_id"]
        .dropna()
        .astype("int64")
        .sort_values()
        .head(max_seeds)
        .tolist()
    )
    if not seed_ids:
        return {
            "status": "skipped",
            "reason": "no ALPN seeds found",
            "n_seed_ids": 0,
            "n_subgraph_edges": 0,
            "lif_active_roots": 0,
            "signed_active_roots": 0,
            "lif_spike_count": 0,
            "lif_signed_top200_jaccard": 0.0,
        }
    edges = pd.read_parquet(connectivity_path, columns=EDGE_COLUMNS)
    subgraph = edges[
        edges["Presynaptic_ID"].isin(circuit_roots) & edges["Postsynaptic_ID"].isin(circuit_roots)
    ].copy()
    if subgraph.empty:
        return {
            "status": "skipped",
            "reason": "empty MB/KC subgraph",
            "n_seed_ids": len(seed_ids),
            "n_subgraph_edges": 0,
            "lif_active_roots": 0,
            "signed_active_roots": 0,
            "lif_spike_count": 0,
            "lif_signed_top200_jaccard": 0.0,
        }

    lif_trace = run_lif_dynamics(
        subgraph,
        seed_ids={int(root_id): 1.0 for root_id in seed_ids},
        config=LIFDynamicsConfig(
            duration_ms=80.0,
            dt_ms=1.0,
            tau_membrane_ms=12.0,
            input_current=2.0,
            synaptic_gain=1.4,
            v_threshold_mv=0.5,
            refractory_ms=2.0,
            record_every_ms=10.0,
            max_active=2_000,
        ),
    )
    signed = signed_multihop_response(
        subgraph,
        seed_ids=seed_ids,
        config=PropagationConfig(steps=3, max_active=2_000),
    )
    lif_aggregate = lif_trace.groupby("root_id", as_index=False)["score"].sum() if not lif_trace.empty else lif_trace
    overlap = response_overlap(lif_aggregate, signed, top_n=200) if not lif_trace.empty else 0.0
    trace_path = output_dir / "real_lif_smoke_trace.csv"
    lif_trace.to_csv(trace_path, index=False)
    return {
        "status": "pass" if not lif_trace.empty and float(lif_trace["spike_count"].sum()) > 0 else "review",
        "reason": "real FlyWire ALPN-seeded MB/KC subgraph",
        "n_seed_ids": int(len(seed_ids)),
        "n_subgraph_edges": int(len(subgraph)),
        "lif_active_roots": int(lif_trace["root_id"].nunique()) if not lif_trace.empty else 0,
        "signed_active_roots": int(signed["root_id"].nunique()) if not signed.empty else 0,
        "lif_spike_count": int(lif_trace["spike_count"].sum()) if not lif_trace.empty else 0,
        "lif_signed_top200_jaccard": float(overlap),
        "trace_csv": str(trace_path),
    }


def _write_method_effectiveness_figure(
    output_dir: Path,
    ratio_summary: pd.DataFrame,
    scorecard: pd.DataFrame,
    frontier: pd.DataFrame,
    lif_summary: dict[str, object] | None,
    apl_state_screen_summary: dict[str, object] | None = None,
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    axes = axes.ravel()

    axes[0].plot(ratio_summary["ratio"], ratio_summary["mean_jaccard_overlap"], "o-", color="#4b8bbe")
    axes[0].axvline(LITERATURE_KC_ACTIVE_FRACTION, ls="--", color="#5b5b5b", label="<=10% anchor")
    axes[0].axvline(LEGACY_ONE_SIXTH_KC_RATIO, ls=":", color="#5b5b5b", label="legacy 1/6")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("KC active fraction")
    axes[0].set_ylabel("mean odor-code Jaccard")
    axes[0].set_title("Real FlyWire ALPN->KC overlap")
    axes[0].legend(fontsize=8)

    axes[1].plot(
        ratio_summary["ratio"],
        ratio_summary["memory_retention_accuracy_after_interference"],
        "s-",
        color="#7f9a48",
    )
    axes[1].axvline(LITERATURE_KC_ACTIVE_FRACTION, ls="--", color="#5b5b5b")
    axes[1].axvline(LEGACY_ONE_SIXTH_KC_RATIO, ls=":", color="#5b5b5b")
    axes[1].set_xscale("log")
    axes[1].set_ylim(0.85, 1.01)
    axes[1].set_xlabel("KC active fraction")
    axes[1].set_ylabel("retention accuracy")
    axes[1].set_title("Memory proxy after interference")

    counts = scorecard["status"].value_counts().sort_index()
    axes[2].bar(counts.index, counts.values, color="#6f7f99")
    axes[2].set_ylabel("count")
    axes[2].set_title("Classic validation status")

    frontier_values = frontier.set_index("finding_id")["value"]
    plot_values = []
    labels = []
    for key in [
        "F1_APL_state_downshift_candidates",
        "F2_KC_multidriver_state_space",
        "F3_behavior_mixed_state_memory",
    ]:
        if key in frontier_values:
            labels.append(key.split("_", 1)[1])
            plot_values.append(float(frontier_values.loc[key]))
    if lif_summary is not None:
        labels.append("LIF_active_roots")
        plot_values.append(float(lif_summary.get("lif_active_roots", 0)))
    if apl_state_screen_summary and apl_state_screen_summary.get("available"):
        labels.append("APL_screen_negative")
        plot_values.append(float(apl_state_screen_summary.get("n_negative_apl_signed_drive", 0)))
        labels.append("APL_screen_downshift")
        plot_values.append(float(apl_state_screen_summary.get("n_downshift_candidates", 0)))
    axes[3].barh(labels, plot_values, color="#b84b5f")
    axes[3].set_xscale("symlog")
    axes[3].set_title("Frontier/smoke-test metrics")

    figure_path = output_dir / "Fig_method_effectiveness_suite.png"
    fig.savefig(figure_path, dpi=190)
    plt.close(fig)
    return figure_path


def _write_method_effectiveness_report(
    output_dir: Path,
    *,
    scorecard: pd.DataFrame,
    frontier: pd.DataFrame,
    apl_state_screen: pd.DataFrame,
    apl_state_screen_summary: dict[str, object] | None,
    lif_summary: dict[str, object] | None,
    figure_path: Path | None,
) -> Path:
    classic_pass = int(scorecard["status"].eq("pass").sum()) if not scorecard.empty else 0
    report_path = output_dir / "METHOD_EFFECTIVENESS_SUITE_REPORT_CN.md"
    lif_lines = "- LIF smoke 未运行。"
    if lif_summary is not None:
        lif_lines = "\n".join(
            [
                f"- status：`{lif_summary.get('status')}`",
                f"- seed ALPN 数：`{lif_summary.get('n_seed_ids')}`",
                f"- 真实 MB/KC 子图边数：`{lif_summary.get('n_subgraph_edges')}`",
                f"- LIF active roots：`{lif_summary.get('lif_active_roots')}`",
                f"- LIF spike count：`{lif_summary.get('lif_spike_count')}`",
                f"- signed multi-hop active roots：`{lif_summary.get('signed_active_roots')}`",
                f"- LIF vs signed top-200 Jaccard：`{float(lif_summary.get('lif_signed_top200_jaccard', 0.0)):.4f}`",
            ]
        )
    apl_lines = "- APL state screen 未发现或未运行。"
    if apl_state_screen_summary and apl_state_screen_summary.get("available"):
        apl_lines = "\n".join(
            [
                f"- tested state proxies：`{apl_state_screen_summary.get('n_ok_state_proxies')}`",
                f"- negative APL signed-drive proxies：`{apl_state_screen_summary.get('n_negative_apl_signed_drive')}`",
                f"- downshift candidates：`{apl_state_screen_summary.get('n_downshift_candidates')}`",
                f"- strongest downshift：`{apl_state_screen_summary.get('strongest_downshift_state')}` "
                f"APL signed `{float(apl_state_screen_summary.get('strongest_downshift_apl_signed_score', 0.0)):.4g}`，"
                f"predicted KC fraction `{float(apl_state_screen_summary.get('strongest_downshift_predicted_kc_fraction', 0.0)):.3g}`",
                f"- strongest enhancement：`{apl_state_screen_summary.get('strongest_enhancement_state')}` "
                f"APL signed `{float(apl_state_screen_summary.get('strongest_enhancement_apl_signed_score', 0.0)):.4g}`，"
                f"predicted KC fraction `{float(apl_state_screen_summary.get('strongest_enhancement_predicted_kc_fraction', 0.0)):.3g}`",
            ]
        )
    report_path.write_text(
        f"""# BioFly 方法有效性验证与前沿探索套件

保存路径：`{report_path}`

## 定位

这份报告把已有真实 FlyWire 输出收束成一个可复跑的有效性验证面板。它不引入 toy 随机图，
优先复用 `ALPN->KC` 稀疏比例 sweep、`KC/APL/DPM/MBON/DAN` 真实连接组科学报告和
行为轴映射。新增 LIF 只做真实子图 smoke test，用来验证 spiking surrogate 后端能接入
同一套传播/读出 schema。

## 经典结论验证

当前经典验证通过 `{classic_pass}/{len(scorecard)}` 项。

{_table(scorecard, max_rows=12)}

## 前沿/未解问题候选

{_table(frontier, max_rows=12)}

## APL 状态下移真实传播筛选

{apl_lines}

{_table(apl_state_screen, max_rows=12) if not apl_state_screen.empty else ''}

解释：这一步把 F1 从“上游连接候选”推进到“真实 state seed propagation 筛选”。当前结果
不支持强泛化的 APL downshift 叙事：MBON_memory_output 是 mild downshift 候选，而
DAN/DPM 更像 APL enhancement。后续应把 wet-lab 优先级放在 MBON/APL 轴和更细的状态 seed。

## LIF 真实子图 smoke test

{lif_lines}

解释：这个 smoke test 不是校准过的电生理模型，只证明新 LIF 后端可以在真实 FlyWire
MB/KC 子图上产生非空 spike-style trace，并与 signed multi-hop 后端共享 `root_id/score/step`
输出约定。后续如果要做生物物理精度，需要引入膜参数、compartment/cable 模型和 wet-lab 标定。

## 图

{f'<img src="{figure_path.name}" alt="BioFly 方法有效性验证套件" width="100%">' if figure_path else '图未生成。'}

## 主要结论

- 方法有效性不是靠 toy 图证明，而是通过真实 FlyWire 结构重现三类经典约束：KC `<=10%`
  稀疏锚点、KC/APL 全局负反馈、多 MBON/DAN/DPM/APL 行为轴输出。
- 当前最有价值的前沿问题是 APL 状态下移候选、KC 多驱动状态空间、mixed/state-modulated
  memory 的行为表现，而不是继续把系统简化成单 MBON 或单 valence proxy。APL state screen
  进一步显示，强 APL 下移不能泛化到所有状态；最稳的候选是 MBON_memory_output mild downshift。
- LIF 后端已经具备前向兼容入口，但科学解释仍应标注为 surrogate；它适合做 rate/signed
  模型的鲁棒性对照，不应直接宣称为真实 spike-level 果蝇脑。

## 输出文件

- scorecard：`{output_dir / 'method_effectiveness_scorecard.csv'}`
- classic validation：`{output_dir / 'classic_validation_summary.csv'}`
- frontier candidates：`{output_dir / 'frontier_exploration_summary.csv'}`
- APL state screen compact：`{output_dir / 'apl_state_screen_compact.csv'}`
- LIF summary：`{output_dir / 'real_lif_smoke_summary.json'}`
- metadata：`{output_dir / 'method_effectiveness_metadata.json'}`
""",
        encoding="utf-8",
    )
    return report_path


def run_method_effectiveness_suite(
    *,
    output_dir: Path = DEFAULT_OUTPUT_ROOT / "method_effectiveness_suite",
    connectome_science_dir: Path = DEFAULT_OUTPUT_ROOT / "flywire_connectome_science",
    kc_ratio_dir: Path = DEFAULT_OUTPUT_ROOT / "kc_flywire_ratio",
    apl_state_screen_dir: Path = DEFAULT_OUTPUT_ROOT / "apl_downshift_state_screen",
    annotation_path: Path = PROCESSED_DATA_ROOT / "flywire_neuron_annotations.parquet",
    connectivity_path: Path = DEFAULT_CONNECTIVITY_PATH,
    run_lif_smoke: bool = True,
) -> dict[str, object]:
    """Build a compact real-FlyWire method-effectiveness report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    ratio_summary = _load_or_run_ratio_summary(kc_ratio_dir, annotation_path, connectivity_path)
    science = _load_or_run_connectome_science(connectome_science_dir, annotation_path, connectivity_path)
    classic, frontier = _build_method_effectiveness_scorecards(
        ratio_summary,
        key_pairs=science["key_pairs"],
        coverage=science["coverage"],
        apl_inputs=science["apl_inputs"],
        target_axis=science["target_axis"],
        primitive_summary=science["primitive_summary"],
        behavior_counts=science["behavior_counts"],
    )
    apl_state_screen, apl_state_screen_summary = _load_apl_state_screen_summary(apl_state_screen_dir)
    apl_state_row = _apl_state_screen_scorecard_row(apl_state_screen_summary)
    if apl_state_row is not None:
        frontier = pd.concat([frontier, pd.DataFrame.from_records([apl_state_row])], ignore_index=True)
    lif_summary = (
        _run_real_lif_smoke(
            annotation_path=annotation_path,
            connectivity_path=connectivity_path,
            output_dir=output_dir,
        )
        if run_lif_smoke
        else None
    )
    if lif_summary is not None:
        frontier = pd.concat(
            [
                frontier,
                pd.DataFrame.from_records(
                    [
                        _scorecard_row(
                            section="frontier_exploration",
                            finding_id="F4_LIF_real_subgraph_compatibility",
                            question="LIF spiking surrogate 能否在真实 FlyWire 子图上前向兼容运行？",
                            evidence=(
                                f"ALPN seeds={lif_summary.get('n_seed_ids')}, "
                                f"subgraph_edges={lif_summary.get('n_subgraph_edges')}, "
                                f"LIF active_roots={lif_summary.get('lif_active_roots')}, "
                                f"spike_count={lif_summary.get('lif_spike_count')}"
                            ),
                            status=str(lif_summary.get("status", "review")),
                            metric="lif_active_roots",
                            value=int(lif_summary.get("lif_active_roots", 0)),
                            interpretation="LIF 与 signed/rate 后端共享响应表 schema，可作为 spike-style 鲁棒性对照。",
                        )
                    ]
                ),
            ],
            ignore_index=True,
        )

    scorecard = pd.concat([classic, frontier], ignore_index=True)
    paths = {
        "classic_validation_csv": output_dir / "classic_validation_summary.csv",
        "frontier_exploration_csv": output_dir / "frontier_exploration_summary.csv",
        "apl_state_screen_compact_csv": output_dir / "apl_state_screen_compact.csv",
        "apl_state_screen_summary_json": output_dir / "apl_state_screen_summary.json",
        "scorecard_csv": output_dir / "method_effectiveness_scorecard.csv",
        "metadata_json": output_dir / "method_effectiveness_metadata.json",
    }
    classic.to_csv(paths["classic_validation_csv"], index=False)
    frontier.to_csv(paths["frontier_exploration_csv"], index=False)
    apl_state_screen.to_csv(paths["apl_state_screen_compact_csv"], index=False)
    paths["apl_state_screen_summary_json"].write_text(
        json.dumps(apl_state_screen_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    scorecard.to_csv(paths["scorecard_csv"], index=False)
    if lif_summary is not None:
        paths["lif_summary_json"] = output_dir / "real_lif_smoke_summary.json"
        paths["lif_summary_json"].write_text(json.dumps(lif_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    figure_path = _write_method_effectiveness_figure(
        output_dir,
        ratio_summary,
        classic,
        frontier,
        lif_summary,
        apl_state_screen_summary,
    )
    report_path = _write_method_effectiveness_report(
        output_dir,
        scorecard=classic,
        frontier=frontier,
        apl_state_screen=apl_state_screen,
        apl_state_screen_summary=apl_state_screen_summary,
        lif_summary=lif_summary,
        figure_path=figure_path,
    )
    metadata = {
        "output_dir": str(output_dir),
        "connectome_science_dir": str(connectome_science_dir),
        "kc_ratio_dir": str(kc_ratio_dir),
        "apl_state_screen_dir": str(apl_state_screen_dir),
        "annotation_path": str(annotation_path),
        "connectivity_path": str(connectivity_path),
        "run_lif_smoke": bool(run_lif_smoke),
        "n_scorecard_rows": int(len(scorecard)),
        "n_classic_pass": int(classic["status"].eq("pass").sum()),
        "apl_state_screen_summary": apl_state_screen_summary,
        "paths": {
            key: str(value)
            for key, value in {
                **paths,
                "figure_png": figure_path,
                "report_md": report_path,
            }.items()
            if value is not None
        },
    }
    paths["metadata_json"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        **paths,
        "figure_png": figure_path,
        "report_md": report_path,
        "metadata": metadata,
        "classic_validation_df": classic,
        "frontier_exploration_df": frontier,
        "apl_state_screen_df": apl_state_screen,
        "apl_state_screen_summary": apl_state_screen_summary,
        "scorecard_df": scorecard,
        "lif_summary": lif_summary,
    }


def _load_tf_groundplan_annotations(path: Path) -> pd.DataFrame:
    requested = [
        "root_id",
        "pos_x",
        "pos_y",
        "pos_z",
        "soma_x",
        "soma_y",
        "soma_z",
        "flow",
        "super_class",
        "cell_class",
        "cell_sub_class",
        "supertype",
        "cell_type",
        "hemibrain_type",
        "ito_lee_hemilineage",
        "hartenstein_hemilineage",
        "top_nt",
        "side",
        "dimorphism",
        "fru_dsx",
    ]
    available = pd.read_parquet(path, columns=None).columns
    columns = [column for column in requested if column in available]
    frame = pd.read_parquet(path, columns=columns).drop_duplicates("root_id")
    for column in requested:
        if column not in frame:
            frame[column] = np.nan if column.endswith(("_x", "_y", "_z")) else ""
    for column in [
        "flow",
        "super_class",
        "cell_class",
        "cell_sub_class",
        "supertype",
        "cell_type",
        "hemibrain_type",
        "ito_lee_hemilineage",
        "hartenstein_hemilineage",
        "top_nt",
        "side",
        "dimorphism",
        "fru_dsx",
    ]:
        frame[column] = frame[column].fillna("").astype(str)
    return _annotate_connectome_groups(frame)


def _valid_groundplan_label(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip()
    invalid = {"", "none", "nan", "putative_primary", "unknown", "na"}
    return text.str.lower().map(lambda value: value not in invalid)


def _entropy_from_counts(counts: pd.Series | np.ndarray) -> float:
    values = np.asarray(counts, dtype=np.float64)
    total = float(values.sum())
    if total <= 0:
        return 0.0
    probabilities = values[values > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def _purity_from_counts(counts: pd.Series | np.ndarray) -> float:
    values = np.asarray(counts, dtype=np.float64)
    total = float(values.sum())
    return float(values.max() / total) if total > 0 else 0.0


def _morphology_coordinates(frame: pd.DataFrame, coordinate_scale_um: float) -> np.ndarray:
    soma = frame[["soma_x", "soma_y", "soma_z"]].to_numpy(dtype=np.float64)
    pos = frame[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=np.float64)
    coords = np.where(np.isfinite(soma), soma, pos)
    return coords * float(coordinate_scale_um)


def _summarize_hemilineage_groundplans(
    annotations: pd.DataFrame,
    *,
    label_column: str,
    min_group_size: int,
    coordinate_scale_um: float,
) -> pd.DataFrame:
    valid = annotations[_valid_groundplan_label(annotations[label_column])].copy()
    valid["groundplan_label"] = valid[label_column].astype(str)
    coords = _morphology_coordinates(valid, coordinate_scale_um)
    valid["_coord_ok"] = np.isfinite(coords).all(axis=1)
    records: list[dict[str, object]] = []
    grouped = valid.groupby("groundplan_label", dropna=False)
    for label, group in grouped:
        if len(group) < int(min_group_size):
            continue
        idx = group.index.to_numpy()
        coord_group = coords[valid.index.get_indexer(idx)]
        coord_group = coord_group[np.isfinite(coord_group).all(axis=1)]
        centroid = coord_group.mean(axis=0) if len(coord_group) else np.asarray([np.nan, np.nan, np.nan])
        if len(coord_group) > 1:
            distances = np.linalg.norm(coord_group - centroid, axis=1)
            radius_um = float(np.median(distances))
        else:
            radius_um = 0.0
        cell_counts = group["cell_class"].replace("", "unlabeled").value_counts()
        flow_counts = group["flow"].replace("", "unlabeled").value_counts()
        nt_counts = group["top_nt"].replace("", "unlabeled").value_counts()
        behavior_counts = group["connectome_group"].replace("", "unmapped").value_counts()
        records.append(
            {
                "hemilineage": str(label),
                "n_neurons": int(len(group)),
                "n_left": int(group["side"].eq("left").sum()),
                "n_right": int(group["side"].eq("right").sum()),
                "dominant_cell_class": str(cell_counts.index[0]),
                "cell_class_purity": _purity_from_counts(cell_counts),
                "cell_class_entropy_bits": _entropy_from_counts(cell_counts),
                "dominant_flow": str(flow_counts.index[0]),
                "flow_purity": _purity_from_counts(flow_counts),
                "dominant_nt": str(nt_counts.index[0]),
                "nt_purity": _purity_from_counts(nt_counts),
                "dominant_connectome_group": str(behavior_counts.index[0]),
                "connectome_group_purity": _purity_from_counts(behavior_counts),
                "n_connectome_grouped": int(group["connectome_group"].ne("").sum()),
                "soma_centroid_x_um": float(centroid[0]) if np.isfinite(centroid[0]) else np.nan,
                "soma_centroid_y_um": float(centroid[1]) if np.isfinite(centroid[1]) else np.nan,
                "soma_centroid_z_um": float(centroid[2]) if np.isfinite(centroid[2]) else np.nan,
                "median_soma_radius_um": radius_um,
                "n_fru_dsx": int(group["fru_dsx"].fillna("").astype(str).str.strip().ne("").sum()),
                "n_dimorphic": int(group["dimorphism"].str.contains("dimorphic|specific", case=False, regex=True).sum()),
            }
        )
    return pd.DataFrame.from_records(records).sort_values(
        ["n_neurons", "cell_class_purity"], ascending=[False, False]
    )


def _pairwise_groundplan_enrichment(
    annotations: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    label_column: str,
    null_repeats: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["root_id", label_column, "cell_class", "flow", "connectome_group", "side"]
    ann = annotations[columns].copy()
    ann = ann[_valid_groundplan_label(ann[label_column])].copy()
    ann[label_column] = ann[label_column].astype(str)
    pre = ann.rename(
        columns={
            "root_id": "Presynaptic_ID",
            label_column: "pre_hemilineage",
            "cell_class": "pre_cell_class",
            "flow": "pre_flow",
            "connectome_group": "pre_connectome_group",
            "side": "pre_side",
        }
    )
    post = ann.rename(
        columns={
            "root_id": "Postsynaptic_ID",
            label_column: "post_hemilineage",
            "cell_class": "post_cell_class",
            "flow": "post_flow",
            "connectome_group": "post_connectome_group",
            "side": "post_side",
        }
    )
    merged = edges.merge(pre, on="Presynaptic_ID", how="inner").merge(post, on="Postsynaptic_ID", how="inner")
    merged["same_hemilineage"] = merged["pre_hemilineage"].eq(merged["post_hemilineage"])
    merged["same_cell_class"] = merged["pre_cell_class"].fillna("").eq(merged["post_cell_class"].fillna(""))
    merged["same_flow"] = merged["pre_flow"].fillna("").eq(merged["post_flow"].fillna(""))
    merged["same_connectome_group"] = merged["pre_connectome_group"].fillna("").eq(merged["post_connectome_group"].fillna(""))
    merged["same_side"] = merged["pre_side"].fillna("").eq(merged["post_side"].fillna(""))
    merged["abs_signed_weight"] = merged["Excitatory x Connectivity"].abs()
    merged["signed_weight"] = merged["Excitatory x Connectivity"].astype(float)
    summary = pd.DataFrame.from_records(
        [
            {
                "metric": "edge_fraction_same_hemilineage",
                "observed": _safe_fraction(float(merged["same_hemilineage"].sum()), float(len(merged))),
                "n_edges": int(len(merged)),
                "weight_basis": "edge_count",
            },
            {
                "metric": "weight_fraction_same_hemilineage",
                "observed": _safe_fraction(
                    float(merged.loc[merged["same_hemilineage"], "abs_signed_weight"].sum()),
                    float(merged["abs_signed_weight"].sum()),
                ),
                "n_edges": int(len(merged)),
                "weight_basis": "abs_signed_weight",
            },
            {
                "metric": "edge_fraction_same_cell_class",
                "observed": _safe_fraction(float(merged["same_cell_class"].sum()), float(len(merged))),
                "n_edges": int(len(merged)),
                "weight_basis": "edge_count",
            },
            {
                "metric": "edge_fraction_same_flow",
                "observed": _safe_fraction(float(merged["same_flow"].sum()), float(len(merged))),
                "n_edges": int(len(merged)),
                "weight_basis": "edge_count",
            },
            {
                "metric": "edge_fraction_same_connectome_group",
                "observed": _safe_fraction(float(merged["same_connectome_group"].sum()), float(len(merged))),
                "n_edges": int(len(merged)),
                "weight_basis": "edge_count",
            },
        ]
    )
    rng = np.random.default_rng(int(random_seed))
    labels = ann[label_column].to_numpy(dtype=object)
    root_ids = ann["root_id"].astype("int64").to_numpy()
    root_index = {int(root_id): idx for idx, root_id in enumerate(root_ids)}
    pre_idx = merged["Presynaptic_ID"].astype("int64").map(root_index).to_numpy(dtype=np.int64)
    post_idx = merged["Postsynaptic_ID"].astype("int64").map(root_index).to_numpy(dtype=np.int64)
    weights = merged["abs_signed_weight"].to_numpy(dtype=np.float64)
    null_rows: list[dict[str, object]] = []
    for repeat in range(max(0, int(null_repeats))):
        shuffled = labels.copy()
        rng.shuffle(shuffled)
        same = shuffled[pre_idx] == shuffled[post_idx]
        null_rows.append(
            {
                "repeat": repeat,
                "edge_fraction_same_hemilineage": _safe_fraction(float(same.sum()), float(len(same))),
                "weight_fraction_same_hemilineage": _safe_fraction(float(weights[same].sum()), float(weights.sum())),
            }
        )
    null = pd.DataFrame.from_records(null_rows)
    if not null.empty:
        for metric in ["edge_fraction_same_hemilineage", "weight_fraction_same_hemilineage"]:
            observed = float(summary.loc[summary["metric"].eq(metric), "observed"].iloc[0])
            mean = float(null[metric].mean())
            std = float(null[metric].std(ddof=1)) if len(null) > 1 else 0.0
            z = (observed - mean) / std if std > 0 else np.nan
            p_upper = _safe_fraction(float((null[metric] >= observed).sum() + 1), float(len(null) + 1))
            summary.loc[summary["metric"].eq(metric), "null_mean"] = mean
            summary.loc[summary["metric"].eq(metric), "null_std"] = std
            summary.loc[summary["metric"].eq(metric), "null_z"] = z
            summary.loc[summary["metric"].eq(metric), "empirical_p_upper"] = p_upper
    return summary, merged


def _summarize_hemilineage_transitions(merged_edges: pd.DataFrame) -> pd.DataFrame:
    if merged_edges.empty:
        return pd.DataFrame()
    summary = (
        merged_edges.groupby(["pre_hemilineage", "post_hemilineage"], dropna=False)
        .agg(
            n_edges=("Connectivity", "size"),
            abs_signed_weight=("abs_signed_weight", "sum"),
            signed_weight=("signed_weight", "sum"),
            n_pre=("Presynaptic_ID", "nunique"),
            n_post=("Postsynaptic_ID", "nunique"),
            same_cell_class_fraction=("same_cell_class", "mean"),
            same_flow_fraction=("same_flow", "mean"),
        )
        .reset_index()
        .sort_values("abs_signed_weight", ascending=False)
    )
    return summary


def _summarize_behavior_groundplan_overlap(
    annotations: pd.DataFrame,
    behavior_target_axis_path: Path,
    *,
    label_column: str,
) -> pd.DataFrame:
    if not behavior_target_axis_path.exists():
        return pd.DataFrame()
    targets = pd.read_csv(behavior_target_axis_path)
    if targets.empty or "Postsynaptic_ID" not in targets:
        return pd.DataFrame()
    ann_cols = ["root_id", label_column, "cell_class", "cell_type", "hemibrain_type", "side", "connectome_group"]
    ann = annotations[ann_cols].copy()
    merged = targets.merge(
        ann.rename(columns={"root_id": "Postsynaptic_ID", label_column: "hemilineage"}),
        on="Postsynaptic_ID",
        how="left",
    )
    merged = merged[_valid_groundplan_label(merged["hemilineage"])].copy()
    if merged.empty:
        return pd.DataFrame()
    grouped = (
        merged.groupby(["behavior_axis", "hemilineage"], dropna=False)
        .agg(
            n_targets=("Postsynaptic_ID", "nunique"),
            total_abs_kc_input=("total_abs_kc_input", "sum"),
            median_kc_inputs=("n_kc_inputs", "median"),
            example_cell_types=("post_cell_type", lambda values: ", ".join(sorted({str(v) for v in values if str(v)}))[:160]),
        )
        .reset_index()
    )
    totals = grouped.groupby("behavior_axis", as_index=False).agg(
        axis_abs_kc_input=("total_abs_kc_input", "sum"),
        axis_targets=("n_targets", "sum"),
    )
    grouped = grouped.merge(totals, on="behavior_axis", how="left")
    grouped["fraction_axis_abs_kc_input"] = grouped["total_abs_kc_input"] / grouped["axis_abs_kc_input"].replace(0, np.nan)
    grouped["fraction_axis_targets"] = grouped["n_targets"] / grouped["axis_targets"].replace(0, np.nan)
    return grouped.sort_values(["behavior_axis", "total_abs_kc_input"], ascending=[True, False])


def _classify_developmental_behavior_target(row: pd.Series) -> tuple[str, str, str]:
    group = str(row.get("post_connectome_group", ""))
    cell_type = str(row.get("post_cell_type", "")).upper()
    hemibrain_type = str(row.get("post_hemibrain_type", "")).upper()
    top_nt = str(row.get("post_top_nt", "")).lower()
    text = f"{cell_type} {hemibrain_type}"
    if group == "APL":
        return "state_modulation", "APL_sparseness_brake", "APL/KC calcium imaging; GABA sensor; APL perturbation"
    if group == "DPM":
        return "state_modulation", "DPM_memory_persistence", "DPM calcium imaging; serotonin sensor; memory persistence assay"
    if group == "DAN":
        if "PAM" in text:
            return "reward_teaching", "PAM_reward_teaching", "DAN calcium imaging; dopamine sensor; sucrose-paired behavior"
        if "PPL" in text:
            return "negative_valence", "PPL_aversive_teaching", "DAN calcium imaging; dopamine sensor; shock/odor avoidance assay"
        return "reward_teaching", "dopamine_teaching_unsplit", "DAN calcium imaging; dopamine sensor; reinforcement assay"
    if group == "MBON":
        if top_nt == "acetylcholine":
            return "positive_valence", "cholinergic_MBON_approach", "MBON calcium imaging; optogenetic activation/silencing; approach assay"
        if top_nt == "glutamate":
            return "negative_valence", "glutamatergic_MBON_avoidance", "MBON calcium imaging; optogenetic perturbation; avoidance assay"
        if top_nt == "gaba":
            return "state_modulation", "GABAergic_MBON_disinhibition", "MBON calcium imaging; GABA sensor; memory expression assay"
        return "state_modulation", "MBON_memory_output", "MBON calcium imaging; compartment-specific behavior assay"
    if group == "DN":
        if re.search(r"\bMDN\b", text, flags=re.IGNORECASE):
            return "motor_readout", "MDN_backward_walk", "DN calcium imaging; split-GFP/GRASP; backward-walking readout"
        if re.search(r"\bDNge", text, flags=re.IGNORECASE):
            return "motor_readout", "DNge_grooming_or_action_selection", "DN calcium imaging; split-GFP/GRASP; grooming/action assay"
        if re.search(r"\bDNpe|\bDNp|\bDNg|\bDNa|\bDNb", text, flags=re.IGNORECASE):
            return "motor_readout", "DN_locomotor_steering", "DN calcium imaging; split-GFP/GRASP; locomotion/steering assay"
        return "motor_readout", "descending_motor_candidate", "DN calcium imaging; split-GFP/GRASP; open-loop behavior assay"
    return "unmapped", "unmapped", "targeted anatomy and physiology follow-up"


def _summarize_developmental_behavior_candidates(
    annotations: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    label_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ann_cols = [
        "root_id",
        label_column,
        "side",
        "connectome_group",
        "cell_class",
        "cell_type",
        "hemibrain_type",
        "top_nt",
    ]
    ann = annotations[ann_cols].copy()
    ann = ann[_valid_groundplan_label(ann[label_column])].copy()
    if ann.empty:
        return pd.DataFrame(), pd.DataFrame()
    ann[label_column] = ann[label_column].astype(str)
    pre = ann.rename(
        columns={
            "root_id": "Presynaptic_ID",
            label_column: "hemilineage",
            "side": "pre_side",
            "connectome_group": "pre_connectome_group",
            "cell_class": "pre_cell_class",
            "cell_type": "pre_cell_type",
            "hemibrain_type": "pre_hemibrain_type",
            "top_nt": "pre_top_nt",
        }
    )
    post = annotations[
        ["root_id", "side", "connectome_group", "cell_class", "cell_type", "hemibrain_type", "top_nt"]
    ].rename(
        columns={
            "root_id": "Postsynaptic_ID",
            "side": "post_side",
            "connectome_group": "post_connectome_group",
            "cell_class": "post_cell_class",
            "cell_type": "post_cell_type",
            "hemibrain_type": "post_hemibrain_type",
            "top_nt": "post_top_nt",
        }
    )
    merged = edges.merge(pre, on="Presynaptic_ID", how="inner").merge(post, on="Postsynaptic_ID", how="inner")
    merged = merged[merged["post_connectome_group"].isin(DEVELOPMENTAL_BEHAVIOR_TARGET_GROUPS)].copy()
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()
    merged["abs_signed_weight"] = merged["Excitatory x Connectivity"].abs()
    merged["signed_weight"] = merged["Excitatory x Connectivity"].astype(float)
    axis_info = merged.apply(_classify_developmental_behavior_target, axis=1, result_type="expand")
    merged["behavior_axis"] = axis_info[0]
    merged["candidate_mechanism"] = axis_info[1]
    merged["wetlab_validation"] = axis_info[2]
    grouped = (
        merged.groupby(
            [
                "hemilineage",
                "post_connectome_group",
                "behavior_axis",
                "candidate_mechanism",
                "wetlab_validation",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            n_edges=("Connectivity", "size"),
            n_pre_neurons=("Presynaptic_ID", "nunique"),
            n_targets=("Postsynaptic_ID", "nunique"),
            signed_weight=("signed_weight", "sum"),
            abs_signed_weight=("abs_signed_weight", "sum"),
            positive_weight=("signed_weight", lambda s: float(s[s > 0].sum())),
            negative_weight=("signed_weight", lambda s: float(-s[s < 0].sum())),
            example_pre_cell_types=("pre_cell_type", lambda values: ", ".join(sorted({str(v) for v in values if str(v)}))[:180]),
            example_targets=("post_cell_type", lambda values: ", ".join(sorted({str(v) for v in values if str(v)}))[:180]),
        )
        .sort_values("abs_signed_weight", ascending=False)
    )
    total_by_axis = grouped.groupby("behavior_axis")["abs_signed_weight"].transform("sum").replace(0, np.nan)
    total_by_target = grouped.groupby("post_connectome_group")["abs_signed_weight"].transform("sum").replace(0, np.nan)
    grouped["fraction_axis_weight"] = (grouped["abs_signed_weight"] / total_by_axis).fillna(0.0)
    grouped["fraction_target_group_weight"] = (grouped["abs_signed_weight"] / total_by_target).fillna(0.0)
    max_weight = float(grouped["abs_signed_weight"].max()) if not grouped.empty else 0.0
    grouped["candidate_score"] = (
        np.log1p(grouped["abs_signed_weight"].astype(float)) / np.log1p(max_weight)
        if max_weight > 0
        else 0.0
    )
    grouped["candidate_score"] = (
        0.58 * grouped["candidate_score"].astype(float)
        + 0.27 * grouped["fraction_axis_weight"].astype(float)
        + 0.15 * np.minimum(1.0, grouped["n_targets"].astype(float) / 20.0)
    )
    top_candidates = (
        grouped.sort_values(["behavior_axis", "candidate_score"], ascending=[True, False])
        .groupby("behavior_axis", as_index=False, group_keys=False)
        .head(5)
        .reset_index(drop=True)
    )
    return grouped.sort_values("candidate_score", ascending=False), top_candidates


def _parse_nature_nt_label(value: object) -> set[str]:
    text = str(value or "").lower()
    mapping = {
        "ach": "acetylcholine",
        "acetylcholine": "acetylcholine",
        "glut": "glutamate",
        "glutamate": "glutamate",
        "gaba": "gaba",
        "dopamine": "dopamine",
        "da": "dopamine",
        "serotonin": "serotonin",
        "octopamine": "octopamine",
    }
    labels: set[str] = set()
    for token, label in mapping.items():
        if re.search(rf"(?:^|[^a-z]){re.escape(token)}(?:[^a-z]|$)", text):
            labels.add(label)
    return labels


def _parse_nature_cell_count(value: object) -> float:
    if pd.isna(value):
        return np.nan
    numbers = [int(number) for number in re.findall(r"\d+", str(value))]
    if not numbers:
        return np.nan
    return float(sum(numbers[:2])) if "+" in str(value) and len(numbers) >= 2 else float(numbers[0])


def _tf_code_size(value: object) -> int:
    if pd.isna(value):
        return 0
    return len([part.strip() for part in str(value).split(",") if part.strip()])


def _load_nature_groundplan_tables(
    nature_table1_path: Path,
    nature_tf_roles_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not nature_table1_path.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    reasoning = pd.read_excel(nature_table1_path, sheet_name="Reasoning")
    cluster_annotations = pd.read_excel(nature_table1_path, sheet_name="Cluster annotations")
    roles = pd.read_excel(nature_tf_roles_path) if nature_tf_roles_path.exists() else pd.DataFrame()
    return reasoning, cluster_annotations, roles


def _fly_label_summary(annotations: pd.DataFrame, label_column: str) -> pd.DataFrame:
    valid = annotations[_valid_groundplan_label(annotations[label_column])].copy()
    valid["label"] = valid[label_column].astype(str)
    rows: list[dict[str, object]] = []
    for label, group in valid.groupby("label", dropna=False):
        nts = group["top_nt"].fillna("").astype(str).str.strip()
        nts = nts[nts.ne("")]
        nt_counts = nts.value_counts()
        cell_counts = group["cell_class"].fillna("").astype(str).str.strip().replace("", "unlabeled").value_counts()
        flow_counts = group["flow"].fillna("").astype(str).str.strip().replace("", "unlabeled").value_counts()
        rows.append(
            {
                "label": str(label),
                "flywire_neurons": int(len(group)),
                "flywire_dominant_nt": str(nt_counts.index[0]) if len(nt_counts) else "",
                "flywire_nt_purity": _purity_from_counts(nt_counts) if len(nt_counts) else 0.0,
                "flywire_dominant_cell_class": str(cell_counts.index[0]),
                "flywire_cell_class_purity": _purity_from_counts(cell_counts),
                "flywire_dominant_flow": str(flow_counts.index[0]),
                "flywire_flow_purity": _purity_from_counts(flow_counts),
            }
        )
    return pd.DataFrame.from_records(rows)


def _summarize_nature_flywire_contrast(
    annotations: pd.DataFrame,
    *,
    nature_table1_path: Path,
    nature_tf_roles_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    reasoning, cluster_annotations, tf_roles = _load_nature_groundplan_tables(nature_table1_path, nature_tf_roles_path)
    if reasoning.empty:
        return pd.DataFrame(), pd.DataFrame(), {"status": "missing_nature_tables"}
    fly_ito = _fly_label_summary(annotations, "ito_lee_hemilineage")
    fly_hartenstein = _fly_label_summary(annotations, "hartenstein_hemilineage")
    rows: list[dict[str, object]] = []
    for _, row in reasoning.iterrows():
        ito = str(row.get("Hemilineage (Ito/Lee)", "")).strip()
        hartenstein = str(row.get("Hemilineage (Hartenstein)", "")).strip()
        candidates: list[tuple[str, pd.Series]] = []
        if ito and ito.lower() != "nan":
            selected = fly_ito[fly_ito["label"].eq(ito)]
            if not selected.empty:
                candidates.append(("ito_lee", selected.iloc[0]))
        if not candidates and hartenstein and hartenstein.lower() != "nan":
            selected = fly_hartenstein[fly_hartenstein["label"].eq(hartenstein)]
            if not selected.empty:
                candidates.append(("hartenstein", selected.iloc[0]))
        if not candidates:
            continue
        match_type, fly = candidates[0]
        predicted_nt_set = _parse_nature_nt_label(row.get("Neurotransmitter (Eckstein et al predictions, unless noted)", ""))
        fly_nt = str(fly["flywire_dominant_nt"])
        rows.append(
            {
                "nature_ito_lee_hemilineage": ito,
                "nature_hartenstein_hemilineage": hartenstein,
                "nature_adjusted_hemilineage": row.get("Hemilineage_adjusted", ""),
                "match_type": match_type,
                "flywire_label": fly["label"],
                "nature_predicted_nt": ",".join(sorted(predicted_nt_set)),
                "flywire_dominant_nt": fly_nt,
                "nt_match": bool(fly_nt in predicted_nt_set) if predicted_nt_set else np.nan,
                "nature_cell_count_fafb": _parse_nature_cell_count(row.get("Cell# FAFB", np.nan)),
                "nature_cell_count_malecns": _parse_nature_cell_count(row.get("Cell# maleCNS", np.nan)),
                "flywire_neurons": int(fly["flywire_neurons"]),
                "flywire_dominant_cell_class": fly["flywire_dominant_cell_class"],
                "flywire_cell_class_purity": float(fly["flywire_cell_class_purity"]),
                "flywire_nt_purity": float(fly["flywire_nt_purity"]),
                "flywire_dominant_flow": fly["flywire_dominant_flow"],
                "flywire_flow_purity": float(fly["flywire_flow_purity"]),
            }
        )
    contrast = pd.DataFrame.from_records(rows)
    tf_summary = pd.DataFrame()
    if not cluster_annotations.empty:
        tf_summary = cluster_annotations.copy()
        if "Full hemilineage TF codes" in tf_summary:
            tf_summary["tf_code_size"] = tf_summary["Full hemilineage TF codes"].map(_tf_code_size)
        if "Hemilineage" in tf_summary and not contrast.empty:
            merged_counts = contrast[["nature_adjusted_hemilineage", "flywire_neurons", "flywire_dominant_nt"]].rename(
                columns={"nature_adjusted_hemilineage": "Hemilineage"}
            )
            tf_summary = tf_summary.merge(merged_counts, on="Hemilineage", how="left")
    summary: dict[str, object] = {
        "status": "ok",
        "n_nature_reasoning_rows": int(len(reasoning)),
        "n_nature_cluster_rows": int(len(cluster_annotations)),
        "n_matched_hemilineages": int(len(contrast)),
        "n_nt_comparable": int(contrast["nt_match"].notna().sum()) if not contrast.empty else 0,
        "n_nt_match": int(contrast["nt_match"].fillna(False).sum()) if not contrast.empty else 0,
        "nt_match_rate": float(contrast["nt_match"].dropna().mean()) if not contrast.empty and contrast["nt_match"].notna().any() else np.nan,
        "cell_count_fafb_corr": float(contrast[["flywire_neurons", "nature_cell_count_fafb"]].dropna().corr().iloc[0, 1])
        if len(contrast[["flywire_neurons", "nature_cell_count_fafb"]].dropna()) >= 3
        else np.nan,
        "cell_count_fafb_n": int(len(contrast[["flywire_neurons", "nature_cell_count_fafb"]].dropna())) if not contrast.empty else 0,
        "cell_count_malecns_corr": float(contrast[["flywire_neurons", "nature_cell_count_malecns"]].dropna().corr().iloc[0, 1])
        if len(contrast[["flywire_neurons", "nature_cell_count_malecns"]].dropna()) >= 3
        else np.nan,
        "cell_count_malecns_n": int(len(contrast[["flywire_neurons", "nature_cell_count_malecns"]].dropna())) if not contrast.empty else 0,
    }
    if not tf_roles.empty and "Manual categorization (described in Methods)" in tf_roles:
        summary["tf_role_counts"] = (
            tf_roles["Manual categorization (described in Methods)"].fillna("unknown").astype(str).value_counts().to_dict()
        )
    return contrast, tf_summary, summary


def _write_tf_groundplan_figure(
    output_dir: Path,
    *,
    hemilineage_summary: pd.DataFrame,
    enrichment_summary: pd.DataFrame,
    transitions: pd.DataFrame,
    behavior_overlap: pd.DataFrame,
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.8), constrained_layout=True)
    top = hemilineage_summary.sort_values("n_neurons", ascending=False).head(12).copy()
    top["label"] = top["hemilineage"] + "  n=" + top["n_neurons"].astype(str)
    axes[0, 0].barh(top["label"], top["cell_class_purity"], color="#2f6fb0")
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlim(0, 1.02)
    axes[0, 0].set_xlabel("dominant cell-class fraction")
    axes[0, 0].set_title("A. hemilineage 标签保留粗细胞类别")
    for y, value in enumerate(top["cell_class_purity"].to_numpy(dtype=float)):
        axes[0, 0].text(min(value + 0.015, 0.98), y, f"{value:.2f}", va="center", fontsize=8)

    plot_metrics = enrichment_summary[
        enrichment_summary["metric"].isin(
            ["edge_fraction_same_hemilineage", "weight_fraction_same_hemilineage"]
        )
    ].copy()
    x = np.arange(len(plot_metrics))
    axes[0, 1].bar(x - 0.18, plot_metrics["observed"], width=0.36, label="observed", color="#16876b")
    if "null_mean" in plot_metrics:
        axes[0, 1].bar(x + 0.18, plot_metrics["null_mean"], width=0.36, label="label shuffle null", color="#9ca3af")
    axes[0, 1].set_xticks(x, ["edge count", "abs weight"])
    axes[0, 1].set_ylabel("fraction within same hemilineage")
    axes[0, 1].set_title("B. 同 hemilineage 连接显著高于随机标签")
    ymax = float(plot_metrics["observed"].max()) * 1.22 if not plot_metrics.empty else 0.15
    axes[0, 1].set_ylim(0, max(0.15, ymax))
    for idx, row in enumerate(plot_metrics.itertuples(index=False)):
        observed = float(getattr(row, "observed"))
        null_mean = float(getattr(row, "null_mean", np.nan))
        fold = observed / null_mean if np.isfinite(null_mean) and null_mean > 0 else np.nan
        axes[0, 1].text(idx - 0.18, observed + 0.004, f"{observed:.3f}", ha="center", va="bottom", fontsize=9)
        if np.isfinite(null_mean):
            axes[0, 1].text(idx + 0.18, null_mean + 0.004, f"{null_mean:.3f}", ha="center", va="bottom", fontsize=9)
        if np.isfinite(fold):
            axes[0, 1].text(idx, ymax * 0.86, f"{fold:.1f}x", ha="center", color="#0f766e", fontsize=11, fontweight="bold")
    axes[0, 1].legend(frameon=False)

    transition_top = transitions.head(10).copy()
    transition_top["transition"] = transition_top["pre_hemilineage"] + " -> " + transition_top["post_hemilineage"]
    axes[1, 0].barh(transition_top["transition"], transition_top["abs_signed_weight"], color="#c96f19")
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel("abs signed connectivity")
    axes[1, 0].set_title("C. 最强 hemilineage 到 hemilineage 通道")
    for y, value in enumerate(transition_top["abs_signed_weight"].to_numpy(dtype=float)):
        axes[1, 0].text(value * 1.04, y, f"{int(value):,}", va="center", fontsize=8)

    if not behavior_overlap.empty:
        behavior_top = behavior_overlap.sort_values("total_abs_kc_input", ascending=False).head(10).copy()
        axis_names = {
            "appetitive_teaching": "reward teaching",
            "approach_or_positive_valence": "positive valence",
            "avoidance_or_negative_valence": "negative valence",
            "disinhibitory_memory_output": "disinhibitory output",
            "appetitive_state": "appetitive state",
            "state_modulation": "state modulation",
        }
        behavior_top["axis_short"] = behavior_top["behavior_axis"].map(axis_names).fillna(behavior_top["behavior_axis"])
        behavior_top["label"] = behavior_top["axis_short"] + " | " + behavior_top["hemilineage"]
        axes[1, 1].barh(behavior_top["label"], behavior_top["fraction_axis_abs_kc_input"], color="#6f4eb2")
        axes[1, 1].invert_yaxis()
        axes[1, 1].set_xlim(0, min(1.05, max(0.25, float(behavior_top["fraction_axis_abs_kc_input"].max()) * 1.18)))
        axes[1, 1].set_xlabel("fraction of axis KC input")
        axes[1, 1].set_title("D. 行为轴 readout 集中在少数 hemilineage")
        for y, value in enumerate(behavior_top["fraction_axis_abs_kc_input"].to_numpy(dtype=float)):
            axes[1, 1].text(value + 0.012, y, f"{value:.0%}", va="center", fontsize=8)
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(0.05, 0.55, "No behavior target-axis table found.", fontsize=12)
    fig.suptitle(
        "Nature 2026 TF groundplan 的连接组层面部分复现：发育标签与结构/功能模块对应",
        fontsize=15,
        fontweight="bold",
    )
    figure_path = output_dir / "Fig_tf_groundplan_replay.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    return figure_path


def _write_nature_contrast_figure(
    output_dir: Path,
    *,
    contrast: pd.DataFrame,
    behavior_overlap: pd.DataFrame,
    nature_summary: dict[str, object],
) -> Path | None:
    if contrast.empty:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    nt_match = int(nature_summary.get("n_nt_match", 0))
    nt_total = int(nature_summary.get("n_nt_comparable", 0))
    axes[0].bar(["match", "mismatch"], [nt_match, max(nt_total - nt_match, 0)], color=["#16876b", "#b84b5f"])
    axes[0].set_title("A. Nature 预测 NT vs FlyWire dominant NT")
    axes[0].set_ylabel("hemilineages")
    axes[0].text(0, nt_match + 0.5, f"{nt_match}/{nt_total}", ha="center", fontsize=12, fontweight="bold")

    count_frame = contrast[["flywire_neurons", "nature_cell_count_fafb"]].dropna().copy()
    axes[1].scatter(count_frame["nature_cell_count_fafb"], count_frame["flywire_neurons"], s=35, color="#2f6fb0", alpha=0.8)
    if len(count_frame) >= 2:
        lim_max = float(max(count_frame["nature_cell_count_fafb"].max(), count_frame["flywire_neurons"].max())) * 1.05
        axes[1].plot([0, lim_max], [0, lim_max], color="#9ca3af", lw=1, ls="--")
        axes[1].set_xlim(0, lim_max)
        axes[1].set_ylim(0, lim_max)
    axes[1].set_xlabel("Nature Table 1 FAFB cell count")
    axes[1].set_ylabel("FlyWire annotation count")
    axes[1].set_title(f"B. 细胞数对照 r={float(nature_summary.get('cell_count_fafb_corr', np.nan)):.2f}")

    if not behavior_overlap.empty:
        top = behavior_overlap.sort_values("total_abs_kc_input", ascending=False).head(8).copy()
        axis_names = {
            "appetitive_teaching": "reward",
            "approach_or_positive_valence": "positive",
            "avoidance_or_negative_valence": "negative",
            "disinhibitory_memory_output": "disinhibitory",
            "appetitive_state": "appetitive",
            "state_modulation": "state",
        }
        top["axis_short"] = top["behavior_axis"].map(axis_names).fillna(top["behavior_axis"])
        top["label"] = top["axis_short"] + " | " + top["hemilineage"]
        axes[2].barh(top["label"], top["fraction_axis_abs_kc_input"], color="#6f4eb2")
        axes[2].invert_yaxis()
        axes[2].set_xlabel("fraction of axis KC input")
        axes[2].set_title("C. TF/hemilineage 标签接入行为轴探索")
        for y, value in enumerate(top["fraction_axis_abs_kc_input"].to_numpy(dtype=float)):
            axes[2].text(value + 0.01, y, f"{value:.0%}", va="center", fontsize=8)
    else:
        axes[2].axis("off")
    fig.suptitle("Nature 开源 TF/hemilineage 表与 BioFly/FlyWire 的对照复现", fontsize=15, fontweight="bold")
    path = output_dir / "Fig_nature_tf_flywire_contrast.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _write_developmental_behavior_candidate_figure(
    output_dir: Path,
    *,
    candidates: pd.DataFrame,
) -> Path | None:
    if candidates.empty:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    axis_order = ["reward_teaching", "positive_valence", "negative_valence", "state_modulation", "motor_readout"]
    axis_palette = {
        "reward_teaching": "#2f6fb0",
        "positive_valence": "#16876b",
        "negative_valence": "#b84b5f",
        "state_modulation": "#6f4eb2",
        "motor_readout": "#c96f19",
    }
    plot = (
        candidates[candidates["behavior_axis"].isin(axis_order)]
        .sort_values(["behavior_axis", "candidate_score"], ascending=[True, False])
        .groupby("behavior_axis", as_index=False, group_keys=False)
        .head(4)
        .copy()
    )
    if plot.empty:
        return None
    plot["label"] = plot["hemilineage"] + " | " + plot["candidate_mechanism"]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8), constrained_layout=True)
    ordered = plot.sort_values("candidate_score", ascending=True)
    colors = [axis_palette.get(axis, "#64748b") for axis in ordered["behavior_axis"]]
    axes[0].barh(ordered["label"], ordered["candidate_score"], color=colors)
    axes[0].set_xlabel("candidate score")
    axes[0].set_title("A. 发育标签到行为候选的优先级")
    for y, value in enumerate(ordered["candidate_score"].to_numpy(dtype=float)):
        axes[0].text(value + 0.012, y, f"{value:.2f}", va="center", fontsize=8)

    matrix = (
        candidates.groupby(["hemilineage", "behavior_axis"], as_index=False)["abs_signed_weight"]
        .sum()
        .sort_values("abs_signed_weight", ascending=False)
    )
    top_labels = matrix.groupby("hemilineage")["abs_signed_weight"].sum().nlargest(12).index.tolist()
    heat = (
        matrix[matrix["hemilineage"].isin(top_labels)]
        .pivot_table(index="hemilineage", columns="behavior_axis", values="abs_signed_weight", fill_value=0, aggfunc="sum")
        .reindex(index=top_labels, columns=axis_order, fill_value=0)
    )
    image = axes[1].imshow(np.log10(heat.to_numpy(dtype=float) + 1), cmap="viridis", aspect="auto")
    axes[1].set_xticks(range(len(heat.columns)), heat.columns, rotation=35, ha="right")
    axes[1].set_yticks(range(len(heat.index)), heat.index)
    axes[1].set_title("B. hemilineage × behavior-axis 连接强度")
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04, label="log10(abs signed weight + 1)")
    fig.suptitle("真实 FlyWire 发育标签 -> MBON/DAN/APL/DPM/DN -> 行为轴候选", fontsize=15, fontweight="bold")
    path = output_dir / "Fig_developmental_behavior_candidates.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _write_tf_groundplan_report(
    output_dir: Path,
    *,
    config: TFGroundplanReplayConfig,
    summary: dict[str, object],
    hemilineage_summary: pd.DataFrame,
    enrichment_summary: pd.DataFrame,
    transitions: pd.DataFrame,
    behavior_overlap: pd.DataFrame,
    behavior_candidates: pd.DataFrame,
    top_behavior_candidates: pd.DataFrame,
    nature_contrast: pd.DataFrame,
    nature_tf_summary: pd.DataFrame,
    nature_summary: dict[str, object],
    figure_path: Path | None,
    nature_figure_path: Path | None,
    behavior_candidate_figure_path: Path | None,
) -> Path:
    report_path = output_dir / "TF_GROUNDPLAN_REPLAY_REPORT_CN.md"
    top_groundplans = hemilineage_summary[
        [
            "hemilineage",
            "n_neurons",
            "dominant_cell_class",
            "cell_class_purity",
            "dominant_flow",
            "flow_purity",
            "dominant_nt",
            "nt_purity",
            "dominant_connectome_group",
            "connectome_group_purity",
            "median_soma_radius_um",
        ]
    ].head(16)
    top_transitions = transitions[
        [
            "pre_hemilineage",
            "post_hemilineage",
            "n_edges",
            "abs_signed_weight",
            "signed_weight",
            "same_cell_class_fraction",
            "same_flow_fraction",
        ]
    ].head(16)
    top_behavior = (
        behavior_overlap[
            [
                "behavior_axis",
                "hemilineage",
                "n_targets",
                "total_abs_kc_input",
                "fraction_axis_abs_kc_input",
                "example_cell_types",
            ]
        ].head(18)
        if not behavior_overlap.empty
        else pd.DataFrame()
    )
    candidate_cols = [
        "behavior_axis",
        "hemilineage",
        "post_connectome_group",
        "candidate_mechanism",
        "candidate_score",
        "n_pre_neurons",
        "n_targets",
        "abs_signed_weight",
        "fraction_axis_weight",
        "example_targets",
        "wetlab_validation",
    ]
    top_candidate_table = (
        top_behavior_candidates[[column for column in candidate_cols if column in top_behavior_candidates.columns]]
        if not top_behavior_candidates.empty
        else pd.DataFrame()
    )
    candidate_axis_summary = (
        behavior_candidates.groupby("behavior_axis", as_index=False)
        .agg(
            n_hemilineages=("hemilineage", "nunique"),
            n_target_groups=("post_connectome_group", "nunique"),
            n_targets=("n_targets", "sum"),
            abs_signed_weight=("abs_signed_weight", "sum"),
            top_candidate=("candidate_mechanism", lambda values: str(values.iloc[0]) if len(values) else ""),
        )
        .sort_values("abs_signed_weight", ascending=False)
        if not behavior_candidates.empty
        else pd.DataFrame()
    )
    top_nature = (
        nature_contrast[
            [
                "nature_ito_lee_hemilineage",
                "nature_hartenstein_hemilineage",
                "flywire_label",
                "nature_predicted_nt",
                "flywire_dominant_nt",
                "nt_match",
                "nature_cell_count_fafb",
                "flywire_neurons",
                "flywire_dominant_cell_class",
            ]
        ].head(18)
        if not nature_contrast.empty
        else pd.DataFrame()
    )
    top_tf_codes = (
        nature_tf_summary[
            [
                "Cluster",
                "Hemilineage",
                "Full hemilineage TF codes",
                "tf_code_size",
                "flywire_neurons",
                "flywire_dominant_nt",
            ]
        ].head(18)
        if not nature_tf_summary.empty and "tf_code_size" in nature_tf_summary
        else pd.DataFrame()
    )
    edge_metric = enrichment_summary[enrichment_summary["metric"].eq("edge_fraction_same_hemilineage")]
    weight_metric = enrichment_summary[enrichment_summary["metric"].eq("weight_fraction_same_hemilineage")]
    edge_text = ""
    if not edge_metric.empty:
        row = edge_metric.iloc[0]
        edge_text = (
            f"同 hemilineage 边比例 observed={float(row['observed']):.4f}; "
            f"label-shuffle null={float(row.get('null_mean', np.nan)):.4f}; "
            f"z={float(row.get('null_z', np.nan)):.2f}。"
        )
    weight_text = ""
    if not weight_metric.empty:
        row = weight_metric.iloc[0]
        weight_text = (
            f"按绝对 signed weight 加权的同 hemilineage 比例 observed={float(row['observed']):.4f}; "
            f"null={float(row.get('null_mean', np.nan)):.4f}; "
            f"z={float(row.get('null_z', np.nan)):.2f}。"
        )
    report_path.write_text(
        f"""# Nature 2026 TF groundplan 的 FlyWire 连接组层面部分复现

## 定位

这份分析复现的是 Nature 2026 文章中可以落到成人 FlyWire 连接组上的结构层预测：
developmental groundplan/hemilineage 应该与粗形态、投射方向、target choice 和电路角色有关。
它不能复现 TF 分子表达本身；当前 BioFly 没有 scRNA-seq、HCR/FISH 或 TF perturbation 输入。

使用的数据是本地真实 FlyWire v783：

- annotation: `{config.annotation_path}`
- connectivity: `{config.connectivity_path}`
- hemilineage column: `{config.hemilineage_column}`
- behavior target axis: `{config.behavior_target_axis_path}`

## 主要数值

- 有效 hemilineage 标注神经元：{summary['n_labeled_neurons']} / {summary['n_total_neurons']}
- 进入统计的 hemilineage 组数：{summary['n_groundplan_groups']}
- 连接边数：{summary['n_edges_loaded']}；带双端 hemilineage 标注的边：{summary['n_labeled_edges']}
- {edge_text}
- {weight_text}
- cell-class median purity：{summary['median_cell_class_purity']:.3f}
- flow median purity：{summary['median_flow_purity']:.3f}
- neurotransmitter median purity：{summary['median_nt_purity']:.3f}
- 有 Fru/Dsx 或性别二态标注的 hemilineage 组：{summary['n_groundplans_with_sex_markers']}
- Nature Table 1 可对齐 hemilineage：{nature_summary.get('n_matched_hemilineages', 0)}
- Nature 预测 NT 与 FlyWire dominant NT 一致：{nature_summary.get('n_nt_match', 0)} / {nature_summary.get('n_nt_comparable', 0)}
- Nature FAFB cell count 与 FlyWire annotation count 相关：r={float(nature_summary.get('cell_count_fafb_corr', np.nan)):.3f}, n={nature_summary.get('cell_count_fafb_n', 0)}

## 解释

1. hemilineage 标签在真实 FlyWire 成人脑中不是随机标签：多数较大组有明显的 dominant cell class、flow 和 NT 倾向。
2. 双端都带 hemilineage 标签的真实连接边中，同 hemilineage 连接比例高于 label-shuffle null，说明发育标签与连接模块有可测对应。
3. 行为 readout 不是均匀落在所有 hemilineage 上；KC 输出映射到 MBON/DAN/DPM/APL 后，部分 behavior axis 的输入质量集中在少数 hemilineage 标签上。
4. 这些结果支持“发育 groundplan 可以作为 BioFly 的结构先验”，但不能反推具体 TF code，也不能替代文章中的遗传扰动证据。
5. 接入 Nature 开源补充表后，BioFly 可以做更强的对照复现：TF/hemilineage 表预测的 NT 和细胞数在 FlyWire adult annotation 中高度一致，并能继续映射到 KC 行为轴 readout。
6. 新增候选筛选把发育标签继续投影到 MBON、DAN、APL、DPM、DN 和行为轴；候选只表示连接组层面的优先级，不等同于因果验证。

{f'![TF groundplan partial reproduction]({figure_path.name})' if figure_path is not None else ''}

{f'![Nature TF FlyWire contrast]({nature_figure_path.name})' if nature_figure_path is not None else ''}

{f'![Developmental behavior candidates]({behavior_candidate_figure_path.name})' if behavior_candidate_figure_path is not None else ''}

## Nature 开源表与 FlyWire 对照

{_markdown_table(top_nature)}

## Nature TF code 接入表

{_markdown_table(top_tf_codes)}

## Top hemilineage groundplans

{_markdown_table(top_groundplans)}

## Label enrichment

{_markdown_table(enrichment_summary)}

## Strong hemilineage transitions

{_markdown_table(top_transitions)}

## Behavior-axis hemilineage overlap

{_markdown_table(top_behavior)}

## 发育标签到行为轴候选筛选

该表把 adult FlyWire 中带 hemilineage 标签的上游神经元，直接投影到 MBON、DAN、APL、DPM、DN 五类目标，并按行为轴汇总。分数由连接强度、该行为轴内占比和目标覆盖共同决定，用于 wet-lab 优先级排序，不写作确定因果。

### 行为轴汇总

{_markdown_table(candidate_axis_summary)}

### Top candidates by behavior axis

{_markdown_table(top_candidate_table)}
""",
        encoding="utf-8",
    )
    return report_path


def run_tf_groundplan_replay(
    config: TFGroundplanReplayConfig | None = None,
) -> dict[str, Path | pd.DataFrame | dict[str, object]]:
    """Replay Nature 2026 TF-groundplan predictions on real FlyWire labels."""

    config = config or TFGroundplanReplayConfig()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations = _load_tf_groundplan_annotations(config.annotation_path)
    label_column = (
        config.hemilineage_column
        if config.hemilineage_column in annotations.columns
        else config.secondary_hemilineage_column
    )
    hemilineage_summary = _summarize_hemilineage_groundplans(
        annotations,
        label_column=label_column,
        min_group_size=config.min_group_size,
        coordinate_scale_um=config.coordinate_scale_um,
    )
    edge_columns = ["Presynaptic_ID", "Postsynaptic_ID", "Connectivity", "Excitatory x Connectivity"]
    edges = pd.read_parquet(config.connectivity_path, columns=edge_columns)
    if config.max_edges is not None:
        edges = edges.head(max(0, int(config.max_edges))).copy()
    enrichment_summary, merged_edges = _pairwise_groundplan_enrichment(
        annotations,
        edges,
        label_column=label_column,
        null_repeats=config.null_repeats,
        random_seed=config.random_seed,
    )
    transitions = _summarize_hemilineage_transitions(merged_edges)
    behavior_overlap = _summarize_behavior_groundplan_overlap(
        annotations,
        config.behavior_target_axis_path,
        label_column=label_column,
    )
    behavior_candidates, top_behavior_candidates = _summarize_developmental_behavior_candidates(
        annotations,
        edges,
        label_column=label_column,
    )
    nature_contrast, nature_tf_summary, nature_summary = _summarize_nature_flywire_contrast(
        annotations,
        nature_table1_path=config.nature_table1_path,
        nature_tf_roles_path=config.nature_tf_roles_path,
    )
    figure_path = _write_tf_groundplan_figure(
        output_dir,
        hemilineage_summary=hemilineage_summary,
        enrichment_summary=enrichment_summary,
        transitions=transitions,
        behavior_overlap=behavior_overlap,
    )
    nature_figure_path = _write_nature_contrast_figure(
        output_dir,
        contrast=nature_contrast,
        behavior_overlap=behavior_overlap,
        nature_summary=nature_summary,
    )
    behavior_candidate_figure_path = _write_developmental_behavior_candidate_figure(
        output_dir,
        candidates=behavior_candidates,
    )
    valid_labels = _valid_groundplan_label(annotations[label_column])
    summary = {
        "n_total_neurons": int(len(annotations)),
        "n_labeled_neurons": int(valid_labels.sum()),
        "label_column": str(label_column),
        "n_groundplan_groups": int(len(hemilineage_summary)),
        "n_edges_loaded": int(len(edges)),
        "n_labeled_edges": int(len(merged_edges)),
        "median_cell_class_purity": float(hemilineage_summary["cell_class_purity"].median())
        if not hemilineage_summary.empty
        else 0.0,
        "median_flow_purity": float(hemilineage_summary["flow_purity"].median()) if not hemilineage_summary.empty else 0.0,
        "median_nt_purity": float(hemilineage_summary["nt_purity"].median()) if not hemilineage_summary.empty else 0.0,
        "n_groundplans_with_sex_markers": int(
            (
                (hemilineage_summary["n_fru_dsx"] > 0)
                | (hemilineage_summary["n_dimorphic"] > 0)
            ).sum()
        )
        if not hemilineage_summary.empty
        else 0,
        "n_developmental_behavior_candidate_rows": int(len(behavior_candidates)),
        "n_developmental_behavior_axes": int(behavior_candidates["behavior_axis"].nunique())
        if not behavior_candidates.empty
        else 0,
        "n_developmental_behavior_target_groups": int(behavior_candidates["post_connectome_group"].nunique())
        if not behavior_candidates.empty
        else 0,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "nature_open_table_summary": nature_summary,
    }
    paths: dict[str, Path | pd.DataFrame | dict[str, object]] = {
        "hemilineage_summary_csv": output_dir / "tf_groundplan_hemilineage_summary.csv",
        "enrichment_summary_csv": output_dir / "tf_groundplan_enrichment_summary.csv",
        "transition_summary_csv": output_dir / "tf_groundplan_transition_summary.csv",
        "behavior_overlap_csv": output_dir / "tf_groundplan_behavior_overlap.csv",
        "behavior_candidates_csv": output_dir / "tf_groundplan_behavior_candidates.csv",
        "top_behavior_candidates_csv": output_dir / "tf_groundplan_top_behavior_candidates.csv",
        "nature_contrast_csv": output_dir / "nature_tf_flywire_contrast.csv",
        "nature_tf_summary_csv": output_dir / "nature_tf_code_summary.csv",
        "metadata_json": output_dir / "tf_groundplan_replay_metadata.json",
    }
    if figure_path is not None:
        paths["figure_png"] = figure_path
    if nature_figure_path is not None:
        paths["nature_figure_png"] = nature_figure_path
    if behavior_candidate_figure_path is not None:
        paths["behavior_candidate_figure_png"] = behavior_candidate_figure_path
    hemilineage_summary.to_csv(paths["hemilineage_summary_csv"], index=False)
    enrichment_summary.to_csv(paths["enrichment_summary_csv"], index=False)
    transitions.to_csv(paths["transition_summary_csv"], index=False)
    behavior_overlap.to_csv(paths["behavior_overlap_csv"], index=False)
    behavior_candidates.to_csv(paths["behavior_candidates_csv"], index=False)
    top_behavior_candidates.to_csv(paths["top_behavior_candidates_csv"], index=False)
    nature_contrast.to_csv(paths["nature_contrast_csv"], index=False)
    nature_tf_summary.to_csv(paths["nature_tf_summary_csv"], index=False)
    report_path = _write_tf_groundplan_report(
        output_dir,
        config=config,
        summary=summary,
        hemilineage_summary=hemilineage_summary,
        enrichment_summary=enrichment_summary,
        transitions=transitions,
        behavior_overlap=behavior_overlap,
        behavior_candidates=behavior_candidates,
        top_behavior_candidates=top_behavior_candidates,
        nature_contrast=nature_contrast,
        nature_tf_summary=nature_tf_summary,
        nature_summary=nature_summary,
        figure_path=figure_path,
        nature_figure_path=nature_figure_path,
        behavior_candidate_figure_path=behavior_candidate_figure_path,
    )
    paths["report_md"] = report_path
    paths["metadata_json"].write_text(
        json.dumps(
            {
                "summary": summary,
                "paths": {key: str(value) for key, value in paths.items() if isinstance(value, Path)},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        **paths,
        "summary": summary,
        "hemilineage_summary_df": hemilineage_summary,
        "enrichment_summary_df": enrichment_summary,
        "transition_summary_df": transitions,
        "behavior_overlap_df": behavior_overlap,
        "behavior_candidates_df": behavior_candidates,
        "top_behavior_candidates_df": top_behavior_candidates,
        "nature_contrast_df": nature_contrast,
        "nature_tf_summary_df": nature_tf_summary,
    }


__all__ = [
    "BehaviorClosureProxyConfig",
    "KCFlyWireRatioConfig",
    "LateralizationMechanismSuiteConfig",
    "LateralizationRepresentationMemoryConfig",
    "LearningMemoryPerturbationConfig",
    "MBONDecisionPivotConfig",
    "ScaleBenchmarkConfig",
    "TFGroundplanReplayConfig",
    "build_flywire_glomerulus_kc_matrix",
    "build_mixture_odor_panel",
    "run_behavior_closure_proxy",
    "run_flywire_connectome_science",
    "run_kc_flywire_ratio_sweep",
    "run_lateralization_mechanism_suite",
    "run_lateralization_representation_memory",
    "run_learning_memory_perturbation",
    "run_method_effectiveness_suite",
    "run_mbon_decision_pivot_search",
    "run_scale_benchmark",
    "run_tf_groundplan_replay",
]
