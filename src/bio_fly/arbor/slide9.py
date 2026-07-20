from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from bio_fly.kc_flywire_ratio_experiment import build_flywire_glomerulus_kc_matrix, build_mixture_odor_panel
from bio_fly.paths import DEFAULT_CONNECTIVITY_PATH, DEFAULT_OUTPUT_ROOT, PROCESSED_DATA_ROOT

from .data_contracts import ArborRunMetadata, KCOdorInputPanel, LifCellParameters
from .lif_recipe import KCLifRecipeInputs, make_kc_lif_recipe
from .runtime import arbor_version, import_arbor, resolve_thread_count


@dataclass(frozen=True)
class ArborKCSparsityConfig:
    annotation_path: Path = field(default_factory=lambda: PROCESSED_DATA_ROOT / "flywire_neuron_annotations.parquet")
    connectivity_path: Path = field(default_factory=lambda: DEFAULT_CONNECTIVITY_PATH)
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_ROOT / "arbor_slide9_kc_sparsity")
    threads: str = "auto"
    seed: int = 0
    n_odors: int = 8
    min_glomeruli_per_odor: int = 2
    max_glomeruli_per_odor: int = 6
    channel_noise_sigma: float = 0.15
    target_ratio: float = 0.10
    pass_min: float = 0.05
    pass_max: float = 0.10
    duration_ms: float = 200.0
    dt_ms: float = 0.1
    lif: LifCellParameters = field(default_factory=LifCellParameters)
    active_readout_mode: str = "spike"


def build_kc_odor_input_panel(config: ArborKCSparsityConfig) -> tuple[KCOdorInputPanel, pd.DataFrame, pd.DataFrame]:
    glomerulus_names, glomerulus_matrix, kc_root_ids, channel_table = build_flywire_glomerulus_kc_matrix(
        config.annotation_path,
        config.connectivity_path,
    )
    odor_names, activity, odor_panel = build_mixture_odor_panel(
        glomerulus_names,
        glomerulus_matrix,
        seed=int(config.seed),
        n_odors=int(config.n_odors),
        min_glomeruli_per_odor=int(config.min_glomeruli_per_odor),
        max_glomeruli_per_odor=int(config.max_glomeruli_per_odor),
        channel_noise_sigma=float(config.channel_noise_sigma),
    )
    return KCOdorInputPanel(odor_names=odor_names, input_matrix=activity, kc_root_ids=kc_root_ids), odor_panel, channel_table


def _spike_frame(spikes, kc_root_ids: np.ndarray, odor_name: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (gid, index), spike_time in spikes:
        gid_int = int(gid)
        if 0 <= gid_int < len(kc_root_ids):
            records.append(
                {
                    "odor_identity": odor_name,
                    "gid": gid_int,
                    "root_id": int(kc_root_ids[gid_int]),
                    "source_index": int(index),
                    "time_ms": float(spike_time),
                }
            )
    return pd.DataFrame.from_records(records, columns=["odor_identity", "gid", "root_id", "source_index", "time_ms"])


def _run_one_odor(arbor_module, panel: KCOdorInputPanel, odor_index: int, config: ArborKCSparsityConfig, threads: int):
    A = arbor_module
    recipe = make_kc_lif_recipe(
        A,
        KCLifRecipeInputs(
            kc_root_ids=panel.kc_root_ids,
            kc_drive=panel.input_matrix[int(odor_index)],
            duration_ms=float(config.duration_ms),
            params=config.lif,
        ),
    )
    allocation = A.proc_allocation(threads=int(threads), gpu_id=None)
    context = A.context(allocation)
    lif_hint = A.partition_hint()
    lif_hint.prefer_gpu = False
    lif_hint.cpu_group_size = 1
    decomposition = A.partition_load_balance(recipe, context, {A.cell_kind.lif: lif_hint})
    simulation = A.simulation(recipe, context=context, domains=decomposition)
    simulation.record(A.spike_recording.all)
    simulation.run(float(config.duration_ms) * A.units.ms, float(config.dt_ms) * A.units.ms)
    spikes = _spike_frame(simulation.spikes(), panel.kc_root_ids, panel.odor_names[int(odor_index)])
    return spikes


def _summarize_per_odor(spikes: pd.DataFrame, panel: KCOdorInputPanel, config: ArborKCSparsityConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    n_kc = int(len(panel.kc_root_ids))
    for odor_index, odor_name in enumerate(panel.odor_names):
        odor_spikes = spikes[spikes["odor_identity"].eq(odor_name)] if not spikes.empty else pd.DataFrame()
        active_kc = int(odor_spikes["root_id"].nunique()) if not odor_spikes.empty else 0
        rows.append(
            {
                "backend": "arbor_lif",
                "odor_identity": odor_name,
                "n_kc_total": n_kc,
                "active_kc": active_kc,
                "active_fraction": active_kc / float(n_kc) if n_kc else 0.0,
                "spike_count": int(len(odor_spikes)),
                "input_nonzero_kc": int(np.count_nonzero(panel.input_matrix[int(odor_index)] > 0)),
                "input_max": float(np.max(panel.input_matrix[int(odor_index)])),
                "active_readout_mode": config.active_readout_mode,
            }
        )
    return pd.DataFrame.from_records(rows)


def _write_report(output_dir: Path, summary: pd.DataFrame, metadata: ArborRunMetadata, config: ArborKCSparsityConfig) -> Path:
    report_path = output_dir / "ARBOR_SLIDE9_KC_SPARSITY_REPORT_CN.md"
    mean_active = float(summary["active_fraction"].mean()) if len(summary) else 0.0
    passed = bool(float(config.pass_min) <= mean_active <= float(config.pass_max))
    report_path.write_text(
        "\n".join(
            [
                "# Arbor Slide9 KC 稀疏编码测试",
                "",
                "本报告使用 Arbor 后端运行 KC LIF spike readout。当前竖切是 point LIF cell，",
                "其并行调度、recipe、event delivery 和 spike recording 由 Arbor 负责；",
                "多房室 cable morphology 会在后续 Phase 3 接入同一 recipe/data contract。",
                "",
                "## 参数",
                "",
                f"- Arbor version: `{metadata.arbor_version}`",
                f"- requested threads: `{metadata.requested_threads}`",
                f"- resolved threads: `{metadata.resolved_threads}`",
                f"- n KC cells: `{metadata.n_cells}`",
                f"- duration_ms: `{metadata.duration_ms}`",
                f"- dt_ms: `{metadata.dt_ms}`",
                f"- synaptic_gain: `{config.lif.synaptic_gain}`",
                f"- v_threshold_mv: `{config.lif.v_threshold_mv}`",
                f"- active readout: `{config.active_readout_mode}`",
                "",
                "## 结果",
                "",
                f"- mean active fraction: `{mean_active:.6f}`",
                f"- pass window: `{config.pass_min:.3f}-{config.pass_max:.3f}`",
                f"- sparsity pass: `{passed}`",
                "",
                summary.to_string(index=False),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return report_path


def run_arbor_kc_sparsity(config: ArborKCSparsityConfig) -> dict[str, Path]:
    arbor_module = import_arbor()
    threads = resolve_thread_count(config.threads)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    panel, odor_panel, channel_table = build_kc_odor_input_panel(config)
    spike_frames = [
        _run_one_odor(arbor_module, panel, odor_index, config, threads)
        for odor_index in range(len(panel.odor_names))
    ]
    spikes = pd.concat(spike_frames, ignore_index=True) if spike_frames else pd.DataFrame()
    summary = _summarize_per_odor(spikes, panel, config)
    elapsed_s = time.perf_counter() - start
    metadata = ArborRunMetadata(
        arbor_version=arbor_version(arbor_module),
        requested_threads=str(config.threads),
        resolved_threads=int(threads),
        n_cells=int(len(panel.kc_root_ids)),
        dt_ms=float(config.dt_ms),
        duration_ms=float(config.duration_ms),
        model="arbor_point_lif_slide9",
    )

    channel_path = output_dir / "flywire_glomerulus_kc_channels.csv"
    odor_panel_path = output_dir / "odor_panel.csv"
    spike_path = output_dir / "arbor_kc_spikes.csv"
    summary_path = output_dir / "arbor_kc_sparsity_summary.csv"
    config_path = output_dir / "arbor_kc_sparsity_config.json"
    metadata_path = output_dir / "arbor_kc_sparsity_metadata.json"

    channel_table.to_csv(channel_path, index=False)
    odor_panel.to_csv(odor_panel_path, index=False)
    spikes.to_csv(spike_path, index=False)
    summary.to_csv(summary_path, index=False)
    config_path.write_text(json.dumps(_config_to_jsonable(config), ensure_ascii=False, indent=2), encoding="utf-8")
    metadata_path.write_text(
        json.dumps({**asdict(metadata), "elapsed_s": elapsed_s}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path = _write_report(output_dir, summary, metadata, config)
    return {
        "channel_table": channel_path,
        "odor_panel": odor_panel_path,
        "spikes": spike_path,
        "summary": summary_path,
        "config": config_path,
        "metadata": metadata_path,
        "report": report_path,
    }


def _config_to_jsonable(config: ArborKCSparsityConfig) -> dict[str, object]:
    values = asdict(config)
    for key in ["annotation_path", "connectivity_path", "output_dir"]:
        values[key] = str(values[key])
    return values
