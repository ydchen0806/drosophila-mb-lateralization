from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from bio_fly.paths import DEFAULT_CONNECTIVITY_PATH, DEFAULT_OUTPUT_ROOT, PROCESSED_DATA_ROOT

from .apl_recipe import KCAplRecipeInputs, make_kc_apl_lif_recipe
from .data_contracts import AplFeedbackParameters, ArborRunMetadata, KCOdorInputPanel, LifCellParameters
from .runtime import arbor_version, import_arbor, resolve_thread_count
from .slide9 import ArborKCSparsityConfig, build_kc_odor_input_panel


@dataclass(frozen=True)
class ArborSlide10Config:
    annotation_path: Path = field(default_factory=lambda: PROCESSED_DATA_ROOT / "flywire_neuron_annotations.parquet")
    connectivity_path: Path = field(default_factory=lambda: DEFAULT_CONNECTIVITY_PATH)
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_ROOT / "arbor_slide10_apl_inhibition")
    threads: str = "auto"
    seed: int = 0
    n_odors: int = 8
    min_glomeruli_per_odor: int = 2
    max_glomeruli_per_odor: int = 6
    channel_noise_sigma: float = 0.15
    duration_ms: float = 200.0
    dt_ms: float = 0.1
    kc_params: LifCellParameters = field(default_factory=LifCellParameters)
    apl_params: AplFeedbackParameters = field(default_factory=AplFeedbackParameters)
    apl_gains: tuple[float, ...] = (1.0, 0.25, 0.0)


def _spike_frame(spikes, panel: KCOdorInputPanel, odor_name: str, apl_gid: int, apl_gain: float) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (gid, index), spike_time in spikes:
        gid_int = int(gid)
        is_apl = gid_int == int(apl_gid)
        root_id = "APL" if is_apl else int(panel.kc_root_ids[gid_int])
        records.append(
            {
                "odor_identity": odor_name,
                "apl_gain": float(apl_gain),
                "gid": gid_int,
                "root_id": root_id,
                "cell_role": "APL" if is_apl else "KC",
                "source_index": int(index),
                "time_ms": float(spike_time),
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=["odor_identity", "apl_gain", "gid", "root_id", "cell_role", "source_index", "time_ms"],
    )


def _run_one_condition(arbor_module, panel: KCOdorInputPanel, odor_index: int, config: ArborSlide10Config, threads: int, apl_gain: float) -> pd.DataFrame:
    A = arbor_module
    recipe = make_kc_apl_lif_recipe(
        A,
        KCAplRecipeInputs(
            kc_root_ids=panel.kc_root_ids,
            kc_drive=panel.input_matrix[int(odor_index)],
            duration_ms=float(config.duration_ms),
            kc_params=config.kc_params,
            apl=AplFeedbackParameters(
                enabled=bool(config.apl_params.enabled),
                apl_gain=float(apl_gain),
                kc_to_apl_weight=float(config.apl_params.kc_to_apl_weight),
                apl_to_kc_weight=float(config.apl_params.apl_to_kc_weight),
                connection_delay_ms=float(config.apl_params.connection_delay_ms),
                apl_cell=config.apl_params.apl_cell,
            ),
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
    return _spike_frame(simulation.spikes(), panel, panel.odor_names[int(odor_index)], int(len(panel.kc_root_ids)), float(apl_gain))


def _summarize(spikes: pd.DataFrame, panel: KCOdorInputPanel) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_odor_rows: list[dict[str, object]] = []
    n_kc = int(len(panel.kc_root_ids))
    for apl_gain, gain_frame in spikes.groupby("apl_gain", dropna=False):
        for odor_name in panel.odor_names:
            frame = gain_frame[gain_frame["odor_identity"].eq(odor_name)]
            kc_spikes = frame[frame["cell_role"].eq("KC")]
            apl_spikes = frame[frame["cell_role"].eq("APL")]
            active_kc = int(kc_spikes["root_id"].nunique()) if not kc_spikes.empty else 0
            per_odor_rows.append(
                {
                    "backend": "arbor_lif_apl",
                    "odor_identity": odor_name,
                    "apl_gain": float(apl_gain),
                    "n_kc_total": n_kc,
                    "active_kc": active_kc,
                    "active_fraction": active_kc / float(n_kc) if n_kc else 0.0,
                    "kc_spike_count": int(len(kc_spikes)),
                    "apl_spike_count": int(len(apl_spikes)),
                }
            )
    per_odor = pd.DataFrame.from_records(per_odor_rows)
    by_gain = (
        per_odor.groupby("apl_gain", as_index=False)
        .agg(
            n_odors=("odor_identity", "nunique"),
            n_kc_total=("n_kc_total", "first"),
            mean_active_kc=("active_kc", "mean"),
            mean_active_fraction=("active_fraction", "mean"),
            min_active_fraction=("active_fraction", "min"),
            max_active_fraction=("active_fraction", "max"),
            mean_kc_spike_count=("kc_spike_count", "mean"),
            mean_apl_spike_count=("apl_spike_count", "mean"),
        )
        .sort_values("apl_gain", ascending=False)
    )
    return per_odor, by_gain


def run_arbor_slide10_apl_inhibition(config: ArborSlide10Config) -> dict[str, Path]:
    arbor_module = import_arbor()
    threads = resolve_thread_count(config.threads)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    panel_config = ArborKCSparsityConfig(
        annotation_path=config.annotation_path,
        connectivity_path=config.connectivity_path,
        output_dir=config.output_dir / "_panel",
        threads=config.threads,
        seed=config.seed,
        n_odors=config.n_odors,
        min_glomeruli_per_odor=config.min_glomeruli_per_odor,
        max_glomeruli_per_odor=config.max_glomeruli_per_odor,
        channel_noise_sigma=config.channel_noise_sigma,
        duration_ms=config.duration_ms,
        dt_ms=config.dt_ms,
        lif=config.kc_params,
    )
    panel, odor_panel, channel_table = build_kc_odor_input_panel(panel_config)
    spike_frames: list[pd.DataFrame] = []
    for apl_gain in config.apl_gains:
        for odor_index in range(len(panel.odor_names)):
            spike_frames.append(_run_one_condition(arbor_module, panel, odor_index, config, threads, float(apl_gain)))
    spikes = pd.concat(spike_frames, ignore_index=True) if spike_frames else pd.DataFrame()
    per_odor, by_gain = _summarize(spikes, panel)
    elapsed_s = time.perf_counter() - start
    metadata = ArborRunMetadata(
        arbor_version=arbor_version(arbor_module),
        requested_threads=str(config.threads),
        resolved_threads=int(threads),
        n_cells=int(len(panel.kc_root_ids) + 1),
        dt_ms=float(config.dt_ms),
        duration_ms=float(config.duration_ms),
        model="arbor_point_lif_slide10_kc_apl",
    )

    paths = {
        "spikes": config.output_dir / "slide10_kc_apl_spikes.csv",
        "per_odor": config.output_dir / "slide10_apl_gain_per_odor.csv",
        "summary": config.output_dir / "slide10_apl_gain_summary.csv",
        "odor_panel": config.output_dir / "odor_panel.csv",
        "channel_table": config.output_dir / "flywire_glomerulus_kc_channels.csv",
        "config": config.output_dir / "slide10_config.json",
        "metadata": config.output_dir / "slide10_metadata.json",
        "report": config.output_dir / "SLIDE10_APL_INHIBITION_CN.md",
    }
    spikes.to_csv(paths["spikes"], index=False)
    per_odor.to_csv(paths["per_odor"], index=False)
    by_gain.to_csv(paths["summary"], index=False)
    odor_panel.to_csv(paths["odor_panel"], index=False)
    channel_table.to_csv(paths["channel_table"], index=False)
    paths["config"].write_text(json.dumps(_config_to_jsonable(config), ensure_ascii=False, indent=2), encoding="utf-8")
    paths["metadata"].write_text(
        json.dumps({**asdict(metadata), "elapsed_s": elapsed_s}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(paths["report"], by_gain, config, metadata)
    return paths


def _write_report(path: Path, summary: pd.DataFrame, config: ArborSlide10Config, metadata: ArborRunMetadata) -> None:
    path.write_text(
        "\n".join(
            [
                "# Arbor Slide10 APL 全局抑制测试",
                "",
                "本测试在同一个 Arbor recipe 中运行 `KC -> APL -> KC` 闭环。",
                "KC 使用 Slide9 固定锚点；APL 是一个 point LIF 全局抑制节点。",
                "",
                "## 参数",
                "",
                f"- Arbor version: `{metadata.arbor_version}`",
                f"- threads: `{metadata.resolved_threads}`",
                f"- n cells: `{metadata.n_cells}`",
                f"- duration_ms: `{metadata.duration_ms}`",
                f"- dt_ms: `{metadata.dt_ms}`",
                f"- kc synaptic_gain: `{config.kc_params.synaptic_gain}`",
                f"- kc v_threshold_mv: `{config.kc_params.v_threshold_mv}`",
                f"- apl enabled: `{config.apl_params.enabled}`",
                f"- apl v_threshold_mv: `{config.apl_params.apl_cell.v_threshold_mv}`",
                f"- kc_to_apl_weight: `{config.apl_params.kc_to_apl_weight}`",
                f"- apl_to_kc_weight: `{config.apl_params.apl_to_kc_weight}`",
                "",
                "## 结果",
                "",
                summary.to_string(index=False),
                "",
            ]
        ),
        encoding="utf-8",
    )


def _config_to_jsonable(config: ArborSlide10Config) -> dict[str, object]:
    values = asdict(config)
    for key in ["annotation_path", "connectivity_path", "output_dir"]:
        values[key] = str(values[key])
    return values
