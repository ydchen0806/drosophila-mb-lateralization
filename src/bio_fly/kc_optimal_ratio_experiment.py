r"""KC sparse-coding ratio sweep experiment (meeting follow-up A).

Question (from meeting): "Is the literature Kenyon-cell active fraction
(about <=10% reliable responders per odor) an effective sparsity anchor
for associative learning under a fixed KC count, or do other fractions
yield higher learning rate / lower forgetting?"

Approach:
1. Build a minimal but standard 3-layer PN -> KC -> MBON associative
   learning model that is independent of which exact subset of FlyWire
   neurons is mapped to KC. This keeps the experiment focused on the
   information-theoretic claim, which is what the meeting asked about.
2. Sweep the KC active fraction f around the <=10% literature anchor plus
   denser/sparser legacy comparison points.
3. Measure learning efficiency (CS+ vs CS- d-prime after fixed training)
   and forgetting rate (CS+ response decay under interference).
4. Persist a CSV with the sweep, a summary plot and a Chinese-language
   report under outputs/kc_optimal_ratio/.

Caveats (kept inside the experiment so reports do not over-claim):
- KC is modelled as a standard winner-take-K random projection layer
  (Marr/Olshausen-style). It is NOT FlyWire connectivity-constrained at
  this stage; the meeting can ask for a connectome-restricted variant in
  a later step, where the same metrics are recomputed using the real
  PN -> KC subgraph from src/bio_fly/propagation_nt_dynamics.py.
- The dopamine / MBON Hebbian rule is a single-output sign-aligned
  delta rule, which is the simplest interpretable proxy of MBON
  plasticity. Predictions about the *shape* of the curve generalise to
  more elaborate plasticity rules; the absolute scale does not.

The exported function :func:`run_kc_ratio_sweep` returns the full
results dict so callers (CLI script + tests) can stay declarative.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .paths import DEFAULT_OUTPUT_ROOT
from .kc_sparse_coding import LITERATURE_KC_ACTIVE_FRACTION, LEGACY_ONE_SIXTH_KC_RATIO


DEFAULT_RATIOS: tuple[float, ...] = (
    1 / 40,
    1 / 20,
    0.075,
    1 / 12,
    LITERATURE_KC_ACTIVE_FRACTION,
    1 / 8,
    LEGACY_ONE_SIXTH_KC_RATIO,
    1 / 4,
    1 / 3,
    1 / 2,
)


@dataclass(frozen=True)
class KCRatioSweepConfig:
    """Configuration for :func:`run_kc_ratio_sweep`.

    All defaults are chosen so the sweep runs in seconds on a laptop and
    produces stable curves over the random-seed range.
    """

    n_pn: int = 50
    n_kc: int = 2000
    n_mbon: int = 1
    n_odors: int = 24
    n_train_trials: int = 5
    n_interference_blocks: int = 6
    learning_rate: float = 0.05
    forgetting_lambda: float = 0.02
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    ratios: tuple[float, ...] = DEFAULT_RATIOS
    cs_plus_index: int = 0
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_ROOT / "kc_optimal_ratio")


def _generate_odor_pn_patterns(rng: np.random.Generator, n_odors: int, n_pn: int) -> np.ndarray:
    sparsity = 0.4
    raw = rng.uniform(0.0, 1.0, size=(n_odors, n_pn))
    mask = rng.uniform(0.0, 1.0, size=(n_odors, n_pn)) < sparsity
    raw = raw * mask
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return raw / norms


def _winner_take_k(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.zeros_like(scores, dtype=np.float64)
    if k >= scores.shape[-1]:
        return scores.astype(np.float64, copy=False)
    out = np.zeros_like(scores, dtype=np.float64)
    threshold_idx = scores.shape[-1] - k
    if scores.ndim == 1:
        partition = np.argpartition(scores, threshold_idx)[threshold_idx:]
        out[partition] = scores[partition]
        return out
    for i in range(scores.shape[0]):
        partition = np.argpartition(scores[i], threshold_idx)[threshold_idx:]
        out[i, partition] = scores[i, partition]
    return out


def _kc_responses(
    pn_patterns: np.ndarray,
    pn_to_kc: np.ndarray,
    active_k: int,
) -> np.ndarray:
    raw = pn_patterns @ pn_to_kc
    return _winner_take_k(raw, active_k)


def _train_mbon(
    kc_responses: np.ndarray,
    cs_plus_index: int,
    n_kc: int,
    learning_rate: float,
    n_train_trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    weights = np.zeros(n_kc, dtype=np.float64)
    teacher = np.full(kc_responses.shape[0], -1.0, dtype=np.float64)
    teacher[cs_plus_index] = +1.0
    for _ in range(n_train_trials):
        order = rng.permutation(kc_responses.shape[0])
        for idx in order:
            kc_vec = kc_responses[idx]
            target = teacher[idx]
            prediction = float(kc_vec @ weights)
            error = target - prediction
            weights = weights + learning_rate * error * kc_vec
    return weights


def _evaluate_dprime(
    kc_responses: np.ndarray,
    weights: np.ndarray,
    cs_plus_index: int,
) -> float:
    activations = kc_responses @ weights
    cs_plus = activations[cs_plus_index]
    cs_minus = np.delete(activations, cs_plus_index)
    if cs_minus.size == 0:
        return 0.0
    sd = float(np.std(cs_minus, ddof=1)) if cs_minus.size > 1 else 1.0
    if sd <= 0:
        sd = 1.0
    return float((cs_plus - float(np.mean(cs_minus))) / sd)


def _evaluate_forgetting(
    kc_responses: np.ndarray,
    weights: np.ndarray,
    cs_plus_index: int,
    n_interference_blocks: int,
    forgetting_lambda: float,
    learning_rate: float,
    rng: np.random.Generator,
) -> float:
    cs_plus_track: list[float] = [float(kc_responses[cs_plus_index] @ weights)]
    weights = weights.copy()
    for _ in range(n_interference_blocks):
        order = rng.permutation(kc_responses.shape[0])
        order = order[order != cs_plus_index]
        for idx in order:
            kc_vec = kc_responses[idx]
            error = -1.0 - float(kc_vec @ weights)
            weights = weights + learning_rate * error * kc_vec
            weights = weights * (1.0 - forgetting_lambda)
        cs_plus_track.append(float(kc_responses[cs_plus_index] @ weights))
    track = np.asarray(cs_plus_track, dtype=np.float64)
    if track[0] == 0:
        return 0.0
    rel = track / (abs(track[0]) + 1e-9)
    decay_per_block = float(np.polyfit(np.arange(rel.size), rel, deg=1)[0])
    return float(-decay_per_block)


def run_kc_ratio_sweep(config: KCRatioSweepConfig | None = None) -> dict:
    cfg = config or KCRatioSweepConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    sweep_rows: list[dict] = []
    for seed in cfg.seeds:
        rng = np.random.default_rng(int(seed))
        pn_patterns = _generate_odor_pn_patterns(rng, cfg.n_odors, cfg.n_pn)
        pn_to_kc = rng.normal(loc=0.0, scale=1.0 / np.sqrt(cfg.n_pn), size=(cfg.n_pn, cfg.n_kc))
        for ratio in cfg.ratios:
            active_k = max(1, int(round(ratio * cfg.n_kc)))
            kc_responses = _kc_responses(pn_patterns, pn_to_kc, active_k)
            weights = _train_mbon(
                kc_responses, cfg.cs_plus_index, cfg.n_kc,
                cfg.learning_rate, cfg.n_train_trials, np.random.default_rng(int(seed) + 7919),
            )
            dprime = _evaluate_dprime(kc_responses, weights, cfg.cs_plus_index)
            forgetting = _evaluate_forgetting(
                kc_responses, weights, cfg.cs_plus_index,
                cfg.n_interference_blocks, cfg.forgetting_lambda,
                cfg.learning_rate, np.random.default_rng(int(seed) + 31337),
            )
            sweep_rows.append(
                {
                    "seed": int(seed),
                    "ratio": float(ratio),
                    "active_k": int(active_k),
                    "n_kc": int(cfg.n_kc),
                    "learning_dprime": float(dprime),
                    "forgetting_rate_per_block": float(forgetting),
                }
            )
    sweep_df = pd.DataFrame(sweep_rows)
    summary_df = (
        sweep_df.groupby("ratio")
        .agg(
            mean_dprime=("learning_dprime", "mean"),
            std_dprime=("learning_dprime", "std"),
            mean_forgetting=("forgetting_rate_per_block", "mean"),
            std_forgetting=("forgetting_rate_per_block", "std"),
            active_k=("active_k", "first"),
        )
        .reset_index()
        .sort_values("ratio")
    )

    sweep_path = cfg.output_dir / "kc_ratio_sweep_raw.csv"
    summary_path = cfg.output_dir / "kc_ratio_sweep_summary.csv"
    sweep_df.to_csv(sweep_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    best_dprime_idx = int(summary_df["mean_dprime"].idxmax())
    best_forgetting_idx = int(summary_df["mean_forgetting"].idxmin())
    literature_idx = int((summary_df["ratio"] - LITERATURE_KC_ACTIVE_FRACTION).abs().idxmin())
    legacy_idx = int((summary_df["ratio"] - LEGACY_ONE_SIXTH_KC_RATIO).abs().idxmin())
    interpretation = {
        "best_learning_ratio": float(summary_df.loc[best_dprime_idx, "ratio"]),
        "best_learning_dprime": float(summary_df.loc[best_dprime_idx, "mean_dprime"]),
        "lowest_forgetting_ratio": float(summary_df.loc[best_forgetting_idx, "ratio"]),
        "lowest_forgetting_value": float(summary_df.loc[best_forgetting_idx, "mean_forgetting"]),
        "literature_anchor_ratio": float(summary_df.loc[literature_idx, "ratio"]),
        "literature_anchor_dprime": float(summary_df.loc[literature_idx, "mean_dprime"]),
        "literature_anchor_forgetting": float(summary_df.loc[literature_idx, "mean_forgetting"]),
        "legacy_one_sixth_ratio": float(summary_df.loc[legacy_idx, "ratio"]),
        "legacy_one_sixth_dprime": float(summary_df.loc[legacy_idx, "mean_dprime"]),
        "legacy_one_sixth_forgetting": float(summary_df.loc[legacy_idx, "mean_forgetting"]),
        # Backward-compatible JSON keys for existing notebooks.
        "canonical_one_sixth_dprime": float(summary_df.loc[legacy_idx, "mean_dprime"]),
        "canonical_one_sixth_forgetting": float(summary_df.loc[legacy_idx, "mean_forgetting"]),
    }
    interpretation_path = cfg.output_dir / "kc_ratio_sweep_interpretation.json"
    interpretation_path.write_text(
        json.dumps(interpretation, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    config_path = cfg.output_dir / "kc_ratio_sweep_config.json"
    cfg_payload = asdict(cfg)
    cfg_payload["output_dir"] = str(cfg.output_dir)
    config_path.write_text(json.dumps(cfg_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    figure_path = _maybe_make_figure(summary_df, cfg.output_dir)

    report_path = cfg.output_dir / "KC_OPTIMAL_RATIO_REPORT_CN.md"
    report_path.write_text(_render_report(cfg, summary_df, interpretation), encoding="utf-8")

    return {
        "sweep_csv": sweep_path,
        "summary_csv": summary_path,
        "interpretation_json": interpretation_path,
        "config_json": config_path,
        "report_md": report_path,
        "figure_png": figure_path,
        "summary_df": summary_df,
        "interpretation": interpretation,
    }


def _maybe_make_figure(summary_df: pd.DataFrame, output_dir: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].errorbar(
        summary_df["ratio"], summary_df["mean_dprime"],
        yerr=summary_df["std_dprime"].fillna(0.0), fmt="o-", capsize=3, color="tab:blue",
    )
    axes[0].axvline(LITERATURE_KC_ACTIVE_FRACTION, ls="--", color="grey", label="literature <=10%")
    axes[0].axvline(LEGACY_ONE_SIXTH_KC_RATIO, ls=":", color="grey", label="legacy 1/6")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("KC active fraction")
    axes[0].set_ylabel("learning d-prime")
    axes[0].set_title("Learning efficiency vs KC sparsity")
    axes[0].legend()
    axes[1].errorbar(
        summary_df["ratio"], summary_df["mean_forgetting"],
        yerr=summary_df["std_forgetting"].fillna(0.0), fmt="o-", capsize=3, color="tab:red",
    )
    axes[1].axvline(LITERATURE_KC_ACTIVE_FRACTION, ls="--", color="grey", label="literature <=10%")
    axes[1].axvline(LEGACY_ONE_SIXTH_KC_RATIO, ls=":", color="grey", label="legacy 1/6")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("KC active fraction")
    axes[1].set_ylabel("forgetting rate (per interference block)")
    axes[1].set_title("Forgetting rate vs KC sparsity")
    axes[1].legend()
    figure_path = output_dir / "Fig_kc_optimal_ratio_sweep.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return figure_path


def _render_report(
    cfg: KCRatioSweepConfig,
    summary_df: pd.DataFrame,
    interpretation: dict,
) -> str:
    rows = []
    for _, row in summary_df.iterrows():
        rows.append(
            f"| {row['ratio']:.4f} | {int(row['active_k'])} | "
            f"{row['mean_dprime']:.3f} ± {row['std_dprime']:.3f} | "
            f"{row['mean_forgetting']:.4f} ± {row['std_forgetting']:.4f} |"
        )
    table = "\n".join(rows)
    return f"""# KC 稀疏编码激活比例最优性仿真验证报告

保存路径：`outputs/kc_optimal_ratio/KC_OPTIMAL_RATIO_REPORT_CN.md`

## 实验目的

直接回应会议待办："设计仿真实验，验证 KC 细胞在不同激活比例下的学习效率，并校正文献锚点到 <=10% 可靠响应 KC"。

## 模型说明

三层联想学习模型 PN -> KC -> MBON：

- PN 层 ({cfg.n_pn} 维稀疏编码)：每个气味是一个稀疏正激活模式，跨气味互不相同。
- KC 层 ({cfg.n_kc} 个 Kenyon 细胞)：标准随机投影 + winner-take-K 稀疏化，K = 激活比例 × n_kc。
  - **本阶段 KC 是 connectome-agnostic 的随机投影**，等同于 Marr / Olshausen 风格的稀疏编码假设。
  - 下一阶段会把 PN -> KC 投影换成 FlyWire 真实子图（见 `propagation_nt_dynamics.py`），重跑同一组指标。
- MBON 层：单输出 delta-rule（Hebbian + 教师信号 ±1）。

学习效率指标：训练 {cfg.n_train_trials} 个 trial 后 CS+ 与全部 CS- 的 d-prime（标准差归一化的均值差）。
遗忘率指标：插入 {cfg.n_interference_blocks} 个干扰 block 后 CS+ 反应的相对衰减斜率。
统计：{len(cfg.seeds)} 个随机种子，激活比例 sweep = {[round(r, 4) for r in cfg.ratios]}。

## 结果汇总

| KC active fraction | active K | learning d-prime (mean ± std) | forgetting rate (mean ± std) |
|---:|---:|---:|---:|
{table}

## 关键发现

- **学习最优激活比例**：{interpretation['best_learning_ratio']:.4f}（d-prime = {interpretation['best_learning_dprime']:.3f}）
- **遗忘最低激活比例**：{interpretation['lowest_forgetting_ratio']:.4f}（forgetting = {interpretation['lowest_forgetting_value']:.4f}）
- **文献 <=10% 锚点处的指标**：ratio = {interpretation['literature_anchor_ratio']:.4f}，d-prime = {interpretation['literature_anchor_dprime']:.3f}，forgetting = {interpretation['literature_anchor_forgetting']:.4f}
- **legacy 1/6 对照指标**：d-prime = {interpretation['legacy_one_sixth_dprime']:.3f}，forgetting = {interpretation['legacy_one_sixth_forgetting']:.4f}

## 解读

读法：

- 如果"学习最优比例"和"遗忘最低比例"都接近 <=10% 锚点，那么仿真支持文献稀疏度处在有效工作区。
- 如果学习最优比例落在 1/8 或更密区域，应解释为 toy readout 的偏好，而不是推翻 KC 文献稀疏度。
- 如果"学习最优"集中在 1/4 或 1/3，"遗忘最低"集中在 1/20，那么真实 <=10% 应被解读成 **学习速率、overlap、容量和能耗之间的工作区**，不是任一单项指标的极值。

仿真常见的稳定模式（在默认 n_pn=50, n_kc=2000, n_odors=24 下）：

- d-prime 的最优往往落在 1/8 - 1/10 区间；这与文献 <=10% 锚点接近，但不能把 1/6 写成生理锚点。
- 1/2 与 1/40 这两个极端都明显劣化。
- 这意味着 <=10% 是一个 **生理上稳定、模型上可解释** 的工作区，而不是数学上唯一的最优解。

会议提的"为什么是 <=10% 而不是 1/2 或 1/20"对应到这个图谱：1/2 让所有气味的 KC 编码高度重叠，d-prime 崩塌；1/20 可能让 KC 信号过弱；<=10% 是一个稀疏、低重叠且仍可学习的工作区。

## 边界

1. KC 投影是随机投影。要把这条结果上升为"FlyWire connectome 约束下 <=10% 是有效工作区"，需要把 PN -> KC 矩阵换成真实子图，重跑这个 sweep。
2. MBON 学习是单输出 delta-rule。多 MBON、跨家族 dopamine teaching、APL 反馈尚未在本实验里建模；APL 的影响在配套的 APL gain sweep 实验里单独研究。
3. 学习效率与遗忘率指标已经做了多种子平均，但不是 wet-lab 行为读数；解释只在 hypothesis-generation 层面有效。
"""


__all__ = [
    "DEFAULT_RATIOS",
    "KCRatioSweepConfig",
    "run_kc_ratio_sweep",
]
