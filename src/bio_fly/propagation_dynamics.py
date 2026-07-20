r"""Rate-style brain-signal dynamics on the FlyWire connectome.

This module upgrades the default :func:`bio_fly.propagation.signed_multihop_response`
from a pure linear multi-hop sweep to an Euler-integrated rate model with
membrane-style time constants and optional axonal delays:

.. math::

    \tau \frac{d v}{d t} = -v + \phi\!\left(W v + I_{\\text{ext}}(t - \\Delta)\right)

* ``v(t)`` is the per-neuron rate vector at time ``t``.
* ``W`` is the FlyWire signed connectivity matrix loaded by
  :func:`bio_fly.propagation.load_connectivity_edges`.
* ``\\tau`` is a per-neuron time constant (default 50 ms).
* ``\\phi`` is a saturation non-linearity (default :func:`numpy.tanh`).
* ``\\Delta`` is a per-edge axonal delay; when omitted, the model degenerates
  to an instantaneous-coupling rate dynamics.
* ``I_{\\text{ext}}(t)`` is the external sensory drive.

In the long-time, no-delay, linearised limit (``\\phi = identity`` and
sufficiently many integration steps), the steady-state response approaches
the standard multi-hop signed propagation. A unit test
(``tests/test_propagation_dynamics.py::test_steady_state_matches_multihop_sign``)
guards that property on a toy graph.

The output schema matches :func:`signed_multihop_response`: a long-format
DataFrame with columns ``root_id``, ``score``, ``step`` so the rest of the
pipeline (readouts, dynamics traces, behaviour adapters) is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd


PRE_COL = "Presynaptic_ID"
POST_COL = "Postsynaptic_ID"
WEIGHT_COL = "Excitatory x Connectivity"


@dataclass(frozen=True)
class RateDynamicsConfig:
    """Configuration for :func:`run_rate_dynamics`.

    Parameters
    ----------
    duration_ms:
        Total simulated time in milliseconds.
    dt_ms:
        Euler integration step in milliseconds. Must satisfy ``dt_ms <= tau_ms / 4``
        for stability with the default tanh non-linearity.
    tau_ms:
        Neuron membrane-style time constant in milliseconds. Either a scalar
        applied to every neuron or a mapping ``{root_id: tau_ms}``.
    delay_ms:
        Per-edge axonal delay in milliseconds. Either a scalar applied to every
        edge or a mapping ``{(pre, post): delay_ms}``. Defaults to 0 ms
        (instantaneous coupling).
    nonlinearity:
        Either ``"tanh"`` (default), ``"linear"`` or a callable. The rate
        update applies it to the synaptic input ``W @ v + I_ext``.
    record_every_ms:
        Trace sampling interval in milliseconds. The returned DataFrame has
        one ``step`` row per recorded sample; recording every step is the
        default behaviour when set to ``None``.
    max_active:
        Top-K abs-score truncation applied at each recording sample to keep
        the trace small. ``0`` disables truncation.
    normalize_each_record:
        If True, the recorded snapshot is L1-normalised across active neurons
        so the score column matches the convention used by
        :func:`signed_multihop_response`.
    """

    duration_ms: float = 200.0
    dt_ms: float = 1.0
    tau_ms: float | Mapping[int, float] = 50.0
    delay_ms: float | Mapping[tuple[int, int], float] = 0.0
    nonlinearity: str | Callable[[np.ndarray], np.ndarray] = "tanh"
    record_every_ms: float | None = 5.0
    max_active: int = 5_000
    normalize_each_record: bool = True
    drive_profile: str = "step"


@dataclass(frozen=True)
class LIFDynamicsConfig:
    """Configuration for :func:`run_lif_dynamics`.

    This is a lightweight leaky integrate-and-fire surrogate that keeps the
    BioFly trace schema compatible with signed/rate propagation.  It is meant
    for spike-style hypothesis checks, not calibrated electrophysiology.
    """

    duration_ms: float = 80.0
    dt_ms: float = 1.0
    tau_membrane_ms: float | Mapping[int, float] = 20.0
    tau_synapse_ms: float = 8.0
    v_rest_mv: float = 0.0
    v_reset_mv: float = 0.0
    v_threshold_mv: float = 1.0
    refractory_ms: float = 3.0
    input_current: float = 1.25
    seed_current_normalization: str = "max"
    synaptic_gain: float = 0.8
    weight_normalization: str = "percentile"
    weight_percentile: float = 99.0
    weight_clip: float = 5.0
    record_every_ms: float | None = 5.0
    max_active: int = 5_000
    normalize_each_record: bool = True
    drive_profile: str = "step"


def _resolve_nonlinearity(spec) -> Callable[[np.ndarray], np.ndarray]:
    if callable(spec):
        return spec
    if spec == "linear":
        return lambda x: x
    if spec == "tanh":
        return np.tanh
    if spec == "relu":
        return lambda x: np.maximum(x, 0.0)
    raise ValueError(f"unknown nonlinearity spec: {spec!r}")


def _drive_envelope(profile: str, t_ms: float, duration_ms: float) -> float:
    if duration_ms <= 0:
        return 0.0
    if profile == "step":
        return 1.0
    if profile == "ramp":
        return float(t_ms / duration_ms)
    if profile == "phasic":
        if duration_ms <= 0:
            return 0.0
        peak = 0.1 * duration_ms
        return float(np.exp(-((t_ms - peak) ** 2) / (2 * (peak / 2 + 1) ** 2)))
    if profile == "pulse":
        return 1.0 if t_ms < 0.5 * duration_ms else 0.0
    if profile == "sustained":
        return 1.0
    raise ValueError(f"unknown drive_profile: {profile!r}")


def _build_index(edges: pd.DataFrame, seed_weights: Mapping[int, float]) -> tuple[np.ndarray, dict[int, int]]:
    seen = pd.unique(
        np.concatenate(
            [
                edges[PRE_COL].to_numpy(np.int64, copy=False),
                edges[POST_COL].to_numpy(np.int64, copy=False),
                np.asarray(list(seed_weights.keys()), dtype=np.int64),
            ]
        )
    )
    seen.sort()
    index_map = {int(root_id): int(i) for i, root_id in enumerate(seen)}
    return seen.astype(np.int64), index_map


def _build_dense_matrix(
    edges: pd.DataFrame,
    index_map: dict[int, int],
) -> np.ndarray:
    n = len(index_map)
    W = np.zeros((n, n), dtype=np.float64)
    pre_idx = edges[PRE_COL].map(index_map).to_numpy(np.int64, copy=False)
    post_idx = edges[POST_COL].map(index_map).to_numpy(np.int64, copy=False)
    weights = edges[WEIGHT_COL].to_numpy(np.float64, copy=False)
    np.add.at(W, (post_idx, pre_idx), weights)
    return W


def _delay_steps_array(
    edges: pd.DataFrame,
    delay_ms: float | Mapping[tuple[int, int], float],
    dt_ms: float,
) -> np.ndarray | None:
    if isinstance(delay_ms, Mapping):
        if not delay_ms:
            return None
        keys = np.array(list(delay_ms.keys()), dtype=np.int64)
        values = np.array(list(delay_ms.values()), dtype=np.float64)
        out = np.zeros(len(edges), dtype=np.int64)
        edge_keys = list(zip(edges[PRE_COL].astype(np.int64), edges[POST_COL].astype(np.int64)))
        lookup = {(int(p), int(q)): float(d) for (p, q), d in zip(map(tuple, keys), values)}
        for i, key in enumerate(edge_keys):
            out[i] = int(round(lookup.get(key, 0.0) / max(dt_ms, 1e-9)))
        return out
    delay_steps = int(round(float(delay_ms) / max(dt_ms, 1e-9)))
    if delay_steps <= 0:
        return None
    return np.full(len(edges), delay_steps, dtype=np.int64)


def _tau_array(
    tau_ms: float | Mapping[int, float],
    root_ids_by_index: np.ndarray,
) -> np.ndarray:
    n = len(root_ids_by_index)
    if isinstance(tau_ms, Mapping):
        out = np.full(n, 50.0, dtype=np.float64)
        for root_id, value in tau_ms.items():
            idx = np.searchsorted(root_ids_by_index, int(root_id))
            if 0 <= idx < n and root_ids_by_index[idx] == int(root_id):
                out[idx] = float(value)
        return out
    return np.full(n, float(tau_ms), dtype=np.float64)


def _scale_lif_weights(weights: np.ndarray, config: LIFDynamicsConfig) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    abs_values = np.abs(values[np.isfinite(values) & (values != 0)])
    mode = str(config.weight_normalization).lower()
    if mode in {"none", "raw"} or abs_values.size == 0:
        divisor = 1.0
    elif mode == "max":
        divisor = float(abs_values.max())
    elif mode == "percentile":
        percentile = float(np.clip(config.weight_percentile, 1.0, 100.0))
        divisor = float(np.percentile(abs_values, percentile))
    else:
        raise ValueError("weight_normalization must be one of: none, max, percentile")
    divisor = max(divisor, 1e-12)
    scaled = values / divisor
    clip = float(config.weight_clip)
    if clip > 0:
        scaled = np.clip(scaled, -clip, clip)
    return scaled * float(config.synaptic_gain)


def _sparse_or_edge_lif_matrix(
    n_nodes: int,
    pre_idx: np.ndarray,
    post_idx: np.ndarray,
    weights: np.ndarray,
):
    try:
        from scipy import sparse

        return sparse.csr_matrix((weights, (post_idx, pre_idx)), shape=(n_nodes, n_nodes), dtype=np.float64)
    except Exception:
        return None


def run_rate_dynamics(
    edges: pd.DataFrame,
    seed_ids: Iterable[int] | Mapping[int, float],
    config: RateDynamicsConfig | None = None,
) -> pd.DataFrame:
    """Run the rate-dynamics model and return a long-format trace.

    The trace has columns ``root_id``, ``score``, ``step``, ``time_ms`` and is
    drop-in compatible with the schema produced by
    :func:`bio_fly.propagation.signed_multihop_response` once ``time_ms`` is
    dropped.
    """

    config = config or RateDynamicsConfig()
    if config.dt_ms <= 0:
        raise ValueError("dt_ms must be > 0")
    if config.duration_ms <= 0:
        raise ValueError("duration_ms must be > 0")

    seed_weights = {
        int(root_id): float(weight)
        for root_id, weight in (
            seed_ids.items() if isinstance(seed_ids, Mapping) else {int(r): 1.0 for r in seed_ids}.items()
        )
        if pd.notna(weight) and float(weight) != 0.0
    }
    if not seed_weights:
        return pd.DataFrame(columns=["root_id", "score", "step", "time_ms"])

    root_ids_by_index, index_map = _build_index(edges, seed_weights)
    W = _build_dense_matrix(edges, index_map)
    n = W.shape[0]
    tau = _tau_array(config.tau_ms, root_ids_by_index)
    phi = _resolve_nonlinearity(config.nonlinearity)

    seed_drive = np.zeros(n, dtype=np.float64)
    abs_total = sum(abs(v) for v in seed_weights.values()) or 1.0
    for root_id, weight in seed_weights.items():
        idx = index_map.get(int(root_id))
        if idx is not None:
            seed_drive[idx] = weight / abs_total

    delay_steps = _delay_steps_array(edges, config.delay_ms, config.dt_ms)
    pre_idx = edges[PRE_COL].map(index_map).to_numpy(np.int64, copy=False)
    post_idx = edges[POST_COL].map(index_map).to_numpy(np.int64, copy=False)
    edge_w = edges[WEIGHT_COL].to_numpy(np.float64, copy=False)
    has_per_edge_delay = delay_steps is not None

    n_steps = int(round(config.duration_ms / config.dt_ms))
    history_len = (int(delay_steps.max()) + 1) if has_per_edge_delay else 1
    history = np.zeros((history_len, n), dtype=np.float64)
    record_interval_steps = max(1, int(round((config.record_every_ms or config.dt_ms) / config.dt_ms)))
    records: list[tuple[int, float, np.ndarray]] = []

    v = np.zeros(n, dtype=np.float64)
    for step in range(n_steps):
        t_ms = step * config.dt_ms
        envelope = _drive_envelope(config.drive_profile, t_ms, config.duration_ms)
        i_ext = seed_drive * envelope

        if has_per_edge_delay:
            delayed_pre_state = np.zeros(n, dtype=np.float64)
            for e in range(len(edges)):
                d = int(delay_steps[e])
                src = pre_idx[e]
                buf_idx = (step - d) % history_len
                delayed_pre_state[post_idx[e]] += edge_w[e] * history[buf_idx, src]
            synaptic_input = delayed_pre_state + i_ext
        else:
            synaptic_input = W @ v + i_ext

        dv = (-v + phi(synaptic_input)) / tau
        v = v + config.dt_ms * dv

        history[step % history_len] = v

        if step % record_interval_steps == 0 or step == n_steps - 1:
            records.append((step, t_ms, v.copy()))

    if not records:
        return pd.DataFrame(columns=["root_id", "score", "step", "time_ms"])

    frames: list[pd.DataFrame] = []
    for record_idx, (step_idx, t_ms, snapshot) in enumerate(records, start=1):
        nonzero = np.flatnonzero(np.abs(snapshot) > 1e-9)
        if nonzero.size == 0:
            continue
        if config.max_active and nonzero.size > config.max_active:
            top = nonzero[np.argsort(-np.abs(snapshot[nonzero]))[: config.max_active]]
            nonzero = top
        scores = snapshot[nonzero].astype(np.float64)
        if config.normalize_each_record:
            total = float(np.abs(scores).sum())
            if total > 0:
                scores = scores / total
        frames.append(
            pd.DataFrame(
                {
                    "root_id": root_ids_by_index[nonzero],
                    "score": scores,
                    "step": record_idx,
                    "time_ms": float(t_ms),
                }
            )
        )

    if not frames:
        return pd.DataFrame(columns=["root_id", "score", "step", "time_ms"])
    return pd.concat(frames, ignore_index=True)


def run_lif_dynamics(
    edges: pd.DataFrame,
    seed_ids: Iterable[int] | Mapping[int, float],
    config: LIFDynamicsConfig | None = None,
) -> pd.DataFrame:
    """Run a leaky integrate-and-fire surrogate on a FlyWire edge table.

    Output columns include the standard ``root_id``, ``score``, ``step`` and
    ``time_ms`` fields plus spike-specific diagnostics.  ``score`` is derived
    from per-record spike counts and can therefore flow into existing readout
    and behaviour adapters without changing their schema.
    """

    config = config or LIFDynamicsConfig()
    if config.dt_ms <= 0:
        raise ValueError("dt_ms must be > 0")
    if config.duration_ms <= 0:
        raise ValueError("duration_ms must be > 0")
    if config.tau_synapse_ms <= 0:
        raise ValueError("tau_synapse_ms must be > 0")
    if config.v_threshold_mv <= config.v_reset_mv:
        raise ValueError("v_threshold_mv must be greater than v_reset_mv")

    seed_weights = {
        int(root_id): float(weight)
        for root_id, weight in (
            seed_ids.items() if isinstance(seed_ids, Mapping) else {int(r): 1.0 for r in seed_ids}.items()
        )
        if pd.notna(weight) and float(weight) != 0.0
    }
    if not seed_weights:
        return pd.DataFrame(
            columns=["root_id", "score", "step", "time_ms", "spike_count", "firing_rate_hz", "mean_voltage_mv"]
        )

    root_ids_by_index, index_map = _build_index(edges, seed_weights)
    n = len(root_ids_by_index)
    tau = _tau_array(config.tau_membrane_ms, root_ids_by_index)
    tau = np.maximum(tau, float(config.dt_ms))

    if edges.empty:
        pre_idx = np.asarray([], dtype=np.int64)
        post_idx = np.asarray([], dtype=np.int64)
        scaled_weights = np.asarray([], dtype=np.float64)
        W = None
    else:
        pre_idx = edges[PRE_COL].map(index_map).to_numpy(np.int64, copy=False)
        post_idx = edges[POST_COL].map(index_map).to_numpy(np.int64, copy=False)
        raw_weights = edges[WEIGHT_COL].to_numpy(np.float64, copy=False)
        scaled_weights = _scale_lif_weights(raw_weights, config)
        W = _sparse_or_edge_lif_matrix(n, pre_idx, post_idx, scaled_weights)

    seed_drive = np.zeros(n, dtype=np.float64)
    norm_mode = str(config.seed_current_normalization).lower()
    if norm_mode == "sum":
        seed_norm = sum(abs(v) for v in seed_weights.values()) or 1.0
    elif norm_mode == "max":
        seed_norm = max(abs(v) for v in seed_weights.values()) or 1.0
    elif norm_mode in {"none", "raw"}:
        seed_norm = 1.0
    else:
        raise ValueError("seed_current_normalization must be one of: max, sum, none")
    for root_id, weight in seed_weights.items():
        idx = index_map.get(int(root_id))
        if idx is not None:
            seed_drive[idx] = float(config.input_current) * float(weight) / seed_norm

    n_steps = int(round(config.duration_ms / config.dt_ms))
    n_steps = max(n_steps, 1)
    record_interval_steps = max(1, int(round((config.record_every_ms or config.dt_ms) / config.dt_ms)))
    refractory_steps = max(0, int(round(float(config.refractory_ms) / float(config.dt_ms))))
    syn_decay = float(np.exp(-float(config.dt_ms) / max(float(config.tau_synapse_ms), 1e-9)))

    v = np.full(n, float(config.v_rest_mv), dtype=np.float64)
    syn_current = np.zeros(n, dtype=np.float64)
    refractory_remaining = np.zeros(n, dtype=np.int64)
    prev_spikes = np.zeros(n, dtype=np.float64)
    window_spikes = np.zeros(n, dtype=np.float64)
    voltage_sum = np.zeros(n, dtype=np.float64)
    window_steps = 0
    records: list[pd.DataFrame] = []

    for step in range(n_steps):
        t_ms = step * float(config.dt_ms)
        envelope = _drive_envelope(config.drive_profile, t_ms, float(config.duration_ms))
        syn_current *= syn_decay
        if np.any(prev_spikes):
            if W is not None:
                syn_current += np.asarray(W @ prev_spikes, dtype=np.float64).reshape(-1)
            elif len(pre_idx):
                active_edges = prev_spikes[pre_idx] != 0.0
                if np.any(active_edges):
                    np.add.at(syn_current, post_idx[active_edges], scaled_weights[active_edges] * prev_spikes[pre_idx[active_edges]])

        external = seed_drive * envelope
        active_mask = refractory_remaining <= 0
        dv = (
            -(v - float(config.v_rest_mv)) + syn_current + external
        ) / tau * float(config.dt_ms)
        v = np.where(active_mask, v + dv, float(config.v_reset_mv))

        spikes_bool = (v >= float(config.v_threshold_mv)) & active_mask
        prev_spikes = spikes_bool.astype(np.float64)
        if np.any(spikes_bool):
            v[spikes_bool] = float(config.v_reset_mv)
            refractory_remaining[spikes_bool] = refractory_steps
        refractory_remaining = np.maximum(refractory_remaining - 1, 0)

        window_spikes += prev_spikes
        voltage_sum += v
        window_steps += 1

        should_record = (step % record_interval_steps == 0 and step > 0) or step == n_steps - 1
        if not should_record:
            continue

        active_indices = np.flatnonzero(window_spikes > 0)
        if active_indices.size:
            spike_counts = window_spikes[active_indices].astype(np.float64)
            if config.max_active and active_indices.size > config.max_active:
                selected = np.argsort(-spike_counts)[: int(config.max_active)]
                active_indices = active_indices[selected]
                spike_counts = spike_counts[selected]
            record_duration_s = max(float(window_steps) * float(config.dt_ms) / 1000.0, 1e-12)
            firing_rate = spike_counts / record_duration_s
            scores = firing_rate.astype(np.float64)
            if config.normalize_each_record:
                total = float(np.abs(scores).sum())
                if total > 0:
                    scores = scores / total
            records.append(
                pd.DataFrame(
                    {
                        "root_id": root_ids_by_index[active_indices],
                        "score": scores,
                        "step": len(records) + 1,
                        "time_ms": float(t_ms),
                        "spike_count": spike_counts,
                        "firing_rate_hz": firing_rate,
                        "mean_voltage_mv": voltage_sum[active_indices] / float(max(window_steps, 1)),
                    }
                )
            )
        window_spikes[:] = 0.0
        voltage_sum[:] = 0.0
        window_steps = 0

    if not records:
        return pd.DataFrame(
            columns=["root_id", "score", "step", "time_ms", "spike_count", "firing_rate_hz", "mean_voltage_mv"]
        )
    return pd.concat(records, ignore_index=True)


def steady_state_response(
    edges: pd.DataFrame,
    seed_ids: Iterable[int] | Mapping[int, float],
    config: RateDynamicsConfig | None = None,
) -> pd.DataFrame:
    """Convenience wrapper returning only the final-time snapshot."""

    cfg = config or RateDynamicsConfig()
    trace = run_rate_dynamics(edges, seed_ids, cfg)
    if trace.empty:
        return pd.DataFrame(columns=["root_id", "score", "step"])
    last_step = int(trace["step"].max())
    return trace[trace["step"] == last_step][["root_id", "score", "step"]].reset_index(drop=True)


__all__ = [
    "LIFDynamicsConfig",
    "RateDynamicsConfig",
    "run_lif_dynamics",
    "run_rate_dynamics",
    "steady_state_response",
]
