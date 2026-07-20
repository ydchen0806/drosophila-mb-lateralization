from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KCOdorInputPanel:
    odor_names: list[str]
    input_matrix: np.ndarray
    kc_root_ids: np.ndarray

    def __post_init__(self) -> None:
        if self.input_matrix.ndim != 2:
            raise ValueError("input_matrix must be two-dimensional.")
        if self.input_matrix.shape[0] != len(self.odor_names):
            raise ValueError("input_matrix row count must match odor_names.")
        if self.input_matrix.shape[1] != len(self.kc_root_ids):
            raise ValueError("input_matrix column count must match kc_root_ids.")


@dataclass(frozen=True)
class LifCellParameters:
    tau_m_ms: float = 20.0
    capacitance_pf: float = 1.0
    resting_potential_mv: float = 0.0
    reset_potential_mv: float = 0.0
    initial_potential_mv: float = 0.0
    refractory_ms: float = 2.0
    v_threshold_mv: float = 0.13
    synaptic_gain: float = 48.0
    input_current: float = 0.0
    input_event_interval_ms: float = 5.0


@dataclass(frozen=True)
class AplFeedbackParameters:
    enabled: bool = True
    apl_gain: float = 1.0
    kc_to_apl_weight: float = 0.001
    apl_to_kc_weight: float = -0.02
    connection_delay_ms: float = 1.0
    apl_cell: LifCellParameters = LifCellParameters(
        tau_m_ms=20.0,
        capacitance_pf=1.0,
        refractory_ms=2.0,
        v_threshold_mv=0.5,
        synaptic_gain=0.0,
        input_event_interval_ms=5.0,
    )


@dataclass(frozen=True)
class ArborRunMetadata:
    arbor_version: str
    requested_threads: str
    resolved_threads: int
    n_cells: int
    dt_ms: float
    duration_ms: float
    model: str
