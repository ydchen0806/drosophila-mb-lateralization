from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data_contracts import LifCellParameters


@dataclass(frozen=True)
class KCLifRecipeInputs:
    kc_root_ids: np.ndarray
    kc_drive: np.ndarray
    duration_ms: float
    params: LifCellParameters


def make_kc_lif_recipe(arbor_module, inputs: KCLifRecipeInputs):
    A = arbor_module

    class BioFlyKCLifRecipe(A.recipe):
        def __init__(self) -> None:
            A.recipe.__init__(self)
            self.kc_root_ids = np.asarray(inputs.kc_root_ids, dtype=np.int64)
            self.kc_drive = np.asarray(inputs.kc_drive, dtype=np.float64)
            self.params = inputs.params
            self.duration_ms = float(inputs.duration_ms)

        def num_cells(self):
            return int(len(self.kc_root_ids))

        def cell_kind(self, gid):
            return A.cell_kind.lif

        def cell_description(self, gid):
            p = self.params
            U = A.units
            return A.lif_cell(
                "spike",
                "input",
                tau_m=float(p.tau_m_ms) * U.ms,
                V_th=float(p.v_threshold_mv) * U.mV,
                C_m=float(p.capacitance_pf) * U.pF,
                E_L=float(p.resting_potential_mv) * U.mV,
                E_R=float(p.reset_potential_mv) * U.mV,
                V_m=float(p.initial_potential_mv) * U.mV,
                t_ref=float(p.refractory_ms) * U.ms,
            )

        def event_generators(self, gid):
            drive = float(self.kc_drive[int(gid)])
            weight = float(self.params.input_current) + float(self.params.synaptic_gain) * drive
            if weight <= 0:
                return []
            interval = max(float(self.params.input_event_interval_ms), 1e-6)
            schedule = A.regular_schedule(0.0 * A.units.ms, interval * A.units.ms, self.duration_ms * A.units.ms)
            target = A.cell_local_label("input")
            return [A.event_generator(target, weight, schedule)]

    return BioFlyKCLifRecipe()
