from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data_contracts import AplFeedbackParameters, LifCellParameters


@dataclass(frozen=True)
class KCAplRecipeInputs:
    kc_root_ids: np.ndarray
    kc_drive: np.ndarray
    duration_ms: float
    kc_params: LifCellParameters
    apl: AplFeedbackParameters = AplFeedbackParameters()


def make_kc_apl_lif_recipe(arbor_module, inputs: KCAplRecipeInputs):
    A = arbor_module

    class BioFlyKCAplLifRecipe(A.recipe):
        def __init__(self) -> None:
            A.recipe.__init__(self)
            self.kc_root_ids = np.asarray(inputs.kc_root_ids, dtype=np.int64)
            self.kc_drive = np.asarray(inputs.kc_drive, dtype=np.float64)
            self.duration_ms = float(inputs.duration_ms)
            self.kc_params = inputs.kc_params
            self.apl = inputs.apl
            self.apl_gid = int(len(self.kc_root_ids))

        def num_cells(self):
            return int(len(self.kc_root_ids) + 1)

        def cell_kind(self, gid):
            return A.cell_kind.lif

        def cell_description(self, gid):
            p = self.apl.apl_cell if int(gid) == self.apl_gid else self.kc_params
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
            gid = int(gid)
            if gid == self.apl_gid:
                return []
            drive = float(self.kc_drive[gid])
            weight = float(self.kc_params.input_current) + float(self.kc_params.synaptic_gain) * drive
            if weight <= 0:
                return []
            interval = max(float(self.kc_params.input_event_interval_ms), 1e-6)
            schedule = A.regular_schedule(0.0 * A.units.ms, interval * A.units.ms, self.duration_ms * A.units.ms)
            return [A.event_generator(A.cell_local_label("input"), weight, schedule)]

        def connections_on(self, gid):
            gid = int(gid)
            if not bool(self.apl.enabled):
                return []
            delay = float(self.apl.connection_delay_ms) * A.units.ms
            target = A.cell_local_label("input")
            if gid == self.apl_gid:
                return [
                    A.connection(A.cell_global_label(kc_gid, "spike"), target, float(self.apl.kc_to_apl_weight), delay)
                    for kc_gid in range(int(len(self.kc_root_ids)))
                ]
            inhibitory_weight = float(self.apl.apl_to_kc_weight) * float(self.apl.apl_gain)
            if inhibitory_weight == 0:
                return []
            return [A.connection(A.cell_global_label(self.apl_gid, "spike"), target, inhibitory_weight, delay)]

    return BioFlyKCAplLifRecipe()
