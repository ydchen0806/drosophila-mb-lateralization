from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cable_builder import build_cable_cell_from_compartments


@dataclass(frozen=True)
class CableKCRecipeInputs:
    kc_root_ids: np.ndarray
    kc_compartments: pd.DataFrame
    kc_synapses: pd.DataFrame
    duration_ms: float
    placed_kc_inputs: pd.DataFrame
    global_conductance_scale: float = 1.0
    alpn_to_kc_conductance_scale: float = 5.0
    input_event_interval_ms: float = 5.0
    detector_threshold_mv: float = -55.0
    vm_mv: float = -65.0
    passive_mechanism: str = "pas"


def make_cable_kc_recipe(arbor_module, inputs: CableKCRecipeInputs):
    A = arbor_module

    class BioFlyCableKCRecipe(A.recipe):
        def __init__(self) -> None:
            A.recipe.__init__(self)
            self.kc_root_ids = np.asarray(inputs.kc_root_ids, dtype=np.int64)
            self.duration_ms = float(inputs.duration_ms)
            self.global_conductance_scale = float(inputs.global_conductance_scale)
            self.alpn_to_kc_conductance_scale = float(inputs.alpn_to_kc_conductance_scale)
            self.input_event_interval_ms = float(inputs.input_event_interval_ms)
            self.kc_compartments = {int(root): frame.copy() for root, frame in inputs.kc_compartments.groupby("root_id")}
            self.kc_synapses = inputs.kc_synapses.copy()
            self.kc_input_compartment = self._pick_kc_input_compartments()
            self.placed_kc_inputs = self._aggregate_placed_kc_inputs(inputs.placed_kc_inputs)
            self._cell_cache: dict[int, object] = {}

        def _aggregate_placed_kc_inputs(self, placed: pd.DataFrame) -> dict[int, list[tuple[int, float, float]]]:
            selected: dict[int, list[tuple[int, float, float]]] = {int(root): [] for root in self.kc_root_ids}
            if placed.empty:
                return selected
            required = {"post_root_id", "post_compartment_id", "relative_weight"}
            missing = required - set(placed.columns)
            if missing:
                raise ValueError(f"Placed KC input table missing required columns: {sorted(missing)}")
            allowed = set(map(int, self.kc_root_ids))
            frame = placed[placed["post_root_id"].isin(allowed)].copy()
            if "conductance_scale" not in frame.columns:
                frame["conductance_scale"] = self.alpn_to_kc_conductance_scale
            grouped = (
                frame.groupby(["post_root_id", "post_compartment_id", "conductance_scale"], as_index=False)["relative_weight"]
                .sum()
                .sort_values(["post_root_id", "post_compartment_id", "conductance_scale"])
            )
            for row in grouped.itertuples(index=False):
                weight = float(row.relative_weight)
                if weight > 0:
                    selected.setdefault(int(row.post_root_id), []).append(
                        (int(row.post_compartment_id), weight, float(row.conductance_scale))
                    )
            return selected

        def _pick_kc_input_compartments(self) -> dict[int, int]:
            selected: dict[int, int] = {}
            required = {"post_root_id", "post_compartment_id"}
            if not required.issubset(self.kc_synapses.columns):
                return selected
            frame = self.kc_synapses[self.kc_synapses["post_root_id"].isin(set(map(int, self.kc_root_ids)))]
            if "sign_or_nt" in frame.columns:
                ach = frame[frame["sign_or_nt"].astype(str).str.lower().eq("ach")]
                if len(ach):
                    frame = ach
            for root_id, group in frame.groupby("post_root_id"):
                counts = group["post_compartment_id"].astype("int64").value_counts()
                if len(counts):
                    selected[int(root_id)] = int(counts.index[0])
            return selected

        def num_cells(self):
            return int(len(self.kc_root_ids))

        def cell_kind(self, gid):
            return A.cell_kind.cable

        def global_properties(self, kind):
            return A.neuron_cable_properties()

        def cell_description(self, gid):
            gid = int(gid)
            if gid in self._cell_cache:
                return self._cell_cache[gid]
            root_id = int(self.kc_root_ids[gid])
            compartment_id = self.kc_input_compartment.get(root_id, 0)
            placed_exc = {compartment_id for compartment_id, _weight, _scale in self.placed_kc_inputs.get(root_id, [])}
            if not placed_exc:
                placed_exc = {compartment_id}
            built = build_cable_cell_from_compartments(
                A,
                self.kc_compartments[root_id],
                vm_mv=float(inputs.vm_mv),
                detector_threshold_mv=float(inputs.detector_threshold_mv),
                detector_compartment_id=0,
                excitatory_compartments=placed_exc,
                passive_mechanism=str(inputs.passive_mechanism),
            )
            self._cell_cache[gid] = built.cell
            return built.cell

        def event_generators(self, gid):
            gid = int(gid)
            root_id = int(self.kc_root_ids[gid])
            schedule = A.regular_schedule(
                0.0 * A.units.ms,
                self.input_event_interval_ms * A.units.ms,
                self.duration_ms * A.units.ms,
            )
            generators = []
            for compartment_id, relative_weight, conductance_scale in self.placed_kc_inputs.get(root_id, []):
                weight = float(relative_weight) * self.global_conductance_scale * float(conductance_scale)
                if weight <= 0:
                    continue
                label = f"exc_{compartment_id}" if int(compartment_id) != 0 else "exc"
                generators.append(A.event_generator(A.cell_local_label(label), weight, schedule))
            return generators

    return BioFlyCableKCRecipe()
