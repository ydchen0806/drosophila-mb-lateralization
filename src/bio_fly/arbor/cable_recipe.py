from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .cable_builder import build_cable_cell_from_compartments


@dataclass(frozen=True)
class CableKCAplRecipeInputs:
    kc_root_ids: np.ndarray
    kc_compartments: pd.DataFrame
    apl_compartments: pd.DataFrame
    kc_synapses: pd.DataFrame
    apl_synapses: pd.DataFrame
    duration_ms: float
    placed_kc_inputs: pd.DataFrame
    apl_root_id: int = 720575940613583001
    apl_root_ids: tuple[int, ...] = field(default_factory=tuple)
    global_conductance_scale: float = 1.0
    alpn_to_kc_conductance_scale: float = 5.0
    kc_to_apl_conductance_scale: float = 0.001
    apl_to_kc_conductance_scale: float = 0.001
    connectome_weight_mode: str = "raw"
    apl_gain: float = 1.0
    connection_delay_ms: float = 1.0
    kc_detector_threshold_mv: float = -55.0
    apl_detector_threshold_mv: float = -55.0
    force_apl_soma_input: bool = False
    apl_detector_at_input: bool = False


def make_cable_kc_apl_recipe(arbor_module, inputs: CableKCAplRecipeInputs):
    A = arbor_module

    class BioFlyCableKCAplRecipe(A.recipe):
        def __init__(self) -> None:
            A.recipe.__init__(self)
            self.kc_root_ids = np.asarray(inputs.kc_root_ids, dtype=np.int64)
            roots = tuple(int(root) for root in inputs.apl_root_ids) or (int(inputs.apl_root_id),)
            self.apl_root_ids = roots
            self.apl_gid_by_root = {int(root): int(len(self.kc_root_ids) + index) for index, root in enumerate(self.apl_root_ids)}
            self.apl_root_by_gid = {gid: root for root, gid in self.apl_gid_by_root.items()}
            self.duration_ms = float(inputs.duration_ms)
            self.global_conductance_scale = float(inputs.global_conductance_scale)
            self.alpn_to_kc_conductance_scale = float(inputs.alpn_to_kc_conductance_scale)
            self.kc_to_apl_conductance_scale = float(inputs.kc_to_apl_conductance_scale)
            self.apl_to_kc_conductance_scale = float(inputs.apl_to_kc_conductance_scale)
            self.connectome_weight_mode = str(inputs.connectome_weight_mode)
            self.apl_gain = float(inputs.apl_gain)
            self.connection_delay_ms = float(inputs.connection_delay_ms)
            self.kc_compartments = {int(root): frame.copy() for root, frame in inputs.kc_compartments.groupby("root_id")}
            self.apl_compartments = {int(root): frame.copy() for root, frame in inputs.apl_compartments.groupby("root_id") if int(root) in set(self.apl_root_ids)}
            self.kc_synapses = inputs.kc_synapses.copy()
            self.apl_synapses = inputs.apl_synapses.copy()
            self.kc_gid_by_root = {int(root): index for index, root in enumerate(self.kc_root_ids)}
            self.kc_input_compartment = self._pick_kc_input_compartments()
            self.placed_kc_inputs = self._aggregate_placed_kc_inputs(inputs.placed_kc_inputs)
            self.kc_inhibitory_inputs = self._aggregate_apl_to_kc_inputs()
            self.apl_excitatory_inputs = self._aggregate_kc_to_apl_inputs()
            self.apl_exc_compartments = {
                (apl_root, kc_root): compartment_id
                for apl_root, rows in self.apl_excitatory_inputs.items()
                for kc_root, compartment_id, _weight in rows
            }
            self.apl_detector_compartments = self._pick_apl_detector_compartments()
            self._cell_cache: dict[int, object] = {}
            self._branch_cache: dict[int, dict[int, int]] = {}
            self._exc_label_cache: dict[int, dict[int, str]] = {}
            self._inh_label_cache: dict[int, dict[int, str]] = {}

        def _pick_apl_detector_compartments(self) -> set[int]:
            if not bool(inputs.apl_detector_at_input):
                return {0}
            values = {int(value) for value in self.apl_exc_compartments.values()}
            return values if values else {0}

        def _pick_kc_input_compartments(self) -> dict[int, int]:
            selected: dict[int, int] = {}
            if {"post_root_id", "post_compartment_id"}.issubset(self.kc_synapses.columns):
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

        def _aggregate_apl_to_kc_inputs(self) -> dict[int, list[tuple[int, int, float]]]:
            selected: dict[int, list[tuple[int, int, float]]] = {int(root): [] for root in self.kc_root_ids}
            required = {"pre_root_id", "post_root_id", "post_compartment_id", "weight"}
            if not required.issubset(self.kc_synapses.columns):
                return selected
            frame = self.kc_synapses[
                self.kc_synapses["pre_root_id"].isin(set(self.apl_root_ids))
                & self.kc_synapses["post_root_id"].isin(set(map(int, self.kc_root_ids)))
            ].copy()
            if frame.empty:
                return selected
            if "attenuation_to_soma" in frame.columns:
                frame["relative_weight"] = self._relative_weight(frame)
            else:
                frame["relative_weight"] = frame["weight"].astype("float64")
            grouped = (
                frame.groupby(["pre_root_id", "post_root_id", "post_compartment_id"], as_index=False)["relative_weight"]
                .sum()
                .sort_values("relative_weight", ascending=False)
            )
            for row in grouped.itertuples(index=False):
                selected.setdefault(int(row.post_root_id), []).append(
                    (int(row.pre_root_id), int(row.post_compartment_id), float(row.relative_weight))
                )
            return selected

        def _aggregate_kc_to_apl_inputs(self) -> dict[int, list[tuple[int, int, float]]]:
            selected: dict[int, list[tuple[int, int, float]]] = {int(root): [] for root in self.apl_root_ids}
            if {"pre_root_id", "post_root_id", "post_compartment_id"}.issubset(self.apl_synapses.columns):
                if bool(inputs.force_apl_soma_input):
                    return {
                        int(apl_root): [(int(kc_root), 0, 1.0) for kc_root in self.kc_root_ids]
                        for apl_root in self.apl_root_ids
                    }
                frame = self.apl_synapses[
                    self.apl_synapses["pre_root_id"].isin(set(map(int, self.kc_root_ids)))
                    & self.apl_synapses["post_root_id"].isin(set(self.apl_root_ids))
                ].copy()
                if frame.empty:
                    return selected
                frame["relative_weight"] = self._relative_weight(frame)
                grouped = (
                    frame.groupby(["post_root_id", "pre_root_id", "post_compartment_id"], as_index=False)["relative_weight"]
                    .sum()
                    .sort_values("relative_weight", ascending=False)
                )
                for row in grouped.itertuples(index=False):
                    selected.setdefault(int(row.post_root_id), []).append(
                        (int(row.pre_root_id), int(row.post_compartment_id), float(row.relative_weight))
                    )
            return selected

        def _relative_weight(self, frame: pd.DataFrame) -> pd.Series:
            weight = frame["weight"].astype("float64")
            if self.connectome_weight_mode == "raw":
                return weight
            if self.connectome_weight_mode == "attenuated":
                if "attenuation_to_soma" not in frame.columns:
                    return weight
                return weight * frame["attenuation_to_soma"].astype("float64")
            raise ValueError(f"Unknown connectome_weight_mode: {self.connectome_weight_mode}")

        def num_cells(self):
            return int(len(self.kc_root_ids) + len(self.apl_root_ids))

        def cell_kind(self, gid):
            return A.cell_kind.cable

        def global_properties(self, kind):
            return A.neuron_cable_properties()

        def _root_for_gid(self, gid: int) -> int:
            return self.apl_root_by_gid[int(gid)] if int(gid) in self.apl_root_by_gid else int(self.kc_root_ids[int(gid)])

        def cell_description(self, gid):
            gid = int(gid)
            if gid in self._cell_cache:
                return self._cell_cache[gid]
            root_id = self._root_for_gid(gid)
            is_apl = gid in self.apl_root_by_gid
            compartments = self.apl_compartments[root_id] if is_apl else self.kc_compartments[root_id]
            if is_apl:
                exc_compartments = {
                    compartment_id
                    for _kc_root, compartment_id, _weight in self.apl_excitatory_inputs.get(int(root_id), [])
                }
                inh_compartments: set[int] = set()
            else:
                exc_compartments = {
                    compartment_id
                    for compartment_id, _weight, _scale in self.placed_kc_inputs.get(root_id, [])
                }
                if not exc_compartments:
                    exc_compartments = {self.kc_input_compartment.get(root_id, 0)}
                inh_compartments = {compartment_id for _apl_root, compartment_id, _weight in self.kc_inhibitory_inputs.get(root_id, [])}
                if not inh_compartments:
                    inh_compartments = {self.kc_input_compartment.get(root_id, 0)}
            built = build_cable_cell_from_compartments(
                A,
                compartments,
                detector_threshold_mv=float(inputs.apl_detector_threshold_mv if is_apl else inputs.kc_detector_threshold_mv),
                detector_compartment_id=0,
                detector_compartment_ids=self.apl_detector_compartments if is_apl else {0},
                excitatory_compartments=exc_compartments,
                inhibitory_compartments=inh_compartments,
            )
            self._cell_cache[gid] = built.cell
            self._branch_cache[gid] = built.compartment_to_branch
            self._exc_label_cache[gid] = built.excitatory_labels
            self._inh_label_cache[gid] = built.inhibitory_labels
            return built.cell

        def event_generators(self, gid):
            gid = int(gid)
            if gid in self.apl_root_by_gid:
                return []
            root_id = int(self.kc_root_ids[gid])
            schedule = A.regular_schedule(0.0 * A.units.ms, 5.0 * A.units.ms, self.duration_ms * A.units.ms)
            generators = []
            for compartment_id, relative_weight, conductance_scale in self.placed_kc_inputs.get(root_id, []):
                weight = float(relative_weight) * self.global_conductance_scale * float(conductance_scale)
                if weight <= 0:
                    continue
                label = f"exc_{compartment_id}" if int(compartment_id) != 0 else "exc"
                generators.append(A.event_generator(A.cell_local_label(label), weight, schedule))
            return generators

        def connections_on(self, gid):
            gid = int(gid)
            delay = self.connection_delay_ms * A.units.ms
            if gid in self.apl_root_by_gid:
                apl_root = self.apl_root_by_gid[gid]
                connections = []
                for kc_root, compartment_id, relative_weight in self.apl_excitatory_inputs.get(int(apl_root), []):
                    kc_gid = self.kc_gid_by_root.get(int(kc_root))
                    if kc_gid is None:
                        continue
                    weight = float(relative_weight) * self.global_conductance_scale * self.kc_to_apl_conductance_scale
                    if weight <= 0:
                        continue
                    label = f"exc_{compartment_id}" if compartment_id != 0 else "exc"
                    connections.append(A.connection(A.cell_global_label(kc_gid, "spike"), A.cell_local_label(label), weight, delay))
                return connections
            if self.apl_to_kc_conductance_scale == 0 or self.apl_gain == 0 or bool(inputs.apl_detector_at_input):
                return []
            root_id = int(self.kc_root_ids[gid])
            connections = []
            for apl_root, compartment_id, relative_weight in self.kc_inhibitory_inputs.get(root_id, []):
                apl_gid = self.apl_gid_by_root.get(int(apl_root))
                if apl_gid is None:
                    continue
                weight = float(relative_weight) * self.global_conductance_scale * self.apl_to_kc_conductance_scale * self.apl_gain
                if weight <= 0:
                    continue
                label = f"inh_{compartment_id}" if compartment_id != 0 else "inh"
                connections.append(A.connection(A.cell_global_label(apl_gid, "spike"), A.cell_local_label(label), weight, delay))
            return connections

    return BioFlyCableKCAplRecipe()
