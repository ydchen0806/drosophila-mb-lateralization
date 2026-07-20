from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CableCellBuildResult:
    cell: object
    compartment_to_branch: dict[int, int]
    excitatory_labels: dict[int, str]
    inhibitory_labels: dict[int, str]


def build_cable_cell_from_compartments(
    arbor_module,
    compartments: pd.DataFrame,
    *,
    detector_label: str = "spike",
    excitatory_label: str = "exc",
    inhibitory_label: str = "inh",
    excitatory_synapse: str | tuple[str, dict[str, float]] = ("expsyn", {"e": 0.0}),
    inhibitory_synapse: str | tuple[str, dict[str, float]] = ("expsyn", {"e": -70.0}),
    vm_mv: float = -65.0,
    detector_threshold_mv: float = -40.0,
    detector_compartment_id: int = 0,
    detector_compartment_ids: set[int] | None = None,
    passive_mechanism: str = "pas",
    excitatory_compartments: set[int] | None = None,
    inhibitory_compartments: set[int] | None = None,
) -> CableCellBuildResult:
    A = arbor_module
    required = {"compartment_id", "parent_compartment_id", "length_um", "mean_radius_um"}
    missing = required - set(compartments.columns)
    if missing:
        raise ValueError(f"Compartment table missing required columns: {sorted(missing)}")
    frame = compartments.sort_values("compartment_id").copy()
    children_by_parent: dict[int, list[int]] = {}
    for row in frame.itertuples(index=False):
        children_by_parent.setdefault(int(row.parent_compartment_id), []).append(int(row.compartment_id))
    child_rank: dict[int, tuple[int, int]] = {}
    for parent_id, children in children_by_parent.items():
        ordered = sorted(children)
        total = len(ordered)
        for index, child_id in enumerate(ordered):
            child_rank[int(child_id)] = (index, total)

    tree = A.segment_tree()
    segment_by_compartment: dict[int, int] = {}
    endpoint_by_compartment: dict[int, tuple[float, float, float]] = {}
    segment_key_by_compartment: dict[int, tuple[float, float, float, float, float, float]] = {}
    for row in frame.itertuples(index=False):
        compartment_id = int(row.compartment_id)
        parent_id = int(row.parent_compartment_id)
        radius = max(float(row.mean_radius_um), 0.05)
        length = max(float(row.length_um), 0.1)
        if parent_id < 0 or parent_id not in segment_by_compartment:
            parent_branch = A.mnpos
            start = (0.0, 0.0, 0.0)
            angle = 0.0
        else:
            parent_branch = segment_by_compartment[parent_id]
            start = endpoint_by_compartment[parent_id]
            index, total = child_rank.get(compartment_id, (0, 1))
            angle = 0.0 if total <= 1 else (2.0 * np.pi * float(index) / float(total)) + 0.37 * float(parent_id % 7)
        end = (start[0] + length * np.cos(angle), start[1] + length * np.sin(angle), start[2] + 0.01 * float(compartment_id % 11))
        segment = tree.append(
            parent_branch,
            A.mpoint(start[0], start[1], start[2], radius),
            A.mpoint(end[0], end[1], end[2], radius),
            1,
        )
        segment_by_compartment[compartment_id] = int(segment)
        endpoint_by_compartment[compartment_id] = end
        segment_key_by_compartment[compartment_id] = _segment_key(start, end)

    compartment_to_branch = _map_compartments_to_branches(A, tree, segment_key_by_compartment)

    decor = A.decor()
    decor.set_property(Vm=float(vm_mv) * A.units.mV)
    decor.paint("(all)", A.density(passive_mechanism))
    detector_ids = {int(detector_compartment_id)} | {int(value) for value in (detector_compartment_ids or set())}
    for detector_id in sorted(detector_ids):
        detector_branch = compartment_to_branch.get(int(detector_id), 0)
        detector_pos = 0.0 if int(detector_id) == 0 else 0.5
        decor.place(
            f"(location {detector_branch} {detector_pos})",
            A.threshold_detector(float(detector_threshold_mv) * A.units.mV),
            detector_label,
        )
    excitatory_labels: dict[int, str] = {}
    inhibitory_labels: dict[int, str] = {}
    exc_compartments = {0} | {int(value) for value in (excitatory_compartments or set())}
    inh_compartments = {0} | {int(value) for value in (inhibitory_compartments or set())}
    for compartment_id in sorted(exc_compartments):
        branch = compartment_to_branch.get(int(compartment_id), 0)
        label = excitatory_label if int(compartment_id) == 0 else f"{excitatory_label}_{int(compartment_id)}"
        decor.place(f"(location {branch} 0.5)", _make_synapse(A, excitatory_synapse), label)
        excitatory_labels[int(compartment_id)] = label
    for compartment_id in sorted(inh_compartments):
        branch = compartment_to_branch.get(int(compartment_id), 0)
        label = inhibitory_label if int(compartment_id) == 0 else f"{inhibitory_label}_{int(compartment_id)}"
        decor.place(f"(location {branch} 0.5)", _make_synapse(A, inhibitory_synapse), label)
        inhibitory_labels[int(compartment_id)] = label
    cell = A.cable_cell(tree, decor)
    cell.discretization(A.cv_policy_fixed_per_branch(1))
    return CableCellBuildResult(cell, compartment_to_branch, excitatory_labels, inhibitory_labels)


def _make_synapse(arbor_module, spec: str | tuple[str, dict[str, float]]):
    if isinstance(spec, tuple):
        name, params = spec
        return arbor_module.synapse(str(name), {str(key): float(value) for key, value in params.items()})
    return arbor_module.synapse(str(spec))


def _segment_key(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> tuple[float, float, float, float, float, float]:
    return (
        round(float(start[0]), 6),
        round(float(start[1]), 6),
        round(float(start[2]), 6),
        round(float(end[0]), 6),
        round(float(end[1]), 6),
        round(float(end[2]), 6),
    )


def _map_compartments_to_branches(
    arbor_module,
    tree,
    segment_key_by_compartment: dict[int, tuple[float, float, float, float, float, float]],
) -> dict[int, int]:
    morphology = arbor_module.morphology(tree)
    branch_by_segment_key: dict[tuple[float, float, float, float, float, float], int] = {}
    for branch in range(int(morphology.num_branches)):
        for segment in morphology.branch_segments(branch):
            key = (
                round(float(segment.prox.x), 6),
                round(float(segment.prox.y), 6),
                round(float(segment.prox.z), 6),
                round(float(segment.dist.x), 6),
                round(float(segment.dist.y), 6),
                round(float(segment.dist.z), 6),
            )
            branch_by_segment_key[key] = int(branch)
    return {
        int(compartment_id): int(branch_by_segment_key.get(key, 0))
        for compartment_id, key in segment_key_by_compartment.items()
    }


def cable_location_for_compartment(arbor_module, branch_by_compartment: dict[int, int], compartment_id: int):
    branch = branch_by_compartment.get(int(compartment_id))
    if branch is None:
        branch = 0
    return arbor_module.location(int(branch), 0.5)


def sample_roots_by_compartment_count(
    compartments: pd.DataFrame,
    *,
    max_roots: int,
    max_compartments_per_root: int | None = None,
    allowed_roots: set[int] | None = None,
) -> list[int]:
    counts = compartments.groupby("root_id").size().sort_values()
    if allowed_roots is not None:
        counts = counts[counts.index.astype("int64").isin({int(root) for root in allowed_roots})]
    if max_compartments_per_root is not None and int(max_compartments_per_root) > 0:
        filtered = counts[counts <= int(max_compartments_per_root)]
        if len(filtered) >= int(max_roots):
            counts = filtered
    return [int(root_id) for root_id in counts.head(int(max_roots)).index]
