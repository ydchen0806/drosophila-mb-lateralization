"""Morphology-aware utilities for FlyWire-derived skeletons.

This module intentionally keeps the first morphology layer lightweight:

* load cached skeleton pickle files already present in local workpacks;
* load whole-brain FlyWire skeleton parquet rows for selected root ids;
* map real FlyWire synapse positions onto postsynaptic skeletons;
* compute graph features such as path distance and branch order;
* export validated SWC files for future NEURON Import3D use;
* run a passive morphology surrogate for proximal/distal comparisons;
* build morphology-weighted edge tables for connectome propagation.

The passive surrogate is not a replacement for NEURON.  It is a deterministic
forward-compatible baseline that lets the existing connectome pipeline account
for input location while keeping NEURON as an optional later backend.
"""

from __future__ import annotations

import heapq
import json
import math
import pickle
import warnings
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .paths import DEFAULT_CONNECTIVITY_PATH, DEFAULT_OUTPUT_ROOT, PROCESSED_DATA_ROOT, PROJECT_ROOT, RAW_DATA_ROOT


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SKELETON_PARQUET_PATHS = (
    RAW_DATA_ROOT / "zenodo_10877326" / "sk_lod1_783_healed_ds2.parquet",
    REPO_ROOT / "data" / "raw" / "zenodo_10877326" / "sk_lod1_783_healed_ds2.parquet",
)
DEFAULT_SYNAPSES_FEATHER_PATHS = (
    RAW_DATA_ROOT / "zenodo_10676866" / "flywire_synapses_783.feather",
    REPO_ROOT / "data" / "raw" / "zenodo_10676866" / "flywire_synapses_783.feather",
)
DEFAULT_ANNOTATION_PATHS = (
    PROCESSED_DATA_ROOT / "flywire_neuron_annotations.parquet",
    REPO_ROOT / "data" / "processed" / "flywire_neuron_annotations.parquet",
)
DEFAULT_CONNECTIVITY_PATHS = (
    DEFAULT_CONNECTIVITY_PATH,
    REPO_ROOT / "external" / "Drosophila_brain_model" / "Connectivity_783.parquet",
)
DEFAULT_KC_SKELETON_DIRS = (
    PROJECT_ROOT / "data" / "processed" / "kc_skeletons_i",
    PROJECT_ROOT
    / "workpacks"
    / "neuron_conn_kc_topology_pack"
    / "data"
    / "FlyWire"
    / "skeletons_i",
    REPO_ROOT
    / "workpacks"
    / "neuron_conn_kc_topology_pack"
    / "data"
    / "FlyWire"
    / "skeletons_i",
)


@dataclass(frozen=True)
class MorphologyGraph:
    """A small undirected cable graph with per-node morphology features."""

    root_id: int
    nodes: pd.DataFrame
    edges: tuple[tuple[int, int], ...]
    soma_node: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict)

    def adjacency(self) -> list[list[int]]:
        adjacency = [[] for _ in range(len(self.nodes))]
        for left, right in self.edges:
            adjacency[int(left)].append(int(right))
            adjacency[int(right)].append(int(left))
        return adjacency

    def coordinates(self) -> np.ndarray:
        return self.nodes[["x_um", "y_um", "z_um"]].to_numpy(dtype=np.float64)

    def edge_lengths(self) -> np.ndarray:
        coords = self.coordinates()
        return np.asarray(
            [float(np.linalg.norm(coords[int(left)] - coords[int(right)])) for left, right in self.edges],
            dtype=np.float64,
        )

    def largest_component(self) -> "MorphologyGraph":
        labels = connected_component_labels(len(self.nodes), self.edges)
        if len(set(labels)) <= 1:
            return self
        counts = pd.Series(labels).value_counts()
        largest_label = int(counts.index[0])
        keep_old = [idx for idx, label in enumerate(labels) if int(label) == largest_label]
        old_to_new = {old: new for new, old in enumerate(keep_old)}
        nodes = self.nodes.iloc[keep_old].copy().reset_index(drop=True)
        nodes["node_id"] = np.arange(len(nodes), dtype=np.int64)
        edges = tuple(
            (old_to_new[left], old_to_new[right])
            for left, right in self.edges
            if left in old_to_new and right in old_to_new
        )
        soma_node = old_to_new.get(int(self.soma_node), 0)
        if "is_soma" in nodes.columns:
            nodes["is_soma"] = False
            nodes.loc[soma_node, "is_soma"] = True
        metadata = dict(self.metadata)
        metadata["component_filter"] = "largest"
        metadata["n_nodes_before_component_filter"] = int(len(self.nodes))
        return MorphologyGraph(
            root_id=self.root_id,
            nodes=nodes,
            edges=edges,
            soma_node=soma_node,
            metadata=metadata,
        )


@dataclass(frozen=True)
class PassiveSurrogateConfig:
    """Parameters for a passive morphology response surrogate."""

    length_constant_um: float = 120.0
    conduction_velocity_um_per_ms: float = 200.0
    base_delay_ms: float = 1.0
    response_tau_ms: float = 8.0
    baseline_mv: float = -65.0
    input_amplitude_mv: float = 8.0
    branch_order_penalty: float = 0.04
    tstop_ms: float = 80.0
    dt_ms: float = 0.1


@dataclass(frozen=True)
class MorphologyNodeMetrics:
    """Reusable per-node morphology arrays for one graph and response config."""

    path_distances_um: np.ndarray
    electrotonic_distances_proxy: np.ndarray
    branch_orders: np.ndarray
    degrees: np.ndarray
    cable_sampling_probabilities: np.ndarray
    attenuation: np.ndarray
    delay_ms: np.ndarray
    soma_peak_delta_mv: np.ndarray
    time_to_peak_ms: np.ndarray


@dataclass(frozen=True)
class KCMorphologyBenchmarkConfig:
    """Configuration for the KC morphology benchmark."""

    skeleton_dir: Path | None = None
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_ROOT / "kc_morphology_benchmark")
    root_ids: tuple[int, ...] = ()
    max_neurons: int | None = 50
    coordinate_scale_um: float = 0.001
    radius_scale_um: float = 0.001
    min_radius_um: float = 0.05
    location_quantiles: tuple[float, ...] = (0.2, 0.5, 0.85)
    response: PassiveSurrogateConfig = field(default_factory=PassiveSurrogateConfig)


@dataclass(frozen=True)
class MorphologyWeightedConnectivityConfig:
    """Configuration for real-synapse morphology-weighted connectivity."""

    root_ids: tuple[int, ...]
    skeleton_parquet: Path | None = None
    synapses_feather: Path | None = None
    annotation_path: Path | None = None
    connectivity_path: Path | None = None
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_ROOT / "morphology_weighted_connectivity")
    max_synapses_per_root: int | None = 20000
    coordinate_scale_um: float = 0.001
    radius_scale_um: float = 0.001
    min_radius_um: float = 0.05
    nearest_node_max_distance_um: float | None = None
    save_mapped_synapses: bool = False
    write_adjusted_connectivity: bool = False
    adjusted_connectivity_mode: str = "scale_existing"
    random_control_repeats: int = 0
    random_control_seed: int = 0
    edge_random_controls: bool = False
    edge_random_control_min_synapses: int = 3
    response: PassiveSurrogateConfig = field(default_factory=PassiveSurrogateConfig)


def _resolve_existing_path(path: Path | None, candidates: Sequence[Path], label: str) -> Path:
    if path is not None:
        resolved = Path(path).expanduser().resolve()
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    checked = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"No {label} found. Pass the path explicitly. Checked:\n{checked}")


def resolve_skeleton_parquet(skeleton_parquet: Path | None = None) -> Path:
    """Return the local whole-brain FlyWire skeleton parquet path."""

    return _resolve_existing_path(skeleton_parquet, DEFAULT_SKELETON_PARQUET_PATHS, "FlyWire skeleton parquet")


def resolve_synapses_feather(synapses_feather: Path | None = None) -> Path:
    """Return the local FlyWire synapse feather path."""

    return _resolve_existing_path(synapses_feather, DEFAULT_SYNAPSES_FEATHER_PATHS, "FlyWire synapses feather")


def resolve_annotation_path(annotation_path: Path | None = None) -> Path:
    """Return the local FlyWire neuron annotation parquet path."""

    return _resolve_existing_path(annotation_path, DEFAULT_ANNOTATION_PATHS, "FlyWire annotation parquet")


def resolve_connectivity_path(connectivity_path: Path | None = None) -> Path:
    """Return the local FlyWire connectivity parquet path."""

    return _resolve_existing_path(connectivity_path, DEFAULT_CONNECTIVITY_PATHS, "FlyWire connectivity parquet")


def resolve_kc_skeleton_dir(skeleton_dir: Path | None = None) -> Path:
    """Return an existing KC skeleton directory or raise a useful error."""

    if skeleton_dir is not None:
        path = Path(skeleton_dir).expanduser().resolve()
        if path.is_dir():
            return path
        raise FileNotFoundError(f"KC skeleton directory does not exist: {path}")
    for candidate in DEFAULT_KC_SKELETON_DIRS:
        if candidate.is_dir():
            return candidate.resolve()
    candidates = "\n".join(str(path) for path in DEFAULT_KC_SKELETON_DIRS)
    raise FileNotFoundError(
        "No KC skeleton cache found. Pass --skeleton-dir explicitly. Checked:\n" + candidates
    )


def list_skeleton_root_ids(skeleton_dir: Path) -> tuple[int, ...]:
    ids: list[int] = []
    for path in sorted(Path(skeleton_dir).glob("*.pkl")):
        try:
            ids.append(int(path.stem))
        except ValueError:
            continue
    return tuple(ids)


def connected_component_labels(n_nodes: int, edges: Sequence[tuple[int, int]]) -> list[int]:
    adjacency = [[] for _ in range(n_nodes)]
    for left, right in edges:
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    labels = [-1] * n_nodes
    label = 0
    for start in range(n_nodes):
        if labels[start] >= 0:
            continue
        stack = [start]
        labels[start] = label
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if labels[neighbor] < 0:
                    labels[neighbor] = label
                    stack.append(neighbor)
        label += 1
    return labels


def load_kc_skeleton_pickle(
    path: Path,
    *,
    root_id: int | None = None,
    coordinate_scale_um: float = 0.001,
    radius_scale_um: float = 0.001,
    min_radius_um: float = 0.05,
    keep_largest_component: bool = True,
) -> MorphologyGraph:
    """Load a KC skeleton pickle from the local topology workpack.

    The expected pickle shape is ``{"features": array, "neighbors": dict}``.
    Feature columns follow the existing workpack convention:

    ``x, y, z, radius, soma_flag, pre_synapse_count, post_synapse_count, partner_hint``.
    Coordinates and radii are scaled into micrometers by the caller-supplied
    scale factors.
    """

    path = Path(path)
    with path.open("rb") as handle:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, message="numpy.core.*")
            payload = pickle.load(handle)
    features = np.asarray(payload["features"], dtype=np.float64)
    if features.ndim != 2 or features.shape[1] < 4:
        raise ValueError(f"Skeleton features must be an Nx4+ matrix: {path}")
    neighbors = payload.get("neighbors", {})
    n_nodes = int(features.shape[0])
    inferred_root_id = int(root_id if root_id is not None else path.stem)

    radius = np.maximum(features[:, 3] * float(radius_scale_um), float(min_radius_um))
    nodes = pd.DataFrame(
        {
            "node_id": np.arange(n_nodes, dtype=np.int64),
            "x_um": features[:, 0] * float(coordinate_scale_um),
            "y_um": features[:, 1] * float(coordinate_scale_um),
            "z_um": features[:, 2] * float(coordinate_scale_um),
            "radius_um": radius,
            "is_soma": features[:, 4] > 0 if features.shape[1] > 4 else False,
            "pre_synapse_count": features[:, 5] if features.shape[1] > 5 else 0.0,
            "post_synapse_count": features[:, 6] if features.shape[1] > 6 else 0.0,
            "partner_hint": features[:, 7] if features.shape[1] > 7 else 0.0,
        }
    )
    soma_candidates = nodes.index[nodes["is_soma"].astype(bool)].tolist()
    soma_node = int(soma_candidates[0]) if soma_candidates else 0

    edge_set: set[tuple[int, int]] = set()
    for left, values in neighbors.items():
        left_i = int(left)
        if left_i < 0 or left_i >= n_nodes:
            continue
        for right in values:
            right_i = int(right)
            if right_i < 0 or right_i >= n_nodes or right_i == left_i:
                continue
            edge_set.add(tuple(sorted((left_i, right_i))))
    graph = MorphologyGraph(
        root_id=inferred_root_id,
        nodes=nodes,
        edges=tuple(sorted(edge_set)),
        soma_node=soma_node,
        metadata={"source": str(path), "format": "workpack_skeleton_pickle"},
    )
    return graph.largest_component() if keep_largest_component else graph


def load_kc_skeleton(
    root_id: int,
    skeleton_dir: Path | None = None,
    **kwargs: object,
) -> MorphologyGraph:
    directory = resolve_kc_skeleton_dir(skeleton_dir)
    path = directory / f"{int(root_id)}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"KC skeleton pickle not found for root_id={root_id}: {path}")
    return load_kc_skeleton_pickle(path, root_id=int(root_id), **kwargs)


def skeleton_parquet_soma_node_ids(
    root_ids: Sequence[int],
    skeleton_parquet: Path | None = None,
) -> dict[int, int]:
    """Read root-id -> soma node-id hints from skeleton parquet metadata."""

    path = resolve_skeleton_parquet(skeleton_parquet)
    import pyarrow.parquet as pq

    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    soma_ids: dict[int, int] = {}
    for root_id in root_ids:
        key = f"{int(root_id)}:soma".encode("utf-8")
        value = metadata.get(key)
        if value is None:
            continue
        try:
            soma_ids[int(root_id)] = int(value.decode("utf-8"))
        except ValueError:
            continue
    return soma_ids


def load_skeleton_parquet_rows(
    root_ids: Sequence[int],
    skeleton_parquet: Path | None = None,
    *,
    columns: Sequence[str] = ("node_id", "parent_id", "radius", "x", "y", "z", "neuron"),
) -> pd.DataFrame:
    """Load skeleton parquet rows for selected root ids.

    The raw file is a whole-brain table, so this function always applies a
    dataset filter on ``neuron`` and should not be used without explicit
    ``root_ids``.
    """

    unique_root_ids = sorted({int(root_id) for root_id in root_ids})
    if not unique_root_ids:
        return pd.DataFrame(columns=list(columns))
    path = resolve_skeleton_parquet(skeleton_parquet)
    import pyarrow.dataset as ds

    dataset = ds.dataset(path, format="parquet")
    table = dataset.to_table(
        columns=list(columns),
        filter=ds.field("neuron").isin(unique_root_ids),
    )
    return table.to_pandas()


def load_annotation_soma_positions(
    root_ids: Sequence[int],
    annotation_path: Path | None = None,
    *,
    coordinate_scale_um: float = 0.001,
) -> dict[int, np.ndarray]:
    """Load soma coordinates from the processed FlyWire annotation table."""

    try:
        path = resolve_annotation_path(annotation_path)
    except FileNotFoundError:
        if annotation_path is None:
            return {}
        raise
    columns = ["root_id", "soma_x", "soma_y", "soma_z"]
    frame = pd.read_parquet(path, columns=columns)
    root_set = {int(root_id) for root_id in root_ids}
    frame = frame[frame["root_id"].astype("int64").isin(root_set)].copy()
    positions: dict[int, np.ndarray] = {}
    for _, row in frame.iterrows():
        coords = np.asarray([row["soma_x"], row["soma_y"], row["soma_z"]], dtype=np.float64)
        if np.isfinite(coords).all():
            positions[int(row["root_id"])] = coords * float(coordinate_scale_um)
    return positions


def select_annotation_root_ids(
    query: str,
    annotation_path: Path | None = None,
    *,
    max_roots: int | None = None,
) -> tuple[int, ...]:
    """Select FlyWire root ids by regex over available annotation text columns."""

    text_query = str(query or "").strip()
    if not text_query:
        return ()
    path = resolve_annotation_path(annotation_path)
    frame = pd.read_parquet(path)
    if "root_id" not in frame.columns:
        raise ValueError(f"Annotation table is missing root_id: {path}")
    text_columns = [
        column
        for column in (
            "super_class",
            "cell_class",
            "cell_sub_class",
            "supertype",
            "cell_type",
            "hemibrain_type",
            "ito_lee_hemilineage",
            "hartenstein_hemilineage",
            "side",
            "top_nt",
            "known_nt",
            "nerve",
            "synonyms",
        )
        if column in frame.columns
    ]
    if text_columns:
        text = frame[text_columns].fillna("").astype(str).agg(" ".join, axis=1)
        mask = text.str.contains(text_query, case=False, regex=True, na=False)
    else:
        mask = pd.Series(False, index=frame.index)
    roots = frame.loc[mask, "root_id"].dropna().astype("int64").drop_duplicates().tolist()
    roots = sorted(int(root_id) for root_id in roots)
    if max_roots is not None:
        roots = roots[: max(0, int(max_roots))]
    return tuple(roots)


def morphology_graph_from_skeleton_rows(
    rows: pd.DataFrame,
    *,
    root_id: int | None = None,
    soma_node_id: int | None = None,
    soma_xyz_um: Sequence[float] | None = None,
    coordinate_scale_um: float = 0.001,
    radius_scale_um: float = 0.001,
    min_radius_um: float = 0.05,
    source_path: Path | None = None,
    keep_largest_component: bool = True,
) -> MorphologyGraph:
    """Convert FlyWire skeleton parquet rows into ``MorphologyGraph``."""

    if root_id is not None and "neuron" in rows.columns:
        frame = rows[rows["neuron"].astype("int64") == int(root_id)].copy()
    else:
        frame = rows.copy()
    if frame.empty:
        raise ValueError(f"No skeleton rows found for root_id={root_id}")
    required = {"node_id", "parent_id", "radius", "x", "y", "z"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Skeleton rows are missing required columns: {missing}")

    if root_id is None:
        if "neuron" not in frame.columns:
            raise ValueError("root_id is required when skeleton rows do not include a neuron column")
        root_values = frame["neuron"].dropna().astype("int64").unique().tolist()
        if len(root_values) != 1:
            raise ValueError(f"Expected rows for one neuron, found {len(root_values)}")
        inferred_root_id = int(root_values[0])
    else:
        inferred_root_id = int(root_id)

    frame = frame.sort_values("node_id").reset_index(drop=True)
    old_node_ids = frame["node_id"].astype("int64").tolist()
    old_to_new = {int(old): idx for idx, old in enumerate(old_node_ids)}
    radius = np.maximum(frame["radius"].to_numpy(dtype=np.float64) * float(radius_scale_um), float(min_radius_um))
    nodes = pd.DataFrame(
        {
            "node_id": np.arange(len(frame), dtype=np.int64),
            "source_node_id": np.asarray(old_node_ids, dtype=np.int64),
            "x_um": frame["x"].to_numpy(dtype=np.float64) * float(coordinate_scale_um),
            "y_um": frame["y"].to_numpy(dtype=np.float64) * float(coordinate_scale_um),
            "z_um": frame["z"].to_numpy(dtype=np.float64) * float(coordinate_scale_um),
            "radius_um": radius,
            "is_soma": False,
            "pre_synapse_count": 0.0,
            "post_synapse_count": 0.0,
            "partner_hint": 0.0,
        }
    )

    edge_set: set[tuple[int, int]] = set()
    root_parent_candidates: list[int] = []
    for index, row in frame.iterrows():
        parent = row["parent_id"]
        if pd.isna(parent) or int(parent) < 0:
            root_parent_candidates.append(int(index))
            continue
        left = int(index)
        right = old_to_new.get(int(parent))
        if right is None or right == left:
            continue
        edge_set.add(tuple(sorted((left, right))))

    soma_node = 0
    if soma_node_id is not None and int(soma_node_id) in old_to_new:
        soma_node = int(old_to_new[int(soma_node_id)])
    elif soma_node_id is not None and 0 <= int(soma_node_id) < len(nodes):
        soma_node = int(soma_node_id)
    elif soma_xyz_um is not None:
        soma_xyz = np.asarray(soma_xyz_um, dtype=np.float64)
        if soma_xyz.shape == (3,) and np.isfinite(soma_xyz).all():
            coords = nodes[["x_um", "y_um", "z_um"]].to_numpy(dtype=np.float64)
            soma_node = int(np.argmin(np.linalg.norm(coords - soma_xyz[None, :], axis=1)))
    elif root_parent_candidates:
        soma_node = int(root_parent_candidates[0])
    elif len(nodes):
        soma_node = int(nodes["radius_um"].astype(float).idxmax())
    nodes.loc[soma_node, "is_soma"] = True

    metadata = {
        "format": "flywire_skeleton_parquet",
        "source": str(source_path) if source_path is not None else "",
    }
    if soma_node_id is not None:
        metadata["soma_source_node_id"] = int(soma_node_id)
    graph = MorphologyGraph(
        root_id=inferred_root_id,
        nodes=nodes,
        edges=tuple(sorted(edge_set)),
        soma_node=soma_node,
        metadata=metadata,
    )
    return graph.largest_component() if keep_largest_component else graph


def load_morphology_graphs_from_parquet(
    root_ids: Sequence[int],
    skeleton_parquet: Path | None = None,
    annotation_path: Path | None = None,
    *,
    coordinate_scale_um: float = 0.001,
    radius_scale_um: float = 0.001,
    min_radius_um: float = 0.05,
    keep_largest_component: bool = True,
) -> dict[int, MorphologyGraph]:
    """Load multiple morphology graphs from the whole-brain skeleton parquet."""

    unique_root_ids = tuple(sorted({int(root_id) for root_id in root_ids}))
    if not unique_root_ids:
        return {}
    skeleton_path = resolve_skeleton_parquet(skeleton_parquet)
    rows = load_skeleton_parquet_rows(unique_root_ids, skeleton_path)
    available = {int(value) for value in rows["neuron"].dropna().astype("int64").unique().tolist()}
    missing = sorted(set(unique_root_ids).difference(available))
    if missing:
        raise FileNotFoundError(f"Skeleton parquet has no rows for root ids: {missing[:10]}")
    soma_node_ids = skeleton_parquet_soma_node_ids(unique_root_ids, skeleton_path)
    soma_positions = load_annotation_soma_positions(
        unique_root_ids,
        annotation_path,
        coordinate_scale_um=coordinate_scale_um,
    )
    graphs: dict[int, MorphologyGraph] = {}
    for root_id in unique_root_ids:
        graphs[int(root_id)] = morphology_graph_from_skeleton_rows(
            rows,
            root_id=int(root_id),
            soma_node_id=soma_node_ids.get(int(root_id)),
            soma_xyz_um=soma_positions.get(int(root_id)),
            coordinate_scale_um=coordinate_scale_um,
            radius_scale_um=radius_scale_um,
            min_radius_um=min_radius_um,
            source_path=skeleton_path,
            keep_largest_component=keep_largest_component,
        )
    return graphs


def load_morphology_graph_from_parquet(root_id: int, **kwargs: object) -> MorphologyGraph:
    """Load one morphology graph from the whole-brain skeleton parquet."""

    return load_morphology_graphs_from_parquet((int(root_id),), **kwargs)[int(root_id)]


def path_distances_to_soma(graph: MorphologyGraph) -> np.ndarray:
    """Compute cable path distance from soma/root with Dijkstra."""

    n_nodes = len(graph.nodes)
    adjacency = [[] for _ in range(n_nodes)]
    coords = graph.coordinates()
    for left, right in graph.edges:
        distance = float(np.linalg.norm(coords[int(left)] - coords[int(right)]))
        adjacency[int(left)].append((int(right), distance))
        adjacency[int(right)].append((int(left), distance))

    distances = np.full(n_nodes, np.inf, dtype=np.float64)
    start = int(graph.soma_node)
    distances[start] = 0.0
    heap: list[tuple[float, int]] = [(0.0, start)]
    while heap:
        distance, node = heapq.heappop(heap)
        if distance > distances[node]:
            continue
        for neighbor, weight in adjacency[node]:
            candidate = distance + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                heapq.heappush(heap, (candidate, neighbor))
    return distances


def electrotonic_distances_to_soma(
    graph: MorphologyGraph,
    *,
    min_radius_um: float = 0.05,
    radius_power: float = 0.5,
) -> np.ndarray:
    """Compute a radius-aware path-distance proxy from soma/root.

    This is a lightweight electrotonic-distance surrogate.  Edge cost is
    Euclidean cable length divided by ``mean_radius_um ** radius_power``.  It
    does not replace a compartmental solver, but it captures the first-order
    fact that thin distal branches are more attenuating than thick cable of
    the same geometric length.
    """

    n_nodes = len(graph.nodes)
    coords = graph.coordinates()
    radii = np.maximum(graph.nodes["radius_um"].to_numpy(dtype=np.float64), float(min_radius_um))
    adjacency = [[] for _ in range(n_nodes)]
    for left, right in graph.edges:
        left_i = int(left)
        right_i = int(right)
        length = float(np.linalg.norm(coords[left_i] - coords[right_i]))
        mean_radius = max(float((radii[left_i] + radii[right_i]) / 2.0), float(min_radius_um))
        weight = length / (mean_radius**float(radius_power))
        adjacency[left_i].append((right_i, weight))
        adjacency[right_i].append((left_i, weight))

    distances = np.full(n_nodes, np.inf, dtype=np.float64)
    start = int(graph.soma_node)
    distances[start] = 0.0
    heap: list[tuple[float, int]] = [(0.0, start)]
    while heap:
        distance, node = heapq.heappop(heap)
        if distance > distances[node]:
            continue
        for neighbor, weight in adjacency[node]:
            candidate = distance + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                heapq.heappush(heap, (candidate, neighbor))
    return distances


def branch_orders_from_soma(graph: MorphologyGraph) -> np.ndarray:
    """Compute branch order as the number of branch points crossed from soma."""

    adjacency = graph.adjacency()
    orders = np.full(len(graph.nodes), -1, dtype=np.int64)
    root = int(graph.soma_node)
    orders[root] = 0
    queue = [root]
    parent = {root: -1}
    for node in queue:
        increment = 1 if parent[node] >= 0 and len(adjacency[node]) > 2 else 0
        for neighbor in adjacency[node]:
            if neighbor in parent:
                continue
            parent[neighbor] = node
            orders[neighbor] = orders[node] + increment
            queue.append(neighbor)
    return orders


def node_metrics_for_graph(
    graph: MorphologyGraph,
    *,
    config: PassiveSurrogateConfig | None = None,
) -> MorphologyNodeMetrics:
    """Compute reusable per-node morphology metrics once for downstream steps."""

    distances = path_distances_to_soma(graph)
    electrotonic = electrotonic_distances_to_soma(graph)
    branch_orders = branch_orders_from_soma(graph)
    degrees = np.asarray([len(values) for values in graph.adjacency()], dtype=np.int64)
    probabilities = _node_sampling_probabilities(graph)
    response_arrays = passive_response_arrays(distances, branch_orders, config)
    return MorphologyNodeMetrics(
        path_distances_um=distances,
        electrotonic_distances_proxy=electrotonic,
        branch_orders=branch_orders,
        degrees=degrees,
        cable_sampling_probabilities=probabilities,
        attenuation=response_arrays["attenuation"],
        delay_ms=response_arrays["delay_ms"],
        soma_peak_delta_mv=response_arrays["soma_peak_delta_mv"],
        time_to_peak_ms=response_arrays["time_to_peak_ms"],
    )


def morphology_feature_summary(
    graph: MorphologyGraph,
    *,
    metrics: MorphologyNodeMetrics | None = None,
) -> dict[str, object]:
    metrics = metrics or node_metrics_for_graph(graph)
    distances = metrics.path_distances_um
    finite = distances[np.isfinite(distances)]
    electrotonic = metrics.electrotonic_distances_proxy
    finite_electrotonic = electrotonic[np.isfinite(electrotonic)]
    branch_orders = metrics.branch_orders
    degrees = metrics.degrees
    edge_lengths = graph.edge_lengths()
    post_counts = graph.nodes["post_synapse_count"].to_numpy(dtype=np.float64)
    pre_counts = graph.nodes["pre_synapse_count"].to_numpy(dtype=np.float64)
    radius = graph.nodes["radius_um"].to_numpy(dtype=np.float64)
    return {
        "root_id": int(graph.root_id),
        "n_nodes": int(len(graph.nodes)),
        "n_edges": int(len(graph.edges)),
        "soma_node": int(graph.soma_node),
        "total_cable_length_um": float(edge_lengths.sum()) if edge_lengths.size else 0.0,
        "max_path_distance_um": float(finite.max()) if finite.size else 0.0,
        "mean_path_distance_um": float(finite.mean()) if finite.size else 0.0,
        "median_path_distance_um": float(np.median(finite)) if finite.size else 0.0,
        "max_electrotonic_distance_proxy": float(finite_electrotonic.max()) if finite_electrotonic.size else 0.0,
        "mean_electrotonic_distance_proxy": float(finite_electrotonic.mean()) if finite_electrotonic.size else 0.0,
        "max_branch_order": int(branch_orders.max()) if branch_orders.size else 0,
        "n_branch_points": int(np.sum(degrees > 2)),
        "n_terminal_points": int(np.sum(degrees == 1)),
        "mean_radius_um": float(radius.mean()) if radius.size else 0.0,
        "median_radius_um": float(np.median(radius)) if radius.size else 0.0,
        "total_pre_synapses": float(pre_counts.sum()),
        "total_post_synapses": float(post_counts.sum()),
        "source": str(graph.metadata.get("source", "")),
    }


def passive_response_at_location(
    path_distance_um: float,
    branch_order: int,
    config: PassiveSurrogateConfig | None = None,
) -> dict[str, float]:
    cfg = config or PassiveSurrogateConfig()
    length_constant = max(float(cfg.length_constant_um), 1e-9)
    velocity = max(float(cfg.conduction_velocity_um_per_ms), 1e-9)
    attenuation = math.exp(-float(path_distance_um) / length_constant)
    attenuation = attenuation / (1.0 + max(int(branch_order), 0) * float(cfg.branch_order_penalty))
    delay_ms = float(cfg.base_delay_ms) + float(path_distance_um) / velocity
    peak_mv = float(cfg.input_amplitude_mv) * attenuation
    return {
        "path_distance_um": float(path_distance_um),
        "branch_order": int(branch_order),
        "attenuation": float(attenuation),
        "delay_ms": float(delay_ms),
        "soma_peak_delta_mv": float(peak_mv),
        "time_to_peak_ms": float(delay_ms + cfg.response_tau_ms),
    }


def passive_response_arrays(
    path_distance_um: np.ndarray,
    branch_order: np.ndarray,
    config: PassiveSurrogateConfig | None = None,
) -> dict[str, np.ndarray]:
    """Vectorized passive surrogate for many morphology locations."""

    cfg = config or PassiveSurrogateConfig()
    path_distance = np.asarray(path_distance_um, dtype=np.float64)
    order = np.maximum(np.asarray(branch_order, dtype=np.float64), 0.0)
    length_constant = max(float(cfg.length_constant_um), 1e-9)
    velocity = max(float(cfg.conduction_velocity_um_per_ms), 1e-9)
    attenuation = np.exp(-path_distance / length_constant)
    attenuation = attenuation / (1.0 + order * float(cfg.branch_order_penalty))
    delay_ms = float(cfg.base_delay_ms) + path_distance / velocity
    peak_mv = float(cfg.input_amplitude_mv) * attenuation
    return {
        "attenuation": attenuation.astype(np.float64),
        "delay_ms": delay_ms.astype(np.float64),
        "soma_peak_delta_mv": peak_mv.astype(np.float64),
        "time_to_peak_ms": (delay_ms + float(cfg.response_tau_ms)).astype(np.float64),
    }


def passive_voltage_trace(
    path_distance_um: float,
    branch_order: int,
    config: PassiveSurrogateConfig | None = None,
) -> pd.DataFrame:
    cfg = config or PassiveSurrogateConfig()
    metrics = passive_response_at_location(path_distance_um, branch_order, cfg)
    t = np.arange(0.0, float(cfg.tstop_ms) + float(cfg.dt_ms), float(cfg.dt_ms))
    shifted = t - float(metrics["delay_ms"])
    alpha = np.zeros_like(t)
    active = shifted >= 0
    tau = max(float(cfg.response_tau_ms), 1e-9)
    x = shifted[active] / tau
    alpha[active] = x * np.exp(1.0 - x)
    v = float(cfg.baseline_mv) + float(metrics["soma_peak_delta_mv"]) * alpha
    return pd.DataFrame({"time_ms": t, "soma_voltage_mv": v})


def location_response_table(
    graph: MorphologyGraph,
    *,
    quantiles: Sequence[float] = (0.2, 0.5, 0.85),
    config: PassiveSurrogateConfig | None = None,
    metrics: MorphologyNodeMetrics | None = None,
) -> pd.DataFrame:
    metrics = metrics or node_metrics_for_graph(graph, config=config)
    distances = metrics.path_distances_um
    branch_orders = metrics.branch_orders
    finite_mask = np.isfinite(distances)
    finite_distances = distances[finite_mask]
    if finite_distances.size == 0:
        return pd.DataFrame()
    labels = ["proximal", "distal"] if len(quantiles) == 2 else ["proximal", "medial", "distal"]
    rows: list[dict[str, object]] = []
    for index, quantile in enumerate(quantiles):
        target = float(np.quantile(finite_distances, float(quantile)))
        node = int(np.nanargmin(np.where(finite_mask, np.abs(distances - target), np.inf)))
        metrics = passive_response_at_location(float(distances[node]), int(branch_orders[node]), config)
        rows.append(
            {
                "root_id": int(graph.root_id),
                "location_label": labels[index] if index < len(labels) else f"q{quantile:g}",
                "location_quantile": float(quantile),
                "node_id": int(node),
                **metrics,
            }
        )
    return pd.DataFrame.from_records(rows)


def synapse_location_summary(
    graph: MorphologyGraph,
    *,
    config: PassiveSurrogateConfig | None = None,
    distal_quantile: float = 0.75,
    metrics: MorphologyNodeMetrics | None = None,
) -> dict[str, object]:
    metrics = metrics or node_metrics_for_graph(graph, config=config)
    distances = metrics.path_distances_um
    branch_orders = metrics.branch_orders
    finite_mask = np.isfinite(distances)
    post_counts = graph.nodes["post_synapse_count"].to_numpy(dtype=np.float64)
    post_counts = np.where(finite_mask, np.maximum(post_counts, 0.0), 0.0)
    total_post = float(post_counts.sum())
    attenuation = metrics.attenuation
    finite_distances = distances[finite_mask]
    distal_threshold = float(np.quantile(finite_distances, distal_quantile)) if finite_distances.size else 0.0
    random_mean_distance = float(finite_distances.mean()) if finite_distances.size else 0.0
    random_mean_attenuation = float(np.nanmean(attenuation[finite_mask])) if finite_distances.size else 0.0
    if total_post > 0:
        mean_distance = float(np.sum(post_counts * distances) / total_post)
        mean_branch_order = float(np.sum(post_counts * np.maximum(branch_orders, 0)) / total_post)
        real_mean_attenuation = float(np.nansum(post_counts * attenuation) / total_post)
        distal_fraction = float(post_counts[distances >= distal_threshold].sum() / total_post)
    else:
        mean_distance = 0.0
        mean_branch_order = 0.0
        real_mean_attenuation = 0.0
        distal_fraction = 0.0
    return {
        "root_id": int(graph.root_id),
        "total_post_synapses": total_post,
        "real_mean_postsynaptic_path_distance_um": mean_distance,
        "real_mean_postsynaptic_branch_order": mean_branch_order,
        "distal_path_threshold_um": distal_threshold,
        "real_distal_postsynaptic_fraction": distal_fraction,
        "real_mean_attenuation": real_mean_attenuation,
        "matched_random_mean_path_distance_um": random_mean_distance,
        "matched_random_mean_attenuation": random_mean_attenuation,
        "real_minus_random_attenuation": real_mean_attenuation - random_mean_attenuation,
    }


def sectionize_morphology_graph(
    graph: MorphologyGraph,
    *,
    metrics: MorphologyNodeMetrics | None = None,
) -> pd.DataFrame:
    """Compress a morphology tree into soma-to-branch cable sections.

    Sections start at soma or branch points and end at the next branch point or
    terminal.  This table is directly useful as a future NEURON section plan
    while keeping today's implementation backend-independent.
    """

    if len(graph.nodes) == 0:
        return pd.DataFrame()
    adjacency = graph.adjacency()
    degrees = np.asarray([len(values) for values in adjacency], dtype=np.int64)
    parent = _bfs_tree_parents(graph)
    children: list[list[int]] = [[] for _ in range(len(graph.nodes))]
    for node, parent_node in enumerate(parent):
        if int(parent_node) >= 0:
            children[int(parent_node)].append(int(node))
    critical = {int(graph.soma_node)}
    critical.update(int(idx) for idx, degree in enumerate(degrees) if int(degree) != 2)

    coords = graph.coordinates()
    radii = graph.nodes["radius_um"].to_numpy(dtype=np.float64)
    pre_counts = graph.nodes["pre_synapse_count"].to_numpy(dtype=np.float64)
    post_counts = graph.nodes["post_synapse_count"].to_numpy(dtype=np.float64)
    branch_orders = metrics.branch_orders if metrics is not None else branch_orders_from_soma(graph)
    path_distances = metrics.path_distances_um if metrics is not None else path_distances_to_soma(graph)

    rows: list[dict[str, object]] = []
    incoming_section_by_end: dict[int, int] = {}
    section_id = 0
    queue = deque([int(graph.soma_node)])
    while queue:
        start = queue.popleft()
        for child in children[start]:
            node_path = [start, child]
            current = child
            while current not in critical and children[current]:
                current = children[current][0]
                node_path.append(current)
            end = int(node_path[-1])
            lengths = [
                float(np.linalg.norm(coords[int(left)] - coords[int(right)]))
                for left, right in zip(node_path[:-1], node_path[1:])
            ]
            section_nodes = np.asarray(node_path, dtype=np.int64)
            parent_section_id = incoming_section_by_end.get(start, -1)
            rows.append(
                {
                    "section_id": int(section_id),
                    "parent_section_id": int(parent_section_id),
                    "root_id": int(graph.root_id),
                    "start_node": int(start),
                    "end_node": int(end),
                    "start_source_node_id": int(graph.nodes.iloc[start].get("source_node_id", start)),
                    "end_source_node_id": int(graph.nodes.iloc[end].get("source_node_id", end)),
                    "n_nodes": int(len(node_path)),
                    "length_um": float(sum(lengths)),
                    "mean_radius_um": float(np.mean(radii[section_nodes])) if len(section_nodes) else 0.0,
                    "branch_order": int(branch_orders[end]) if len(branch_orders) else 0,
                    "end_path_distance_um": float(path_distances[end]) if np.isfinite(path_distances[end]) else np.nan,
                    "pre_synapse_count": float(pre_counts[section_nodes].sum()),
                    "post_synapse_count": float(post_counts[section_nodes].sum()),
                    "node_path": ",".join(str(int(node)) for node in node_path),
                    "source_node_path": ",".join(
                        str(int(graph.nodes.iloc[int(node)].get("source_node_id", int(node)))) for node in node_path
                    ),
                }
            )
            incoming_section_by_end[end] = int(section_id)
            section_id += 1
            if end in critical:
                queue.append(end)
    return pd.DataFrame.from_records(rows)


def _nearest_node_indices(
    node_coords: np.ndarray,
    query_coords: np.ndarray,
    *,
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    if query_coords.size == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
    try:
        from scipy.spatial import cKDTree

        distances, indices = cKDTree(node_coords).query(query_coords, k=1)
        return np.asarray(indices, dtype=np.int64), np.asarray(distances, dtype=np.float64)
    except ImportError:
        indices: list[np.ndarray] = []
        distances: list[np.ndarray] = []
        for start in range(0, len(query_coords), int(chunk_size)):
            chunk = query_coords[start : start + int(chunk_size)]
            delta = chunk[:, None, :] - node_coords[None, :, :]
            dist = np.linalg.norm(delta, axis=2)
            idx = np.argmin(dist, axis=1)
            indices.append(idx.astype(np.int64))
            distances.append(dist[np.arange(len(chunk)), idx].astype(np.float64))
        return np.concatenate(indices), np.concatenate(distances)


def map_synapses_to_morphology(
    graph: MorphologyGraph,
    synapses: pd.DataFrame,
    *,
    role: str = "post",
    coordinate_scale_um: float = 0.001,
    nearest_node_max_distance_um: float | None = None,
    config: PassiveSurrogateConfig | None = None,
    metrics: MorphologyNodeMetrics | None = None,
) -> pd.DataFrame:
    """Map real synapse coordinates to nearest morphology nodes."""

    if role not in {"post", "pre"}:
        raise ValueError("role must be 'post' or 'pre'")
    if synapses.empty:
        return synapses.copy()
    root_col = f"{role}_pt_root_id"
    position_cols = [f"{role}_pt_position_x", f"{role}_pt_position_y", f"{role}_pt_position_z"]
    missing = [column for column in [root_col, *position_cols] if column not in synapses.columns]
    if missing:
        raise ValueError(f"Synapse table is missing required columns: {missing}")
    subset = synapses[synapses[root_col].astype("int64") == int(graph.root_id)].copy()
    if subset.empty:
        return subset

    query = subset[position_cols].to_numpy(dtype=np.float64) * float(coordinate_scale_um)
    nearest_indices, nearest_distances = _nearest_node_indices(graph.coordinates(), query)
    metrics = metrics or node_metrics_for_graph(graph, config=config)
    path_distances = metrics.path_distances_um
    electrotonic = metrics.electrotonic_distances_proxy
    branch_orders = metrics.branch_orders
    source_ids = graph.nodes.get("source_node_id", pd.Series(graph.nodes.index, index=graph.nodes.index))
    mapped = subset.reset_index(drop=True)
    mapped[f"{role}_nearest_node"] = nearest_indices.astype(np.int64)
    mapped[f"{role}_nearest_source_node_id"] = source_ids.iloc[nearest_indices].to_numpy(dtype=np.int64)
    mapped[f"{role}_nearest_node_distance_um"] = nearest_distances.astype(np.float64)
    mapped[f"{role}_path_distance_um"] = path_distances[nearest_indices].astype(np.float64)
    mapped[f"{role}_electrotonic_distance_proxy"] = electrotonic[nearest_indices].astype(np.float64)
    mapped[f"{role}_branch_order"] = branch_orders[nearest_indices].astype(np.int64)
    mapped[f"{role}_attenuation"] = metrics.attenuation[nearest_indices].astype(np.float64)
    mapped[f"{role}_delay_ms"] = metrics.delay_ms[nearest_indices].astype(np.float64)
    mapped[f"{role}_soma_peak_delta_mv"] = metrics.soma_peak_delta_mv[nearest_indices].astype(np.float64)
    if nearest_node_max_distance_um is not None:
        mapped = mapped[mapped[f"{role}_nearest_node_distance_um"] <= float(nearest_node_max_distance_um)].copy()
    return mapped


def load_synapses_for_root_ids(
    root_ids: Sequence[int],
    synapses_feather: Path | None = None,
    *,
    role: str = "post",
    max_synapses_per_root: int | None = None,
) -> pd.DataFrame:
    """Load FlyWire synapse rows where selected roots are pre or post."""

    if role not in {"post", "pre"}:
        raise ValueError("role must be 'post' or 'pre'")
    unique_root_ids = sorted({int(root_id) for root_id in root_ids})
    if not unique_root_ids:
        return pd.DataFrame()
    path = resolve_synapses_feather(synapses_feather)
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.ipc as ipc

    root_col = f"{role}_pt_root_id"
    value_set = pa.array(unique_root_ids, type=pa.int64())
    tables = []
    with ipc.open_file(str(path)) as reader:
        for batch_index in range(reader.num_record_batches):
            batch = reader.get_batch(batch_index)
            mask = pc.is_in(batch.column(root_col), value_set=value_set)
            if pc.sum(mask).as_py() == 0:
                continue
            tables.append(pa.Table.from_batches([batch.filter(mask)]))
    if not tables:
        return pd.DataFrame()
    frame = pa.concat_tables(tables, promote_options="default").to_pandas()
    if max_synapses_per_root is not None:
        limit = max(0, int(max_synapses_per_root))
        frame = (
            frame.sort_values(["connection_score", "cleft_score"], ascending=[False, False])
            .groupby(root_col, group_keys=False)
            .head(limit)
            .reset_index(drop=True)
        )
    return frame


def aggregate_morphology_weighted_edges(mapped_synapses: pd.DataFrame, *, role: str = "post") -> pd.DataFrame:
    """Aggregate mapped synapses into a morphology-weighted edge table."""

    if role not in {"post", "pre"}:
        raise ValueError("role must be 'post' or 'pre'")
    if mapped_synapses.empty:
        return pd.DataFrame()
    required = [
        "pre_pt_root_id",
        "post_pt_root_id",
        f"{role}_path_distance_um",
        f"{role}_branch_order",
        f"{role}_attenuation",
        f"{role}_delay_ms",
    ]
    missing = [column for column in required if column not in mapped_synapses.columns]
    if missing:
        raise ValueError(f"Mapped synapse table is missing required columns: {missing}")
    group_cols = ["pre_pt_root_id", "post_pt_root_id"]
    rows: list[dict[str, object]] = []
    nt_cols = [column for column in ["ach", "gaba", "glut", "oct", "ser", "da"] if column in mapped_synapses.columns]
    for (pre_root, post_root), subset in mapped_synapses.groupby(group_cols):
        row: dict[str, object] = {
            "pre_root_id": int(pre_root),
            "post_root_id": int(post_root),
            "raw_synapse_count": int(len(subset)),
            "morphology_weighted_synapse_count": float(subset[f"{role}_attenuation"].sum()),
            f"mean_{role}_path_distance_um": float(subset[f"{role}_path_distance_um"].mean()),
            f"mean_{role}_branch_order": float(subset[f"{role}_branch_order"].mean()),
            f"mean_{role}_attenuation": float(subset[f"{role}_attenuation"].mean()),
            f"mean_{role}_delay_ms": float(subset[f"{role}_delay_ms"].mean()),
        }
        if "neuropil" in subset.columns and not subset["neuropil"].dropna().empty:
            row["dominant_neuropil"] = str(subset["neuropil"].dropna().mode().iloc[0])
        for column in nt_cols:
            row[f"mean_{column}_probability"] = float(subset[column].mean())
        if nt_cols:
            means = {column: float(subset[column].mean()) for column in nt_cols}
            dominant_nt = max(means, key=means.get)
            row["dominant_nt"] = dominant_nt
            row["dominant_nt_mean_probability"] = means[dominant_nt]
        rows.append(row)
    edges = pd.DataFrame.from_records(rows)
    if edges.empty:
        return edges
    return edges.sort_values(
        ["post_root_id", "morphology_weighted_synapse_count", "raw_synapse_count"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def _node_sampling_probabilities(graph: MorphologyGraph) -> np.ndarray:
    """Approximate cable-length-weighted node probabilities for random controls."""

    n_nodes = len(graph.nodes)
    weights = np.zeros(n_nodes, dtype=np.float64)
    if n_nodes == 0:
        return weights
    coords = graph.coordinates()
    for left, right in graph.edges:
        left_i = int(left)
        right_i = int(right)
        length = float(np.linalg.norm(coords[left_i] - coords[right_i]))
        if np.isfinite(length) and length > 0:
            weights[left_i] += length / 2.0
            weights[right_i] += length / 2.0
    if float(weights.sum()) <= 0:
        weights[:] = 1.0 / float(n_nodes)
    else:
        weights = weights / float(weights.sum())
    return weights


def random_synapse_location_controls(
    graph: MorphologyGraph,
    mapped_synapses: pd.DataFrame,
    *,
    role: str = "post",
    repeats: int = 100,
    seed: int = 0,
    config: PassiveSurrogateConfig | None = None,
    metrics: MorphologyNodeMetrics | None = None,
) -> pd.DataFrame:
    """Compare real mapped synapse positions to cable-length-matched random nodes."""

    if role not in {"post", "pre"}:
        raise ValueError("role must be 'post' or 'pre'")
    if repeats <= 0 or mapped_synapses.empty:
        return pd.DataFrame()
    subset = mapped_synapses[mapped_synapses[f"{role}_pt_root_id"].astype("int64") == int(graph.root_id)].copy()
    if subset.empty:
        return pd.DataFrame()

    n_synapses = int(len(subset))
    metrics = metrics or node_metrics_for_graph(graph, config=config)
    path_distances = metrics.path_distances_um
    branch_orders = metrics.branch_orders
    real_mean_path = float(subset[f"{role}_path_distance_um"].mean())
    real_mean_branch = float(subset[f"{role}_branch_order"].mean())
    real_mean_attenuation = float(subset[f"{role}_attenuation"].mean())
    probabilities = metrics.cable_sampling_probabilities
    finite_nodes = np.where(np.isfinite(path_distances))[0]
    if len(finite_nodes) == 0:
        return pd.DataFrame()
    probabilities = probabilities[finite_nodes]
    probabilities = probabilities / float(probabilities.sum()) if float(probabilities.sum()) > 0 else None

    rng = np.random.default_rng(int(seed) + int(graph.root_id) % 1_000_000)
    rows: list[dict[str, object]] = []
    for repeat in range(int(repeats)):
        sampled = rng.choice(finite_nodes, size=n_synapses, replace=True, p=probabilities)
        random_mean_path = float(np.mean(path_distances[sampled]))
        random_mean_branch = float(np.mean(branch_orders[sampled]))
        random_mean_attenuation = float(np.mean(metrics.attenuation[sampled]))
        rows.append(
            {
                "root_id": int(graph.root_id),
                "repeat": int(repeat),
                "n_synapses": n_synapses,
                "real_mean_path_distance_um": real_mean_path,
                "random_mean_path_distance_um": random_mean_path,
                "real_minus_random_path_distance_um": real_mean_path - random_mean_path,
                "real_mean_branch_order": real_mean_branch,
                "random_mean_branch_order": random_mean_branch,
                "real_minus_random_branch_order": real_mean_branch - random_mean_branch,
                "real_mean_attenuation": real_mean_attenuation,
                "random_mean_attenuation": random_mean_attenuation,
                "real_minus_random_attenuation": real_mean_attenuation - random_mean_attenuation,
            }
        )
    return pd.DataFrame.from_records(rows)


def summarize_random_synapse_controls(control_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize repeat-level random controls per postsynaptic root."""

    if control_rows.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for root_id, subset in control_rows.groupby("root_id"):
        rows.append(
            {
                "root_id": int(root_id),
                "repeats": int(len(subset)),
                "n_synapses": int(subset["n_synapses"].iloc[0]),
                "real_mean_path_distance_um": float(subset["real_mean_path_distance_um"].iloc[0]),
                "random_mean_path_distance_um": float(subset["random_mean_path_distance_um"].mean()),
                "real_minus_random_path_distance_um": float(subset["real_minus_random_path_distance_um"].mean()),
                "real_mean_branch_order": float(subset["real_mean_branch_order"].iloc[0]),
                "random_mean_branch_order": float(subset["random_mean_branch_order"].mean()),
                "real_minus_random_branch_order": float(subset["real_minus_random_branch_order"].mean()),
                "real_mean_attenuation": float(subset["real_mean_attenuation"].iloc[0]),
                "random_mean_attenuation": float(subset["random_mean_attenuation"].mean()),
                "real_minus_random_attenuation": float(subset["real_minus_random_attenuation"].mean()),
                "random_attenuation_std": float(subset["random_mean_attenuation"].std(ddof=0)),
                "attenuation_z_score": _safe_z_score(
                    float(subset["real_mean_attenuation"].iloc[0]),
                    float(subset["random_mean_attenuation"].mean()),
                    float(subset["random_mean_attenuation"].std(ddof=0)),
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def edge_synapse_location_random_controls(
    graph: MorphologyGraph,
    mapped_synapses: pd.DataFrame,
    *,
    role: str = "post",
    repeats: int = 100,
    seed: int = 0,
    min_synapses: int = 3,
    config: PassiveSurrogateConfig | None = None,
    metrics: MorphologyNodeMetrics | None = None,
) -> pd.DataFrame:
    """Run cable-length random controls for each pre->post edge on one morphology."""

    if role != "post":
        raise ValueError("edge controls currently support role='post'")
    if repeats <= 0 or mapped_synapses.empty:
        return pd.DataFrame()
    subset = mapped_synapses[mapped_synapses["post_pt_root_id"].astype("int64") == int(graph.root_id)].copy()
    if subset.empty:
        return pd.DataFrame()

    metrics = metrics or node_metrics_for_graph(graph, config=config)
    path_distances = metrics.path_distances_um
    branch_orders = metrics.branch_orders
    finite_nodes = np.where(np.isfinite(path_distances))[0]
    if len(finite_nodes) == 0:
        return pd.DataFrame()
    probabilities = metrics.cable_sampling_probabilities[finite_nodes]
    probabilities = probabilities / float(probabilities.sum()) if float(probabilities.sum()) > 0 else None
    rng = np.random.default_rng(int(seed) + int(graph.root_id) % 1_000_000 + 17)

    rows: list[dict[str, object]] = []
    for pre_root, edge_subset in subset.groupby("pre_pt_root_id"):
        n_synapses = int(len(edge_subset))
        if n_synapses < int(min_synapses):
            continue
        random_attenuation = np.empty(int(repeats), dtype=np.float64)
        random_path = np.empty(int(repeats), dtype=np.float64)
        random_branch = np.empty(int(repeats), dtype=np.float64)
        for repeat in range(int(repeats)):
            sampled = rng.choice(finite_nodes, size=n_synapses, replace=True, p=probabilities)
            random_attenuation[repeat] = float(np.mean(metrics.attenuation[sampled]))
            random_path[repeat] = float(np.mean(path_distances[sampled]))
            random_branch[repeat] = float(np.mean(branch_orders[sampled]))
        real_mean_attenuation = float(edge_subset["post_attenuation"].mean())
        random_mean_attenuation = float(np.mean(random_attenuation))
        random_std_attenuation = float(np.std(random_attenuation, ddof=0))
        rows.append(
            {
                "pre_root_id": int(pre_root),
                "post_root_id": int(graph.root_id),
                "n_synapses": n_synapses,
                "repeats": int(repeats),
                "real_mean_path_distance_um": float(edge_subset["post_path_distance_um"].mean()),
                "random_mean_path_distance_um": float(np.mean(random_path)),
                "real_minus_random_path_distance_um": float(
                    edge_subset["post_path_distance_um"].mean() - np.mean(random_path)
                ),
                "real_mean_branch_order": float(edge_subset["post_branch_order"].mean()),
                "random_mean_branch_order": float(np.mean(random_branch)),
                "real_minus_random_branch_order": float(edge_subset["post_branch_order"].mean() - np.mean(random_branch)),
                "real_mean_attenuation": real_mean_attenuation,
                "random_mean_attenuation": random_mean_attenuation,
                "real_minus_random_attenuation": real_mean_attenuation - random_mean_attenuation,
                "random_attenuation_std": random_std_attenuation,
                "attenuation_z_score": _safe_z_score(
                    real_mean_attenuation,
                    random_mean_attenuation,
                    random_std_attenuation,
                ),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame.from_records(rows).sort_values(
        ["post_root_id", "attenuation_z_score"],
        ascending=[True, True],
    ).reset_index(drop=True)


def _safe_z_score(value: float, mean: float, std: float) -> float:
    if not np.isfinite(std) or std <= 1e-12:
        return 0.0
    return float((value - mean) / std)


def build_morphology_weighted_connectivity(
    config: MorphologyWeightedConnectivityConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[int, MorphologyGraph],
    dict[int, MorphologyNodeMetrics],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load skeletons and real synapses, then build morphology-weighted edges."""

    graphs = load_morphology_graphs_from_parquet(
        config.root_ids,
        skeleton_parquet=config.skeleton_parquet,
        annotation_path=config.annotation_path,
        coordinate_scale_um=config.coordinate_scale_um,
        radius_scale_um=config.radius_scale_um,
        min_radius_um=config.min_radius_um,
    )
    metrics_by_root = {int(root_id): node_metrics_for_graph(graph, config=config.response) for root_id, graph in graphs.items()}
    synapses = load_synapses_for_root_ids(
        config.root_ids,
        config.synapses_feather,
        role="post",
        max_synapses_per_root=config.max_synapses_per_root,
    )
    if synapses.empty:
        synapses_by_post: dict[int, pd.DataFrame] = {}
    else:
        synapses_by_post = {
            int(root_id): subset.reset_index(drop=True)
            for root_id, subset in synapses.groupby("post_pt_root_id", sort=False)
        }
    mapped_frames = [
        map_synapses_to_morphology(
            graph,
            synapses_by_post.get(int(graph.root_id), synapses.iloc[0:0]),
            role="post",
            coordinate_scale_um=config.coordinate_scale_um,
            nearest_node_max_distance_um=config.nearest_node_max_distance_um,
            config=config.response,
            metrics=metrics_by_root[int(graph.root_id)],
        )
        for graph in graphs.values()
    ]
    mapped = pd.concat(mapped_frames, ignore_index=True) if mapped_frames else pd.DataFrame()
    edges = aggregate_morphology_weighted_edges(mapped, role="post")
    control_frames = [
        random_synapse_location_controls(
            graph,
            mapped,
            role="post",
            repeats=config.random_control_repeats,
            seed=config.random_control_seed,
            config=config.response,
            metrics=metrics_by_root[int(graph.root_id)],
        )
        for graph in graphs.values()
        if config.random_control_repeats > 0
    ]
    controls = pd.concat(control_frames, ignore_index=True) if control_frames else pd.DataFrame()
    control_summary = summarize_random_synapse_controls(controls)
    edge_control_frames = [
        edge_synapse_location_random_controls(
            graph,
            mapped,
            role="post",
            repeats=config.random_control_repeats,
            seed=config.random_control_seed,
            min_synapses=config.edge_random_control_min_synapses,
            config=config.response,
            metrics=metrics_by_root[int(graph.root_id)],
        )
        for graph in graphs.values()
        if config.random_control_repeats > 0 and config.edge_random_controls
    ]
    edge_controls = pd.concat(edge_control_frames, ignore_index=True) if edge_control_frames else pd.DataFrame()
    return edges, mapped, graphs, metrics_by_root, controls, control_summary, edge_controls


def apply_morphology_weights_to_connectivity(
    connectivity_edges: pd.DataFrame,
    morphology_edges: pd.DataFrame,
    *,
    mode: str = "scale_existing",
    weight_column: str = "Excitatory x Connectivity",
) -> pd.DataFrame:
    """Merge morphology-weighted edge strength into an existing connectome table.

    ``mode='scale_existing'`` keeps the signed connectome weight and multiplies
    it by ``morphology_weighted_synapse_count / raw_synapse_count`` for edges
    that have mapped postsynaptic synapses.  Edges without morphology rows are
    left unchanged, which keeps the function forward-compatible with partial
    root-id runs.
    """

    if mode != "scale_existing":
        raise ValueError("Only mode='scale_existing' is currently supported")
    required_connectivity = {"Presynaptic_ID", "Postsynaptic_ID", weight_column}
    missing_connectivity = sorted(required_connectivity.difference(connectivity_edges.columns))
    if missing_connectivity:
        raise ValueError(f"Connectivity table is missing required columns: {missing_connectivity}")
    required_morphology = {
        "pre_root_id",
        "post_root_id",
        "raw_synapse_count",
        "morphology_weighted_synapse_count",
    }
    missing_morphology = sorted(required_morphology.difference(morphology_edges.columns))
    if missing_morphology:
        raise ValueError(f"Morphology edge table is missing required columns: {missing_morphology}")

    factors = morphology_edges[
        ["pre_root_id", "post_root_id", "raw_synapse_count", "morphology_weighted_synapse_count"]
    ].copy()
    raw = factors["raw_synapse_count"].astype(float).replace(0.0, np.nan)
    factors["morphology_weight_factor"] = (
        factors["morphology_weighted_synapse_count"].astype(float) / raw
    ).fillna(1.0)
    factors = factors.rename(columns={"pre_root_id": "Presynaptic_ID", "post_root_id": "Postsynaptic_ID"})

    adjusted = connectivity_edges.merge(
        factors[["Presynaptic_ID", "Postsynaptic_ID", "morphology_weight_factor"]],
        on=["Presynaptic_ID", "Postsynaptic_ID"],
        how="left",
    )
    adjusted["morphology_weight_factor"] = adjusted["morphology_weight_factor"].fillna(1.0)
    adjusted["raw_connectivity_weight"] = adjusted[weight_column].astype(float)
    adjusted[weight_column] = adjusted["raw_connectivity_weight"] * adjusted["morphology_weight_factor"].astype(float)
    return adjusted


def summarize_morphology_connectivity_adjustment(
    adjusted_connectivity: pd.DataFrame,
    *,
    weight_column: str = "Excitatory x Connectivity",
) -> pd.DataFrame:
    """Summarize the edges changed by morphology weighting."""

    required = {"Presynaptic_ID", "Postsynaptic_ID", weight_column, "raw_connectivity_weight", "morphology_weight_factor"}
    missing = sorted(required.difference(adjusted_connectivity.columns))
    if missing:
        raise ValueError(f"Adjusted connectivity table is missing required columns: {missing}")
    changed = adjusted_connectivity[adjusted_connectivity["morphology_weight_factor"].astype(float) != 1.0].copy()
    if changed.empty:
        return pd.DataFrame(
            columns=[
                "Presynaptic_ID",
                "Postsynaptic_ID",
                "raw_connectivity_weight",
                "adjusted_connectivity_weight",
                "morphology_weight_factor",
                "absolute_weight_delta",
                "relative_abs_weight_delta",
            ]
        )
    changed = changed.rename(columns={weight_column: "adjusted_connectivity_weight"})
    changed["absolute_weight_delta"] = (
        changed["adjusted_connectivity_weight"].astype(float) - changed["raw_connectivity_weight"].astype(float)
    )
    denominator = changed["raw_connectivity_weight"].astype(float).abs().replace(0.0, np.nan)
    changed["relative_abs_weight_delta"] = (changed["absolute_weight_delta"].astype(float).abs() / denominator).fillna(0.0)
    columns = [
        "Presynaptic_ID",
        "Postsynaptic_ID",
        "raw_connectivity_weight",
        "adjusted_connectivity_weight",
        "morphology_weight_factor",
        "absolute_weight_delta",
        "relative_abs_weight_delta",
    ]
    return changed[columns].sort_values("relative_abs_weight_delta", ascending=False).reset_index(drop=True)


def run_morphology_weighted_connectivity(
    config: MorphologyWeightedConnectivityConfig,
) -> dict[str, Path]:
    """Persist morphology-weighted edge tables and section summaries."""

    output_dir = config.output_dir
    table_dir = output_dir / "tables"
    metadata_dir = output_dir / "metadata"
    for directory in (table_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    edges, mapped, graphs, metrics_by_root, controls, control_summary, edge_controls = build_morphology_weighted_connectivity(config)
    features = pd.DataFrame.from_records(
        [morphology_feature_summary(graph, metrics=metrics_by_root[int(graph.root_id)]) for graph in graphs.values()]
    )
    sections = pd.concat(
        [sectionize_morphology_graph(graph, metrics=metrics_by_root[int(graph.root_id)]) for graph in graphs.values()],
        ignore_index=True,
    ) if graphs else pd.DataFrame()

    edge_path = table_dir / "morphology_weighted_edges.csv"
    feature_path = table_dir / "morphology_features.csv"
    section_path = table_dir / "morphology_sections.csv"
    edges.to_csv(edge_path, index=False)
    features.to_csv(feature_path, index=False)
    sections.to_csv(section_path, index=False)

    control_path: Path | None = None
    control_summary_path: Path | None = None
    if not controls.empty:
        control_path = table_dir / "synapse_location_random_controls.csv"
        control_summary_path = table_dir / "synapse_location_random_control_summary.csv"
        controls.to_csv(control_path, index=False)
        control_summary.to_csv(control_summary_path, index=False)

    edge_control_path: Path | None = None
    if not edge_controls.empty:
        edge_control_path = table_dir / "edge_synapse_location_random_controls.csv"
        edge_controls.to_csv(edge_control_path, index=False)

    mapped_path: Path | None = None
    if config.save_mapped_synapses:
        mapped_path = table_dir / "mapped_synapses.csv"
        mapped.to_csv(mapped_path, index=False)

    adjusted_path: Path | None = None
    adjustment_summary_path: Path | None = None
    if config.write_adjusted_connectivity:
        connectivity_path = resolve_connectivity_path(config.connectivity_path)
        connectivity = pd.read_parquet(connectivity_path)
        adjusted = apply_morphology_weights_to_connectivity(
            connectivity,
            edges,
            mode=config.adjusted_connectivity_mode,
        )
        adjusted_path = table_dir / "connectivity_morphology_adjusted.parquet"
        adjusted.to_parquet(adjusted_path, index=False)
        adjustment_summary = summarize_morphology_connectivity_adjustment(adjusted)
        adjustment_summary_path = table_dir / "connectivity_morphology_adjustment_summary.csv"
        adjustment_summary.to_csv(adjustment_summary_path, index=False)

    payload = asdict(config)
    payload["skeleton_parquet"] = str(resolve_skeleton_parquet(config.skeleton_parquet))
    payload["synapses_feather"] = str(resolve_synapses_feather(config.synapses_feather))
    if config.write_adjusted_connectivity or config.connectivity_path is not None:
        payload["connectivity_path"] = str(resolve_connectivity_path(config.connectivity_path))
    try:
        payload["annotation_path"] = str(resolve_annotation_path(config.annotation_path))
    except FileNotFoundError:
        payload["annotation_path"] = ""
    payload["output_dir"] = str(output_dir)
    payload["response"] = asdict(config.response)
    config_path = metadata_dir / "morphology_weighted_connectivity_config.json"
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = output_dir / "MORPHOLOGY_WEIGHTED_CONNECTIVITY_REPORT.md"
    report_path.write_text(
        render_morphology_weighted_connectivity_report(
            edges,
            features,
            sections,
            mapped,
            control_summary,
            edge_controls,
            config,
        ),
        encoding="utf-8",
    )

    result = {
        "edges_csv": edge_path,
        "features_csv": feature_path,
        "sections_csv": section_path,
        "config_json": config_path,
        "report_md": report_path,
    }
    if mapped_path is not None:
        result["mapped_synapses_csv"] = mapped_path
    if control_path is not None:
        result["random_controls_csv"] = control_path
    if control_summary_path is not None:
        result["random_control_summary_csv"] = control_summary_path
    if edge_control_path is not None:
        result["edge_random_controls_csv"] = edge_control_path
    if adjusted_path is not None:
        result["adjusted_connectivity_parquet"] = adjusted_path
    if adjustment_summary_path is not None:
        result["adjustment_summary_csv"] = adjustment_summary_path
    return result


def render_morphology_weighted_connectivity_report(
    edges: pd.DataFrame,
    features: pd.DataFrame,
    sections: pd.DataFrame,
    mapped_synapses: pd.DataFrame,
    control_summary: pd.DataFrame,
    edge_controls: pd.DataFrame,
    config: MorphologyWeightedConnectivityConfig,
) -> str:
    n_roots = int(len(config.root_ids))
    n_edges = int(len(edges))
    n_synapses = int(len(mapped_synapses))
    mean_attenuation = (
        float(mapped_synapses["post_attenuation"].mean())
        if not mapped_synapses.empty and "post_attenuation" in mapped_synapses.columns
        else 0.0
    )
    mean_sections = float(sections.groupby("root_id")["section_id"].count().mean()) if not sections.empty else 0.0
    mean_cable = float(features["total_cable_length_um"].mean()) if not features.empty else 0.0
    mean_real_minus_random_attenuation = (
        float(control_summary["real_minus_random_attenuation"].mean()) if not control_summary.empty else 0.0
    )
    mean_control_z = float(control_summary["attenuation_z_score"].mean()) if not control_summary.empty else 0.0
    n_edge_controls = int(len(edge_controls))
    n_strong_edge_outliers = (
        int((edge_controls["attenuation_z_score"].abs() >= 2.0).sum())
        if not edge_controls.empty and "attenuation_z_score" in edge_controls.columns
        else 0
    )
    return f"""# Morphology Weighted Connectivity

## Scope

This run maps real FlyWire synapse coordinates onto selected postsynaptic
skeletons from the whole-brain skeleton parquet, then writes an edge table with
both raw synapse count and passive morphology-weighted synapse count.

## Summary

| metric | value |
|---|---:|
| postsynaptic root ids requested | {n_roots} |
| mapped synapses | {n_synapses} |
| pre -> post edges | {n_edges} |
| mean mapped postsynaptic attenuation | {mean_attenuation:.6f} |
| mean real-minus-random attenuation | {mean_real_minus_random_attenuation:.6f} |
| mean attenuation z-score vs random | {mean_control_z:.3f} |
| edge-level random-control rows | {n_edge_controls} |
| edge-level abs(z) >= 2 rows | {n_strong_edge_outliers} |
| mean cable length per neuron (um) | {mean_cable:.3f} |
| mean cable sections per neuron | {mean_sections:.2f} |

## Outputs

- `tables/morphology_weighted_edges.csv`: edge-level raw and morphology-weighted counts.
- `tables/morphology_features.csv`: per-neuron cable, branch and radius features.
- `tables/morphology_sections.csv`: soma-to-branch section plan for a future compartmental backend.
- `tables/synapse_location_random_control_summary.csv`: optional real-vs-random synapse placement control.
- `tables/edge_synapse_location_random_controls.csv`: optional pre->post edge-level structural null control.
- `tables/connectivity_morphology_adjusted.parquet`: optional partial edge-weight adjusted connectome.
- `tables/connectivity_morphology_adjustment_summary.csv`: optional changed-edge impact summary.
- `metadata/morphology_weighted_connectivity_config.json`: run parameters.

## Model Boundary

`morphology_weighted_synapse_count` is the sum of passive attenuation values at
the mapped postsynaptic synapse locations.  It is designed for relative ranking,
connectome propagation weighting and sensitivity analysis.  It is not a claim
of measured membrane voltage, active conductance or spike timing.

The random-control table samples skeleton nodes in proportion to local cable
length.  It asks whether real mapped synapses are more proximal or distal than a
matched random distribution on the same morphology; it is a structural null
model, not an animal-causal test.
"""


def _bfs_tree_parents(graph: MorphologyGraph) -> np.ndarray:
    adjacency = graph.adjacency()
    parent = np.full(len(graph.nodes), -2, dtype=np.int64)
    root = int(graph.soma_node)
    parent[root] = -1
    queue = [root]
    for node in queue:
        for neighbor in adjacency[node]:
            if parent[neighbor] != -2:
                continue
            parent[neighbor] = node
            queue.append(neighbor)
    if np.any(parent == -2):
        missing = np.where(parent == -2)[0].tolist()[:10]
        raise ValueError(f"Graph is disconnected; unreachable SWC nodes include {missing}")
    return parent


def write_swc(graph: MorphologyGraph, path: Path) -> Path:
    """Write a graph as a conservative SWC tree."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parent = _bfs_tree_parents(graph)
    nodes = graph.nodes
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# root_id {int(graph.root_id)}\n")
        handle.write("# id type x y z radius parent\n")
        for node_id, row in nodes.iterrows():
            swc_id = int(node_id) + 1
            node_type = 1 if int(node_id) == int(graph.soma_node) else 3
            parent_id = -1 if parent[int(node_id)] < 0 else int(parent[int(node_id)]) + 1
            handle.write(
                f"{swc_id} {node_type} "
                f"{float(row['x_um']):.6f} {float(row['y_um']):.6f} {float(row['z_um']):.6f} "
                f"{max(float(row['radius_um']), 1e-6):.6f} {parent_id}\n"
            )
    validate_swc(path)
    return path


def read_swc(path: Path) -> pd.DataFrame:
    rows: list[list[float]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) != 7:
                raise ValueError(f"Invalid SWC row with {len(parts)} fields: {line!r}")
            rows.append([float(part) for part in parts])
    frame = pd.DataFrame(rows, columns=["id", "type", "x", "y", "z", "radius", "parent"])
    for column in ["id", "type", "parent"]:
        frame[column] = frame[column].astype(int)
    return frame


def validate_swc(path: Path) -> None:
    frame = read_swc(path)
    if frame.empty:
        raise ValueError(f"SWC file is empty: {path}")
    if frame["id"].duplicated().any():
        raise ValueError(f"SWC contains duplicate ids: {path}")
    roots = frame[frame["parent"] == -1]
    if len(roots) != 1:
        raise ValueError(f"SWC must contain exactly one root, found {len(roots)}: {path}")
    if not np.isfinite(frame[["x", "y", "z", "radius"]].to_numpy(dtype=float)).all():
        raise ValueError(f"SWC contains non-finite coordinates/radii: {path}")
    if (frame["radius"] <= 0).any():
        raise ValueError(f"SWC radii must be positive: {path}")
    ids = set(frame["id"].astype(int).tolist())
    bad_parents = sorted({int(parent) for parent in frame["parent"] if int(parent) != -1 and int(parent) not in ids})
    if bad_parents:
        raise ValueError(f"SWC parent ids not present in file: {bad_parents[:10]}")


def run_kc_morphology_benchmark(config: KCMorphologyBenchmarkConfig | None = None) -> dict[str, Path]:
    cfg = config or KCMorphologyBenchmarkConfig()
    skeleton_dir = resolve_kc_skeleton_dir(cfg.skeleton_dir)
    output_dir = cfg.output_dir
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    metadata_dir = output_dir / "metadata"
    for directory in (table_dir, figure_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    root_ids = list(cfg.root_ids) if cfg.root_ids else list_skeleton_root_ids(skeleton_dir)
    if cfg.max_neurons is not None:
        root_ids = root_ids[: max(0, int(cfg.max_neurons))]
    if not root_ids:
        raise ValueError(f"No KC skeleton root ids found in {skeleton_dir}")

    feature_rows: list[dict[str, object]] = []
    response_frames: list[pd.DataFrame] = []
    synapse_rows: list[dict[str, object]] = []
    for root_id in root_ids:
        graph = load_kc_skeleton(
            int(root_id),
            skeleton_dir=skeleton_dir,
            coordinate_scale_um=cfg.coordinate_scale_um,
            radius_scale_um=cfg.radius_scale_um,
            min_radius_um=cfg.min_radius_um,
        )
        feature_rows.append(morphology_feature_summary(graph))
        response_frames.append(
            location_response_table(graph, quantiles=cfg.location_quantiles, config=cfg.response)
        )
        synapse_rows.append(synapse_location_summary(graph, config=cfg.response))

    features = pd.DataFrame.from_records(feature_rows)
    responses = pd.concat(response_frames, ignore_index=True) if response_frames else pd.DataFrame()
    synapses = pd.DataFrame.from_records(synapse_rows)

    feature_path = table_dir / "kc_morphology_features.csv"
    response_path = table_dir / "kc_location_responses.csv"
    synapse_path = table_dir / "kc_synapse_location_summary.csv"
    features.to_csv(feature_path, index=False)
    responses.to_csv(response_path, index=False)
    synapses.to_csv(synapse_path, index=False)

    config_path = metadata_dir / "kc_morphology_benchmark_config.json"
    payload = asdict(cfg)
    payload["skeleton_dir"] = str(skeleton_dir)
    payload["output_dir"] = str(output_dir)
    payload["response"] = asdict(cfg.response)
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    figure_path = make_kc_morphology_figure(responses, synapses, figure_dir / "Fig_kc_morphology_response.png")
    report_path = output_dir / "KC_MORPHOLOGY_BENCHMARK_REPORT.md"
    report_path.write_text(
        render_kc_morphology_report(features, responses, synapses, cfg, skeleton_dir),
        encoding="utf-8",
    )

    return {
        "features_csv": feature_path,
        "responses_csv": response_path,
        "synapse_summary_csv": synapse_path,
        "config_json": config_path,
        "figure_png": figure_path,
        "report_md": report_path,
    }


def make_kc_morphology_figure(responses: pd.DataFrame, synapses: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if responses.empty:
        return output_path
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    for label, subset in responses.groupby("location_label"):
        ax.scatter(
            subset["path_distance_um"],
            subset["soma_peak_delta_mv"],
            s=18,
            alpha=0.65,
            label=str(label),
        )
    ax.set_xlabel("Path distance to soma (um)")
    ax.set_ylabel("Predicted soma peak delta (mV)")
    ax.set_title("Passive morphology surrogate")
    ax.legend(fontsize=8)

    ax = axes[1]
    if not synapses.empty:
        ax.hist(
            synapses["real_mean_postsynaptic_path_distance_um"],
            bins=24,
            color="tab:blue",
            alpha=0.75,
        )
    ax.set_xlabel("Mean postsynaptic path distance (um)")
    ax.set_ylabel("KC count")
    ax.set_title("Real input-location distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def render_kc_morphology_report(
    features: pd.DataFrame,
    responses: pd.DataFrame,
    synapses: pd.DataFrame,
    config: KCMorphologyBenchmarkConfig,
    skeleton_dir: Path,
) -> str:
    n = int(len(features))
    mean_cable = float(features["total_cable_length_um"].mean()) if n else 0.0
    mean_max_distance = float(features["max_path_distance_um"].mean()) if n else 0.0
    mean_real_distance = (
        float(synapses["real_mean_postsynaptic_path_distance_um"].mean()) if not synapses.empty else 0.0
    )
    proximal_amp = _mean_response_for_label(responses, "proximal")
    distal_amp = _mean_response_for_label(responses, "distal")
    return f"""# KC Morphology Benchmark

## Scope

This run used cached FlyWire-derived Kenyon-cell skeletons from:

`{skeleton_dir}`

It computes graph morphology features, exports a passive morphology-response
surrogate, and keeps SWC export compatible with a later NEURON backend.

## Summary

| metric | value |
|---|---:|
| KC skeletons analyzed | {n} |
| mean total cable length (um) | {mean_cable:.3f} |
| mean max soma path distance (um) | {mean_max_distance:.3f} |
| mean real postsynaptic path distance (um) | {mean_real_distance:.3f} |
| mean proximal soma peak delta (mV) | {proximal_amp:.4f} |
| mean distal soma peak delta (mV) | {distal_amp:.4f} |

## Model Boundary

The response values are a deterministic passive surrogate:

`attenuation = exp(-path_distance / length_constant) / branch_order_penalty`

They are suitable for relative proximal-vs-distal and real-vs-random input
comparisons. They are not spike-level electrophysiology and should not be
reported as real membrane responses unless an explicit NEURON passive or active
backend is run.

## Parameters

```json
{json.dumps(asdict(config.response), indent=2)}
```
"""


def _mean_response_for_label(responses: pd.DataFrame, label: str) -> float:
    if responses.empty or "location_label" not in responses.columns:
        return 0.0
    subset = responses[responses["location_label"] == label]
    if subset.empty:
        return 0.0
    return float(subset["soma_peak_delta_mv"].mean())
