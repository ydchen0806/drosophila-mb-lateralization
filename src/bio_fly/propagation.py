from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping
import warnings

import numpy as np
import pandas as pd

from .paths import DEFAULT_CONNECTIVITY_PATH


EDGE_COLUMNS = ["Presynaptic_ID", "Postsynaptic_ID", "Excitatory x Connectivity"]
EDGE_INDEX_COLUMNS = [
    "Presynaptic_ID",
    "Postsynaptic_ID",
    "Presynaptic_Index",
    "Postsynaptic_Index",
    "Excitatory x Connectivity",
]


@dataclass(frozen=True)
class PropagationConfig:
    steps: int = 3
    max_active: int = 5_000
    normalize_each_step: bool = True


@dataclass(frozen=True)
class PropagationSummary:
    condition: str
    seed_ids: str
    silence_ids: str
    active_neurons: int
    positive_mass: float
    negative_mass: float
    absolute_mass: float
    max_abs_score: float
    readout_score: float


@dataclass(frozen=True)
class TorchPropagationGraph:
    matrix: object
    root_ids_by_index: np.ndarray
    root_to_index: dict[int, int]
    device: str


def load_connectivity_edges(connectivity_path: Path = DEFAULT_CONNECTIVITY_PATH) -> pd.DataFrame:
    return pd.read_parquet(connectivity_path, columns=EDGE_COLUMNS)


def load_connectivity_edges_with_indices(connectivity_path: Path = DEFAULT_CONNECTIVITY_PATH) -> pd.DataFrame:
    return pd.read_parquet(connectivity_path, columns=EDGE_INDEX_COLUMNS)


def build_torch_propagation_graph(
    connectivity_path: Path = DEFAULT_CONNECTIVITY_PATH,
    device: str = "cuda",
    dtype: str = "float32",
) -> TorchPropagationGraph:
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA propagation but torch.cuda.is_available() is False")
    edges = load_connectivity_edges_with_indices(connectivity_path)
    max_index = int(max(edges["Presynaptic_Index"].max(), edges["Postsynaptic_Index"].max()))
    n_nodes = max_index + 1

    index_to_root = (
        pd.concat(
            [
                edges[["Presynaptic_Index", "Presynaptic_ID"]].rename(
                    columns={"Presynaptic_Index": "index", "Presynaptic_ID": "root_id"}
                ),
                edges[["Postsynaptic_Index", "Postsynaptic_ID"]].rename(
                    columns={"Postsynaptic_Index": "index", "Postsynaptic_ID": "root_id"}
                ),
            ],
            ignore_index=True,
        )
        .drop_duplicates("index")
        .sort_values("index")
    )
    root_ids_by_index = np.zeros(n_nodes, dtype=np.int64)
    root_ids_by_index[index_to_root["index"].to_numpy(dtype=np.int64)] = index_to_root["root_id"].to_numpy(
        dtype=np.int64
    )
    root_to_index = {
        int(root_id): int(index)
        for index, root_id in zip(index_to_root["index"].to_numpy(), index_to_root["root_id"].to_numpy())
    }

    torch_dtype = torch.float32 if dtype == "float32" else torch.float64
    indices = torch.as_tensor(
        np.vstack(
            [
                edges["Postsynaptic_Index"].to_numpy(dtype=np.int64),
                edges["Presynaptic_Index"].to_numpy(dtype=np.int64),
            ]
        ),
        dtype=torch.long,
        device=device,
    )
    values = torch.as_tensor(
        edges["Excitatory x Connectivity"].to_numpy(dtype=np.float32 if dtype == "float32" else np.float64),
        dtype=torch_dtype,
        device=device,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse invariant checks are implicitly disabled.*")
        matrix = torch.sparse_coo_tensor(
            indices,
            values,
            size=(n_nodes, n_nodes),
            device=device,
            check_invariants=False,
        ).coalesce()
    return TorchPropagationGraph(
        matrix=matrix,
        root_ids_by_index=root_ids_by_index,
        root_to_index=root_to_index,
        device=device,
    )


def _id_set(values: Iterable[int] | None) -> set[int]:
    if values is None:
        return set()
    return {int(value) for value in values if pd.notna(value)}


def _normalize_frontier(frontier: pd.DataFrame) -> pd.DataFrame:
    absolute_total = frontier["score"].abs().sum()
    if absolute_total > 0:
        frontier = frontier.copy()
        frontier["score"] = frontier["score"] / absolute_total
    return frontier


def _seed_weight_map(seed_ids: Iterable[int] | Mapping[int, float]) -> dict[int, float]:
    if isinstance(seed_ids, Mapping):
        weights = {int(root_id): float(weight) for root_id, weight in seed_ids.items() if pd.notna(weight)}
    else:
        seed_set = _id_set(seed_ids)
        weights = {root_id: 1.0 for root_id in seed_set}
    return {root_id: weight for root_id, weight in weights.items() if weight != 0.0}


def _initial_seed_frontier(seed_ids: Iterable[int] | Mapping[int, float]) -> pd.DataFrame:
    weights = _seed_weight_map(seed_ids)
    if not weights:
        return pd.DataFrame(columns=["root_id", "score"])
    frontier = pd.DataFrame({"root_id": list(weights), "score": list(weights.values())})
    return _normalize_frontier(frontier)


def _limit_frontier(frontier: pd.DataFrame, max_active: int) -> pd.DataFrame:
    if len(frontier) <= max_active:
        return frontier
    limited = frontier.assign(abs_score=frontier["score"].abs())
    limited = limited.nlargest(max_active, "abs_score").drop(columns=["abs_score"])
    return limited.reset_index(drop=True)


def signed_multihop_response(
    edges: pd.DataFrame,
    seed_ids: Iterable[int] | Mapping[int, float],
    config: PropagationConfig | None = None,
    silence_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    config = config or PropagationConfig()
    silence_set = _id_set(silence_ids)
    active = _initial_seed_frontier(seed_ids)
    if active.empty:
        return pd.DataFrame(columns=["root_id", "score", "step"])
    if silence_set:
        active = active[~active["root_id"].isin(silence_set)].copy()

    filtered_edges = edges
    if silence_set:
        filtered_edges = edges[
            ~edges["Presynaptic_ID"].isin(silence_set) & ~edges["Postsynaptic_ID"].isin(silence_set)
        ].copy()

    step_frames: list[pd.DataFrame] = []
    for step_number in range(1, config.steps + 1):
        if active.empty:
            break
        frontier_edges = filtered_edges[filtered_edges["Presynaptic_ID"].isin(active["root_id"])]
        if frontier_edges.empty:
            break
        propagated = frontier_edges.merge(active, left_on="Presynaptic_ID", right_on="root_id", how="inner")
        propagated["score"] = propagated["score"] * propagated["Excitatory x Connectivity"].astype("float64")
        next_active = (
            propagated.groupby("Postsynaptic_ID", as_index=False)["score"]
            .sum()
            .rename(columns={"Postsynaptic_ID": "root_id"})
        )
        next_active = next_active[next_active["score"] != 0].copy()
        if config.normalize_each_step:
            next_active = _normalize_frontier(next_active)
        next_active = _limit_frontier(next_active, config.max_active)
        step_frame = next_active.copy()
        step_frame["step"] = step_number
        step_frames.append(step_frame)
        active = next_active

    if not step_frames:
        return pd.DataFrame(columns=["root_id", "score", "step"])
    return pd.concat(step_frames, ignore_index=True)


def signed_multihop_response_torch(
    graph: TorchPropagationGraph,
    seed_ids: Iterable[int] | Mapping[int, float],
    config: PropagationConfig | None = None,
    silence_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    import torch

    config = config or PropagationConfig()
    seed_weights = {
        int(root_id): float(weight)
        for root_id, weight in _seed_weight_map(seed_ids).items()
        if int(root_id) in graph.root_to_index
    }
    seed_indices = [graph.root_to_index[root_id] for root_id in seed_weights]
    silence_indices = [
        graph.root_to_index[int(root_id)] for root_id in _id_set(silence_ids) if int(root_id) in graph.root_to_index
    ]
    if not seed_indices:
        return pd.DataFrame(columns=["root_id", "score", "step"])

    n_nodes = graph.matrix.shape[0]
    active = torch.zeros(n_nodes, dtype=graph.matrix.dtype, device=graph.device)
    seed_tensor = torch.as_tensor(seed_indices, dtype=torch.long, device=graph.device)
    seed_score_tensor = torch.as_tensor(
        [seed_weights[root_id] for root_id in seed_weights],
        dtype=graph.matrix.dtype,
        device=graph.device,
    )
    seed_total = torch.sum(torch.abs(seed_score_tensor))
    if float(seed_total.item()) > 0:
        seed_score_tensor = seed_score_tensor / seed_total
    active[seed_tensor] = seed_score_tensor
    if silence_indices:
        silence_tensor = torch.as_tensor(silence_indices, dtype=torch.long, device=graph.device)
        active[silence_tensor] = 0
    else:
        silence_tensor = None

    step_frames: list[pd.DataFrame] = []
    for step_number in range(1, config.steps + 1):
        if not bool(torch.any(active != 0).item()):
            break
        next_active = torch.sparse.mm(graph.matrix, active.unsqueeze(1)).squeeze(1)
        if silence_tensor is not None:
            next_active[silence_tensor] = 0
        next_active = torch.where(next_active != 0, next_active, torch.zeros_like(next_active))
        if config.normalize_each_step:
            absolute_total = torch.sum(torch.abs(next_active))
            if float(absolute_total.item()) > 0:
                next_active = next_active / absolute_total
        nonzero_indices = torch.nonzero(next_active, as_tuple=False).flatten()
        if nonzero_indices.numel() == 0:
            break
        if nonzero_indices.numel() > config.max_active:
            _, selected_positions = torch.topk(torch.abs(next_active[nonzero_indices]), k=config.max_active)
            active_indices = nonzero_indices[selected_positions]
        else:
            active_indices = nonzero_indices
        active_scores = next_active[active_indices]
        cpu_indices = active_indices.detach().cpu().numpy()
        cpu_scores = active_scores.detach().cpu().numpy()
        step_frames.append(
            pd.DataFrame(
                {
                    "root_id": graph.root_ids_by_index[cpu_indices],
                    "score": cpu_scores,
                    "step": step_number,
                }
            )
        )
        active = torch.zeros_like(active)
        active[active_indices] = active_scores

    if not step_frames:
        return pd.DataFrame(columns=["root_id", "score", "step"])
    return pd.concat(step_frames, ignore_index=True)


def summarize_response(
    condition: str,
    seed_ids: Iterable[int],
    response: pd.DataFrame,
    silence_ids: Iterable[int] | None = None,
    readout_ids: Iterable[int] | None = None,
) -> PropagationSummary:
    silence_set = _id_set(silence_ids)
    readout_set = _id_set(readout_ids)
    seed_set = _id_set(seed_ids)
    if response.empty:
        return PropagationSummary(
            condition=condition,
            seed_ids=";".join(map(str, sorted(seed_set))),
            silence_ids=";".join(map(str, sorted(silence_set))),
            active_neurons=0,
            positive_mass=0.0,
            negative_mass=0.0,
            absolute_mass=0.0,
            max_abs_score=0.0,
            readout_score=0.0,
        )

    aggregate = response.groupby("root_id", as_index=False)["score"].sum()
    readout_score = 0.0
    if readout_set:
        readout_score = float(aggregate.loc[aggregate["root_id"].isin(readout_set), "score"].sum())

    return PropagationSummary(
        condition=condition,
        seed_ids=";".join(map(str, sorted(seed_set))),
        silence_ids=";".join(map(str, sorted(silence_set))),
        active_neurons=int(len(aggregate)),
        positive_mass=float(aggregate.loc[aggregate["score"] > 0, "score"].sum()),
        negative_mass=float(aggregate.loc[aggregate["score"] < 0, "score"].sum()),
        absolute_mass=float(aggregate["score"].abs().sum()),
        max_abs_score=float(aggregate["score"].abs().max()),
        readout_score=readout_score,
    )


def response_overlap(left_response: pd.DataFrame, right_response: pd.DataFrame, top_n: int = 200) -> float:
    if left_response.empty or right_response.empty:
        return 0.0
    left_top = set(
        left_response.assign(abs_score=left_response["score"].abs())
        .nlargest(top_n, "abs_score")["root_id"]
        .astype(int)
        .tolist()
    )
    right_top = set(
        right_response.assign(abs_score=right_response["score"].abs())
        .nlargest(top_n, "abs_score")["root_id"]
        .astype(int)
        .tolist()
    )
    union = left_top | right_top
    if not union:
        return 0.0
    return len(left_top & right_top) / len(union)


def summaries_to_frame(summaries: list[PropagationSummary]) -> pd.DataFrame:
    return pd.DataFrame.from_records([asdict(summary) for summary in summaries])
