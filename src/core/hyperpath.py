"""Shortest hyperpath computation on explicit H-CEG DAGs.

This module implements the shortest B-hyperpath algorithm for the explicit
reachable-state H-CEG DAG. In the current H-CEG construction, each hyperedge
has one source state and one target state, so the shortest hyperpath reduces to
shortest-path relaxation on a DAG in topological order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.utils.types import Hyperpath, State

if TYPE_CHECKING:
    from src.core.hceg import HCEGDAG, HCEGHyperedge


@dataclass(frozen=True)
class HyperpathResult:
    """Shortest hyperpath result in log-space.

    Attributes:
        hyperpath: Reconstructed hyperpath from source to sink.
        total_weight: Sum of edge weights along the shortest hyperpath.
    """

    hyperpath: Hyperpath
    total_weight: float

    @property
    def bound(self) -> float:
        """Return the pessimistic estimate in the original scale."""
        return float(2.0**self.total_weight)


def shortest_hyperpath(graph: HCEGDAG) -> HyperpathResult:
    """Compute the minimum-weight hyperpath from source to sink.

    Edge weights are assumed to already be in log2-space, so path weights are
    accumulated by addition.

    Args:
        graph: Explicit H-CEG DAG in topological order.

    Returns:
        Shortest hyperpath result containing the reconstructed hyperpath, total
        log-space weight, and induced bound.

    Raises:
        ValueError: If the graph is malformed or if the sink is unreachable.
    """
    _validate_graph(graph)

    if graph.source == graph.sink:
        hyperpath = Hyperpath(states=(graph.source,), blocks=(), weights=())
        _validate_hyperpath(hyperpath, graph)
        return HyperpathResult(hyperpath=hyperpath, total_weight=0.0)

    outgoing = _build_outgoing_hyperedges(graph)
    distances = {vertex: math.inf for vertex in graph.vertices}
    predecessors: dict[State, HCEGHyperedge] = {}
    distances[graph.source] = 0.0

    for vertex in graph.vertices:
        current_distance = distances[vertex]
        if math.isinf(current_distance):
            continue
        for hyperedge in outgoing[vertex]:
            candidate = current_distance + hyperedge.weight
            if candidate < distances[hyperedge.target]:
                distances[hyperedge.target] = candidate
                predecessors[hyperedge.target] = hyperedge

    total_weight = distances[graph.sink]
    if math.isinf(total_weight):
        msg = f"H-CEG sink is unreachable for query {graph.query_name!r}"
        raise ValueError(msg)

    hyperpath = _reconstruct_hyperpath(graph, predecessors)
    _validate_hyperpath(hyperpath, graph)
    return HyperpathResult(hyperpath=hyperpath, total_weight=total_weight)


def _build_outgoing_hyperedges(
    graph: HCEGDAG,
) -> dict[State, list[HCEGHyperedge]]:
    outgoing: dict[State, list[HCEGHyperedge]] = {
        vertex: [] for vertex in graph.vertices
    }
    for hyperedge in graph.hyperedges:
        outgoing[hyperedge.source].append(hyperedge)
    return outgoing


def _reconstruct_hyperpath(
    graph: HCEGDAG,
    predecessors: dict[State, HCEGHyperedge],
) -> Hyperpath:
    path_edges: list[HCEGHyperedge] = []
    current = graph.sink

    while current != graph.source:
        if current not in predecessors:
            msg = f"Failed to reconstruct hyperpath for query {graph.query_name!r}"
            raise ValueError(msg)
        hyperedge = predecessors[current]
        path_edges.append(hyperedge)
        current = hyperedge.source

    path_edges.reverse()
    states = tuple([graph.source, *[hyperedge.target for hyperedge in path_edges]])
    blocks = tuple(hyperedge.block for hyperedge in path_edges)
    weights = tuple(hyperedge.weight for hyperedge in path_edges)
    return Hyperpath(states=states, blocks=blocks, weights=weights)


def _validate_graph(graph: HCEGDAG) -> None:
    if not graph.vertices:
        msg = f"H-CEG DAG for query {graph.query_name!r} must contain vertices"
        raise ValueError(msg)
    if graph.source not in graph.vertices:
        msg = f"H-CEG source is missing from vertices for query {graph.query_name!r}"
        raise ValueError(msg)
    if graph.sink not in graph.vertices:
        msg = f"H-CEG sink is missing from vertices for query {graph.query_name!r}"
        raise ValueError(msg)

    vertex_to_index = {vertex: index for index, vertex in enumerate(graph.vertices)}
    for hyperedge in graph.hyperedges:
        if hyperedge.source not in vertex_to_index:
            msg = (
                f"Hyperedge source is missing from vertices for query "
                f"{graph.query_name!r}: {sorted(hyperedge.source.relations)!r}"
            )
            raise ValueError(msg)
        if hyperedge.target not in vertex_to_index:
            msg = (
                f"Hyperedge target is missing from vertices for query "
                f"{graph.query_name!r}: {sorted(hyperedge.target.relations)!r}"
            )
            raise ValueError(msg)
        if vertex_to_index[hyperedge.source] >= vertex_to_index[hyperedge.target]:
            msg = (
                f"H-CEG hyperedge violates topological order for query "
                f"{graph.query_name!r}"
            )
            raise ValueError(msg)


def _validate_hyperpath(hyperpath: Hyperpath, graph: HCEGDAG) -> None:
    expected_num_states = len(hyperpath.blocks) + 1
    if len(hyperpath.states) != expected_num_states:
        msg = "Hyperpath must contain exactly one more state than blocks"
        raise ValueError(msg)
    if len(hyperpath.weights) != len(hyperpath.blocks):
        msg = "Hyperpath weights and blocks must have equal length"
        raise ValueError(msg)
    if hyperpath.states[0] != graph.source:
        msg = "Hyperpath must start at the graph source"
        raise ValueError(msg)
    if hyperpath.states[-1] != graph.sink:
        msg = "Hyperpath must end at the graph sink"
        raise ValueError(msg)

    for index, block in enumerate(hyperpath.blocks):
        source = hyperpath.states[index]
        target = hyperpath.states[index + 1]
        expected_target = source.union(block)
        if target != expected_target:
            msg = (
                "Hyperpath state transition does not match source.union(block) at "
                f"step {index}"
            )
            raise ValueError(msg)
