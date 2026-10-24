"""Reachable-state H-CEG DAG construction.

This module builds the explicit H-CEG DAG over relation states. Vertices are
reachable join states, represented as subsets of query relations. Directed
hyperedges add a valid block of new relations, with interface attributes
computed as ``A(S) ∩ A(F)``.

When the state count approaches ``max_states``, a beam-search pruning
strategy retains only the most promising states (smallest accumulated
weight) rather than aborting.
"""

from __future__ import annotations

import heapq
import logging
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from src.utils.types import Block, Interface, QueryHypergraph, State

LOGGER = logging.getLogger(__name__)
DEFAULT_MAX_STATES = 10_000

WeightFn = Callable[[State, Block, Interface], float]
BlockWeightMap = Mapping[tuple[Block, Interface], float]


@dataclass(frozen=True)
class HCEGHyperedge:
    """A directed H-CEG hyperedge from one relation state to another.

    Attributes:
        source: Source relation state.
        target: Target relation state after adding ``block``.
        block: Block of relations added along this transition.
        interface: Interface attributes ``A(source) ∩ A(block)``.
        weight: Edge weight supplied by the chosen H-CEG variant.
    """

    source: State
    target: State
    block: Block
    interface: Interface
    weight: float


@dataclass(frozen=True)
class HCEGDAG:
    """Explicit reachable-state DAG for H-CEG.

    Attributes:
        query_name: Name of the underlying query hypergraph.
        vertices: Reachable relation states in topological order.
        hyperedges: Hyperedges between reachable relation states.
        source: Empty relation state.
        sink: Full relation state containing all query relations.
        pruned_count: Number of states discarded by beam pruning during
            construction. ``0`` means the DAG is complete and the shortest
            hyperpath is exact; a positive value means the bound is a
            beam-search approximation and may not be the true shortest path.
    """

    query_name: str
    vertices: tuple[State, ...]
    hyperedges: tuple[HCEGHyperedge, ...]
    source: State
    sink: State
    pruned_count: int = 0


def build_hceg_dag(
    query: QueryHypergraph,
    blocks: Iterable[Block],
    weight_fn: WeightFn,
    *,
    max_states: int = DEFAULT_MAX_STATES,
    pruning: str = "beam",
) -> HCEGDAG:
    """Build the reachable-state H-CEG DAG for a query.

    The builder performs a BFS from the empty state and only materializes
    states reachable by repeatedly adding disjoint valid blocks from the given
    block family. For non-empty source states, each block must have a non-empty
    interface with the current state.

    Args:
        query: Query hypergraph whose relation states define the DAG.
        blocks: Candidate block family used for extensions.
        weight_fn: Callable returning the edge weight for
            ``(state, block, interface)``.
        max_states: Maximum allowed number of reachable states before
            pruning or aborting.
        pruning: Pruning strategy when state count exceeds ``max_states``.
            ``"abort"`` raises ValueError (legacy behavior).
            ``"beam"`` uses priority-bounded BFS, retaining only the
            most promising states by accumulated weight. Default ``"beam"``.

    Returns:
        Explicit H-CEG DAG over reachable relation states.

    Raises:
        ValueError: If ``max_states`` is less than 1, or if a provided block is
            invalid for ``query``, or if pruning is ``"abort"`` and the
            reachable-state count exceeds the threshold.
    """
    if max_states < 1:
        msg = "max_states must be at least 1"
        raise ValueError(msg)
    if pruning not in ("abort", "beam"):
        msg = f"pruning must be 'abort' or 'beam', got {pruning!r}"
        raise ValueError(msg)

    source = State(frozenset())
    sink = State(query.relation_names)
    ordered_blocks = _normalize_blocks(query, blocks)

    if pruning == "beam":
        return _build_hceg_dag_beam(
            query, ordered_blocks, weight_fn, max_states, source, sink
        )
    return _build_hceg_dag_abort(
        query, ordered_blocks, weight_fn, max_states, source, sink
    )


def _build_hceg_dag_abort(
    query: QueryHypergraph,
    ordered_blocks: tuple[Block, ...],
    weight_fn: WeightFn,
    max_states: int,
    source: State,
    sink: State,
) -> HCEGDAG:
    """Build DAG with abort-on-overflow (legacy behavior)."""
    visited: set[State] = {source}
    queue: deque[State] = deque([source])
    hyperedges: list[HCEGHyperedge] = []

    while queue:
        state = queue.popleft()
        for block in ordered_blocks:
            if not block.isdisjoint(state):
                continue

            interface = query.interface(state, block)
            if state.relations and not interface:
                continue

            target = state.union(block)
            weight = weight_fn(state, block, interface)
            hyperedges.append(
                HCEGHyperedge(
                    source=state,
                    target=target,
                    block=block,
                    interface=interface,
                    weight=weight,
                )
            )

            if target in visited:
                continue

            visited.add(target)
            if len(visited) > max_states:
                msg = f"H-CEG DAG exceeded {max_states} states for query {query.name!r}"
                raise ValueError(msg)
            queue.append(target)

    return _finalize_dag(query.name, visited, hyperedges, source, sink)


def _build_hceg_dag_beam(
    query: QueryHypergraph,
    ordered_blocks: tuple[Block, ...],
    weight_fn: WeightFn,
    max_states: int,
    source: State,
    sink: State,
) -> HCEGDAG:
    """Build DAG with beam-search pruning.

    Uses a priority queue ordered by accumulated weight. When the state
    count exceeds ``max_states``, only the ``max_states`` states with the
    smallest accumulated weight are retained, ensuring the sink and source
    are always kept.
    """
    # Priority queue: (accumulated_weight, tiebreaker, state)
    counter = 0
    pq: list[tuple[float, int, State]] = [(0.0, counter, source)]
    best_weight: dict[State, float] = {source: 0.0}
    visited: set[State] = set()
    hyperedges: list[HCEGHyperedge] = []
    pruned_count = 0

    while pq:
        acc_weight, _, state = heapq.heappop(pq)
        if state in visited:
            continue
        visited.add(state)

        for block in ordered_blocks:
            if not block.isdisjoint(state):
                continue

            interface = query.interface(state, block)
            if state.relations and not interface:
                continue

            target = state.union(block)
            edge_weight = weight_fn(state, block, interface)
            hyperedges.append(
                HCEGHyperedge(
                    source=state,
                    target=target,
                    block=block,
                    interface=interface,
                    weight=edge_weight,
                )
            )

            new_acc = acc_weight + edge_weight
            if target not in best_weight or new_acc < best_weight[target]:
                best_weight[target] = new_acc
                counter += 1
                heapq.heappush(pq, (new_acc, counter, target))

        # Beam pruning: if too many pending states, keep only the best.
        # Dropping frontier states only removes candidate hyperpaths, so the
        # surviving minimum can only be larger: the bound stays valid (T1).
        if len(best_weight) > max_states * 2:
            # Keep source, sink, and the max_states best by weight
            sorted_states = sorted(best_weight.items(), key=lambda x: x[1])
            keep: set[State] = {source, sink}
            for s, _ in sorted_states[:max_states]:
                keep.add(s)
            # Drop unexpanded frontier states outright -- retaining them (as an
            # earlier version did) made the threshold inert and the DAG grew in
            # full. Already-visited states keep their emitted hyperedges.
            dropped = {s for _, _, s in pq if s not in keep and s not in visited}
            if dropped:
                pq = [(w, c, s) for w, c, s in pq if s not in dropped]
                heapq.heapify(pq)
                pruned_count += len(dropped)
            best_weight = {s: w for s, w in best_weight.items()
                           if s in keep or s in visited}

    if pruned_count > 0:
        LOGGER.info(
            "Beam pruning removed %d low-promise states for query %r",
            pruned_count,
            query.name,
        )

    # Filter hyperedges to only include those between visited states
    valid_hyperedges = [
        he for he in hyperedges
        if he.source in visited and he.target in visited
    ]

    return _finalize_dag(
        query.name, visited, valid_hyperedges, source, sink, pruned_count=pruned_count
    )


def _finalize_dag(
    query_name: str,
    visited: set[State],
    hyperedges: list[HCEGHyperedge],
    source: State,
    sink: State,
    pruned_count: int = 0,
) -> HCEGDAG:
    """Sort and package visited states and hyperedges into an HCEGDAG."""
    vertices = tuple(sorted(visited, key=_state_sort_key))
    ordered_hyperedges = tuple(sorted(hyperedges, key=_hyperedge_sort_key))

    LOGGER.debug(
        "Built H-CEG DAG for %r with %d states and %d hyperedges",
        query_name,
        len(vertices),
        len(ordered_hyperedges),
    )
    return HCEGDAG(
        query_name=query_name,
        vertices=vertices,
        hyperedges=ordered_hyperedges,
        source=source,
        sink=sink,
        pruned_count=pruned_count,
    )


def build_pessimistic_hceg(
    query: QueryHypergraph,
    blocks: Iterable[Block],
    block_stats: BlockWeightMap,
    *,
    max_states: int = DEFAULT_MAX_STATES,
) -> HCEGDAG:
    """Build an H-CEG_M DAG using log2(mu(F | I)) edge weights.

    Args:
        query: Query hypergraph to build over.
        blocks: Candidate block family.
        block_stats: Mapping from ``(block, interface)`` to ``log2(mu(F | I))``.
        max_states: Maximum allowed reachable-state count.

    Returns:
        Reachable-state H-CEG DAG with pessimistic edge weights.
    """

    def weight_fn(_state: State, block: Block, interface: Interface) -> float:
        return _lookup_block_weight(block_stats, block, interface, "pessimistic")

    return build_hceg_dag(
        query,
        blocks,
        weight_fn,
        max_states=max_states,
    )


def build_optimistic_hceg(
    query: QueryHypergraph,
    blocks: Iterable[Block],
    block_stats: BlockWeightMap,
    *,
    max_states: int = DEFAULT_MAX_STATES,
) -> HCEGDAG:
    """Build an H-CEG_O DAG using average extension-rate edge weights.

    Args:
        query: Query hypergraph to build over.
        blocks: Candidate block family.
        block_stats: Mapping from ``(block, interface)`` to average extension
            rate.
        max_states: Maximum allowed reachable-state count.

    Returns:
        Reachable-state H-CEG DAG with optimistic edge weights.
    """

    def weight_fn(_state: State, block: Block, interface: Interface) -> float:
        return _lookup_block_weight(block_stats, block, interface, "optimistic")

    return build_hceg_dag(
        query,
        blocks,
        weight_fn,
        max_states=max_states,
    )


def _normalize_blocks(
    query: QueryHypergraph, blocks: Iterable[Block]
) -> tuple[Block, ...]:
    unique_blocks = set(blocks)
    for block in unique_blocks:
        if not query.is_valid_block(block):
            msg = f"Invalid block for query {query.name!r}: {sorted(block.relations)!r}"
            raise ValueError(msg)
    return tuple(sorted(unique_blocks, key=_block_sort_key))


def _lookup_block_weight(
    block_stats: BlockWeightMap,
    block: Block,
    interface: Interface,
    mode: str,
) -> float:
    key = (block, interface)
    if key not in block_stats:
        msg = (
            f"Missing {mode} block statistic for block={sorted(block.relations)!r}, "
            f"interface={sorted(interface)!r}"
        )
        raise KeyError(msg)
    return block_stats[key]


def _state_sort_key(state: State) -> tuple[int, tuple[str, ...]]:
    return len(state), tuple(sorted(state.relations))


def _block_sort_key(block: Block) -> tuple[int, tuple[str, ...]]:
    return len(block), tuple(sorted(block.relations))


def _hyperedge_sort_key(
    hyperedge: HCEGHyperedge,
) -> tuple[
    tuple[int, tuple[str, ...]],
    tuple[int, tuple[str, ...]],
    tuple[int, tuple[str, ...]],
    tuple[str, ...],
]:
    return (
        _state_sort_key(hyperedge.source),
        _state_sort_key(hyperedge.target),
        _block_sort_key(hyperedge.block),
        tuple(sorted(hyperedge.interface)),
    )
