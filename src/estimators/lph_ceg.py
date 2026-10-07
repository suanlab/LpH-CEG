"""LpH-CEG estimator: Lp-norm enhanced H-CEG cardinality estimation.

Combines H-CEG's multi-way block extensions with LpBound's Lp-norm
degree sequence statistics.  For each candidate hyperpath, an LP
selects optimal (p, w) weights to minimize the cardinality upper bound.

Supports two strategies:
- **Hyperpath-then-LP** (default): Find the ℓ∞-shortest hyperpath,
  then LP-optimize its weights.  Fast, often near-optimal.
- **Joint optimization** (explore_k > 1): Enumerate the K best
  hyperpaths by ℓ∞ weight, LP-optimize each, and return the
  tightest bound across all candidates.  Strictly tighter or equal.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.core.block import enumerate_blocks
from src.core.hceg import DEFAULT_MAX_STATES, build_hceg_dag
from src.estimators.lp_bound import optimize_hyperpath_weights
from src.statistics.lp_stats import compute_all_lp_block_extensions

if TYPE_CHECKING:
    from src.core.hceg import HCEGDAG, HCEGHyperedge
    from src.statistics.block_stats import RelationDataMap
    from src.statistics.lp_stats import LpBlockExtensionStats
    from src.utils.types import (
        Block,
        EstimationResult,
        Hyperpath,
        Interface,
        QueryHypergraph,
    )

LOGGER = logging.getLogger(__name__)

DEFAULT_P_VALUES: tuple[float, ...] = (1.0, 2.0, float("inf"))


# ---------------------------------------------------------------------------
# K-best hyperpath enumeration
# ---------------------------------------------------------------------------


def _enumerate_k_best_hyperpaths(
    dag: HCEGDAG,
    k: int,
) -> list[tuple[Hyperpath, float]]:
    """Enumerate up to k shortest hyperpaths in the DAG.

    Uses Yen's-style successive shortest path approach adapted for
    DAGs: after finding each shortest path, edges along that path are
    penalized to diversify the next candidate.

    Returns list of (hyperpath, total_weight) pairs sorted by weight.
    """
    from src.core.hyperpath import shortest_hyperpath as _sp
    from src.utils.types import Hyperpath, State

    if k <= 1:
        result = _sp(dag)
        return [(result.hyperpath, result.total_weight)]

    # Build outgoing edge map
    outgoing: dict[State, list[HCEGHyperedge]] = {v: [] for v in dag.vertices}
    for he in dag.hyperedges:
        outgoing[he.source].append(he)

    # Enumerate all source-to-sink paths via DFS (DAG is small)
    all_paths: list[tuple[Hyperpath, float]] = []
    stack: list[tuple[State, list[HCEGHyperedge]]] = [(dag.source, [])]

    while stack:
        current, path_edges = stack.pop()
        if current == dag.sink:
            states = tuple([dag.source, *[he.target for he in path_edges]])
            blocks = tuple(he.block for he in path_edges)
            weights = tuple(he.weight for he in path_edges)
            hp = Hyperpath(states=states, blocks=blocks, weights=weights)
            total = sum(he.weight for he in path_edges)
            all_paths.append((hp, total))
            continue

        for he in outgoing[current]:
            stack.append((he.target, [*path_edges, he]))

            # Safety cap to avoid combinatorial explosion
            if len(all_paths) >= k * 10:
                break
        if len(all_paths) >= k * 10:
            break

    all_paths.sort(key=lambda x: x[1])
    return all_paths[:k]


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------


def estimate_lph_ceg(
    query: QueryHypergraph,
    relation_data: RelationDataMap,
    *,
    max_block_size: int = 3,
    max_states: int = DEFAULT_MAX_STATES,
    p_values: tuple[float, ...] = DEFAULT_P_VALUES,
    apply_selectivity: bool = True,
    explore_k: int = 1,
    lp_stats: dict[tuple[Block, Interface], LpBlockExtensionStats] | None = None,
) -> EstimationResult:
    """Estimate cardinality using LpH-CEG: LP-optimized Lp-norm bounds.

    Args:
        query: Query hypergraph.
        relation_data: Exact relation tuples.
        max_block_size: Maximum block size for enumeration.
        max_states: Maximum H-CEG DAG states.
        p_values: Lp-norm exponents for the LP.
        apply_selectivity: Scale bound by relation selectivities.
        explore_k: Number of candidate hyperpaths to LP-optimize.
            1 = standard hyperpath-then-LP (default).
            >1 = joint optimization over K best hyperpaths.
        lp_stats: Precomputed ``(block, interface)`` statistics covering
            every block of size ``<= max_block_size``. When given, the
            statistics-collection step is skipped, which is how a
            workload-level statistics catalogue is shared across queries
            (``scripts/stats_catalogue.py``). ``None`` computes them here.

    Returns:
        Estimation result with the tightest bound found.
    """
    from src.utils.types import EstimationResult, State

    # Step 1: Enumerate blocks and compute all Lp stats (or take them as given)
    blocks = enumerate_blocks(query, max_block_size=max_block_size)
    all_lp_stats = (
        lp_stats
        if lp_stats is not None
        else compute_all_lp_block_extensions(
            query, blocks, relation_data, p_values=p_values
        )
    )

    # Step 2: Build H-CEG DAG with standard max-degree weights
    def weight_fn(state: State, block: Block, interface: Interface) -> float:
        key = (block, interface)
        if key not in all_lp_stats:
            msg = (
                "Missing Lp block extension stats for "
                f"block={sorted(block.relations)!r}, "
                f"interface={sorted(interface)!r}"
            )
            raise KeyError(msg)
        return all_lp_stats[key].log_max_extension

    dag = build_hceg_dag(query, blocks, weight_fn, max_states=max_states)

    # Step 3: Find candidate hyperpaths
    candidates = _enumerate_k_best_hyperpaths(dag, explore_k)

    # Step 4: Standard bound from the ℓ∞-shortest hyperpath
    best_hp, standard_log = candidates[0]
    standard_bound = 2.0**standard_log

    # Step 5: LP-optimize each candidate, keep the tightest
    best_lp = optimize_hyperpath_weights(
        query, best_hp, all_lp_stats, p_values=p_values
    )
    best_bound = (
        min(standard_bound, best_lp.bound) if best_lp.feasible else standard_bound
    )
    best_lp_result = best_lp
    chosen_hp = best_hp
    paths_explored = len(candidates)

    for hp, _w in candidates[1:]:
        lp_result = optimize_hyperpath_weights(
            query, hp, all_lp_stats, p_values=p_values
        )
        if lp_result.feasible and lp_result.bound < best_bound:
            best_bound = lp_result.bound
            best_lp_result = lp_result
            chosen_hp = hp

    # Step 6: Determine method variant
    if best_bound < standard_bound - 1e-10:
        method_detail = "lph_ceg_lp"
    else:
        method_detail = "lph_ceg_standard"
        best_bound = standard_bound

    # Apply selectivity correction
    selectivity_factor = 1.0
    if apply_selectivity:
        for rel in query.relations:
            selectivity_factor *= rel.selectivity
        best_bound *= selectivity_factor

    metadata: dict[str, object] = {
        "method_variant": method_detail,
        "max_block_size": max_block_size,
        "p_values": list(p_values),
        "standard_bound": standard_bound,
        "standard_log_bound": standard_log,
        "lp_bound": best_lp_result.bound,
        "lp_log_bound": best_lp_result.log_bound,
        "lp_feasible": best_lp_result.feasible,
        "lp_p_values_used": list(best_lp_result.p_values_used),
        "lp_weights_per_step": list(best_lp_result.weights_per_step),
        "improvement_ratio": standard_bound / max(best_lp_result.bound, 1e-10),
        "selectivity_factor": selectivity_factor,
        "num_blocks": len(blocks),
        "num_states": len(dag.vertices),
        "paths_explored": paths_explored,
        "explore_k": explore_k,
    }

    return EstimationResult(
        estimate=best_bound,
        hyperpath=chosen_hp,
        method="lph_ceg",
        metadata=metadata,
    )
