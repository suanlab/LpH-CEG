from __future__ import annotations

from src.core.block import enumerate_blocks
from src.core.hceg import DEFAULT_MAX_STATES, build_hceg_dag
from src.core.hyperpath import shortest_hyperpath
from src.statistics.block_stats import RelationDataMap, compute_all_block_extensions
from src.utils.types import (
    Block,
    EstimationResult,
    Hyperpath,
    Interface,
    QueryHypergraph,
    State,
)


def estimate_pessimistic(
    query: QueryHypergraph,
    relation_data: RelationDataMap,
    *,
    max_block_size: int = 3,
    max_states: int = DEFAULT_MAX_STATES,
    apply_selectivity: bool = False,
) -> EstimationResult:
    """Estimate query cardinality with the exact pessimistic H-CEG_M bound.

    With ``apply_selectivity=False`` (the default) the returned estimate is an
    unconditional valid upper bound on the join size. ``apply_selectivity=True``
    is an *optional tightening* that multiplies the bound by the product of
    relation selectivities; it remains a valid upper bound ONLY under the
    contract that ``relation_data`` holds the unfiltered base tuples and each
    ``selectivity`` is the true fraction of tuples surviving a predicate that
    has NOT yet been applied to that data. If ``relation_data`` is already
    filtered, or the selectivity is mis-specified, the scaled value can fall
    below the true cardinality and is no longer a guaranteed bound. The default
    is therefore the guarantee-preserving mode.

    Args:
        query: Query hypergraph whose cardinality upper bound is estimated.
        relation_data: Exact input tuples for each relation in ``query``.
        max_block_size: Maximum block size used during block enumeration.
        max_states: Maximum number of reachable H-CEG states to materialize.
        apply_selectivity: If True, multiply the bound by relation
            selectivities (optional, non-guaranteed tightening; see above).
            Default False (guarantee-preserving).

    Returns:
        Estimation result whose ``estimate`` equals the exact pessimistic
        H-CEG_M upper bound, optionally adjusted for filter selectivity.
    """
    blocks = enumerate_blocks(query, max_block_size=max_block_size)
    all_stats = compute_all_block_extensions(query, blocks, relation_data)

    def weight_fn(_state: State, block: Block, interface: Interface) -> float:
        key = (block, interface)
        if key not in all_stats:
            msg = (
                "Missing exact block extension stats for "
                f"block={sorted(block.relations)!r}, interface={sorted(interface)!r}"
            )
            raise KeyError(msg)
        return all_stats[key].log_max_extension

    # Empty block join => empty query result. The full join projected onto
    # attr(F) is contained in the block join, so if any block is empty the
    # answer is exactly 0. Without this the log-space DAG would floor
    # log(mu = 0) to 0, i.e. treat the block as a factor of 1: still a valid
    # bound, but needlessly loose on a case we can answer exactly.
    if any(stat.max_extension == 0 for stat in all_stats.values()):
        return EstimationResult(
            estimate=0.0,
            hyperpath=Hyperpath(states=(), blocks=(), weights=()),
            method="pessimistic",
            metadata={
                "method_variant": "hceg_m",
                "max_block_size": max_block_size,
                "num_blocks": len(blocks),
                "empty_block": True,
                "bound": 0.0,
                "exact_shortest_path": True,
            },
        )

    dag = build_hceg_dag(query, blocks, weight_fn, max_states=max_states)
    hyperpath_result = shortest_hyperpath(dag)
    bound = hyperpath_result.bound
    total_weight_log2 = hyperpath_result.total_weight

    # Apply selectivity correction
    selectivity_factor = 1.0
    if apply_selectivity:
        selectivity_factor = _compute_selectivity_factor(query)
        bound *= selectivity_factor

    metadata: dict[str, object] = {
        "method_variant": "hceg_m",
        "max_block_size": max_block_size,
        "num_blocks": len(blocks),
        "num_states": len(dag.vertices),
        "num_hyperedges": len(dag.hyperedges),
        "pruned_count": dag.pruned_count,
        "exact_shortest_path": dag.pruned_count == 0,
        "total_weight_log2": total_weight_log2,
        "bound": bound,
        "selectivity_factor": selectivity_factor,
    }
    return EstimationResult(
        estimate=bound,
        hyperpath=hyperpath_result.hyperpath,
        method="pessimistic",
        metadata=metadata,
    )


def _compute_selectivity_factor(query: QueryHypergraph) -> float:
    """Compute the combined selectivity factor from all relations.

    The selectivity factor is the product of each relation's selectivity.
    When all selectivities are 1.0 (no filters), the factor is 1.0.
    """
    factor = 1.0
    for relation in query.relations:
        factor *= relation.selectivity
    return factor
