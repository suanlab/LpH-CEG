from __future__ import annotations

from src.core.block import enumerate_blocks
from src.core.hceg import DEFAULT_MAX_STATES, build_hceg_dag
from src.core.hyperpath import shortest_hyperpath
from src.statistics.block_stats import RelationDataMap, compute_all_block_extensions
from src.utils.types import Block, EstimationResult, Interface, QueryHypergraph, State


def estimate_pessimistic(
    query: QueryHypergraph,
    relation_data: RelationDataMap,
    *,
    max_block_size: int = 3,
    max_states: int = DEFAULT_MAX_STATES,
    apply_selectivity: bool = True,
) -> EstimationResult:
    """Estimate query cardinality with the exact pessimistic H-CEG_M bound.

    When ``apply_selectivity`` is True and any relation has
    ``selectivity < 1.0``, the estimate is scaled down by the product of
    selectivities. This provides a tighter bound when filter predicates
    reduce the effective relation sizes.

    Args:
        query: Query hypergraph whose cardinality upper bound is estimated.
        relation_data: Exact input tuples for each relation in ``query``.
        max_block_size: Maximum block size used during block enumeration.
        max_states: Maximum number of reachable H-CEG states to materialize.
        apply_selectivity: If True, multiply the bound by relation
            selectivities. Default True.

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
