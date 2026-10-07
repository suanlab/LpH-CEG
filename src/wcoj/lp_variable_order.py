"""LpH-CEG-guided variable ordering for WCOJ.

Uses LP-optimized Lp-norm residual bounds to select the variable that
minimizes the residual cardinality upper bound at each step.  Falls back
to the standard H-CEG ordering when LP gives no improvement.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.core.hceg import DEFAULT_MAX_STATES
from src.estimators.lph_ceg import estimate_lph_ceg
from src.wcoj.residual_bound import (
    _connected_components,
    _normalize_prefix,
    _project_residual_relations,
)

if TYPE_CHECKING:
    from src.statistics.block_stats import RelationDataMap
    from src.utils.types import AttributeName, QueryHypergraph

LOGGER = logging.getLogger(__name__)

DEFAULT_LP_P_VALUES: tuple[float, ...] = (1.0, 2.0, float("inf"))


def lp_variable_ordering(
    query: QueryHypergraph,
    relation_data: RelationDataMap,
    *,
    max_block_size: int = 1,
    max_states: int = DEFAULT_MAX_STATES,
    p_values: tuple[float, ...] = DEFAULT_LP_P_VALUES,
) -> list[AttributeName]:
    """Return an LpH-CEG-guided variable ordering for WCOJ.

    Greedy algorithm: at each step, pick the variable whose removal
    minimizes the LP-optimized residual bound.

    Args:
        query: Query hypergraph.
        relation_data: Exact relation tuples.
        max_block_size: Maximum block size for residual H-CEG.
        max_states: Maximum H-CEG DAG states.
        p_values: Lp-norm exponents for the LP.

    Returns:
        Variable ordering as a list of attribute names.
    """
    ordering: list[AttributeName] = []
    chosen: set[AttributeName] = set()
    all_attrs = query.all_attributes

    while len(ordering) < len(all_attrs):
        remaining = sorted(all_attrs - chosen)

        if len(remaining) == 1:
            ordering.append(remaining[0])
            break

        best_attr = remaining[0]
        best_bound = float("inf")

        for candidate in remaining:
            candidate_prefix = chosen | {candidate}
            bound = _lp_residual_bound(
                query,
                relation_data,
                candidate_prefix,
                max_block_size=max_block_size,
                max_states=max_states,
                p_values=p_values,
            )
            if bound < best_bound:
                best_bound = bound
                best_attr = candidate

        ordering.append(best_attr)
        chosen.add(best_attr)

    return ordering


def _lp_residual_bound(
    query: QueryHypergraph,
    relation_data: RelationDataMap,
    prefix: set[AttributeName],
    *,
    max_block_size: int,
    max_states: int,
    p_values: tuple[float, ...],
) -> float:
    """Compute the LP-optimized residual bound after fixing a prefix."""
    from src.utils.types import QueryHypergraph as QueryHG

    prefix_set = _normalize_prefix(query, frozenset(prefix))
    if not prefix_set:
        return estimate_lph_ceg(
            query,
            relation_data,
            max_block_size=max_block_size,
            max_states=max_states,
            p_values=p_values,
        ).estimate

    residual_relations, residual_data = _project_residual_relations(
        query, relation_data, prefix_set
    )
    if not residual_relations:
        return 1.0

    total_bound = 1.0
    for component_relations in _connected_components(residual_relations):
        component_query = QueryHG(
            relations=component_relations,
            name=f"{query.name}_residual",
        )
        component_data = {
            rel.name: residual_data[rel.name] for rel in component_relations
        }
        component_bound = estimate_lph_ceg(
            component_query,
            component_data,
            max_block_size=max_block_size,
            max_states=max_states,
            p_values=p_values,
        ).estimate
        total_bound *= component_bound

    return total_bound
