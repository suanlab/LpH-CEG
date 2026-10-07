from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.hceg import DEFAULT_MAX_STATES
from src.estimators.pessimistic import estimate_pessimistic

if TYPE_CHECKING:
    from src.statistics.block_stats import RelationDataMap
    from src.utils.types import AttributeName, QueryHypergraph


def variable_ordering(
    query: QueryHypergraph,
    relation_data: RelationDataMap,
    *,
    max_block_size: int = 3,
    max_states: int = DEFAULT_MAX_STATES,
) -> list[AttributeName]:
    """Return a deterministic H-CEG-guided variable ordering for WCOJ."""
    estimation = estimate_pessimistic(
        query,
        relation_data,
        max_block_size=max_block_size,
        max_states=max_states,
    )
    hyperpath_blocks = estimation.hyperpath.blocks
    block_attributes = tuple(query.attributes_of(block) for block in hyperpath_blocks)
    relation_coverage = _count_relation_coverage(query)

    chosen_attributes: set[AttributeName] = set()
    ordering: list[AttributeName] = []

    while len(ordering) < len(query.all_attributes):
        remaining_attributes = sorted(query.all_attributes - chosen_attributes)
        next_attribute = min(
            remaining_attributes,
            key=lambda attribute: _priority_tuple(
                attribute,
                block_attributes,
                chosen_attributes,
                relation_coverage,
            ),
        )
        ordering.append(next_attribute)
        chosen_attributes.add(next_attribute)

    return ordering


def _priority_tuple(
    attribute: AttributeName,
    block_attributes: tuple[frozenset[AttributeName], ...],
    chosen_attributes: set[AttributeName],
    relation_coverage: dict[AttributeName, int],
) -> tuple[int, int, int, AttributeName]:
    remaining_block_indices = _remaining_block_indices(
        block_attributes, chosen_attributes
    )
    containing_indices = [
        index
        for index in remaining_block_indices
        if attribute in block_attributes[index]
    ]

    earliest_block_index = len(block_attributes)
    if containing_indices:
        earliest_block_index = containing_indices[0]

    return (
        earliest_block_index,
        -len(containing_indices),
        -relation_coverage[attribute],
        attribute,
    )


def _remaining_block_indices(
    block_attributes: tuple[frozenset[AttributeName], ...],
    chosen_attributes: set[AttributeName],
) -> list[int]:
    return [
        index
        for index, attributes in enumerate(block_attributes)
        if not attributes <= chosen_attributes
    ]


def _count_relation_coverage(query: QueryHypergraph) -> dict[AttributeName, int]:
    coverage = {attribute: 0 for attribute in query.all_attributes}
    for relation in query.relations:
        for attribute in relation.attributes:
            coverage[attribute] += 1
    return coverage
