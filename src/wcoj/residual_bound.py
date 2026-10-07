from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, TypeAlias

from src.core.hceg import DEFAULT_MAX_STATES
from src.estimators.pessimistic import estimate_pessimistic
from src.utils.types import AttributeName, QueryHypergraph, Relation, RelationName

if TYPE_CHECKING:
    from collections.abc import Collection

    from src.statistics.block_stats import RelationDataMap

ResidualRelationRows: TypeAlias = tuple[tuple[str, ...], ...]
ResidualRelationPayload: TypeAlias = tuple[
    tuple[AttributeName, ...], ResidualRelationRows
]
ResidualRelationDataMap: TypeAlias = dict[RelationName, ResidualRelationPayload]


def build_residual_query(
    query: QueryHypergraph,
    relation_data: RelationDataMap,
    prefix: Collection[AttributeName],
) -> tuple[QueryHypergraph | None, RelationDataMap]:
    """Build the structural residual query after fixing a variable prefix.

    Args:
        query: Original query hypergraph.
        relation_data: Exact relation tuples aligned with ``query``.
        prefix: Fixed prefix attributes to project away.

    Returns:
        A residual query/data pair. The query is ``None`` when all relations are
        fully covered by the prefix.

    Raises:
        KeyError: If relation data is missing for a query relation.
        ValueError: If ``prefix`` contains attributes outside the query, if
            relation payloads do not match the query schema, or if the residual
            projection is disconnected.
    """
    residual_relations, residual_data = _project_residual_relations(
        query,
        relation_data,
        prefix,
    )
    if not residual_relations:
        return None, residual_data

    if len(_connected_components(residual_relations)) > 1:
        msg = "Residual projection is disconnected; use residual_bound() instead"
        raise ValueError(msg)

    residual_query = QueryHypergraph(
        relations=residual_relations,
        name=_residual_query_name(query.name),
    )
    return residual_query, residual_data


def residual_bound(
    query: QueryHypergraph,
    relation_data: RelationDataMap,
    prefix: Collection[AttributeName],
    *,
    max_block_size: int = 3,
    max_states: int = DEFAULT_MAX_STATES,
) -> float:
    """Compute a prefix-conditioned structural residual upper bound for WCOJ.

    Args:
        query: Original query hypergraph.
        relation_data: Exact relation tuples aligned with ``query``.
        prefix: Fixed prefix attributes to project away structurally.
        max_block_size: Maximum block size used during H-CEG block enumeration.
        max_states: Maximum number of reachable H-CEG states to materialize.

    Returns:
        A pessimistic H-CEG_M upper bound on the residual query after removing
        the fixed prefix attributes.

    Raises:
        KeyError: If relation data is missing for a query relation.
        ValueError: If ``prefix`` contains attributes outside the query or if a
            relation payload does not match the query schema.
    """
    prefix_set = _normalize_prefix(query, prefix)
    if not prefix_set:
        return estimate_pessimistic(
            query,
            relation_data,
            max_block_size=max_block_size,
            max_states=max_states,
        ).estimate

    residual_relations, residual_data = _project_residual_relations(
        query,
        relation_data,
        prefix_set,
    )
    if not residual_relations:
        return 1.0

    total_bound = 1.0
    for component_relations in _connected_components(residual_relations):
        component_query = QueryHypergraph(
            relations=component_relations,
            name=_residual_query_name(query.name),
        )
        component_data = {
            relation.name: residual_data[relation.name]
            for relation in component_relations
        }
        total_bound *= estimate_pessimistic(
            component_query,
            component_data,
            max_block_size=max_block_size,
            max_states=max_states,
        ).estimate

    return total_bound


def _project_residual_relations(
    query: QueryHypergraph,
    relation_data: RelationDataMap,
    prefix: Collection[AttributeName],
) -> tuple[tuple[Relation, ...], RelationDataMap]:
    prefix_set = _normalize_prefix(query, prefix)
    residual_relations: list[Relation] = []
    residual_data: ResidualRelationDataMap = {}

    for relation in query.relations:
        if relation.name not in relation_data:
            msg = f"Missing relation data for {relation.name!r}"
            raise KeyError(msg)

        attribute_order, rows = relation_data[relation.name]
        _validate_payload_against_relation(relation, attribute_order, rows)

        remaining_attrs = frozenset(relation.attributes - prefix_set)
        if not remaining_attrs:
            continue

        projected_order = tuple(
            attribute for attribute in attribute_order if attribute in remaining_attrs
        )
        index_map = {
            attribute: index for index, attribute in enumerate(attribute_order)
        }
        projected_indices = [index_map[attribute] for attribute in projected_order]
        projected_rows = tuple(
            sorted({tuple(row[index] for index in projected_indices) for row in rows})
        )

        residual_relations.append(
            Relation(
                name=relation.name,
                attributes=remaining_attrs,
                cardinality=len(projected_rows),
            )
        )
        residual_data[relation.name] = (projected_order, projected_rows)

    return tuple(residual_relations), residual_data


def _normalize_prefix(
    query: QueryHypergraph,
    prefix: Collection[AttributeName],
) -> frozenset[AttributeName]:
    prefix_set = frozenset(prefix)
    if prefix_set <= query.all_attributes:
        return prefix_set
    invalid_attributes = sorted(prefix_set - query.all_attributes)
    msg = f"Prefix attributes must be contained in the query: {invalid_attributes!r}"
    raise ValueError(msg)


def _validate_payload_against_relation(
    relation: Relation,
    attribute_order: tuple[AttributeName, ...],
    rows: Collection[tuple[str, ...]],
) -> None:
    if frozenset(attribute_order) != relation.attributes:
        msg = (
            f"Relation data for {relation.name!r} has attributes "
            f"{attribute_order!r}, expected {sorted(relation.attributes)!r}"
        )
        raise ValueError(msg)

    if len(attribute_order) != len(set(attribute_order)):
        msg = f"Relation data for {relation.name!r} has duplicate attributes"
        raise ValueError(msg)

    expected_width = len(attribute_order)
    for row in rows:
        if len(row) == expected_width:
            continue
        msg = (
            f"Tuple width mismatch for relation {relation.name!r}: expected "
            f"{expected_width}, got {len(row)}"
        )
        raise ValueError(msg)


def _connected_components(
    relations: tuple[Relation, ...],
) -> tuple[tuple[Relation, ...], ...]:
    relation_by_name: dict[RelationName, Relation] = {
        relation.name: relation for relation in relations
    }
    adjacency: dict[RelationName, set[RelationName]] = {
        relation.name: set() for relation in relations
    }

    for relation in relations:
        for other in relations:
            if relation.name == other.name:
                continue
            if relation.attributes & other.attributes:
                adjacency[relation.name].add(other.name)

    remaining_names = set(relation_by_name)
    components: list[tuple[Relation, ...]] = []
    while remaining_names:
        start = min(remaining_names)
        queue: deque[RelationName] = deque([start])
        component_names: list[RelationName] = []
        remaining_names.remove(start)

        while queue:
            current = queue.popleft()
            component_names.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor not in remaining_names:
                    continue
                remaining_names.remove(neighbor)
                queue.append(neighbor)

        components.append(
            tuple(relation_by_name[name] for name in sorted(component_names))
        )

    components.sort(
        key=lambda component: tuple(relation.name for relation in component)
    )
    return tuple(components)


def _residual_query_name(query_name: str) -> str:
    if not query_name:
        return "residual"
    return f"{query_name}_residual"
