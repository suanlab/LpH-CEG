"""Binary CEG_O optimistic estimator with 3x3 configuration space.

This module implements the optimistic Cardinality Estimation Graph (CEG_O)
for binary queries. It uses average degree statistics (``avg_degree``) instead
of max degree, and provides 9 configurations from the cross product of
3 edge-aggregation methods and 3 hop strategies.

The 3x3 configuration space:

    +-----------+----------+----------+----------+
    |           | min-aggr | avg-aggr | max-aggr |
    +-----------+----------+----------+----------+
    | min-hop   |   ...    |   ...    |   ...    |
    | max-hop   |   ...    |   ...    |   ...    |
    | all-hops  |   ...    |   ...    |   ...    |
    +-----------+----------+----------+----------+

- **Edge aggregation** (columns): When multiple relations provide edges
  between the same attribute subset pair, collapse their ``avg_degree``
  weights using min, mean, or max.
- **Hop strategy** (rows): In the collapsed graph, select paths from
  the empty set to the full attribute set using minimum-product,
  maximum-product, or average over all path products.

Primary configurations (empirical from Salihoglu et al.):
- ``max-hop-max``: Best for acyclic queries and triangle cycles.
- ``max-hop-min``: Best for large cycles (size > 3).

References:
    Salihoglu et al., "Cardinality Estimation Graphs", CACM 2026.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import TypeAlias

from src.statistics.binary_stats import BinaryDegreeStats
from src.utils.types import AttributeName, QueryHypergraph, Relation, RelationName

logger = logging.getLogger(__name__)

AttributeSet: TypeAlias = frozenset[AttributeName]
BinaryDegreeKey: TypeAlias = tuple[AttributeSet, AttributeSet]
RelationDegreeStats: TypeAlias = Mapping[BinaryDegreeKey, BinaryDegreeStats]
DegreeStatsByRelation: TypeAlias = Mapping[RelationName, RelationDegreeStats]

HOP_STRATEGIES: frozenset[str] = frozenset({"min-hop", "max-hop", "all-hops"})
AGGREGATIONS: frozenset[str] = frozenset({"min-aggr", "avg-aggr", "max-aggr"})

_MAX_PATHS: int = 100_000


@dataclass(frozen=True)
class CEGOEdge:
    """A binary CEG_O edge between attribute subsets.

    Attributes:
        source: Source attribute subset.
        target: Target attribute subset with one additional attribute.
        relation_name: Relation used for the binary extension.
        conditioning_attrs: Attributes from the source that condition
            the degree statistic.
        added_attribute: The newly introduced attribute.
        weight: Average degree (raw, not log-space).
    """

    source: AttributeSet
    target: AttributeSet
    relation_name: RelationName
    conditioning_attrs: AttributeSet
    added_attribute: AttributeName
    weight: float


@dataclass(frozen=True)
class CEGOGraph:
    """Explicit binary CEG_O DAG with multi-edges.

    The graph may contain multiple edges between the same source-target
    pair, one per relation that can provide the extension.  Edge weights
    are raw average degrees.

    Attributes:
        query_name: Name of the query hypergraph.
        vertices: All attribute subsets in topological order.
        edges: All edges (possibly multiple per source-target pair).
        source: Empty attribute set (start vertex).
        sink: Full attribute set (target vertex).
    """

    query_name: str
    vertices: tuple[AttributeSet, ...]
    edges: tuple[CEGOEdge, ...]
    source: AttributeSet
    sink: AttributeSet


@dataclass(frozen=True)
class CEGOEstimate:
    """CEG_O optimistic cardinality estimate.

    Attributes:
        bound: The estimated upper bound on join cardinality.
        hop_strategy: Path selection strategy used.
        aggregation: Edge aggregation method used.
        path_count: Number of paths considered in the estimate.
    """

    bound: float
    hop_strategy: str
    aggregation: str
    path_count: int


def _validate_binary_relation(relation: Relation) -> None:
    """Validate that a relation is binary (exactly 2 attributes).

    Raises:
        ValueError: If the relation does not have exactly 2 attributes.
    """
    if len(relation.attributes) != 2:
        msg = (
            "CEG_O baseline only supports binary relations, got relation "
            f"{relation.name!r} with attributes {sorted(relation.attributes)!r}"
        )
        raise ValueError(msg)


def _enumerate_attribute_subsets(
    attributes: frozenset[AttributeName],
) -> tuple[AttributeSet, ...]:
    """Enumerate all attribute subsets in DAG topological order.

    Subsets are sorted by cardinality then lexicographically, which is a
    valid topological order because every edge adds exactly one attribute.

    Args:
        attributes: Query attribute universe.

    Returns:
        All subsets of ``attributes`` in deterministic order.
    """
    ordered = tuple(sorted(attributes))
    subsets: list[AttributeSet] = []
    for size in range(len(ordered) + 1):
        for subset in combinations(ordered, size):
            subsets.append(frozenset(subset))
    return tuple(subsets)


def build_ceg_o_graph(
    query: QueryHypergraph,
    degree_stats_by_relation: DegreeStatsByRelation,
) -> CEGOGraph:
    """Build the explicit binary CEG_O DAG for a query.

    Same vertex/edge structure as CEG_M (attribute-subset vertices,
    single-attribute extension edges) but with ``avg_degree`` weights
    instead of log-space ``log_max_degree`` weights.

    Args:
        query: Binary query hypergraph.
        degree_stats_by_relation: Per-relation binary degree statistics
            from :func:`~src.statistics.binary_stats.compute_all_binary_degrees`.

    Returns:
        Explicit CEG_O graph with ``avg_degree`` edge weights.

    Raises:
        ValueError: If a relation in the query is not binary.
        KeyError: If a relation is missing from ``degree_stats_by_relation``
            or if the required degree-stat key is absent.
    """
    vertices = _enumerate_attribute_subsets(query.all_attributes)
    edges: list[CEGOEdge] = []

    for relation in query.relations:
        _validate_binary_relation(relation)
        if relation.name not in degree_stats_by_relation:
            msg = f"Missing degree statistics for relation {relation.name!r}"
            raise KeyError(msg)

    for source in vertices:
        for relation in query.relations:
            relation_stats = degree_stats_by_relation[relation.name]
            for added_attribute in sorted(relation.attributes - source):
                target = frozenset(set(source) | {added_attribute})
                conditioning_attrs = frozenset(
                    source & (relation.attributes - {added_attribute})
                )
                key = (conditioning_attrs, frozenset({added_attribute}))
                if key not in relation_stats:
                    msg = (
                        f"Missing binary degree statistics for relation "
                        f"{relation.name!r}, conditioning="
                        f"{sorted(conditioning_attrs)!r}, "
                        f"target={[added_attribute]!r}"
                    )
                    raise KeyError(msg)
                stats = relation_stats[key]
                edges.append(
                    CEGOEdge(
                        source=source,
                        target=target,
                        relation_name=relation.name,
                        conditioning_attrs=conditioning_attrs,
                        added_attribute=added_attribute,
                        weight=stats.avg_degree,
                    )
                )

    logger.debug(
        "CEG_O graph for %r: %d vertices, %d edges",
        query.name,
        len(vertices),
        len(edges),
    )
    return CEGOGraph(
        query_name=query.name,
        vertices=vertices,
        edges=tuple(edges),
        source=frozenset(),
        sink=query.all_attributes,
    )


def _collapse_edge_weight(weights: list[float], aggregation: str) -> float:
    """Collapse multiple edge weights into a single weight.

    Args:
        weights: Non-empty list of ``avg_degree`` values from different
            relations providing the same attribute extension.
        aggregation: One of ``"min-aggr"``, ``"avg-aggr"``, ``"max-aggr"``.

    Returns:
        Collapsed weight.
    """
    if aggregation == "min-aggr":
        return min(weights)
    if aggregation == "avg-aggr":
        return sum(weights) / len(weights)
    return max(weights)


def _build_collapsed_adjacency(
    graph: CEGOGraph,
    aggregation: str,
) -> dict[AttributeSet, dict[AttributeSet, float]]:
    """Build adjacency map with collapsed multi-edge weights.

    Groups edges by ``(source, target)`` and applies the aggregation
    strategy to produce a single weight per pair.

    Args:
        graph: CEG_O graph with potentially multiple edges per pair.
        aggregation: Edge collapse strategy.

    Returns:
        Mapping ``vertex -> {neighbor: collapsed_weight}``.
    """
    multi_edges: dict[tuple[AttributeSet, AttributeSet], list[float]] = {}
    for edge in graph.edges:
        pair = (edge.source, edge.target)
        if pair not in multi_edges:
            multi_edges[pair] = []
        multi_edges[pair].append(edge.weight)

    adjacency: dict[AttributeSet, dict[AttributeSet, float]] = {
        v: {} for v in graph.vertices
    }
    for (source, target), weights in multi_edges.items():
        adjacency[source][target] = _collapse_edge_weight(weights, aggregation)

    return adjacency


def _shortest_path_product(
    graph: CEGOGraph,
    adjacency: dict[AttributeSet, dict[AttributeSet, float]],
) -> tuple[float, int]:
    """Find the minimum-product path via DAG relaxation.

    Args:
        graph: CEG_O graph (vertices in topological order).
        adjacency: Collapsed adjacency map.

    Returns:
        ``(minimum_product, 1)`` — path count is always 1.

    Raises:
        ValueError: If the sink is unreachable.
    """
    dist: dict[AttributeSet, float] = {}
    dist[graph.source] = 1.0

    for vertex in graph.vertices:
        if vertex not in dist:
            continue
        current = dist[vertex]
        for target, weight in adjacency[vertex].items():
            candidate = current * weight
            if target not in dist or candidate < dist[target]:
                dist[target] = candidate

    if graph.sink not in dist:
        msg = f"CEG_O sink is unreachable for query {graph.query_name!r}"
        raise ValueError(msg)

    return dist[graph.sink], 1


def _longest_path_product(
    graph: CEGOGraph,
    adjacency: dict[AttributeSet, dict[AttributeSet, float]],
) -> tuple[float, int]:
    """Find the maximum-product path via DAG relaxation.

    Args:
        graph: CEG_O graph (vertices in topological order).
        adjacency: Collapsed adjacency map.

    Returns:
        ``(maximum_product, 1)`` — path count is always 1.

    Raises:
        ValueError: If the sink is unreachable.
    """
    dist: dict[AttributeSet, float] = {}
    dist[graph.source] = 1.0

    for vertex in graph.vertices:
        if vertex not in dist:
            continue
        current = dist[vertex]
        for target, weight in adjacency[vertex].items():
            candidate = current * weight
            if target not in dist or candidate > dist[target]:
                dist[target] = candidate

    if graph.sink not in dist:
        msg = f"CEG_O sink is unreachable for query {graph.query_name!r}"
        raise ValueError(msg)

    return dist[graph.sink], 1


def _all_paths_estimates(
    graph: CEGOGraph,
    adjacency: dict[AttributeSet, dict[AttributeSet, float]],
) -> list[float]:
    """Enumerate all source-to-sink path products via iterative DFS.

    Args:
        graph: CEG_O graph.
        adjacency: Collapsed adjacency map.

    Returns:
        List of product estimates, one per path.

    Raises:
        ValueError: If the number of paths exceeds ``_MAX_PATHS``.
    """
    estimates: list[float] = []
    stack: list[tuple[AttributeSet, float]] = [(graph.source, 1.0)]

    while stack:
        vertex, product = stack.pop()
        if vertex == graph.sink:
            estimates.append(product)
            if len(estimates) > _MAX_PATHS:
                msg = (
                    f"Path enumeration exceeded {_MAX_PATHS} paths "
                    f"for query {graph.query_name!r}; query may be too large "
                    f"for the all-hops strategy"
                )
                raise ValueError(msg)
            continue
        for target, weight in adjacency[vertex].items():
            stack.append((target, product * weight))

    return estimates


def estimate_ceg_o(
    query: QueryHypergraph,
    degree_stats_by_relation: DegreeStatsByRelation,
    hop_strategy: str = "max-hop",
    aggregation: str = "max-aggr",
) -> CEGOEstimate:
    """Compute CEG_O optimistic estimate for a given configuration.

    The estimate combines two independent dimensions:

    1. **Edge aggregation** (``aggregation``): When multiple relations
       provide edges between the same attribute-subset pair, collapse
       their ``avg_degree`` values using min, mean, or max.

    2. **Hop strategy** (``hop_strategy``): In the collapsed graph,
       select paths from the empty set to the full attribute set:

       - ``"min-hop"``: minimum-product path (tightest bound).
       - ``"max-hop"``: maximum-product path (loosest bound).
       - ``"all-hops"``: average over all path products.

    Primary configurations (empirically best):

    - ``max-hop`` / ``max-aggr``: Best for acyclic queries and triangles.
    - ``max-hop`` / ``min-aggr``: Best for large cycles (size > 3).

    Args:
        query: Binary query hypergraph.
        degree_stats_by_relation: Per-relation binary degree statistics.
        hop_strategy: Path selection strategy.
        aggregation: Edge-weight collapsing strategy.

    Returns:
        CEG_O estimate with configuration metadata.

    Raises:
        ValueError: If ``hop_strategy`` or ``aggregation`` is invalid,
            or if the sink is unreachable.
        KeyError: If required degree statistics are missing.
    """
    if hop_strategy not in HOP_STRATEGIES:
        msg = (
            f"Invalid hop_strategy {hop_strategy!r}, "
            f"expected one of {sorted(HOP_STRATEGIES)!r}"
        )
        raise ValueError(msg)
    if aggregation not in AGGREGATIONS:
        msg = (
            f"Invalid aggregation {aggregation!r}, "
            f"expected one of {sorted(AGGREGATIONS)!r}"
        )
        raise ValueError(msg)

    graph = build_ceg_o_graph(query, degree_stats_by_relation)
    adjacency = _build_collapsed_adjacency(graph, aggregation)

    bound: float
    path_count: int

    if hop_strategy == "min-hop":
        bound, path_count = _shortest_path_product(graph, adjacency)
    elif hop_strategy == "max-hop":
        bound, path_count = _longest_path_product(graph, adjacency)
    else:
        estimates = _all_paths_estimates(graph, adjacency)
        if not estimates:
            msg = f"No paths found for query {query.name!r}"
            raise ValueError(msg)
        bound = sum(estimates) / len(estimates)
        path_count = len(estimates)

    logger.debug(
        "CEG_O [%s/%s] query=%r: bound=%.4f, paths=%d",
        hop_strategy,
        aggregation,
        query.name,
        bound,
        path_count,
    )
    return CEGOEstimate(
        bound=bound,
        hop_strategy=hop_strategy,
        aggregation=aggregation,
        path_count=path_count,
    )


def estimate_all_configurations(
    query: QueryHypergraph,
    degree_stats_by_relation: DegreeStatsByRelation,
) -> dict[tuple[str, str], CEGOEstimate]:
    """Compute CEG_O estimates for all 9 configurations.

    Iterates over the full 3x3 space of ``(hop_strategy, aggregation)``
    combinations and returns a dictionary keyed by the pair.

    Args:
        query: Binary query hypergraph.
        degree_stats_by_relation: Per-relation binary degree statistics.

    Returns:
        Dictionary mapping ``(hop_strategy, aggregation)`` to the
        corresponding :class:`CEGOEstimate`.
    """
    results: dict[tuple[str, str], CEGOEstimate] = {}
    for hop in sorted(HOP_STRATEGIES):
        for aggr in sorted(AGGREGATIONS):
            results[(hop, aggr)] = estimate_ceg_o(
                query, degree_stats_by_relation, hop, aggr
            )
    return results
