"""Binary CEG_M baseline via shortest paths on attribute subsets.

This module implements the pessimistic Cardinality Estimation Graph (CEG_M)
for binary queries. Vertices are attribute subsets and each directed edge adds
exactly one attribute using a binary relation already present in the query.

Edge weights are stored in log2-space using Task 4 binary degree statistics.
The shortest path from the empty attribute set to the full attribute set is the
MOLP optimum from Theorem 5.1 of the CEG paper, so the final upper bound is
``2 ** shortest_path_weight``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import TypeAlias

from scipy.optimize import linprog

from src.statistics.binary_stats import BinaryDegreeStats
from src.utils.types import AttributeName, QueryHypergraph, Relation, RelationName

AttributeSet: TypeAlias = frozenset[AttributeName]
BinaryDegreeKey: TypeAlias = tuple[AttributeSet, AttributeSet]
RelationDegreeStats: TypeAlias = Mapping[BinaryDegreeKey, BinaryDegreeStats]
DegreeStatsByRelation: TypeAlias = Mapping[RelationName, RelationDegreeStats]


@dataclass(frozen=True)
class CEGMEdge:
    """A binary CEG_M edge between attribute subsets.

    Attributes:
        source: Source attribute subset.
        target: Target attribute subset with one additional attribute.
        relation_name: Relation used for the binary extension.
        conditioning_attrs: Attributes from the source that are also present in
            the relation and condition the degree statistic.
        added_attribute: The newly introduced attribute.
        weight: Edge weight in log2-space.
    """

    source: AttributeSet
    target: AttributeSet
    relation_name: RelationName
    conditioning_attrs: AttributeSet
    added_attribute: AttributeName
    weight: float


@dataclass(frozen=True)
class CEGMGraph:
    """Explicit binary CEG_M DAG."""

    query_name: str
    vertices: tuple[AttributeSet, ...]
    edges: tuple[CEGMEdge, ...]
    source: AttributeSet
    sink: AttributeSet


@dataclass(frozen=True)
class CEGMPath:
    """Shortest path solution in the binary CEG_M graph."""

    vertices: tuple[AttributeSet, ...]
    edges: tuple[CEGMEdge, ...]
    total_weight: float

    @property
    def bound(self) -> float:
        """Return the pessimistic estimate in the original scale."""
        return float(2.0**self.total_weight)


@dataclass(frozen=True)
class MOLPSolution:
    """Optimal solution of the MOLP verification LP."""

    optimal_value: float
    subset_values: dict[AttributeSet, float]

    @property
    def bound(self) -> float:
        """Return the LP optimum in the original scale."""
        return float(2.0**self.optimal_value)


@dataclass(frozen=True)
class CEGMEstimate:
    """Full CEG_M estimation result for one query."""

    graph: CEGMGraph
    shortest_path: CEGMPath
    bound: float
    molp_solution: MOLPSolution | None = None


def enumerate_attribute_subsets(
    attributes: frozenset[AttributeName],
) -> tuple[AttributeSet, ...]:
    """Enumerate all attribute subsets in DAG topological order.

    The order is by subset size and then lexicographically, which is a valid
    topological order because every CEG_M edge adds exactly one attribute.

    Args:
        attributes: Query attribute universe.

    Returns:
        All subsets of ``attributes`` in a deterministic order.
    """
    ordered_attributes = tuple(sorted(attributes))
    subsets: list[AttributeSet] = []
    for size in range(len(ordered_attributes) + 1):
        for subset in combinations(ordered_attributes, size):
            subsets.append(frozenset(subset))
    return tuple(subsets)


def build_ceg_m_graph(
    query: QueryHypergraph,
    degree_stats_by_relation: DegreeStatsByRelation,
) -> CEGMGraph:
    """Build the explicit binary CEG_M DAG for a query.

    Args:
        query: Binary query hypergraph.
        degree_stats_by_relation: Per-relation binary degree statistics from
            Task 4. Each relation must provide entries keyed by
            ``(conditioning_attrs, frozenset({added_attribute}))``.

    Returns:
        Explicit CEG_M graph with all attribute-subset vertices and binary
        extension edges.

    Raises:
        ValueError: If a relation in the query is not binary.
        KeyError: If a relation is missing degree statistics required by a CEG
            edge.
    """
    vertices = enumerate_attribute_subsets(query.all_attributes)
    edges: list[CEGMEdge] = []

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
                        "Missing binary degree statistics for relation "
                        f"{relation.name!r}, conditioning="
                        f"{sorted(conditioning_attrs)!r}, "
                        f"target={[added_attribute]!r}"
                    )
                    raise KeyError(msg)
                stats = relation_stats[key]
                edges.append(
                    CEGMEdge(
                        source=source,
                        target=target,
                        relation_name=relation.name,
                        conditioning_attrs=conditioning_attrs,
                        added_attribute=added_attribute,
                        weight=stats.log_max_degree,
                    )
                )

    return CEGMGraph(
        query_name=query.name,
        vertices=vertices,
        edges=tuple(edges),
        source=frozenset(),
        sink=query.all_attributes,
    )


def shortest_path_ceg_m(graph: CEGMGraph) -> CEGMPath:
    """Compute the minimum-weight path from ``∅`` to all attributes.

    Args:
        graph: Explicit binary CEG_M DAG.

    Returns:
        Shortest path in log2-space.

    Raises:
        ValueError: If the sink is unreachable.
    """
    outgoing: dict[AttributeSet, list[CEGMEdge]] = {
        vertex: [] for vertex in graph.vertices
    }
    for edge in graph.edges:
        outgoing[edge.source].append(edge)

    distances = {vertex: math.inf for vertex in graph.vertices}
    predecessors: dict[AttributeSet, CEGMEdge] = {}
    distances[graph.source] = 0.0

    for vertex in graph.vertices:
        current_distance = distances[vertex]
        if math.isinf(current_distance):
            continue
        for edge in outgoing[vertex]:
            candidate = current_distance + edge.weight
            if candidate < distances[edge.target]:
                distances[edge.target] = candidate
                predecessors[edge.target] = edge

    if math.isinf(distances[graph.sink]):
        msg = f"CEG_M sink is unreachable for query {graph.query_name!r}"
        raise ValueError(msg)

    path_edges: list[CEGMEdge] = []
    path_vertices: list[AttributeSet] = [graph.sink]
    current = graph.sink
    while current != graph.source:
        edge = predecessors[current]
        path_edges.append(edge)
        current = edge.source
        path_vertices.append(current)

    path_edges.reverse()
    path_vertices.reverse()
    return CEGMPath(
        vertices=tuple(path_vertices),
        edges=tuple(path_edges),
        total_weight=distances[graph.sink],
    )


def solve_molp_linprog(graph: CEGMGraph) -> MOLPSolution:
    """Solve the MOLP verification LP with SciPy.

    The LP uses one variable ``s_X`` per attribute subset ``X``. It enforces:

    - ``s_∅ = 0``
    - monotonicity ``s_X <= s_{X U {a}}``
    - CEG_M edge constraints ``s_target <= s_source + weight``

    Maximizing ``s_A`` over these constraints reproduces the shortest-path
    optimum from Theorem 5.1 in the binary setting.

    Args:
        graph: Explicit binary CEG_M DAG.

    Returns:
        Optimal LP solution in log2-space.

    Raises:
        RuntimeError: If ``scipy.optimize.linprog`` fails.
    """
    vertex_to_index = {vertex: index for index, vertex in enumerate(graph.vertices)}
    num_vertices = len(graph.vertices)
    a_ub: list[list[float]] = []
    b_ub: list[float] = []

    for source in graph.vertices:
        for attribute in sorted(graph.sink - source):
            target = frozenset(set(source) | {attribute})
            row = [0.0] * num_vertices
            row[vertex_to_index[source]] = 1.0
            row[vertex_to_index[target]] = -1.0
            a_ub.append(row)
            b_ub.append(0.0)

    for edge in graph.edges:
        row = [0.0] * num_vertices
        row[vertex_to_index[edge.target]] = 1.0
        row[vertex_to_index[edge.source]] = -1.0
        a_ub.append(row)
        b_ub.append(edge.weight)

    a_eq = [[0.0] * num_vertices]
    a_eq[0][vertex_to_index[graph.source]] = 1.0
    b_eq = [0.0]

    objective = [0.0] * num_vertices
    objective[vertex_to_index[graph.sink]] = -1.0

    result = linprog(
        c=objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(None, None)] * num_vertices,
        method="highs",
    )
    if not result.success:
        msg = f"linprog failed for query {graph.query_name!r}: {result.message}"
        raise RuntimeError(msg)

    subset_values = {
        vertex: float(result.x[vertex_to_index[vertex]]) for vertex in graph.vertices
    }
    return MOLPSolution(
        optimal_value=subset_values[graph.sink],
        subset_values=subset_values,
    )


def estimate_ceg_m(
    query: QueryHypergraph,
    degree_stats_by_relation: DegreeStatsByRelation,
    *,
    verify_equivalence: bool = False,
    atol: float = 1e-9,
) -> CEGMEstimate:
    """Estimate a query bound with the binary CEG_M baseline.

    Args:
        query: Binary query hypergraph.
        degree_stats_by_relation: Per-relation Task 4 degree statistics.
        verify_equivalence: Whether to solve the MOLP verification LP and
            compare it to the shortest-path optimum.
        atol: Absolute tolerance for the shortest-path / LP comparison.

    Returns:
        CEG_M estimate containing the graph, shortest path, and optional LP
        verification result.

    Raises:
        ValueError: If the LP and shortest path differ beyond ``atol``.
    """
    graph = build_ceg_m_graph(query, degree_stats_by_relation)
    shortest_path = shortest_path_ceg_m(graph)
    molp_solution: MOLPSolution | None = None

    if verify_equivalence:
        molp_solution = solve_molp_linprog(graph)
        if not math.isclose(
            shortest_path.total_weight, molp_solution.optimal_value, abs_tol=atol
        ):
            msg = (
                f"MOLP mismatch for query {query.name!r}: shortest path "
                f"{shortest_path.total_weight:.12f} vs LP "
                f"{molp_solution.optimal_value:.12f}"
            )
            raise ValueError(msg)

    return CEGMEstimate(
        graph=graph,
        shortest_path=shortest_path,
        bound=shortest_path.bound,
        molp_solution=molp_solution,
    )


def verify_molp_equivalence(
    query: QueryHypergraph,
    degree_stats_by_relation: DegreeStatsByRelation,
    *,
    atol: float = 1e-9,
) -> tuple[float, float]:
    """Return shortest-path and LP optima after validating equivalence.

    Args:
        query: Binary query hypergraph.
        degree_stats_by_relation: Per-relation Task 4 degree statistics.
        atol: Absolute tolerance for the equality check.

    Returns:
        Pair ``(shortest_path_value, molp_value)`` in log2-space.
    """
    estimate = estimate_ceg_m(
        query,
        degree_stats_by_relation,
        verify_equivalence=True,
        atol=atol,
    )
    if estimate.molp_solution is None:
        msg = "Expected MOLP verification result to be available"
        raise ValueError(msg)
    return estimate.shortest_path.total_weight, estimate.molp_solution.optimal_value


def _validate_binary_relation(relation: Relation) -> None:
    if len(relation.attributes) != 2:
        msg = (
            "CEG_M baseline only supports binary relations, got relation "
            f"{relation.name!r} with attributes {sorted(relation.attributes)!r}"
        )
        raise ValueError(msg)
