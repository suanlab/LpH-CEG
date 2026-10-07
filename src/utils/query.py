"""Query builder functions and standard query templates for H-CEG.

This module provides factory functions to construct common query patterns
used in cardinality estimation benchmarks and tests.
"""

from __future__ import annotations

from src.utils.types import QueryHypergraph, Relation


def triangle_query() -> QueryHypergraph:
    """Create a triangle query: R(a,b), S(b,c), T(a,c).

    This is a classic 3-relation query with a cycle structure.
    All three attributes {a, b, c} are present.

    Returns:
        QueryHypergraph: Triangle query with 3 relations and 3 attributes.

    Example:
        >>> q = triangle_query()
        >>> len(q.relations)
        3
        >>> q.all_attributes
        frozenset({'a', 'b', 'c'})
    """
    return QueryHypergraph(
        relations=(
            Relation("R", frozenset({"a", "b"}), 100),
            Relation("S", frozenset({"b", "c"}), 200),
            Relation("T", frozenset({"a", "c"}), 150),
        ),
        name="triangle",
    )


def star_query(n: int) -> QueryHypergraph:
    """Create a star query with n+1 relations.

    The star has one central relation R(a, b_1) and n leaf relations
    S_i(a, c_i) for i=1..n. All relations share the central attribute 'a'.

    Args:
        n: Number of leaf relations (must be >= 1).

    Returns:
        QueryHypergraph: Star query with n+1 relations.

    Raises:
        ValueError: If n < 1.

    Example:
        >>> q = star_query(3)
        >>> len(q.relations)
        4
        >>> q.all_attributes
        frozenset({'a', 'b', 'c1', 'c2', 'c3'})
    """
    if n < 1:
        msg = "Star query must have at least 1 leaf relation (n >= 1)"
        raise ValueError(msg)

    relations: list[Relation] = [
        Relation("R", frozenset({"a", "b"}), 100),
    ]

    for i in range(1, n + 1):
        leaf_attr = f"c{i}"
        relations.append(Relation(f"S{i}", frozenset({"a", leaf_attr}), 100 + 50 * i))

    return QueryHypergraph(
        relations=tuple(relations),
        name=f"star_{n}",
    )


def path_query(n: int) -> QueryHypergraph:
    """Create a path query with n relations forming a linear chain.

    Relations form a chain: R_1(a_1, a_2), R_2(a_2, a_3), ..., R_n(a_n, a_{n+1}).
    Each consecutive pair shares exactly one attribute.

    Args:
        n: Number of relations in the path (must be >= 1).

    Returns:
        QueryHypergraph: Path query with n relations.

    Raises:
        ValueError: If n < 1.

    Example:
        >>> q = path_query(3)
        >>> len(q.relations)
        3
        >>> q.all_attributes
        frozenset({'a1', 'a2', 'a3', 'a4'})
    """
    if n < 1:
        msg = "Path query must have at least 1 relation (n >= 1)"
        raise ValueError(msg)

    relations: list[Relation] = []

    for i in range(1, n + 1):
        attr1 = f"a{i}"
        attr2 = f"a{i + 1}"
        relations.append(Relation(f"R{i}", frozenset({attr1, attr2}), 100 + 50 * i))

    return QueryHypergraph(
        relations=tuple(relations),
        name=f"path_{n}",
    )


def cycle_query(n: int) -> QueryHypergraph:
    """Create a cycle query with n relations forming a cycle.

    Relations form a cycle: R_1(a_1, a_2), R_2(a_2, a_3), ..., R_n(a_n, a_1).
    The last relation closes the cycle by connecting back to the first attribute.

    Args:
        n: Number of relations in the cycle (must be >= 3).

    Returns:
        QueryHypergraph: Cycle query with n relations.

    Raises:
        ValueError: If n < 3.

    Example:
        >>> q = cycle_query(4)
        >>> len(q.relations)
        4
        >>> q.all_attributes
        frozenset({'a1', 'a2', 'a3', 'a4'})
    """
    if n < 3:
        msg = "Cycle query must have at least 3 relations (n >= 3)"
        raise ValueError(msg)

    relations: list[Relation] = []

    for i in range(1, n + 1):
        attr1 = f"a{i}"
        attr2 = f"a{(i % n) + 1}"
        relations.append(Relation(f"R{i}", frozenset({attr1, attr2}), 100 + 50 * i))

    return QueryHypergraph(
        relations=tuple(relations),
        name=f"cycle_{n}",
    )
