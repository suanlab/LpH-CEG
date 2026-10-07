"""Binary degree statistics for CEG baseline.

Computes the generalized degree statistic deg(X, Y, R) used as
edge weights in the Cardinality Estimation Graph (CEG).

Definition:
    deg(X, Y, R) = max over all X-value assignments a_X
                   of |{ pi_Y(t) : t in R, pi_X(t) = a_X }|

    i.e. the maximum number of distinct Y-extensions for any X-value.

References:
    Salihoglu et al., "Cardinality Estimation Graphs", CACM 2026.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.utils.types import AttributeName

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BinaryDegreeStats:
    """Binary degree statistics deg(X, Y, R).

    Attributes:
        max_degree: Maximum number of distinct Y-extensions over all
            X-value assignments. This is deg(X, Y, R) used in CEG_M.
        avg_degree: Average number of distinct Y-extensions across all
            distinct X-value assignments. Used in CEG_O.
        count: Number of distinct X-value assignments observed.
        log_max_degree: log2(max_degree) for log-space computation.
            Equals 0.0 when max_degree <= 1.
    """

    max_degree: int
    avg_degree: float
    count: int
    log_max_degree: float

    @property
    def is_empty(self) -> bool:
        """Whether this was computed from an empty relation."""
        return self.count == 0


def _extract_projection(
    row: tuple[str, ...],
    attr_indices: list[int],
) -> tuple[str, ...]:
    return tuple(row[i] for i in attr_indices)


def _resolve_attr_indices(
    attribute_order: tuple[AttributeName, ...],
    attrs: frozenset[AttributeName],
) -> list[int]:
    index_map = {name: idx for idx, name in enumerate(attribute_order)}
    indices: list[int] = []
    for attr in sorted(attrs):
        if attr not in index_map:
            msg = f"Attribute {attr!r} not found in attribute_order {attribute_order!r}"
            raise ValueError(msg)
        indices.append(index_map[attr])
    return sorted(indices)


def compute_binary_degree(
    tuples: Sequence[tuple[str, ...]],
    attribute_order: tuple[AttributeName, ...],
    from_attrs: frozenset[AttributeName],
    to_attrs: frozenset[AttributeName],
) -> BinaryDegreeStats:
    """Compute binary degree statistics deg(X, Y, R).

    For each distinct X-value assignment in the relation, counts the
    number of distinct Y-value extensions. Returns the maximum and
    average of these counts.

    Args:
        tuples: Relation data as a sequence of tuples. Each tuple has
            one value per attribute in attribute_order.
        attribute_order: Ordered attribute names defining column positions.
            Must match the width of each tuple.
        from_attrs: Source attribute set X. May be empty (computes global
            degree to Y).
        to_attrs: Target attribute set Y. Must be non-empty.

    Returns:
        BinaryDegreeStats with max/avg degree, count, and log values.

    Raises:
        ValueError: If to_attrs is empty, or attributes not in
            attribute_order, or from_attrs and to_attrs overlap.

    Examples:
        >>> stats = compute_binary_degree(
        ...     tuples=[("a", "b"), ("a", "c"), ("d", "b")],
        ...     attribute_order=("x", "y"),
        ...     from_attrs=frozenset({"x"}),
        ...     to_attrs=frozenset({"y"}),
        ... )
        >>> stats.max_degree
        2
        >>> stats.avg_degree
        1.5
    """
    if not to_attrs:
        msg = "to_attrs (Y) must be non-empty"
        raise ValueError(msg)

    if from_attrs & to_attrs:
        msg = (
            f"from_attrs and to_attrs must be disjoint, got overlap: "
            f"{from_attrs & to_attrs!r}"
        )
        raise ValueError(msg)

    if not tuples:
        logger.debug("Empty relation: returning zero-valued stats")
        return BinaryDegreeStats(
            max_degree=0,
            avg_degree=0.0,
            count=0,
            log_max_degree=0.0,
        )

    from_indices = _resolve_attr_indices(attribute_order, from_attrs)
    to_indices = _resolve_attr_indices(attribute_order, to_attrs)

    degree_map: dict[tuple[str, ...], set[tuple[str, ...]]] = defaultdict(set)

    for row in tuples:
        x_val = _extract_projection(row, from_indices) if from_indices else ()
        y_val = _extract_projection(row, to_indices)
        degree_map[x_val].add(y_val)

    degrees = [len(y_set) for y_set in degree_map.values()]
    max_degree = max(degrees)
    count = len(degrees)
    avg_degree = sum(degrees) / count
    log_max = math.log2(max_degree) if max_degree >= 2 else 0.0

    logger.debug(
        "deg(X=%s, Y=%s): max=%d, avg=%.4f, count=%d",
        sorted(from_attrs),
        sorted(to_attrs),
        max_degree,
        avg_degree,
        count,
    )

    return BinaryDegreeStats(
        max_degree=max_degree,
        avg_degree=avg_degree,
        count=count,
        log_max_degree=log_max,
    )


def compute_all_binary_degrees(
    tuples: Sequence[tuple[str, ...]],
    attribute_order: tuple[AttributeName, ...],
) -> dict[tuple[frozenset[AttributeName], frozenset[AttributeName]], BinaryDegreeStats]:
    """Compute binary degree stats for all valid (X, Y) pairs in a relation.

    Enumerates all pairs where X and Y are disjoint non-overlapping
    single-attribute subsets. This covers the standard binary CEG edges.

    Args:
        tuples: Relation data as a sequence of tuples.
        attribute_order: Ordered attribute names defining column positions.

    Returns:
        Dictionary mapping (from_attrs, to_attrs) pairs to their
        BinaryDegreeStats. Includes empty from_attrs for global degrees.
    """
    attrs = frozenset(attribute_order)
    result: dict[
        tuple[frozenset[AttributeName], frozenset[AttributeName]], BinaryDegreeStats
    ] = {}

    for attr_y in attrs:
        to_set = frozenset({attr_y})

        stats = compute_binary_degree(tuples, attribute_order, frozenset(), to_set)
        result[(frozenset(), to_set)] = stats

        for attr_x in attrs:
            if attr_x == attr_y:
                continue
            from_set = frozenset({attr_x})
            stats = compute_binary_degree(tuples, attribute_order, from_set, to_set)
            result[(from_set, to_set)] = stats

    return result
