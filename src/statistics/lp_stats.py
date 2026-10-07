"""Lp-norm block extension statistics for LpH-CEG.

Extends the standard block extension statistics with full degree sequence
collection and Lp-norm computation for multiple p-values.  The core
primitive is ``LpBlockExtensionStats``, which stores the degree sequence
and precomputed norms alongside the legacy max/avg statistics.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from src.statistics.block_stats import (
    _enumerate_interfaces,
    _extract_projection,
    _join_block_relations,
    _resolve_attr_indices,
    _validate_relation_payload,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from src.statistics.block_stats import RelationDataMap
    from src.utils.types import AttributeName, Block, Interface, QueryHypergraph

LOGGER = logging.getLogger(__name__)

DEFAULT_P_VALUES: tuple[float, ...] = (1.0, 2.0, float("inf"))

LpNormMap: TypeAlias = dict[float, float]


@dataclass(frozen=True)
class LpBlockExtensionStats:
    """Lp-norm block extension statistics ``mu_p(F | I)``.

    Attributes:
        degree_sequence: Sorted (descending) degree sequence d(a_I) for
            each distinct interface value a_I.
        lp_norms: Mapping from p to ``||deg||_p``.
        interface_size: Number of attributes in the interface.
        count: Number of distinct interface values.
        max_extension: Maximum degree (= l∞ norm).
        avg_extension: Average degree (= l1 norm / count).
        log_max_extension: log₂ of max degree (for log-space DAG weights).
    """

    degree_sequence: tuple[int, ...]
    lp_norms: LpNormMap
    interface_size: int
    count: int
    max_extension: int
    avg_extension: float
    log_max_extension: float

    def norm(self, p: float) -> float:
        """Return the lp-norm, computing on the fly if not precomputed."""
        if p in self.lp_norms:
            return self.lp_norms[p]
        return _compute_lp_norm(self.degree_sequence, p)

    def log_norm(self, p: float) -> float:
        """Return log₂ of the lp-norm."""
        n = self.norm(p)
        if n <= 0.0:
            return 0.0
        return math.log2(n)

    @property
    def is_simple(self) -> bool:
        """Whether the degree sequence is simple (interface size ≤ 1).

        LpBound tightness is guaranteed only for simple degree sequences.
        """
        return self.interface_size <= 1


def _compute_lp_norm(degree_sequence: tuple[int, ...], p: float) -> float:
    """Compute lp-norm of a degree sequence."""
    if not degree_sequence:
        return 0.0
    if p == float("inf"):
        return float(degree_sequence[0])  # already sorted descending
    if p == 1.0:
        return float(sum(degree_sequence))
    return float(sum(d**p for d in degree_sequence) ** (1.0 / p))


def _collect_degree_sequence(
    joined_tuples: list[tuple[str, ...]] | tuple[tuple[str, ...], ...],
    joined_attrs: tuple[AttributeName, ...],
    interface: Interface,
) -> tuple[int, ...]:
    """Collect the degree sequence for a block/interface pair.

    Returns:
        Tuple of extension counts sorted in descending order.
    """
    if not joined_tuples:
        return ()

    interface_indices = _resolve_attr_indices(joined_attrs, interface)
    extension_attrs = frozenset(joined_attrs) - interface
    extension_indices = _resolve_attr_indices(joined_attrs, extension_attrs)

    extension_map: dict[tuple[str, ...], set[tuple[str, ...]]] = defaultdict(set)
    for row in joined_tuples:
        interface_value = (
            _extract_projection(row, interface_indices) if interface_indices else ()
        )
        extension_value = (
            _extract_projection(row, extension_indices) if extension_indices else ()
        )
        extension_map[interface_value].add(extension_value)

    sizes = sorted((len(v) for v in extension_map.values()), reverse=True)
    return tuple(sizes)


def compute_lp_block_extension(
    query: QueryHypergraph,
    block: Block,
    interface: Interface,
    relation_data: RelationDataMap,
    *,
    p_values: tuple[float, ...] = DEFAULT_P_VALUES,
) -> LpBlockExtensionStats:
    """Compute Lp-norm block extension statistics for a block/interface pair.

    Args:
        query: Query hypergraph defining valid relations and block schemas.
        block: Connected non-empty block of query relations.
        interface: Interface attributes to condition on.
        relation_data: Mapping from relation name to
            ``(attribute_order, tuples)``.
        p_values: Lp-norm exponents to precompute.

    Returns:
        Full Lp-norm extension statistics including degree sequence.

    Raises:
        KeyError: If relation data is missing for a relation in ``block``.
        ValueError: If the block is invalid or interface is not contained
            in the block attributes.
    """
    if not query.is_valid_block(block):
        msg = f"Invalid block for query {query.name!r}: {sorted(block.relations)!r}"
        raise ValueError(msg)

    block_attributes = query.attributes_of(block)
    if not interface <= block_attributes:
        msg = (
            f"Interface {sorted(interface)!r} must be contained in block "
            f"attributes {sorted(block_attributes)!r}"
        )
        raise ValueError(msg)

    for relation_name in sorted(block.relations):
        if relation_name not in relation_data:
            msg = f"Missing relation data for {relation_name!r}"
            raise KeyError(msg)
        _validate_relation_payload(query, relation_name, relation_data[relation_name])

    joined_tuples, joined_attrs = _join_block_relations(query, block, relation_data)
    degree_seq = _collect_degree_sequence(joined_tuples, joined_attrs, interface)

    if not degree_seq:
        norms = {p: 0.0 for p in p_values}
        return LpBlockExtensionStats(
            degree_sequence=(),
            lp_norms=norms,
            interface_size=len(interface),
            count=0,
            max_extension=0,
            avg_extension=0.0,
            log_max_extension=0.0,
        )

    norms = {p: _compute_lp_norm(degree_seq, p) for p in p_values}
    max_ext = degree_seq[0]
    count = len(degree_seq)
    avg_ext = sum(degree_seq) / count
    log_max = math.log2(max_ext) if max_ext >= 2 else 0.0

    LOGGER.debug(
        "lp_mu(F|I) block=%s, interface=%s: norms=%s, count=%d, max=%d",
        sorted(block.relations),
        sorted(interface),
        {p: f"{v:.2f}" for p, v in norms.items()},
        count,
        max_ext,
    )

    return LpBlockExtensionStats(
        degree_sequence=degree_seq,
        lp_norms=norms,
        interface_size=len(interface),
        count=count,
        max_extension=max_ext,
        avg_extension=avg_ext,
        log_max_extension=log_max,
    )


def compute_all_lp_block_extensions(
    query: QueryHypergraph,
    blocks: Iterable[Block],
    relation_data_map: RelationDataMap,
    *,
    p_values: tuple[float, ...] = DEFAULT_P_VALUES,
) -> dict[tuple[Block, Interface], LpBlockExtensionStats]:
    """Compute Lp-norm extension stats for every interface of each block.

    Block join results are cached to avoid redundant computation.

    Args:
        query: Query hypergraph.
        blocks: Blocks to compute statistics for.
        relation_data_map: Relation data for all query relations.
        p_values: Lp-norm exponents to precompute.

    Returns:
        Mapping from ``(block, interface)`` to Lp extension statistics.
    """
    result: dict[tuple[Block, Interface], LpBlockExtensionStats] = {}
    unique_blocks = tuple(
        sorted(
            set(blocks),
            key=lambda b: (len(b), tuple(sorted(b.relations))),
        )
    )

    join_cache: dict[
        frozenset[str],
        tuple[list[tuple[str, ...]], tuple[AttributeName, ...]],
    ] = {}

    for block in unique_blocks:
        if not query.is_valid_block(block):
            msg = f"Invalid block for query {query.name!r}: {sorted(block.relations)!r}"
            raise ValueError(msg)

        for rel_name in sorted(block.relations):
            if rel_name not in relation_data_map:
                msg = f"Missing relation data for {rel_name!r}"
                raise KeyError(msg)
            _validate_relation_payload(query, rel_name, relation_data_map[rel_name])

        cache_key = block.relations
        if cache_key in join_cache:
            joined_tuples, joined_attrs = join_cache[cache_key]
        else:
            joined_tuples, joined_attrs = _join_block_relations(
                query, block, relation_data_map
            )
            join_cache[cache_key] = (joined_tuples, joined_attrs)

        block_attrs = frozenset(joined_attrs)
        for interface in _enumerate_interfaces(block_attrs):
            deg_seq = _collect_degree_sequence(joined_tuples, joined_attrs, interface)

            if not deg_seq:
                norms = {p: 0.0 for p in p_values}
                result[(block, interface)] = LpBlockExtensionStats(
                    degree_sequence=(),
                    lp_norms=norms,
                    interface_size=len(interface),
                    count=0,
                    max_extension=0,
                    avg_extension=0.0,
                    log_max_extension=0.0,
                )
                continue

            norms = {p: _compute_lp_norm(deg_seq, p) for p in p_values}
            max_ext = deg_seq[0]
            count = len(deg_seq)
            avg_ext = sum(deg_seq) / count
            log_max = math.log2(max_ext) if max_ext >= 2 else 0.0

            result[(block, interface)] = LpBlockExtensionStats(
                degree_sequence=deg_seq,
                lp_norms=norms,
                interface_size=len(interface),
                count=count,
                max_extension=max_ext,
                avg_extension=avg_ext,
                log_max_extension=log_max,
            )

    return result


# ---------------------------------------------------------------------------
# L4: Predicate-aware filtered degree sequences
# ---------------------------------------------------------------------------


def compute_filtered_lp_block_extension(
    query: QueryHypergraph,
    block: Block,
    interface: Interface,
    relation_data: RelationDataMap,
    *,
    predicates: dict[str, set[str]] | None = None,
    p_values: tuple[float, ...] = DEFAULT_P_VALUES,
) -> LpBlockExtensionStats:
    """Compute Lp-norm stats with predicate filtering.

    Before computing degree sequences, tuples that fail predicate
    filters are removed.  This yields tighter bounds when WHERE
    clauses restrict the effective data.

    Args:
        query: Query hypergraph.
        block: Block of relations.
        interface: Interface attributes.
        relation_data: Full relation data.
        predicates: Optional per-attribute value filters. Maps
            attribute name → set of allowed values.  Tuples with
            attribute values outside the allowed set are dropped.
        p_values: Lp-norm exponents.

    Returns:
        Lp-norm extension statistics on the filtered data.
    """
    if predicates is None:
        return compute_lp_block_extension(
            query, block, interface, relation_data, p_values=p_values
        )

    # Build filtered relation data
    filtered: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]] = {}
    for rel_name in block.relations:
        attrs, tuples = relation_data[rel_name]
        kept: list[tuple[str, ...]] = []
        for row in tuples:
            ok = True
            for ai, attr in enumerate(attrs):
                if attr in predicates and row[ai] not in predicates[attr]:
                    ok = False
                    break
            if ok:
                kept.append(row)
        filtered[rel_name] = (attrs, kept)

    # Copy non-block relations unchanged
    for rname in relation_data:
        if rname not in filtered:
            attrs_r, tuples_r = relation_data[rname]
            filtered[rname] = (attrs_r, list(tuples_r))

    return compute_lp_block_extension(
        query, block, interface, filtered, p_values=p_values
    )


# ---------------------------------------------------------------------------
# L2: Histogram-based approximate Lp-norms
# ---------------------------------------------------------------------------


def approximate_lp_norms_from_histogram(
    histogram: list[tuple[int, int]],
    p_values: tuple[float, ...] = DEFAULT_P_VALUES,
) -> LpNormMap:
    """Approximate Lp-norms from a degree histogram.

    Database catalogs (e.g., pg_statistic) store degree distributions
    as histograms rather than full degree sequences.  This function
    approximates Lp-norms from histogram buckets.

    Args:
        histogram: List of (degree_value, count) pairs.  Each pair
            means ``count`` interface values have degree ``degree_value``.
            E.g., [(1, 100), (5, 20), (10, 3)] means 100 values with
            degree 1, 20 with degree 5, 3 with degree 10.
        p_values: Lp-norm exponents to compute.

    Returns:
        Mapping from p to approximate Lp-norm.

    Example:
        >>> hist = [(1, 100), (5, 20), (10, 3)]
        >>> norms = approximate_lp_norms_from_histogram(hist)
        >>> norms[float('inf')]  # max degree
        10.0
        >>> norms[1.0]  # sum of all degrees
        230.0
    """
    if not histogram:
        return {p: 0.0 for p in p_values}

    norms: LpNormMap = {}
    for p in p_values:
        if p == float("inf"):
            norms[p] = float(max(d for d, _ in histogram))
        elif p == 1.0:
            norms[p] = float(sum(d * c for d, c in histogram))
        else:
            norms[p] = float(sum(c * d**p for d, c in histogram) ** (1.0 / p))
    return norms


def lp_stats_from_histogram(
    histogram: list[tuple[int, int]],
    interface_size: int = 1,
    p_values: tuple[float, ...] = DEFAULT_P_VALUES,
) -> LpBlockExtensionStats:
    """Build LpBlockExtensionStats from a degree histogram.

    Useful for integrating with database catalogs that provide
    histogram-based statistics rather than full degree sequences.

    Args:
        histogram: List of (degree_value, count) pairs.
        interface_size: Number of interface attributes.
        p_values: Lp-norm exponents.

    Returns:
        Approximate Lp-norm block extension statistics.
    """
    if not histogram:
        return LpBlockExtensionStats(
            degree_sequence=(),
            lp_norms={p: 0.0 for p in p_values},
            interface_size=interface_size,
            count=0,
            max_extension=0,
            avg_extension=0.0,
            log_max_extension=0.0,
        )

    # Expand histogram to degree sequence (for small histograms)
    degree_seq = tuple(
        sorted(
            (d for d, c in histogram for _ in range(c)),
            reverse=True,
        )
    )
    norms = approximate_lp_norms_from_histogram(histogram, p_values)
    max_ext = degree_seq[0] if degree_seq else 0
    count = sum(c for _, c in histogram)
    total_deg = sum(d * c for d, c in histogram)
    avg_ext = total_deg / count if count else 0.0
    log_max = math.log2(max_ext) if max_ext >= 2 else 0.0

    return LpBlockExtensionStats(
        degree_sequence=degree_seq,
        lp_norms=norms,
        interface_size=interface_size,
        count=count,
        max_extension=max_ext,
        avg_extension=avg_ext,
        log_max_extension=log_max,
    )
