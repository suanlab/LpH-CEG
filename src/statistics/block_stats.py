"""Exact block extension statistics for H-CEG."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import TypeAlias

from src.utils.types import (
    AttributeName,
    Block,
    Interface,
    QueryHypergraph,
    RelationName,
)

RelationRows: TypeAlias = Sequence[tuple[str, ...]]
RelationPayload: TypeAlias = tuple[tuple[AttributeName, ...], RelationRows]
RelationDataMap: TypeAlias = Mapping[RelationName, RelationPayload]

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BlockExtensionStats:
    """Block extension statistics ``mu(F | I)``."""

    max_extension: int
    avg_extension: float
    count: int
    log_max_extension: float

    @property
    def is_empty(self) -> bool:
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


def _validate_relation_payload(
    query: QueryHypergraph,
    relation_name: RelationName,
    payload: RelationPayload,
) -> None:
    attribute_order, tuples = payload
    relation = query.get_relation(relation_name)

    if frozenset(attribute_order) != relation.attributes:
        msg = (
            f"Relation data for {relation_name!r} has attributes "
            f"{attribute_order!r}, expected {sorted(relation.attributes)!r}"
        )
        raise ValueError(msg)

    if len(attribute_order) != len(set(attribute_order)):
        msg = f"Relation data for {relation_name!r} has duplicate attributes"
        raise ValueError(msg)

    expected_width = len(attribute_order)
    for row in tuples:
        if len(row) != expected_width:
            msg = (
                f"Tuple width mismatch for relation {relation_name!r}: "
                f"expected {expected_width}, got {len(row)}"
            )
            raise ValueError(msg)


def _natural_join(
    tuples_a: RelationRows,
    attrs_a: tuple[AttributeName, ...],
    tuples_b: RelationRows,
    attrs_b: tuple[AttributeName, ...],
) -> tuple[list[tuple[str, ...]], tuple[AttributeName, ...]]:
    shared_attrs = tuple(sorted(set(attrs_a) & set(attrs_b)))
    output_attrs = attrs_a + tuple(attr for attr in attrs_b if attr not in attrs_a)

    if not tuples_a or not tuples_b:
        return [], output_attrs

    unique_a = sorted(set(tuples_a))
    unique_b = sorted(set(tuples_b))

    if not shared_attrs:
        result = sorted({row_a + row_b for row_a in unique_a for row_b in unique_b})
        return result, output_attrs

    index_a = {attr: idx for idx, attr in enumerate(attrs_a)}
    index_b = {attr: idx for idx, attr in enumerate(attrs_b)}
    shared_indices_a = [index_a[attr] for attr in shared_attrs]
    shared_indices_b = [index_b[attr] for attr in shared_attrs]
    extra_indices_b = [index_b[attr] for attr in attrs_b if attr not in index_a]

    # Build hash index on the smaller relation for faster joins
    if len(unique_a) > len(unique_b):
        # Index side B, probe side A (default)
        buckets: dict[tuple[str, ...], list[tuple[str, ...]]] = defaultdict(list)
        for row_b in unique_b:
            key = _extract_projection(row_b, shared_indices_b)
            buckets[key].append(row_b)

        result_rows: set[tuple[str, ...]] = set()
        for row_a in unique_a:
            key = _extract_projection(row_a, shared_indices_a)
            matches = buckets.get(key)
            if matches is None:
                continue
            for row_b in matches:
                extension = tuple(row_b[idx] for idx in extra_indices_b)
                result_rows.add(row_a + extension)
    else:
        # Index side A, probe side B
        buckets_a: dict[tuple[str, ...], list[tuple[str, ...]]] = defaultdict(list)
        for row_a in unique_a:
            key = _extract_projection(row_a, shared_indices_a)
            buckets_a[key].append(row_a)

        result_rows = set()
        for row_b in unique_b:
            key = _extract_projection(row_b, shared_indices_b)
            matches_a = buckets_a.get(key)
            if matches_a is None:
                continue
            extension = tuple(row_b[idx] for idx in extra_indices_b)
            for row_a in matches_a:
                result_rows.add(row_a + extension)

    return sorted(result_rows), output_attrs


def _join_block_relations(
    query: QueryHypergraph,
    block: Block,
    relation_data: RelationDataMap,
) -> tuple[list[tuple[str, ...]], tuple[AttributeName, ...]]:
    relation_names = tuple(sorted(block.relations))
    if not relation_names:
        msg = "Block must be non-empty"
        raise ValueError(msg)

    first_relation = relation_names[0]
    first_payload = relation_data[first_relation]
    current_attrs, current_tuples = first_payload
    joined_tuples = sorted(set(current_tuples))
    joined_attrs = current_attrs

    for relation_name in relation_names[1:]:
        next_attrs, next_tuples = relation_data[relation_name]
        joined_tuples, joined_attrs = _natural_join(
            joined_tuples,
            joined_attrs,
            next_tuples,
            next_attrs,
        )

    return joined_tuples, joined_attrs


def _compute_extension_stats(
    joined_tuples: RelationRows,
    joined_attrs: tuple[AttributeName, ...],
    interface: Interface,
) -> BlockExtensionStats:
    if not joined_tuples:
        LOGGER.debug("Empty block join: returning zero-valued stats")
        return BlockExtensionStats(
            max_extension=0,
            avg_extension=0.0,
            count=0,
            log_max_extension=0.0,
        )

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

    extension_sizes = [len(values) for values in extension_map.values()]
    max_extension = max(extension_sizes)
    count = len(extension_sizes)
    avg_extension = sum(extension_sizes) / count
    log_max_extension = math.log2(max_extension) if max_extension >= 2 else 0.0

    LOGGER.debug(
        "mu(F|I) with interface=%s: max=%d, avg=%.4f, count=%d",
        sorted(interface),
        max_extension,
        avg_extension,
        count,
    )

    return BlockExtensionStats(
        max_extension=max_extension,
        avg_extension=avg_extension,
        count=count,
        log_max_extension=log_max_extension,
    )


def _enumerate_interfaces(
    attrs: frozenset[AttributeName],
) -> tuple[Interface, ...]:
    ordered_attrs = tuple(sorted(attrs))
    interfaces: list[Interface] = []
    for size in range(len(ordered_attrs) + 1):
        for subset in combinations(ordered_attrs, size):
            interfaces.append(frozenset(subset))
    return tuple(interfaces)


def compute_block_extension(
    query: QueryHypergraph,
    block: Block,
    interface: Interface,
    relation_data: RelationDataMap,
) -> BlockExtensionStats:
    """Compute exact block extension statistics for a block/interface pair.

    Args:
        query: Query hypergraph defining valid relations and block schemas.
        block: Connected non-empty block of query relations.
        interface: Interface attributes to condition on.
        relation_data: Mapping from relation name to ``(attribute_order, tuples)``.

    Returns:
        Exact block extension statistics for ``mu(F | I)``.

    Raises:
        KeyError: If relation data is missing for a relation in ``block``.
        ValueError: If the block is invalid, the interface is not contained in the
            block attributes, or relation payloads do not match the query schema.
    """
    if not query.is_valid_block(block):
        msg = f"Invalid block for query {query.name!r}: {sorted(block.relations)!r}"
        raise ValueError(msg)

    block_attributes = query.attributes_of(block)
    if not interface <= block_attributes:
        msg = (
            f"Interface {sorted(interface)!r} must be contained in block attributes "
            f"{sorted(block_attributes)!r}"
        )
        raise ValueError(msg)

    for relation_name in sorted(block.relations):
        if relation_name not in relation_data:
            msg = f"Missing relation data for {relation_name!r}"
            raise KeyError(msg)
        _validate_relation_payload(query, relation_name, relation_data[relation_name])

    joined_tuples, joined_attrs = _join_block_relations(query, block, relation_data)
    return _compute_extension_stats(joined_tuples, joined_attrs, interface)


def _streaming_join_stats(
    query: QueryHypergraph,
    block: Block,
    relation_data: RelationDataMap,
) -> dict[Interface, BlockExtensionStats]:
    """Compute extension stats for ALL interfaces via streaming join.

    For pairwise blocks, performs a hash-probe join that never
    materializes the full join result. Instead, each join tuple is
    immediately dispatched to per-interface counters, keeping only
    the distinct extension sets in memory.

    Memory: O(distinct_interface_values * avg_extensions) per interface.
    Time: O(|R join S|) -- same as materialized join, but no list storage.
    """
    relation_names = tuple(sorted(block.relations))

    if len(relation_names) == 1:
        attrs_order, rows = relation_data[relation_names[0]]
        joined_tuples = sorted(set(rows))
        joined_attrs = attrs_order
        return _aggregate_all_interfaces(joined_tuples, joined_attrs)

    # Multi-relation: streaming hash-probe join with inline aggregation
    first_payload = relation_data[relation_names[0]]
    current_attrs, current_tuples = first_payload
    current_rows = sorted(set(current_tuples))
    current_joined_attrs = current_attrs

    for rname in relation_names[1:]:
        next_attrs, next_tuples = relation_data[rname]
        next_rows = sorted(set(next_tuples))

        output_attrs = current_joined_attrs + tuple(
            a for a in next_attrs if a not in current_joined_attrs
        )
        shared = tuple(sorted(set(current_joined_attrs) & set(next_attrs)))

        if not shared:
            # Cross join — must materialize (rare for connected blocks)
            joined: list[tuple[str, ...]] = sorted(
                {ra + rb for ra in current_rows for rb in next_rows}
            )
            current_rows = joined
            current_joined_attrs = output_attrs
            continue

        idx_a = {a: i for i, a in enumerate(current_joined_attrs)}
        idx_b = {a: i for i, a in enumerate(next_attrs)}
        shared_a = [idx_a[a] for a in shared]
        shared_b = [idx_b[a] for a in shared]
        extra_b = [idx_b[a] for a in next_attrs if a not in idx_a]

        # If this is the LAST relation in the block, stream directly
        # to interface counters without materializing
        if rname == relation_names[-1]:
            # Build hash index on smaller side
            buckets: dict[tuple[str, ...], list[tuple[str, ...]]]
            buckets = defaultdict(list)
            for rb in next_rows:
                key = tuple(rb[i] for i in shared_b)
                buckets[key].append(rb)

            return _streaming_probe_aggregate(
                current_rows, current_joined_attrs,
                buckets, shared_a, extra_b, output_attrs,
            )

        # Intermediate join — must materialize for next step
        buckets_mid: dict[tuple[str, ...], list[tuple[str, ...]]]
        buckets_mid = defaultdict(list)
        for rb in next_rows:
            key = tuple(rb[i] for i in shared_b)
            buckets_mid[key].append(rb)

        new_rows: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()
        for ra in current_rows:
            key = tuple(ra[i] for i in shared_a)
            for rb in buckets_mid.get(key, []):
                ext = tuple(rb[i] for i in extra_b)
                row = ra + ext
                if row not in seen:
                    seen.add(row)
                    new_rows.append(row)
        current_rows = new_rows
        current_joined_attrs = output_attrs

    return _aggregate_all_interfaces(current_rows, current_joined_attrs)


def _streaming_probe_aggregate(
    probe_rows: list[tuple[str, ...]],
    probe_attrs: tuple[AttributeName, ...],
    build_buckets: dict[tuple[str, ...], list[tuple[str, ...]]],
    shared_probe_indices: list[int],
    extra_build_indices: list[int],
    output_attrs: tuple[AttributeName, ...],
) -> dict[Interface, BlockExtensionStats]:
    """Probe hash index and aggregate interface stats without materialization.

    For each probe row, looks up matching build rows and immediately
    updates all interface counters. No join result list is created.
    """
    block_attrs = frozenset(output_attrs)
    all_interfaces = _enumerate_interfaces(block_attrs)

    iface_mappings: list[tuple[Interface, list[int], list[int]]] = []
    for interface in all_interfaces:
        iface_idx = _resolve_attr_indices(output_attrs, interface)
        ext_attrs = frozenset(output_attrs) - interface
        ext_idx = _resolve_attr_indices(output_attrs, ext_attrs)
        iface_mappings.append((interface, iface_idx, ext_idx))

    counters: dict[Interface, dict[tuple[str, ...], set[tuple[str, ...]]]] = {
        iface: defaultdict(set) for iface, _, _ in iface_mappings
    }

    seen: set[tuple[str, ...]] = set()
    for probe_row in probe_rows:
        key = tuple(probe_row[i] for i in shared_probe_indices)
        for build_row in build_buckets.get(key, []):
            ext = tuple(build_row[i] for i in extra_build_indices)
            joined_row = probe_row + ext
            if joined_row in seen:
                continue
            seen.add(joined_row)

            for interface, iface_idx, ext_idx in iface_mappings:
                iv = tuple(joined_row[i] for i in iface_idx) if iface_idx else ()
                ev = tuple(joined_row[i] for i in ext_idx) if ext_idx else ()
                counters[interface][iv].add(ev)

    return _build_stats_from_counters(counters, iface_mappings)


def _aggregate_all_interfaces(
    joined_tuples: list[tuple[str, ...]],
    joined_attrs: tuple[AttributeName, ...],
) -> dict[Interface, BlockExtensionStats]:
    """Aggregate extension stats for all interfaces from materialized tuples."""
    block_attrs = frozenset(joined_attrs)
    all_interfaces = _enumerate_interfaces(block_attrs)

    iface_mappings: list[tuple[Interface, list[int], list[int]]] = []
    for interface in all_interfaces:
        iface_idx = _resolve_attr_indices(joined_attrs, interface)
        ext_attrs = frozenset(joined_attrs) - interface
        ext_idx = _resolve_attr_indices(joined_attrs, ext_attrs)
        iface_mappings.append((interface, iface_idx, ext_idx))

    counters: dict[Interface, dict[tuple[str, ...], set[tuple[str, ...]]]] = {
        iface: defaultdict(set) for iface, _, _ in iface_mappings
    }

    for row in joined_tuples:
        for interface, iface_idx, ext_idx in iface_mappings:
            iv = tuple(row[i] for i in iface_idx) if iface_idx else ()
            ev = tuple(row[i] for i in ext_idx) if ext_idx else ()
            counters[interface][iv].add(ev)

    return _build_stats_from_counters(counters, iface_mappings)


def _build_stats_from_counters(
    counters: dict[Interface, dict[tuple[str, ...], set[tuple[str, ...]]]],
    iface_mappings: list[tuple[Interface, list[int], list[int]]],
) -> dict[Interface, BlockExtensionStats]:
    """Convert counter dicts to BlockExtensionStats."""
    results: dict[Interface, BlockExtensionStats] = {}
    for interface, _, _ in iface_mappings:
        ext_map = counters[interface]
        if not ext_map:
            results[interface] = BlockExtensionStats(
                max_extension=0, avg_extension=0.0,
                count=0, log_max_extension=0.0,
            )
            continue
        sizes = [len(v) for v in ext_map.values()]
        mx = max(sizes)
        cnt = len(sizes)
        avg = sum(sizes) / cnt
        results[interface] = BlockExtensionStats(
            max_extension=mx, avg_extension=avg,
            count=cnt,
            log_max_extension=math.log2(mx) if mx >= 2 else 0.0,
        )
    return results


def compute_all_block_extensions(
    query: QueryHypergraph,
    blocks: Iterable[Block],
    relation_data_map: RelationDataMap,
) -> dict[tuple[Block, Interface], BlockExtensionStats]:
    """Compute exact extension stats for every interface of each block.

    Uses streaming aggregation: join tuples are scanned once to update
    all interface counters simultaneously, reducing memory overhead.
    Block join results are cached to avoid redundant computation.
    """
    result: dict[tuple[Block, Interface], BlockExtensionStats] = {}
    unique_blocks = tuple(
        sorted(
            set(blocks), key=lambda block: (len(block), tuple(sorted(block.relations)))
        )
    )

    # Cache block join results by frozenset of relation names
    join_cache: dict[
        frozenset[str],
        dict[Interface, BlockExtensionStats],
    ] = {}

    for block in unique_blocks:
        if not query.is_valid_block(block):
            msg = f"Invalid block for query {query.name!r}: {sorted(block.relations)!r}"
            raise ValueError(msg)

        for relation_name in sorted(block.relations):
            if relation_name not in relation_data_map:
                msg = f"Missing relation data for {relation_name!r}"
                raise KeyError(msg)
            _validate_relation_payload(
                query, relation_name, relation_data_map[relation_name]
            )

        cache_key = block.relations
        if cache_key in join_cache:
            iface_stats = join_cache[cache_key]
        else:
            iface_stats = _streaming_join_stats(
                query, block, relation_data_map
            )
            join_cache[cache_key] = iface_stats

        for interface, stats in iface_stats.items():
            result[(block, interface)] = stats

    return result
