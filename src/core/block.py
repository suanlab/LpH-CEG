"""Block enumeration utilities for H-CEG."""

from __future__ import annotations

import logging
from itertools import combinations

from src.utils.types import Block, QueryHypergraph

LOGGER = logging.getLogger(__name__)
DEFAULT_MAX_BLOCKS_WARN = 1000


def is_valid_block(query: QueryHypergraph, block: Block) -> bool:
    """Return whether a block is non-empty, in-query, and connected.

    This is a thin wrapper around ``QueryHypergraph.is_valid_block`` so block
    validation stays consistent across the codebase.

    Args:
        query: Query hypergraph that defines the relation universe.
        block: Candidate relation block.

    Returns:
        True if the block is valid for ``query``.
    """
    return query.is_valid_block(block)


def enumerate_singleton_blocks(query: QueryHypergraph) -> tuple[Block, ...]:
    """Enumerate all singleton relation blocks in deterministic order.

    Args:
        query: Query hypergraph whose relations define the singleton blocks.

    Returns:
        Tuple of singleton blocks sorted by relation name.
    """
    ordered_relation_names = sorted(query.relation_names)
    return tuple(
        Block(frozenset({relation_name})) for relation_name in ordered_relation_names
    )


def enumerate_blocks(
    query: QueryHypergraph,
    max_block_size: int = 3,
    *,
    max_blocks_warn: int = DEFAULT_MAX_BLOCKS_WARN,
) -> tuple[Block, ...]:
    """Enumerate all valid blocks up to a maximum size.

    Blocks are enumerated exactly over sizes ``1..max_block_size`` after
    capping the requested size by the number of relations in the query. Output
    order is deterministic: first by block size, then lexicographically by the
    sorted relation names inside each block.

    Args:
        query: Query hypergraph whose relation subsets are enumerated.
        max_block_size: Maximum block cardinality to enumerate. Must be at
            least 1.
        max_blocks_warn: Warning threshold for the total number of enumerated
            valid blocks.

    Returns:
        Tuple of valid blocks of size ``1..max_block_size``.

    Raises:
        ValueError: If ``max_block_size`` is less than 1.
    """
    if max_block_size < 1:
        msg = "max_block_size must be at least 1"
        raise ValueError(msg)

    ordered_relation_names = tuple(sorted(query.relation_names))
    effective_max_block_size = min(max_block_size, len(ordered_relation_names))
    blocks: list[Block] = []

    for block_size in range(1, effective_max_block_size + 1):
        for relation_names in combinations(ordered_relation_names, block_size):
            block = Block(frozenset(relation_names))
            if is_valid_block(query, block):
                blocks.append(block)

    blocks.sort(key=_block_sort_key)

    if len(blocks) > max_blocks_warn:
        LOGGER.warning(
            "Enumerated %d valid blocks for query %r with max_block_size=%d; "
            "this exceeds warning threshold %d",
            len(blocks),
            query.name,
            max_block_size,
            max_blocks_warn,
        )

    return tuple(blocks)


def _block_sort_key(block: Block) -> tuple[int, tuple[str, ...]]:
    return len(block), tuple(sorted(block.relations))
