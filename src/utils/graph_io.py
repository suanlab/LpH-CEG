"""Graph I/O utilities for loading edge lists and relational data.

Supports SNAP edge list format and CSV format. Provides streaming
(chunked) reading for large files and metadata extraction (|V|, |E|,
directed/undirected detection).

Formats:
    SNAP edge list: Lines of ``src dst`` (tab or space separated).
        Comment lines start with ``#``. Metadata (nodes, edges,
        directed/undirected) may appear in comment headers.
    CSV: Configurable delimiter, optional header row.

References:
    Stanford Large Network Dataset Collection (SNAP):
    https://snap.stanford.edu/data/
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)

# Chunk size for streaming reads (number of tuples per chunk)
DEFAULT_CHUNK_SIZE: int = 100_000


@dataclass(frozen=True)
class GraphMetadata:
    """Metadata extracted from a loaded graph.

    Attributes:
        node_count: Number of distinct nodes (|V|).
        edge_count: Number of edges (|E|) — directed count.
        is_directed: Whether the graph is directed. Detected from SNAP
            headers or inferred by checking edge symmetry.
        source_path: Path to the source file.
        extra: Additional metadata from file headers.
    """

    node_count: int
    edge_count: int
    is_directed: bool
    source_path: Path
    extra: dict[str, str] = field(default_factory=dict)


def _detect_directed_from_header(header_lines: list[str]) -> bool | None:
    """Detect directed/undirected from SNAP comment header lines.

    Args:
        header_lines: Comment lines (with ``#`` stripped) from SNAP file.

    Returns:
        True if directed, False if undirected, None if indeterminate.
    """
    for line in header_lines:
        lower = line.lower()
        if "undirected" in lower:
            return False
        if "directed" in lower:
            return True
    return None


def _infer_directed(edges: list[tuple[str, ...]]) -> bool:
    """Infer directedness by checking edge symmetry.

    If every edge (u, v) has a reverse (v, u), treat as undirected.
    Otherwise directed.

    Args:
        edges: List of edge tuples (at least 2 columns each).

    Returns:
        True if directed (asymmetric), False if undirected (symmetric).
    """
    if not edges:
        return True  # Default to directed for empty graphs
    edge_set: set[tuple[str, str]] = set()
    for e in edges:
        edge_set.add((e[0], e[1]))
    return any((v, u) not in edge_set for u, v in edge_set)


def _compute_metadata(
    edges: list[tuple[str, ...]],
    path: Path,
    is_directed_hint: bool | None,
    extra: dict[str, str],
) -> GraphMetadata:
    """Build GraphMetadata from a loaded edge list.

    Args:
        edges: Loaded edge tuples.
        path: Source file path.
        is_directed_hint: Directedness from header (None = infer).
        extra: Extra metadata from headers.

    Returns:
        GraphMetadata with node/edge counts and directedness.
    """
    nodes: set[str] = set()
    for e in edges:
        if len(e) >= 2:
            nodes.add(e[0])
            nodes.add(e[1])

    is_directed = (
        is_directed_hint if is_directed_hint is not None else _infer_directed(edges)
    )

    return GraphMetadata(
        node_count=len(nodes),
        edge_count=len(edges),
        is_directed=is_directed,
        source_path=path,
        extra=extra,
    )


def load_snap_streamed(
    path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Generator[list[tuple[str, ...]], None, None]:
    """Stream SNAP edge list in chunks.

    Yields chunks of edge tuples from a SNAP-format file. Comment lines
    (starting with ``#``) are skipped. Supports both tab and space
    delimiters.

    Args:
        path: Path to the SNAP edge list file.
        chunk_size: Number of edges per yielded chunk.

    Yields:
        Lists of edge tuples, each list up to ``chunk_size`` long.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If chunk_size < 1.
    """
    if chunk_size < 1:
        msg = f"chunk_size must be >= 1, got {chunk_size}"
        raise ValueError(msg)

    path = Path(path)
    if not path.exists():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)

    chunk: list[tuple[str, ...]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = tuple(stripped.split())
            if len(parts) >= 2:
                chunk.append(parts)
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
    if chunk:
        yield chunk


def load_snap(
    path: Path,
) -> tuple[list[tuple[str, ...]], GraphMetadata]:
    """Load a SNAP edge list file.

    Reads the entire file, extracting edges and metadata from comment
    headers. Supports tab and space delimiters. Comment lines start
    with ``#``.

    Args:
        path: Path to the SNAP edge list file.

    Returns:
        Tuple of (edges, metadata) where edges is a list of string
        tuples and metadata contains |V|, |E|, directedness.

    Raises:
        FileNotFoundError: If path does not exist.

    Examples:
        >>> edges, meta = load_snap(Path("web-Google.txt"))
        >>> len(edges)
        5105039
        >>> meta.node_count
        875713
    """
    path = Path(path)
    if not path.exists():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)

    edges: list[tuple[str, ...]] = []
    header_lines: list[str] = []
    extra: dict[str, str] = {}

    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                comment_text = stripped.lstrip("#").strip()
                header_lines.append(comment_text)
                # Try to extract key: value metadata from comments
                kv_match = re.match(r"^(\w[\w\s]*\w)\s*:\s*(.+)$", comment_text)
                if kv_match:
                    extra[kv_match.group(1).strip()] = kv_match.group(2).strip()
                continue
            parts = tuple(stripped.split())
            if len(parts) >= 2:
                edges.append(parts)

    is_directed_hint = _detect_directed_from_header(header_lines)

    logger.info(
        "Loaded SNAP file %s: %d edges, directed_hint=%s",
        path,
        len(edges),
        is_directed_hint,
    )

    metadata = _compute_metadata(edges, path, is_directed_hint, extra)
    return edges, metadata


def load_csv_streamed(
    path: Path,
    has_header: bool = False,
    delimiter: str = ",",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Generator[list[tuple[str, ...]], None, None]:
    """Stream CSV file in chunks.

    Yields chunks of row tuples from a CSV file. Optionally skips
    the header row.

    Args:
        path: Path to the CSV file.
        has_header: Whether the first row is a header (skipped).
        delimiter: Column delimiter character.
        chunk_size: Number of rows per yielded chunk.

    Yields:
        Lists of row tuples, each list up to ``chunk_size`` long.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If chunk_size < 1.
    """
    if chunk_size < 1:
        msg = f"chunk_size must be >= 1, got {chunk_size}"
        raise ValueError(msg)

    path = Path(path)
    if not path.exists():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)

    chunk: list[tuple[str, ...]] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            next(reader, None)
        for row in reader:
            if not row:
                continue
            chunk.append(tuple(row))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


def load_csv(
    path: Path,
    has_header: bool = False,
    delimiter: str = ",",
) -> tuple[list[tuple[str, ...]], GraphMetadata]:
    """Load a CSV file as relational tuples.

    Reads the entire CSV file, treating each row as a tuple of string
    values. The first two columns are used for node counting and
    directedness inference (assuming edge-list semantics).

    Args:
        path: Path to the CSV file.
        has_header: Whether the first row is a header (skipped).
        delimiter: Column delimiter character.

    Returns:
        Tuple of (rows, metadata) where rows is a list of string
        tuples and metadata contains |V|, |E|, directedness.

    Raises:
        FileNotFoundError: If path does not exist.

    Examples:
        >>> rows, meta = load_csv(Path("edges.csv"), has_header=True)
        >>> meta.edge_count
        100
    """
    path = Path(path)
    if not path.exists():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)

    header_names: list[str] = []
    rows: list[tuple[str, ...]] = []

    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            header_row = next(reader, None)
            if header_row is not None:
                header_names = header_row
        for row in reader:
            if not row:
                continue
            rows.append(tuple(row))

    extra: dict[str, str] = {}
    if header_names:
        extra["header"] = delimiter.join(header_names)

    logger.info("Loaded CSV file %s: %d rows", path, len(rows))

    metadata = _compute_metadata(rows, path, None, extra)
    return rows, metadata
