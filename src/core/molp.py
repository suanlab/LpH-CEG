"""MOLP / CEG_M for relations of any arity.

The Maximum-degree Optimization LP of Joglekar and Re, which Chen et al. (PVLDB
2022, Theorem 5.1) show equals the shortest path in the pessimistic CEG_M, has a
variable ``s_Z`` per attribute subset ``Z`` and, for every relation ``R`` and every
``X subset Y subset attr(R)``, the degree constraint

    s_{Y u E} <= s_{X u E} + log deg_R(Y | X),
    deg_R(Y | X) = max_x |pi_Y sigma_{X=x} R|,

plus projection constraints ``s_X <= s_Y`` for ``X subset Y``. Its optimum is the
shortest path from the empty set to the full attribute set in the graph whose edges
are ``Z -> Z u Y`` of weight ``log deg_R(Y | Z n Y)``: taking ``X = Z n Y`` is optimal
because ``deg_R(Y | X)`` can only shrink as ``X`` grows, and since every weight is
non-negative, a path never gains by projecting attributes away.

Unlike the singleton-block H-CEG chain (``estimate_pessimistic(max_block_size=1)``),
this admits steps that add only *part* of a relation (for example ``|pi_a R|``) and
steps with an empty interface, both of which can be strictly tighter. The repository's
``src.core.ceg_m`` computes the same quantity but only for binary relations; this
module is the n-ary version used as the CEG_M baseline on STATS-CEB and JOB-light.
"""

from __future__ import annotations

import heapq
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import TypeAlias

import pandas as pd

logger = logging.getLogger(__name__)

Rows: TypeAlias = Sequence[tuple[str, ...]]
# A derived-relation constraint: the block's attributes, and the degree of the block's
# join given an interface (a subset of those attributes), i.e. mu(F | I).
BlockFactor: TypeAlias = tuple[frozenset[str], Callable[[frozenset[str]], float]]
RelationData: TypeAlias = Mapping[str, tuple[tuple[str, ...], Rows]]


@dataclass(frozen=True)
class MOLPResult:
    """The MOLP bound and the edges of one optimal path.

    Attributes:
        bound: ``2 ** shortest_path_weight``; 0.0 when some relation is empty.
        path: ``(relation, added attributes, conditioning attributes, degree)``
            per step of an optimal path, in order.
        n_attributes: Number of query attributes (the state space is their power set).
        n_settled: Number of attribute subsets Dijkstra settled.
    """

    bound: float
    path: tuple[tuple[str, frozenset[str], frozenset[str], int], ...]
    n_attributes: int
    n_settled: int


def _subsets(attrs: tuple[str, ...]) -> list[tuple[str, ...]]:
    return [c for k in range(len(attrs) + 1) for c in combinations(attrs, k)]


def degree_table(
    attrs: tuple[str, ...], rows: Sequence[tuple[str, ...]]
) -> dict[tuple[frozenset[str], frozenset[str]], int]:
    """``deg_R(Y | X)`` for every ``X subset Y subset attr(R)`` with ``Y`` non-empty.

    Args:
        attrs: The relation's attribute names.
        rows: Its tuples (set semantics; callers encode bags with a row-id attribute).

    Returns:
        Mapping ``(X, Y) -> max_x |pi_Y sigma_{X=x} R|``.
    """
    frame = pd.DataFrame(list(rows), columns=list(attrs))
    table: dict[tuple[frozenset[str], frozenset[str]], int] = {}
    for y in _subsets(attrs):
        if not y:
            continue
        projected = frame[list(y)].drop_duplicates()
        for x in _subsets(y):
            if len(x) == len(y):
                continue
            if not x:
                deg = len(projected)
            else:
                deg = int(projected.groupby(list(x), sort=False).size().max())
            table[(frozenset(x), frozenset(y))] = deg
    return table


def estimate_molp(data: RelationData,
                  blocks: Sequence[BlockFactor] = ()) -> MOLPResult:
    """MOLP (= CEG_M) bound on the natural join of ``data``.

    With ``blocks``, MOLP runs over the schema augmented with derived block relations
    (Chen et al., PVLDB 2022, Sec. 5.1.1): each block contributes the edges
    ``Z -> Z u attr(F)`` of weight ``log mu(F | Z n attr(F))``, a valid degree
    constraint on the block's join. The search stays over attribute subsets, so it
    scales with the number of attributes, not of relations.

    Args:
        data: ``relation -> (attributes, rows)``; attributes with equal names join.
        blocks: Derived-relation constraints ``(attributes, degree-given-interface)``.

    Returns:
        The bound and an optimal path.
    """
    names = sorted(data)
    if any(not data[name][1] for name in names):
        return MOLPResult(0.0, (), 0, 0)
    all_attrs = sorted({a for name in names for a in data[name][0]})
    bit = {a: 1 << i for i, a in enumerate(all_attrs)}
    full = (1 << len(all_attrs)) - 1

    # Per relation: (mask(Y), mask(X)) -> log2 degree, keyed by Y then X.
    edges: list[tuple[str, int, dict[int, tuple[float, int]]]] = []
    for name in names:
        attrs, rows = data[name]
        by_y: dict[int, dict[int, tuple[float, int]]] = {}
        for (x, y), deg in degree_table(attrs, rows).items():
            ymask = sum(bit[a] for a in y)
            xmask = sum(bit[a] for a in x)
            by_y.setdefault(ymask, {})[xmask] = (math.log2(deg), deg)
        for ymask, by_x in by_y.items():
            edges.append((name, ymask, by_x))

    dist: dict[int, float] = {0: 0.0}
    back: dict[int, tuple[int, str, int, int, int]] = {}
    heap: list[tuple[float, int]] = [(0.0, 0)]
    settled: set[int] = set()
    block_masks = [(sum(bit[a] for a in attrs), attrs, fn) for attrs, fn in blocks]
    block_cache: dict[tuple[int, int], float] = {}

    while heap:
        d, state = heapq.heappop(heap)
        if state in settled:
            continue
        settled.add(state)
        if state == full:
            break
        for name, ymask, by_x in edges:
            if ymask & ~state == 0:
                continue
            xmask = state & ymask
            weight, deg = by_x[xmask]
            nxt = state | ymask
            nd = d + weight
            if nd < dist.get(nxt, math.inf) - 1e-12:
                dist[nxt] = nd
                back[nxt] = (state, name, ymask, xmask, deg)
                heapq.heappush(heap, (nd, nxt))
        for index, (fmask, fattrs, fn) in enumerate(block_masks):
            if fmask & ~state == 0:
                continue
            xmask = state & fmask
            ck = (index, xmask)
            if ck not in block_cache:
                iface = frozenset(a for a in fattrs if bit[a] & xmask)
                block_cache[ck] = float(fn(iface))
            bdeg = block_cache[ck]
            if bdeg <= 0:
                return MOLPResult(0.0, (), len(all_attrs), len(settled))
            nxt = state | fmask
            nd = d + math.log2(bdeg)
            if nd < dist.get(nxt, math.inf) - 1e-12:
                dist[nxt] = nd
                back[nxt] = (state, f"block{index}", fmask, xmask, round(bdeg))
                heapq.heappush(heap, (nd, nxt))

    path: list[tuple[str, frozenset[str], frozenset[str], int]] = []
    node = full
    while node:
        prev, name, ymask, xmask, deg = back[node]
        path.append((
            name,
            frozenset(a for a in all_attrs if bit[a] & ymask & ~prev),
            frozenset(a for a in all_attrs if bit[a] & xmask),
            deg,
        ))
        node = prev
    path.reverse()
    return MOLPResult(2.0 ** dist[full], tuple(path), len(all_attrs), len(settled))
