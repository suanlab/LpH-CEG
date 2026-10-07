"""Verification of the Cyclic Improvement theorem on non-uniform cycles.

The theorem (main.tex, "Cyclic Improvement") was corrected to carry an
explicit boundary factor ``(N_1/N_k)^{1/3}`` that vanishes only for
uniform-cardinality cycles.  These tests lock in the two properties that
must hold regardless of cardinality uniformity:

1. The realized LpH-CEG bound is a *valid* upper bound (>= true count)
   even on deliberately non-uniform cycles -- i.e. the boundary term does
   not make the bound underestimate.
2. The bound ordering ``CEG_M >= cover-LP >= entropic-LpBound >= true``
   holds on both uniform and non-uniform cycles.
"""

from __future__ import annotations

import random

from src.core.block import enumerate_blocks
from src.estimators.lpbound_entropic import estimate_lpbound_entropic
from src.estimators.lph_ceg import estimate_lph_ceg
from src.estimators.pessimistic import estimate_pessimistic
from src.statistics.block_stats import _join_block_relations
from src.utils.query import cycle_query

P = (1.0, 2.0, float("inf"))


def _power_law_relation(
    attrs: tuple[str, str], n_vertices: int, n_edges: int, *, alpha: float, seed: int
) -> tuple[tuple[str, str], list[tuple[str, str]]]:
    rng = random.Random(seed)
    w = [1.0 / (i + 1) ** alpha for i in range(n_vertices)]
    tot = sum(w)
    cum: list[float] = []
    acc = 0.0
    for x in w:
        acc += x / tot
        cum.append(acc)

    def pick() -> str:
        r = rng.random()
        lo, hi = 0, len(cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] < r:
                lo = mid + 1
            else:
                hi = mid
        return str(lo)

    seen: set[tuple[str, str]] = set()
    tries = 0
    while len(seen) < n_edges and tries < n_edges * 40:
        a, b = pick(), pick()
        if a != b:
            seen.add((a, b))
        tries += 1
    return attrs, sorted(seen)


def _nonuniform_cycle4(seed: int = 7):
    """Cycle-4 where each relation has a different size and skew."""
    q = cycle_query(4)
    # deliberately different cardinalities and skews per relation
    specs = [(120, 600, 1.0), (90, 200, 2.0), (120, 800, 0.8), (60, 150, 2.5)]
    data: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]] = {}
    for i, rel in enumerate(q.relations):
        attrs = tuple(sorted(rel.attributes))
        nv, ne, alpha = specs[i]
        data[rel.name] = _power_law_relation(
            (attrs[0], attrs[1]), nv, ne, alpha=alpha, seed=seed + 13 * i
        )
    return q, data


def _true_count(q, data) -> int:
    blocks = enumerate_blocks(q, max_block_size=len(q.relations))
    full = max(blocks, key=lambda b: len(b.relations))
    rows, _ = _join_block_relations(q, full, data)
    return len(rows)


def test_nonuniform_cycle_bound_is_valid() -> None:
    """SB-1: bound stays a valid upper bound on non-uniform cycles."""
    q, data = _nonuniform_cycle4()
    cover = estimate_lph_ceg(
        q, data, max_block_size=1, p_values=P, apply_selectivity=False
    ).estimate
    true = _true_count(q, data)
    assert cover >= true - 1e-6, f"cover {cover} underestimates true {true}"


def test_nonuniform_cycle_bound_ordering() -> None:
    """SB-1: CEG_M >= cover-LP >= entropic >= true on a non-uniform cycle."""
    q, data = _nonuniform_cycle4()
    ceg = estimate_pessimistic(q, data, max_block_size=1).estimate
    cover = estimate_lph_ceg(
        q, data, max_block_size=1, p_values=P, apply_selectivity=False
    ).estimate
    ent = estimate_lpbound_entropic(q, data, p_values=P)
    true = _true_count(q, data)
    assert ceg >= cover - 1e-6
    assert cover >= ent - 1e-6
    assert ent >= true - 1e-6


def test_uniform_cycle_ordering() -> None:
    """Uniform cycle: boundary factor is 1; ordering still holds."""
    q = cycle_query(4)
    data = {}
    for i, rel in enumerate(q.relations):
        attrs = tuple(sorted(rel.attributes))
        data[rel.name] = _power_law_relation(
            (attrs[0], attrs[1]), 100, 400, alpha=1.0, seed=42 + 7 * i
        )
    ceg = estimate_pessimistic(q, data, max_block_size=1).estimate
    cover = estimate_lph_ceg(
        q, data, max_block_size=1, p_values=P, apply_selectivity=False
    ).estimate
    ent = estimate_lpbound_entropic(q, data, p_values=P)
    assert ceg >= cover >= ent >= 1.0
    # uniform skewed cycle should show real improvement
    assert ceg / cover > 1.5
