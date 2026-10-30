"""Edge cases of the guaranteed path: bags, parallel atoms, fallbacks.

Each test pins one input class the released API must handle: duplicate
rows (relations are sets), parallel atoms over the same attribute pair,
the size and solver fallbacks of ``guaranteed_block_bound``, and degrees
large enough that ``d**p`` at p = 10 would overflow an integer-free path.
"""

from __future__ import annotations

import itertools
import math
import random

import pytest

import src.estimators.flow_bound as fb
from src.estimators.flow_bound import (
    block_constraints,
    guaranteed_block_bound,
    polymatroid_bound,
)
from src.estimators.lpbound_entropic import estimate_lpbound_entropic
from src.estimators.lph_ceg import estimate_lph_ceg
from src.utils.query import cycle_query
from src.utils.types import QueryHypergraph, Relation

P3 = (1.0, 2.0, math.inf)
P11 = (*(float(p) for p in range(1, 11)), math.inf)


def _random_data(q: QueryHypergraph, seed: int, n: int = 40, dom: int = 6):  # type: ignore[no-untyped-def]
    rng = random.Random(seed)
    return {
        r.name: (
            tuple(sorted(r.attributes)),
            [(str(rng.randrange(dom)), str(rng.randrange(dom))) for _ in range(n)],
        )
        for r in q.relations
    }


def _count(q: QueryHypergraph, data) -> int:  # type: ignore[no-untyped-def]
    """Brute-force join size under set semantics."""
    attrs = sorted(q.all_attributes)
    sets = {k: {tuple(r) for r in rows} for k, (_, rows) in data.items()}
    dom = sorted({v for rows in sets.values() for r in rows for v in r})
    n = 0
    for assign in itertools.product(dom, repeat=len(attrs)):
        a = dict(zip(attrs, assign, strict=True))
        if all(
            tuple(a[x] for x in schema) in sets[k] for k, (schema, _) in data.items()
        ):
            n += 1
    return n


def _poly(q: QueryHypergraph, data, p: tuple[float, ...] = P3) -> float:  # type: ignore[no-untyped-def]
    cons = block_constraints(q, data, max_block_size=2, p_values=p)
    return float(2.0 ** polymatroid_bound(sorted(q.all_attributes), cons))


def test_duplicate_rows_do_not_change_any_bound() -> None:
    q = cycle_query(4)
    data = _random_data(q, 3)
    dedup = {k: (a, list(dict.fromkeys(rows))) for k, (a, rows) in data.items()}
    bag = {k: (a, rows + rows[:10]) for k, (a, rows) in data.items()}
    for bs in (1, 2):
        assert estimate_lph_ceg(q, bag, max_block_size=bs).estimate == pytest.approx(
            estimate_lph_ceg(q, dedup, max_block_size=bs).estimate, rel=1e-9
        )
    assert _poly(q, bag) == pytest.approx(_poly(q, dedup), rel=1e-9)
    assert estimate_lpbound_entropic(q, bag) == pytest.approx(
        estimate_lpbound_entropic(q, dedup), rel=1e-9
    )


@pytest.mark.parametrize("seed", range(5))
def test_parallel_atoms_stay_valid(seed: int) -> None:
    """R(a,b), S(a,b) plus a triangle closing through c."""
    q = QueryHypergraph(
        (
            Relation("R", frozenset("ab"), 0, 1.0),
            Relation("S", frozenset("ab"), 0, 1.0),
            Relation("T", frozenset("bc"), 0, 1.0),
            Relation("U", frozenset("ac"), 0, 1.0),
        ),
        "parallel",
    )
    data = _random_data(q, seed, n=25, dom=5)
    truth = _count(q, data)
    assert estimate_lph_ceg(q, data, max_block_size=2).estimate >= truth - 1e-9
    assert _poly(q, data) >= truth * (1 - 1e-9)


def test_guaranteed_bound_falls_back_beyond_the_size_limit() -> None:
    q = cycle_query(4)
    data = _random_data(q, 1)
    cons = block_constraints(q, data, max_block_size=2, p_values=P3)
    cover = math.log2(estimate_lph_ceg(q, data, max_block_size=2).estimate)
    g = guaranteed_block_bound(
        q.all_attributes, cons, None, cover, max_polymatroid_vars=3
    )
    assert g.method == "cover_fallback"
    assert g.log_bound == cover


def test_guaranteed_bound_falls_back_when_the_solver_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q = cycle_query(4)
    data = _random_data(q, 2)
    cons = block_constraints(q, data, max_block_size=2, p_values=P3)
    cover = math.log2(estimate_lph_ceg(q, data, max_block_size=2).estimate)

    def fail(*a: object, **kw: object) -> float:
        raise ValueError("solver failed")

    monkeypatch.setattr(fb, "polymatroid_bound", fail)
    g = guaranteed_block_bound(q.all_attributes, cons, None, cover)
    assert g.method == "cover_fallback"
    assert g.log_bound == cover


def test_large_degrees_at_eleven_norms_stay_finite_and_valid() -> None:
    """A hub of degree 400 makes d**10 about 1e26: the bounds must stay
    finite (no overflow) and above the exact count."""
    q = cycle_query(3)
    names = [r.name for r in q.relations]
    hub = [("h", str(i)) for i in range(400)]
    data = {}
    for r in q.relations:
        schema = tuple(sorted(r.attributes))
        data[r.name] = (schema, [*hub, ("x", "y")])
    truth = _count_triangle(data, names)
    for bs in (1, 2):
        est = estimate_lph_ceg(q, data, max_block_size=bs, p_values=P11).estimate
        assert math.isfinite(est)
        assert est >= truth - 1e-9
    poly = _poly(q, data, P11)
    assert math.isfinite(poly)
    assert poly >= truth * (1 - 1e-9)


def _count_triangle(data, names: list[str]) -> int:  # type: ignore[no-untyped-def]
    """Exact triangle count without enumerating the full domain."""
    rel = {n: (data[n][0], {tuple(r) for r in data[n][1]}) for n in names}
    (s0, r0), (s1, r1), (s2, r2) = (rel[n] for n in names)
    attrs = sorted(set(s0) | set(s1) | set(s2))
    n = 0
    for t0 in r0:
        a = dict(zip(s0, t0, strict=True))
        for t1 in r1:
            b = dict(zip(s1, t1, strict=True))
            if any(a[k] != b[k] for k in a.keys() & b.keys()):
                continue
            ab = {**a, **b}
            if len(ab) == len(attrs) and tuple(ab[k] for k in s2) in r2:
                n += 1
    return n
