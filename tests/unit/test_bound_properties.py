"""Randomized property checks tying the four programs together.

On small random cycles (fixed seeds, exact join counts by enumeration):

* the entropic bound is the polymatroid optimum over single-relation
  statistics, and LpBound's flow LP reaches it (``flow1 == poly1``);
* the cover LP is a valid bound no tighter than that optimum
  (``cover1 >= poly1``), at block size 1 and 2;
* the order-relaxed flow LP over the ``bs<=2`` catalog never undercuts the
  polymatroid bound over the unrelaxed catalog (``flow2 >= poly2``), which
  is the per-query certificate the paper relies on for l_p constraints;
* every bound is at least the true count.
"""

from __future__ import annotations

import itertools
import math
import random

import pytest

from src.core.block import enumerate_blocks
from src.estimators.flow_bound import (
    block_constraints,
    cycle_candidate_orders,
    flow_bound,
    flow_bound_over_orders,
    polymatroid_bound,
)
from src.estimators.lph_ceg import estimate_lph_ceg
from src.statistics.lp_stats import compute_all_lp_block_extensions
from src.utils.query import cycle_query

P_SETS = ((1.0, 2.0, math.inf), (1.0, 2.0, 3.0, 4.0, math.inf))
TOL = 1e-6  # relative, on bounds


def _random_cycle(k: int, seed: int):  # type: ignore[no-untyped-def]
    rng = random.Random(seed)
    q = cycle_query(k)
    data = {}
    for rel in q.relations:
        attrs = tuple(sorted(rel.attributes))
        # skewed source degrees over a small domain
        rows = {
            (str(min(int(rng.paretovariate(1.1)), 6)), str(rng.randrange(7)))
            for _ in range(rng.randrange(8, 20))
        }
        data[rel.name] = (attrs, sorted(rows))
    return q, data


def _true_count(q, data) -> int:  # type: ignore[no-untyped-def]
    attrs = sorted(q.all_attributes)
    domain = sorted({v for _, rows in data.values() for row in rows for v in row})
    tables = [(attrs_r, set(rows)) for attrs_r, rows in data.values()]
    total = 0
    for values in itertools.product(domain, repeat=len(attrs)):
        m = dict(zip(attrs, values, strict=True))
        total += all(tuple(m[a] for a in ar) in rows for ar, rows in tables)
    return total


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("k", [3, 4])
def test_programs_order_and_validity(k: int, seed: int) -> None:
    q, data = _random_cycle(k, 100 * k + seed)
    truth = _true_count(q, data)
    attrs = sorted(q.all_attributes)
    orders = cycle_candidate_orders(sorted(attrs, key=lambda a: int(a.lstrip("a_"))))
    stats = compute_all_lp_block_extensions(
        q, enumerate_blocks(q, max_block_size=2), data, p_values=P_SETS[-1]
    )
    for ps in P_SETS:
        cons1 = block_constraints(
            q, data, max_block_size=1, p_values=ps, lp_stats=stats
        )
        cons2 = block_constraints(
            q, data, max_block_size=2, p_values=ps, lp_stats=stats
        )
        poly1 = 2.0 ** polymatroid_bound(attrs, cons1)
        poly2 = 2.0 ** polymatroid_bound(attrs, cons2)
        flow1 = flow_bound(attrs, cons1).bound
        flow2 = flow_bound_over_orders(attrs, cons2, orders)[0].bound
        cover1 = estimate_lph_ceg(
            q, data, max_block_size=1, p_values=ps, lp_stats=stats
        ).estimate
        cover2 = estimate_lph_ceg(
            q, data, max_block_size=2, p_values=ps, lp_stats=stats
        ).estimate

        assert flow1 == pytest.approx(poly1, rel=TOL)
        assert cover1 >= poly1 * (1 - TOL)
        assert poly2 <= poly1 * (1 + TOL)
        assert flow2 >= poly2 * (1 - TOL)
        for bound in (poly1, poly2, flow1, flow2, cover1, cover2):
            assert bound >= truth * (1 - TOL)
