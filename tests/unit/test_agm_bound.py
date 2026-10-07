"""Validation tests for the true AGM fractional-edge-cover bound.

Hard gates on small instances where brute-force evaluation is feasible:

* G1 — the AGM bound is a valid upper bound (``>= true_count``);
* G2 — for an equal-cardinality triangle, AGM == ``N ** 1.5`` (closed form);
* G3 — determinism.
"""

from __future__ import annotations

import math

import pytest

from scripts.large_scale_comparison import make_cycle_data
from src.estimators.agm_bound import estimate_agm
from src.statistics.block_stats import _join_block_relations
from src.utils.query import triangle_query
from src.utils.types import Block

SMALL_VERTICES = 40
SMALL_EDGES = 150
SEED = 42
REL_TOL = 1e-6


def _true_count(query, data) -> int:  # type: ignore[no-untyped-def]
    """Exact result cardinality via a full natural join (tiny inputs only)."""
    block = Block(frozenset(query.relation_names))
    rows, _ = _join_block_relations(query, block, data)
    return len(rows)


@pytest.mark.parametrize("k", [3, 4, 5])
class TestAgmGates:
    """G1 / G3 on triangle and cycle-4/5."""

    def _instance(self, k: int):  # type: ignore[no-untyped-def]
        return make_cycle_data(k, SMALL_VERTICES, SMALL_EDGES, alpha=1.0, seed=SEED)

    def test_g1_valid_upper_bound(self, k: int) -> None:
        """G1: the AGM bound never undershoots the true count."""
        query, data = self._instance(k)
        agm = estimate_agm(query, data)
        true_count = _true_count(query, data)
        assert agm >= true_count - REL_TOL, (
            f"cycle-{k}: AGM {agm} < true_count {true_count}"
        )

    def test_g3_deterministic(self, k: int) -> None:
        """G3: identical inputs yield identical bounds."""
        query, data = self._instance(k)
        assert estimate_agm(query, data) == estimate_agm(query, data)


class TestAgmClosedForm:
    """G2: equal-cardinality triangle has AGM == N ** 1.5."""

    def test_equal_triangle_closed_form(self) -> None:
        """Three relations with equal |R| = N give AGM == N ** 1.5."""
        n = 12
        query = triangle_query()
        # Triangle: R(a,b), S(b,c), T(c,a).  Give each relation exactly N
        # distinct rows so |R| = |S| = |T| = N.
        rows = [(str(i), str(i + 1)) for i in range(n)]
        data = {
            "R": (("a", "b"), list(rows)),
            "S": (("b", "c"), list(rows)),
            "T": (("c", "a"), list(rows)),
        }
        # Sanity: all three relations really have N tuples.
        assert all(len(data[name][1]) == n for name in ("R", "S", "T"))

        agm = estimate_agm(query, data)
        expected = float(n) ** 1.5
        assert math.isclose(agm, expected, rel_tol=REL_TOL), (
            f"AGM {agm} != N**1.5 {expected}"
        )

    def test_valid_upper_bound_on_equal_triangle(self) -> None:
        """The closed-form instance still respects the upper-bound gate."""
        n = 12
        query = triangle_query()
        rows = [(str(i), str(i + 1)) for i in range(n)]
        data = {
            "R": (("a", "b"), list(rows)),
            "S": (("b", "c"), list(rows)),
            "T": (("c", "a"), list(rows)),
        }
        agm = estimate_agm(query, data)
        assert agm >= _true_count(query, data) - REL_TOL
