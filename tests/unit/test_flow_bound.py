"""Tests for the flow-bound inference over block statistics."""

from __future__ import annotations

import itertools
import math

import pytest

from scripts import statistics_hierarchy as sh
from src.estimators.flow_bound import (
    DegreeConstraint,
    block_constraints,
    cyclic_orders,
    flow_bound,
    flow_bound_over_orders,
    polymatroid_bound,
)
from src.estimators.lpbound_entropic import estimate_lpbound_entropic

INF = math.inf


def _path_constraints(n_r: int, n_s: int, maxdeg: int) -> list[DegreeConstraint]:
    ab, bc = frozenset("ab"), frozenset("bc")
    return [
        DegreeConstraint(frozenset(), ab, 1.0, math.log2(n_r)),
        DegreeConstraint(frozenset(), bc, 1.0, math.log2(n_s)),
        DegreeConstraint(frozenset("b"), bc, INF, math.log2(maxdeg)),
    ]


class TestFlowBoundAnalytic:
    def test_two_relation_path_takes_cheaper_route_to_c(self) -> None:
        # h(abc) <= h(ab) + min(h(bc), h(c|b)) = log R + min(log S, log maxdeg)
        res = flow_bound("abc", _path_constraints(100, 50, 5))
        assert res.log_bound == pytest.approx(math.log2(100) + math.log2(5))
        res = flow_bound("abc", _path_constraints(100, 3, 5))
        assert res.log_bound == pytest.approx(math.log2(100) + math.log2(3))

    def test_unreachable_sink_is_infinite(self) -> None:
        cons = [DegreeConstraint(frozenset(), frozenset("ab"), 1.0, 3.0)]
        assert flow_bound("abc", cons).log_bound == INF

    def test_lp_norm_bundle_matches_polymatroid(self) -> None:
        # (1/2) h(b) + h(bc|b) <= log ||deg||_2: the bundle edge ∅ -> b of
        # capacity delta/2 is exactly what the Shannon-cone LP allows.
        cons = [
            *_path_constraints(100, 50, 5),
            DegreeConstraint(frozenset("b"), frozenset("bc"), 2.0, math.log2(12)),
        ]
        assert flow_bound("abc", cons).log_bound == pytest.approx(
            polymatroid_bound("abc", cons)
        )


class TestFlowBoundOnData:
    @pytest.fixture(scope="class")
    def triangle(self):  # type: ignore[no-untyped-def]
        q, d = sh.build(sh.FAMILIES["triangle"], seed=101)
        return q, d

    def test_bs1_equals_entropic_lpbound(self, triangle) -> None:  # type: ignore[no-untyped-def]
        q, d = triangle
        cons = block_constraints(q, d, max_block_size=1, p_values=sh.P_VALUES)
        ent = math.log2(estimate_lpbound_entropic(q, d, p_values=sh.P_VALUES))
        assert flow_bound(sorted(q.all_attributes), cons).log_bound == pytest.approx(
            ent, abs=1e-6
        )

    def test_simple_block_constraints_are_exact(self, triangle) -> None:  # type: ignore[no-untyped-def]
        q, d = triangle
        attrs = sorted(q.all_attributes)
        cons = block_constraints(q, d, max_block_size=2, p_values=sh.P_VALUES)
        assert any(not c.is_simple for c in cons)
        simple = [c for c in cons if c.is_simple]
        assert flow_bound(attrs, cons).log_bound == pytest.approx(
            polymatroid_bound(attrs, simple), abs=1e-6
        )

    def test_every_order_is_valid_and_best_order_is_tight(self, triangle) -> None:  # type: ignore[no-untyped-def]
        q, d = triangle
        attrs = sorted(q.all_attributes)
        cons = block_constraints(q, d, max_block_size=2, p_values=sh.P_VALUES)
        poly = polymatroid_bound(attrs, cons)
        exact = math.log2(sh.true_count(q, d))
        for order in itertools.permutations(attrs):
            for prefix in (False, True):
                res = flow_bound(attrs, cons, order=order, prefix_sources=prefix)
                assert res.log_bound >= poly - 1e-9
                assert res.log_bound >= exact - 1e-9
        best, order = flow_bound_over_orders(attrs, cons, cyclic_orders(attrs))
        assert order is not None
        assert best.log_bound == pytest.approx(poly, abs=1e-6)
        assert best.n_nonsimple_used > 0

    def test_order_with_interfaces_last_uses_no_nonsimple(self, triangle) -> None:  # type: ignore[no-untyped-def]
        q, d = triangle
        attrs = sorted(q.all_attributes)
        cons = [
            c
            for c in block_constraints(q, d, max_block_size=2, p_values=sh.P_VALUES)
            if c.is_simple or c.conditioning == frozenset("bc")
        ]
        # 'a' precedes both interface attributes, so Y' is empty
        res = flow_bound(attrs, cons, order=("a", "b", "c"))
        assert res.n_nonsimple_used == 0
        assert res.log_bound == pytest.approx(
            flow_bound(attrs, cons).log_bound, abs=1e-9
        )


def test_empty_block_join_gives_zero_bound() -> None:
    """A block whose join is empty certifies an empty query: bound 0."""
    from src.estimators.flow_bound import (
        block_constraints,
        flow_bound,
        flow_bound_over_orders,
        polymatroid_bound,
    )
    from src.utils.types import QueryHypergraph, Relation

    # triangle R(a,b), S(b,c), T(a,c) where R and S share no b value
    data = {
        "R": (("a", "b"), [("1", "x"), ("2", "x")]),
        "S": (("b", "c"), [("y", "5"), ("y", "6")]),
        "T": (("a", "c"), [("1", "5"), ("2", "6")]),
    }
    q = QueryHypergraph(
        relations=(
            Relation("R", frozenset({"a", "b"}), 2),
            Relation("S", frozenset({"b", "c"}), 2),
            Relation("T", frozenset({"a", "c"}), 2),
        ),
        name="empty-join",
    )
    cons2 = block_constraints(q, data, max_block_size=2)
    assert any(c.log_norm == float("-inf") for c in cons2)
    assert flow_bound(["a", "b", "c"], cons2).bound == 0.0
    orders = [("a", "b", "c"), ("b", "a", "c")]
    assert flow_bound_over_orders(["a", "b", "c"], cons2, orders)[0].bound == 0.0
    assert polymatroid_bound(["a", "b", "c"], cons2) == float("-inf")
    # single-relation statistics are non-empty, so bs=1 stays positive
    cons1 = block_constraints(q, data, max_block_size=1)
    assert all(c.log_norm > float("-inf") for c in cons1)
    assert flow_bound(["a", "b", "c"], cons1).bound > 0.0


def test_guaranteed_bound_returns_the_polymatroid_optimum() -> None:
    """Small queries get the polymatroid value; the flow value is only a candidate."""
    from src.estimators.flow_bound import guaranteed_block_bound

    cons = _path_constraints(100, 80, 3)
    attrs = ("a", "b", "c")
    res = guaranteed_block_bound(attrs, cons, cyclic_orders(attrs), 99.0)
    assert res.method == "polymatroid"
    assert res.log_bound == pytest.approx(polymatroid_bound(attrs, cons))
    assert res.flow_certified is True  # simple constraints: flow is exact


def test_guaranteed_bound_falls_back_to_the_cover_bound() -> None:
    """Beyond the polymatroid size limit the caller's cover bound is returned."""
    from src.estimators.flow_bound import guaranteed_block_bound

    cons = _path_constraints(100, 80, 3)
    attrs = ("a", "b", "c")
    res = guaranteed_block_bound(
        attrs, cons, cyclic_orders(attrs), 42.0, max_polymatroid_vars=2
    )
    assert res.method == "cover_fallback"
    assert res.fallback_reason == "size"
    assert res.log_bound == 42.0
    assert res.flow_certified is None


def test_guaranteed_bound_without_the_flow_candidate() -> None:
    """The flow candidate is optional and never needed for the return."""
    from src.estimators.flow_bound import guaranteed_block_bound

    cons = _path_constraints(100, 80, 3)
    attrs = ("a", "b", "c")
    res = guaranteed_block_bound(attrs, cons, None, 99.0)
    assert res.method == "polymatroid"
    assert res.flow_log_bound is None and res.flow_certified is None


def test_guaranteed_bound_falls_back_when_the_polymatroid_lp_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A solver failure on a small query returns the cover bound, not an error."""
    import src.estimators.flow_bound as fb

    def fail(*args: object, **kwargs: object) -> float:
        raise ValueError("polymatroid LP failed: test")

    monkeypatch.setattr(fb, "polymatroid_bound", fail)
    cons = _path_constraints(100, 80, 3)
    res = fb.guaranteed_block_bound(("a", "b", "c"), cons, None, 7.0)
    assert res.method == "cover_fallback"
    assert res.fallback_reason == "solver"
    assert res.log_bound == 7.0
