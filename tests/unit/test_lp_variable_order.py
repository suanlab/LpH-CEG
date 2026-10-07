"""Unit tests for LpH-CEG-guided WCOJ variable ordering."""

from __future__ import annotations

from src.utils.query import cycle_query, triangle_query
from src.wcoj.lp_variable_order import lp_variable_ordering
from src.wcoj.variable_order import variable_ordering

_TRIANGLE_DATA: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]] = {
    "R": (("a", "b"), [("1", "10"), ("1", "11"), ("2", "20")]),
    "S": (("b", "c"), [("10", "100"), ("11", "100"), ("20", "200")]),
    "T": (("c", "a"), [("100", "1"), ("200", "2")]),
}


class TestLpVariableOrdering:
    """Tests for lp_variable_ordering."""

    def test_returns_all_attributes(self) -> None:
        """Ordering must contain all query attributes exactly once."""
        q = triangle_query()
        ordering = lp_variable_ordering(q, _TRIANGLE_DATA, max_block_size=1)
        assert set(ordering) == set(q.all_attributes)
        assert len(ordering) == len(q.all_attributes)

    def test_deterministic(self) -> None:
        """Same input must produce same ordering."""
        q = triangle_query()
        o1 = lp_variable_ordering(q, _TRIANGLE_DATA, max_block_size=1)
        o2 = lp_variable_ordering(q, _TRIANGLE_DATA, max_block_size=1)
        assert o1 == o2

    def test_cycle4_all_attributes(self) -> None:
        """LP ordering on cycle-4 must cover all 4 attributes."""
        q = cycle_query(4)
        data = {
            "R1": (("a1", "a2"), [("1", "2"), ("2", "3"), ("3", "1")]),
            "R2": (("a2", "a3"), [("2", "3"), ("3", "1"), ("1", "2")]),
            "R3": (("a3", "a4"), [("3", "1"), ("1", "2"), ("2", "3")]),
            "R4": (("a1", "a4"), [("1", "1"), ("2", "2"), ("3", "3")]),
        }
        ordering = lp_variable_ordering(q, data, max_block_size=1)
        assert set(ordering) == {"a1", "a2", "a3", "a4"}


class TestLpVsStandardOrdering:
    """Compare LP ordering against standard H-CEG ordering."""

    def test_may_differ_on_cycle(self) -> None:
        """On cyclic queries, LP and standard orderings may differ."""
        q = cycle_query(4)
        data = {
            "R1": (("a1", "a2"), [("1", "2"), ("2", "3"), ("3", "1"), ("1", "3")]),
            "R2": (("a2", "a3"), [("2", "3"), ("3", "1"), ("1", "2"), ("3", "2")]),
            "R3": (("a3", "a4"), [("3", "1"), ("1", "2"), ("2", "3"), ("1", "3")]),
            "R4": (("a1", "a4"), [("1", "1"), ("2", "2"), ("3", "3"), ("1", "3")]),
        }
        lp_order = lp_variable_ordering(q, data, max_block_size=1)
        std_order = variable_ordering(q, data, max_block_size=1)
        # Both must be valid (contain all attributes)
        assert set(lp_order) == set(q.all_attributes)
        assert set(std_order) == set(q.all_attributes)
        # They may or may not differ — we just verify both run correctly
