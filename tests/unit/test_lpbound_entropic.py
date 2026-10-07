"""Validation tests for the real entropic (polymatroid) LpBound baseline.

These tests enforce the correctness gates required for the entropic LpBound
to serve as an honest comparison baseline.  On small triangle / cycle
instances (where brute-force evaluation is feasible) we check:

* G1 — the bound is a valid upper bound (``>= true_count``);
* G2 — the bound is no looser than the AGM bound;
* G3 — the bound is tighter-or-equal than our block-cover LP (``bs=1``);
* G4 — the bound is deterministic across repeated runs.
"""

from __future__ import annotations

import math

import pytest

from scripts.large_scale_comparison import make_cycle_data
from src.estimators.lpbound_entropic import (
    MAX_ATTRIBUTES,
    estimate_lpbound_entropic,
)
from src.estimators.lph_ceg import estimate_lph_ceg
from src.statistics.block_stats import _join_block_relations
from src.utils.types import Block

P_VALUES: tuple[float, ...] = (1.0, 2.0, float("inf"))

# Small instances so the brute-force natural join is cheap.
SMALL_VERTICES = 40
SMALL_EDGES = 150
SEED = 42

# Generous floating-point slack for inequality gates.
REL_TOL = 1e-6


def _true_count(query, data) -> int:  # type: ignore[no-untyped-def]
    """Exact result cardinality via a full natural join (tiny inputs only)."""
    block = Block(frozenset(query.relation_names))
    rows, _ = _join_block_relations(query, block, data)
    return len(rows)


def _agm_bound(query, data) -> float:  # type: ignore[no-untyped-def]
    """AGM bound for a simple cycle: product of |R_i|^{1/2}.

    A simple cycle's optimal fractional edge cover assigns weight 1/2 to
    every edge (each attribute is covered by its two incident relations),
    so the AGM bound is ``prod_i |R_i|^{1/2}``.
    """
    agm = 1.0
    for relation in query.relations:
        card = len(set(data[relation.name][1]))
        agm *= card**0.5
    return agm


@pytest.mark.parametrize("k", [3, 4, 5])
class TestEntropicLpBoundGates:
    """Hard validation gates G1-G4 on triangle and cycle-4/5."""

    def _instance(self, k: int):  # type: ignore[no-untyped-def]
        return make_cycle_data(k, SMALL_VERTICES, SMALL_EDGES, alpha=1.0, seed=SEED)

    def test_g1_valid_upper_bound(self, k: int) -> None:
        """G1: entropic LpBound is a genuine upper bound on the true count."""
        query, data = self._instance(k)
        bound = estimate_lpbound_entropic(query, data, p_values=P_VALUES)
        true_count = _true_count(query, data)
        assert bound >= true_count - REL_TOL, (
            f"cycle-{k}: bound {bound} < true_count {true_count}"
        )

    def test_g2_at_most_agm(self, k: int) -> None:
        """G2: entropic LpBound never exceeds the AGM bound."""
        query, data = self._instance(k)
        bound = estimate_lpbound_entropic(query, data, p_values=P_VALUES)
        agm = _agm_bound(query, data)
        assert bound <= agm * (1.0 + REL_TOL), f"cycle-{k}: bound {bound} > AGM {agm}"

    def test_g3_tighter_than_cover(self, k: int) -> None:
        """G3: entropic LpBound is tighter-or-equal than our cover LP."""
        query, data = self._instance(k)
        bound = estimate_lpbound_entropic(query, data, p_values=P_VALUES)
        cover = estimate_lph_ceg(
            query,
            data,
            max_block_size=1,
            p_values=P_VALUES,
            apply_selectivity=False,
        ).estimate
        assert bound <= cover * (1.0 + REL_TOL), (
            f"cycle-{k}: entropic {bound} > cover {cover}"
        )

    def test_g4_deterministic(self, k: int) -> None:
        """G4: identical inputs yield identical bounds."""
        query, data = self._instance(k)
        first = estimate_lpbound_entropic(query, data, p_values=P_VALUES)
        second = estimate_lpbound_entropic(query, data, p_values=P_VALUES)
        assert first == second


class TestEntropicLpBoundBasics:
    """Structural / guardrail tests."""

    def test_positive_finite_bound(self) -> None:
        """The bound is positive and finite on a standard triangle."""
        query, data = make_cycle_data(3, SMALL_VERTICES, SMALL_EDGES, seed=SEED)
        bound = estimate_lpbound_entropic(query, data, p_values=P_VALUES)
        assert math.isfinite(bound)
        assert bound > 0.0

    def test_rejects_too_many_attributes(self) -> None:
        """Queries with more than MAX_ATTRIBUTES attributes are rejected."""
        # A cycle-k query has exactly k attributes.
        query, data = make_cycle_data(MAX_ATTRIBUTES + 1, 20, 40, seed=SEED)
        with pytest.raises(ValueError, match="intractable"):
            estimate_lpbound_entropic(query, data, p_values=P_VALUES)
