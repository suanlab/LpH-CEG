"""Unit tests for Lp-norm block extension statistics (src/statistics/lp_stats.py).

Tests the Lp-norm computation of block extension factor mu_p(F | I). Covers:
- Degree sequence collection matches hand-computed values
- Lp-norm computation for p=1, 2, inf
- Simple vs non-simple interface detection
- Consistency with legacy max-degree stats
- Empty join handling
- All-block bulk computation
"""

from __future__ import annotations

import math

import pytest

from src.statistics.lp_stats import (
    compute_all_lp_block_extensions,
    compute_lp_block_extension,
)
from src.utils.query import triangle_query
from src.utils.types import Block

# R(a,b) join S(b,c) = {(1,10,100), (1,11,100), (2,20,200)}
_TEST_DATA: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]] = {
    "R": (("a", "b"), [("1", "10"), ("1", "11"), ("2", "20")]),
    "S": (("b", "c"), [("10", "100"), ("11", "100"), ("20", "200")]),
    "T": (("c", "a"), [("100", "1"), ("200", "2")]),
}

_P_VALUES = (1.0, 2.0, float("inf"))


class TestDegreeSequence:
    """Tests that degree sequences are correctly collected."""

    def test_pair_block_interface_a(self) -> None:
        """deg({R,S}|{a}): a=1 → 2, a=2 → 1. Sorted desc: (2, 1)."""
        q = triangle_query()
        block = Block(frozenset({"R", "S"}))
        stats = compute_lp_block_extension(
            q, block, frozenset({"a"}), _TEST_DATA, p_values=_P_VALUES
        )
        assert stats.degree_sequence == (2, 1)
        assert stats.count == 2

    def test_empty_interface(self) -> None:
        """deg({R,S}|∅): one group → (3,)."""
        q = triangle_query()
        block = Block(frozenset({"R", "S"}))
        stats = compute_lp_block_extension(
            q, block, frozenset(), _TEST_DATA, p_values=_P_VALUES
        )
        assert stats.degree_sequence == (3,)
        assert stats.count == 1

    def test_singleton_block(self) -> None:
        """deg({R}|{a}): a=1 → 2, a=2 → 1. Same as binary."""
        q = triangle_query()
        block = Block(frozenset({"R"}))
        stats = compute_lp_block_extension(
            q, block, frozenset({"a"}), _TEST_DATA, p_values=_P_VALUES
        )
        assert stats.degree_sequence == (2, 1)

    def test_full_interface(self) -> None:
        """deg({R}|{a,b}): every group has exactly 1 extension → all 1s."""
        q = triangle_query()
        block = Block(frozenset({"R"}))
        stats = compute_lp_block_extension(
            q, block, frozenset({"a", "b"}), _TEST_DATA, p_values=_P_VALUES
        )
        assert all(d == 1 for d in stats.degree_sequence)


class TestLpNorms:
    """Tests that Lp-norms are correctly computed."""

    def test_l1_norm(self) -> None:
        """l1 = sum of degrees = join size projected onto extension attrs."""
        q = triangle_query()
        block = Block(frozenset({"R", "S"}))
        stats = compute_lp_block_extension(
            q, block, frozenset({"a"}), _TEST_DATA, p_values=_P_VALUES
        )
        # degree seq = (2, 1), l1 = 2 + 1 = 3
        assert stats.lp_norms[1.0] == pytest.approx(3.0)

    def test_l2_norm(self) -> None:
        """l2 = sqrt(sum d_i^2)."""
        q = triangle_query()
        block = Block(frozenset({"R", "S"}))
        stats = compute_lp_block_extension(
            q, block, frozenset({"a"}), _TEST_DATA, p_values=_P_VALUES
        )
        # degree seq = (2, 1), l2 = sqrt(4 + 1) = sqrt(5)
        assert stats.lp_norms[2.0] == pytest.approx(math.sqrt(5))

    def test_linf_norm(self) -> None:
        """l∞ = max degree."""
        q = triangle_query()
        block = Block(frozenset({"R", "S"}))
        stats = compute_lp_block_extension(
            q, block, frozenset({"a"}), _TEST_DATA, p_values=_P_VALUES
        )
        # degree seq = (2, 1), l∞ = 2
        assert stats.lp_norms[float("inf")] == pytest.approx(2.0)

    def test_linf_equals_max_extension(self) -> None:
        """l∞ norm must equal max_extension (legacy compatibility)."""
        q = triangle_query()
        block = Block(frozenset({"R", "S"}))
        stats = compute_lp_block_extension(
            q, block, frozenset({"a"}), _TEST_DATA, p_values=_P_VALUES
        )
        assert stats.lp_norms[float("inf")] == pytest.approx(float(stats.max_extension))

    def test_norm_ordering(self) -> None:
        """l1 ≥ l2 ≥ l∞ for any degree sequence (power mean inequality)."""
        q = triangle_query()
        block = Block(frozenset({"R", "S"}))
        stats = compute_lp_block_extension(
            q, block, frozenset({"a"}), _TEST_DATA, p_values=_P_VALUES
        )
        assert stats.lp_norms[1.0] >= stats.lp_norms[2.0]
        assert stats.lp_norms[2.0] >= stats.lp_norms[float("inf")]

    def test_on_the_fly_norm(self) -> None:
        """norm() method computes p=3 on the fly even if not precomputed."""
        q = triangle_query()
        block = Block(frozenset({"R", "S"}))
        stats = compute_lp_block_extension(
            q, block, frozenset({"a"}), _TEST_DATA, p_values=(1.0, float("inf"))
        )
        # p=3 not precomputed
        assert 3.0 not in stats.lp_norms
        # but norm() computes it: (2^3 + 1^3)^(1/3) = 9^(1/3)
        assert stats.norm(3.0) == pytest.approx(9.0 ** (1.0 / 3.0))


class TestSimpleInterface:
    """Tests the simple/non-simple interface detection."""

    def test_empty_interface_is_simple(self) -> None:
        q = triangle_query()
        block = Block(frozenset({"R"}))
        stats = compute_lp_block_extension(
            q, block, frozenset(), _TEST_DATA, p_values=_P_VALUES
        )
        assert stats.is_simple

    def test_single_attr_interface_is_simple(self) -> None:
        q = triangle_query()
        block = Block(frozenset({"R"}))
        stats = compute_lp_block_extension(
            q, block, frozenset({"a"}), _TEST_DATA, p_values=_P_VALUES
        )
        assert stats.is_simple

    def test_multi_attr_interface_is_not_simple(self) -> None:
        q = triangle_query()
        block = Block(frozenset({"R", "S"}))
        # interface {a, b} has size 2 → non-simple
        stats = compute_lp_block_extension(
            q, block, frozenset({"a", "b"}), _TEST_DATA, p_values=_P_VALUES
        )
        assert not stats.is_simple


class TestEmptyJoin:
    """Tests handling of empty join results."""

    def test_empty_join(self) -> None:
        """Disjoint relations produce empty join → zero stats."""
        q = triangle_query()
        empty_data: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]] = {
            "R": (("a", "b"), [("1", "10")]),
            "S": (("b", "c"), [("99", "100")]),  # no matching b values
            "T": (("c", "a"), [("100", "1")]),
        }
        block = Block(frozenset({"R", "S"}))
        stats = compute_lp_block_extension(
            q, block, frozenset({"a"}), empty_data, p_values=_P_VALUES
        )
        assert stats.degree_sequence == ()
        assert stats.count == 0
        assert stats.max_extension == 0
        assert stats.lp_norms[1.0] == 0.0
        assert stats.lp_norms[float("inf")] == 0.0


class TestBulkComputation:
    """Tests compute_all_lp_block_extensions."""

    def test_all_blocks(self) -> None:
        """Bulk computation returns stats for all block/interface pairs."""
        q = triangle_query()
        blocks = [
            Block(frozenset({"R"})),
            Block(frozenset({"S"})),
            Block(frozenset({"R", "S"})),
        ]
        all_stats = compute_all_lp_block_extensions(
            q, blocks, _TEST_DATA, p_values=_P_VALUES
        )
        # Each block has multiple interfaces
        assert len(all_stats) > 0

        # Spot check: {R,S} with interface {a}
        key = (Block(frozenset({"R", "S"})), frozenset({"a"}))
        assert key in all_stats
        assert all_stats[key].degree_sequence == (2, 1)

    def test_consistency_with_single(self) -> None:
        """Bulk results match single-block computation."""
        q = triangle_query()
        block = Block(frozenset({"R", "S"}))
        interface = frozenset({"a"})

        single = compute_lp_block_extension(
            q, block, interface, _TEST_DATA, p_values=_P_VALUES
        )
        bulk = compute_all_lp_block_extensions(
            q, [block], _TEST_DATA, p_values=_P_VALUES
        )

        bulk_stats = bulk[(block, interface)]
        assert bulk_stats.degree_sequence == single.degree_sequence
        assert bulk_stats.lp_norms == single.lp_norms
        assert bulk_stats.max_extension == single.max_extension


class TestLogNorm:
    """Tests log_norm helper."""

    def test_log_norm_positive(self) -> None:
        q = triangle_query()
        block = Block(frozenset({"R", "S"}))
        stats = compute_lp_block_extension(
            q, block, frozenset({"a"}), _TEST_DATA, p_values=_P_VALUES
        )
        # log2(2) = 1.0
        assert stats.log_norm(float("inf")) == pytest.approx(1.0)

    def test_log_norm_l2(self) -> None:
        q = triangle_query()
        block = Block(frozenset({"R", "S"}))
        stats = compute_lp_block_extension(
            q, block, frozenset({"a"}), _TEST_DATA, p_values=_P_VALUES
        )
        # log2(sqrt(5))
        assert stats.log_norm(2.0) == pytest.approx(math.log2(math.sqrt(5)))


class TestFilteredDegreeSequence:
    """Tests for predicate-aware filtered statistics (L4)."""

    def test_no_filter_matches_original(self) -> None:
        """With no predicates, filtered stats equal unfiltered."""
        from src.statistics.lp_stats import (
            compute_filtered_lp_block_extension,
        )

        q = triangle_query()
        block = Block(frozenset({"R"}))
        original = compute_lp_block_extension(
            q, block, frozenset({"a"}), _TEST_DATA, p_values=_P_VALUES
        )
        filtered = compute_filtered_lp_block_extension(
            q, block, frozenset({"a"}), _TEST_DATA, predicates=None, p_values=_P_VALUES
        )
        assert original.max_extension == filtered.max_extension
        assert original.degree_sequence == filtered.degree_sequence

    def test_filter_reduces_degrees(self) -> None:
        """Filtering on attribute values should reduce degree counts."""
        from src.statistics.lp_stats import (
            compute_filtered_lp_block_extension,
        )

        q = triangle_query()
        block = Block(frozenset({"R"}))
        # Only allow a="1" — filters out row (2, 20)
        filtered = compute_filtered_lp_block_extension(
            q,
            block,
            frozenset({"a"}),
            _TEST_DATA,
            predicates={"a": {"1"}},
            p_values=_P_VALUES,
        )
        # a=1 has degree 2 (b=10,11). a=2 filtered out.
        assert filtered.degree_sequence == (2,)
        assert filtered.max_extension == 2
        assert filtered.count == 1

    def test_filter_all_gives_empty(self) -> None:
        """Filtering everything out gives empty stats."""
        from src.statistics.lp_stats import (
            compute_filtered_lp_block_extension,
        )

        q = triangle_query()
        block = Block(frozenset({"R"}))
        filtered = compute_filtered_lp_block_extension(
            q,
            block,
            frozenset({"a"}),
            _TEST_DATA,
            predicates={"a": {"999"}},
            p_values=_P_VALUES,
        )
        assert filtered.degree_sequence == ()
        assert filtered.max_extension == 0


class TestHistogramApproximation:
    """Tests for histogram-based approximate Lp-norms (L2)."""

    def test_simple_histogram(self) -> None:
        """Basic histogram → Lp-norm computation."""
        from src.statistics.lp_stats import approximate_lp_norms_from_histogram

        # 100 values with degree 1, 20 with degree 5, 3 with degree 10
        hist = [(1, 100), (5, 20), (10, 3)]
        norms = approximate_lp_norms_from_histogram(hist, (1.0, 2.0, float("inf")))

        assert norms[float("inf")] == 10.0
        assert norms[1.0] == 100 * 1 + 20 * 5 + 3 * 10  # 230
        # l2 = sqrt(100*1 + 20*25 + 3*100) = sqrt(900) = 30
        assert norms[2.0] == pytest.approx(30.0)

    def test_empty_histogram(self) -> None:
        """Empty histogram gives zero norms."""
        from src.statistics.lp_stats import approximate_lp_norms_from_histogram

        norms = approximate_lp_norms_from_histogram([], (1.0, float("inf")))
        assert norms[1.0] == 0.0
        assert norms[float("inf")] == 0.0

    def test_stats_from_histogram(self) -> None:
        """lp_stats_from_histogram produces valid LpBlockExtensionStats."""
        from src.statistics.lp_stats import lp_stats_from_histogram

        hist = [(2, 5), (10, 1)]
        stats = lp_stats_from_histogram(hist, interface_size=1)

        assert stats.max_extension == 10
        assert stats.count == 6
        assert stats.avg_extension == pytest.approx((2 * 5 + 10 * 1) / 6)
        assert stats.is_simple  # interface_size=1

    def test_histogram_matches_exact(self) -> None:
        """Histogram norms match exact computation for known sequences."""
        from src.statistics.lp_stats import (
            _compute_lp_norm,
            approximate_lp_norms_from_histogram,
        )

        # degree sequence: (5, 5, 3, 3, 3, 1, 1)
        # histogram: [(5, 2), (3, 3), (1, 2)]
        seq = (5, 5, 3, 3, 3, 1, 1)
        hist = [(5, 2), (3, 3), (1, 2)]
        p_vals = (1.0, 2.0, 3.0, float("inf"))

        exact = {p: _compute_lp_norm(seq, p) for p in p_vals}
        approx = approximate_lp_norms_from_histogram(hist, p_vals)

        for p in p_vals:
            assert approx[p] == pytest.approx(exact[p], rel=1e-10)


class TestJointOptimization:
    """Tests for K-best hyperpath joint optimization (L3)."""

    def test_explore_k1_matches_default(self) -> None:
        """explore_k=1 should match default behavior."""
        from src.estimators.lph_ceg import estimate_lph_ceg

        q = triangle_query()
        r1 = estimate_lph_ceg(q, _TEST_DATA, max_block_size=1, explore_k=1)
        r_default = estimate_lph_ceg(q, _TEST_DATA, max_block_size=1)
        assert r1.estimate == pytest.approx(r_default.estimate, rel=1e-6)

    def test_explore_k10_no_worse(self) -> None:
        """explore_k=10 should be <= explore_k=1 (never worse)."""
        from src.estimators.lph_ceg import estimate_lph_ceg

        q = triangle_query()
        r1 = estimate_lph_ceg(q, _TEST_DATA, max_block_size=1, explore_k=1)
        r10 = estimate_lph_ceg(q, _TEST_DATA, max_block_size=1, explore_k=10)
        assert r10.estimate <= r1.estimate + 1e-6

    @pytest.mark.parametrize(("k", "bs"), [(6, 1), (6, 2), (8, 2)])
    def test_explore_k_never_looser_on_cycles(self, k: int, bs: int) -> None:
        """More candidates may only tighten: the shortest path is always kept.

        Regression: the candidate DFS once dropped the l_inf-shortest
        hyperpath, so explore_k>1 was looser on cycle-6/8 at bs=2.
        """
        import random

        from src.estimators.lph_ceg import estimate_lph_ceg
        from src.utils.query import cycle_query

        q = cycle_query(k)
        rng = random.Random(k * 10 + bs)
        data = {}
        for rel in q.relations:
            rows = {
                (str(int(rng.paretovariate(1.2)) % 40), str(rng.randrange(40)))
                for _ in range(160)
            }
            data[rel.name] = (tuple(sorted(rel.attributes)), sorted(rows))
        r1 = estimate_lph_ceg(q, data, max_block_size=bs, explore_k=1)
        r3 = estimate_lph_ceg(q, data, max_block_size=bs, explore_k=3)
        assert r3.estimate <= r1.estimate * (1 + 1e-9)

    def test_metadata_tracks_paths(self) -> None:
        """Metadata should report paths_explored and explore_k."""
        from src.estimators.lph_ceg import estimate_lph_ceg

        q = triangle_query()
        r = estimate_lph_ceg(q, _TEST_DATA, max_block_size=1, explore_k=5)
        assert "paths_explored" in r.metadata
        assert r.metadata["explore_k"] == 5
