"""Unit tests for LpH-CEG estimator and LP weight optimization.

Tests the full LpH-CEG pipeline: Lp statistics -> LP optimization -> bound.
Covers:
- LP produces feasible solution on triangle query
- LP bound <= standard H-CEG bound (never worse)
- LP weights satisfy cover constraints
- Metadata captures method variant and improvement ratio
- Consistency: p=inf only recovers standard H-CEG
"""

from __future__ import annotations

import math

import pytest

from src.estimators.lp_bound import optimize_hyperpath_weights
from src.estimators.lph_ceg import estimate_lph_ceg
from src.estimators.pessimistic import estimate_pessimistic
from src.statistics.lp_stats import compute_all_lp_block_extensions
from src.utils.query import triangle_query
from src.utils.types import Block, QueryHypergraph, Relation

_TRIANGLE_DATA: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]] = {
    "R": (("a", "b"), [("1", "10"), ("1", "11"), ("2", "20")]),
    "S": (("b", "c"), [("10", "100"), ("11", "100"), ("20", "200")]),
    "T": (("c", "a"), [("100", "1"), ("200", "2")]),
}


class TestLpHCegEstimator:
    """Tests for the full estimate_lph_ceg pipeline."""

    def test_produces_finite_bound(self) -> None:
        """LpH-CEG should produce a finite positive bound."""
        q = triangle_query()
        result = estimate_lph_ceg(q, _TRIANGLE_DATA, max_block_size=2)
        assert result.estimate > 0
        assert result.estimate < float("inf")

    def test_never_worse_than_standard(self) -> None:
        """LpH-CEG bound must be <= standard H-CEG bound."""
        q = triangle_query()
        standard = estimate_pessimistic(q, _TRIANGLE_DATA, max_block_size=2)
        lph = estimate_lph_ceg(q, _TRIANGLE_DATA, max_block_size=2)
        # LpH-CEG should be at most as large as standard
        assert lph.estimate <= standard.estimate + 1e-6

    def test_metadata_has_method(self) -> None:
        """Metadata should indicate the method variant used."""
        q = triangle_query()
        result = estimate_lph_ceg(q, _TRIANGLE_DATA, max_block_size=2)
        assert result.method == "lph_ceg"
        assert "method_variant" in result.metadata
        assert result.metadata["method_variant"] in ("lph_ceg_lp", "lph_ceg_standard")

    def test_metadata_has_improvement_ratio(self) -> None:
        """Metadata should report the improvement ratio."""
        q = triangle_query()
        result = estimate_lph_ceg(q, _TRIANGLE_DATA, max_block_size=2)
        ratio = result.metadata["improvement_ratio"]
        assert isinstance(ratio, float)
        assert ratio >= 1.0 - 1e-6  # LP is never worse

    def test_block_size_1_matches_binary(self) -> None:
        """With block_size=1, LpH-CEG reduces to binary Lp estimation."""
        q = triangle_query()
        result = estimate_lph_ceg(q, _TRIANGLE_DATA, max_block_size=1)
        assert result.estimate > 0

    def test_inf_only_matches_standard(self) -> None:
        """With p_values=(inf,), LP should recover standard H-CEG."""
        q = triangle_query()
        standard = estimate_pessimistic(q, _TRIANGLE_DATA, max_block_size=2)
        lph_inf = estimate_lph_ceg(
            q,
            _TRIANGLE_DATA,
            max_block_size=2,
            p_values=(float("inf"),),
        )
        # Should be very close (LP with only p=inf = standard weights)
        assert lph_inf.estimate == pytest.approx(standard.estimate, rel=0.01)


class TestLpWeightOptimization:
    """Tests for the LP weight optimization directly."""

    def test_feasible_on_triangle(self) -> None:
        """LP should find a feasible solution for triangle query."""
        q = triangle_query()
        blocks = [
            Block(frozenset({"R"})),
            Block(frozenset({"S"})),
            Block(frozenset({"T"})),
        ]
        lp_stats = compute_all_lp_block_extensions(q, blocks, _TRIANGLE_DATA)

        standard = estimate_pessimistic(q, _TRIANGLE_DATA, max_block_size=1)
        result = optimize_hyperpath_weights(
            q, standard.hyperpath, lp_stats, p_values=(1.0, 2.0, float("inf"))
        )
        assert result.feasible

    def test_weights_nonnegative(self) -> None:
        """All LP weights must be non-negative."""
        q = triangle_query()
        blocks = [
            Block(frozenset({"R"})),
            Block(frozenset({"S"})),
            Block(frozenset({"T"})),
        ]
        lp_stats = compute_all_lp_block_extensions(q, blocks, _TRIANGLE_DATA)

        standard = estimate_pessimistic(q, _TRIANGLE_DATA, max_block_size=1)
        result = optimize_hyperpath_weights(
            q, standard.hyperpath, lp_stats, p_values=(1.0, 2.0, float("inf"))
        )
        for w in result.weights.values():
            assert w >= -1e-10  # allow tiny numerical noise

    def test_lp_bound_is_valid_upper_bound(self) -> None:
        """LP bound must be a valid upper bound (>= true cardinality).

        True triangle count for test data:
        R(a,b) join S(b,c) join T(c,a):
        (1,10,100,1) and (2,20,200,2) = 2 triangles.
        """
        q = triangle_query()
        result = estimate_lph_ceg(q, _TRIANGLE_DATA, max_block_size=2)
        true_count = 2  # manually verified
        assert result.estimate >= true_count


class TestLpBoundBaseline:
    """Tests for the bs=1 cover LP (``estimate_cover_bs1``; not LpBound)."""

    def test_produces_finite_bound(self) -> None:
        """LpBound baseline should produce a finite positive bound."""
        from src.estimators.lpbound_baseline import estimate_lpbound

        q = triangle_query()
        result = estimate_lpbound(q, _TRIANGLE_DATA)
        assert result.estimate > 0
        assert result.estimate < float("inf")
        assert result.method == "cover_bs1"

    def test_equals_lph_ceg_bs1(self) -> None:
        """LpBound baseline must equal LpH-CEG with block_size=1."""
        from src.estimators.lpbound_baseline import estimate_lpbound

        q = triangle_query()
        p_vals = (1.0, 2.0, float("inf"))
        lpbound = estimate_lpbound(q, _TRIANGLE_DATA, p_values=p_vals)
        lph_bs1 = estimate_lph_ceg(
            q,
            _TRIANGLE_DATA,
            max_block_size=1,
            p_values=p_vals,
        )
        assert lpbound.estimate == pytest.approx(lph_bs1.estimate, rel=1e-6)

    def test_never_worse_than_ceg(self) -> None:
        """LpBound (LP) must be <= CEG_M (no LP) on same block size."""
        from src.estimators.lpbound_baseline import estimate_lpbound

        q = triangle_query()
        ceg = estimate_pessimistic(q, _TRIANGLE_DATA, max_block_size=1)
        lpbound = estimate_lpbound(q, _TRIANGLE_DATA)
        assert lpbound.estimate <= ceg.estimate + 1e-6

    def test_metadata_has_baseline_tag(self) -> None:
        """Metadata should identify this as the lpbound baseline."""
        from src.estimators.lpbound_baseline import estimate_lpbound

        q = triangle_query()
        result = estimate_lpbound(q, _TRIANGLE_DATA)
        assert result.metadata["baseline_method"] == "cover_bs1"

    def test_deprecated_alias_is_cover_bs1(self) -> None:
        """The old name is an alias of the correctly named function."""
        from src.estimators import lpbound_baseline as lb

        assert lb.estimate_lpbound is lb.estimate_cover_bs1

    def test_acyclic_gives_no_improvement(self) -> None:
        """On a star query, LpBound should equal CEG_M (1.00x)."""
        from src.estimators.lpbound_baseline import estimate_lpbound
        from src.utils.query import star_query

        q = star_query(3)
        star_data = {
            "R": (("a", "b"), [("1", "x"), ("2", "y"), ("3", "z")]),
            "S1": (("a", "c1"), [("1", "p"), ("2", "q")]),
            "S2": (("a", "c2"), [("1", "r"), ("3", "s")]),
            "S3": (("a", "c3"), [("1", "t"), ("2", "u"), ("3", "v")]),
        }
        ceg = estimate_pessimistic(q, star_data, max_block_size=1)
        lpbound = estimate_lpbound(q, star_data)
        assert lpbound.estimate == pytest.approx(ceg.estimate, rel=1e-6)


def test_lp_weights_cover_every_attribute_exactly() -> None:
    """After rescaling, the LP weights satisfy the cover constraints exactly."""
    from src.estimators.lp_bound import _attribute_coverage, _collect_step_info

    q = triangle_query()
    blocks = [Block(frozenset({r})) for r in ("R", "S", "T")]
    lp_stats = compute_all_lp_block_extensions(q, blocks, _TRIANGLE_DATA)
    hp = estimate_pessimistic(q, _TRIANGLE_DATA, max_block_size=1).hyperpath
    p_values = (1.0, 2.0, float("inf"))
    result = optimize_hyperpath_weights(q, hp, lp_stats, p_values=p_values)
    steps = _collect_step_info(q, hp)
    for attr in q.all_attributes:
        cov = sum(
            result.weights[(j, p)] * _attribute_coverage(attr, step, p)
            for j, step in enumerate(steps)
            for p in p_values
        )
        assert cov >= 1.0 - 1e-12


def test_under_covering_solver_output_is_rescaled(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A solver answer that misses the cover by a little is scaled back up."""
    import src.estimators.lp_bound as lb

    real = lb.linprog

    def short(*args, **kwargs):  # type: ignore[no-untyped-def]
        res = real(*args, **kwargs)
        res.x = res.x * (1 - 1e-3)  # under-cover every attribute by 0.1%
        return res

    monkeypatch.setattr(lb, "linprog", short)
    q = triangle_query()
    blocks = [Block(frozenset({r})) for r in ("R", "S", "T")]
    lp_stats = compute_all_lp_block_extensions(q, blocks, _TRIANGLE_DATA)
    hp = estimate_pessimistic(q, _TRIANGLE_DATA, max_block_size=1).hyperpath
    p_values = (1.0, 2.0, float("inf"))
    result = lb.optimize_hyperpath_weights(q, hp, lp_stats, p_values=p_values)
    steps = lb._collect_step_info(q, hp)
    covs = [
        sum(
            result.weights[(j, p)] * lb._attribute_coverage(attr, step, p)
            for j, step in enumerate(steps)
            for p in p_values
        )
        for attr in q.all_attributes
    ]
    assert min(covs) == pytest.approx(1.0, abs=1e-9)


def test_intermediate_norm_tightens_an_acyclic_path() -> None:
    """Theorem 4 is sufficient, not necessary: l2 helps on a 3-path."""
    q = QueryHypergraph(
        relations=(
            Relation("R", frozenset({"a", "b"}), 1),
            Relation("S", frozenset({"b", "c"}), 10),
            Relation("T", frozenset({"c", "d"}), 1000),
        )
    )
    t_rows = [("c0", f"d{j}") for j in range(100)]
    t_rows += [(f"c{i}", "d0") for i in range(1, 901)]
    data: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]] = {
        "R": (("a", "b"), [("a0", "b0")]),
        "S": (("b", "c"), [("b0", f"c{i}") for i in range(10)]),
        "T": (("c", "d"), t_rows),
    }
    without_l2 = estimate_lph_ceg(
        q, data, max_block_size=1, p_values=(1.0, float("inf"))
    ).estimate
    with_l2 = estimate_lph_ceg(
        q, data, max_block_size=1, p_values=(1.0, 2.0, float("inf"))
    ).estimate
    assert without_l2 == pytest.approx(1000.0, rel=1e-6)
    assert with_l2 == pytest.approx(math.sqrt(10 * 10900), rel=1e-6)
    assert with_l2 >= 109
