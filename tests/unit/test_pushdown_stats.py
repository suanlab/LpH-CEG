"""Tests for join-free block statistics (``src/statistics/pushdown_stats``)."""

from __future__ import annotations

import math
import random

import pytest

from scripts import statistics_hierarchy as sh
from src.core.block import enumerate_blocks
from src.estimators.flow_bound import block_constraints, flow_bound_over_orders
from src.statistics.lp_stats import compute_all_lp_block_extensions
from src.statistics.pushdown_stats import (
    codegree_block_extension,
    codegree_interface,
    codegree_l2_upper_bound,
    compute_pushdown_block_extensions,
    compute_sparse_block_extensions,
    join_free_interfaces,
    sketch_codegree_l2,
)
from src.utils.types import Block, QueryHypergraph, Relation

P = (1.0, 2.0, math.inf)


def _query(*rels: tuple[str, str]) -> QueryHypergraph:
    return QueryHypergraph(
        tuple(Relation(n, frozenset(a), 0, 1.0) for n, a in rels), "t"
    )


def _rows(rng: random.Random, n: int, *ranges: int) -> list[tuple[str, ...]]:
    return [tuple(str(rng.randint(0, r)) for r in ranges) for _ in range(n)]


def _same(a: object, b: object) -> bool:
    return (
        a.count == b.count  # type: ignore[attr-defined]
        and a.max_extension == b.max_extension  # type: ignore[attr-defined]
        and all(
            a.norm(p) == pytest.approx(b.norm(p), rel=1e-12)  # type: ignore[attr-defined]
            for p in P
        )
    )


class TestClassification:
    def test_binary_block_only_the_codegree_interface_needs_the_join(self) -> None:
        q = _query(("R", "ab"), ("S", "bc"))
        kinds = join_free_interfaces(q, Block(frozenset({"R", "S"})))
        assert len(kinds) == 8
        assert [sorted(i) for i, k in kinds.items() if k is None] == [["a", "c"]]
        assert kinds[frozenset("b")] == "shared"
        assert kinds[frozenset("a")] == "one_side"
        assert kinds[frozenset()] == "one_side"
        assert codegree_interface(q, Block(frozenset({"R", "S"}))) == frozenset("ac")

    def test_rejects_non_pair_blocks(self) -> None:
        q = _query(("R", "ab"), ("S", "bc"), ("T", "ca"))
        with pytest.raises(ValueError):
            join_free_interfaces(q, Block(frozenset({"R", "S", "T"})))


class TestEquality:
    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_binary_block_matches_materialized_join(self, seed: int) -> None:
        rng = random.Random(seed)
        q = _query(("R", "ab"), ("S", "bc"))
        rows_r = _rows(rng, 300, 15, 12) + [("0", "0")] * 5  # duplicates
        rows_s = _rows(rng, 300, 20, 15)  # dangling b values on both sides
        data = {"R": (("a", "b"), rows_r), "S": (("b", "c"), rows_s)}
        block = Block(frozenset({"R", "S"}))
        full = compute_all_lp_block_extensions(q, [block], data, p_values=P)
        pushed = compute_pushdown_block_extensions(q, block, data, p_values=P)
        assert len(pushed) == 7
        for key, st in pushed.items():
            assert _same(st, full[key]), sorted(key[1])
        # one-side interfaces also reproduce the degree sequence itself
        a_stats = pushed[(block, frozenset("a"))]
        assert a_stats.degree_sequence == full[(block, frozenset("a"))].degree_sequence

    @pytest.mark.parametrize("seed", [4, 5, 6])
    def test_sparse_product_gives_every_interface(self, seed: int) -> None:
        rng = random.Random(seed)
        q = _query(("R", "ab"), ("S", "bc"))
        rows_r = _rows(rng, 400, 12, 10) + [("0", "0")] * 3  # duplicates
        rows_s = _rows(rng, 400, 14, 18)
        data = {"R": (("a", "b"), rows_r), "S": (("b", "c"), rows_s)}
        block = Block(frozenset({"R", "S"}))
        ps = (*(float(p) for p in range(1, 11)), math.inf)
        full = compute_all_lp_block_extensions(q, [block], data, p_values=ps)
        sparse = compute_sparse_block_extensions(q, block, data, p_values=ps)
        assert sparse.keys() == full.keys()
        for key, st in sparse.items():
            assert st.count == full[key].count, sorted(key[1])
            assert st.max_extension == full[key].max_extension
            for p in ps:
                assert st.norm(p) == pytest.approx(full[key].norm(p), rel=1e-12)
        ac = codegree_block_extension(q, block, data, p_values=ps)
        assert ac.degree_sequence == full[(block, frozenset("ac"))].degree_sequence

    def test_codegree_of_disjoint_relations_is_empty(self) -> None:
        q = _query(("R", "ab"), ("S", "bc"))
        data = {"R": (("a", "b"), [("1", "x")]), "S": (("b", "c"), [("y", "2")])}
        st = codegree_block_extension(q, Block(frozenset({"R", "S"})), data, p_values=P)
        assert st.count == 0 and st.norm(2.0) == 0.0

    def test_ternary_relations_with_two_shared_attributes(self) -> None:
        rng = random.Random(9)
        q = _query(("R", "abc"), ("S", "bcd"))
        data = {
            "R": (("a", "b", "c"), _rows(rng, 250, 6, 5, 5)),
            "S": (("b", "c", "d"), _rows(rng, 250, 5, 5, 8)),
        }
        block = Block(frozenset({"R", "S"}))
        kinds = join_free_interfaces(q, block)
        assert sum(k is None for k in kinds.values()) == 3
        full = compute_all_lp_block_extensions(q, [block], data, p_values=P)
        for key, st in compute_pushdown_block_extensions(
            q, block, data, p_values=P
        ).items():
            assert _same(st, full[key]), sorted(key[1])

    def test_single_relation_block_delegates(self) -> None:
        q = _query(("R", "ab"), ("S", "bc"))
        data = {"R": (("a", "b"), [("1", "2"), ("1", "3")]), "S": (("b", "c"), [])}
        block = Block(frozenset({"R"}))
        out = compute_pushdown_block_extensions(q, block, data, p_values=P)
        assert out[(block, frozenset("a"))].max_extension == 2


class TestCodegreeSurrogates:
    def _case(self, seed: int):  # type: ignore[no-untyped-def]
        rng = random.Random(seed)
        q = _query(("R", "ab"), ("S", "bc"))
        data = {
            "R": (("a", "b"), _rows(rng, 400, 30, 25)),
            "S": (("b", "c"), _rows(rng, 400, 25, 30)),
        }
        block = Block(frozenset({"R", "S"}))
        exact = compute_all_lp_block_extensions(q, [block], data, p_values=P)[
            (block, frozenset("ac"))
        ].norm(2.0)
        return q, block, data, exact

    @pytest.mark.parametrize("seed", [1, 2, 3, 4])
    def test_triangle_bound_is_an_upper_bound(self, seed: int) -> None:
        q, block, data, exact = self._case(seed)
        assert codegree_l2_upper_bound(q, block, data) >= exact * (1 - 1e-12)

    def test_sketch_is_close_with_enough_probes(self) -> None:
        q, block, data, exact = self._case(5)
        est, rel_se = sketch_codegree_l2(q, block, data, n_probes=256, seed=1)
        assert est == pytest.approx(exact, rel=0.15)
        assert rel_se < 0.15


class TestBoundRecovery:
    def test_l2_codegree_alone_recovers_full_bs2_on_triangle(self) -> None:
        q, d = sh.build(sh.FAMILIES["triangle"], seed=101)
        attrs = sorted(q.all_attributes)
        blocks = enumerate_blocks(q, max_block_size=2)
        stats = compute_all_lp_block_extensions(q, blocks, d, p_values=P)
        free = {
            k: v
            for k, v in stats.items()
            if len(k[0].relations) == 1
            or join_free_interfaces(q, k[0])[k[1]] is not None
        }
        cons_full = block_constraints(
            q, d, max_block_size=2, p_values=P, lp_stats=stats
        )
        cons_free = block_constraints(q, d, max_block_size=2, p_values=P, lp_stats=free)
        l2_join = [c for c in cons_full if c not in set(cons_free) and c.p == 2.0]
        orders = [tuple(attrs), tuple(reversed(attrs)), ("b", "a", "c")]
        full = flow_bound_over_orders(attrs, cons_full, orders)[0].log_bound
        only_free = flow_bound_over_orders(attrs, cons_free, orders)[0].log_bound
        free_l2 = flow_bound_over_orders(attrs, cons_free + l2_join, orders)[
            0
        ].log_bound
        assert only_free > full + 0.3
        assert free_l2 == pytest.approx(full, abs=1e-6)
