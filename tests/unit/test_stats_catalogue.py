"""Tests for the amortised statistics catalogue (scripts/stats_catalogue.py)."""

from __future__ import annotations

from scripts.stats_catalogue import StatsCatalogue, canonical_key, relation_sources
from src.core.block import enumerate_blocks
from src.estimators.lph_ceg import estimate_lph_ceg
from src.utils.types import Block, QueryHypergraph, Relation

P = (1.0, 2.0, float("inf"))


def _pair_query(name: str, a: str, b: str, c: str) -> QueryHypergraph:
    return QueryHypergraph(
        relations=(
            Relation("R0_X", frozenset({a, b}), 4),
            Relation("R1_Y", frozenset({b, c}), 4),
        ),
        name=name,
    )


def test_canonical_key_is_relabel_invariant() -> None:
    q1 = _pair_query("q1", "v0", "v1", "v2")
    q2 = _pair_query("q2", "v5", "v3", "v9")
    src1 = {"R0_X": ("X", "v0", "v1"), "R1_Y": ("Y", "v1", "v2")}
    src2 = {"R0_X": ("X", "v5", "v3"), "R1_Y": ("Y", "v3", "v9")}
    blk = Block(frozenset({"R0_X", "R1_Y"}))
    assert canonical_key(blk, frozenset({"v0"}), src1) == canonical_key(
        blk, frozenset({"v5"}), src2
    )
    assert canonical_key(blk, frozenset({"v0"}), src1) != canonical_key(
        blk, frozenset({"v2"}), src1
    )
    assert q1.name != q2.name  # distinct queries share the key


def test_canonical_key_respects_orientation() -> None:
    blk = Block(frozenset({"R0_X", "R1_Y"}))
    fwd = {"R0_X": ("X", "v0", "v1"), "R1_Y": ("Y", "v1", "v2")}
    rev = {"R0_X": ("X", "v1", "v0"), "R1_Y": ("Y", "v1", "v2")}
    assert canonical_key(blk, frozenset(), fwd) != canonical_key(blk, frozenset(), rev)


def test_relation_sources_follows_native_orientation() -> None:
    q = {"relations": ["cbg", "gig", "cbg"], "var_types": ["C", "G", "G"]}
    # cbg is natively (C, G): relation 0 binds src->v0, tgt->v1; relation 2
    # joins v2 (G) to v0 (C), so it is transposed: src->v0, tgt->v2.
    src = relation_sources(q)
    assert src["R0_cbg"] == ("cbg", "v0", "v1")
    assert src["R2_cbg"] == ("cbg", "v0", "v2")


def test_catalogue_hits_reproduce_uncached_bound() -> None:
    rows_x = [("a1", "b1"), ("a1", "b2"), ("a2", "b2"), ("a3", "b3")]
    rows_y = [("b1", "c1"), ("b2", "c1"), ("b2", "c2"), ("b3", "c3")]
    rows_z = [("c1", "a1"), ("c2", "a2"), ("c3", "a3"), ("c1", "a3")]

    def tri(
        name: str, u: str, v: str, w: str
    ) -> tuple[
        QueryHypergraph,
        dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]],
        dict[str, tuple[str, str, str]],
    ]:
        q = QueryHypergraph(
            relations=(
                Relation("R0_X", frozenset({u, v}), 4),
                Relation("R1_Y", frozenset({v, w}), 4),
                Relation("R2_Z", frozenset({w, u}), 4),
            ),
            name=name,
        )
        data = {
            "R0_X": (
                tuple(sorted((u, v))),
                rows_x if u < v else [(b, a) for a, b in rows_x],
            ),
            "R1_Y": (
                tuple(sorted((v, w))),
                rows_y if v < w else [(b, a) for a, b in rows_y],
            ),
            "R2_Z": (
                tuple(sorted((w, u))),
                rows_z if w < u else [(b, a) for a, b in rows_z],
            ),
        }
        src = {"R0_X": ("X", u, v), "R1_Y": ("Y", v, w), "R2_Z": ("Z", w, u)}
        return q, data, src

    cat = StatsCatalogue(P)
    q1, d1, s1 = tri("t1", "v0", "v1", "v2")
    q2, d2, s2 = tri("t2", "v2", "v0", "v1")  # relabelled copy of t1
    b1 = enumerate_blocks(q1, max_block_size=2)
    b2 = enumerate_blocks(q2, max_block_size=2)
    st1, m1, h1 = cat.stats_for(q1, d1, b1, s1)
    st2, m2, h2 = cat.stats_for(q2, d2, b2, s2)
    assert m1 == len(b1) and h1 == 0
    assert m2 == 0 and h2 == len(b2)
    ref = estimate_lph_ceg(q2, d2, max_block_size=2, p_values=P).estimate
    got = estimate_lph_ceg(q2, d2, max_block_size=2, p_values=P, lp_stats=st2).estimate
    assert got == ref
    assert (
        estimate_lph_ceg(q1, d1, max_block_size=2, p_values=P, lp_stats=st1).estimate
        == ref
    )
