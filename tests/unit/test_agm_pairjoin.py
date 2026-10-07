"""AGM over pair-join-enriched hypergraphs (the 'known trick' baseline)."""

from __future__ import annotations

from scripts.agm_pairjoin_baseline import agm_with_pair_joins, pair_join_size
from scripts.statistics_hierarchy import FAMILIES, build
from src.estimators.agm_bound import estimate_agm


def test_pair_join_size_matches_brute_force() -> None:
    _query, data = build(FAMILIES["triangle"], seed=7)
    (ca, ra), (cb, rb) = data["R0"], data["R1"]
    common = [c for c in ca if c in cb]
    brute = sum(
        1
        for x in ra
        for y in rb
        if all(x[ca.index(c)] == y[cb.index(c)] for c in common)
    )
    assert pair_join_size(data, "R0", "R1") == brute


def test_enriched_agm_never_looser_than_agm() -> None:
    for fam in ("triangle", "cycle4", "k4"):
        query, data = build(FAMILIES[fam], seed=11)
        assert agm_with_pair_joins(query, data) <= estimate_agm(query, data) + 1e-6


def test_enriched_agm_bounded_by_any_covering_pair_join() -> None:
    # on the triangle {R0, R1} already covers every attribute, so the enriched
    # AGM can pick that hyperedge alone
    query, data = build(FAMILIES["triangle"], seed=3)
    assert agm_with_pair_joins(query, data) <= pair_join_size(data, "R0", "R1") + 1e-6
