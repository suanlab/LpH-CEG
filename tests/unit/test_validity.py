"""Validity regression gate: the cover bound must never undercut the truth.

This is the fast, permanent version of ``scripts/validity_search.py`` (the
C0 gate). It exhaustively enumerates *every* domain-2 database for the
triangle (complete coverage: 3,375 databases) and checks that the LpH-CEG
estimate is a valid upper bound (>= exact join size) at every block size.
A single failure here falsifies the paper's "guaranteed upper bound" claim.
"""

from __future__ import annotations

import itertools

from scripts.wcoj_e2e_experiment import generic_join
from src.estimators.lph_ceg import estimate_lph_ceg
from src.utils.types import QueryHypergraph, Relation
from src.wcoj.evaluation import lexicographic_ordering

P_VALUES = (1.0, 2.0, float("inf"))
TRIANGLE = [("a", "b"), ("b", "c"), ("a", "c")]


def _query(edges: list[tuple[str, str]]) -> QueryHypergraph:
    rels = tuple(
        Relation(f"R{i}", frozenset({u, v}), 0) for i, (u, v) in enumerate(edges)
    )
    return QueryHypergraph(relations=rels, name="shape")


def test_bound_valid_on_all_domain2_triangles() -> None:
    """Exhaustive: estimate >= true for every domain-2 triangle database."""
    query = _query(TRIANGLE)
    pairs = [(str(x), str(y)) for x in range(2) for y in range(2)]
    per_rel = [
        combo
        for r in range(1, len(pairs) + 1)
        for combo in itertools.combinations(pairs, r)
    ]
    checked = 0
    for combo in itertools.product(per_rel, repeat=len(TRIANGLE)):
        data = {
            f"R{i}": (tuple(sorted((u, v))), list(combo[i]))
            for i, (u, v) in enumerate(TRIANGLE)
        }
        true = generic_join(query, data, lexicographic_ordering(query)).result_count
        if true == 0:
            continue
        checked += 1
        for bs in (1, 2, 3):
            est = estimate_lph_ceg(
                query, data, max_block_size=bs, p_values=P_VALUES
            ).estimate
            assert est >= true - 1e-6 * max(1.0, true), (
                f"validity violation: bs={bs} true={true} est={est} data={data}"
            )
    assert checked > 2000  # sanity: we really enumerated the space (nonzero joins)
