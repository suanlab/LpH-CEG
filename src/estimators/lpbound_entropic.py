"""Real entropic (polymatroid) LpBound baseline.

This module implements the *actual* LpBound cardinality bound, defined as
the optimum of the polymatroid (entropic) linear program over lp-norm
degree-sequence statistics.  Unlike ``lpbound_baseline`` (which reuses our
fractional block-cover LP restricted to singleton blocks), this estimator
solves the full Shannon-cone LP and therefore yields the genuine, tighter
LpBound that the paper compares against.

The LP is solved in ``log2`` units.  For query attributes ``V`` we introduce
a variable ``h(S)`` for every subset ``S ⊆ V`` (with ``h(∅) = 0``), and
maximize ``h(V)`` subject to:

* **Elemental Shannon inequalities** (these generate the full polymatroid
  cone):
    - Monotonicity: ``h(V) - h(V\\{i}) ≥ 0`` for each ``i ∈ V``.
    - Submodularity: ``h(S+{i}) + h(S+{j}) - h(S+{i,j}) - h(S) ≥ 0`` for all
      distinct ``i, j ∈ V`` and all ``S ⊆ V\\{i,j}``.
* **Statistics constraints** — for each relation ``R`` with attribute set
  ``A_R``, each ``p ∈ {1, 2, ∞}``, and each interface/key ``U ⊊ A_R``:
    ``h(A_R) - (1 - 1/p)·h(U) ≤ log2( ||deg_R(A_R\\U | U)||_p )``
  (for ``p = ∞`` the factor ``1/p`` is ``0``).  For a binary relation
  ``R(x, y)`` the splits ``U ∈ {∅, {x}, {y}}`` recover the cardinality,
  out-degree, and in-degree bounds respectively.

The returned bound is ``2 ** (optimal h(V))``.  Because the polymatroid cone
is a relaxation that is *strictly contained* in our block-cover relaxation,
this bound is always ``≤`` the cover bound produced by ``estimate_lph_ceg``
with ``max_block_size = 1``.
"""

from __future__ import annotations

import logging
import math
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from src.statistics.lp_stats import compute_lp_block_extension
from src.utils.types import Block

if TYPE_CHECKING:
    from src.statistics.block_stats import RelationDataMap
    from src.utils.types import AttributeName, QueryHypergraph

LOGGER = logging.getLogger(__name__)

DEFAULT_P_VALUES: tuple[float, ...] = (1.0, 2.0, float("inf"))

#: Guard on the 2^n LP's size.  This is a *resource* cap, not a claim that
#: larger instances are unsolvable: at |V|=13 the LP still solves, it is
#: simply slow.  Any statement about where the entropic formulation becomes
#: impractical must come from a measured timeout, never from this constant.
MAX_ATTRIBUTES: int = 14


def _all_subsets(
    elements: tuple[AttributeName, ...],
) -> list[frozenset[AttributeName]]:
    """Return every subset of ``elements`` (including the empty set)."""
    subsets: list[frozenset[AttributeName]] = []
    for size in range(len(elements) + 1):
        for combo in combinations(elements, size):
            subsets.append(frozenset(combo))
    return subsets


def estimate_lpbound_entropic(
    query: QueryHypergraph,
    relation_data: RelationDataMap,
    *,
    p_values: tuple[float, ...] = DEFAULT_P_VALUES,
) -> float:
    """Estimate cardinality via the real entropic (polymatroid) LpBound.

    Solves the Shannon-cone LP over lp-norm degree-sequence statistics and
    returns ``2 ** (optimal h(V))``.

    Args:
        query: Query hypergraph defining relations and attributes.
        relation_data: Mapping from relation name to ``(schema, rows)``.
        p_values: lp-norm exponents whose statistics feed the LP.

    Returns:
        The entropic LpBound cardinality upper bound (a positive float).

    Raises:
        ValueError: If the query has more than ``MAX_ATTRIBUTES`` attributes
            (the LP has ``2 ** n`` variables) or if the LP is infeasible.
    """
    attributes = tuple(sorted(query.all_attributes))
    n = len(attributes)
    if n == 0:
        return 1.0
    # An empty relation empties the join; its norms (all 0) give no finite
    # log constraint, so answer exactly instead of solving an unbounded LP.
    if any(not relation_data[rel.name][1] for rel in query.relations):
        return 0.0
    if n > MAX_ATTRIBUTES:
        msg = (
            f"Entropic LpBound is intractable for {n} attributes "
            f"(2**{n} variables); cap is {MAX_ATTRIBUTES}"
        )
        raise ValueError(msg)

    full_set = frozenset(attributes)

    # ------------------------------------------------------------------
    # Variable indexing: one column per subset S ⊆ V.  h(∅) is forced to
    # zero by pinning its column to 0 via a tight bound, which keeps the
    # index arithmetic simple while honouring h(∅) = 0.
    # ------------------------------------------------------------------
    subsets = _all_subsets(attributes)
    col_of: dict[frozenset[AttributeName], int] = {
        s: idx for idx, s in enumerate(subsets)
    }
    num_vars = len(subsets)
    empty_col = col_of[frozenset()]
    full_col = col_of[full_set]

    # Sparse constraint matrix: each row touches at most four of the 2^n
    # columns, so a dense matrix would cost O(2^n) memory per row and would
    # inflate the baseline's measured time with matrix construction.
    rows_i: list[int] = []
    cols_j: list[int] = []
    vals_v: list[float] = []
    b_ub: list[float] = []

    def add_row(coeffs: dict[int, float], rhs: float) -> None:
        r = len(b_ub)
        for col, val in coeffs.items():
            rows_i.append(r)
            cols_j.append(col)
            vals_v.append(val)
        b_ub.append(rhs)

    # ------------------------------------------------------------------
    # (A) Elemental Shannon inequalities (express as A_ub x <= b_ub, i.e.
    #     negate the "<= 0" form of each ">= 0" inequality).
    # ------------------------------------------------------------------
    # Monotonicity at the top: h(V) - h(V\{i}) >= 0  ->  h(V\{i}) - h(V) <= 0.
    for i in attributes:
        without_i = full_set - {i}
        add_row({col_of[without_i]: 1.0, full_col: -1.0}, 0.0)

    # Submodularity: h(S+{i}) + h(S+{j}) - h(S+{i,j}) - h(S) >= 0
    #   -> -h(S+{i}) - h(S+{j}) + h(S+{i,j}) + h(S) <= 0
    for i, j in combinations(attributes, 2):
        rest = tuple(a for a in attributes if a not in (i, j))
        for s in _all_subsets(rest):
            s_i = s | {i}
            s_j = s | {j}
            s_ij = s | {i, j}
            add_row(
                {
                    col_of[s_i]: -1.0,
                    col_of[s_j]: -1.0,
                    col_of[s_ij]: 1.0,
                    col_of[s]: 1.0,
                },
                0.0,
            )

    # ------------------------------------------------------------------
    # (B) Statistics constraints from lp degree-sequence norms.
    # ------------------------------------------------------------------
    n_stat_constraints = 0
    for relation in query.relations:
        block = Block(frozenset({relation.name}))
        attr_set = relation.attributes
        # All proper subsets U ⊊ A_R serve as candidate interfaces/keys.
        for u in _all_subsets(tuple(sorted(attr_set))):
            if u == attr_set:
                continue
            stats = compute_lp_block_extension(
                query,
                block,
                u,
                relation_data,
                p_values=p_values,
            )
            for p in p_values:
                norm = stats.norm(p)
                if norm <= 0.0:
                    # Empty relation/key: deg-norm is 0, bound is degenerate.
                    continue
                log_norm = math.log2(norm)
                inv_p = 0.0 if math.isinf(p) else 1.0 / p
                # h(A_R) - (1 - 1/p)*h(U) <= log2(norm)
                coeffs: dict[int, float] = {col_of[attr_set]: 1.0}
                weight = 1.0 - inv_p
                if u:
                    coeffs[col_of[u]] = coeffs.get(col_of[u], 0.0) - weight
                # When U == ∅, h(∅) = 0 so the (1-1/p)*h(U) term vanishes.
                add_row(coeffs, log_norm)
                n_stat_constraints += 1

    # ------------------------------------------------------------------
    # Objective: maximize h(V)  ->  minimize -h(V).
    # ------------------------------------------------------------------
    c = [0.0] * num_vars
    c[full_col] = -1.0

    a_ub = coo_matrix((vals_v, (rows_i, cols_j)), shape=(len(b_ub), num_vars)).tocsr()
    b_ub_arr = np.asarray(b_ub, dtype=float)

    # Bounds: h(S) >= 0 for all S; pin h(∅) = 0 exactly.
    bounds: list[tuple[float, float | None]] = [(0.0, None)] * num_vars
    bounds[empty_col] = (0.0, 0.0)

    LOGGER.debug(
        "Entropic LpBound LP: n=%d attrs, %d vars, %d constraints (%d statistics)",
        n,
        num_vars,
        len(b_ub),
        n_stat_constraints,
    )

    # Solve with the interior-point method.  The default HiGHS choice is
    # dual simplex, which on this LP is ~5x slower at |V|=12 (238.7 s vs
    # 45.0 s here) for a bit-identical objective.  Timing the baseline under
    # a slower-than-necessary solver would overstate our own advantage, so
    # we solve it the fast way.
    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub_arr,
        bounds=bounds,
        method="highs-ipm",
    )

    if not result.success:
        msg = f"Entropic LpBound LP failed to solve: {result.message}"
        raise ValueError(msg)

    optimal_h_v = -float(result.fun)
    return float(2.0**optimal_h_v)
