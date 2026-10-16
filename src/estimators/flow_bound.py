"""Flow-bound inference over block statistics (polynomial-size LP).

Background.  The entropic LpBound solves a Shannon-cone LP with ``2**|V|``
variables (``lpbound_entropic``).  Im, Moseley, Ngo and Pruhs (PACMMOD 2025)
show that for *simple* degree constraints -- conditioning on at most one
attribute -- the same polymatroid bound is the optimum of a polynomial-size
network-flow LP, and LpBound (Zhang et al., SIGMOD 2025, ``LPflow``) extends
that LP to ``lp``-norm degree sequences, again exactly.  Both results are
stated for single-relation statistics.  This module applies the flow LP to
*block* statistics: the ``lp`` norms of the join of a block of relations
conditioned on an interface, the statistics the block-cover LP consumes.

A block constraint ``(I, Y, p, c)`` reads ``(1/p)·h(I) + h(Y | I) <= c`` with
``c = log2 ||deg(Y \\ I | I)||_p``.  It is *simple* when ``|I| <= 1``; those
enter the flow network exactly (an edge ``I -> Y`` of capacity ``delta``
plus, for ``p < inf``, an edge ``∅ -> I`` of capacity ``delta / p``).  A
constraint with ``|I| >= 2`` is non-simple; Im et al. (Thm. 1.3c) show such
constraints already make the polymatroid bound as hard as the general case.
Following their Section 7 "flow bound", a non-simple constraint is relaxed
along an attribute order ``pi`` to ``(I, Y', p, c)`` where ``Y'`` keeps the
attributes of ``Y`` that come after every attribute of ``I``; it carries no
flow but reduces the unit demand at each sink ``t ∈ Y'`` by ``delta``.  For
l_inf (degree) constraints the resulting value upper-bounds the polymatroid
bound of the relaxed constraint set, hence the true cardinality (Im et al.,
Thm. 1.5a).  For l_p constraints with ``p < inf`` this is not proved; the
paper certifies each reported bound a posteriori by checking it against
``polymatroid_bound`` over the unrelaxed constraints, which is valid for
every p (``scripts/poly_certify.py``, ``scripts/gcare_poly_certify.py``).

The module also ships the generic ``2**|V|`` polymatroid LP over the same
constraint list so both inferences can be compared on identical statistics.
All values are in ``log2`` units unless stated otherwise.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from src.core.block import enumerate_blocks
from src.statistics.lp_stats import DEFAULT_P_VALUES, compute_all_lp_block_extensions

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from src.statistics.block_stats import RelationDataMap
    from src.statistics.lp_stats import LpBlockExtensionStats
    from src.utils.types import AttributeName, Block, Interface, QueryHypergraph

AttrSet = frozenset[str]


@dataclass(frozen=True)
class DegreeConstraint:
    """One ``lp`` degree constraint ``(1/p)·h(I) + h(Y|I) <= log_norm``.

    Attributes:
        conditioning: The interface ``I`` (a proper subset of ``target``).
        target: The block attribute set ``Y``.
        p: The norm exponent (``math.inf`` for the max degree).
        log_norm: ``log2 ||deg(Y \\ I | I)||_p``.
    """

    conditioning: AttrSet
    target: AttrSet
    p: float
    log_norm: float

    @property
    def is_simple(self) -> bool:
        """Whether the constraint conditions on at most one attribute."""
        return len(self.conditioning) <= 1


@dataclass(frozen=True)
class FlowBoundResult:
    """Outcome of one flow-LP solve.

    Attributes:
        log_bound: Optimal objective in ``log2`` units (``inf`` if infeasible).
        n_variables: Number of LP columns.
        n_rows: Number of LP inequality rows.
        n_simple: Simple constraints used (with flow edges).
        n_nonsimple_used: Non-simple constraints with non-empty ``Y'``.
        solve_seconds: Wall-clock time of LP construction plus solve.
    """

    log_bound: float
    n_variables: int
    n_rows: int
    n_simple: int
    n_nonsimple_used: int
    solve_seconds: float

    @property
    def bound(self) -> float:
        """Cardinality upper bound ``2 ** log_bound``."""
        return float(2.0**self.log_bound)


def block_constraints(
    query: QueryHypergraph,
    relation_data: RelationDataMap,
    *,
    max_block_size: int = 2,
    p_values: tuple[float, ...] = DEFAULT_P_VALUES,
    lp_stats: dict[tuple[Block, Interface], LpBlockExtensionStats] | None = None,
) -> list[DegreeConstraint]:
    """Turn block ``lp`` statistics into a flat list of degree constraints.

    Every block of at most ``max_block_size`` relations contributes, for each
    proper interface ``I`` of its attribute set ``Y`` and each ``p``, the
    constraint ``(1/p)·h(I) + h(Y|I) <= log2 ||deg(Y\\I | I)||_p``.  A
    statistic with no positive norm at all certifies that the block join is
    empty, hence so is the query; it becomes one constraint with
    ``log_norm = -inf``, on which :func:`flow_bound` and
    :func:`polymatroid_bound` return a zero bound.  Individual zero norms of
    an otherwise non-empty statistic (a partial-``p`` surrogate) are skipped.

    Args:
        query: Query hypergraph.
        relation_data: Relation name to ``(schema, rows)``.
        max_block_size: Largest block to collect statistics for.
        p_values: Norm exponents.
        lp_stats: Precomputed statistics (e.g. from a catalogue); collected
            here when ``None``.

    Returns:
        The constraint list, deterministic in block/interface/p order.
    """
    blocks = enumerate_blocks(query, max_block_size)
    stats = (
        compute_all_lp_block_extensions(query, blocks, relation_data, p_values=p_values)
        if lp_stats is None
        else lp_stats
    )
    out: list[DegreeConstraint] = []
    for block in blocks:
        target = frozenset(query.attributes_of(block))
        for interface_size in range(len(target)):
            for combo in combinations(sorted(target), interface_size):
                interface = frozenset(combo)
                st = stats.get((block, interface))
                if st is None:
                    continue
                if all(st.norm(p) <= 0.0 for p in p_values) and not any(
                    v > 0.0 for v in st.lp_norms.values()
                ):
                    out.append(DegreeConstraint(interface, target, 1.0, -math.inf))
                    continue
                for p in p_values:
                    norm = st.norm(p)
                    if norm <= 0.0:
                        continue
                    out.append(DegreeConstraint(interface, target, p, math.log2(norm)))
    return out


def _inv_p(p: float) -> float:
    return 0.0 if math.isinf(p) else 1.0 / p


def flow_bound(
    attributes: Iterable[AttributeName],
    constraints: Sequence[DegreeConstraint],
    *,
    order: Sequence[AttributeName] | None = None,
    prefix_sources: bool = False,
) -> FlowBoundResult:
    """Solve the flow LP of Im et al. / LpBound over ``constraints``.

    Simple constraints (``|I| <= 1``) are modelled exactly.  Non-simple ones
    are relaxed along ``order`` as described in the module docstring; when
    ``order`` is ``None`` they are dropped, which makes the optimum exactly
    the polymatroid bound of the simple sub-family (Im et al., Thm. 1.1;
    Zhang et al., Thm. 4.5).

    Args:
        attributes: The query attributes ``V`` (each becomes a sink).
        constraints: Degree constraints over subsets of ``V``.
        order: Attribute order ``pi`` used to relax non-simple constraints.
        prefix_sources: Im et al.'s optional strengthening: the unit of flow
            for sink ``t`` may originate at ``∅`` or at any singleton that
            precedes ``t`` in ``order`` (the supplies sum to at most one).
            Requires ``order``.

    Returns:
        The LP optimum and its size.  ``log_bound`` is ``inf`` when some sink
        cannot receive a unit of flow (no usable constraints cover it), and
        ``-inf`` (bound ``0``) when a constraint certifies an empty join.
    """
    t0 = time.perf_counter()
    if any(c.log_norm == -math.inf for c in constraints):
        return FlowBoundResult(-math.inf, 0, 0, 0, 0, time.perf_counter() - t0)
    sinks = tuple(sorted(attributes))
    position = {a: i for i, a in enumerate(order)} if order is not None else None
    empty: AttrSet = frozenset()

    # --- edges with capacity in delta: (src, dst) -> {constraint idx: coef}
    cap_edges: dict[tuple[AttrSet, AttrSet], dict[int, float]] = {}
    # demand reduction at sink t: t -> {constraint idx}
    reduction: dict[str, list[int]] = {t: [] for t in sinks}
    n_simple = 0
    n_nonsimple_used = 0

    def add_cap(src: AttrSet, dst: AttrSet, j: int, coef: float) -> None:
        cap_edges.setdefault((src, dst), {})
        cap_edges[(src, dst)][j] = cap_edges[(src, dst)].get(j, 0.0) + coef

    for j, con in enumerate(constraints):
        inv_p = _inv_p(con.p)
        if con.is_simple:
            n_simple += 1
            add_cap(con.conditioning, con.target, j, 1.0)
            if con.conditioning and inv_p > 0.0:
                add_cap(empty, con.conditioning, j, inv_p)
            continue
        if position is None:
            continue
        last = max(position[a] for a in con.conditioning)
        y_prime = [a for a in con.target - con.conditioning if position[a] > last]
        if not y_prime:
            continue
        n_nonsimple_used += 1
        for attr in y_prime:
            reduction[attr].append(j)
        if inv_p > 0.0:
            add_cap(empty, con.conditioning, j, inv_p)

    # --- nodes: ∅, singletons, every set touched by a capacitated edge
    node_set: set[AttrSet] = {empty, *(frozenset({a}) for a in sinks)}
    for src, dst in cap_edges:
        node_set.add(src)
        node_set.add(dst)
    nodes = sorted(node_set, key=lambda s: (len(s), tuple(sorted(s))))
    node_id = {s: i for i, s in enumerate(nodes)}
    # uncapacitated monotonicity edges B -> A for A ⊊ B
    free_edges = [(b, a) for a in nodes for b in nodes if a < b]
    edges: list[tuple[AttrSet, AttrSet]] = [*cap_edges, *free_edges]
    n_cap = len(cap_edges)

    m = len(constraints)
    n_e = len(edges)
    n_t = len(sinks)
    n_nodes = len(nodes)
    # supply variables s[u, t] for prefix sources (u = ∅ or a singleton
    # before sink t); column m + n_e*n_t + t*(n_t+1) + u_index
    use_supply = prefix_sources and position is not None
    n_vars = m + n_e * n_t + (n_t * (n_t + 1) if use_supply else 0)

    def fvar(e: int, t: int) -> int:
        return m + t * n_e + e

    def svar(u: int, t: int) -> int:  # u = 0 for ∅, 1 + sink index otherwise
        return m + n_e * n_t + t * (n_t + 1) + u

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs: list[float] = []
    row = 0

    def put(r: int, c: int, v: float) -> None:
        rows.append(r)
        cols.append(c)
        vals.append(v)

    # capacity: f[e,t] - sum_j coef_j delta_j <= 0
    for e, (src, dst) in enumerate(edges[:n_cap]):
        coefs = cap_edges[(src, dst)]
        for t in range(n_t):
            put(row, fvar(e, t), 1.0)
            for j, coef in coefs.items():
                put(row, j, -coef)
            rhs.append(0.0)
            row += 1

    # conservation: -excess_t(Z) <= b_Z  (b = 1 at ∅, -1 + reductions at t,
    # 0 elsewhere); excess = inflow - outflow.
    for t, sink in enumerate(sinks):
        sink_node = node_id[frozenset({sink})]
        base = row
        for e, (src, dst) in enumerate(edges):
            put(base + node_id[dst], fvar(e, t), -1.0)  # inflow to dst
            put(base + node_id[src], fvar(e, t), 1.0)  # outflow from src
        sources: dict[int, int] = {node_id[empty]: 0}
        if use_supply and position is not None:
            for u, other in enumerate(sinks):
                if position[other] < position[sink]:
                    sources[node_id[frozenset({other})]] = 1 + u
        for z in range(n_nodes):
            if z in sources and use_supply:
                put(base + z, svar(sources[z], t), -1.0)  # -excess <= s
                rhs.append(0.0)
            elif z == node_id[empty]:
                rhs.append(1.0)
            elif z == sink_node:
                for j in reduction[sink]:
                    put(base + z, j, -1.0)
                rhs.append(-1.0)
            else:
                rhs.append(0.0)
        row += n_nodes
        if use_supply:  # sum of supplies <= 1
            for u in sources.values():
                put(row, svar(u, t), 1.0)
            rhs.append(1.0)
            row += 1

    a_ub = coo_matrix((vals, (rows, cols)), shape=(row, n_vars)).tocsr()
    c = np.zeros(n_vars)
    for j, con in enumerate(constraints):
        c[j] = con.log_norm
    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=np.asarray(rhs),
        bounds=(0.0, None),
        method="highs",
    )
    if result.status == 2:  # infeasible: a sink is unreachable
        log_bound = math.inf
    elif not result.success:
        msg = f"flow LP failed: {result.message}"
        raise ValueError(msg)
    else:
        log_bound = float(result.fun)
    return FlowBoundResult(
        log_bound=log_bound,
        n_variables=n_vars,
        n_rows=row,
        n_simple=n_simple,
        n_nonsimple_used=n_nonsimple_used,
        solve_seconds=time.perf_counter() - t0,
    )


def polymatroid_bound(
    attributes: Iterable[AttributeName],
    constraints: Sequence[DegreeConstraint],
) -> float:
    """Maximise ``h(V)`` over the Shannon cone subject to ``constraints``.

    This is the ``2**|V|``-variable program of ``lpbound_entropic`` with an
    arbitrary constraint list, used as the exact reference for
    :func:`flow_bound`.

    Args:
        attributes: The query attributes ``V``.
        constraints: Degree constraints over subsets of ``V``.

    Returns:
        The optimum in ``log2`` units; ``-inf`` when a constraint certifies an
        empty join.

    Raises:
        ValueError: If the LP does not solve.
    """
    if any(c.log_norm == -math.inf for c in constraints):
        return -math.inf
    attrs = tuple(sorted(attributes))
    subsets: list[AttrSet] = [
        frozenset(cmb)
        for size in range(len(attrs) + 1)
        for cmb in combinations(attrs, size)
    ]
    col_of = {s: i for i, s in enumerate(subsets)}
    full = frozenset(attrs)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs: list[float] = []

    def add_row(coeffs: dict[int, float], b: float) -> None:
        r = len(rhs)
        for col, v in coeffs.items():
            rows.append(r)
            cols.append(col)
            vals.append(v)
        rhs.append(b)

    for i in attrs:  # h(V\i) - h(V) <= 0
        add_row({col_of[full - {i}]: 1.0, col_of[full]: -1.0}, 0.0)
    for i, j in combinations(attrs, 2):  # submodularity
        rest = tuple(a for a in attrs if a not in (i, j))
        for size in range(len(rest) + 1):
            for cmb in combinations(rest, size):
                s = frozenset(cmb)
                add_row(
                    {
                        col_of[s | {i}]: -1.0,
                        col_of[s | {j}]: -1.0,
                        col_of[s | {i, j}]: 1.0,
                        col_of[s]: 1.0,
                    },
                    0.0,
                )
    for con in constraints:  # h(Y) - (1 - 1/p) h(I) <= log_norm
        coeffs = {col_of[con.target]: 1.0}
        if con.conditioning:
            coeffs[col_of[con.conditioning]] = -(1.0 - _inv_p(con.p))
        add_row(coeffs, con.log_norm)

    n_vars = len(subsets)
    c = np.zeros(n_vars)
    c[col_of[full]] = -1.0
    bounds: list[tuple[float, float | None]] = [(0.0, None)] * n_vars
    bounds[col_of[frozenset()]] = (0.0, 0.0)
    a_ub = coo_matrix((vals, (rows, cols)), shape=(len(rhs), n_vars)).tocsr()
    result = linprog(
        c, A_ub=a_ub, b_ub=np.asarray(rhs), bounds=bounds, method="highs-ipm"
    )
    if not result.success:
        msg = f"polymatroid LP failed: {result.message}"
        raise ValueError(msg)
    return -float(result.fun)


def cyclic_orders(attributes: Sequence[AttributeName]) -> list[tuple[str, ...]]:
    """All rotations of ``attributes`` and of its reverse.

    For a cycle query labelled in walk order these are the orders under which
    every block ``{v_i, v_{i+1}, v_{i+2}}`` keeps one usable ``|I| = 2``
    constraint (the one conditioning on the two earlier attributes).
    """
    seq = tuple(attributes)
    n = len(seq)
    out: list[tuple[str, ...]] = []
    for base in (seq, seq[::-1]):
        for k in range(n):
            out.append(base[k:] + base[:k])
    return out


def alternating_orders(attributes: Sequence[AttributeName]) -> list[tuple[str, ...]]:
    """Every rotation/reflection of ``attributes`` taken at stride two.

    For a cycle labelled in walk order this lists the *bipartite* orders:
    the vertices of one colour class first, then the other, so that every
    vertex of the second class has both of its neighbours earlier and its
    ``|I| = 2`` block constraint (the count of a vertex given both
    neighbours) survives the relaxation.  On even cycles these orders reach
    the polymatroid bound; on odd cycles one vertex necessarily loses it.
    """
    seq = tuple(attributes)
    n = len(seq)
    out: list[tuple[str, ...]] = []
    for base in (seq, seq[::-1]):
        for k in range(n):
            rot = base[k:] + base[:k]
            out.append(rot[0::2] + rot[1::2])
    return out


def cycle_candidate_orders(
    attributes: Sequence[AttributeName],
) -> list[tuple[str, ...]]:
    """Walk orders plus alternating orders of a cycle (``4|V|`` candidates)."""
    seen: set[tuple[str, ...]] = set()
    out: list[tuple[str, ...]] = []
    for order in [*cyclic_orders(attributes), *alternating_orders(attributes)]:
        if order not in seen:
            seen.add(order)
            out.append(order)
    return out


def flow_bound_over_orders(
    attributes: Iterable[AttributeName],
    constraints: Sequence[DegreeConstraint],
    orders: Iterable[Sequence[AttributeName]],
    *,
    prefix_sources: bool = False,
) -> tuple[FlowBoundResult, tuple[str, ...] | None]:
    """Minimise :func:`flow_bound` over candidate orders.

    Args:
        attributes: The query attributes.
        constraints: Degree constraints.
        orders: Candidate attribute orders for the non-simple relaxation.
        prefix_sources: Passed to :func:`flow_bound`.

    Returns:
        The best result and the order achieving it (``None`` if no order was
        supplied, in which case non-simple constraints are dropped).
    """
    best = flow_bound(attributes, constraints, order=None)
    best_order: tuple[str, ...] | None = None
    total = best.solve_seconds
    for order in orders:
        res = flow_bound(
            attributes, constraints, order=order, prefix_sources=prefix_sources
        )
        total += res.solve_seconds
        if res.log_bound < best.log_bound - 1e-9:
            best, best_order = res, tuple(order)
    return (
        FlowBoundResult(
            log_bound=best.log_bound,
            n_variables=best.n_variables,
            n_rows=best.n_rows,
            n_simple=best.n_simple,
            n_nonsimple_used=best.n_nonsimple_used,
            solve_seconds=total,
        ),
        best_order,
    )
