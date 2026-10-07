"""Block extension statistics without materializing the block join.

For a two-relation block ``{R, S}`` with shared attributes ``B``, the degree
sequence of an interface ``I`` is a function of *single-relation* degree
tables whenever ``I`` lies inside one relation or contains ``B``:

* ``I ⊆ attrs(R)``, ``B ⊄ I`` ("one-side"): each interface value ``i`` extends
  to ``Σ_{t ∈ R, t[I]=i} deg_S(t[B])`` distinct tuples, where ``deg_S(b)`` is
  the number of distinct ``attrs(S) \\ B`` values under ``b``.  One pass over
  ``S`` and one over ``R``.
* ``B ⊆ I`` ("shared"): interface values are pairs ``(i_R, i_S)`` agreeing on
  ``B``, with degree ``d_R(i_R) · d_S(i_S)``; the ``lp`` norms factor as
  ``Σ_b (Σ_{i_R ∋ b} d_R^p)(Σ_{i_S ∋ b} d_S^p)`` and are computed in closed
  form, again in one pass per relation.

The remaining interfaces (``I`` straddles both relations without covering
``B``; on a binary block ``R(a,b), S(b,c)`` this is exactly ``{a, c}``) count
common neighbours and need the join.  :func:`join_free_interfaces` classifies
every interface of a block; :func:`compute_pushdown_block_extensions` computes
the join-free ones in ``O(|R| + |S|)``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

from src.statistics.block_stats import _enumerate_interfaces
from src.statistics.lp_stats import (
    DEFAULT_P_VALUES,
    LpBlockExtensionStats,
    _compute_lp_norm,
    compute_all_lp_block_extensions,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from src.statistics.block_stats import RelationDataMap
    from src.utils.types import AttributeName, Block, Interface, QueryHypergraph

JoinFreeKind = str  # "one_side" | "shared"


def _proj(row: tuple[str, ...], idx: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(row[i] for i in idx)


def _indices(
    schema: tuple[AttributeName, ...], attrs: Iterable[str]
) -> tuple[int, ...]:
    return tuple(schema.index(a) for a in sorted(attrs))


def join_free_interfaces(
    query: QueryHypergraph, block: Block
) -> dict[Interface, JoinFreeKind | None]:
    """Classify every proper interface of a two-relation block.

    Returns:
        Interface -> ``"one_side"``, ``"shared"`` or ``None`` (needs the join).

    Raises:
        ValueError: If the block does not have exactly two relations.
    """
    rels = sorted(block.relations)
    if len(rels) != 2:
        msg = f"pushdown statistics are defined for two-relation blocks, got {rels}"
        raise ValueError(msg)
    a_r = frozenset(query.get_relation(rels[0]).attributes)
    a_s = frozenset(query.get_relation(rels[1]).attributes)
    shared = a_r & a_s
    out: dict[Interface, JoinFreeKind | None] = {}
    for interface in _enumerate_interfaces(a_r | a_s):
        if shared <= interface:
            out[interface] = "shared"
        elif interface <= a_r or interface <= a_s:
            out[interface] = "one_side"
        else:
            out[interface] = None
    return out


def _distinct_rows(rows: Iterable[tuple[str, ...]]) -> list[tuple[str, ...]]:
    return list(dict.fromkeys(rows))


def _degree_table(
    rows: list[tuple[str, ...]], key_idx: tuple[int, ...], ext_idx: tuple[int, ...]
) -> dict[tuple[str, ...], int]:
    """``key value -> number of distinct extension values`` in one pass."""
    seen: dict[tuple[str, ...], set[tuple[str, ...]]] = defaultdict(set)
    for row in rows:
        seen[_proj(row, key_idx)].add(_proj(row, ext_idx))
    return {k: len(v) for k, v in seen.items()}


def _make_stats(
    degree_sequence: tuple[int, ...],
    norms: dict[float, float],
    interface_size: int,
    count: int,
    max_ext: int,
) -> LpBlockExtensionStats:
    total = norms.get(1.0)
    avg = (total / count) if (total is not None and count) else 0.0
    return LpBlockExtensionStats(
        degree_sequence=degree_sequence,
        lp_norms=norms,
        interface_size=interface_size,
        count=count,
        max_extension=max_ext,
        avg_extension=avg,
        log_max_extension=math.log2(max_ext) if max_ext >= 2 else 0.0,
    )


def _one_side(
    rows_r: list[tuple[str, ...]],
    schema_r: tuple[AttributeName, ...],
    rows_s: list[tuple[str, ...]],
    schema_s: tuple[AttributeName, ...],
    shared: frozenset[str],
    interface: Interface,
    p_values: tuple[float, ...],
) -> LpBlockExtensionStats:
    ext_s = frozenset(schema_s) - shared
    deg_s = _degree_table(rows_s, _indices(schema_s, shared), _indices(schema_s, ext_s))
    i_idx = _indices(schema_r, interface)
    b_idx = _indices(schema_r, shared)
    acc: dict[tuple[str, ...], int] = defaultdict(int)
    for row in rows_r:
        d = deg_s.get(_proj(row, b_idx), 0)
        if d:
            acc[_proj(row, i_idx)] += d
    seq = tuple(sorted(acc.values(), reverse=True))
    if not seq:
        return _make_stats((), {p: 0.0 for p in p_values}, len(interface), 0, 0)
    norms = {p: _compute_lp_norm(seq, p) for p in p_values}
    return _make_stats(seq, norms, len(interface), len(seq), seq[0])


def _shared(
    rows_r: list[tuple[str, ...]],
    schema_r: tuple[AttributeName, ...],
    rows_s: list[tuple[str, ...]],
    schema_s: tuple[AttributeName, ...],
    shared: frozenset[str],
    interface: Interface,
    p_values: tuple[float, ...],
) -> LpBlockExtensionStats:
    finite = [p for p in p_values if not math.isinf(p)]

    def side(
        rows: list[tuple[str, ...]], schema: tuple[AttributeName, ...]
    ) -> dict[tuple[str, ...], tuple[dict[float, float], int, int]]:
        """b -> (Σ_i d(i)^p per finite p, max_i d(i), number of i under b)."""
        i_attrs = interface & frozenset(schema)
        ext = frozenset(schema) - interface
        deg = _degree_table(rows, _indices(schema, i_attrs), _indices(schema, ext))
        b_pos = tuple(sorted(i_attrs).index(a) for a in sorted(shared))
        out: dict[tuple[str, ...], tuple[dict[float, float], int, int]] = {}
        for i_val, d in deg.items():
            b = _proj(i_val, b_pos)
            sums, mx, n = out.get(b, ({p: 0.0 for p in finite}, 0, 0))
            for p in finite:
                sums[p] += float(d) ** p
            out[b] = (sums, max(mx, d), n + 1)
        return out

    side_r = side(rows_r, schema_r)
    side_s = side(rows_s, schema_s)
    pow_sums = {p: 0.0 for p in finite}
    max_ext = 0
    count = 0
    for b, (sr, mr, nr) in side_r.items():
        hit = side_s.get(b)
        if hit is None:
            continue
        ss, ms, ns = hit
        for p in finite:
            pow_sums[p] += sr[p] * ss[p]
        max_ext = max(max_ext, mr * ms)
        count += nr * ns
    norms = {p: pow_sums[p] ** (1.0 / p) for p in finite}
    if any(math.isinf(p) for p in p_values):
        norms[math.inf] = float(max_ext)
    # The sequence itself is not materialized: its length is the size of the
    # interface projection of the join (the join itself when I = all attrs).
    return _make_stats((), norms, len(interface), count, max_ext)


def compute_pushdown_block_extensions(
    query: QueryHypergraph,
    block: Block,
    relation_data: RelationDataMap,
    *,
    p_values: tuple[float, ...] = DEFAULT_P_VALUES,
) -> dict[tuple[Block, Interface], LpBlockExtensionStats]:
    """Join-free ``lp`` statistics for every join-free interface of ``block``.

    Single-relation blocks delegate to :func:`compute_all_lp_block_extensions`
    (no join is involved there).  Two-relation blocks return only the
    interfaces classified join-free by :func:`join_free_interfaces`.
    """
    if len(block.relations) == 1:
        return compute_all_lp_block_extensions(
            query, [block], relation_data, p_values=p_values
        )
    kinds = join_free_interfaces(query, block)
    rels = sorted(block.relations)
    schema_r, raw_r = relation_data[rels[0]]
    schema_s, raw_s = relation_data[rels[1]]
    rows_r = _distinct_rows(raw_r)
    rows_s = _distinct_rows(raw_s)
    shared = frozenset(schema_r) & frozenset(schema_s)
    out: dict[tuple[Block, Interface], LpBlockExtensionStats] = {}
    for interface, kind in kinds.items():
        if kind is None:
            continue
        if kind == "shared":
            st = _shared(
                rows_r, schema_r, rows_s, schema_s, shared, interface, p_values
            )
        elif interface <= frozenset(schema_r):
            st = _one_side(
                rows_r, schema_r, rows_s, schema_s, shared, interface, p_values
            )
        else:
            st = _one_side(
                rows_s, schema_s, rows_r, schema_r, shared, interface, p_values
            )
        out[(block, interface)] = st
    return out


# ---------------------------------------------------------------------------
# The join-required statistic on a binary block, without the join.
#
# On R(a, b), S(b, c) the interface {a, c} extends to the common neighbours
# codeg(a, c) = |N_R(a) ∩ N_S(c)| = (M N)_{ac}, with M and N the 0/1 incidence
# matrices of R and S.  ||codeg||_2 = ||M N||_F.  Two linear-time surrogates:
#
# * a deterministic upper bound from the triangle inequality,
#   ||codeg(a, ·)||_2 <= Σ_{b ∈ N_R(a)} sqrt(deg_S(b)), summed over a (and the
#   symmetric orientation; the smaller is kept) -- valid, one pass each;
# * a Hutchinson sketch, ||M N||_F^2 = E ||M N g||^2 for Rademacher g, one
#   pass over S and one over R per probe -- an estimate, not a bound.


def _binary_block(
    query: QueryHypergraph, block: Block, relation_data: RelationDataMap
) -> tuple[
    list[tuple[str, ...]], tuple[int, int], list[tuple[str, ...]], tuple[int, int]
]:
    """Rows of R and S with (own, shared) column indices; binary blocks only."""
    rels = sorted(block.relations)
    if len(rels) != 2:
        msg = f"expected a two-relation block, got {rels}"
        raise ValueError(msg)
    schema_r, raw_r = relation_data[rels[0]]
    schema_s, raw_s = relation_data[rels[1]]
    shared = frozenset(schema_r) & frozenset(schema_s)
    if len(schema_r) != 2 or len(schema_s) != 2 or len(shared) != 1:
        msg = (
            "codegree surrogates are defined for binary relations sharing one attribute"
        )
        raise ValueError(msg)
    (b,) = shared
    ib_r, ib_s = schema_r.index(b), schema_s.index(b)
    return (
        _distinct_rows(raw_r),
        (1 - ib_r, ib_r),
        _distinct_rows(raw_s),
        (1 - ib_s, ib_s),
    )


def codegree_interface(query: QueryHypergraph, block: Block) -> Interface:
    """The join-required interface ``{a, c}`` of a binary block."""
    rels = sorted(block.relations)
    a_r = frozenset(query.get_relation(rels[0]).attributes)
    a_s = frozenset(query.get_relation(rels[1]).attributes)
    return (a_r | a_s) - (a_r & a_s)


def codegree_l2_upper_bound(
    query: QueryHypergraph, block: Block, relation_data: RelationDataMap
) -> float:
    """Deterministic one-pass upper bound on ``||codeg||_2`` (triangle inequality)."""
    rows_r, (ia, ib_r), rows_s, (ic, ib_s) = _binary_block(query, block, relation_data)

    def one_way(
        rows_x: list[tuple[str, ...]],
        own_x: int,
        sh_x: int,
        rows_y: list[tuple[str, ...]],
        own_y: int,
        sh_y: int,
    ) -> float:
        deg_y: dict[str, int] = defaultdict(int)
        for row in rows_y:
            deg_y[row[sh_y]] += 1
        acc: dict[str, float] = defaultdict(float)
        for row in rows_x:
            acc[row[own_x]] += math.sqrt(deg_y.get(row[sh_x], 0))
        return math.sqrt(sum(v * v for v in acc.values()))

    return min(
        one_way(rows_r, ia, ib_r, rows_s, ic, ib_s),
        one_way(rows_s, ic, ib_s, rows_r, ia, ib_r),
    )


def sketch_codegree_l2(
    query: QueryHypergraph,
    block: Block,
    relation_data: RelationDataMap,
    *,
    n_probes: int,
    seed: int = 0,
) -> tuple[float, float]:
    """Hutchinson estimate of ``||codeg||_2`` from ``n_probes`` Rademacher probes.

    Returns:
        ``(estimate, relative_std_error)`` of the squared norm's mean; the
        estimate is ``sqrt(mean_k ||M N g_k||^2)``.
    """
    import random

    rows_r, (ia, ib_r), rows_s, (ic, ib_s) = _binary_block(query, block, relation_data)
    rng = random.Random(seed)
    c_vals = sorted({row[ic] for row in rows_s})
    samples: list[float] = []
    for _ in range(n_probes):
        g = {c: (1.0 if rng.random() < 0.5 else -1.0) for c in c_vals}
        v: dict[str, float] = defaultdict(float)
        for row in rows_s:
            v[row[ib_s]] += g[row[ic]]
        u: dict[str, float] = defaultdict(float)
        for row in rows_r:
            u[row[ia]] += v.get(row[ib_r], 0.0)
        samples.append(sum(x * x for x in u.values()))
    mean = sum(samples) / n_probes
    if n_probes > 1 and mean > 0:
        var = sum((x - mean) ** 2 for x in samples) / (n_probes - 1)
        rel_se = math.sqrt(var / n_probes) / mean
    else:
        rel_se = float("nan")
    return math.sqrt(mean), rel_se
