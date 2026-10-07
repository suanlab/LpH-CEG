"""Mechanically check the Step-5 proof obligation over the Shannon cone.

Theorem 1 (Composition Inequality) reduces, via Steps 1--4, to a single
linear implication.  Writing $E_j = A_j \\setminus I_j$ for the attributes
first added at hyperpath step $j$, Step 3 supplies, for every step $j$ and
every $p$,

    (1/p) h(I_j) + h(A_j) - h(I_j)  <=  log2 mu_p(B_j | I_j).

Step 5 then asserts that the attribute-wise cover constraint is enough to
conclude

    (OBLIGATION)   sum_{j,p} w_{j,p} [ (1/p) h(I_j) + h(A_j) - h(I_j) ]  >=  h(V).

An earlier draft declared this by appealing to "subadditivity", which does
not establish it: the cover constraint is *weaker* than a per-step
normalisation (see `step5_lemma.md`), and that slack is precisely what
produces the improvement on cyclic queries.  The obligation is now proved
(`docs/paper/extended_version.tex`, Lemma 3); this script is the independent
mechanical check of the same statement.

This script decides the obligation mechanically.  For a fixed weight
vector the obligation is linear in h, so we minimise its slack over the
polymatroid cone (all h satisfying the elemental Shannon inequalities),
normalised by h(V) = 1:

    minimise   sum_{j,p} w_{j,p} [ h(A_j) - (1 - 1/p) h(I_j) ]  -  h(V)

    subject to  h(V) - h(V \\ {i}) >= 0                        (monotone)
                h(A+i) + h(A+j) - h(A+ij) - h(A) >= 0          (submodular)
                h(empty) = 0,  h(V) = 1

Since the entropic region is contained in the polymatroid cone, a
minimum of >= 0 is a *certificate* that the obligation is Shannon-provable
for those weights -- strictly stronger than testing random databases,
which only probes entropic h.  A negative minimum exhibits a polymatroid
witness and proves no Shannon-type derivation exists (the bound may still
hold for entropic h, so this refutes the proof, not necessarily Theorem 1).

The check is also made *complete in the weights*.  The obligation is
affine in w, and W = {w >= 0 : cover} has recession cone R^N_{>=0} (the
coverage matrix is entrywise >= 0) whose extreme rays each satisfy the
left-hand side >= 0 termwise.  The certified set is a convex cone, so
checking every vertex of W plus those rays decides the obligation for
*all* feasible weights -- not merely sampled ones.

Scope: the verdict is exact per (shape, hyperpath), for the shapes in
FAMILIES and |V| small enough that 2^|V| stays tractable.  It is a finite
verification, not a general theorem; see docs/step5_lemma.md.

Output: results/validity/shannon_implication.csv
"""

from __future__ import annotations

import csv
import itertools
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from src.estimators.lp_bound import _collect_step_info, optimize_hyperpath_weights
from src.estimators.lph_ceg import estimate_lph_ceg
from src.statistics.lp_stats import compute_all_lp_block_extensions
from src.utils.types import QueryHypergraph, Relation

P_VALUES = (1.0, 2.0, float("inf"))
TOL = 1e-7
MAX_VERTICES = 20000


def _vec_to_w(v: np.ndarray) -> dict[tuple[int, float], float]:
    """Convert a flat weight vector back to the {(step, p): w} form."""
    n_p = len(P_VALUES)
    return {
        (j, p): float(v[j * n_p + pi])
        for j in range(len(v) // n_p)
        for pi, p in enumerate(P_VALUES)
    }


# ---------------------------------------------------------------------------
# Polymatroid (Shannon) cone
# ---------------------------------------------------------------------------
_ELEMENTAL_CACHE: dict[int, list[np.ndarray]] = {}


def elemental_inequalities(n: int) -> list[np.ndarray]:
    """Rows g with g . h >= 0 generating the polymatroid cone on n elements.

    Uses the standard elemental (minimal) generating set: monotonicity at
    the top, and elemental submodularity.  Cached per n: the wide check
    solves hundreds of thousands of these LPs.
    """
    if n in _ELEMENTAL_CACHE:
        return _ELEMENTAL_CACHE[n]
    full = (1 << n) - 1
    rows: list[np.ndarray] = []

    # Monotonicity: h(V) - h(V \ {i}) >= 0
    for i in range(n):
        g = np.zeros(1 << n)
        g[full] += 1.0
        g[full & ~(1 << i)] -= 1.0
        rows.append(g)

    # Elemental submodularity:
    #   h(A+i) + h(A+j) - h(A+i+j) - h(A) >= 0,  A subset of V \ {i,j}
    for i, j in itertools.combinations(range(n), 2):
        rest = [k for k in range(n) if k != i and k != j]
        for r in range(len(rest) + 1):
            for sub in itertools.combinations(rest, r):
                a = 0
                for k in sub:
                    a |= 1 << k
                g = np.zeros(1 << n)
                g[a | (1 << i)] += 1.0
                g[a | (1 << j)] += 1.0
                g[a | (1 << i) | (1 << j)] -= 1.0
                g[a] -= 1.0
                rows.append(g)
    _ELEMENTAL_CACHE[n] = rows
    return rows


@dataclass(frozen=True)
class Step:
    """One hyperpath step, as attribute bitmasks."""

    block_mask: int  # A_j  (all attributes of the block)
    iface_mask: int  # I_j  (interface with the current join state)


def check_obligation(
    steps: list[Step],
    weights: dict[tuple[int, float], float],
    n: int,
) -> tuple[float, bool]:
    """Minimise the obligation's slack over the polymatroid cone.

    Returns (min_slack, holds).  holds is True iff min_slack >= -TOL, i.e.
    the obligation is provable from Shannon inequalities alone.
    """
    return _min_slack(steps, weights, n, subtract_hv=True)


def _min_slack(
    steps: list[Step],
    weights: dict[tuple[int, float], float],
    n: int,
    *,
    subtract_hv: bool,
) -> tuple[float, bool]:
    n_vars = 1 << n
    full = n_vars - 1

    # Objective: sum_{j,p} w_{j,p} [ h(A_j) - (1 - 1/p) h(I_j) ] - h(V)
    c = np.zeros(n_vars)
    for j, st in enumerate(steps):
        for p in P_VALUES:
            w = weights.get((j, p), 0.0)
            if w == 0.0:
                continue
            inv_p = 0.0 if p == float("inf") else 1.0 / p
            c[st.block_mask] += w
            c[st.iface_mask] -= w * (1.0 - inv_p)
    if subtract_hv:
        c[full] -= 1.0

    # Polymatroid constraints as -g . h <= 0
    rows = elemental_inequalities(n)
    a_ub = np.vstack([-g for g in rows])
    b_ub = np.zeros(len(rows))

    # h(empty) = 0, h(V) = 1  (the cone is scale-invariant, so fixing the
    # scale loses nothing and makes the program bounded)
    a_eq = np.zeros((2, n_vars))
    a_eq[0, 0] = 1.0
    a_eq[1, full] = 1.0
    b_eq = np.array([0.0, 1.0])

    res = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(0.0, None)] * n_vars,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"polymatroid LP failed: {res.message}")
    return float(res.fun), bool(res.fun >= -TOL)


# ---------------------------------------------------------------------------
# Query construction (same generator as the other ablations)
# ---------------------------------------------------------------------------
def cyc(k: int) -> list[tuple[str, str]]:
    v = [chr(ord("a") + i) for i in range(k)]
    return [(v[i], v[(i + 1) % k]) for i in range(k)]


FAMILIES: dict[str, list[tuple[str, str]]] = {
    "triangle": cyc(3),
    "cycle4": cyc(4),
    "cycle5": cyc(5),
    "cycle6": cyc(6),
    "k4": list(itertools.combinations("abcd", 2)),
    "diamond": [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"), ("a", "c")],
    "path4": [("a", "b"), ("b", "c"), ("c", "d")],
    "star3": [("a", "b"), ("a", "c"), ("a", "d")],
    # |V| = 5-7: extends the certified region, narrowing where any
    # counterexample to Step 5 could still live.
    "chordal5": [*cyc(5), ("a", "c")],
    "k5": list(itertools.combinations("abcde", 2)),
    "chordal6": [*cyc(6), ("a", "d")],
    "cycle7": cyc(7),
}


def gen_edges(n_v: int, n_e: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    w = [1.0 / (i + 1) for i in range(n_v)]
    tot = sum(w)
    pr = [x / tot for x in w]

    def pick() -> str:
        r = rng.random()
        c = 0.0
        for i, p in enumerate(pr):
            c += p
            if r <= c:
                return str(i)
        return str(n_v - 1)

    s: set[tuple[str, str]] = set()
    while len(s) < n_e:
        a, b = pick(), pick()
        if a != b:
            s.add((a, b))
    return sorted(s)


def build(edges: list[tuple[str, str]], nv: int, ne: int, seed: int):
    rels, data = [], {}
    for i, (u, v) in enumerate(edges):
        rows = gen_edges(nv, ne, seed + i * 7)
        attrs = tuple(sorted((u, v)))
        swap = attrs != (u, v)
        rels.append(Relation(f"R{i}", frozenset({u, v}), len(rows)))
        data[f"R{i}"] = (attrs, [(y, x) for x, y in rows] if swap else rows)
    return QueryHypergraph(relations=tuple(rels), name="q"), data


def cover_matrix(
    steps: list[Step], attrs: list[str], idx: dict[str, int]
) -> np.ndarray:
    """Rows = attributes, cols = (step, p); entry = coverage of that attribute."""
    n_p = len(P_VALUES)
    c = np.zeros((len(attrs), len(steps) * n_p))
    for ai, a in enumerate(attrs):
        bit = 1 << idx[a]
        for j, st in enumerate(steps):
            for pi, p in enumerate(P_VALUES):
                if st.block_mask & bit and not (st.iface_mask & bit):
                    cov = 1.0
                elif st.iface_mask & bit:
                    cov = 0.0 if p == float("inf") else 1.0 / p
                else:
                    cov = 0.0
                c[ai, j * n_p + pi] = cov
    return c


def enumerate_vertices(cov: np.ndarray, max_vertices: int) -> list[np.ndarray]:
    """Enumerate vertices of W = {w >= 0 : cov @ w >= 1}.

    A vertex is determined by N linearly independent active constraints
    drawn from the cover rows (cov_i . w = 1) and the sign rows (w_k = 0).
    The recession cone of W is the nonnegative orthant (cov is entrywise
    >= 0), whose extreme rays are the unit vectors; those are handled
    separately by `rays_nonnegative`.
    """
    n_c, n = cov.shape
    rows = [(cov[i], 1.0) for i in range(n_c)]
    rows += [(np.eye(n)[k], 0.0) for k in range(n)]

    seen: set[tuple[int, ...]] = set()
    verts: list[np.ndarray] = []
    for combo in itertools.combinations(range(len(rows)), n):
        a = np.array([rows[i][0] for i in combo])
        b = np.array([rows[i][1] for i in combo])
        if abs(np.linalg.det(a)) < 1e-9:
            continue
        try:
            w = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:  # pragma: no cover - guarded by det
            continue
        if np.any(w < -1e-9) or np.any(cov @ w < 1.0 - 1e-7):
            continue
        key = tuple(np.round(w, 7))
        if key in seen:
            continue
        seen.add(key)
        verts.append(np.maximum(w, 0.0))
        if len(verts) >= max_vertices:
            break
    return verts


def rays_nonnegative(steps: list[Step], n: int) -> bool:
    """Every single term (1/p)h(I)+h(A|I) must be >= 0 over the cone.

    These are the extreme rays of W's recession cone, so the vertex check
    extends to all of W exactly when this holds.
    """
    for j in range(len(steps)):
        for p in P_VALUES:
            slack, _ = check_obligation_raw(steps, {(j, p): 1.0}, n)
            if slack < -TOL:
                return False
    return True


def check_obligation_raw(
    steps: list[Step], weights: dict[tuple[int, float], float], n: int
) -> tuple[float, bool]:
    """Like check_obligation but without the -h(V) term (ray check)."""
    return _min_slack(steps, weights, n, subtract_hv=False)


def random_feasible_weights(
    steps: list[Step], n: int, attrs: list[str], idx: dict[str, int], seed: int
) -> dict[tuple[int, float], float] | None:
    """Draw a random cover-feasible weight vector via an LP with a random objective."""
    rng = random.Random(seed)
    m = len(steps)
    n_p = len(P_VALUES)
    n_vars = m * n_p

    a_ub = np.zeros((len(attrs), n_vars))
    b_ub = np.full(len(attrs), -1.0)
    for ai, a in enumerate(attrs):
        bit = 1 << idx[a]
        for j, st in enumerate(steps):
            for pi, p in enumerate(P_VALUES):
                if st.block_mask & bit and not (st.iface_mask & bit):
                    cov = 1.0
                elif st.iface_mask & bit:
                    cov = 0.0 if p == float("inf") else 1.0 / p
                else:
                    cov = 0.0
                a_ub[ai, j * n_p + pi] = -cov

    c = np.array([rng.uniform(0.1, 1.0) for _ in range(n_vars)])
    res = linprog(
        c, A_ub=a_ub, b_ub=b_ub, bounds=[(0.0, None)] * n_vars, method="highs"
    )
    if not res.success:
        return None
    return {
        (j, p): float(res.x[j * n_p + pi])
        for j in range(m)
        for pi, p in enumerate(P_VALUES)
    }


def main() -> None:
    print("Step-5 obligation: mechanical check over the Shannon (polymatroid) cone")
    print("=" * 78)
    print(f"  {'family':<10} {'|V|':>4} {'weights':<14} {'min slack':>12}  verdict")
    print(f"  {'-' * 72}")

    rows_out: list[dict[str, object]] = []

    for fam, edges in FAMILIES.items():
        query, data = build(edges, 60, 240, seed=101)
        attrs = sorted(query.all_attributes)
        n = len(attrs)
        idx = {a: i for i, a in enumerate(attrs)}

        result = estimate_lph_ceg(query, data, max_block_size=1, p_values=P_VALUES)
        hp = result.hyperpath
        info = _collect_step_info(query, hp)
        steps = [
            Step(
                block_mask=sum(
                    1 << idx[a] for a in (s.interface_attrs | s.extension_attrs)
                ),
                iface_mask=sum(1 << idx[a] for a in s.interface_attrs),
            )
            for s in info
        ]

        # (a) the weights the estimator actually emits, over several databases
        cases: list[tuple[str, dict[tuple[int, float], float]]] = []
        for seed in (101, 202, 303):
            _, d = build(edges, 60, 240, seed=seed)
            stats = compute_all_lp_block_extensions(
                query, hp.blocks, d, p_values=P_VALUES
            )
            lp = optimize_hyperpath_weights(query, hp, stats, p_values=P_VALUES)
            cases.append((f"LP-opt(s{seed})", lp.weights))

        # (b) random cover-feasible weights
        for seed in (11, 22, 33, 44, 55):
            w = random_feasible_weights(steps, n, attrs, idx, seed)
            if w is not None:
                cases.append((f"random(s{seed})", w))

        for label, w in cases:
            slack, holds = check_obligation(steps, w, n)
            verdict = (
                "HOLDS (Shannon-provable)" if holds else "REFUTED (polymatroid witness)"
            )
            print(f"  {fam:<10} {n:>4} {label:<14} {slack:>12.6f}  {verdict}")
            rows_out.append(
                {
                    "family": fam,
                    "n_attrs": n,
                    "weights": label,
                    "min_slack": f"{slack:.6f}",
                    "shannon_provable": holds,
                }
            )

        # (c) COMPLETE check: every vertex of the cover polytope.  Together
        # with the ray check this decides the obligation for *all* feasible
        # weights, not just sampled ones.
        cov = cover_matrix(steps, attrs, idx)
        verts = enumerate_vertices(cov, MAX_VERTICES)
        exhausted = len(verts) < MAX_VERTICES
        rays_ok = rays_nonnegative(steps, n)
        worst = min(
            (check_obligation(steps, _vec_to_w(v), n)[0] for v in verts), default=0.0
        )
        all_ok = worst >= -TOL and rays_ok
        scope = (
            "ALL feasible weights" if exhausted else f"first {MAX_VERTICES} vertices"
        )
        print(
            f"  {fam:<10} {n:>4} {'VERTICES(' + str(len(verts)) + ')':<14} "
            f"{worst:>12.6f}  {'HOLDS' if all_ok else 'REFUTED'} over {scope}"
        )
        rows_out.append(
            {
                "family": fam,
                "n_attrs": n,
                "weights": f"all-vertices({len(verts)})",
                "min_slack": f"{worst:.6f}",
                "shannon_provable": all_ok,
            }
        )

    n_hold = sum(1 for r in rows_out if r["shannon_provable"])
    print(f"  {'-' * 72}")
    print(f"  {n_hold}/{len(rows_out)} weight vectors are Shannon-provable")
    if n_hold < len(rows_out):
        bad = {r["family"] for r in rows_out if not r["shannon_provable"]}
        print(f"  REFUTED on: {sorted(bad)}")
        print(
            "  => Step 5 does NOT follow from Shannon inequalities for these weights."
        )
    else:
        print("  => Step 5's obligation is certified over the full polymatroid cone")
        print("     for every weight vector tested (stronger than database testing).")

    out = Path(__file__).resolve().parent.parent / "results" / "validity"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "shannon_implication.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows_out)
    print(f"\nWrote {out / 'shannon_implication.csv'} ({len(rows_out)} checks)")


if __name__ == "__main__":
    main()
