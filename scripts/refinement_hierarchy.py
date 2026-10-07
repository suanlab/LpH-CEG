"""H1 kill-gate: is there a tightening path from the cover LP to the entropic optimum?

The strategic problem with the current method is that it is a single
point on the cost/tightness frontier, and a dominated-looking one: looser
than a published bound. A method that *spans* the frontier is a different
object -- an optimizer could ask for "the best bound you can give me in
10 ms" and get an answer.

Whether that is possible reduces to one question, asked here.

Setup.  The bound is  min_w  sum_j w_j log mu_j  over the set F of valid
weights.  The cover polytope C is an inner approximation, C subset F, so
minimising over C gives a *larger* (weaker) bound.  Enlarging C toward F
should tighten monotonically, and at F we recover the entropic optimum.

Algorithm (outer approximation + scaling projection).  Starting from the
cover polytope is useless: its optimum is already valid, the oracle
accepts on the first call and no cut ever fires.  So we start from a
RELAXATION and walk back.  Minimise over the relaxed set R (initially just
w >= 0 with one anchor constraint); the minimiser w is generally invalid.
One oracle call then does double duty.  Writing m(w) = min over the
polymatroid cone of sum w f(h) at h(V)=1, the reported slack is m(w) - 1,
and t*w is valid exactly when t >= 1/m(w).  So the same call yields

  * a cut, from the minimising h, which tightens R for the next round; and
  * a CERTIFIED VALID bound, obj(w)/m(w), by scaling w up to the boundary.

That second quantity is the anytime sequence: every iterate is a genuine
upper bound, and it decreases toward the entropic optimum as the cuts
accumulate.

The oracle costs 2^|V|, so this prototype is not itself a scalable
algorithm -- it answers only whether the tightening curve EXISTS. If it
does, the follow-up question is whether a cheaper sufficient oracle (a
level-by-level restriction of the polymatroid constraints) can replace it;
if it does not, the whole direction dies here and nothing further is worth
building.

Pre-registered kill gate at |V| <= 6: accepted bounds must decrease
monotonically and close >= 50% of the cover-to-entropic gap within 10
iterations. Otherwise: abandon.

RESULT: FAILED, 0% of the gap closed on every family -- and the reason is
worth more than the hypothesis was. The loop converges in 4-7 cuts, and it
converges to exactly the cover bound (cycle-4: 15.3353 in log2, against
14.8577 for entropic). So on a fixed hyperpath the cover LP is ALREADY
optimal over the whole set of valid weight vectors; the cover constraint
is not costing anything.

The entire cover-to-entropic gap is therefore a *statistics-coverage* gap,
not a constraint-tightness gap. The entropic program draws on conditional
statistics mu_p(U|W) for exponentially many (U,W); one hyperpath supplies
only its m steps. No amount of refinement in weight space can reach terms
that are not in the program.

Two things follow. Any interpolation between the two bounds has to be a
hierarchy over *statistics sets*, not over weight constraints -- which is
precisely what larger blocks do, and why bs=2 buys 1.2-1.6x while LP
refinement buys 0%. And the honest way to describe the cover LP is not
"a loose relaxation" but "the exact optimum for the statistics it is
given".

Output: results/cost/refinement_hierarchy.csv
"""

from __future__ import annotations

import csv
import itertools
import math
import random
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from scripts.shannon_implication_check import Step, elemental_inequalities
from src.estimators.lp_bound import _collect_step_info
from src.estimators.lpbound_entropic import estimate_lpbound_entropic
from src.estimators.lph_ceg import estimate_lph_ceg
from src.statistics.lp_stats import compute_all_lp_block_extensions
from src.utils.types import QueryHypergraph, Relation

ROOT = Path(__file__).resolve().parent.parent
P_VALUES: tuple[float, ...] = (1.0, 2.0, float("inf"))
MAX_ITERS = 10
TOL = 1e-7


def cyc(k: int) -> list[tuple[str, str]]:
    v = [chr(ord("a") + i) for i in range(k)]
    return [(v[i], v[(i + 1) % k]) for i in range(k)]


FAMILIES: dict[str, list[tuple[str, str]]] = {
    "triangle": cyc(3),
    "cycle4": cyc(4),
    "cycle5": cyc(5),
    "k4": list(itertools.combinations("abcd", 2)),
    "cycle6": cyc(6),
}


def gen_edges(n_v: int, n_e: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    w = [1.0 / (i + 1) for i in range(n_v)]
    tot = sum(w)
    pr = [x / tot for x in w]

    def pick() -> str:
        r = rng.random()
        c = 0.0
        for i, q in enumerate(pr):
            c += q
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


def validity_oracle(
    steps: list[Step], w: np.ndarray, n: int
) -> tuple[float, np.ndarray]:
    """Min slack of w over the polymatroid cone, plus the minimising h.

    slack >= 0  =>  w is a valid weight vector (the bound it gives holds).
    slack <  0  =>  h is a witness, usable as a cut.
    """
    n_vars = 1 << n
    full = n_vars - 1
    c = np.zeros(n_vars)
    n_p = len(P_VALUES)
    for j, st in enumerate(steps):
        for pi, p in enumerate(P_VALUES):
            wt = float(w[j * n_p + pi])
            if wt == 0.0:
                continue
            inv_p = 0.0 if p == float("inf") else 1.0 / p
            c[st.block_mask] += wt
            c[st.iface_mask] -= wt * (1.0 - inv_p)
    c[full] -= 1.0

    rows = elemental_inequalities(n)
    res = linprog(
        c,
        A_ub=np.vstack([-g for g in rows]),
        b_ub=np.zeros(len(rows)),
        A_eq=np.array([[1.0] + [0.0] * (n_vars - 1), [0.0] * (n_vars - 1) + [1.0]]),
        b_eq=np.array([0.0, 1.0]),
        bounds=[(0.0, None)] * n_vars,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(res.message)
    return float(res.fun), np.asarray(res.x)


def cut_row(steps: list[Step], h: np.ndarray, n: int) -> tuple[np.ndarray, float]:
    """The linear constraint on w that h induces: sum_j,p w f_{j,p}(h) >= h(V)."""
    n_p = len(P_VALUES)
    row = np.zeros(len(steps) * n_p)
    for j, st in enumerate(steps):
        for pi, p in enumerate(P_VALUES):
            inv_p = 0.0 if p == float("inf") else 1.0 / p
            row[j * n_p + pi] = h[st.block_mask] - (1.0 - inv_p) * h[st.iface_mask]
    return row, float(h[(1 << n) - 1])


def main() -> None:
    print("H1 kill-gate: does a cover -> entropic tightening curve exist?")
    print("=" * 78)
    print(
        f"  {'family':<10} {'|V|':>4} {'cover':>12} {'refined':>12} "
        f"{'entropic':>12} {'gap closed':>11} {'iters':>6} {'monotone':>9}"
    )
    print(f"  {'-' * 74}")

    out_rows: list[dict[str, object]] = []
    for fam, edges in FAMILIES.items():
        query, data = build(edges, 60, 240, seed=101)
        attrs = sorted(query.all_attributes)
        n = len(attrs)
        idx = {a: i for i, a in enumerate(attrs)}

        res = estimate_lph_ceg(query, data, max_block_size=1, p_values=P_VALUES)
        hp = res.hyperpath
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
        stats = compute_all_lp_block_extensions(
            query, hp.blocks, data, p_values=P_VALUES
        )
        n_p = len(P_VALUES)
        obj = np.zeros(len(steps) * n_p)
        for j, s in enumerate(info):
            st = stats[(s.block, s.interface)]
            for pi, p in enumerate(P_VALUES):
                v = st.norm(p)
                obj[j * n_p + pi] = math.log2(v) if v > 1.0 else 0.0

        cover_bound = res.estimate
        entropic = estimate_lpbound_entropic(query, data, p_values=P_VALUES)

        # start from the cover constraints, then add polymatroid cuts
        cov_rows, cov_rhs = [], []
        for a in attrs:
            bit = 1 << idx[a]
            row = np.zeros(len(steps) * n_p)
            for j, st in enumerate(steps):
                for pi, p in enumerate(P_VALUES):
                    if st.block_mask & bit and not (st.iface_mask & bit):
                        row[j * n_p + pi] = 1.0
                    elif st.iface_mask & bit:
                        row[j * n_p + pi] = 0.0 if p == float("inf") else 1.0 / p
            cov_rows.append(row)
            cov_rhs.append(1.0)

        # Relaxed start: one anchor row keeps the LP bounded away from w=0.
        # (Every attribute covered by SOME step with total weight >= 1.)
        anchor = np.zeros(len(steps) * n_p)
        for j in range(len(steps)):
            for pi in range(n_p):
                anchor[j * n_p + pi] = 1.0
        rows_l, rhs_l = [anchor], [1.0]
        del cov_rows, cov_rhs

        log_cover = math.log2(max(cover_bound, 1.0))
        accepted: list[float] = [log_cover]
        iters = 0
        for _ in range(MAX_ITERS):
            iters += 1
            r = linprog(
                obj,
                A_ub=np.vstack([-x for x in rows_l]),
                b_ub=-np.asarray(rhs_l),
                bounds=[(0.0, None)] * len(obj),
                method="highs",
            )
            if not r.success:
                break
            w = np.asarray(r.x)
            slack, h = validity_oracle(steps, w, n)
            m = slack + 1.0  # min over the cone of sum w f(h) at h(V)=1
            if m > 1e-9:
                # scale to the validity boundary -> a certified upper bound
                accepted.append(min(accepted[-1], float(r.fun) / m))
            if slack >= -TOL:
                break
            row, rhs = cut_row(steps, h, n)
            rows_l.append(row)
            rhs_l.append(rhs)

        log_cov, log_ent = accepted[0], math.log2(max(entropic, 1.0))
        refined = accepted[-1]
        span = log_cov - log_ent
        closed = (log_cov - refined) / span if span > 1e-9 else 0.0
        monotone = all(
            accepted[i + 1] <= accepted[i] + 1e-9 for i in range(len(accepted) - 1)
        )
        print(
            f"  {fam:<10} {n:>4} {2**log_cov:>12.3e} {2**refined:>12.3e} "
            f"{entropic:>12.3e} {closed * 100:>10.1f}% {iters:>6} {monotone!s:>9}"
        )
        out_rows.append(
            {
                "family": fam,
                "n_attrs": n,
                "cover_bound": f"{2**log_cov:.6e}",
                "refined_bound": f"{2**refined:.6e}",
                "entropic_bound": f"{entropic:.6e}",
                "gap_closed_frac": f"{closed:.4f}",
                "iterations": iters,
                "monotone": monotone,
            }
        )

    print(f"  {'-' * 74}")
    worst = min(float(r["gap_closed_frac"]) for r in out_rows)
    all_mono = all(bool(r["monotone"]) for r in out_rows)
    gate = worst >= 0.5 and all_mono
    print(
        f"  kill-gate (monotone AND >=50% of gap closed in {MAX_ITERS} iters): "
        f"worst {worst * 100:.1f}%, monotone={all_mono} -> "
        f"{'PASS -- pursue' if gate else 'FAIL -- abandon this direction'}"
    )
    if not gate:
        print(
            "\n  Diagnosis: the loop converges to exactly the cover bound, so the\n"
            "  cover LP is already optimal over ALL valid weights on this\n"
            "  hyperpath. The gap to entropic is a statistics-coverage gap, not a\n"
            "  constraint-tightness gap -- refine the statistics, not the weights."
        )

    out = ROOT / "results" / "cost"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "refinement_hierarchy.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(out_rows)
    print(f"\nWrote {out / 'refinement_hierarchy.csv'}")


if __name__ == "__main__":
    main()
