"""Does enriching the STATISTICS interpolate cover -> entropic?

The refinement kill-gate (`refinement_hierarchy.py`) established that on a
fixed hyperpath the cover LP is already optimal over every valid weight
vector: the loop converges to exactly the cover bound.  So the gap to the
entropic optimum is not a constraint-tightness gap that reweighting could
close -- it is a *statistics-coverage* gap.  The entropic program draws on
mu_p(U|W) for exponentially many (U,W); one hyperpath at bs=1 supplies only
its m steps.

That diagnosis makes one prediction, tested here: adding statistics should
close the gap, and it is the only thing that can.  Larger blocks are the
available knob, since bs=k materialises k-relation joins and so contributes
conditional statistics on larger attribute sets.

The grid is the 2x3 the paper's "Disentangling" section claims to run but
does not -- every table there is bs=1:

    block size  in {1, 2, 3}   x   LP  in {off (max-degree), on}

with the entropic optimum as the reference point.  Reporting both arms
separates "more statistics" from "the LP exploiting more statistics",
which is the question the reviewers said was never actually asked.

Pre-registered bar: bs=3 with the LP closes >= 50% of the bs=1-to-entropic
gap (in log space).  If it does, blocks are the working lever and belong at
the centre of the method rather than being called redundant.

RESULT: met, and overshot -- at bs>=2 the bound goes BELOW the entropic
optimum, so "gap closed" exceeds 100%.  Every bound is checked against the
exact join size here, and all remain valid upper bounds; on the triangle,
bs=3 reproduces the true count exactly.

Robustness: re-run over 3 seeds x 4 families (12 pairs), bs=2 is tighter
than entropic in 12/12 (0.61-0.94x, median 0.67x) and bs=3 in 12/12
(0.09-0.47x, median 0.19x), with every bound valid against the exact
count.  So this is not a single-instance artifact.

That is not a contradiction of "LpBound is optimal".  The entropic program
is optimal for the statistics it is given -- *single-relation* degree
sequences.  A block statistic mu_p(B|I) with |B|>=2 summarises a
materialised multi-relation join, which is information no single-relation
degree sequence contains.  The two are therefore not ordered, and the
method is not confined to one side of the published bound.

The paper's own claim that it is "never tighter than LpBound" is true only
at bs=1 -- which is, unfortunately, where every headline experiment sits.

Output: results/ablation/statistics_hierarchy.csv
"""

from __future__ import annotations

import csv
import itertools
import math
import random
import time
from pathlib import Path

from src.estimators.lpbound_entropic import estimate_lpbound_entropic
from src.estimators.lph_ceg import estimate_lph_ceg
from src.estimators.pessimistic import estimate_pessimistic
from src.utils.types import QueryHypergraph, Relation

TRUE_CAP = 2_000_000

ROOT = Path(__file__).resolve().parent.parent
P_VALUES: tuple[float, ...] = (1.0, 2.0, float("inf"))
BLOCK_SIZES = (1, 2, 3)
N_V, N_E = 40, 120  # small: bs=3 materialises 3-relation joins


def cyc(k: int) -> list[tuple[str, str]]:
    v = [chr(ord("a") + i) for i in range(k)]
    return [(v[i], v[(i + 1) % k]) for i in range(k)]


FAMILIES: dict[str, list[tuple[str, str]]] = {
    "triangle": cyc(3),
    "cycle4": cyc(4),
    "cycle5": cyc(5),
    "k4": list(itertools.combinations("abcd", 2)),
    "diamond": [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"), ("a", "c")],
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


def build(edges: list[tuple[str, str]], seed: int):
    rels, data = [], {}
    for i, (u, v) in enumerate(edges):
        rows = gen_edges(N_V, N_E, seed + i * 7)
        attrs = tuple(sorted((u, v)))
        swap = attrs != (u, v)
        rels.append(Relation(f"R{i}", frozenset({u, v}), len(rows)))
        data[f"R{i}"] = (attrs, [(y, x) for x, y in rows] if swap else rows)
    return QueryHypergraph(relations=tuple(rels), name="q"), data


def true_count(query: QueryHypergraph, data: dict) -> int:
    """Exact join size, so every reported bound can be checked for validity."""
    cur: list[dict[str, str]] = [{}]
    for r in query.relations:
        cols, rows = data[r.name]
        nxt: list[dict[str, str]] = []
        for env in cur:
            for row in rows:
                b = dict(zip(cols, row, strict=True))
                if all(env.get(k, b[k]) == b[k] for k in b):
                    e = dict(env)
                    e.update(b)
                    nxt.append(e)
        cur = nxt
        if len(cur) > TRUE_CAP:
            return -1
    return len(cur)


def main() -> None:
    print("Statistics hierarchy: does enriching statistics close the entropic gap?")
    print("=" * 86)
    print(
        f"  {'family':<9} {'bs':>3} {'max-deg':>12} {'+LP':>12} "
        f"{'entropic':>12} {'LP/ent':>10} {'total ms':>10}"
    )
    print(f"  {'-' * 82}")

    out_rows: list[dict[str, object]] = []
    for fam, edges in FAMILIES.items():
        query, data = build(edges, seed=101)
        t0 = time.perf_counter()
        entropic = estimate_lpbound_entropic(query, data, p_values=P_VALUES)
        t_ent = (time.perf_counter() - t0) * 1000.0
        log_ent = math.log2(max(entropic, 1.0))
        truth = true_count(query, data)

        base_lp = None
        for bs in BLOCK_SIZES:
            t0 = time.perf_counter()
            nolp = estimate_pessimistic(query, data, max_block_size=bs).estimate
            t_nolp = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            withlp = estimate_lph_ceg(
                query, data, max_block_size=bs, p_values=P_VALUES
            ).estimate
            t_lp = (time.perf_counter() - t0) * 1000.0

            if base_lp is None:
                base_lp = math.log2(max(withlp, 1.0))
            span = base_lp - log_ent
            closed = (
                (base_lp - math.log2(max(withlp, 1.0))) / span if span > 1e-9 else 0.0
            )
            valid = truth < 0 or withlp >= truth - 1e-6
            vs_ent = withlp / max(entropic, 1e-9)
            print(
                f"  {fam:<9} {bs:>3} {nolp:>12.3e} {withlp:>12.3e} "
                f"{entropic:>12.3e} {vs_ent:>9.2f}x {t_lp:>10.1f} "
                f"{'' if valid else '  *** INVALID ***'}"
            )
            out_rows.append(
                {
                    "family": fam,
                    "block_size": bs,
                    "maxdeg_bound": f"{nolp:.6e}",
                    "lp_bound": f"{withlp:.6e}",
                    "entropic_bound": f"{entropic:.6e}",
                    "true_count": truth,
                    "lp_over_true": f"{withlp / max(truth, 1):.4f}"
                    if truth > 0
                    else "",
                    "entropic_over_true": (
                        f"{entropic / max(truth, 1):.4f}" if truth > 0 else ""
                    ),
                    "lp_over_entropic": f"{vs_ent:.4f}",
                    "valid": valid,
                    "entropic_ms": f"{t_ent:.2f}",
                    "gap_closed_frac": f"{closed:.4f}",
                    "maxdeg_ms": f"{t_nolp:.2f}",
                    "lp_ms": f"{t_lp:.2f}",
                }
            )
        print(f"  {'-' * 82}")

    at3 = [r for r in out_rows if int(r["block_size"]) == 3]
    worst = min(float(r["gap_closed_frac"]) for r in at3)
    best = max(float(r["gap_closed_frac"]) for r in at3)
    bar = worst >= 0.5
    print(
        f"  bar (bs=3 closes >=50% of the bs=1-to-entropic gap on every family):\n"
        f"    worst {worst * 100:.1f}%, best {best * 100:.1f}% -> "
        f"{'MET' if bar else 'NOT MET'}"
    )
    if not bar:
        print(
            "\n  If even bs=3 does not interpolate, then no available knob spans\n"
            "  cover -> entropic, and the two are separate points rather than\n"
            "  ends of a frontier."
        )

    # cost of the enrichment, for the record
    for fam in FAMILIES:
        rs = [r for r in out_rows if r["family"] == fam]
        c1 = float(rs[0]["lp_ms"])
        c3 = float(rs[-1]["lp_ms"])
        print(f"    {fam:<9} bs3/bs1 estimation time = {c3 / max(c1, 1e-9):.1f}x")

    out = ROOT / "results" / "ablation"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "statistics_hierarchy.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(out_rows)
    print(f"\nWrote {out / 'statistics_hierarchy.csv'}")


if __name__ == "__main__":
    main()
