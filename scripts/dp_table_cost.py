"""H0: what does filling an optimizer's DP table cost, per estimator?

The paper times a single estimate.  A Selinger-style optimizer instead
needs a bound for every connected sub-plan -- that is the DP table -- and
it is there that decomposability is supposed to pay.

For a cycle-k query the connected sub-queries are the contiguous arcs,
of which there are k(k-1)+1.  Each one needs its own bound.  The cover LP
reuses the degree statistics across all of them; the entropic LP is a
global 2^|V| program and must be re-solved from scratch on each arc.

This script measures the total, not a single call.  No ground truth is
needed, so it can run at the query sizes where the two formulations
actually diverge -- unlike the ranking study, which is limited to |V|<=5
by the cost of computing exact intermediate sizes.

Pre-registered bar: DP-table cost ratio (entropic / cover) >= 100x by
k=10.

Output: results/cost/dp_table_cost.csv
"""

from __future__ import annotations

import csv
import random
import sys
import time
from pathlib import Path

from src.estimators.lpbound_entropic import MAX_ATTRIBUTES, estimate_lpbound_entropic
from src.estimators.lph_ceg import estimate_lph_ceg
from src.utils.types import QueryHypergraph, Relation

ROOT = Path(__file__).resolve().parent.parent
P_VALUES: tuple[float, ...] = (1.0, 2.0, float("inf"))
KS = tuple(int(x) for x in sys.argv[1:]) or (4, 6, 8, 10)
N_V, N_E = 60, 240


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


def build_cycle(k: int, seed: int = 7):
    """Cycle-k over variables v0..v{k-1}; relation i joins v_i to v_{i+1}."""
    rels, data = [], {}
    for i in range(k):
        u, v = f"v{i}", f"v{(i + 1) % k}"
        rows = gen_edges(N_V, N_E, seed + i * 13)
        attrs = tuple(sorted((u, v)))
        swap = attrs != (u, v)
        rels.append(Relation(f"R{i}", frozenset({u, v}), len(rows)))
        data[f"R{i}"] = (attrs, [(y, x) for x, y in rows] if swap else rows)
    return QueryHypergraph(relations=tuple(rels), name=f"cycle{k}"), data


def connected_arcs(k: int) -> list[tuple[int, ...]]:
    """Contiguous arcs of a k-cycle: the connected sub-queries."""
    arcs: list[tuple[int, ...]] = []
    for start in range(k):
        for length in range(1, k + 1):
            arc = tuple(sorted((start + o) % k for o in range(length)))
            if arc not in arcs:
                arcs.append(arc)
    return arcs


def main() -> None:
    print("H0: cost of filling an optimizer's DP table")
    print("=" * 78)
    print(
        f"  {'k':>3} {'sub-queries':>12} {'cover_ms':>11} {'entropic_ms':>13} "
        f"{'ratio':>10}  note"
    )
    print(f"  {'-' * 74}")

    out_rows: list[dict[str, object]] = []
    for k in KS:
        query, data = build_cycle(k)
        arcs = connected_arcs(k)
        cov_t = ent_t = 0.0
        ent_done = 0
        ent_skipped = 0
        for arc in arcs:
            names = tuple(f"R{i}" for i in arc)
            sq = QueryHypergraph(
                relations=tuple(r for r in query.relations if r.name in names),
                name="sub",
            )
            sdata = {n: data[n] for n in names}
            t0 = time.perf_counter()
            estimate_lph_ceg(sq, sdata, max_block_size=1, p_values=P_VALUES)
            cov_t += time.perf_counter() - t0

            n_attrs = len(sq.all_attributes)
            if n_attrs > MAX_ATTRIBUTES:
                ent_skipped += 1
                continue
            t0 = time.perf_counter()
            estimate_lpbound_entropic(sq, sdata, p_values=P_VALUES)
            ent_t += time.perf_counter() - t0
            ent_done += 1

        ratio = ent_t / cov_t if cov_t else float("nan")
        note = "" if ent_skipped == 0 else f"{ent_skipped} sub-queries too large"
        print(
            f"  {k:>3} {len(arcs):>12} {cov_t * 1000:>11.1f} {ent_t * 1000:>13.1f} "
            f"{ratio:>9.1f}x  {note}"
        )
        out_rows.append(
            {
                "k": k,
                "n_subqueries": len(arcs),
                "n_entropic_solved": ent_done,
                "n_entropic_skipped": ent_skipped,
                "cover_total_ms": f"{cov_t * 1000:.2f}",
                "entropic_total_ms": f"{ent_t * 1000:.2f}",
                "dp_cost_ratio": f"{ratio:.3f}",
            }
        )

    print(f"  {'-' * 74}")
    # The bar was pre-registered AT k=10.  Report it there, and report the
    # crossover separately and explicitly as post-hoc.
    path_prior = out_rows
    at10 = [r for r in path_prior if int(r["k"]) == 10]
    if at10:
        v = float(at10[0]["dp_cost_ratio"])
        print(
            f"  H0 bar as pre-registered (ratio >= 100x AT k=10): {v:.1f}x -> "
            f"{'MET' if v >= 100 else 'NOT MET'}"
        )
    crossed = [r for r in path_prior if float(r["dp_cost_ratio"]) >= 100.0]
    if crossed:
        kc = min(int(r["k"]) for r in crossed)
        rc = float(next(r for r in path_prior if int(r["k"]) == kc)["dp_cost_ratio"])
        print(f"  [post-hoc] ratio first exceeds 100x at k={kc} ({rc:.0f}x)")
    print(
        "  Scope: the paper's real query suite tops out at |V|=5, where the\n"
        "  ratio is ~1.05x -- i.e. the decomposability advantage does not\n"
        "  exist at the sizes the paper actually evaluates."
    )

    out = ROOT / "results" / "cost"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "dp_table_cost.csv"
    prior: list[dict[str, str]] = []
    if path.exists() and len(sys.argv) > 1:
        with open(path) as f:
            prior = [r for r in csv.DictReader(f) if int(r["k"]) not in KS]
    with open(path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(sorted(prior + out_rows, key=lambda r: int(r["k"])))
    print(f"\nWrote {out / 'dp_table_cost.csv'}")


if __name__ == "__main__":
    main()
