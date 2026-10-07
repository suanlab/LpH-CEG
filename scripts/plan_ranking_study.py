"""H5 + H0: does looseness matter for plan choice, and what does reuse cost?

Two questions the paper never asks, both answered by the same measurement.

**H5 (does tightness matter?)**  The paper treats being 1.99x looser than
the entropic optimum as a cost.  But an optimizer does not consume the
absolute number -- it ranks plans.  A bound that is uniformly 2x loose
ranks plans exactly as a tight one does.  And our own gap predictor
(R^2=0.87) says the looseness *is* systematic rather than noise.  So we
measure rank agreement with the true plan cost, not q-error.

**H0 (what does a plan search actually cost?)**  A DP optimizer needs a
bound per *sub-plan*, not one per query.  The cover LP was designed to
decompose, so sub-plan bounds come from the same statistics; the entropic
LP has to be re-solved from scratch on each sub-plan, because its
2^|V| program is global.  Costing a whole search, rather than one
estimate, is where decomposability is supposed to pay -- and it has never
been measured.

Protocol.  For each query we enumerate connected left-deep orders, and
score each order by the sum of its prefix cardinality estimates -- the
standard intermediate-size cost model.  Rankings are then compared, by
Kendall tau, against the ranking induced by the TRUE intermediate sizes.

Pre-registered bars:
  H5  tau(cover, true) >= tau(entropic, true) - 0.05  (cover ranks plans
      no worse than the tight bound, to within noise)
  H0  total plan-search cost of the entropic route >= 100x the cover route

Output: results/cost/plan_ranking.csv
"""

from __future__ import annotations

import csv
import itertools
import random
import time
from pathlib import Path

from src.estimators.lpbound_entropic import estimate_lpbound_entropic
from src.estimators.lph_ceg import estimate_lph_ceg
from src.utils.types import QueryHypergraph, Relation

ROOT = Path(__file__).resolve().parent.parent
P_VALUES: tuple[float, ...] = (1.0, 2.0, float("inf"))
MAX_ORDERS = 24  # cap on enumerated orders per query
TRUE_CAP = 400_000  # give up on true intermediate sizes beyond this


def cyc(k: int) -> list[tuple[str, str]]:
    v = [chr(ord("a") + i) for i in range(k)]
    return [(v[i], v[(i + 1) % k]) for i in range(k)]


FAMILIES: dict[str, list[tuple[str, str]]] = {
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


def build(edges: list[tuple[str, str]], nv: int, ne: int, seed: int):
    rels, data = [], {}
    for i, (u, v) in enumerate(edges):
        rows = gen_edges(nv, ne, seed + i * 7)
        attrs = tuple(sorted((u, v)))
        swap = attrs != (u, v)
        rels.append(Relation(f"R{i}", frozenset({u, v}), len(rows)))
        data[f"R{i}"] = (attrs, [(y, x) for x, y in rows] if swap else rows)
    return QueryHypergraph(relations=tuple(rels), name="q"), data


def sub_query(query: QueryHypergraph, names: tuple[str, ...]) -> QueryHypergraph:
    keep = tuple(r for r in query.relations if r.name in names)
    return QueryHypergraph(relations=keep, name="sub")


def connected_orders(query: QueryHypergraph, cap: int) -> list[tuple[str, ...]]:
    """Left-deep orders whose every prefix is connected."""
    names = [r.name for r in query.relations]
    attrs = {r.name: set(r.attributes) for r in query.relations}
    out: list[tuple[str, ...]] = []
    for perm in itertools.permutations(names):
        seen = set(attrs[perm[0]])
        ok = True
        for nm in perm[1:]:
            if not (seen & attrs[nm]):
                ok = False
                break
            seen |= attrs[nm]
        if ok:
            out.append(perm)
            if len(out) >= cap:
                break
    return out


def true_prefix_sizes(
    order: tuple[str, ...], data: dict, attrs: dict[str, tuple[str, ...]]
) -> list[int] | None:
    """Exact intermediate sizes, or None if the join blows past TRUE_CAP."""
    cur: list[dict[str, str]] = [{}]
    sizes: list[int] = []
    for nm in order:
        cols, rows = data[nm]
        nxt: list[dict[str, str]] = []
        for env in cur:
            for row in rows:
                b = dict(zip(cols, row, strict=True))
                if all(env.get(k, b[k]) == b[k] for k in b):
                    e = dict(env)
                    e.update(b)
                    nxt.append(e)
                    if len(nxt) > TRUE_CAP:
                        return None
        cur = nxt
        sizes.append(len(cur))
    del attrs
    return sizes


def kendall_tau(a: list[float], b: list[float]) -> float:
    n = len(a)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 or db == 0:
                continue
            if (da > 0) == (db > 0):
                conc += 1
            else:
                disc += 1
    tot = conc + disc
    return (conc - disc) / tot if tot else float("nan")


def main() -> None:
    print("H5/H0: plan-ranking fidelity and the cost of a whole plan search")
    print("=" * 78)
    print(
        f"  {'query':<10} {'orders':>7} {'tau_cov':>8} {'tau_ent':>8} "
        f"{'cover_ms':>10} {'entropic_ms':>12} {'ratio':>9}"
    )
    print(f"  {'-' * 74}")

    out_rows: list[dict[str, object]] = []
    for fam, edges in FAMILIES.items():
        query, data = build(edges, 40, 90, seed=11)
        attrs = {r.name: tuple(sorted(r.attributes)) for r in query.relations}
        orders = connected_orders(query, MAX_ORDERS)

        cov_cost, ent_cost, true_cost = [], [], []
        cov_t = ent_t = 0.0
        n_calls = 0
        for order in orders:
            ts = true_prefix_sizes(order, data, attrs)
            if ts is None:
                continue
            c_sum = e_sum = 0.0
            for i in range(1, len(order) + 1):
                sq = sub_query(query, order[:i])
                sdata = {n: data[n] for n in order[:i]}
                t0 = time.perf_counter()
                c_sum += estimate_lph_ceg(
                    sq, sdata, max_block_size=1, p_values=P_VALUES
                ).estimate
                cov_t += time.perf_counter() - t0
                t0 = time.perf_counter()
                e_sum += estimate_lpbound_entropic(sq, sdata, p_values=P_VALUES)
                ent_t += time.perf_counter() - t0
                n_calls += 1
            cov_cost.append(c_sum)
            ent_cost.append(e_sum)
            true_cost.append(float(sum(ts)))

        if len(true_cost) < 3:
            print(f"  {fam:<10} (too few evaluable orders)")
            continue

        tau_c = kendall_tau(cov_cost, true_cost)
        tau_e = kendall_tau(ent_cost, true_cost)
        ratio = ent_t / cov_t if cov_t else float("nan")
        print(
            f"  {fam:<10} {len(true_cost):>7} {tau_c:>8.3f} {tau_e:>8.3f} "
            f"{cov_t * 1000:>10.1f} {ent_t * 1000:>12.1f} {ratio:>8.1f}x"
        )
        out_rows.append(
            {
                "family": fam,
                "n_orders": len(true_cost),
                "n_bound_calls": n_calls,
                "tau_cover_vs_true": f"{tau_c:.4f}",
                "tau_entropic_vs_true": f"{tau_e:.4f}",
                "tau_cover_vs_entropic": f"{kendall_tau(cov_cost, ent_cost):.4f}",
                "cover_total_ms": f"{cov_t * 1000:.2f}",
                "entropic_total_ms": f"{ent_t * 1000:.2f}",
                "search_cost_ratio": f"{ratio:.2f}",
            }
        )

    print(f"  {'-' * 74}")
    if out_rows:
        d_tau = [
            float(r["tau_cover_vs_true"]) - float(r["tau_entropic_vs_true"])
            for r in out_rows
        ]
        worst = min(d_tau)
        ratios = [float(r["search_cost_ratio"]) for r in out_rows]
        h5 = worst >= -0.05
        h0 = min(ratios) >= 100.0
        print(
            f"  H5 (cover ranks no worse than entropic, -0.05 tol): "
            f"worst delta = {worst:+.3f} -> {'MET' if h5 else 'NOT MET'}"
        )
        print(
            f"  H0 (search cost ratio >= 100x): min = {min(ratios):.1f}x "
            f"-> {'MET' if h0 else 'NOT MET'}"
        )

    out = ROOT / "results" / "cost"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "plan_ranking.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(out_rows)
    print(f"\nWrote {out / 'plan_ranking.csv'}")


if __name__ == "__main__":
    main()
