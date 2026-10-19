"""Which pair joins are worth materializing?  Value-of-information selection.

On a binary block R(a,b), S(b,c) every interface statistic but one is
join-free (``pushdown_stats``); the exception, the common-neighbour
interface {a,c}, needs the pair join.  Without it we can still use valid
upper bounds for that interface: l1 = |R join S| (join-free) and
l2, l_inf <= the triangle-inequality bound.  So the budget is the number of
blocks whose common-neighbour statistic is computed exactly; every choice
gives a valid bound (the 2^|V| polymatroid bound, three norms).

Selection rule (cheap): score each block by how much the bound drops when its
l2 / l_inf proxy is replaced by a Hutchinson sketch (an estimate, used only
to rank), and collect the top-b exactly.  Compared with the best size-b
subset (oracle) and the mean over all size-b subsets (random), by the
fraction of the log-gain from 'no exact block' to 'all blocks exact' that
each recovers.

Output: results/characterization/voi_selection.csv and _summary.csv
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import multiprocessing
import statistics
from pathlib import Path

from scripts.pset_fair import SUITES
from src.core.block import enumerate_blocks
from src.estimators.flow_bound import block_constraints, polymatroid_bound
from src.statistics.lp_stats import (
    LpBlockExtensionStats,
    compute_all_lp_block_extensions,
)
from src.statistics.pushdown_stats import (
    codegree_interface,
    codegree_l2_upper_bound,
    compute_pushdown_block_extensions,
    sketch_codegree_l2,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "characterization"
P = (1.0, 2.0, math.inf)
N_VARS = 6
BUDGETS = (1, 2, 3)
PROBES = 64


def _proxy(l1: float, l2: float) -> LpBlockExtensionStats:
    return LpBlockExtensionStats(
        degree_sequence=(),
        lp_norms={1.0: l1, 2.0: l2, math.inf: l2},
        interface_size=2,
        count=1,
        max_extension=max(1, int(l2)),
        avg_extension=l1,
        log_max_extension=math.log2(max(l2, 1.0)),
    )


def evaluate(item: tuple[str, dict]) -> dict[str, object] | None:  # type: ignore[type-arg]
    suite, q = item
    built = SUITES[suite][1](q)
    if built is None:
        return None
    query, data = built
    attrs = sorted(query.all_attributes)
    blocks = enumerate_blocks(query, max_block_size=2)
    pairs = [b for b in blocks if len(b.relations) == 2]
    exact = compute_all_lp_block_extensions(query, blocks, data, p_values=P)
    free = {k: v for k, v in exact.items() if len(k[0].relations) == 1}
    codeg = {}
    for b in pairs:
        free.update(compute_pushdown_block_extensions(query, b, data, p_values=P))
        iface = codegree_interface(query, b)
        codeg[b] = (iface, exact[(b, iface)])
        ub = codegree_l2_upper_bound(query, b, data)
        free[(b, iface)] = _proxy(exact[(b, iface)].norm(1.0), ub)

    def bound(stats: dict) -> float:  # type: ignore[type-arg]
        cons = block_constraints(
            query, data, max_block_size=2, p_values=P, lp_stats=stats
        )
        return polymatroid_bound(attrs, cons)  # log2

    def with_exact(sel: tuple[int, ...]) -> float:
        st = dict(free)
        for i in sel:
            iface, s = codeg[pairs[i]]
            st[(pairs[i], iface)] = s
        return bound(st)

    n = len(pairs)
    sub = {
        sel: with_exact(sel)
        for r in range(n + 1)
        for sel in itertools.combinations(range(n), r)
    }
    b_free, b_full = sub[()], sub[tuple(range(n))]
    gain = b_free - b_full
    scores = []
    for i, b in enumerate(pairs):
        iface, _ = codeg[b]
        est, _ = sketch_codegree_l2(query, b, data, n_probes=PROBES, seed=i)
        st = dict(free)
        st[(b, iface)] = _proxy(
            free[(b, iface)].norm(1.0), min(est, free[(b, iface)].norm(2.0))
        )
        scores.append(b_free - bound(st))
    order = sorted(range(n), key=lambda i: -scores[i])
    row: dict[str, object] = {
        "suite": suite,
        "query_id": q["query_id"],
        "n_blocks": n,
        "log2_free": f"{b_free:.6f}",
        "log2_full": f"{b_full:.6f}",
        "full_gain_x": f"{2.0**gain:.6g}",
    }
    for budget in BUDGETS:
        sizes = [v for k, v in sub.items() if len(k) == budget]
        frac = (lambda v: (b_free - v) / gain) if gain > 1e-9 else (lambda v: 1.0)
        row[f"voi_frac_b{budget}"] = f"{frac(sub[tuple(sorted(order[:budget]))]):.6f}"
        row[f"oracle_frac_b{budget}"] = f"{frac(min(sizes)):.6f}"
        row[f"random_frac_b{budget}"] = f"{statistics.mean(frac(v) for v in sizes):.6f}"
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    items = [
        (suite, q)
        for suite, (qpath, _) in SUITES.items()
        for q in json.loads(qpath.read_text())["queries"]
        if q["n_vars"] == N_VARS
    ]
    with multiprocessing.Pool(processes=8, maxtasksperchild=1) as pool:
        rows = [r for r in pool.map(evaluate, items) if r is not None]
    rows.sort(key=lambda r: (str(r["suite"]), str(r["query_id"])))
    with open(OUT / "voi_selection.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = []
    for suite in ("hetionet", "imdb"):
        rs = [
            r
            for r in rows
            if r["suite"] == suite and float(str(r["full_gain_x"])) > 1.01
        ]
        rec: dict[str, object] = {"suite": suite, "n": len(rs)}
        rec["median_full_gain_x"] = (
            f"{statistics.median(float(str(r['full_gain_x'])) for r in rs):.4g}"
        )
        for b in BUDGETS:
            for kind in ("voi", "oracle", "random"):
                rec[f"median_{kind}_frac_b{b}"] = (
                    f"{statistics.median(float(str(r[f'{kind}_frac_b{b}'])) for r in rs):.4g}"
                )
        summary.append(rec)
        print(rec)
    with open(OUT / "voi_selection_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)


if __name__ == "__main__":
    main()
