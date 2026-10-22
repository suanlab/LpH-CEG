"""What does sketch-guided selection of pair joins save in collection time?

``voi_selection.py`` shows that joining only the three blocks (of six, at
|V|=6) whose sketched common-neighbour norm lowers the bound most recovers
the full bs<=2 bound on the median query.  This measures the collection
cost of that plan against collecting every bs<=2 statistic exactly, on the
same queries, in the same process (so the ratio is robust to machine load):

  full     = every single-relation and pair-block statistic, exactly
             (``compute_all_lp_block_extensions``, what the bs<=2 bounds use)
  selected = single-relation statistics
           + the join-free interfaces of every pair block (pushdown)
           + a 64-probe sketch of every pair block's common-neighbour norm
           + ranking the blocks (one polymatroid solve per block)
           + the exact statistics of the top three blocks

Three norms, as in ``voi_selection.py``.  The selected plan's bound and the
fraction of the full gain it recovers are recomputed here, so cost and
benefit come from the same run.

Output: results/characterization/voi_cost.csv and voi_cost_summary.csv
"""

from __future__ import annotations

import csv
import json
import multiprocessing
import statistics
import time
from pathlib import Path

from scripts.pset_fair import SUITES
from scripts.voi_selection import PROBES, P, _proxy
from src.core.block import enumerate_blocks
from src.estimators.flow_bound import block_constraints, polymatroid_bound
from src.statistics.lp_stats import compute_all_lp_block_extensions
from src.statistics.pushdown_stats import (
    codegree_interface,
    codegree_l2_upper_bound,
    compute_pushdown_block_extensions,
    sketch_codegree_l2,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "characterization"
N_VARS = 6
BUDGET = 3


def evaluate(item: tuple[str, dict]) -> dict[str, object] | None:  # type: ignore[type-arg]
    suite, q = item
    built = SUITES[suite][1](q)
    if built is None:
        return None
    query, data = built
    attrs = sorted(query.all_attributes)
    blocks = enumerate_blocks(query, max_block_size=2)
    singles = [b for b in blocks if len(b.relations) == 1]
    pairs = [b for b in blocks if len(b.relations) == 2]

    def bound(stats: dict) -> float:  # type: ignore[type-arg]
        cons = block_constraints(
            query, data, max_block_size=2, p_values=P, lp_stats=stats
        )
        return polymatroid_bound(attrs, cons)  # log2

    # full: every statistic exactly, timed block by block so the cost of any
    # subset of pair blocks is the sum of its blocks' times
    t0 = time.perf_counter()
    exact = compute_all_lp_block_extensions(query, singles, data, p_values=P)
    t_singles = time.perf_counter() - t0
    t_block: list[float] = []
    for b in pairs:
        t0 = time.perf_counter()
        exact.update(compute_all_lp_block_extensions(query, [b], data, p_values=P))
        t_block.append(time.perf_counter() - t0)
    t_full = t_singles + sum(t_block)

    # selected plan, step by step
    t0 = time.perf_counter()
    free = {k: v for k, v in exact.items() if len(k[0].relations) == 1}
    for b in pairs:
        free.update(compute_pushdown_block_extensions(query, b, data, p_values=P))
        iface = codegree_interface(query, b)
        ub = codegree_l2_upper_bound(query, b, data)
        # the l1 norm of the common-neighbour sequence is the join size, which
        # is join-free (sum_b deg_R(b) deg_S(b)); it is read from the exact run
        # rather than recomputed, so its (small) cost is not charged here
        free[(b, iface)] = _proxy(exact[(b, iface)].norm(1.0), ub)
    t_free = t_singles + time.perf_counter() - t0

    t0 = time.perf_counter()
    sketched = []
    for i, b in enumerate(pairs):
        est, _ = sketch_codegree_l2(query, b, data, n_probes=PROBES, seed=i)
        sketched.append(est)
    t_sketch = time.perf_counter() - t0

    t0 = time.perf_counter()
    b_free = bound(free)
    scores = []
    for b, est in zip(pairs, sketched, strict=True):
        iface = codegree_interface(query, b)
        st = dict(free)
        st[(b, iface)] = _proxy(
            free[(b, iface)].norm(1.0), min(est, free[(b, iface)].norm(2.0))
        )
        scores.append(b_free - bound(st))
    top = sorted(range(len(pairs)), key=lambda i: -scores[i])[:BUDGET]
    t_rank = time.perf_counter() - t0

    chosen = {k: v for k, v in exact.items() if k[0] in {pairs[i] for i in top}}
    t_exact_top = sum(t_block[i] for i in top)

    sel = dict(free)
    sel.update(chosen)
    b_sel, b_full = bound(sel), bound(exact)
    gain = b_free - b_full
    frac = (b_free - b_sel) / gain if gain > 1e-9 else 1.0
    t_sel = t_free + t_sketch + t_rank + t_exact_top
    return {
        "suite": suite,
        "query_id": q["query_id"],
        "n_blocks": len(pairs),
        "full_gain_x": f"{2.0**gain:.6g}",
        "selected_frac_of_gain": f"{frac:.6f}",
        "t_full_s": f"{t_full:.3f}",
        "max_block_share": f"{max(t_block) / max(t_full, 1e-9):.4f}",
        "t_free_s": f"{t_free:.3f}",
        "t_sketch_s": f"{t_sketch:.3f}",
        "t_rank_s": f"{t_rank:.3f}",
        "t_exact_top_s": f"{t_exact_top:.3f}",
        "t_selected_s": f"{t_sel:.3f}",
        "selected_over_full_time": f"{t_sel / max(t_full, 1e-9):.6f}",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    items = [
        (suite, q)
        for suite, (qpath, _) in SUITES.items()
        for q in json.loads(qpath.read_text())["queries"]
        if q["n_vars"] == N_VARS
    ]
    with multiprocessing.Pool(processes=4, maxtasksperchild=1) as pool:
        rows = [r for r in pool.map(evaluate, items, chunksize=1) if r is not None]
    rows.sort(key=lambda r: (str(r["suite"]), str(r["query_id"])))
    with open(OUT / "voi_cost.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = []
    for suite in ("hetionet", "imdb"):
        rs = [r for r in rows if r["suite"] == suite]
        gainful = [r for r in rs if float(str(r["full_gain_x"])) > 1.01]

        def med(col: str, xs: list[dict[str, object]] = rs) -> str:
            return f"{statistics.median(float(str(r[col])) for r in xs):.4g}"

        summary.append(
            {
                "suite": suite,
                "n": len(rs),
                "n_gainful": len(gainful),
                "median_t_full_s": med("t_full_s"),
                "median_t_selected_s": med("t_selected_s"),
                "median_selected_over_full_time": med("selected_over_full_time"),
                "median_selected_frac_of_gain_gainful": med(
                    "selected_frac_of_gain", gainful
                ),
            }
        )
        print(summary[-1])
    with open(OUT / "voi_cost_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)


if __name__ == "__main__":
    main()
