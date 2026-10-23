"""How much of the block-statistics gain lies outside a proxy for CorrBound's?

Interfaces of a block of two binary relations R(a, b), S(b, c), grouped by
what they count:

  size      {}                       the pair-join size |R join S|
  products  {b}, {a,b}, {b,c}, {a,b,c}   functions of deg_R(b) and deg_S(b)
            alone: their l_p norms are inner products
            sum_b deg_R(b)^q deg_S(b)^r with (q, r) in {(1,1), (p,p), (1,p),
            (p,1)}, an exact proxy for (a subset of) CorrBound's statistics
  one_side  {a}, {c}                 which a's reach which b's (join-free)
  codeg     {a, c}                   common-neighbour counts (needs the join)

Same queries (Hetionet and IMDb long suites), eleven norms, the 2^|V|
polymatroid program.  Catalogs, cumulative from LpBound's single-relation
statistics:

  bs1 -> +size -> +products (the CorrBound proxy) -> +one_side
      -> +codeg (= the full bs<=2 catalog)

and, the other way round, bs1 + one_side + codeg (only the statistics outside
that proxy).  The proxy is exact (no sketches) but has only these exponent
pairs, no per-relation p_i and no k-way products.  The
statistics come from the sparse-product path (equal to the materialized
join's, tests/unit/test_pushdown_stats.py).

Output: results/pushdown/ablation_overlap.csv and _summary.csv
"""

from __future__ import annotations

import csv
import json
import multiprocessing
import signal
import statistics
import time
from pathlib import Path

from scripts.pset_fair import P11, SUITES
from src.core.block import enumerate_blocks
from src.estimators.flow_bound import block_constraints, polymatroid_bound
from src.statistics.lp_stats import compute_all_lp_block_extensions
from src.statistics.pushdown_stats import (
    compute_sparse_block_extensions,
    join_free_interfaces,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "pushdown"
WORKERS = 8
TASK_TIMEOUT_S = 3600
STEPS = {
    "bs1": (),
    "size": ("size",),
    "products": ("size", "products"),
    "one_side": ("size", "products", "one_side"),
    "full": ("size", "products", "one_side", "codeg"),
    "new_only": ("one_side", "codeg"),
}


class _TaskTimeoutError(Exception):
    pass


def _alarm(signum, frame):  # type: ignore[no-untyped-def]
    raise _TaskTimeoutError


def group(kind: str | None, interface: frozenset[str]) -> str:
    if kind is None:
        return "codeg"
    if not interface:
        return "size"
    return "products" if kind == "shared" else "one_side"


def evaluate(item: tuple[str, dict]) -> dict[str, object]:  # type: ignore[type-arg]
    suite, q = item
    row: dict[str, object] = {
        "suite": suite,
        "query_id": q["query_id"],
        "n_vars": q["n_vars"],
    }
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(TASK_TIMEOUT_S)
    t0 = time.perf_counter()
    try:
        query, data = SUITES[suite][1](q)
        attrs = sorted(query.all_attributes)
        blocks = enumerate_blocks(query, max_block_size=2)
        singles = [b for b in blocks if len(b.relations) == 1]
        stats = dict(
            compute_all_lp_block_extensions(query, singles, data, p_values=P11)
        )
        tagged: dict[str, dict] = {
            g: {} for g in ("size", "products", "one_side", "codeg")
        }  # type: ignore[type-arg]
        for b in blocks:
            if len(b.relations) != 2:
                continue
            kinds = join_free_interfaces(query, b)
            for key, st in compute_sparse_block_extensions(
                query, b, data, p_values=P11
            ).items():
                tagged[group(kinds[key[1]], key[1])][key] = st
        for name, groups in STEPS.items():
            st = dict(stats)
            for g in groups:
                st.update(tagged[g])
            cons = block_constraints(
                query,
                data,
                max_block_size=2 if groups else 1,
                p_values=P11,
                lp_stats=st,
            )
            row[f"log2_{name}"] = f"{polymatroid_bound(attrs, cons):.6f}"
        row["status"] = "ok"
    except _TaskTimeoutError:
        row["status"] = f"over {TASK_TIMEOUT_S} s"
    finally:
        signal.alarm(0)
    row["seconds"] = f"{time.perf_counter() - t0:.1f}"
    return row


def summarise(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    def lg(r: dict[str, object], c: str) -> float:
        return float(str(r[f"log2_{c}"]))

    out: list[dict[str, object]] = []
    for suite in ("hetionet", "imdb"):
        rs = [r for r in rows if r["suite"] == suite and r["status"] == "ok"]
        rec: dict[str, object] = {"suite": suite, "n": len(rs)}
        for name in STEPS:
            if name == "bs1":
                continue
            gains = [lg(r, "bs1") - lg(r, name) for r in rs]
            shares = [
                (lg(r, "bs1") - lg(r, name)) / max(lg(r, "bs1") - lg(r, "full"), 1e-9)
                for r in rs
            ]
            rec[f"median_gain_{name}"] = f"{statistics.median(2**g for g in gains):.4g}"
            rec[f"median_share_{name}"] = f"{statistics.median(shares):.4g}"
        beyond = [lg(r, "products") - lg(r, "full") for r in rs]
        rec["median_full_over_products_gain"] = (
            f"{statistics.median(2**g for g in beyond):.4g}"
        )
        rec["n_full_tighter_than_products"] = sum(g > 1e-6 for g in beyond)
        rec["median_share_beyond_products"] = (
            f"{statistics.median(1 - (lg(r, 'bs1') - lg(r, 'products')) / max(lg(r, 'bs1') - lg(r, 'full'), 1e-9) for r in rs):.4g}"
        )
        out.append(rec)
    return out


def _write(name: str, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(OUT / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    items = [
        (suite, q)
        for suite in ("hetionet", "imdb")
        for q in json.loads(SUITES[suite][0].read_text())["queries"]
    ]
    with multiprocessing.Pool(processes=WORKERS, maxtasksperchild=1) as pool:
        rows = pool.map(evaluate, items, chunksize=1)
    rows.sort(key=lambda r: (str(r["suite"]), str(r["query_id"])))
    _write("ablation_overlap.csv", rows)
    summary = summarise(rows)
    _write("ablation_overlap_summary.csv", summary)
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
