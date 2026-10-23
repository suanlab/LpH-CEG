"""Which block statistic carries the gain, in the headline configuration?

``pushdown_stats_experiment.py`` attributes the bs=2 gain to the
common-neighbour l2 with three norms and the order-relaxed flow LP.  The
headline numbers of the paper use eleven norms (P = {1,...,10,inf}) and the
2^|V| polymatroid program.  This repeats the ablation in that setting, on
both long suites (Hetionet, IMDb; |V| = 6, 8, 10):

  bs1        single-relation statistics (= LpBound's entropic bound)
  free       bs1 + the join-free interfaces of every pair block
  free+l2    free + the l2 norm of each block's common-neighbour sequence
  l2only     bs1 + that l2 alone
  full       the full bs<=2 catalogue

all solved by the same polymatroid program.  A catalogue "recovers the full
bound" when it is within 1.0001x of ``full`` (the criterion of the
three-norm ablation).  Each query runs under a wall-clock cap; queries over
the cap are recorded, not dropped.

Output: results/pushdown/ablation_p11_poly.csv and _summary.csv
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
from scripts.pushdown_stats_experiment import split_stats
from src.core.block import enumerate_blocks
from src.estimators.flow_bound import block_constraints, polymatroid_bound
from src.statistics.lp_stats import compute_all_lp_block_extensions

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "pushdown"
WORKERS = 8
TASK_TIMEOUT_S = 3600
EQ = 1.0001
CATALOGUES = ("bs1", "free", "free_l2", "l2only", "full")


class _TaskTimeoutError(Exception):
    pass


def _alarm(signum, frame):  # type: ignore[no-untyped-def]
    raise _TaskTimeoutError


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
        built = SUITES[suite][1](q)
        assert built is not None
        query, data = built
        attrs = sorted(query.all_attributes)
        blocks = enumerate_blocks(query, max_block_size=2)
        stats = compute_all_lp_block_extensions(query, blocks, data, p_values=P11)
        free, _, _, _ = split_stats(query, stats)

        def cons(st: dict, bs: int = 2) -> list:  # type: ignore[type-arg]
            return block_constraints(
                query, data, max_block_size=bs, p_values=P11, lp_stats=st
            )

        c1, c_free, c_full = cons(stats, 1), cons(free), cons(stats)
        free_set = set(c_free)
        join_l2 = [c for c in c_full if c not in free_set and c.p == 2.0]
        catalogues = {
            "bs1": c1,
            "free": c_free,
            "free_l2": c_free + join_l2,
            "l2only": c1 + join_l2,
            "full": c_full,
        }
        for name, cs in catalogues.items():
            row[f"log2_{name}"] = f"{polymatroid_bound(attrs, cs):.6f}"
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
        for nv in ("6", "8", "10", "all"):
            rs = [
                r
                for r in rows
                if r["suite"] == suite
                and r["status"] == "ok"
                and (nv == "all" or str(r["n_vars"]) == nv)
            ]
            n_all = sum(
                1
                for r in rows
                if r["suite"] == suite and (nv == "all" or str(r["n_vars"]) == nv)
            )
            if not rs:
                continue

            def share(r: dict[str, object], c: str) -> float:
                gain = lg(r, "bs1") - lg(r, "full")
                return (lg(r, "bs1") - lg(r, c)) / gain if gain > 1e-9 else 1.0

            rec: dict[str, object] = {
                "suite": suite,
                "n_vars": nv,
                "n": n_all,
                "n_ok": len(rs),
                "median_full_gain_x": f"{statistics.median(2 ** (lg(r, 'bs1') - lg(r, 'full')) for r in rs):.4g}",
            }
            for c in ("free", "free_l2", "l2only"):
                rec[f"n_{c}_eq_full"] = sum(
                    2 ** (lg(r, c) - lg(r, "full")) < EQ for r in rs
                )
                rec[f"median_{c}_over_full"] = (
                    f"{statistics.median(2 ** (lg(r, c) - lg(r, 'full')) for r in rs):.4g}"
                )
                rec[f"median_{c}_gain_share"] = (
                    f"{statistics.median(share(r, c) for r in rs):.4g}"
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
    _write("ablation_p11_poly.csv", rows)
    summary = summarise(rows)
    _write("ablation_p11_poly_summary.csv", summary)
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
