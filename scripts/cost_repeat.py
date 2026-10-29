"""Repeated, thread-pinned runs of the independent cost table.

``cost_accuracy_indep.py`` times each (query, catalogue) task once, with
eight workers and the BLAS/OpenMP thread pools left at their defaults.  This
reruns the same tasks ``REPS`` times, each repetition in its own shuffled
order, with every native thread pool pinned to one thread (set before numpy
is imported, so the forked workers inherit it), and reports the spread of the
per-repetition medians and a query-bootstrap interval for the collection
ratio against LpBound's statistics.  The server is shared with other users;
each task records the 1-minute load average before and after it.  Bounds must agree across repetitions
(they are deterministic); timings are what is repeated.

Output: results/cost/cost_repeat.csv (one row per task and repetition) and
cost_repeat_summary.csv
"""

from __future__ import annotations

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_var] = "1"

import csv  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import random  # noqa: E402
import statistics  # noqa: E402
from pathlib import Path  # noqa: E402

from scripts.cost_accuracy_indep import CATALOGUES, evaluate  # noqa: E402
from scripts.pset_fair import SUITES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "cost"
WORKERS = 8
REPS = 3
SEED = 2027
BOOT = 2000


def _run(item: tuple[int, str, dict, str]) -> dict[str, object]:  # type: ignore[type-arg]
    rep, suite, q, name = item
    load_before = os.getloadavg()[0]
    row = evaluate((suite, q, name))
    # the server is shared: record the 1-minute load around each task
    return {
        "rep": rep,
        **row,
        "load1_before": f"{load_before:.1f}",
        "load1_after": f"{os.getloadavg()[0]:.1f}",
    }


def summarise(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    def f(r: dict[str, object], c: str) -> float:
        return float(str(r[c]))

    ok: dict[tuple[str, str, int], dict[str, dict[str, object]]] = {}
    for r in rows:
        if r["status"] == "ok":
            key = (str(r["suite"]), str(r["query_id"]), int(str(r["rep"])))
            ok.setdefault(key, {})[str(r["catalogue"])] = r
    out: list[dict[str, object]] = []
    rng = random.Random(SEED)
    for suite in ("hetionet", "imdb"):
        qids = sorted(
            {
                q
                for (s, q, _) in ok
                if s == suite
                and all(
                    len(ok.get((s, q, k), {})) == len(CATALOGUES) for k in range(REPS)
                )
            }
        )
        for name in CATALOGUES:
            per_rep = [
                statistics.median(
                    f(ok[(suite, q, k)][name], "collect_s")
                    / f(ok[(suite, q, k)]["lpbound"], "collect_s")
                    for q in qids
                )
                for k in range(REPS)
            ]
            # per query: the median over repetitions of the ratio
            per_q = [
                statistics.median(
                    f(ok[(suite, q, k)][name], "collect_s")
                    / f(ok[(suite, q, k)]["lpbound"], "collect_s")
                    for k in range(REPS)
                )
                for q in qids
            ]
            boot = sorted(
                statistics.median(rng.choices(per_q, k=len(per_q))) for _ in range(BOOT)
            )
            cv = [
                statistics.pstdev(xs) / statistics.mean(xs)
                for q in qids
                if statistics.mean(
                    xs := [f(ok[(suite, q, k)][name], "collect_s") for k in range(REPS)]
                )
                > 0
            ]
            bound_spread = max(
                max(f(ok[(suite, q, k)][name], "log2_bound") for k in range(REPS))
                - min(f(ok[(suite, q, k)][name], "log2_bound") for k in range(REPS))
                for q in qids
            )
            out.append(
                {
                    "suite": suite,
                    "catalogue": name,
                    "n_queries": len(qids),
                    "reps": REPS,
                    "rep_median_collect_over_lpbound": ";".join(
                        f"{x:.4g}" for x in per_rep
                    ),
                    "min_rep_median": f"{min(per_rep):.4g}",
                    "max_rep_median": f"{max(per_rep):.4g}",
                    "median_collect_over_lpbound": f"{statistics.median(per_q):.4g}",
                    "boot95_lo": f"{boot[int(0.025 * BOOT)]:.4g}",
                    "boot95_hi": f"{boot[int(0.975 * BOOT) - 1]:.4g}",
                    "median_cv_collect_s": f"{statistics.median(cv):.3g}",
                    "max_abs_log2_bound_spread": f"{bound_spread:.2e}",
                    "median_load1": f"{statistics.median(float(str(ok[(suite, q, k)][name]['load1_before'])) for q in qids for k in range(REPS)):.1f}",
                }
            )
        out.append(
            {
                "suite": suite,
                "catalogue": "checks",
                "n_tasks_not_ok": sum(
                    1 for r in rows if r["suite"] == suite and r["status"] != "ok"
                ),
            }
        )
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
    rows: list[dict[str, object]] = []
    for rep in range(REPS):
        items = [
            (rep, suite, q, name)
            for suite in ("hetionet", "imdb")
            for q in json.loads(SUITES[suite][0].read_text())["queries"]
            for name in CATALOGUES
        ]
        random.Random(SEED + rep).shuffle(items)
        with multiprocessing.Pool(processes=WORKERS, maxtasksperchild=1) as pool:
            rows += pool.map(_run, items, chunksize=1)
        print(f"rep {rep} done", flush=True)
    rows.sort(
        key=lambda r: (
            str(r["suite"]),
            str(r["query_id"]),
            str(r["catalogue"]),
            int(str(r["rep"])),
        )
    )
    _write("cost_repeat.csv", rows)
    summary = summarise(rows)
    _write("cost_repeat_summary.csv", summary)
    for s in summary:
        print(s)


if __name__ == "__main__":
    import sys

    if "--summarise" in sys.argv:
        stored = list(csv.DictReader((OUT / "cost_repeat.csv").open()))
        _write("cost_repeat_summary.csv", summarise(stored))  # type: ignore[arg-type]
    else:
        main()
