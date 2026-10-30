"""Do the headline margins survive independent data slices?

The Hetionet long suite caps every relation at its first 20K rows in file
order, and the IMDb suite keeps the 200 most-voted feature films.  Query
counts are not independent data conditions: every query of a suite reads
the same slice.  This reruns the headline comparison -- LpBound's entropic
bound against the bs<=2 polymatroid bound, eleven norms on both sides, pair
statistics from the sparse collector -- on slices fixed before any result
was seen:

  hetionet/prefix       the registered first-20K-row slice (same code path)
  hetionet/seed1..10    each relation a uniform 20K-row sample of the full
                        file, one seed per slice (seed 1 and 2 at |V|=6 are
                        ``slice_robustness.py``'s)
  imdb/top200           the registered 200 most-voted films
  imdb/rank201 ... rank1601   200 films from lower popularity ranks
  imdb/random5000_s1, _s2     200 films drawn uniformly from the top 5,000

(IMDb strata: ``prepare_imdb_cycles.py --rank-offset / --random-from`` into
``data/imdb_strata/``.)  Every registered query runs on every slice; exact
counts under set semantics are taken where DuckDB finishes within the cap,
and timeouts are recorded, not dropped.

Output: results/suite/slice_independent.csv and _summary.csv
"""

from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing
import random
import signal
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING

import scripts.evaluate_imdb_suite as eis
import scripts.evaluate_long_suite as els
import scripts.evaluate_query_suite as eqs
from scripts.pset_fair import P11, SUITES
from src.core.block import enumerate_blocks
from src.estimators.flow_bound import block_constraints, polymatroid_bound
from src.statistics.lp_stats import compute_all_lp_block_extensions
from src.statistics.pushdown_stats import compute_sparse_block_extensions

if TYPE_CHECKING:
    from src.statistics.lp_stats import LpBlockExtensionStats
    from src.utils.types import Block, Interface

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "suite"
STRATA = ROOT / "data" / "imdb_strata"
WORKERS = 8
DUCKDB_THREADS = 2
EXACT_TIMEOUT_S = 60.0
TASK_TIMEOUT_S = 3600
HETIONET_SEEDS = tuple(range(1, 11))
IMDB_STRATA = (
    "rank201",
    "rank401",
    "rank801",
    "rank1601",
    "random5000_s1",
    "random5000_s2",
)


class _TaskTimeoutError(Exception):
    pass


def _alarm(signum, frame):  # type: ignore[no-untyped-def]
    raise _TaskTimeoutError


def conditions() -> list[tuple[str, str]]:
    return [
        ("hetionet", "prefix"),
        *(("hetionet", f"seed{s}") for s in HETIONET_SEEDS),
        ("imdb", "top200"),
        *(("imdb", s) for s in IMDB_STRATA),
    ]


def _sampled_loader(seed: int):  # type: ignore[no-untyped-def]
    def load(fname: str, base: Path, has_header: bool = True) -> list[tuple[str, str]]:
        with open(base / fname) as f:
            if has_header:
                f.readline()
            rows = [
                (p[0], p[1])
                for p in (ln.rstrip("\n").replace("\t", " ").split() for ln in f)
                if len(p) >= 2
            ]
        if len(rows) <= eqs.MAX_ROWS:
            return rows
        # per-file seed: the same relation gets the same sample in every query
        return random.Random(f"{seed}:{fname}").sample(rows, eqs.MAX_ROWS)

    return load


def _use_condition(suite: str, cond: str) -> None:
    """Point the suite's loader at the slice (each task is a fresh process)."""
    if suite == "hetionet" and cond != "prefix":
        eqs.load_edges = _sampled_loader(int(cond.removeprefix("seed")))
    if suite == "imdb" and cond != "top200":
        eis.DATA = STRATA / cond
        eis._cache.clear()


def _exact(query, data) -> int | None:  # type: ignore[no-untyped-def]
    real_connect = els.duckdb.connect

    def connect(*a, **kw):  # type: ignore[no-untyped-def]
        con = real_connect(*a, **kw)
        con.execute(f"SET threads TO {DUCKDB_THREADS}")
        return con

    els.duckdb.connect = connect  # type: ignore[assignment]
    try:
        return els.exact_count(query, data, EXACT_TIMEOUT_S)
    finally:
        els.duckdb.connect = real_connect  # type: ignore[assignment]


def evaluate(item: tuple[str, str, dict]) -> dict[str, object]:  # type: ignore[type-arg]
    suite, cond, q = item
    row: dict[str, object] = {
        "suite": suite,
        "condition": cond,
        "query_id": q["query_id"],
        "n_vars": q["n_vars"],
    }
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(TASK_TIMEOUT_S)
    t0 = time.perf_counter()
    try:
        _use_condition(suite, cond)
        built = SUITES[suite][1](q)
        assert built is not None
        query, data = built
        row["min_rows"] = min(len(rows) for _, rows in data.values())
        if row["min_rows"] == 0:  # an empty relation: the join is empty
            row.update(log2_lpbound="-inf", log2_block="-inf", true_count=0)
            row["status"] = "ok"
            return row
        attrs = sorted(query.all_attributes)
        blocks = enumerate_blocks(query, max_block_size=2)
        singles = [b for b in blocks if len(b.relations) == 1]
        stats: dict[tuple[Block, Interface], LpBlockExtensionStats] = dict(
            compute_all_lp_block_extensions(query, singles, data, p_values=P11)
        )
        for b in blocks:
            if len(b.relations) == 2:
                stats.update(
                    compute_sparse_block_extensions(query, b, data, p_values=P11)
                )
        for name, bs in (("lpbound", 1), ("block", 2)):
            cons = block_constraints(
                query, data, max_block_size=bs, p_values=P11, lp_stats=stats
            )
            row[f"log2_{name}"] = f"{polymatroid_bound(attrs, cons):.6f}"
        row["bounds_s"] = f"{time.perf_counter() - t0:.1f}"
        exact = _exact(query, data)
        row["true_count"] = "" if exact is None else exact
        row["status"] = "ok"
    except _TaskTimeoutError:
        row["status"] = f"over {TASK_TIMEOUT_S} s"
    finally:
        signal.alarm(0)
    return row


def summarise(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    def lg(r: dict[str, object], c: str) -> float:
        return float(str(r[f"log2_{c}"]))

    out: list[dict[str, object]] = []
    for suite, cond in conditions():
        for nv in ("6", "8", "10", "all"):
            sel = [
                r
                for r in rows
                if r["suite"] == suite
                and r["condition"] == cond
                and (nv == "all" or str(r["n_vars"]) == nv)
            ]
            ok = [r for r in sel if r["status"] == "ok"]
            if not ok:
                continue
            # an empty join gives 0 on both sides: no ratio
            pos = [r for r in ok if lg(r, "lpbound") > float("-inf")]
            gains = [2 ** (lg(r, "lpbound") - lg(r, "block")) for r in pos]
            chk = [r for r in ok if str(r["true_count"]) not in ("",)]
            nz = [r for r in chk if int(str(r["true_count"])) > 0]
            out.append(
                {
                    "suite": suite,
                    "condition": cond,
                    "n_vars": nv,
                    "n": len(sel),
                    "n_ok": len(ok),
                    "n_empty_relation": len(ok) - len(pos),
                    "median_gain_over_lpbound": (
                        f"{statistics.median(gains):.4g}" if gains else ""
                    ),
                    "n_below_lpbound": sum(
                        lg(r, "block") < lg(r, "lpbound") - 1e-9 for r in pos
                    ),
                    "n_above_lpbound": sum(
                        lg(r, "block") > lg(r, "lpbound") + 1e-6 for r in pos
                    ),
                    "n_checked": len(chk),
                    "n_empty_join": len(chk) - len(nz),
                    "n_invalid": sum(
                        2 ** lg(r, "block") < int(str(r["true_count"])) * (1 - 1e-9)
                        for r in chk
                    ),
                    "median_block_over_truth": (
                        f"{statistics.median(2 ** lg(r, 'block') / int(str(r['true_count'])) for r in nz):.4g}"
                        if nz
                        else ""
                    ),
                    "median_lpbound_over_truth": (
                        f"{statistics.median(2 ** lg(r, 'lpbound') / int(str(r['true_count'])) for r in nz):.4g}"
                        if nz
                        else ""
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
    eis.verify_registration()
    # the strata must be the ones whose hashes are recorded with the results
    expected = {
        (r["stratum"], r["relation"]): r["sha256"]
        for r in csv.DictReader((OUT / "imdb_strata_manifest.csv").open())
    }
    for (stratum, rel), sha in expected.items():
        f = STRATA / stratum / f"{rel}.tsv"
        if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest() != sha:
            raise SystemExit(f"REFUSING TO RUN: {f} missing or changed")
    items = [
        (suite, cond, q)
        for suite, cond in conditions()
        for q in json.loads(SUITES[suite][0].read_text())["queries"]
    ]
    random.Random(2026).shuffle(items)
    with multiprocessing.Pool(processes=WORKERS, maxtasksperchild=1) as pool:
        rows = pool.map(evaluate, items, chunksize=1)
    rows.sort(key=lambda r: (str(r["suite"]), str(r["condition"]), str(r["query_id"])))
    _write("slice_independent.csv", rows)
    summary = summarise(rows)
    _write("slice_independent_summary.csv", summary)
    for s in summary:
        if s["n_vars"] == "all":
            print(s)


if __name__ == "__main__":
    import sys

    if "--summarise" in sys.argv:
        stored = list(csv.DictReader((OUT / "slice_independent.csv").open()))
        _write("slice_independent_summary.csv", summarise(stored))  # type: ignore[arg-type]
    else:
        main()
