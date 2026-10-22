"""Block statistics under predicates: is a predicate-free catalog still useful?

Block statistics are collected without predicates (Section VII).  Under a
selection they remain valid upper bounds -- filtering a relation can only
remove rows of a pair join, so every degree, and every l_p norm, can only
shrink -- but they are looser.  This measures how much looser, on the
IMDb |V|=6 cycles with a production-year predicate on every movie variable
(two predicates of different selectivity over the 200 films):

  lpbound   entropic bound on single-relation statistics of the FILTERED
            relations (exact, as if LpBound had per-predicate histograms)
  catalog   polymatroid bound on the filtered single-relation statistics
            plus the UNFILTERED pair-block statistics (a predicate-free
            block catalog, as a system would hold it)
  oracle    polymatroid bound on block statistics of the filtered
            relations (one block catalog per predicate)

Eleven norms on every side.  Each filtered query is counted exactly by DuckDB
(same SQL as the suites), so every bound with a count is checked.

Output: results/imdb/predicate_blocks.csv and predicate_blocks_summary.csv
"""

from __future__ import annotations

import csv
import json
import multiprocessing
import os
import statistics
from pathlib import Path

from scripts.evaluate_imdb_suite import build_imdb_query
from scripts.evaluate_long_suite import exact_count
from scripts.pset_fair import P11, SUITES
from src.core.block import enumerate_blocks
from src.estimators.flow_bound import block_constraints, polymatroid_bound
from src.statistics.lp_stats import compute_all_lp_block_extensions

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "imdb"
# JOB's IMDb CSVs, as in prepare_imdb_cycles.py; override with LPH_JOB_DIR.
JOB_DIR = Path(os.environ.get("LPH_JOB_DIR", str(ROOT / "data" / "job_full")))
N_VARS = 6
WORKERS = 8
EXACT_TIMEOUT_S = 60.0
# name -> predicate on production_year; selectivity over the 200 films is
# reported in the summary
PREDICATES = {
    "year>=2005": lambda y: y >= 2005,
    "year<1995": lambda y: y < 1995,
}


def _years() -> dict[str, int]:
    movies: set[int] = set()
    for f in (ROOT / "data" / "imdb_cycles").glob("*.tsv"):
        with open(f) as fh:
            next(fh)
            for line in fh:
                for x in line.rstrip("\n").split("\t"):
                    if x.startswith("M"):
                        movies.add(int(x[1:]))
    years: dict[str, int] = {}
    with open(
        JOB_DIR / "title.csv", newline="", encoding="utf-8", errors="replace"
    ) as fh:
        for row in csv.reader(fh):
            if row and row[0].isdigit() and int(row[0]) in movies and row[4]:
                years[f"M{row[0]}"] = int(row[4])
    return years


YEARS = _years()


def _filter(data: dict, keep) -> dict:  # type: ignore[no-untyped-def,type-arg]
    """Drop rows whose movie endpoint fails the predicate."""
    out = {}
    for name, (attrs, rows) in data.items():
        out[name] = (
            attrs,
            [r for r in rows if all(keep(YEARS[x]) for x in r if x.startswith("M"))],
        )
    return out


def evaluate(item: tuple[str, dict]) -> dict[str, object]:  # type: ignore[type-arg]
    pname, q = item
    query, data_u = build_imdb_query(q)
    data_f = _filter(data_u, PREDICATES[pname])
    attrs = sorted(query.all_attributes)
    blocks = enumerate_blocks(query, max_block_size=2)
    pairs = [b for b in blocks if len(b.relations) == 2]
    st_f = compute_all_lp_block_extensions(query, blocks, data_f, p_values=P11)
    st_u = compute_all_lp_block_extensions(query, pairs, data_u, p_values=P11)
    mixed = {k: v for k, v in st_f.items() if len(k[0].relations) == 1}
    mixed.update(st_u)

    def poly(bs: int, stats: dict) -> float:  # type: ignore[type-arg]
        cons = block_constraints(
            query, data_f, max_block_size=bs, p_values=P11, lp_stats=stats
        )
        return 2.0 ** polymatroid_bound(attrs, cons)

    lpbound = poly(1, st_f)
    catalog = poly(2, mixed)
    oracle = poly(2, st_f)
    truth = exact_count(query, data_f, EXACT_TIMEOUT_S)
    return {
        "predicate": pname,
        "query_id": q["query_id"],
        "rows_filtered_over_unfiltered": f"{sum(len(r) for _, r in data_f.values()) / max(1, sum(len(r) for _, r in data_u.values())):.4f}",
        "lpbound": f"{lpbound:.6e}",
        "catalog": f"{catalog:.6e}",
        "oracle": f"{oracle:.6e}",
        "true_count": "" if truth is None else truth,
    }


def summarise(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    def f(r: dict[str, object], c: str) -> float:
        return float(str(r[c]))

    out: list[dict[str, object]] = []
    for pname, keep in PREDICATES.items():
        rs = [r for r in rows if r["predicate"] == pname]
        nz = [r for r in rs if f(r, "lpbound") > 0]
        chk = [r for r in rs if str(r["true_count"]) not in ("", "0")]
        out.append(
            {
                "predicate": pname,
                "films_kept": sum(1 for y in YEARS.values() if keep(y)),
                "n": len(rs),
                "n_empty": sum(str(r["true_count"]) == "0" for r in rs),
                "median_catalog_over_lpbound": f"{statistics.median(f(r, 'catalog') / f(r, 'lpbound') for r in nz):.4g}",
                "median_oracle_over_lpbound": f"{statistics.median(f(r, 'oracle') / f(r, 'lpbound') for r in nz):.4g}",
                "median_catalog_over_oracle": f"{statistics.median(f(r, 'catalog') / max(f(r, 'oracle'), 1e-300) for r in nz if f(r, 'oracle') > 0):.4g}",
                "n_catalog_below_lpbound": sum(
                    f(r, "catalog") < f(r, "lpbound") for r in nz
                ),
                "n_checked": len(chk),
                "n_invalid": sum(
                    f(r, b) < f(r, "true_count") * (1 - 1e-9)
                    for r in chk
                    for b in ("lpbound", "catalog", "oracle")
                ),
            }
        )
    return out


def _write(name: str, rows: list[dict[str, object]]) -> None:
    with open(OUT / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    queries = [
        q
        for q in json.loads(SUITES["imdb"][0].read_text())["queries"]
        if q["n_vars"] == N_VARS
    ]
    items = [(p, q) for p in PREDICATES for q in queries]
    with multiprocessing.Pool(processes=WORKERS, maxtasksperchild=1) as pool:
        rows = pool.map(evaluate, items, chunksize=1)
    rows.sort(key=lambda r: (str(r["predicate"]), str(r["query_id"])))
    _write("predicate_blocks.csv", rows)
    summary = summarise(rows)
    _write("predicate_blocks_summary.csv", summary)
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
