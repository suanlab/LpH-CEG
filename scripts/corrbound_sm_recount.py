"""Recount the 400 CorrBound subgraph-matching queries with DuckDB.

The artifact's truecardinality_dblp.csv sits 0.3--1.6% below the counts its
own SQL returns on its own CSVs (checked on three queries), so validity is
checked against a recount.  Each query is counted over the label-filtered
edge relations that corrbound_sm_compare.py builds (the vertex atoms are key
lookups with a label predicate), which returns the artifact SQL's count.

Output: results/corrbound/sm_recount.csv (query, count, artifact_truth)
"""

from __future__ import annotations

import csv
import os
import tempfile
import time

import duckdb
import pandas as pd

import scripts.corrbound_sm_compare as cmp

THREADS = 4


def count(con: duckdb.DuckDBPyConnection, name: str) -> int:
    sql = (
        cmp.CB
        / "benchmarks"
        / "workloads"
        / "subgraph_matching"
        / "dblp"
        / f"{name}.sql"
    ).read_text()
    labels, edges = cmp.parse(sql)
    first: dict[int, str] = {}
    conds: list[str] = []
    for k, (a, b) in enumerate(edges):
        df = pd.DataFrame(
            cmp._EDGES.get((labels[a], labels[b]), []), columns=["s", "t"]
        )
        con.register(f"e{k}", df)
        for v, col in ((a, "s"), (b, "t")):
            if v in first:
                conds.append(f"{first[v]} = e{k}.{col}")
            else:
                first[v] = f"e{k}.{col}"
    q = (
        "SELECT COUNT(*) FROM "
        + ", ".join(f"e{k}" for k in range(len(edges)))
        + " WHERE "
        + " AND ".join(conds)
    )
    n = int(con.execute(q).fetchone()[0])  # type: ignore[index]
    for k in range(len(edges)):
        con.unregister(f"e{k}")
    return n


def main() -> None:
    cmp.OUT.mkdir(parents=True, exist_ok=True)
    cmp._load_dblp()
    truth = cmp._paper("truecardinality")
    tmp = tempfile.mkdtemp(prefix="sm_recount_")
    con = duckdb.connect()
    con.execute(f"SET threads TO {THREADS}")
    con.execute("SET enable_progress_bar = false")
    con.execute(f"SET temp_directory = '{tmp}'")
    names = sorted(
        f[:-4]
        for f in os.listdir(
            cmp.CB / "benchmarks" / "workloads" / "subgraph_matching" / "dblp"
        )
        if f.endswith(".sql") and "_8_" in f
    )
    rows = []
    for name in names:
        t0 = time.perf_counter()
        n = count(con, name)
        rows.append(
            {
                "query": name,
                "count": n,
                "artifact_truth": int(truth[name]),
                "seconds": f"{time.perf_counter() - t0:.2f}",
            }
        )
    with open(cmp.OUT / "sm_recount.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    diff = [r for r in rows if r["count"] != r["artifact_truth"]]
    print(f"{len(rows)} queries; {len(diff)} differ from the artifact's truth")


if __name__ == "__main__":
    main()
