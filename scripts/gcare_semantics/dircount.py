"""Directed labelled homomorphism count (same rule as scripts/gcare_suite.py) for
every query (cyclic or not) in the given G-CARE query directories.  Read-only.

usage: python dircount.py <out.csv> yago/Graph_3 yago/Petal_9 ...
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import csv
import os
import sys
import threading

import duckdb

ROOT = str(ROOT / "data" / "gcare") + "/"
TIMEOUT = 120.0


def parse(path):
    verts, edges = {}, []
    for line in open(path):
        t = line.split()
        if t and t[0] == "v":
            verts[int(t[1])] = (int(t[2]), int(t[3]))
        elif t and t[0] == "e":
            edges.append((int(t[1]), int(t[2]), int(t[3])))
    return verts, edges


def rel_sql(ds, verts, a, b, label):
    joins, where = "", [f"e.label = {label}"]
    (la, _), (lb, _) = verts[a], verts[b]
    if la != -1:
        joins += f" JOIN {ds}_vlabel va ON va.vid = e.src AND va.label = {la}"
    if lb != -1:
        joins += f" JOIN {ds}_vlabel vb ON vb.vid = e.dst AND vb.label = {lb}"
    return (
        f"SELECT DISTINCT e.src AS x, e.dst AS y FROM {ds}_edges e{joins} WHERE "
        + " AND ".join(where)
    )


def count(con, ds, verts, edges):
    names, conds, owner = [], [], {}
    for i, (a, b, lab) in enumerate(edges):
        n = f"R{i}"
        con.execute(
            f"CREATE OR REPLACE TEMP TABLE {n} AS {rel_sql(ds, verts, a, b, lab)}"
        )
        names.append(n)
        for col, var in (("x", a), ("y", b)):
            if var in owner:
                conds.append(f"{n}.{col} = {owner[var]}")
            else:
                owner[var] = f"{n}.{col}"
    sql = f"SELECT count(*) FROM {', '.join(names)}" + (
        " WHERE " + " AND ".join(conds) if conds else ""
    )
    out = [None]

    def go():
        try:
            out[0] = int(con.execute(sql).fetchone()[0])
        except Exception as exc:  # interrupt or OOM
            out[0] = f"err:{type(exc).__name__}"

    th = threading.Thread(target=go, daemon=True)
    th.start()
    th.join(TIMEOUT)
    if th.is_alive():
        con.interrupt()
        th.join()
        return "timeout"
    return out[0]


def main():
    out_csv = sys.argv[1]
    con = duckdb.connect(ROOT + "gcare.duckdb", read_only=True)
    con.execute("SET threads TO 4")
    con.execute("SET memory_limit = '16GB'")
    rows = []
    for d in sys.argv[2:]:
        ds = d.split("/")[0]
        for f in sorted(os.listdir(ROOT + "queries/" + d)):
            p = f"{d}/{f}"
            verts, edges = parse(ROOT + "queries/" + p)
            c = count(con, ds, verts, edges)
            tp = ROOT + "truth/" + p
            truth = open(tp).read().split()[0] if os.path.exists(tp) else ""
            rows.append(
                {
                    "path": p,
                    "count": c,
                    "gcare_true": truth,
                    "n_v": len(verts),
                    "n_e": len(edges),
                }
            )
            print(rows[-1], flush=True)
            with open(out_csv, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)


if __name__ == "__main__":
    main()
