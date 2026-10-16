"""Recount the G-CARE YAGO queries whose published count differs from ours.

``gcare_suite.py`` counts directed, labelled homomorphisms with DuckDB over
``gcare.duckdb``.  On 49 of the 192 YAGO queries it evaluates, the count
differs from G-CARE's published true cardinality.  To make the validity
check not depend on that one counter, this recounts those queries from
G-CARE's raw text file (``yago/yago.txt``; no DuckDB): relations are the
distinct (src, dst) pairs of an edge label, filtered by vertex labels and
bound data vertices, and the count is a sum-product by variable
elimination over pandas factors (min-width order).

Output: results/gcare/gcare_recount.csv
"""

from __future__ import annotations

import csv
import resource
import sys

import pandas as pd

from scripts.gcare_suite import GCARE, OUT, parse

MEMORY_LIMIT_BYTES = 40 * 2**30


def _factors(verts, edges, e, vl):  # type: ignore[no-untyped-def]
    out = []
    for a, b, lab in edges:
        r = e[e.l == lab][["s", "d"]].drop_duplicates()
        (la, da), (lb, db) = verts[a], verts[b]
        if la != -1:
            r = r[r.s.isin(vl[la])]
        if lb != -1:
            r = r[r.d.isin(vl[lb])]
        if da != -1:
            r = r[r.s == da]
        if db != -1:
            r = r[r.d == db]
        if a == b:
            r = r[r.s == r.d][["s"]].rename(columns={"s": f"v{a}"})
        else:
            r = r.rename(columns={"s": f"v{a}", "d": f"v{b}"})
        out.append(r.assign(w=1))
    return out


def count(verts, edges, e, vl) -> int:  # type: ignore[no-untyped-def]
    """Number of homomorphisms, by variable elimination."""
    factors = _factors(verts, edges, e, vl)
    remaining = {f"v{v}" for v in verts}

    def cost(v: str) -> tuple[int, int]:
        touch = [f for f in factors if v in f.columns]
        width = len({c for f in touch for c in f.columns} - {v, "w"})
        return width, sum(len(f) for f in touch)

    while remaining:
        v = min(sorted(remaining), key=cost)
        touch = [f for f in factors if v in f.columns]
        factors = [f for f in factors if v not in f.columns]
        acc = touch[0]
        for f in touch[1:]:
            on = [c for c in acc.columns if c in f.columns and c != "w"]
            acc = acc.merge(f, on=on)
            acc["w"] = acc.pop("w_x") * acc.pop("w_y")
        keep = [c for c in acc.columns if c not in (v, "w")]
        acc = (
            acc.groupby(keep, as_index=False)["w"].sum()
            if keep
            else pd.DataFrame({"w": [acc.w.sum()]})
        )
        factors.append(acc)
        remaining.discard(v)
    total = 1
    for f in factors:
        total *= int(f.w.sum())
    return total


def main() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    ev = pd.read_csv(OUT / "gcare_eval.csv")
    y = ev[(ev.dataset == "yago") & ev.cover_bs1_p11.notna()]
    y = y[y.exact.notna()]
    todo = y if "--all" in sys.argv else y[y.gcare_true != y.exact]
    queries = {r.path: parse(GCARE / "queries" / r.path) for r in todo.itertuples()}
    need_el = {lab for _, es in queries.values() for *_, lab in es}
    need_vl = {lab for vs, _ in queries.values() for lab, _ in vs.values() if lab != -1}
    src, dst, labs = [], [], []
    vl: dict[int, set[int]] = {lab: set() for lab in need_vl}
    with open(GCARE / "yago" / "yago.txt") as fh:
        for line in fh:
            if line[0] == "e":
                _, s, d, lab = line.split()
                if int(lab) in need_el:
                    src.append(int(s))
                    dst.append(int(d))
                    labs.append(int(lab))
            elif line[0] == "v" and need_vl:
                t = line.split()
                for lab in t[2:]:
                    if int(lab) in vl:
                        vl[int(lab)].add(int(t[1]))
    e = pd.DataFrame({"s": src, "d": dst, "l": labs})
    rows: list[dict[str, object]] = []
    for r in todo.itertuples():
        try:
            c: object = count(*queries[r.path], e, vl)
        except MemoryError:
            c = "memory_limit"
        rows.append(
            {
                "path": r.path,
                "petal": "/Petal_" in r.path,
                "duckdb_exact": int(r.exact),
                "recount": c,
                "gcare_true": int(r.gcare_true),
            }
        )
        print(rows[-1], flush=True)
        with open(OUT / "gcare_recount.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    agree = sum(str(r["recount"]) == str(r["duckdb_exact"]) for r in rows)
    print(f"recount == duckdb on {agree}/{len(rows)}")


if __name__ == "__main__":
    main()
