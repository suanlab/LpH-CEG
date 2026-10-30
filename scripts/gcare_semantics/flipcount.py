"""For one query, count homomorphisms (DuckDB, same rule as gcare_suite) for every
subset of edges reversed; report subsets reproducing G-CARE's truth.  Read-only.

usage: python flipcount.py yago/Petal_9/f_Q_3_4.txt [max_flips]
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import itertools
import sys
import time

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dircount
from dircount import ROOT, count, parse

dircount.TIMEOUT = 20.0


def main():
    p = sys.argv[1]
    kmax = int(sys.argv[2]) if len(sys.argv) > 2 else 99
    verts, edges = parse(ROOT + "queries/" + p)
    truth = int(open(ROOT + "truth/" + p).read().split()[0])
    con = duckdb.connect(ROOT + "gcare.duckdb", read_only=True)
    con.execute("SET threads TO 3")
    res = {}
    t0 = time.time()
    for k in range(0, min(kmax, len(edges)) + 1):
        for flip in itertools.combinations(range(len(edges)), k):
            es = [
                (b, a, l) if i in flip else (a, b, l)
                for i, (a, b, l) in enumerate(edges)
            ]
            c = count(con, "yago", verts, es)
            res[flip] = c
            if c == truth or k == 0:
                print("flip", flip, "count", c, "truth", truth, flush=True)
    hits = [f for f, c in res.items() if c == truth]
    print(
        p,
        "truth",
        truth,
        "base",
        res[()],
        "variants",
        len(res),
        "hits",
        hits,
        f"{time.time() - t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
