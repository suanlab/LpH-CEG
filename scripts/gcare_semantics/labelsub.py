from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
"""For one query, replace one edge's label by each other label (0..90); report hits
equal to G-CARE's truth.  DuckDB, same rule as gcare_suite.  Read-only."""
import sys
import time

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dircount
from dircount import ROOT, count, parse

dircount.TIMEOUT = 10.0
p = sys.argv[1]
verts, edges = parse(ROOT + "queries/" + p)
truth = int(open(ROOT + "truth/" + p).read().split()[0])
con = duckdb.connect(ROOT + "gcare.duckdb", read_only=True)
con.execute("SET threads TO 4")
labs = [
    r[0]
    for r in con.execute("select distinct label from yago_edges order by 1").fetchall()
]
t0 = time.time()
hits = []
n = 0
for i, (a, b, l) in enumerate(edges):
    for l2 in labs:
        if l2 == l:
            continue
        es = list(edges)
        es[i] = (a, b, l2)
        c = count(con, "yago", verts, es)
        n += 1
        if c == truth:
            hits.append((i, l, l2))
            print("hit", i, l, "->", l2, flush=True)
print(
    p,
    "truth",
    truth,
    "variants",
    n,
    "hits",
    hits,
    f"{time.time() - t0:.0f}s",
    flush=True,
)
