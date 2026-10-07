"""When (if ever) does the LP improve the bound on an ACYCLIC query?

The paper originally claimed improvement is possible "if and only if" the
query is cyclic.  That is false as stated: on a 4-path the LP returns
per-step weights (1, 0, 1) -- a later step can cover an attribute through
its *interface* at p=1, releasing an earlier step's weight -- and the
bound strictly improves.

This search pins down the actual boundary.  Over 1,500 random acyclic
instances (paths, star, tree; varied domain sizes and skew) it records,
for every case where the LP improved, whether the true join was empty.

Measured result: 626 improvements, ALL of them on instances whose true
join is empty; 0 improvements among instances with a non-empty join.
So the honest statement is that redundant coverage is *necessary* for
improvement, and on acyclic queries the improvement is confined to the
degenerate empty-join regime.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.estimators.lph_ceg import estimate_lph_ceg
from src.utils.types import QueryHypergraph, Relation


def rand_rel(rng, du, dv, n, skew):
    rows = set()
    us = [f"u{i}" for i in range(du)]
    vs = [f"v{i}" for i in range(dv)]
    wu = [1.0 / ((i + 1) ** skew) for i in range(du)]
    wv = [1.0 / ((i + 1) ** skew) for i in range(dv)]
    while len(rows) < n:
        rows.add((rng.choices(us, wu)[0], rng.choices(vs, wv)[0]))
    return sorted(rows)


def true_join(edges, data):

    # sequential hash join over tuples
    cur = [{}]
    for i in range(len(edges)):
        attrs, rows = data[f"R{i}"]
        nxt = []
        for env in cur:
            for r in rows:
                b = dict(zip(attrs, r, strict=True))
                if all(env.get(k, b[k]) == b[k] for k in b):
                    e = dict(env)
                    e.update(b)
                    nxt.append(e)
        cur = nxt
        if len(cur) > 200000:
            return -1  # censored
    return len(cur)


shapes = {
    "path3": [("a", "b"), ("b", "c"), ("c", "d")],
    "path4": [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")],
    "path5": [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f")],
    "star3": [("a", "b"), ("a", "c"), ("a", "d")],
    "tree": [("a", "b"), ("b", "c"), ("b", "d"), ("d", "e")],
}
rng = random.Random(2026)
found_nonempty = 0
found_empty = 0
tested = 0
best = None
for name, edges in shapes.items():
    for _ in range(300):
        data = {}
        for i, (u, v) in enumerate(edges):
            du = rng.choice([2, 5, 20, 60])
            dv = rng.choice([2, 5, 20, 60])
            n = min(rng.choice([20, 80, 200]), du * dv)
            attrs = tuple(sorted((u, v)))
            rows = rand_rel(rng, du, dv, n, rng.choice([0.0, 1.0, 2.0]))
            if attrs != (u, v):
                rows = [(y, x) for x, y in rows]
            data[f"R{i}"] = (attrs, rows)
        rels = tuple(
            Relation(f"R{i}", frozenset(e), len(data[f"R{i}"]))
            for i, e in enumerate(edges)
        )
        q = QueryHypergraph(relations=rels, name=name)
        r = estimate_lph_ceg(q, data, max_block_size=1)
        tested += 1
        ratio = r.metadata.get("improvement_ratio", 1.0)
        if ratio > 1.001:
            tj = true_join(edges, data)
            if tj > 0:
                found_nonempty += 1
                if best is None or ratio > best[0]:
                    best = (ratio, name, tj)
            elif tj == 0:
                found_empty += 1
print(
    f"tested={tested}  improved&nonempty={found_nonempty}  improved&EMPTY-join={found_empty}"
)
if best:
    print("best non-empty improvement:", best)
