"""Recount G-CARE queries under alternative semantics (scratch, read-only on repo).

Variants (flags):
  directed  : query edge (a,b,l) matches data edge a->b with label l (else either dir)
  injective : distinct query vertices map to distinct data vertices
  bag       : parallel duplicate data edges (same s,d,l) counted with multiplicity
  vfirst    : a query vertex label matches only the FIRST label on the data v-line
Count is by backtracking (exact), with a wall-clock timeout per query.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import collections
import sys
import time

import duckdb

DB = str(ROOT / "data" / "gcare" / "gcare.duckdb")
Q = str(ROOT / "data" / "gcare" / "queries") + "/"
T = str(ROOT / "data" / "gcare" / "truth") + "/"


def parse(path):
    verts, edges = {}, []
    for line in open(path):
        t = line.split()
        if t and t[0] == "v":
            verts[int(t[1])] = (int(t[2]), int(t[3]))
        elif t and t[0] == "e":
            edges.append((int(t[1]), int(t[2]), int(t[3])))
    return verts, edges


class Timeout(Exception):
    pass


class Graph:
    def __init__(self, ds, labels, vlabels, con):
        self.ds = ds
        lab = ",".join(map(str, sorted(labels)))
        rows = con.execute(
            f"SELECT src, dst, label, count(*) FROM {ds}_edges WHERE label IN ({lab}) "
            "GROUP BY ALL"
        ).fetchall()
        # out[l][s] -> {d: mult}, inn[l][d] -> {s: mult}
        self.out = collections.defaultdict(lambda: collections.defaultdict(dict))
        self.inn = collections.defaultdict(lambda: collections.defaultdict(dict))
        for s, d, l, m in rows:
            self.out[l][s][d] = m
            self.inn[l][d][s] = m
        self.vl_any, self.vl_first = {}, {}
        for vl in vlabels:
            self.vl_any[vl] = {
                r[0]
                for r in con.execute(
                    f"SELECT vid FROM {ds}_vlabel WHERE label = {vl}"
                ).fetchall()
            }
        self.first = None

    def load_first_labels(self, path, vlabels):
        want = set(vlabels)
        self.vl_first = {vl: set() for vl in want}
        with open(path) as fh:
            for line in fh:
                if line[0] == "v":
                    t = line.split()
                    if len(t) > 2 and int(t[2]) in want:
                        self.vl_first[int(t[2])].add(int(t[1]))


def count(
    g,
    verts,
    edges,
    directed=True,
    injective=False,
    bag=False,
    vfirst=False,
    timeout=120.0,
):
    t0 = time.time()
    vlab = g.vl_first if vfirst else g.vl_any
    # order: start at vertex with most incident edges, then greedy connected
    inc = collections.Counter()
    for a, b, _ in edges:
        inc[a] += 1
        inc[b] += 1
    order = [max(verts, key=lambda v: (inc[v], -v))]
    while len(order) < len(verts):
        cand = [v for v in verts if v not in order]
        conn = lambda v: sum(
            1 for a, b, _ in edges if (a == v and b in order) or (b == v and a in order)
        )
        order.append(max(cand, key=lambda v: (conn(v), inc[v], -v)))
    pos = {v: i for i, v in enumerate(order)}

    def nbrs(l, x, forward):
        """data vertices y with edge x->y (forward) or y->x, label l, with mult."""
        res = collections.Counter()
        if directed or True:
            src = g.out[l] if forward else g.inn[l]
            for y, m in src.get(x, {}).items():
                res[y] += m if bag else 1
        if not directed:
            src = g.inn[l] if forward else g.out[l]
            for y, m in src.get(x, {}).items():
                res[y] += m if bag else 1
            if not bag:
                for y in res:
                    res[y] = 1
        return res

    # candidates for first vertex: endpoints of any incident edge
    v0 = order[0]
    first_edges = [(a, b, l) for a, b, l in edges if v0 in (a, b)]
    a, b, l = first_edges[0]
    cands = set()
    if directed:
        cands = set(g.out[l]) if a == v0 else set(g.inn[l])
    else:
        cands = set(g.out[l]) | set(g.inn[l])
    lab0 = verts[v0][0]
    if lab0 != -1:
        cands &= vlab[lab0]

    # per position: edges to earlier vertices
    back = []
    for i, v in enumerate(order):
        es = []
        for a, b, l in edges:
            if a == v and b == v:
                es.append(("self", l, None))
            elif a == v and pos[b] < i:
                es.append(("in", l, b))  # earlier b, need v->b: v in inn[l][b]
            elif b == v and pos[a] < i:
                es.append(("out", l, a))  # earlier a, need a->v
        back.append(es)

    assign = {}
    used = set()
    total = 0
    steps = [0]

    def weight_self(x, l):
        m = g.out[l].get(x, {}).get(x, 0)
        return m if bag else (1 if m else 0)

    def rec(i, w):
        nonlocal total
        steps[0] += 1
        if steps[0] & 0xFFFF == 0 and time.time() - t0 > timeout:
            raise Timeout
        if i == len(order):
            total += w
            return
        v = order[i]
        es = back[i]
        if i == 0:
            cand = {x: 1 for x in cands}
        else:
            cand = None
            for kind, l, u in es:
                if kind == "self":
                    continue
                xu = assign[u]
                # kind out: a=u earlier, need u->v: forward from u
                nb = nbrs(l, xu, forward=(kind == "out"))
                if cand is None:
                    cand = dict(nb)
                else:
                    cand = {y: cand[y] * nb[y] for y in cand if y in nb}
                if not cand:
                    return
            if cand is None:
                raise RuntimeError("disconnected order")
        lab = verts[v][0]
        for x, m in cand.items():
            if lab != -1 and x not in vlab[lab]:
                continue
            if injective and x in used:
                continue
            ww = m
            for kind, l, _ in es:
                if kind == "self":
                    ww *= weight_self(x, l)
            if i == 0:
                for a2, b2, l2 in edges:
                    pass
            if ww == 0:
                continue
            assign[v] = x
            used.add(x)
            rec(i + 1, w * ww)
            used.discard(x)
            del assign[v]

    rec(0, 1)
    return total


VARIANTS = {
    "base": dict(),
    "injective": dict(injective=True),
    "undirected": dict(directed=False),
    "undir_inj": dict(directed=False, injective=True),
    "bag": dict(bag=True),
    "vfirst": dict(vfirst=True),
}


def main():
    ds = sys.argv[1]
    paths = sys.argv[2:]
    timeout = 120.0
    qs = {p: parse(Q + p) for p in paths}
    labels = {l for _, es in qs.values() for *_, l in es}
    vls = {lab for vs, _ in qs.values() for lab, _ in vs.values() if lab != -1}
    con = duckdb.connect(DB, read_only=True)
    con.execute("SET threads TO 2")
    g = Graph(ds, labels, vls, con)
    if vls:
        g.load_first_labels(str(ROOT / "data" / "gcare" / ds / f"{ds}.txt"), vls)
    con.close()
    print("path," + ",".join(VARIANTS) + ",gcare_true", flush=True)
    for p in paths:
        verts, edges = qs[p]
        out = []
        for name, kw in VARIANTS.items():
            if name == "vfirst" and not vls:
                out.append("same")
                continue
            try:
                out.append(str(count(g, verts, edges, timeout=timeout, **kw)))
            except Timeout:
                out.append("timeout")
        truth = open(T + p).read().split()[0]
        print(p + "," + ",".join(out) + "," + truth, flush=True)


if __name__ == "__main__":
    main()
