"""C2: predict the cover/entropic tightness gap from query structure and skew.

The paper currently *observes* that the cover-LP's looseness over the
entropic optimum grows with skew (1.3x synthetic -> 8.8x extreme real) but
offers no predictive model. This experiment runs a factorial study over
query shape, size, and degree-skew, and fits a simple model of the log gap
from cheap structural features. Success bar (pre-registered in TOP_PLAN):
held-out log-gap R^2 >= 0.6 OR Spearman rho >= 0.7 between a skew feature
and the gap. If met, C2 becomes a characterization; if not, we keep it as
an observation.

Features (all computable before solving the entropic LP):
  n_mid   -- number of non-boundary (middle) extension edges (~ query size)
  l2linf  -- max over relations of ||deg||_inf / ||deg||_2-normalised skew
  maxskew -- max over relations of maxdeg / meandeg
  ntri    -- number of triangles in the query graph (cyclic-intermediate density)

Output: results/characterization/skew_gap.csv
"""

from __future__ import annotations

import csv
import itertools
import math
import random
from collections import Counter
from pathlib import Path

from src.estimators.lpbound_entropic import MAX_ATTRIBUTES, estimate_lpbound_entropic
from src.estimators.lph_ceg import estimate_lph_ceg
from src.utils.types import QueryHypergraph, Relation

P_VALUES: tuple[float, ...] = (1.0, 2.0, float("inf"))
RelData = dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]]


def cycle_edges(k: int) -> list[tuple[str, str]]:
    v = [f"v{i}" for i in range(k)]
    return [(v[i], v[(i + 1) % k]) for i in range(k)]


def clique_edges(k: int) -> list[tuple[str, str]]:
    v = [f"v{i}" for i in range(k)]
    return list(itertools.combinations(v, 2))


def chordal_cycle_edges(k: int) -> list[tuple[str, str]]:
    e = cycle_edges(k)
    e.append(("v0", "v2"))  # one chord -> a triangle v0-v1-v2
    return e


SHAPES = {
    "cycle": cycle_edges,
    "clique": clique_edges,
    "chordal": chordal_cycle_edges,
}


def gen_edges_alpha(
    n_v: int, n_e: int, alpha: float, seed: int
) -> list[tuple[str, str]]:
    """Power-law edges with tunable skew exponent alpha (larger = more skew)."""
    rng = random.Random(seed)
    weights = [1.0 / (i + 1) ** alpha for i in range(n_v)]
    total = sum(weights)
    probs = [w / total for w in weights]

    def pick() -> str:
        r = rng.random()
        c = 0.0
        for i, p in enumerate(probs):
            c += p
            if r <= c:
                return str(i)
        return str(n_v - 1)

    tuples: set[tuple[str, str]] = set()
    guard = 0
    while len(tuples) < n_e and guard < n_e * 50:
        u, v = pick(), pick()
        guard += 1
        if u != v:
            tuples.add((u, v))
    return sorted(tuples)


def build(
    edges: list[tuple[str, str]], n_v: int, n_e: int, alpha: float, seed: int
) -> tuple[QueryHypergraph, RelData]:
    rels: list[Relation] = []
    data: RelData = {}
    for i, (u, v) in enumerate(edges):
        rows = gen_edges_alpha(n_v, n_e, alpha, seed + i * 101)
        attrs = tuple(sorted((u, v)))
        swapped = attrs != (u, v)
        rels.append(Relation(f"R{i}", frozenset({u, v}), len(rows)))
        data[f"R{i}"] = (attrs, [(y, x) for x, y in rows] if swapped else rows)
    return QueryHypergraph(relations=tuple(rels), name="q"), data


def rel_skew(rows: list[tuple[str, str]]) -> float:
    """max/mean out-degree over the first attribute (>=1)."""
    deg = Counter(r[0] for r in rows)
    if not deg:
        return 1.0
    vals = list(deg.values())
    return max(vals) / (sum(vals) / len(vals))


def n_triangles(edges: list[tuple[str, str]]) -> int:
    adj: dict[str, set[str]] = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    verts = list(adj)
    tri = 0
    for a, b, c in itertools.combinations(verts, 3):
        if b in adj[a] and c in adj[a] and c in adj[b]:
            tri += 1
    return tri


def ols_r2(x_rows: list[list[float]], y: list[float]) -> float:
    """In-house OLS with intercept; returns R^2. No external deps."""
    import numpy as np

    x = np.array([[1.0, *r] for r in x_rows])
    yv = np.array(y)
    beta, *_ = np.linalg.lstsq(x, yv, rcond=None)
    pred = x @ beta
    ss_res = float(((yv - pred) ** 2).sum())
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(z: list[float]) -> list[float]:
        order = sorted(range(len(z)), key=lambda i: z[i])
        rk = [0.0] * len(z)
        for pos, idx in enumerate(order):
            rk[idx] = float(pos)
        return rk

    ra, rb = ranks(a), ranks(b)
    n = len(a)
    mean_a = sum(ra) / n
    mean_b = sum(rb) / n
    cov = sum((ra[i] - mean_a) * (rb[i] - mean_b) for i in range(n))
    va = math.sqrt(sum((r - mean_a) ** 2 for r in ra))
    vb = math.sqrt(sum((r - mean_b) ** 2 for r in rb))
    return cov / (va * vb) if va > 0 and vb > 0 else 0.0


def main() -> None:
    print("C2: skew/structure -> tightness-gap characterization")
    print("=" * 74)

    n_v, n_e = 60, 220
    seeds = (1, 2, 3)
    configs: list[tuple[str, int, float, int]] = []
    for shape in SHAPES:
        for k in (3, 4, 5, 6):
            if shape == "clique" and k > 5:
                continue  # |V|=k for clique; keep entropic tractable
            for alpha in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
                for s in seeds:
                    configs.append((shape, k, alpha, s))

    rows: list[dict[str, object]] = []
    feats: list[list[float]] = []
    log_gap: list[float] = []
    maxskews: list[float] = []
    for shape, k, alpha, s in configs:
        edges = SHAPES[shape](k)
        nv = len({x for e in edges for x in e})
        if nv > MAX_ATTRIBUTES:
            continue
        q, data = build(edges, n_v, n_e, alpha, seed=1000 * s + k)
        cover = estimate_lph_ceg(q, data, max_block_size=1, p_values=P_VALUES).estimate
        try:
            ent = estimate_lpbound_entropic(q, data, p_values=P_VALUES)
        except ValueError:
            continue
        if ent <= 0 or cover <= 0:
            continue
        gap = cover / ent
        maxskew = max(rel_skew(data[r.name][1]) for r in q.relations)
        meanskew = sum(rel_skew(data[r.name][1]) for r in q.relations) / len(
            q.relations
        )
        ntri = n_triangles(edges)
        n_mid = len(edges)
        feat = [float(n_mid), math.log(maxskew), math.log(meanskew), float(ntri)]
        feats.append(feat)
        log_gap.append(math.log(gap))
        maxskews.append(maxskew)
        rows.append(
            {
                "shape": shape,
                "k": k,
                "n_vars": nv,
                "alpha": alpha,
                "seed": s,
                "n_edges": n_mid,
                "n_triangles": ntri,
                "max_skew": f"{maxskew:.4f}",
                "mean_skew": f"{meanskew:.4f}",
                "cover": f"{cover:.6e}",
                "entropic": f"{ent:.6e}",
                "gap": f"{gap:.4f}",
                "log_gap": f"{math.log(gap):.4f}",
            }
        )

    # Fit predictor + train/test split for an honest held-out R^2.
    r2_full = ols_r2(feats, log_gap)
    rng = random.Random(0)
    idx = list(range(len(feats)))
    rng.shuffle(idx)
    cut = int(0.7 * len(idx))
    tr, te = idx[:cut], idx[cut:]
    import numpy as np

    xt = np.array([[1.0, *feats[i]] for i in tr])
    beta, *_ = np.linalg.lstsq(xt, np.array([log_gap[i] for i in tr]), rcond=None)
    xe = np.array([[1.0, *feats[i]] for i in te])
    pred = xe @ beta
    ye = np.array([log_gap[i] for i in te])
    ss_res = float(((ye - pred) ** 2).sum())
    ss_tot = float(((ye - ye.mean()) ** 2).sum())
    r2_heldout = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rho = spearman(maxskews, [math.exp(g) for g in log_gap])

    print(f"  configs evaluated: {len(rows)}")
    print(
        f"  gap range: {min(math.exp(g) for g in log_gap):.2f}x "
        f"-- {max(math.exp(g) for g in log_gap):.2f}x"
    )
    print(f"  in-sample R^2 (log-gap ~ structure+skew): {r2_full:.3f}")
    print(f"  held-out R^2 (70/30 split):               {r2_heldout:.3f}")
    print(f"  Spearman rho(max_skew, gap):              {rho:.3f}")
    bar = r2_heldout >= 0.6 or rho >= 0.7
    print(f"  SUCCESS BAR (R^2>=0.6 or rho>=0.7): {'MET' if bar else 'NOT MET'}")

    out = Path(__file__).resolve().parent.parent / "results" / "characterization"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "skew_gap.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out / "skew_gap_fit.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["n_configs", len(rows)])
        w.writerow(["r2_in_sample", f"{r2_full:.4f}"])
        w.writerow(["r2_heldout", f"{r2_heldout:.4f}"])
        w.writerow(["spearman_maxskew_gap", f"{rho:.4f}"])
        w.writerow(["success_bar_met", int(bar)])
    print(f"\nWrote {out / 'skew_gap.csv'} ({len(rows)} rows) + skew_gap_fit.csv")


if __name__ == "__main__":
    main()
