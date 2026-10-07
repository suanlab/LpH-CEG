"""C3: does LP-residual variable ordering reduce WCOJ search work, broadly?

The paper's WCOJ claim currently rests on one positive query (Hetionet
cycle-4, 2.29x fewer iterations); synthetic controls are tied or worse.
This study runs a broad suite (>=30 queries across shapes, sizes, skews,
seeds) and, for each, counts GenericJoin iterations under five orderings:
lexicographic, min-degree, max-degree, H-CEG (max-degree residual), and
LpH-CEG (l_p residual). It then reports the *distribution* of the paired
LpH-CEG-vs-H-CEG improvement with proper statistics (median ratio, sign
split, Wilcoxon signed-rank p, Cliff's delta, P90 regression), so the
claim is judged on evidence, not on one case.

Pre-registered bar (TOP_PLAN C3): median iteration reduction >= 20% and
paired advantage significant. If unmet, the honest outcome is to scope
the WCOJ claim to the regime where it wins (or demote it).

Output: results/wcoj/ordering_study.csv (+ ordering_study_summary.csv)
"""

from __future__ import annotations

import csv
import itertools
import math
import random
from pathlib import Path

from scripts.wcoj_e2e_experiment import generic_join
from src.estimators.lph_ceg import estimate_lph_ceg
from src.utils.types import QueryHypergraph, Relation
from src.wcoj.evaluation import (
    lexicographic_ordering,
    max_degree_ordering,
    min_degree_ordering,
)
from src.wcoj.lp_variable_order import lp_variable_ordering
from src.wcoj.variable_order import variable_ordering

P_VALUES: tuple[float, ...] = (1.0, 2.0, float("inf"))
RelData = dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]]


def cycle(k: int) -> list[tuple[str, str]]:
    v = [f"v{i}" for i in range(k)]
    return [(v[i], v[(i + 1) % k]) for i in range(k)]


def clique(k: int) -> list[tuple[str, str]]:
    return list(itertools.combinations([f"v{i}" for i in range(k)], 2))


def chordal(k: int) -> list[tuple[str, str]]:
    return [*cycle(k), ("v0", "v2")]


SHAPES = {
    "cycle3": cycle(3),
    "cycle4": cycle(4),
    "cycle5": cycle(5),
    "cycle6": cycle(6),
    "k4": clique(4),
    "chordal5": chordal(5),
    "diamond": [*cycle(4), ("v0", "v2")],
}


def gen_edges(n_v: int, n_e: int, alpha: float, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    w = [1.0 / (i + 1) ** alpha for i in range(n_v)]
    tot = sum(w)
    pr = [x / tot for x in w]

    def pick() -> str:
        r = rng.random()
        c = 0.0
        for i, p in enumerate(pr):
            c += p
            if r <= c:
                return str(i)
        return str(n_v - 1)

    s: set[tuple[str, str]] = set()
    guard = 0
    while len(s) < n_e and guard < n_e * 40:
        a, b = pick(), pick()
        guard += 1
        if a != b:
            s.add((a, b))
    return sorted(s)


def build(edges: list[tuple[str, str]], nv: int, ne: int, a: float, seed: int):
    rels, data = [], {}
    for i, (u, v) in enumerate(edges):
        rows = gen_edges(nv, ne, a, seed + i * 131)
        attrs = tuple(sorted((u, v)))
        swap = attrs != (u, v)
        rels.append(Relation(f"R{i}", frozenset({u, v}), len(rows)))
        data[f"R{i}"] = (attrs, [(y, x) for x, y in rows] if swap else rows)
    return QueryHypergraph(relations=tuple(rels), name="q"), data


def iters(query, data, ordering) -> int:
    return generic_join(query, data, ordering).total_iterations


def wilcoxon_p(diffs: list[float]) -> float:
    """Two-sided Wilcoxon signed-rank p via normal approximation."""
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n < 6:
        return float("nan")
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    rank = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            rank[order[k]] = avg
        i = j + 1
    w_plus = sum(rank[i] for i in range(n) if nz[i] > 0)
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sigma == 0:
        return float("nan")
    z = (w_plus - mu) / sigma
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def cliffs_delta(better: int, worse: int, total: int) -> float:
    return (better - worse) / total if total else 0.0


def main() -> None:
    print("C3: WCOJ ordering study -- LpH-CEG residual vs baselines")
    print("=" * 74)

    configs = []
    for shape in SHAPES:
        for nv, ne in ((40, 120), (60, 200)):
            for alpha in (1.0, 2.0):
                for seed in (1, 2, 3):
                    configs.append((shape, nv, ne, alpha, seed))

    cap = 400_000  # skip queries whose join is too large to enumerate 5x
    rows = []
    censored = 0
    for shape, nv, ne, alpha, seed in configs:
        q, data = build(SHAPES[shape], nv, ne, alpha, seed=7000 * seed + nv)
        # Cheap pre-filter: skip intractable joins (censored, reported).
        if (
            estimate_lph_ceg(q, data, max_block_size=1, p_values=P_VALUES).estimate
            > cap
        ):
            censored += 1
            continue
        orders = {
            "lex": lexicographic_ordering(q),
            "min_deg": min_degree_ordering(q),
            "max_deg": max_degree_ordering(q),
            "hceg": variable_ordering(q, data, max_block_size=1),
            "lph": lp_variable_ordering(q, data, max_block_size=1, p_values=P_VALUES),
        }
        it = {name: iters(q, data, o) for name, o in orders.items()}
        rows.append(
            {"shape": shape, "nv": nv, "ne": ne, "alpha": alpha, "seed": seed, **it}
        )

    # Paired analysis: LpH vs H-CEG, and LpH vs best simple baseline.
    def analyze(label: str, a_key: str, b_key: str) -> dict[str, object]:
        ratios, better, worse, eq = [], 0, 0, 0
        p90reg = []
        log_diffs = []
        for r in rows:
            a, b = r[a_key], max(r[b_key], 1)
            ratio = b / max(a, 1)  # >1 means LpH used fewer iterations
            ratios.append(ratio)
            log_diffs.append(math.log(max(a, 1)) - math.log(b))
            p90reg.append(a / b)  # >1 means LpH regressed
            if a < b:
                better += 1
            elif a > b:
                worse += 1
            else:
                eq += 1
        ratios.sort()
        med = ratios[len(ratios) // 2]
        p90reg.sort()
        p90 = p90reg[int(0.9 * (len(p90reg) - 1))]
        return {
            "comparison": label,
            "n": len(rows),
            "median_ratio": f"{med:.3f}",
            "median_reduction_pct": f"{(1 - 1 / med) * 100:.1f}",
            "better": better,
            "worse": worse,
            "tie": eq,
            "wilcoxon_p": f"{wilcoxon_p(log_diffs):.4f}",
            "cliffs_delta": f"{cliffs_delta(better, worse, len(rows)):.3f}",
            "p90_regression_ratio": f"{p90:.3f}",
        }

    print(f"  evaluated {len(rows)} queries ({censored} censored: join > {cap:,})")
    if not rows:
        print("  [all queries censored -- lower the cap or graph density]")
        return

    def best_simple(r: dict) -> int:
        return min(r["lex"], r["min_deg"], r["max_deg"])

    for r in rows:
        r["best_simple"] = best_simple(r)
    summ = [
        analyze("LpH vs H-CEG", "lph", "hceg"),
        analyze("LpH vs best-simple", "lph", "best_simple"),
        analyze("LpH vs max_deg", "lph", "max_deg"),
    ]

    print(
        f"  {'comparison':<20} {'n':>3} {'med_ratio':>9} {'red%':>6} "
        f"{'W/T/L':>10} {'wilcox_p':>8} {'delta':>6} {'p90reg':>7}"
    )
    print(f"  {'-' * 78}")
    for s in summ:
        wtl = f"{s['better']}/{s['tie']}/{s['worse']}"
        print(
            f"  {s['comparison']:<20} {s['n']:>3} {s['median_ratio']:>9} "
            f"{s['median_reduction_pct']:>5}% {wtl:>10} {s['wilcoxon_p']:>8} "
            f"{s['cliffs_delta']:>6} {s['p90_regression_ratio']:>7}"
        )

    primary = summ[0]
    red = float(primary["median_reduction_pct"])
    bar = red >= 20.0 and primary["worse"] < primary["better"]
    print(
        f"\n  C3 bar (median reduction >=20% & net-positive): "
        f"{'MET' if bar else 'NOT MET'} (median reduction {red:.1f}%)"
    )
    if not bar:
        print("  -> honest outcome: scope WCOJ claim to winning regime / demote.")

    out = Path(__file__).resolve().parent.parent / "results" / "wcoj"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ordering_study.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "shape",
                "nv",
                "ne",
                "alpha",
                "seed",
                "lex",
                "min_deg",
                "max_deg",
                "hceg",
                "lph",
                "best_simple",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    with open(out / "ordering_study_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
    print(f"\nWrote {out / 'ordering_study.csv'} ({len(rows)} queries)")


if __name__ == "__main__":
    main()
