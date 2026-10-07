"""Extend the headline cycle table along the statistics axis.

The paper's central claim is that which statistics you materialise moves
the bound further than which program combines them.  Its headline table,
however, reports only bs=1 -- single-relation degree sequences -- which is
the setting the claim says is the weaker of the two.  A reviewer would
reasonably ask why the flagship table sits on the secondary axis.

This measures the same protocol as `tab:synthetic` (cycle-k, |V|=200,
|E|=800, 10 seeds, P={1,2,inf}) with the statistics axis added: max-degree
and cover LP at bs=1, and the cover LP at bs=2.

Reported per k: the improvement each axis buys over the max-degree
baseline, so the two are directly comparable in the same units.

Output: results/revision/headline_bs2.csv
"""

from __future__ import annotations

import csv
import random
import statistics
import time
from pathlib import Path

from src.estimators.lph_ceg import estimate_lph_ceg
from src.estimators.pessimistic import estimate_pessimistic
from src.utils.types import QueryHypergraph, Relation

ROOT = Path(__file__).resolve().parent.parent
P_VALUES: tuple[float, ...] = (1.0, 2.0, float("inf"))
KS = (3, 4, 5, 6, 7, 8)
N_V, N_E = 200, 800
SEEDS = tuple(range(10))


def gen_edges(n_v: int, n_e: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    w = [1.0 / (i + 1) for i in range(n_v)]
    tot = sum(w)
    pr = [x / tot for x in w]

    def pick() -> str:
        r = rng.random()
        c = 0.0
        for i, q in enumerate(pr):
            c += q
            if r <= c:
                return str(i)
        return str(n_v - 1)

    s: set[tuple[str, str]] = set()
    while len(s) < n_e:
        a, b = pick(), pick()
        if a != b:
            s.add((a, b))
    return sorted(s)


def build_cycle(k: int, seed: int):
    rels, data = [], {}
    for i in range(k):
        u, v = f"v{i}", f"v{(i + 1) % k}"
        rows = gen_edges(N_V, N_E, seed * 1000 + i * 7)
        attrs = tuple(sorted((u, v)))
        swap = attrs != (u, v)
        rels.append(Relation(f"R{i}", frozenset({u, v}), len(rows)))
        data[f"R{i}"] = (attrs, [(y, x) for x, y in rows] if swap else rows)
    return QueryHypergraph(relations=tuple(rels), name=f"cycle{k}"), data


def main() -> None:
    print("Headline cycle table, with the statistics axis added")
    print("=" * 84)
    print(
        f"  {'k':>2} {'CEG_M':>12} {'LP bs=1':>12} {'LP bs=2':>12} "
        f"{'infer gain':>11} {'stats gain':>11} {'bs2 ms':>9}"
    )
    print(f"  {'-' * 80}")

    rows_out: list[dict[str, object]] = []
    for k in KS:
        cegm, lp1, lp2, t2 = [], [], [], []
        for seed in SEEDS:
            q, d = build_cycle(k, seed)
            cegm.append(estimate_pessimistic(q, d, max_block_size=1).estimate)
            lp1.append(
                estimate_lph_ceg(q, d, max_block_size=1, p_values=P_VALUES).estimate
            )
            t0 = time.perf_counter()
            lp2.append(
                estimate_lph_ceg(q, d, max_block_size=2, p_values=P_VALUES).estimate
            )
            t2.append((time.perf_counter() - t0) * 1000.0)
        m_c, m_1, m_2 = (
            statistics.mean(cegm),
            statistics.mean(lp1),
            statistics.mean(lp2),
        )
        infer = m_c / max(m_1, 1e-9)  # what the LP buys at fixed statistics
        stats = m_1 / max(m_2, 1e-9)  # what richer statistics buy on top
        print(
            f"  {k:>2} {m_c:>12.3e} {m_1:>12.3e} {m_2:>12.3e} "
            f"{infer:>10.1f}x {stats:>10.2f}x {statistics.median(t2):>9.0f}"
        )
        rows_out.append(
            {
                "k": k,
                "n_seeds": len(SEEDS),
                "cegm_mean": f"{m_c:.6e}",
                "lp_bs1_mean": f"{m_1:.6e}",
                "lp_bs2_mean": f"{m_2:.6e}",
                "lp_bs2_std": f"{statistics.pstdev(lp2):.6e}",
                "inference_gain": f"{infer:.4f}",
                "statistics_gain": f"{stats:.4f}",
                "total_gain": f"{m_c / max(m_2, 1e-9):.4f}",
                "bs2_median_ms": f"{statistics.median(t2):.2f}",
            }
        )

    print(f"  {'-' * 80}")
    sg = [float(r["statistics_gain"]) for r in rows_out]
    print(
        f"  statistics axis buys {min(sg):.2f}-{max(sg):.2f}x on top of the LP "
        f"(median {statistics.median(sg):.2f}x)"
    )
    print(
        "  Note: at this scale the two axes are complementary -- the LP gain is\n"
        "  large because CEG_M is weak, and the statistics gain sits on top of it."
    )

    out = ROOT / "results" / "revision"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "headline_bs2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nWrote {out / 'headline_bs2.csv'}")


if __name__ == "__main__":
    main()
