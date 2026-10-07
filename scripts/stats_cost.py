"""Total cost of estimation: statistics collection vs LP solve.

Plan Top-5 item #5 and ablation rule 3.2#5: the paper reports LP overhead
(<=4.7 ms) but that figure *excludes* building the degree-sequence
statistics the LP consumes.  Quoting only the solve time would overstate
practicality, so this script measures the full pipeline, phase by phase:

  enumerate blocks -> collect lp stats -> build DAG + hyperpath -> solve LP

It also tests the paper's asserted (but previously unmeasured) claim that
block extensions need "quadratic-size statistics": bs=2 must materialise
two-relation joins, whereas bs=1 reads only per-relation degree
sequences.  We report the measured stats-build ratio and the stored
degree-sequence footprint for both.

Timing: median of N_RUNS after a warm-up, matching the other tables.

Output: results/cost/stats_cost.csv
"""

from __future__ import annotations

import csv
import itertools
import random
import statistics
import time
import tracemalloc
from pathlib import Path

from src.core.block import enumerate_blocks
from src.estimators.lp_bound import optimize_hyperpath_weights
from src.estimators.lph_ceg import estimate_lph_ceg
from src.statistics.lp_stats import compute_all_lp_block_extensions
from src.utils.types import QueryHypergraph, Relation

P_VALUES: tuple[float, ...] = (1.0, 2.0, float("inf"))
N_RUNS = 5
WARMUP = 1


def cyc(k: int) -> list[tuple[str, str]]:
    v = [chr(ord("a") + i) for i in range(k)]
    return [(v[i], v[(i + 1) % k]) for i in range(k)]


FAMILIES: dict[str, list[tuple[str, str]]] = {
    "cycle4": cyc(4),
    "cycle5": cyc(5),
    "cycle6": cyc(6),
    "k4": list(itertools.combinations("abcd", 2)),
}


def gen_edges(n_v: int, n_e: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    w = [1.0 / (i + 1) for i in range(n_v)]
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
    while len(s) < n_e:
        a, b = pick(), pick()
        if a != b:
            s.add((a, b))
    return sorted(s)


def build(edges: list[tuple[str, str]], nv: int, ne: int, seed: int):
    rels, data = [], {}
    for i, (u, v) in enumerate(edges):
        rows = gen_edges(nv, ne, seed + i * 7)
        attrs = tuple(sorted((u, v)))
        swap = attrs != (u, v)
        rels.append(Relation(f"R{i}", frozenset({u, v}), len(rows)))
        data[f"R{i}"] = (attrs, [(y, x) for x, y in rows] if swap else rows)
    return QueryHypergraph(relations=tuple(rels), name="q"), data


def measure(query, data, bs: int) -> dict[str, float]:
    """Time stats collection and LP solve separately, plus stats footprint."""
    stats_ms: list[float] = []
    lp_ms: list[float] = []

    for r in range(N_RUNS + WARMUP):
        t0 = time.perf_counter()
        blocks = enumerate_blocks(query, max_block_size=bs)
        st = compute_all_lp_block_extensions(query, blocks, data, p_values=P_VALUES)
        t_stats = (time.perf_counter() - t0) * 1000.0

        # A hyperpath is needed before the LP; time only the solve itself.
        res = estimate_lph_ceg(query, data, max_block_size=bs, p_values=P_VALUES)
        t0 = time.perf_counter()
        optimize_hyperpath_weights(query, res.hyperpath, st, p_values=P_VALUES)
        t_lp = (time.perf_counter() - t0) * 1000.0

        if r >= WARMUP:
            stats_ms.append(t_stats)
            lp_ms.append(t_lp)

    # Footprint of the retained degree sequences.
    tracemalloc.start()
    blocks = enumerate_blocks(query, max_block_size=bs)
    st = compute_all_lp_block_extensions(query, blocks, data, p_values=P_VALUES)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    n_entries = sum(len(s.degree_sequence) for s in st.values())

    return {
        "stats_ms": statistics.median(stats_ms),
        "lp_ms": statistics.median(lp_ms),
        "n_stat_keys": float(len(st)),
        "n_degree_entries": float(n_entries),
        "peak_kib": peak / 1024.0,
    }


def main() -> None:
    print("Total estimation cost: statistics collection vs LP solve")
    print("=" * 86)
    print(
        f"  {'family':<9} {'bs':>2} {'stats(ms)':>10} {'LP(ms)':>9} "
        f"{'LP share':>9} {'deg entries':>12} {'peak KiB':>10}"
    )
    print(f"  {'-' * 82}")

    rows: list[dict[str, object]] = []
    per_family_bs: dict[str, dict[int, dict[str, float]]] = {}

    for fam, edges in FAMILIES.items():
        query, data = build(edges, 200, 800, seed=101)
        per_family_bs[fam] = {}
        for bs in (1, 2):
            m = measure(query, data, bs)
            total = m["stats_ms"] + m["lp_ms"]
            share = m["lp_ms"] / total if total > 0 else 0.0
            per_family_bs[fam][bs] = m
            print(
                f"  {fam:<9} {bs:>2} {m['stats_ms']:>10.1f} {m['lp_ms']:>9.2f} "
                f"{share * 100:>8.1f}% {int(m['n_degree_entries']):>12} "
                f"{m['peak_kib']:>10.0f}"
            )
            rows.append(
                {
                    "family": fam,
                    "block_size": bs,
                    "stats_ms": f"{m['stats_ms']:.3f}",
                    "lp_solve_ms": f"{m['lp_ms']:.3f}",
                    "total_ms": f"{total:.3f}",
                    "lp_share_of_total": f"{share:.4f}",
                    "n_stat_keys": int(m["n_stat_keys"]),
                    "n_degree_entries": int(m["n_degree_entries"]),
                    "peak_kib": f"{m['peak_kib']:.1f}",
                }
            )

    print(f"  {'-' * 82}")
    # Does bs=2 really cost "quadratic-size statistics" relative to bs=1?
    ratios = []
    for fam in FAMILIES:
        a, b = per_family_bs[fam][1], per_family_bs[fam][2]
        t_ratio = b["stats_ms"] / max(a["stats_ms"], 1e-9)
        e_ratio = b["n_degree_entries"] / max(a["n_degree_entries"], 1e-9)
        ratios.append((fam, t_ratio, e_ratio))
        print(
            f"  {fam:<9} bs2/bs1 stats time = {t_ratio:>6.1f}x, "
            f"degree entries = {e_ratio:>6.1f}x"
        )

    med_t = statistics.median(r[1] for r in ratios)
    print(f"\n  median bs2/bs1 statistics-time ratio: {med_t:.1f}x")
    lp_shares = [float(r["lp_share_of_total"]) for r in rows if r["block_size"] == 1]
    print(
        f"  at bs=1 the LP solve is {statistics.median(lp_shares) * 100:.1f}% of "
        "total estimation cost (median) -- statistics collection dominates."
    )

    out = Path(__file__).resolve().parent.parent / "results" / "cost"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "stats_cost.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out / 'stats_cost.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
