"""LP scalability analysis: cycle size and p-value count vs solve time."""

from __future__ import annotations

import csv
import random
import statistics
import time
from pathlib import Path

from src.estimators.lph_ceg import estimate_lph_ceg
from src.estimators.pessimistic import estimate_pessimistic
from src.utils.query import cycle_query

# Timing is wall-clock and noisy; report the median of this many runs.
N_TIMING_RUNS = 7


def gen_edges(n_v: int, n_e: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    weights = [1.0 / (i + 1) for i in range(n_v)]
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
    while len(tuples) < n_e:
        u, v = pick(), pick()
        if u != v:
            tuples.add((u, v))
    return sorted(tuples)


def main() -> None:
    n_v, n_e = 100, 400

    print(f"Part 1: Cycle size vs LP solve time (median of {N_TIMING_RUNS} runs)")
    print("=" * 75)
    header = (
        f"{'k':>3}  {'#rels':>5}  {'#attrs':>6}  "
        f"{'H-CEG(ms)':>10}  {'LpH(ms)':>10}  "
        f"{'overhead':>9}  {'improvement':>12}"
    )
    print(header)
    print("-" * 75)

    rows_out: list[dict[str, object]] = []
    for k in [3, 4, 5, 6, 7, 8, 10]:
        q = cycle_query(k)
        data: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]] = {}
        for i, rel in enumerate(q.relations):
            attrs = tuple(sorted(rel.attributes))
            data[rel.name] = (attrs, gen_edges(n_v, n_e, 42 + i * 7))

        # Median of N timed runs (single-run wall-clock is noisy); the
        # improvement ratio itself is deterministic.
        std_ms: list[float] = []
        lph_ms: list[float] = []
        std = estimate_pessimistic(q, data, max_block_size=1)
        lph = estimate_lph_ceg(
            q, data, max_block_size=1, p_values=(1.0, 2.0, float("inf"))
        )
        for _ in range(N_TIMING_RUNS):
            t0 = time.perf_counter()
            estimate_pessimistic(q, data, max_block_size=1)
            std_ms.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            estimate_lph_ceg(
                q, data, max_block_size=1, p_values=(1.0, 2.0, float("inf"))
            )
            lph_ms.append((time.perf_counter() - t0) * 1000)
        t_std = statistics.median(std_ms)
        t_lph = statistics.median(lph_ms)

        ratio = std.estimate / max(lph.estimate, 1e-10)
        overhead = t_lph / max(t_std, 0.01)

        print(
            f"{k:>3}  {len(q.relations):>5}  {len(q.all_attributes):>6}  "
            f"{t_std:>10.1f}  {t_lph:>10.1f}  "
            f"{overhead:>8.1f}x  {ratio:>11.1f}x"
        )
        rows_out.append(
            {
                "k": k,
                "n_rels": len(q.relations),
                "n_attrs": len(q.all_attributes),
                "hceg_ms": f"{t_std:.4f}",
                "lph_ms": f"{t_lph:.4f}",
                "overhead": f"{overhead:.4f}",
                "improvement": f"{ratio:.4f}",
            }
        )

    out_path = (
        Path(__file__).resolve().parent.parent
        / "results"
        / "scalability"
        / "lp_scalability.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "k",
                "n_rels",
                "n_attrs",
                "hceg_ms",
                "lph_ms",
                "overhead",
                "improvement",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\n(wrote {out_path})")

    print()
    print("Part 2: Number of p-values vs solve time (cycle-6)")
    print("=" * 75)
    header2 = f"{'#p':>4}  {'p-values':>30}  {'LpH(ms)':>10}  {'improvement':>12}"
    print(header2)
    print("-" * 60)

    q6 = cycle_query(6)
    data6: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]] = {}
    for i, rel in enumerate(q6.relations):
        attrs = tuple(sorted(rel.attributes))
        data6[rel.name] = (attrs, gen_edges(n_v, n_e, 42 + i * 7))

    std6 = estimate_pessimistic(q6, data6, max_block_size=1)

    for pvals in [
        (float("inf"),),
        (1.0, float("inf")),
        (1.0, 2.0, float("inf")),
        (1.0, 2.0, 3.0, float("inf")),
        (1.0, 1.5, 2.0, 3.0, 5.0, float("inf")),
    ]:
        t0 = time.perf_counter()
        lph = estimate_lph_ceg(q6, data6, max_block_size=1, p_values=pvals)
        t_lph = (time.perf_counter() - t0) * 1000
        ratio = std6.estimate / max(lph.estimate, 1e-10)
        pstr = ",".join("inf" if p == float("inf") else str(p) for p in pvals)
        print(f"{len(pvals):>4}  {pstr:>30}  {t_lph:>10.1f}  {ratio:>11.1f}x")


if __name__ == "__main__":
    main()
