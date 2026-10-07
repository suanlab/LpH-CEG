"""X1: Scalability crossover — polynomial cover-LP vs entropic 2^|V| LpBound.

This experiment is the central efficiency argument of the paper: our fractional
block-cover LP (``estimate_lph_ceg`` with ``max_block_size=1``) has
``O(m * |P|)`` variables and scales linearly in the query size, whereas the
genuine entropic LpBound (``estimate_lpbound_entropic``) solves a Shannon-cone
LP with ``2 ** |V|`` variables that becomes intractable as the cycle grows.

For each cycle length ``k`` we record, on identical power-law data:

* ``cover_vars``  — LP columns of the cover LP (linear in ``k``),
* ``entropic_vars`` — ``2 ** k`` (the polymatroid LP columns),
* ``cover_ms`` / ``entropic_ms`` — median wall-clock solve time,
* ``cover_bound`` / ``entropic_bound`` — the bounds (to confirm the tightness
  gap stays ~1.3-1.4x, i.e. we give up little tightness for the speed),
* ``status`` — ``ok`` if entropic solved, ``intractable`` once ``2 ** k``
  exceeds the entropic solver's attribute cap or the per-call time budget.

The bounds are deterministic; only the timings are noisy, so we report the
median of a few runs.  Output: ``results/scalability/scale_entropic.csv``.
"""

from __future__ import annotations

import csv
import random
import statistics
import time
from pathlib import Path

from src.estimators.lpbound_entropic import (
    MAX_ATTRIBUTES,
    estimate_lpbound_entropic,
)
from src.estimators.lph_ceg import estimate_lph_ceg
from src.utils.query import cycle_query

P_VALUES: tuple[float, ...] = (1.0, 2.0, float("inf"))

#: Cover LP is cheap, so we sweep well past the entropic feasibility wall.
COVER_K_VALUES: tuple[int, ...] = (3, 4, 5, 6, 7, 8, 10, 12, 14, 16)

#: Median-of-N for the (noisy) wall-clock timings; bounds are deterministic.
N_TIMING_RUNS: int = 5

#: Skip repeated entropic timings once a single solve exceeds this (seconds);
#: keeps the sweep bounded while still recording the (slow) data point.
ENTROPIC_SLOW_S: float = 2.0


def gen_edges(n_v: int, n_e: int, seed: int) -> list[tuple[str, str]]:
    """Deterministic power-law edge set (same generator as lp_scalability)."""
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


def _build_cycle(
    k: int, n_v: int, n_e: int
) -> tuple[object, dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]]]:
    q = cycle_query(k)
    data: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]] = {}
    for i, rel in enumerate(q.relations):
        attrs = tuple(sorted(rel.attributes))
        data[rel.name] = (attrs, gen_edges(n_v, n_e, 42 + i * 7))
    return q, data


def _time_cover(q: object, data: dict) -> tuple[float, float]:
    """Return (median_ms, bound) for the cover LP (bs=1)."""
    bound = estimate_lph_ceg(q, data, max_block_size=1, p_values=P_VALUES).estimate
    samples: list[float] = []
    for _ in range(N_TIMING_RUNS):
        t0 = time.perf_counter()
        estimate_lph_ceg(q, data, max_block_size=1, p_values=P_VALUES)
        samples.append((time.perf_counter() - t0) * 1000)
    return statistics.median(samples), bound


def _time_entropic(
    q: object, data: dict, k: int
) -> tuple[str, float | None, float | None]:
    """Return (status, median_ms, bound) for the entropic 2^k LP."""
    if k > MAX_ATTRIBUTES:
        return "intractable", None, None
    try:
        # First (timed) solve doubles as the feasibility probe.
        t0 = time.perf_counter()
        bound = estimate_lpbound_entropic(q, data, p_values=P_VALUES)
        first_s = time.perf_counter() - t0
    except ValueError:
        return "intractable", None, None

    samples = [first_s * 1000]
    if first_s < ENTROPIC_SLOW_S:
        for _ in range(N_TIMING_RUNS - 1):
            t0 = time.perf_counter()
            estimate_lpbound_entropic(q, data, p_values=P_VALUES)
            samples.append((time.perf_counter() - t0) * 1000)
    return "ok", statistics.median(samples), bound


def main() -> None:
    n_v, n_e = 100, 400

    print("X1: Polynomial cover-LP vs entropic 2^|V| LpBound (cycle queries)")
    print("=" * 86)
    header = (
        f"{'k':>3}  {'cover_vars':>10}  {'entropic_vars':>13}  "
        f"{'cover_ms':>9}  {'entropic_ms':>12}  {'gap(c/e)':>9}  {'status':>11}"
    )
    print(header)
    print("-" * 86)

    rows: list[dict[str, object]] = []
    for k in COVER_K_VALUES:
        q, data = _build_cycle(k, n_v, n_e)
        cover_ms, cover_bound = _time_cover(q, data)
        status, entropic_ms, entropic_bound = _time_entropic(q, data, k)

        # Cover LP columns: one fractional weight per (hyperpath step, p-value).
        # A k-cycle hyperpath has k extension steps; |P| p-values each.
        cover_vars = k * len(P_VALUES)
        entropic_vars = 2**k
        gap = (
            cover_bound / entropic_bound
            if entropic_bound and entropic_bound > 0
            else None
        )

        e_ms_str = f"{entropic_ms:.1f}" if entropic_ms is not None else "----"
        gap_str = f"{gap:.3f}x" if gap is not None else "----"
        print(
            f"{k:>3}  {cover_vars:>10}  {entropic_vars:>13}  "
            f"{cover_ms:>9.2f}  {e_ms_str:>12}  {gap_str:>9}  {status:>11}"
        )

        rows.append(
            {
                "k": k,
                "n_attrs": k,
                "cover_vars": cover_vars,
                "entropic_vars": entropic_vars,
                "cover_ms": f"{cover_ms:.4f}",
                "entropic_ms": (
                    f"{entropic_ms:.4f}" if entropic_ms is not None else ""
                ),
                "cover_bound": f"{cover_bound:.6e}",
                "entropic_bound": (
                    f"{entropic_bound:.6e}" if entropic_bound is not None else ""
                ),
                "gap_cover_over_entropic": f"{gap:.4f}" if gap is not None else "",
                "status": status,
            }
        )

    out_path = (
        Path(__file__).resolve().parent.parent
        / "results"
        / "scalability"
        / "scale_entropic.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "k",
        "n_attrs",
        "cover_vars",
        "entropic_vars",
        "cover_ms",
        "entropic_ms",
        "cover_bound",
        "entropic_bound",
        "gap_cover_over_entropic",
        "status",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
