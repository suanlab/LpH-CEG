"""Large-scale 4-way comparison: CEG vs LpBound vs H-CEG vs LpH-CEG.

Runs cycle-k queries on synthetic power-law graphs at multiple scales
(100 to 10K vertices) and compares all four estimators:

  1. CEG_M     (bs=1, no LP)  — standard binary max-degree
  2. LpBound   (bs=1, LP)     — binary extensions + LP optimization
  3. H-CEG_M   (bs=2, no LP)  — multi-way blocks, max-degree
  4. LpH-CEG   (bs=1, LP)     — binary extensions + LP (same as LpBound)
  5. LpH-CEG   (bs=2, LP)     — multi-way blocks + LP (full framework)

Also measures: LP overhead, degree skew sensitivity, p-value sensitivity.
"""

from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.estimators.lpbound_baseline import estimate_lpbound
from src.estimators.lph_ceg import estimate_lph_ceg
from src.estimators.pessimistic import estimate_pessimistic
from src.utils.query import cycle_query, star_query

if TYPE_CHECKING:
    from src.utils.types import QueryHypergraph

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_power_law_relation(
    name: str,
    attrs: tuple[str, ...],
    n_vertices: int,
    n_tuples: int,
    *,
    alpha: float = 1.0,
    seed: int = 42,
) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
    """Generate a relation with power-law (Zipf) value distribution.

    Args:
        alpha: Power-law exponent. Higher = more skew.
    """
    rng = random.Random(seed)
    weights = [1.0 / (i + 1) ** alpha for i in range(n_vertices)]
    total = sum(weights)
    cumulative = []
    acc = 0.0
    for w in weights:
        acc += w / total
        cumulative.append(acc)

    def pick() -> str:
        r = rng.random()
        lo, hi = 0, len(cumulative) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cumulative[mid] < r:
                lo = mid + 1
            else:
                hi = mid
        return str(lo)

    tuples: set[tuple[str, ...]] = set()
    attempts = 0
    max_attempts = n_tuples * 20
    while len(tuples) < n_tuples and attempts < max_attempts:
        row = tuple(pick() for _ in attrs)
        if len(set(row)) == len(row):  # no self-loops
            tuples.add(row)
        attempts += 1

    return attrs, sorted(tuples)


def make_cycle_data(
    k: int,
    n_vertices: int,
    n_edges: int,
    *,
    alpha: float = 1.0,
    seed: int = 42,
) -> tuple[QueryHypergraph, dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]]]:
    """Create a cycle-k query with synthetic power-law data."""
    q = cycle_query(k)
    data: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]] = {}
    for i, rel in enumerate(q.relations):
        attrs = tuple(sorted(rel.attributes))
        data[rel.name] = generate_power_law_relation(
            rel.name,
            attrs,
            n_vertices,
            n_edges,
            alpha=alpha,
            seed=seed + i * 7,
        )
    return q, data


def make_star_data(
    n_leaves: int,
    n_vertices: int,
    n_edges: int,
    *,
    alpha: float = 1.0,
    seed: int = 42,
) -> tuple[QueryHypergraph, dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]]]:
    """Create a star query with synthetic power-law data."""
    q = star_query(n_leaves)
    data: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]] = {}
    for i, rel in enumerate(q.relations):
        attrs = tuple(sorted(rel.attributes))
        data[rel.name] = generate_power_law_relation(
            rel.name,
            attrs,
            n_vertices,
            n_edges,
            alpha=alpha,
            seed=seed + i * 7,
        )
    return q, data


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


@dataclass
class Result:
    """Single experiment result row."""

    experiment: str
    query: str
    n_vertices: int
    n_edges: int
    method: str
    bound: float
    time_ms: float
    ratio_vs_ceg: float
    metadata: dict[str, object]


def run_estimator(
    name: str,
    fn: object,
    query: QueryHypergraph,
    data: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]],
    **kwargs: object,
) -> tuple[float, float, dict[str, object]]:
    """Run an estimator and return (bound, time_ms, metadata)."""
    t0 = time.perf_counter()
    result = fn(query, data, **kwargs)  # type: ignore[operator]
    elapsed = (time.perf_counter() - t0) * 1000
    meta = dict(result.metadata) if result.metadata else {}
    return result.estimate, elapsed, meta


P_VALUES_DEFAULT: tuple[float, ...] = (1.0, 2.0, float("inf"))


def run_4way_comparison(
    experiment: str,
    query: QueryHypergraph,
    data: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]],
    n_vertices: int,
    n_edges: int,
    *,
    include_bs2: bool = True,
) -> list[Result]:
    """Run 4-way (or 5-way) comparison on one query."""
    results: list[Result] = []

    # 1. CEG_M (bs=1, no LP)
    b_ceg, t_ceg, m_ceg = run_estimator(
        "CEG_M",
        estimate_pessimistic,
        query,
        data,
        max_block_size=1,
    )

    # 2. LpBound (bs=1, LP)
    b_lpb, t_lpb, m_lpb = run_estimator(
        "LpBound",
        estimate_lpbound,
        query,
        data,
        p_values=P_VALUES_DEFAULT,
    )

    # 3. H-CEG_M (bs=2, no LP) — only if include_bs2
    b_hceg, t_hceg, m_hceg = (b_ceg, 0.0, {})
    if include_bs2:
        b_hceg, t_hceg, m_hceg = run_estimator(
            "H-CEG_M",
            estimate_pessimistic,
            query,
            data,
            max_block_size=2,
        )

    # 4. LpH-CEG (bs=1, LP) — same as LpBound but tagged differently
    # (already computed as LpBound above, skip)

    # 5. LpH-CEG (bs=2, LP) — only if include_bs2
    b_lph2, t_lph2, m_lph2 = (b_lpb, 0.0, {})
    if include_bs2:
        b_lph2, t_lph2, m_lph2 = run_estimator(
            "LpH-CEG",
            estimate_lph_ceg,
            query,
            data,
            max_block_size=2,
            p_values=P_VALUES_DEFAULT,
        )

    qname = query.name or "?"

    results.append(
        Result(
            experiment,
            qname,
            n_vertices,
            n_edges,
            "CEG_M (bs=1)",
            b_ceg,
            t_ceg,
            1.0,
            m_ceg,
        )
    )
    results.append(
        Result(
            experiment,
            qname,
            n_vertices,
            n_edges,
            "Cover-LP(bs1)",
            b_lpb,
            t_lpb,
            b_ceg / max(b_lpb, 1e-10),
            m_lpb,
        )
    )
    if include_bs2:
        results.append(
            Result(
                experiment,
                qname,
                n_vertices,
                n_edges,
                "H-CEG_M (bs=2)",
                b_hceg,
                t_hceg,
                b_ceg / max(b_hceg, 1e-10),
                m_hceg,
            )
        )
        results.append(
            Result(
                experiment,
                qname,
                n_vertices,
                n_edges,
                "LpH-CEG (bs=2,LP)",
                b_lph2,
                t_lph2,
                b_ceg / max(b_lph2, 1e-10),
                m_lph2,
            )
        )

    return results


def print_results(results: list[Result]) -> None:
    """Print results as a formatted table."""
    print(f"\n{'Method':<25} {'Bound':>15} {'Time':>8} {'vs CEG':>10}")
    print("-" * 62)
    for r in results:
        print(
            f"  {r.method:<23} {r.bound:>15.0f} {r.time_ms:>6.1f}ms {r.ratio_vs_ceg:>9.2f}x"
        )


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def experiment_cycle_scalability() -> list[Result]:
    """Exp 1: Cycle-k for k=3..8, measuring improvement vs cycle size."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Cycle Scalability (k=3..8, 200V, 800E)")
    print("=" * 70)

    all_results: list[Result] = []
    n_v, n_e = 200, 800

    for k in [3, 4, 5, 6, 7, 8]:
        q, data = make_cycle_data(k, n_v, n_e, seed=42)
        print(f"\n  Cycle-{k} ({n_v}V, {n_e}E):")
        results = run_4way_comparison(
            "cycle_scale", q, data, n_v, n_e, include_bs2=(k <= 6)
        )
        print_results(results)
        all_results.extend(results)

    return all_results


def experiment_data_scalability() -> list[Result]:
    """Exp 2: Fixed cycle-4, varying data size (100 to 5K vertices)."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Data Scalability (cycle-4, 100V to 5KV)")
    print("=" * 70)

    all_results: list[Result] = []

    for n_v, n_e in [
        (100, 400),
        (200, 800),
        (500, 2000),
        (1000, 4000),
        (2000, 8000),
        (5000, 20000),
    ]:
        q, data = make_cycle_data(4, n_v, n_e, seed=42)
        actual_edges = {r: len(data[r][1]) for r in data}
        print(f"\n  Cycle-4 ({n_v}V, target {n_e}E, actual {actual_edges}):")
        # Skip bs=2 for large data (block join is expensive)
        include_bs2 = n_v <= 1000
        results = run_4way_comparison(
            "data_scale", q, data, n_v, n_e, include_bs2=include_bs2
        )
        print_results(results)
        all_results.extend(results)

    return all_results


def experiment_skew_sensitivity() -> list[Result]:
    """Exp 3: Fixed cycle-4 200V, varying power-law exponent alpha."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Degree Skew Sensitivity (cycle-4, alpha=0.5..3.0)")
    print("=" * 70)

    all_results: list[Result] = []
    n_v, n_e = 200, 800

    for alpha in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        q, data = make_cycle_data(4, n_v, n_e, alpha=alpha, seed=42)
        print(f"\n  Cycle-4 ({n_v}V, {n_e}E, alpha={alpha}):")
        results = run_4way_comparison(f"skew_alpha={alpha}", q, data, n_v, n_e)
        print_results(results)
        all_results.extend(results)

    return all_results


def experiment_acyclic_control() -> list[Result]:
    """Exp 4: Acyclic queries (star, path) — LP should give 1.00x."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Acyclic Control (star/path — expect 1.00x LP)")
    print("=" * 70)

    all_results: list[Result] = []
    n_v, n_e = 200, 800

    # Star-3
    q_star, data_star = make_star_data(3, n_v, n_e, seed=42)
    print(f"\n  Star-3 ({n_v}V, {n_e}E):")
    results = run_4way_comparison("acyclic", q_star, data_star, n_v, n_e)
    print_results(results)
    all_results.extend(results)

    # Star-5
    q_star5, data_star5 = make_star_data(5, n_v, n_e, seed=42)
    print(f"\n  Star-5 ({n_v}V, {n_e}E):")
    results = run_4way_comparison("acyclic", q_star5, data_star5, n_v, n_e)
    print_results(results)
    all_results.extend(results)

    return all_results


def experiment_block_lp_interaction() -> list[Result]:
    """Exp 5: Block size + LP interaction on cycle-4."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Block Size + LP Interaction (cycle-4, 200V)")
    print("=" * 70)

    n_v, n_e = 200, 800
    q, data = make_cycle_data(4, n_v, n_e, seed=42)

    all_results: list[Result] = []

    configs = [
        ("CEG_M (bs=1)", estimate_pessimistic, {"max_block_size": 1}),
        ("Cover-LP(bs1)", estimate_lpbound, {"p_values": P_VALUES_DEFAULT}),
        ("H-CEG_M (bs=2)", estimate_pessimistic, {"max_block_size": 2}),
        (
            "LpH-CEG (bs=2,LP)",
            estimate_lph_ceg,
            {"max_block_size": 2, "p_values": P_VALUES_DEFAULT},
        ),
        ("H-CEG_M (bs=3)", estimate_pessimistic, {"max_block_size": 3}),
        (
            "LpH-CEG (bs=3,LP)",
            estimate_lph_ceg,
            {"max_block_size": 3, "p_values": P_VALUES_DEFAULT},
        ),
    ]

    print(f"\n  Cycle-4 ({n_v}V, {n_e}E) — varying block size:")
    ceg_bound = None
    for name, fn, kwargs in configs:
        bound, t_ms, meta = run_estimator(name, fn, q, data, **kwargs)
        if ceg_bound is None:
            ceg_bound = bound
        ratio = ceg_bound / max(bound, 1e-10)
        r = Result("block_lp", q.name or "?", n_v, n_e, name, bound, t_ms, ratio, meta)
        all_results.append(r)

    print_results(all_results)
    return all_results


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def export_csv(results: list[Result], path: Path) -> None:
    """Export results to CSV for plotting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "experiment",
                "query",
                "n_vertices",
                "n_edges",
                "method",
                "bound",
                "time_ms",
                "ratio_vs_ceg",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.experiment,
                    r.query,
                    r.n_vertices,
                    r.n_edges,
                    r.method,
                    f"{r.bound:.2f}",
                    f"{r.time_ms:.2f}",
                    f"{r.ratio_vs_ceg:.4f}",
                ]
            )
    print(f"\nResults exported to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("LpH-CEG Large-Scale 4-Way Comparison")
    print("CEG_M vs LpBound vs H-CEG_M vs LpH-CEG")
    print("=" * 70)

    all_results: list[Result] = []

    all_results.extend(experiment_cycle_scalability())
    all_results.extend(experiment_data_scalability())
    all_results.extend(experiment_skew_sensitivity())
    all_results.extend(experiment_acyclic_control())
    all_results.extend(experiment_block_lp_interaction())

    # Export
    export_csv(all_results, Path("results/large_scale/comparison.csv"))

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nTotal experiments: {len(all_results)}")

    # Best improvements
    lp_results = [r for r in all_results if "LP" in r.method and r.ratio_vs_ceg > 1.01]
    if lp_results:
        best = max(lp_results, key=lambda r: r.ratio_vs_ceg)
        print(
            f"Best LP improvement: {best.ratio_vs_ceg:.1f}x ({best.method} on {best.query})"
        )


if __name__ == "__main__":
    main()
