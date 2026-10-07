"""Revision experiments addressing reviewer feedback.

R1: Multi-seed experiments with mean ± std (Reviewers B, E)
R2: Ground truth q-error on real data (Reviewer E)
R3: PANDA baseline (Reviewer C)
R4: Fair comparison at fixed block size (Reviewer E)

All synthetic experiments run across N_SEEDS seeds.
"""

from __future__ import annotations

import csv
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from src.estimators.agm_bound import estimate_agm as estimate_agm_true
from src.estimators.lpbound_baseline import estimate_lpbound
from src.estimators.pessimistic import estimate_pessimistic
from src.utils.query import cycle_query, star_query

if TYPE_CHECKING:
    from src.utils.types import QueryHypergraph

# Human-facing label for the cover-LP proxy (LpH-CEG at bs=1).  This is NOT
# the genuine entropic LpBound (see src/estimators/lpbound_entropic.py), so we
# do not call it "LpBound" in outputs.
COVER_LP_LABEL = "Cover-LP(bs1)"
# Human-facing label for the PANDA proxy, which is just CEG_M (pessimistic, bs=1).
PANDA_PROXY_LABEL = "PANDA-proxy (= CEG_M)"

N_SEEDS = 10
P_VALUES: tuple[float, ...] = (1.0, 2.0, float("inf"))


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
    """Generate a relation with power-law (Zipf) value distribution."""
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
        if len(set(row)) == len(row):
            tuples.add(row)
        attempts += 1

    return attrs, sorted(tuples)


def make_cycle_data(
    k: int,
    n_v: int,
    n_e: int,
    *,
    alpha: float = 1.0,
    seed: int = 42,
) -> tuple[QueryHypergraph, dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]]]:
    """Create cycle-k query with synthetic data."""
    q = cycle_query(k)
    data: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]] = {}
    for i, rel in enumerate(q.relations):
        attrs = tuple(sorted(rel.attributes))
        data[rel.name] = generate_power_law_relation(
            rel.name,
            attrs,
            n_v,
            n_e,
            alpha=alpha,
            seed=seed + i * 7,
        )
    return q, data


# ---------------------------------------------------------------------------
# Baselines: PANDA proxy (= CEG_M) and the true AGM fractional-edge-cover bound
# ---------------------------------------------------------------------------


def estimate_panda_proxy(
    query: QueryHypergraph,
    relation_data: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]],
) -> float:
    """PANDA-proxy bound (= CEG_M, pessimistic with bs=1).

    This is *not* a faithful PANDA implementation: it reuses CEG_M (max-degree,
    binary blocks) as a stand-in.  Outputs label it "PANDA-proxy (= CEG_M)" so
    the proxy nature is explicit.
    """
    result = estimate_pessimistic(query, relation_data, max_block_size=1)
    return result.estimate


def estimate_agm(
    query: QueryHypergraph,
    relation_data: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]],
) -> float:
    """True AGM fractional-edge-cover bound.

    Delegates to :func:`src.estimators.agm_bound.estimate_agm`, which solves
    the fractional-edge-cover LP ``min prod_e |R_e|**x_e`` subject to every
    attribute being covered with total weight >= 1.  For a cycle this yields
    ``prod_e |R_e|**(1/2)``, not the (much looser) plain product of
    cardinalities used previously.
    """
    return estimate_agm_true(query, relation_data)


# ---------------------------------------------------------------------------
# Brute-force join for ground truth
# ---------------------------------------------------------------------------


def count_true_join(
    query: QueryHypergraph,
    data: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]],
    max_intermediate: int = 5_000_000,
) -> int | None:
    """Brute-force join. Returns None if intermediate exceeds cap."""
    rels = list(query.relations)
    if not rels:
        return 0

    first = rels[0]
    attrs_0, tuples_0 = data[first.name]
    current = [dict(zip(attrs_0, t, strict=False)) for t in tuples_0]

    for rel in rels[1:]:
        attrs_r, tuples_r = data[rel.name]
        if not current:
            return 0
        current_attrs = set(current[0].keys())
        shared = sorted(current_attrs & set(attrs_r))

        idx: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
        for t in tuples_r:
            d = dict(zip(attrs_r, t, strict=False))
            key = tuple(d[a] for a in shared)
            idx[key].append(d)

        joined: list[dict[str, str]] = []
        for ct in current:
            key = tuple(ct[a] for a in shared)
            for nt in idx.get(key, []):
                joined.append({**ct, **nt})
            if len(joined) > max_intermediate:
                return None
        current = joined

    return len(current)


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------


@dataclass
class MultiSeedResult:
    """Aggregated results across seeds."""

    experiment: str
    query: str
    n_vertices: int
    n_edges: int
    methods: dict[str, list[float]] = field(default_factory=dict)
    true_counts: list[int | None] = field(default_factory=list)

    def add(self, method: str, value: float) -> None:
        self.methods.setdefault(method, []).append(value)

    def mean(self, method: str) -> float:
        return statistics.mean(self.methods[method])

    def std(self, method: str) -> float:
        vals = self.methods[method]
        return statistics.stdev(vals) if len(vals) > 1 else 0.0

    def ratio(self, base: str, target: str) -> tuple[float, float]:
        """Return mean and std of per-seed ratios."""
        ratios = [
            b / max(t, 1e-10)
            for b, t in zip(self.methods[base], self.methods[target], strict=False)
        ]
        return statistics.mean(ratios), (
            statistics.stdev(ratios) if len(ratios) > 1 else 0.0
        )

    def mean_true(self) -> float | None:
        valid = [t for t in self.true_counts if t is not None]
        return statistics.mean(valid) if valid else None


def run_multi_seed(
    experiment: str,
    k: int,
    n_v: int,
    n_e: int,
    *,
    alpha: float = 1.0,
    n_seeds: int = N_SEEDS,
    include_truth: bool = False,
    include_panda: bool = True,
) -> MultiSeedResult:
    """Run one experiment across multiple seeds."""
    result = MultiSeedResult(experiment, f"cycle_{k}", n_v, n_e)

    for s in range(n_seeds):
        seed = 1000 + s * 100
        q, data = make_cycle_data(k, n_v, n_e, alpha=alpha, seed=seed)

        ceg = estimate_pessimistic(q, data, max_block_size=1).estimate
        lpb = estimate_lpbound(q, data, p_values=P_VALUES).estimate

        result.add("CEG_M", ceg)
        result.add(COVER_LP_LABEL, lpb)

        if include_panda:
            agm = estimate_agm(q, data)
            result.add("AGM", agm)

        if include_truth:
            true_count = count_true_join(q, data)
            result.true_counts.append(true_count)

    return result


def fmt(val: float) -> str:
    """Format large numbers compactly."""
    if val >= 1e9:
        return f"{val / 1e9:.1f}B"
    if val >= 1e6:
        return f"{val / 1e6:.1f}M"
    if val >= 1e3:
        return f"{val / 1e3:.0f}K"
    return f"{val:.0f}"


def print_multi_result(r: MultiSeedResult) -> None:
    """Print one multi-seed result row."""
    ceg_m, ceg_s = r.mean("CEG_M"), r.std("CEG_M")
    lpb_m, lpb_s = r.mean(COVER_LP_LABEL), r.std(COVER_LP_LABEL)
    ratio_m, ratio_s = r.ratio("CEG_M", COVER_LP_LABEL)

    line = f"  {r.query:<10} {r.n_vertices:>5}V"
    line += f"  CEG: {fmt(ceg_m):>8}±{fmt(ceg_s):<8}"
    line += f"  Cover-LP(bs1): {fmt(lpb_m):>8}±{fmt(lpb_s):<8}"
    line += f"  ratio: {ratio_m:.2f}±{ratio_s:.2f}x"

    if "AGM" in r.methods:
        agm_m = r.mean("AGM")
        line += f"  AGM: {fmt(agm_m):>8}"

    true_m = r.mean_true()
    if true_m is not None:
        qerr_ceg = ceg_m / max(true_m, 1)
        qerr_lpb = lpb_m / max(true_m, 1)
        line += (
            f"  true: {fmt(true_m):>6}  q(CEG): {qerr_ceg:.1f}x  q(LP): {qerr_lpb:.1f}x"
        )

    print(line)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def exp1_cycle_scalability() -> list[MultiSeedResult]:
    """R1: Cycle scalability with error bars."""
    print("\n" + "=" * 70)
    print("EXP 1: Cycle Scalability (k=3..8, 200V, 800E, 10 seeds)")
    print("=" * 70)

    results = []
    for k in [3, 4, 5, 6, 7, 8]:
        r = run_multi_seed("cycle_scale", k, 200, 800, include_truth=(k <= 5))
        print_multi_result(r)
        results.append(r)
    return results


def exp2_data_scalability() -> list[MultiSeedResult]:
    """R1: Data scalability with error bars."""
    print("\n" + "=" * 70)
    print("EXP 2: Data Scalability (cycle-4, 100V-5KV, 10 seeds)")
    print("=" * 70)

    results = []
    for n_v, n_e in [(100, 400), (200, 800), (500, 2000), (1000, 4000), (2000, 8000)]:
        r = run_multi_seed("data_scale", 4, n_v, n_e, include_truth=(n_v <= 200))
        print_multi_result(r)
        results.append(r)
    return results


def exp3_skew_sensitivity() -> list[MultiSeedResult]:
    """R1: Skew sensitivity with error bars."""
    print("\n" + "=" * 70)
    print("EXP 3: Skew Sensitivity (cycle-4, 200V, alpha=0.5-3.0, 10 seeds)")
    print("=" * 70)

    results = []
    for alpha in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        r = run_multi_seed(
            f"skew_a={alpha}", 4, 200, 800, alpha=alpha, include_truth=True
        )
        print_multi_result(r)
        results.append(r)
    return results


def exp4_acyclic_control() -> list[MultiSeedResult]:
    """R1: Acyclic control with error bars."""
    print("\n" + "=" * 70)
    print("EXP 4: Acyclic Control (star-3, star-5, 200V, 10 seeds)")
    print("=" * 70)

    results = []
    for n_leaves in [3, 5]:
        result = MultiSeedResult("acyclic", f"star_{n_leaves}", 200, 800)
        for s in range(N_SEEDS):
            seed = 1000 + s * 100
            q = star_query(n_leaves)
            data: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]] = {}
            for i, rel in enumerate(q.relations):
                attrs = tuple(sorted(rel.attributes))
                data[rel.name] = generate_power_law_relation(
                    rel.name,
                    attrs,
                    200,
                    800,
                    seed=seed + i * 7,
                )
            ceg = estimate_pessimistic(q, data, max_block_size=1).estimate
            lpb = estimate_lpbound(q, data, p_values=P_VALUES).estimate
            result.add("CEG_M", ceg)
            result.add(COVER_LP_LABEL, lpb)

        print_multi_result(result)
        results.append(result)
    return results


def exp5_fair_comparison() -> list[MultiSeedResult]:
    """R4: Fair comparison at fixed bs=1 (LP vs no-LP)."""
    print("\n" + "=" * 70)
    print("EXP 5: Fair Comparison at Fixed bs=1 (cycle-4, 200V, 10 seeds)")
    print("=" * 70)
    print("  (All methods use bs=1. Isolates LP effect at same block size.)")

    results = []
    for k in [3, 4, 5, 6]:
        result = MultiSeedResult("fair_bs1", f"cycle_{k}", 200, 800)
        for s in range(N_SEEDS):
            seed = 1000 + s * 100
            q, data = make_cycle_data(k, 200, 800, seed=seed)

            agm = estimate_agm(q, data)
            ceg = estimate_pessimistic(q, data, max_block_size=1).estimate
            lpb = estimate_lpbound(q, data, p_values=P_VALUES).estimate

            result.add("AGM", agm)
            result.add("CEG_M", ceg)
            result.add(COVER_LP_LABEL, lpb)

        agm_m = result.mean("AGM")
        ceg_m = result.mean("CEG_M")
        lpb_m, lpb_s = result.mean(COVER_LP_LABEL), result.std(COVER_LP_LABEL)
        r_m, r_s = result.ratio("CEG_M", COVER_LP_LABEL)
        print(
            f"  {result.query:<10}  AGM: {fmt(agm_m):>10}  CEG_M: {fmt(ceg_m):>10}"
            f"  Cover-LP(bs1): {fmt(lpb_m):>10}±{fmt(lpb_s):<6}  LP ratio: {r_m:.2f}±{r_s:.2f}x"
        )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def export_csv(all_results: list[MultiSeedResult], path: Path) -> None:
    """Export aggregated results with mean ± std."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "experiment",
                "query",
                "n_vertices",
                "n_edges",
                "n_seeds",
                "ceg_mean",
                "ceg_std",
                "cover_lp_bs1_mean",
                "cover_lp_bs1_std",
                "ratio_mean",
                "ratio_std",
                "agm_mean",
                "true_mean",
            ]
        )
        for r in all_results:
            ceg_m, ceg_s = r.mean("CEG_M"), r.std("CEG_M")
            lpb_m, lpb_s = r.mean(COVER_LP_LABEL), r.std(COVER_LP_LABEL)
            rat_m, rat_s = r.ratio("CEG_M", COVER_LP_LABEL)
            agm_m = r.mean("AGM") if "AGM" in r.methods else ""
            true_m = r.mean_true() or ""
            writer.writerow(
                [
                    r.experiment,
                    r.query,
                    r.n_vertices,
                    r.n_edges,
                    N_SEEDS,
                    f"{ceg_m:.2f}",
                    f"{ceg_s:.2f}",
                    f"{lpb_m:.2f}",
                    f"{lpb_s:.2f}",
                    f"{rat_m:.4f}",
                    f"{rat_s:.4f}",
                    f"{agm_m:.2f}" if agm_m else "",
                    f"{true_m:.2f}" if true_m else "",
                ]
            )
    print(f"\nResults exported to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print(f"REVISION EXPERIMENTS ({N_SEEDS} seeds per experiment)")
    print("Addressing: R1 (error bars), R2 (q-error), R3 (PANDA/AGM), R4 (fair)")
    print("=" * 70)

    all_results: list[MultiSeedResult] = []

    all_results.extend(exp1_cycle_scalability())
    all_results.extend(exp2_data_scalability())
    all_results.extend(exp3_skew_sensitivity())
    all_results.extend(exp4_acyclic_control())
    all_results.extend(exp5_fair_comparison())

    export_csv(all_results, Path("results/revision/multi_seed.csv"))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total experiment groups: {len(all_results)}")
    print(f"Total individual runs: {len(all_results) * N_SEEDS}")

    lp_results = [r for r in all_results if r.ratio("CEG_M", COVER_LP_LABEL)[0] > 1.01]
    if lp_results:
        best = max(lp_results, key=lambda r: r.ratio("CEG_M", COVER_LP_LABEL)[0])
        m, s = best.ratio("CEG_M", COVER_LP_LABEL)
        print(
            f"Best LP improvement: {m:.1f}±{s:.1f}x ({best.query}, {best.n_vertices}V)"
        )


if __name__ == "__main__":
    main()
