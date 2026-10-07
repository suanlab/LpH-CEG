"""K-best joint-optimization artifact (C-5).

Measures the additional tightening obtained by LP-optimizing the K best
hyperpaths (``explore_k=10``) instead of just the single shortest one
(``explore_k=1``).  For cyclic queries the K-best joint optimization can pick
a better extension order, yielding a tighter bound on top of the single-path
LP improvement.

Writes ``results/large_scale/explore_k.csv`` with columns:
    query, n_vertices, n_edges, bound_k1, bound_k10, additional_improvement_pct
"""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.large_scale_comparison import make_cycle_data
from src.estimators.lph_ceg import estimate_lph_ceg

P_VALUES: tuple[float, ...] = (1.0, 2.0, float("inf"))
SEED = 42

# (cycle k, n_vertices, n_edges)
CONFIGS: tuple[tuple[int, int, int], ...] = (
    (4, 100, 400),
    (4, 200, 800),
    (5, 100, 400),
    (5, 200, 800),
)


def _bound(k: int, n_v: int, n_e: int, explore_k: int) -> float:
    query, data = make_cycle_data(k, n_v, n_e, alpha=1.0, seed=SEED)
    return estimate_lph_ceg(
        query,
        data,
        max_block_size=1,
        p_values=P_VALUES,
        explore_k=explore_k,
    ).estimate


def main() -> None:
    print("=" * 78)
    print("K-best joint optimization: explore_k=1 vs explore_k=10")
    print("=" * 78)
    header = (
        f"{'query':<10} {'V':>6} {'E':>6} {'bound_k1':>16} "
        f"{'bound_k10':>16} {'add_improve%':>13}"
    )
    print(header)
    print("-" * 78)

    rows: list[dict[str, object]] = []
    for k, n_v, n_e in CONFIGS:
        b1 = _bound(k, n_v, n_e, explore_k=1)
        b10 = _bound(k, n_v, n_e, explore_k=10)
        # Additional improvement of k=10 over k=1 (positive = tighter bound).
        add_pct = (b1 - b10) / b1 * 100.0 if b1 > 0 else 0.0
        qname = f"cycle-{k}"
        print(
            f"{qname:<10} {n_v:>6} {n_e:>6} {b1:>16.2f} {b10:>16.2f} {add_pct:>12.2f}%"
        )
        rows.append(
            {
                "query": qname,
                "n_vertices": n_v,
                "n_edges": n_e,
                "bound_k1": f"{b1:.4f}",
                "bound_k10": f"{b10:.4f}",
                "additional_improvement_pct": f"{add_pct:.4f}",
            }
        )

    out_path = (
        Path(__file__).resolve().parent.parent
        / "results"
        / "large_scale"
        / "explore_k.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query",
                "n_vertices",
                "n_edges",
                "bound_k1",
                "bound_k10",
                "additional_improvement_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
