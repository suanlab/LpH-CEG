"""Validity search for the flow bound over block statistics (B3 gate).

The flow LP in ``src/estimators/flow_bound.py`` is Im et al.'s network-flow
program (exact for simple constraints, Thm. 1.1) with LpBound's ``lp``
bundles (Thm. 4.5) and, for the non-simple ``|I| = 2`` block constraints, the
order-relaxation of Im et al. Section 7.  The ``lp`` bundle on a non-simple
constraint is our own extension, so before any number is reported the
combination is checked against the exact ``2**|V|`` polymatroid LP on random
small queries, for EVERY attribute order:

* ``flow(pi) >= poly(DC_pi)``: the flow value never drops below the
  polymatroid bound of the relaxed constraint set it certifies
  (Im et al., Thm. 1.5a) -- the validity obligation;
* ``poly(DC_pi) >= poly(DC)``: the relaxation only loosens;
* ``min_pi flow(pi) >= log2(true count)``: an upper bound on real data;
* ``flow(simple only) == poly(simple only)`` and, at ``bs = 1``,
  ``flow == entropic LpBound``: the exactness theorems reproduce;
* the same checks with ``prefix_sources`` (Im et al.'s optional
  strengthening) and with randomly rescaled constraint costs, which breaks
  the data-consistency of the statistics and exercises the LP itself.

Writes ``results/flow/flow_validity.csv`` (one row per random instance) and
``results/flow/flow_validity_summary.csv``.
"""

from __future__ import annotations

import csv
import itertools
import math
import random
import time
from pathlib import Path

from scripts import statistics_hierarchy as sh
from src.estimators.flow_bound import (
    DegreeConstraint,
    block_constraints,
    flow_bound,
    polymatroid_bound,
)
from src.estimators.lpbound_entropic import estimate_lpbound_entropic

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "flow"
N_TRIALS = 40
N_RANDOM_COST_ORDERS = 6
TOL = 1e-6


def relaxed(
    cons: list[DegreeConstraint], order: tuple[str, ...]
) -> list[DegreeConstraint]:
    """The relaxed constraint set ``DC_pi`` certified by ``flow(pi)``."""
    pos = {a: i for i, a in enumerate(order)}
    out: list[DegreeConstraint] = []
    for c in cons:
        if c.is_simple:
            out.append(c)
            continue
        last = max(pos[a] for a in c.conditioning)
        y_prime = frozenset(a for a in c.target - c.conditioning if pos[a] > last)
        if y_prime:
            out.append(
                DegreeConstraint(
                    c.conditioning, c.conditioning | y_prime, c.p, c.log_norm
                )
            )
    return out


def random_connected_graph(
    rng: random.Random, n_v: int, n_e: int
) -> list[tuple[str, str]]:
    """A random connected simple graph on ``n_v`` vertices, ``n_e`` edges."""
    v = [chr(ord("a") + i) for i in range(n_v)]
    perm = v[:]
    rng.shuffle(perm)
    edges = {tuple(sorted((perm[i], perm[i + 1]))) for i in range(n_v - 1)}
    while len(edges) < n_e:
        a, b = rng.sample(v, 2)
        edges.add(tuple(sorted((a, b))))
    return sorted(edges)  # type: ignore[arg-type]


def main() -> None:
    rng = random.Random(7)
    rows: list[dict[str, object]] = []
    t_start = time.time()
    print("Flow-bound validity search (all attribute orders, |V| <= 5)")
    print("=" * 88)
    for trial in range(N_TRIALS):
        n_v = rng.choice([3, 4, 4, 5])
        n_e = rng.randint(n_v, min(n_v * (n_v - 1) // 2, n_v + 2))
        edges = random_connected_graph(rng, n_v, n_e)
        sh.N_V = rng.choice([20, 40, 60])
        sh.N_E = rng.choice([60, 120, 200])
        q, d = sh.build(edges, seed=rng.randint(0, 10**6))
        attrs = sorted(q.all_attributes)
        cons = block_constraints(q, d, max_block_size=2, p_values=sh.P_VALUES)
        cons1 = block_constraints(q, d, max_block_size=1, p_values=sh.P_VALUES)
        poly_all = polymatroid_bound(attrs, cons)
        poly_simple = polymatroid_bound(attrs, [c for c in cons if c.is_simple])
        flow_simple = flow_bound(attrs, cons).log_bound
        flow1 = flow_bound(attrs, cons1).log_bound
        ent1 = math.log2(estimate_lpbound_entropic(q, d, p_values=sh.P_VALUES))
        exact = math.log2(max(1, sh.true_count(q, d)))

        worst_flow_below_relaxed = -math.inf
        worst_relaxed_below_all = -math.inf
        worst_prefix_below_relaxed = -math.inf
        worst_flow_above_relaxed = -math.inf
        best = best_prefix = math.inf
        n_orders = 0
        for order in itertools.permutations(attrs):
            f = flow_bound(attrs, cons, order=order).log_bound
            fp = flow_bound(attrs, cons, order=order, prefix_sources=True).log_bound
            pr = polymatroid_bound(attrs, relaxed(cons, order))
            worst_flow_below_relaxed = max(worst_flow_below_relaxed, pr - f)
            worst_prefix_below_relaxed = max(worst_prefix_below_relaxed, pr - fp)
            worst_relaxed_below_all = max(worst_relaxed_below_all, poly_all - pr)
            worst_flow_above_relaxed = max(worst_flow_above_relaxed, f - pr)
            best = min(best, f)
            best_prefix = min(best_prefix, fp)
            n_orders += 1
        # random costs: the LP must still certify a valid inequality
        cons2 = [
            DegreeConstraint(
                c.conditioning, c.target, c.p, c.log_norm * rng.uniform(0.3, 1.5)
            )
            for c in cons
        ]
        worst_random_cost = -math.inf
        for _ in range(N_RANDOM_COST_ORDERS):
            order = tuple(rng.sample(attrs, len(attrs)))
            pr = polymatroid_bound(attrs, relaxed(cons2, order))
            f = flow_bound(attrs, cons2, order=order).log_bound
            fp = flow_bound(attrs, cons2, order=order, prefix_sources=True).log_bound
            worst_random_cost = max(worst_random_cost, pr - f, pr - fp)

        row = {
            "trial": trial,
            "n_vars": n_v,
            "n_rels": n_e,
            "n_constraints": len(cons),
            "n_orders": n_orders,
            "exact_log2": f"{exact:.6f}",
            "poly_all": f"{poly_all:.6f}",
            "poly_simple": f"{poly_simple:.6f}",
            "flow_simple": f"{flow_simple:.6f}",
            "flow_best": f"{best:.6f}",
            "flow_prefix_best": f"{best_prefix:.6f}",
            "flow1": f"{flow1:.6f}",
            "entropic1": f"{ent1:.6f}",
            "gap_flow_best_minus_poly_all": f"{best - poly_all:.6e}",
            "gap_prefix_best_minus_poly_all": f"{best_prefix - poly_all:.6e}",
            "worst_flow_below_relaxed": f"{worst_flow_below_relaxed:.3e}",
            "worst_prefix_below_relaxed": f"{worst_prefix_below_relaxed:.3e}",
            "worst_relaxed_below_all": f"{worst_relaxed_below_all:.3e}",
            "worst_flow_above_relaxed": f"{worst_flow_above_relaxed:.3e}",
            "worst_random_cost_below_relaxed": f"{worst_random_cost:.3e}",
            "flow_simple_eq_poly_simple": abs(flow_simple - poly_simple) < TOL,
            "flow1_eq_entropic1": abs(flow1 - ent1) < TOL,
            "valid_vs_exact": best >= exact - TOL,
            "flow_reaches_poly_all": best - poly_all < TOL,
        }
        rows.append(row)
        print(
            f"trial {trial:2d} |V|={n_v} |E|={n_e} orders={n_orders:3d}: "
            f"poly-all {poly_all:.4f} flow-best {best:.4f} "
            f"gap {best - poly_all:+.2e} prefix {best_prefix - poly_all:+.2e} "
            f"worst-below {worst_flow_below_relaxed:+.1e} exact {exact:.3f}",
            flush=True,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "flow_validity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_pairs = sum(int(r["n_orders"]) for r in rows)
    summary = {
        "n_instances": len(rows),
        "n_order_pairs": n_pairs,
        "n_random_cost_pairs": N_RANDOM_COST_ORDERS * len(rows),
        "worst_flow_below_relaxed": max(
            float(r["worst_flow_below_relaxed"]) for r in rows
        ),
        "worst_prefix_below_relaxed": max(
            float(r["worst_prefix_below_relaxed"]) for r in rows
        ),
        "worst_random_cost_below_relaxed": max(
            float(r["worst_random_cost_below_relaxed"]) for r in rows
        ),
        "worst_relaxed_below_all": max(
            float(r["worst_relaxed_below_all"]) for r in rows
        ),
        "max_flow_above_relaxed": max(
            float(r["worst_flow_above_relaxed"]) for r in rows
        ),
        "n_valid_vs_exact": sum(bool(r["valid_vs_exact"]) for r in rows),
        "n_flow_simple_eq_poly_simple": sum(
            bool(r["flow_simple_eq_poly_simple"]) for r in rows
        ),
        "n_flow1_eq_entropic1": sum(bool(r["flow1_eq_entropic1"]) for r in rows),
        "n_flow_reaches_poly_all": sum(bool(r["flow_reaches_poly_all"]) for r in rows),
        "n_prefix_improves": sum(
            float(r["gap_prefix_best_minus_poly_all"])
            < float(r["gap_flow_best_minus_poly_all"]) - TOL
            for r in rows
        ),
        "max_gap_bits": max(float(r["gap_flow_best_minus_poly_all"]) for r in rows),
        "seconds": round(time.time() - t_start, 1),
    }
    with (OUT_DIR / "flow_validity_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)
    print()
    for k, v in summary.items():
        print(f"  {k:36s} {v}")
    print(f"\nWrote {OUT_DIR / 'flow_validity.csv'}")


if __name__ == "__main__":
    main()
