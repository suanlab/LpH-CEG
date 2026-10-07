"""C0 gate: adversarial search for a Composition-Inequality validity violation.

The paper's central claim is that ``estimate_lph_ceg`` returns a *guaranteed
upper bound*: estimate >= true join size, always. A single counterexample
falsifies that claim and the whole "guaranteed" framing. Unit tests check a
handful of instances; this harness instead searches thousands of random
small databases across cyclic / overlapping query shapes and block sizes,
computing the EXACT join size by full enumeration (GenericJoin) and flagging
any instance where the estimate drops below the true count.

Small domains keep the exact count computable while still producing real
skew and cycle closure. We test the regimes where the Composition
Inequality's non-trivial step (attribute-wise cover => h(V) lower bound)
actually bites: triangles, longer cycles, cliques, and chordal patterns,
at block sizes 1..3.

Reports the minimal violating instance if any; exits nonzero on violation so
it can act as a CI gate. Output: results/validity/validity_search.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import random
import sys
from pathlib import Path

from scripts.wcoj_e2e_experiment import generic_join
from src.estimators.lph_ceg import estimate_lph_ceg
from src.utils.types import QueryHypergraph, Relation
from src.wcoj.evaluation import lexicographic_ordering

P_VALUES: tuple[float, ...] = (1.0, 2.0, float("inf"))
RelData = dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]]

# Query shapes as edge lists over named variables. All are cyclic/overlapping
# (the regime where validity is non-trivial).
SHAPES: dict[str, list[tuple[str, str]]] = {
    "triangle": [("a", "b"), ("b", "c"), ("a", "c")],
    "cycle4": [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")],
    "cycle5": [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("a", "e")],
    "diamond": [("a", "b"), ("b", "c"), ("a", "c"), ("a", "d"), ("c", "d")],
    "k4": [
        ("a", "b"),
        ("a", "c"),
        ("a", "d"),
        ("b", "c"),
        ("b", "d"),
        ("c", "d"),
    ],
    "chordal5": [
        ("a", "b"),
        ("b", "c"),
        ("c", "d"),
        ("d", "e"),
        ("a", "e"),
        ("a", "c"),
    ],
}


def build_query(edges: list[tuple[str, str]]) -> QueryHypergraph:
    rels = tuple(
        Relation(f"R{i}", frozenset({u, v}), 0) for i, (u, v) in enumerate(edges)
    )
    return QueryHypergraph(relations=rels, name="shape")


def random_data(edges: list[tuple[str, str]], dom: int, rng: random.Random) -> RelData:
    """Random relation contents over domain {0..dom-1} for each attribute."""
    data: RelData = {}
    for i, (u, v) in enumerate(edges):
        attrs = tuple(sorted((u, v)))
        all_pairs = [(str(x), str(y)) for x in range(dom) for y in range(dom)]
        # keep each pair with a random probability -> varied density & skew
        keep = rng.random() * 0.8 + 0.1  # in [0.1, 0.9]
        rows = [p for p in all_pairs if rng.random() < keep]
        if not rows:  # avoid empty relation (trivially 0 join)
            rows = [rng.choice(all_pairs)]
        data[f"R{i}"] = (attrs, rows)
    return data


def true_count(query: QueryHypergraph, data: RelData) -> int:
    return generic_join(query, data, lexicographic_ordering(query)).result_count


def check_instance(
    query: QueryHypergraph, data: RelData, block_sizes: tuple[int, ...]
) -> list[tuple[int, int, float]]:
    """Return [(bs, true, estimate)] for any bs where estimate < true."""
    true = true_count(query, data)
    if true == 0:
        return []
    bad: list[tuple[int, int, float]] = []
    for bs in block_sizes:
        est = estimate_lph_ceg(
            query, data, max_block_size=bs, p_values=P_VALUES
        ).estimate
        if est < true - 1e-6 * max(1.0, true):
            bad.append((bs, true, est))
    return bad


def exhaustive(
    shape: str, dom: int, block_sizes: tuple[int, ...], cap: int
) -> tuple[int, list[dict[str, object]]]:
    """Enumerate EVERY database over domain ``dom`` for ``shape`` (complete).

    Each relation content is an arbitrary subset of the dom*dom possible
    pairs; we take the product across relations. Feasible only for small
    (shape, dom); ``cap`` bounds the enumeration.
    """
    edges = SHAPES[shape]
    query = build_query(edges)
    pairs = [(str(x), str(y)) for x in range(dom) for y in range(dom)]
    per_rel = list(
        itertools.chain.from_iterable(
            itertools.combinations(pairs, r) for r in range(1, len(pairs) + 1)
        )
    )
    total = len(per_rel) ** len(edges)
    if total > cap:
        return 0, []  # too large; skip (random search covers it)
    checked = 0
    viols: list[dict[str, object]] = []
    for combo in itertools.product(per_rel, repeat=len(edges)):
        data: RelData = {}
        for i, (u, v) in enumerate(edges):
            data[f"R{i}"] = (tuple(sorted((u, v))), list(combo[i]))
        checked += 1
        for bs, true, est in check_instance(query, data, block_sizes):
            viols.append(
                {
                    "trial": -1,
                    "shape": shape,
                    "block_size": bs,
                    "domain": dom,
                    "true": true,
                    "estimate": f"{est:.6f}",
                    "ratio": f"{est / true:.6f}",
                    "data": repr({k: v[1] for k, v in data.items()}),
                }
            )
    return checked, viols


def run(
    n_trials: int,
    dom_choices: tuple[int, ...],
    block_sizes: tuple[int, ...],
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = random.Random(seed)
    violations: list[dict[str, object]] = []
    summary: dict[tuple[str, int], dict[str, float]] = {}

    for shape, edges in SHAPES.items():
        query = build_query(edges)
        for bs in block_sizes:
            key = (shape, bs)
            summary[key] = {"trials": 0, "violations": 0, "min_ratio": float("inf")}

    for t in range(n_trials):
        shape = rng.choice(list(SHAPES))
        edges = SHAPES[shape]
        dom = rng.choice(dom_choices)
        query = build_query(edges)
        data = random_data(edges, dom, rng)
        true = true_count(query, data)
        for bs in block_sizes:
            est = estimate_lph_ceg(
                query, data, max_block_size=bs, p_values=P_VALUES
            ).estimate
            key = (shape, bs)
            summary[key]["trials"] += 1
            ratio = est / true if true > 0 else float("inf")
            summary[key]["min_ratio"] = min(summary[key]["min_ratio"], ratio)
            # Validity: estimate must be >= true (allow tiny float slack).
            if true > 0 and est < true - 1e-6 * max(1.0, true):
                summary[key]["violations"] += 1
                violations.append(
                    {
                        "trial": t,
                        "shape": shape,
                        "block_size": bs,
                        "domain": dom,
                        "true": true,
                        "estimate": f"{est:.6f}",
                        "ratio": f"{ratio:.6f}",
                        "data": repr({k: v[1] for k, v in data.items()}),
                    }
                )

    rows = [
        {
            "shape": s,
            "block_size": bs,
            "trials": int(v["trials"]),
            "violations": int(v["violations"]),
            "min_estimate_over_true": (
                f"{v['min_ratio']:.4f}" if v["min_ratio"] != float("inf") else "n/a"
            ),
        }
        for (s, bs), v in sorted(summary.items())
    ]
    return rows, violations


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    print("C0 gate: Composition-Inequality validity search")
    print("=" * 72)

    bs = (1, 2, 3)
    # (1) EXHAUSTIVE: every domain-2 database for triangle and cycle4 (complete).
    print("  [1] Exhaustive domain-2 enumeration (complete coverage):")
    exhaustive_viol: list[dict[str, object]] = []
    exhaustive_total = 0
    for shape in ("triangle", "cycle4"):
        checked, v = exhaustive(shape, dom=2, block_sizes=bs, cap=200_000)
        exhaustive_total += checked
        exhaustive_viol.extend(v)
        print(f"      {shape:<9} d=2: {checked:>7} databases, {len(v)} violation(s)")

    # (2) RANDOM: thousands of larger/denser instances across all shapes.
    print(f"\n  [2] Random search: {args.trials} trials over shapes x domains\n")
    rows, violations = run(
        args.trials, dom_choices=(3, 4, 5, 6), block_sizes=bs, seed=args.seed
    )
    violations = exhaustive_viol + violations

    print(f"  {'shape':<10} {'bs':>3} {'trials':>7} {'viol':>5} {'min est/true':>13}")
    print(f"  {'-' * 44}")
    for r in rows:
        flag = "  <-- VIOLATION" if int(r["violations"]) > 0 else ""
        print(
            f"  {r['shape']:<10} {r['block_size']:>3} {r['trials']:>7} "
            f"{r['violations']:>5} {r['min_estimate_over_true']:>13}{flag}"
        )

    out = Path(__file__).resolve().parent.parent / "results" / "validity"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "validity_search.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "shape",
                "block_size",
                "trials",
                "violations",
                "min_estimate_over_true",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    random_checks = sum(int(r["trials"]) for r in rows)
    total_t = exhaustive_total * len(bs) + random_checks
    total_v = len(violations)
    print(
        f"\n  TOTAL: {total_v} violation(s) across "
        f"{exhaustive_total} exhaustive + {random_checks // len(bs)} random "
        f"instances (x{len(bs)} block sizes = {total_t} bound checks)"
    )
    if violations:
        v = min(violations, key=lambda x: float(x["ratio"]))
        print(
            f"  MINIMAL VIOLATION: shape={v['shape']} bs={v['block_size']} "
            f"true={v['true']} est={v['estimate']} ratio={v['ratio']}"
        )
        print(f"  data={v['data']}")
        with open(out / "validity_violations.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(violations[0].keys()))
            w.writeheader()
            w.writerows(violations)
        print(f"\n  Wrote {len(violations)} violations to validity_violations.csv")
        sys.exit(1)
    print("  RESULT: no validity violation found (bound held on every instance).")
    print(f"Wrote {out / 'validity_search.csv'}")


if __name__ == "__main__":
    main()
