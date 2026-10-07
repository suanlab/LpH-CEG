"""Ablation: does "P={1,2,inf} suffices" hold across query families?

The paper's p-value sensitivity result (Fig. pvalue) is measured on cycle-6
only. Phase 3.2 of our plan forbids declaring "suffices" from one family.
This ablation repeats the p-set sweep on three shape families (cycle,
clique, chordal) at multiple sizes/seeds and reports, per family, the
improvement over max-degree (CEG_M / cover) at each p-set. It then checks
the two sub-claims:
  (a) adding l_2 (going {1,inf} -> {1,2,inf}) helps materially, and
  (b) adding p-values beyond {1,2,inf} adds < 5%.

Output: results/ablation/pvalue_ablation.csv
"""

from __future__ import annotations

import csv
import itertools
import random
from pathlib import Path

from src.estimators.lph_ceg import estimate_lph_ceg
from src.estimators.pessimistic import estimate_pessimistic
from src.utils.types import QueryHypergraph, Relation

RelData = dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]]

INF = float("inf")
PSETS: list[tuple[float, ...]] = [
    (INF,),
    (1.0, INF),
    (1.0, 2.0, INF),
    (1.0, 2.0, 3.0, INF),
    (1.0, 1.5, 2.0, 3.0, 5.0, INF),
]
PSET_LABELS = ["{inf}", "{1,inf}", "{1,2,inf}", "{1,2,3,inf}", "{1,1.5,2,3,5,inf}"]


def cyc(k: int) -> list[tuple[str, str]]:
    v = [chr(ord("a") + i) for i in range(k)]
    return [(v[i], v[(i + 1) % k]) for i in range(k)]


FAMILIES: dict[str, list[tuple[str, str]]] = {
    "cycle4": cyc(4),
    "cycle6": cyc(6),
    "k4": list(itertools.combinations("abcd", 2)),
    "chordal5": [*cyc(5), ("a", "c")],
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


def main() -> None:
    print("Ablation: p-set sufficiency across query families")
    print("=" * 78)
    header = "  " + f"{'family':<10}" + "".join(f"{lbl:>12}" for lbl in PSET_LABELS)
    print(header)
    print(f"  {'-' * 74}")

    rows_out: list[dict[str, object]] = []
    all_l2_helps, all_extra_small = True, True
    for fam, edges in FAMILIES.items():
        # mean improvement over seeds for each p-set
        per_pset: list[float] = []
        for pset in PSETS:
            vals = []
            for seed in (1, 2, 3):
                q, data = build(edges, 100, 400, seed=100 * seed)
                ceg = estimate_pessimistic(q, data, max_block_size=1).estimate
                cover = estimate_lph_ceg(
                    q, data, max_block_size=1, p_values=pset
                ).estimate
                vals.append(ceg / max(cover, 1e-9))
            per_pset.append(sum(vals) / len(vals))
        print(f"  {fam:<10}" + "".join(f"{v:>11.2f}x" for v in per_pset))
        # sub-claims: l2 helps (idx1 -> idx2), extra p adds <5% (idx2 -> idx4)
        l2_gain = per_pset[2] / max(per_pset[1], 1e-9)
        extra_gain = per_pset[4] / max(per_pset[2], 1e-9)
        l2_helps = l2_gain >= 1.1
        extra_small = extra_gain <= 1.05
        all_l2_helps = all_l2_helps and l2_helps
        all_extra_small = all_extra_small and extra_small
        rows_out.append(
            {
                "family": fam,
                **{PSET_LABELS[i]: f"{per_pset[i]:.4f}" for i in range(len(PSETS))},
                "l2_gain": f"{l2_gain:.3f}",
                "extra_gain_beyond_1_2_inf": f"{extra_gain:.3f}",
                "l2_helps": l2_helps,
                "extra_small": extra_small,
            }
        )

    print(f"  {'-' * 74}")
    print(f"  (a) l_2 helps in ALL families: {all_l2_helps}")
    print(f"  (b) p>{{1,2,inf}} adds <5% in ALL families: {all_extra_small}")
    verdict = (
        "'P={1,2,inf} suffices' holds across families"
        if all_l2_helps and all_extra_small
        else "claim is family-dependent -- scope it"
    )
    print(f"  => {verdict}")

    out = Path(__file__).resolve().parent.parent / "results" / "ablation"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "pvalue_ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nWrote {out / 'pvalue_ablation.csv'} ({len(rows_out)} families)")


if __name__ == "__main__":
    main()
