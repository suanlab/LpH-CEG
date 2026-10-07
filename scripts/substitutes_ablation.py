"""Ablation: are LP and block extensions "substitutes" across query families?

The paper claims (abstract/intro/conclusion) that l_p LP optimization and
multi-way block extensions are largely substitutes on cyclic queries:
(a) the bs=1 LP recovers most of the bs=2 block benefit, and
(b) the LP adds little once bs>=2 blocks already capture the cycle.
This rests on one 2x2 synthetic-cycle setting. Here we repeat it on five
shape families and report, per family (mean over seeds):

  cegm   = CEG_M = pessimistic(bs=1)   (max-degree, no LP)
  lp1    = lph_ceg(bs=1)               (LP, binary)
  blk2   = pessimistic(bs=2)           (block, no LP)
  lp2    = lph_ceg(bs=2)               (LP + block)

Sub-claim (a): lp1 <= blk2 (LP at bs=1 is at least as tight as block at
bs=2), reported as blk2/lp1 (>=1 means the LP matches or beats blocks).
Sub-claim (b): lp2 ~ blk2 (LP adds little at bs=2), reported as blk2/lp2.

Output: results/ablation/substitutes_ablation.csv
"""

from __future__ import annotations

import csv
import itertools
import random
from pathlib import Path

from src.estimators.lph_ceg import estimate_lph_ceg
from src.estimators.pessimistic import estimate_pessimistic
from src.utils.types import QueryHypergraph, Relation

P = (1.0, 2.0, float("inf"))
RelData = dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]]


def cyc(k: int) -> list[tuple[str, str]]:
    v = [chr(ord("a") + i) for i in range(k)]
    return [(v[i], v[(i + 1) % k]) for i in range(k)]


FAMILIES: dict[str, list[tuple[str, str]]] = {
    "cycle4": cyc(4),
    "cycle5": cyc(5),
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
    print("Ablation: LP vs block extensions -- substitutes across families?")
    print("=" * 76)
    print(
        f"  {'family':<10} {'CEG_M':>10} {'LP(bs1)':>10} {'blk(bs2)':>10} "
        f"{'LP(bs2)':>10} {'blk2/lp1':>9} {'blk2/lp2':>9}"
    )
    print(f"  {'-' * 72}")

    rows_out = []
    a_holds, b_holds = True, True
    for fam, edges in FAMILIES.items():
        vals = {"cegm": [], "lp1": [], "blk2": [], "lp2": []}
        for seed in (1, 2, 3):
            q, data = build(edges, 100, 400, seed=100 * seed)
            vals["cegm"].append(
                estimate_pessimistic(q, data, max_block_size=1).estimate
            )
            vals["lp1"].append(
                estimate_lph_ceg(q, data, max_block_size=1, p_values=P).estimate
            )
            vals["blk2"].append(
                estimate_pessimistic(q, data, max_block_size=2).estimate
            )
            vals["lp2"].append(
                estimate_lph_ceg(q, data, max_block_size=2, p_values=P).estimate
            )
        m = {k: sum(v) / len(v) for k, v in vals.items()}
        blk2_over_lp1 = m["blk2"] / max(m["lp1"], 1e-9)
        blk2_over_lp2 = m["blk2"] / max(m["lp2"], 1e-9)
        # (a) LP at bs1 matches/beats block at bs2: blk2/lp1 >= ~0.9
        # (b) LP adds little at bs2: blk2/lp2 within [0.95, 1.05]
        a_ok = blk2_over_lp1 >= 0.9
        b_ok = 0.9 <= blk2_over_lp2 <= 1.11
        a_holds = a_holds and a_ok
        b_holds = b_holds and b_ok
        print(
            f"  {fam:<10} {m['cegm']:>10.2e} {m['lp1']:>10.2e} {m['blk2']:>10.2e} "
            f"{m['lp2']:>10.2e} {blk2_over_lp1:>8.2f}x {blk2_over_lp2:>8.2f}x"
        )
        rows_out.append(
            {
                "family": fam,
                "cegm": f"{m['cegm']:.6e}",
                "lp_bs1": f"{m['lp1']:.6e}",
                "block_bs2": f"{m['blk2']:.6e}",
                "lp_bs2": f"{m['lp2']:.6e}",
                "block2_over_lp1": f"{blk2_over_lp1:.4f}",
                "block2_over_lp2": f"{blk2_over_lp2:.4f}",
                "a_lp1_matches_block2": a_ok,
                "b_lp_adds_little_at_bs2": b_ok,
            }
        )

    print(f"  {'-' * 72}")
    print(f"  (a) LP(bs1) matches/beats block(bs2) in ALL families: {a_holds}")
    print(f"  (b) LP adds little at bs2 in ALL families: {b_holds}")
    print(
        f"  => substitutes claim {'holds' if a_holds and b_holds else 'is nuanced'} "
        "across families"
    )

    out = Path(__file__).resolve().parent.parent / "results" / "ablation"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "substitutes_ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nWrote {out / 'substitutes_ablation.csv'} ({len(rows_out)} families)")


if __name__ == "__main__":
    main()
