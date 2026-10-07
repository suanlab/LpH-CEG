"""Which matters more: the statistics you collect, or how you infer from them?

SUPERSEDED for the paper's Table (tab:stats-vs-inf): the ``entropic_block``
cell here runs LpBound on a *greedy pairing* of adjacent relations, which is
neither the cover LP's hyperpath blocks nor the full bs=2 catalogue, so it is
not "the same statistics".  The paper now reads every cell from
``results/flow/flow_2x2.csv``, where both programs see one identical
catalogue and the block optimum is ``poly_bs2`` (the 2^|V| polymatroid bound
over every bs<=2 block and interface).  This script is kept for the
``cover_block < entropic_single`` pins, which do not depend on that cell.

`statistics_hierarchy.py` found that the cover LP at bs>=2 produces bounds
below the entropic LpBound.  Read carelessly that says "we beat LpBound",
which would be wrong.  This is the control that settles it: give the
ENTROPIC LP the same block statistics, by materialising the pairwise joins
into virtual relations and running it on those.

The 2x2 is  statistics {single-relation, block}  x  inference {cover LP,
entropic LP}, and it separates two things the earlier experiment conflated.

Measured (40 vertices / 120 edges, seed 101):

  family     cover/single  cover/block  entropic/single  entropic/block
  triangle         1083          668            806             426
  cycle4           9871         4716           7436            1386
  diamond          9769         4560           7436            1386

Two readings, and only the second is defensible:

  * WRONG: "the cover LP beats the entropic bound".  It does beat the
    entropic bound *as published*, but only because that bound is computed
    from single-relation degree sequences.
  * RIGHT: at equal statistics the entropic LP wins by 1.6-3.4x, exactly as
    its optimality requires.  What the block result really shows is that
    the STATISTICS axis dominates the INFERENCE axis -- a cheap LP on rich
    statistics (4716) beats the optimal LP on poor statistics (7436).

So the payoff is in deciding which statistics to materialise, not in
refining the program that consumes them -- which is the same conclusion the
refinement kill-gate reached from the other direction.  It also identifies
the strongest method in the grid, entropic-on-blocks, which is nobody's
published method and which we do not claim.

Output: results/ablation/statistics_vs_inference.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

from scripts import statistics_hierarchy as sh
from src.estimators.lpbound_entropic import estimate_lpbound_entropic
from src.estimators.lph_ceg import estimate_lph_ceg
from src.utils.types import QueryHypergraph, Relation

ROOT = Path(__file__).resolve().parent.parent
FAMILIES = ("triangle", "cycle4", "diamond", "k4")
SEEDS = (101, 202, 303)


def join2(data: dict, a: str, b: str):
    """Materialise the join of two binary relations into one virtual relation."""
    (ca, ra), (cb, rb) = data[a], data[b]
    shared = [c for c in ca if c in cb]
    if not shared:
        return None
    s = shared[0]
    ia, ib = ca.index(s), cb.index(s)
    idx: dict[str, list[tuple[str, ...]]] = {}
    for row in rb:
        idx.setdefault(row[ib], []).append(row)
    cols = tuple(sorted(set(ca) | set(cb)))
    out: set[tuple[str, ...]] = set()
    for row in ra:
        for m in idx.get(row[ia], []):
            env = dict(zip(ca, row, strict=True))
            env.update(dict(zip(cb, m, strict=True)))
            out.add(tuple(env[c] for c in cols))
    return cols, sorted(out)


def blocked_query(query: QueryHypergraph, data: dict):
    """Pair adjacent relations into virtual joined relations."""
    by_name = {r.name: r for r in query.relations}
    names = [r.name for r in query.relations]
    used: set[str] = set()
    rels: list[Relation] = []
    out: dict = {}
    for i, n1 in enumerate(names):
        if n1 in used:
            continue
        partner = None
        for n2 in names[i + 1 :]:
            if n2 in used:
                continue
            if set(by_name[n1].attributes) & set(by_name[n2].attributes):
                partner = n2
                break
        if partner:
            j = join2(data, n1, partner)
            if j:
                cols, rows = j
                nm = f"{n1}_{partner}"
                rels.append(Relation(nm, frozenset(cols), len(rows)))
                out[nm] = (cols, rows)
                used.update({n1, partner})
                continue
        rels.append(by_name[n1])
        out[n1] = data[n1]
        used.add(n1)
    return QueryHypergraph(relations=tuple(rels), name="blocks"), out


def main() -> None:
    print("Statistics vs inference: the 2x2 the block result needs")
    print("=" * 82)
    print(
        f"  {'family':<9} {'seed':>5} {'cov/single':>11} {'cov/block':>11} "
        f"{'ent/single':>11} {'ent/block':>11} {'ent wins?':>10}"
    )
    print(f"  {'-' * 78}")

    rows: list[dict[str, object]] = []
    for fam in FAMILIES:
        for seed in SEEDS:
            q, d = sh.build(sh.FAMILIES[fam], seed=seed)
            c1 = estimate_lph_ceg(q, d, max_block_size=1, p_values=sh.P_VALUES).estimate
            c2 = estimate_lph_ceg(q, d, max_block_size=2, p_values=sh.P_VALUES).estimate
            e1 = estimate_lpbound_entropic(q, d, p_values=sh.P_VALUES)
            qb, db = blocked_query(q, d)
            e2 = estimate_lpbound_entropic(qb, db, p_values=sh.P_VALUES)
            wins = e2 < c2
            print(
                f"  {fam:<9} {seed:>5} {c1:>11.4g} {c2:>11.4g} "
                f"{e1:>11.4g} {e2:>11.4g} {wins!s:>10}"
            )
            rows.append(
                {
                    "family": fam,
                    "seed": seed,
                    "cover_single": f"{c1:.6e}",
                    "cover_block": f"{c2:.6e}",
                    "entropic_single": f"{e1:.6e}",
                    "entropic_block": f"{e2:.6e}",
                    "cover_block_vs_entropic_single": f"{c2 / max(e1, 1e-9):.4f}",
                    "entropic_block_vs_cover_block": f"{e2 / max(c2, 1e-9):.4f}",
                    "entropic_wins_at_equal_stats": wins,
                }
            )

    print(f"  {'-' * 78}")
    n = len(rows)
    a = sum(1 for r in rows if float(r["cover_block_vs_entropic_single"]) < 1.0)
    b = sum(1 for r in rows if r["entropic_wins_at_equal_stats"])
    print(f"  cheap LP on rich stats beats optimal LP on poor stats: {a}/{n}")
    print(f"  at EQUAL statistics the entropic LP wins:               {b}/{n}")
    print("\n  => the statistics axis dominates the inference axis; we do not")
    print("     beat LpBound's method, we change what it is given.")

    out = ROOT / "results" / "ablation"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "statistics_vs_inference.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out / 'statistics_vs_inference.csv'}")


if __name__ == "__main__":
    main()
