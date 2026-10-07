"""W2: diverse real subgraph patterns (not just cycles) over Hetionet.

C1 re-scopes the efficiency argument to graph-pattern / subgraph-matching
cardinality estimation. That scope is only credible if LpH-CEG works across
the *shapes* real subgraph queries take -- cliques and chordal patterns,
whose intermediate sub-queries are themselves cyclic -- not just simple
cycles (Table realcycles). This experiment builds those shapes from real
Hetionet edges and measures the improvement over max-degree (CEG_M / cover)
and the tightness vs the entropic optimum (cover / entropic).

Patterns use the compound--gene metaedges (cbg / CdG / CuG, all
Compound-Gene) and the gene--gene interaction graph (gig), so a compound
joined to three mutually interacting genes forms a genuine real 4-clique:
  K4 on {C, G1, G2, G3}:
    C-G1 (cbg), C-G2 (CdG), C-G3 (CuG), G1-G2 (gig), G1-G3 (gig), G2-G3 (gig)
Dropping edges yields diamonds and chordal patterns.

Output: results/real/subgraph_patterns.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.estimators.lpbound_entropic import MAX_ATTRIBUTES, estimate_lpbound_entropic
from src.estimators.lph_ceg import estimate_lph_ceg
from src.estimators.pessimistic import estimate_pessimistic
from src.utils.types import QueryHypergraph, Relation

P_VALUES: tuple[float, ...] = (1.0, 2.0, float("inf"))
MAX_ROWS = 30000
DATA = Path(__file__).resolve().parent.parent / "data" / "hetionet"
RelData = dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]]


def load_edges(fname: str, max_rows: int = MAX_ROWS) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    with open(DATA / fname) as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                edges.append((parts[0], parts[1]))
                if len(edges) >= max_rows:
                    break
    return edges


# Each pattern: (label, shape, [(rel_name, file, (u, v))]).
PATTERNS: list[tuple[str, str, list[tuple[str, str, tuple[str, str]]]]] = [
    (
        "Triangle C-G-G",
        "triangle",
        [
            ("CbG", "cbg_edges.tsv", ("C", "G1")),
            ("GiG12", "gig_edges.tsv", ("G1", "G2")),
            ("CdG", "CdG_edges.tsv", ("C", "G2")),
        ],
    ),
    (
        "Diamond (2 triangles)",
        "diamond",
        [
            ("CbG", "cbg_edges.tsv", ("C", "G1")),
            ("CdG", "CdG_edges.tsv", ("C", "G2")),
            ("CuG", "CuG_edges.tsv", ("C", "G3")),
            ("GiG12", "gig_edges.tsv", ("G1", "G2")),
            ("GiG13", "gig_edges.tsv", ("G1", "G3")),
        ],
    ),
    (
        "K4 4-clique",
        "clique",
        [
            ("CbG", "cbg_edges.tsv", ("C", "G1")),
            ("CdG", "CdG_edges.tsv", ("C", "G2")),
            ("CuG", "CuG_edges.tsv", ("C", "G3")),
            ("GiG12", "gig_edges.tsv", ("G1", "G2")),
            ("GiG13", "gig_edges.tsv", ("G1", "G3")),
            ("GiG23", "gig_edges.tsv", ("G2", "G3")),
        ],
    ),
    (
        "Chordal 4-cycle",
        "chordal",
        [
            ("CbG", "cbg_edges.tsv", ("C", "G1")),
            ("GiG12", "gig_edges.tsv", ("G1", "G2")),
            ("GiG23", "gig_edges.tsv", ("G2", "G3")),
            ("CuG", "CuG_edges.tsv", ("C", "G3")),
            ("GiG13", "gig_edges.tsv", ("G1", "G3")),
        ],
    ),
]


def build(
    spec: list[tuple[str, str, tuple[str, str]]],
) -> tuple[QueryHypergraph, RelData]:
    rels: list[Relation] = []
    data: RelData = {}
    for rel_name, fname, (u, v) in spec:
        edges = load_edges(fname)
        sa = tuple(sorted((u, v)))
        rows = edges if sa == (u, v) else [(b, a) for a, b in edges]
        rels.append(Relation(rel_name, frozenset({u, v}), len(edges)))
        data[rel_name] = (sa, rows)
    return QueryHypergraph(relations=tuple(rels), name="pattern"), data


def main() -> None:
    if not DATA.exists():
        print(f"[Hetionet data not found at {DATA}; cannot run]")
        return

    print("W2: diverse REAL subgraph patterns over Hetionet")
    print("=" * 78)
    print(
        f"  {'pattern':<24} {'shape':>9} {'|V|':>4} {'#rel':>5} "
        f"{'CEG/cover':>10} {'cover/ent':>10}"
    )
    print(f"  {'-' * 76}")

    rows_out: list[dict[str, object]] = []
    for label, shape, spec in PATTERNS:
        q, data = build(spec)
        nv = len(q.all_attributes)
        ceg = estimate_pessimistic(q, data, max_block_size=1).estimate
        cover = estimate_lph_ceg(q, data, max_block_size=1, p_values=P_VALUES).estimate
        improve = ceg / max(cover, 1e-9)
        gap: float | None = None
        if nv <= MAX_ATTRIBUTES:
            try:
                ent = estimate_lpbound_entropic(q, data, p_values=P_VALUES)
                gap = cover / ent if ent > 0 else None
            except ValueError:
                gap = None
        gap_s = f"{gap:.2f}x" if gap is not None else "----"
        print(
            f"  {label:<24} {shape:>9} {nv:>4} {len(q.relations):>5} "
            f"{improve:>9.2f}x {gap_s:>10}"
        )
        rows_out.append(
            {
                "pattern": label,
                "shape": shape,
                "n_vars": nv,
                "n_rels": len(q.relations),
                "ceg_m": f"{ceg:.6e}",
                "cover_lp": f"{cover:.6e}",
                "improvement_vs_cegm": f"{improve:.4f}",
                "gap_cover_over_entropic": f"{gap:.4f}" if gap is not None else "",
            }
        )

    out = DATA.parent.parent / "results" / "real" / "subgraph_patterns.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "pattern",
                "shape",
                "n_vars",
                "n_rels",
                "ceg_m",
                "cover_lp",
                "improvement_vs_cegm",
                "gap_cover_over_entropic",
            ],
        )
        w.writeheader()
        w.writerows(rows_out)
    imps = [float(r["improvement_vs_cegm"]) for r in rows_out]
    print(f"  {'-' * 76}")
    print(
        f"  improvement {min(imps):.2f}x--{max(imps):.2f}x across "
        f"{len(rows_out)} real pattern shapes (cliques/chordal included)"
    )
    print(f"\nWrote {out} ({len(rows_out)} rows)")


if __name__ == "__main__":
    main()
