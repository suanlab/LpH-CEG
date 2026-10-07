"""X3: Cyclic-rich real-data benchmark over the Hetionet knowledge graph.

The paper's real-data story previously rested on a single Hetionet cycle-4
(6.7x). Hetionet's metagraph has many edge types over four node types
(Compound, Disease, Gene, Anatomy), so we can construct a *suite* of
genuinely heterogeneous closed metapaths (cycles) of length 3-5 and measure
the LpH-CEG improvement (max-degree CEG_M vs. the cover-LP, bs=1) on each.

This characterizes *when* LpH-CEG helps on real data. The improvement is
*structural*, not a simple function of relation size or degree skew: per
Theorem 3 it materializes only when the hyperpath places a skewed relation
in a middle (non-closing) position, so the LP fires on some real cycles
(up to 7.6x) and correctly returns 1.00x on others---never spuriously.

Metaedges used (column order verified against the TSV headers):
  ctd: Compound-Disease   dag: Disease-Gene     cbg/CdG: Compound-Gene
  gig: Gene-Gene          AdG: Anatomy-Gene     DlA: Disease-Anatomy

Output: ``results/real/cyclic_real.csv``.
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.estimators.lph_ceg import estimate_lph_ceg
from src.estimators.pessimistic import estimate_pessimistic
from src.utils.types import QueryHypergraph, Relation

P_VALUES: tuple[float, ...] = (1.0, 2.0, float("inf"))
MAX_ROWS: int = 30000  # deterministic per-relation cap (first N rows)
DATA = Path(__file__).resolve().parent.parent / "data" / "hetionet"
RelData = dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]]


def load_edges(fname: str, max_rows: int = MAX_ROWS) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    with open(DATA / fname) as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                edges.append((parts[0], parts[1]))
                if len(edges) >= max_rows:
                    break
    return edges


# Each cycle: (label, length, node-type signature, [(rel_name, file, (attr_u, attr_v))]).
# Attribute names encode the vertex identity so shared vertices join correctly.
CYCLES: list[tuple[str, str, list[tuple[str, str, tuple[str, str]]]]] = [
    (
        "C-D-G triangle",
        "CDG",
        [
            ("CtD", "ctd_edges.tsv", ("C", "D")),
            ("DaG", "dag_edges.tsv", ("D", "G")),
            ("CbG", "cbg_edges.tsv", ("C", "G")),
        ],
    ),
    (
        "C-G-G triangle",
        "CGG",
        [
            ("CbG", "cbg_edges.tsv", ("C", "G1")),
            ("GiG", "gig_edges.tsv", ("G1", "G2")),
            ("CdG", "CdG_edges.tsv", ("C", "G2")),
        ],
    ),
    (
        "D-G-A triangle",
        "DGA",
        [
            ("DaG", "dag_edges.tsv", ("D", "G")),
            ("AdG", "AdG_edges.tsv", ("A", "G")),
            ("DlA", "DlA_edges.tsv", ("D", "A")),
        ],
    ),
    (
        "C-D-G-G cycle-4",
        "CDGG",
        [
            ("CtD", "ctd_edges.tsv", ("C", "D")),
            ("DaG", "dag_edges.tsv", ("D", "G1")),
            ("GiG", "gig_edges.tsv", ("G1", "G2")),
            ("CbG", "cbg_edges.tsv", ("C", "G2")),
        ],
    ),
    (
        "D-G-G-A cycle-4",
        "DGGA",
        [
            ("DaG", "dag_edges.tsv", ("D", "G1")),
            ("GiG", "gig_edges.tsv", ("G1", "G2")),
            ("AdG", "AdG_edges.tsv", ("A", "G2")),
            ("DlA", "DlA_edges.tsv", ("D", "A")),
        ],
    ),
    (
        "C-D-A-G-G cycle-5",
        "CDAGG",
        [
            ("CtD", "ctd_edges.tsv", ("C", "D")),
            ("DlA", "DlA_edges.tsv", ("D", "A")),
            ("AdG", "AdG_edges.tsv", ("A", "G1")),
            ("GiG", "gig_edges.tsv", ("G1", "G2")),
            ("CbG", "cbg_edges.tsv", ("C", "G2")),
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
        attrs = (u, v)
        rels.append(Relation(rel_name, frozenset({u, v}), len(edges)))
        # store with sorted schema (swap columns to match) so the row order
        # lines up with the estimator's expected attribute order
        sorted_attrs = tuple(sorted(attrs))
        rows = edges if sorted_attrs == attrs else [(b, a) for a, b in edges]
        data[rel_name] = (sorted_attrs, rows)
    return QueryHypergraph(relations=tuple(rels), name="real_cycle"), data


def main() -> None:
    if not DATA.exists():
        print(f"[Hetionet data not found at {DATA}; cannot run X3]")
        return

    print("X3: Hetionet heterogeneous cyclic benchmark (CEG_M vs cover-LP, bs=1)")
    print("=" * 84)
    print(f"  {'cycle':<20} {'len':>3} {'types':>7} {'rel sizes':>26} {'improve':>9}")
    print(f"  {'-' * 20} {'-' * 3} {'-' * 7} {'-' * 26} {'-' * 9}")

    rows_out: list[dict[str, object]] = []
    for label, sig, spec in CYCLES:
        q, data = build(spec)
        sizes = [len(data[r.name][1]) for r in q.relations]
        ceg_m = estimate_pessimistic(q, data, max_block_size=1).estimate
        cover = estimate_lph_ceg(q, data, max_block_size=1, p_values=P_VALUES).estimate
        improve = ceg_m / max(cover, 1e-9)
        length = len(spec)

        sizes_str = "/".join(str(s) for s in sizes)
        if len(sizes_str) > 26:
            sizes_str = sizes_str[:23] + "..."
        print(f"  {label:<20} {length:>3} {sig:>7} {sizes_str:>26} {improve:>8.2f}x")
        rows_out.append(
            {
                "cycle": label,
                "length": length,
                "node_types": sig,
                "rel_sizes": "/".join(str(s) for s in sizes),
                "ceg_m": f"{ceg_m:.6e}",
                "cover_lp": f"{cover:.6e}",
                "improvement": f"{improve:.4f}",
            }
        )

    out_path = (
        Path(__file__).resolve().parent.parent / "results" / "real" / "cyclic_real.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cycle",
                "length",
                "node_types",
                "rel_sizes",
                "ceg_m",
                "cover_lp",
                "improvement",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    improves = [float(r["improvement"]) for r in rows_out]
    print(f"  {'-' * 64}")
    print(
        f"  improvement range {min(improves):.2f}x--{max(improves):.2f}x "
        f"across {len(rows_out)} real heterogeneous cycles"
    )
    print(f"\nWrote {out_path} ({len(rows_out)} rows)")


if __name__ == "__main__":
    main()
