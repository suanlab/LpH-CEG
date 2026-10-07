"""C1 SPOF: is the cover-LP-vs-entropic crossover a real-workload phenomenon?

A reviewer's sharpest attack on the efficiency argument (Section vs-lpbound)
is that the crossover was shown only on *synthetic* power-law cycles, and
that real queries are small enough that the entropic LpBound LP is cheap.
This experiment answers both parts honestly.

Two regimes, two honest conclusions:

1. **Relational OLAP (JOB).** Join-Order-Benchmark cyclic queries have very
   few join *variables* (|V| approx 4: a co-actor cycle is
   {m1, m2, actor, director}). The entropic LP has 2^4 = 16 variables and
   is trivially cheap. On this workload our efficiency advantage does NOT
   apply -- we say so plainly. (The value there is the per-step
   decomposition for WCOJ ordering, not solve cost.)

2. **Graph-pattern / subgraph matching (the regime cyclic-query bounds
   actually target).** Cardinality estimation for subgraph patterns over a
   knowledge graph routinely reaches |V| = 10-15 (long cycles, multi-hop
   paths, cliques). Here we build cycle patterns of growing length over the
   *real* Hetionet gene-interaction graph and show that the SAME crossover
   appears on real degree sequences: the entropic 2^|V| LP explodes while
   the cover LP stays flat. The crossover is a property of the LP structure,
   not of synthetic data.

The cycle patterns reuse the real GiG (gene-interacts-gene, ~147K edges)
relation for the middle edges and real CdG/CbG (compound-gene) relations to
close, giving a genuine biomedical degree distribution at every |V|.

Output: results/scalability/crossover_real.csv
"""

from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

from src.estimators.lpbound_entropic import MAX_ATTRIBUTES, estimate_lpbound_entropic
from src.estimators.lph_ceg import estimate_lph_ceg
from src.utils.types import QueryHypergraph, Relation

P_VALUES: tuple[float, ...] = (1.0, 2.0, float("inf"))
DATA = Path(__file__).resolve().parent.parent / "data" / "hetionet"
MAX_ROWS = 20000  # per-relation cap (real GiG is ~147K; cap for tractability)
N_TIMING = 3
ENTROPIC_SLOW_S = 2.0
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


def build_real_cycle(
    k: int,
    gig: list[tuple[str, str]],
    cdg: list[tuple[str, str]],
    cbg: list[tuple[str, str]],
) -> tuple[QueryHypergraph, RelData]:
    """Real Hetionet cycle on variables C, G0..G_{k-2} (|V| = k).

    Edges: CdG(C,G0), GiG(G0,G1), ..., GiG(G_{k-3},G_{k-2}), CbG(G_{k-2},C).
    Middle edges reuse the real gene-interaction degree distribution.
    """
    rels: list[Relation] = []
    data: RelData = {}

    def add(name: str, a: str, b: str, rows: list[tuple[str, str]]) -> None:
        attrs = tuple(sorted((a, b)))
        swapped = attrs != (a, b)
        rels.append(Relation(name, frozenset({a, b}), len(rows)))
        data[name] = (attrs, [(y, x) for x, y in rows] if swapped else rows)

    add("CdG", "C", "G0", cdg)
    for i in range(k - 2):
        add(f"GiG{i}", f"G{i}", f"G{i + 1}", gig)
    add("CbG", f"G{k - 2}", "C", cbg)
    return QueryHypergraph(relations=tuple(rels), name=f"real_cycle_{k}"), data


def time_cover(q: QueryHypergraph, data: RelData) -> tuple[float, float]:
    bound = estimate_lph_ceg(q, data, max_block_size=1, p_values=P_VALUES).estimate
    s = []
    for _ in range(N_TIMING):
        t0 = time.perf_counter()
        estimate_lph_ceg(q, data, max_block_size=1, p_values=P_VALUES)
        s.append((time.perf_counter() - t0) * 1000)
    return statistics.median(s), bound


def time_entropic(
    q: QueryHypergraph, data: RelData, k: int
) -> tuple[str, float | None, float | None]:
    if k > MAX_ATTRIBUTES:
        return "intractable", None, None
    try:
        t0 = time.perf_counter()
        bound = estimate_lpbound_entropic(q, data, p_values=P_VALUES)
        first = time.perf_counter() - t0
    except ValueError:
        return "intractable", None, None
    s = [first * 1000]
    if first < ENTROPIC_SLOW_S:
        for _ in range(N_TIMING - 1):
            t0 = time.perf_counter()
            estimate_lpbound_entropic(q, data, p_values=P_VALUES)
            s.append((time.perf_counter() - t0) * 1000)
    return "ok", statistics.median(s), bound


def main() -> None:
    if not DATA.exists():
        print(f"[Hetionet data not found at {DATA}; cannot run]")
        return

    print("C1: cover-LP vs entropic crossover on REAL Hetionet graph patterns")
    print("=" * 84)
    print("  JOB relational cyclic queries have |V|~4 -> entropic 2^4=16 vars,")
    print("  trivially cheap; the efficiency argument targets graph patterns")
    print("  where |V| grows. Below: real Hetionet cycles of increasing |V|.\n")

    gig = load_edges("gig_edges.tsv")
    cdg = load_edges("CdG_edges.tsv")
    cbg = load_edges("cbg_edges.tsv")

    header = (
        f"  {'|V|=k':>5} {'cover_ms':>9} {'entropic_ms':>12} "
        f"{'gap(c/e)':>9} {'status':>12}"
    )
    print(header)
    print(f"  {'-' * 52}")

    rows: list[dict[str, object]] = []
    for k in (3, 4, 5, 6, 7, 8, 10, 12, 14, 16):
        q, data = build_real_cycle(k, gig, cdg, cbg)
        cover_ms, cover_b = time_cover(q, data)
        status, ent_ms, ent_b = time_entropic(q, data, k)
        gap = cover_b / ent_b if ent_b and ent_b > 0 else None
        ent_s = f"{ent_ms:.1f}" if ent_ms is not None else "----"
        gap_s = f"{gap:.3f}x" if gap is not None else "----"
        print(f"  {k:>5} {cover_ms:>9.2f} {ent_s:>12} {gap_s:>9} {status:>12}")
        rows.append(
            {
                "k": k,
                "n_attrs": k,
                "cover_vars": 3 * k,
                "entropic_vars": 2**k,
                "cover_ms": f"{cover_ms:.4f}",
                "entropic_ms": f"{ent_ms:.4f}" if ent_ms is not None else "",
                "cover_bound": f"{cover_b:.6e}",
                "entropic_bound": f"{ent_b:.6e}" if ent_b is not None else "",
                "gap_cover_over_entropic": f"{gap:.4f}" if gap is not None else "",
                "status": status,
            }
        )

    out = DATA.parent.parent / "results" / "scalability" / "crossover_real.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "k",
                "n_attrs",
                "cover_vars",
                "entropic_vars",
                "cover_ms",
                "entropic_ms",
                "cover_bound",
                "entropic_bound",
                "gap_cover_over_entropic",
                "status",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
