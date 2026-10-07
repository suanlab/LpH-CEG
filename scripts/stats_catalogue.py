"""B2: amortised block-statistics catalogue over the long-cycle suite.

Every query in benchmarks/query_suite_v2_long.json is a cycle over the same
Hetionet tables, so a bs=2 block (two relation instances joined on one
variable, plus its interface) is the *same physical statistic* whenever the
underlying (table, orientation) pairs and the interface coincide -- whatever
attribute names the query happens to use.  A system would materialise each
such statistic once per database; our per-query evaluation script recomputed
it for every query, which is where the "25-57 s per query at bs=2" cost in
the long-suite table comes from.

This script replays the suite in registry order with a workload-level
catalogue keyed by the canonical (relabel-invariant) identity of each
(block, interface), and records per query:

  * catalogue misses (statistics actually computed) and hits,
  * marginal statistics seconds (misses only) and LP/DAG seconds,
  * cumulative catalogue size,
  * the bs=2 bound, which must equal results/suite/long_suite_eval.csv
    (cover_bs2) -- the equality check is what certifies that the canonical
    key really identifies the same statistic.

Outputs results/suite/stats_catalogue.csv (per query) and
results/suite/stats_catalogue_summary.csv (per |V|, cold vs warm halves),
compared against the entropic solve time from the same evaluation.
"""

from __future__ import annotations

import csv
import itertools
import json
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.evaluate_query_suite import P_VALUES, build_hetionet_query
from scripts.register_query_suite import HETIONET_RELATIONS
from src.core.block import enumerate_blocks
from src.estimators.lph_ceg import estimate_lph_ceg
from src.statistics.block_stats import _enumerate_interfaces
from src.statistics.lp_stats import (
    LpBlockExtensionStats,
    compute_all_lp_block_extensions,
)

if TYPE_CHECKING:
    from src.utils.types import Block, Interface, QueryHypergraph, RelationDataMap

ROOT = Path(__file__).resolve().parent.parent
SUITE = ROOT / "benchmarks" / "query_suite_v2_long.json"
EVAL = ROOT / "results" / "suite" / "long_suite_eval.csv"
OUT_DIR = ROOT / "results" / "suite"
MAX_BLOCK_SIZE = 2

# (table name, attribute bound to the table's native source column,
#  attribute bound to its native target column)
RelationSource = tuple[str, str, str]
# canonical key: sorted (table, src-slot, tgt-slot) triples + sorted
# interface slots, minimised over all relabellings of the block's attributes
CatalogueKey = tuple[tuple[tuple[str, int, int], ...], tuple[int, ...]]


def relation_sources(q: dict) -> dict[str, RelationSource]:
    """Map each relation instance of a registered query to its table identity.

    Mirrors the orientation rule of ``build_hetionet_query``: relation ``i``
    joins ``v{i}`` to ``v{i+1}``; the file's native (source, target) columns
    bind to ``(v{i}, v{i+1})`` when the types agree and are transposed
    otherwise.
    """
    rels, types = q["relations"], q["var_types"]
    k = len(rels)
    out: dict[str, RelationSource] = {}
    for i, rname in enumerate(rels):
        native = HETIONET_RELATIONS[rname]
        want = (types[i], types[(i + 1) % k])
        u, v = f"v{i}", f"v{(i + 1) % k}"
        src, tgt = (u, v) if native == want else (v, u)
        out[f"R{i}_{rname}"] = (rname, src, tgt)
    return out


def canonical_key(
    block: Block, interface: Interface, sources: dict[str, RelationSource]
) -> CatalogueKey:
    """Relabel-invariant identity of ``(block, interface)``.

    Attributes of the block are renamed to slots ``0..m-1`` by every
    bijection; the lexicographically smallest resulting description is the
    key, so two blocks over the same tables with the same wiring and the same
    interface collide regardless of the attribute names their queries use.
    """
    triples = [sources[r] for r in block.relations]
    attrs = sorted({a for _, s, t in triples for a in (s, t)})
    best: CatalogueKey | None = None
    for perm in itertools.permutations(range(len(attrs))):
        slot = dict(zip(attrs, perm, strict=True))
        key: CatalogueKey = (
            tuple(sorted((name, slot[s], slot[t]) for name, s, t in triples)),
            tuple(sorted(slot[a] for a in interface)),
        )
        if best is None or key < best:
            best = key
    assert best is not None
    return best


class StatsCatalogue:
    """Workload-level store of ``LpBlockExtensionStats`` by canonical key."""

    def __init__(self, p_values: tuple[float, ...]) -> None:
        self.p_values = p_values
        self.store: dict[CatalogueKey, LpBlockExtensionStats] = {}
        self.hits = 0
        self.misses = 0
        self.build_seconds = 0.0

    def stats_for(
        self,
        query: QueryHypergraph,
        data: RelationDataMap,
        blocks: tuple[Block, ...],
        sources: dict[str, RelationSource],
    ) -> tuple[dict[tuple[Block, Interface], LpBlockExtensionStats], int, int]:
        """Assemble the full statistics dict for ``blocks``.

        A block whose interfaces are not all catalogued is computed once (all
        interfaces at once, as the per-query path does) and stored under its
        canonical keys.  Returns the dict plus (misses, hits) for this call.
        """
        out: dict[tuple[Block, Interface], LpBlockExtensionStats] = {}
        misses = hits = 0
        for block in blocks:
            attrs = frozenset(
                a for r in block.relations for a in query.get_relation(r).attributes
            )
            keyed = {
                i: canonical_key(block, i, sources)
                for i in _enumerate_interfaces(attrs)
            }
            if any(k not in self.store for k in keyed.values()):
                t0 = time.perf_counter()
                fresh = compute_all_lp_block_extensions(
                    query, [block], data, p_values=self.p_values
                )
                self.build_seconds += time.perf_counter() - t0
                for (_, i), st in fresh.items():
                    self.store[keyed[i]] = st
                misses += 1
            else:
                hits += 1
            for i, k in keyed.items():
                out[(block, i)] = self.store[k]
        self.misses += misses
        self.hits += hits
        return out, misses, hits


def main() -> None:
    with open(SUITE) as fh:
        suite = json.load(fh)
    with open(EVAL) as fh:
        prior = {r["query_id"]: r for r in csv.DictReader(fh)}

    cat = StatsCatalogue(P_VALUES)
    rows: list[dict[str, object]] = []
    print("Amortised statistics catalogue over the long-cycle suite (bs<=2)")
    print("=" * 78)
    print(
        f"  {'query':<9} {'|V|':>3} {'miss':>4} {'hit':>4} {'|cat|':>6} "
        f"{'stats_s':>8} {'lp_s':>7} {'total_s':>8} {'cold_s':>8} {'ent_s':>7} "
        f"{'bound=eval':>10}"
    )
    for idx, q in enumerate(suite["queries"]):
        built = build_hetionet_query(q)
        if built is None:
            continue
        query, data = built
        sources = relation_sources(q)
        blocks = enumerate_blocks(query, max_block_size=MAX_BLOCK_SIZE)

        t0 = time.perf_counter()
        stats, misses, hits = cat.stats_for(query, data, blocks, sources)
        stats_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        res = estimate_lph_ceg(
            query,
            data,
            max_block_size=MAX_BLOCK_SIZE,
            p_values=P_VALUES,
            lp_stats=stats,
        )
        lp_s = time.perf_counter() - t1

        p = prior[q["query_id"]]
        ref = float(p["cover_bs2"])
        rel = abs(res.estimate - ref) / ref
        equal = rel < 1e-6
        rows.append(
            {
                "order": idx,
                "query_id": q["query_id"],
                "n_vars": len(q["relations"]),
                "n_blocks": len(blocks),
                "misses": misses,
                "hits": hits,
                "catalogue_size": len(cat.store),
                "stats_s": f"{stats_s:.4f}",
                "lp_s": f"{lp_s:.4f}",
                "total_s": f"{stats_s + lp_s:.4f}",
                "cold_bs2_s": p["cover_bs2_s"],
                "entropic_s": p["entropic_s"],
                "bound": f"{res.estimate:.6e}",
                "bound_eval": f"{ref:.6e}",
                "rel_diff": f"{rel:.2e}",
                "bound_equal": equal,
            }
        )
        print(
            f"  {q['query_id']:<9} {len(q['relations']):>3} {misses:>4} {hits:>4} "
            f"{len(cat.store):>6} {stats_s:>8.3f} {lp_s:>7.3f} "
            f"{stats_s + lp_s:>8.3f} {float(p['cover_bs2_s']):>8.2f} "
            f"{float(p['entropic_s']):>7.3f} {equal!s:>10}",
            flush=True,
        )

    with open(OUT_DIR / "stats_catalogue.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    half = n // 2
    summary: list[dict[str, object]] = []

    def med(rs: list[dict[str, object]], col: str) -> float:
        return statistics.median(float(str(r[col])) for r in rs)

    for label, sel in (
        ("all", rows),
        ("first_half", rows[:half]),
        ("second_half", rows[half:]),
    ):
        for k in ["all", *sorted({int(str(r["n_vars"])) for r in rows})]:
            rs = [r for r in sel if k == "all" or int(str(r["n_vars"])) == k]
            if not rs:
                continue
            summary.append(
                {
                    "subset": label,
                    "n_vars": k,
                    "n_queries": len(rs),
                    "misses": sum(int(str(r["misses"])) for r in rs),
                    "hits": sum(int(str(r["hits"])) for r in rs),
                    "median_stats_s": f"{med(rs, 'stats_s'):.4f}",
                    "median_lp_s": f"{med(rs, 'lp_s'):.4f}",
                    "median_total_s": f"{med(rs, 'total_s'):.4f}",
                    "max_total_s": f"{max(float(str(r['total_s'])) for r in rs):.4f}",
                    "median_cold_bs2_s": f"{med(rs, 'cold_bs2_s'):.4f}",
                    "median_entropic_s": f"{med(rs, 'entropic_s'):.4f}",
                    "n_total_below_entropic": sum(
                        1
                        for r in rs
                        if float(str(r["total_s"])) < float(str(r["entropic_s"]))
                    ),
                    "n_bound_equal": sum(1 for r in rs if r["bound_equal"]),
                }
            )
    with open(OUT_DIR / "stats_catalogue_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print("-" * 78)
    print(
        f"  catalogue: {len(cat.store)} entries from {cat.misses} block builds "
        f"({cat.hits} hits); build time {cat.build_seconds:.1f} s total; "
        f"bounds equal to long_suite_eval on "
        f"{sum(1 for r in rows if r['bound_equal'])}/{n}"
    )
    for s in summary:
        print(
            f"  {s['subset']:<11} |V|={s['n_vars']!s:<3} n={s['n_queries']:>3} "
            f"miss/hit {s['misses']}/{s['hits']}  median total {s['median_total_s']} s "
            f"(cold {s['median_cold_bs2_s']} s, entropic {s['median_entropic_s']} s), "
            f"below entropic {s['n_total_below_entropic']}/{s['n_queries']}"
        )
    print(f"\nWrote {OUT_DIR / 'stats_catalogue.csv'} and stats_catalogue_summary.csv")


if __name__ == "__main__":
    main()
