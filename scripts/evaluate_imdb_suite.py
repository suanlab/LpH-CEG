"""Evaluate the pre-registered IMDb cycle suite: the paper's comparisons on a
second real dataset.

One pass per registered query (hash-verified first), on the typed IMDb
relations of `prepare_imdb_cycles.py`:

* the long-suite estimators -- max-degree, cover LP at bs=1 and bs=2, the
  entropic LpBound -- and the exact join size (DuckDB, EXACT_TIMEOUT_S);
* the flow LP over the bs=1 statistics (LpBound's LPflow), over the full
  bs<=2 catalogue (best of the 4|V| cycle orders), and the 2^|V|
  polymatroid bound over that catalogue;
* the CorrBound proxy: bs=1 statistics plus pair-join sizes, flow LP;
* B4's split: the join-free bs=2 statistics alone, and with the l2 norm of
  the common-neighbour sequence added.

Censoring as on Hetionet: ``exact_timeout`` leaves validity unknown, it is
not assumed.  Outputs results/imdb/imdb_suite_eval.csv and
imdb_suite_summary.csv.

Usage:
    PYTHONPATH=. python scripts/evaluate_imdb_suite.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing
import resource
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING

import scripts.evaluate_long_suite as els
from scripts.evaluate_query_suite import P_VALUES
from scripts.flow_bound_blocks import _walk_order
from scripts.prepare_imdb_cycles import IMDB_RELATIONS
from scripts.pushdown_stats_experiment import split_stats
from src.core.block import enumerate_blocks
from src.estimators.flow_bound import (
    block_constraints,
    cycle_candidate_orders,
    flow_bound,
    flow_bound_over_orders,
    polymatroid_bound,
)
from src.estimators.lpbound_entropic import estimate_lpbound_entropic
from src.estimators.lph_ceg import estimate_lph_ceg
from src.estimators.pessimistic import estimate_pessimistic
from src.statistics.lp_stats import compute_all_lp_block_extensions
from src.utils.types import QueryHypergraph, Relation

if TYPE_CHECKING:
    from src.statistics.block_stats import RelationDataMap

ROOT = Path(__file__).resolve().parent.parent
SUITE = ROOT / "benchmarks" / "query_suite_v3_imdb.json"
DIGEST = ROOT / "benchmarks" / "query_suite_v3_imdb.sha256"
DATA = ROOT / "data" / "imdb_cycles"
OUT = ROOT / "results" / "imdb"
EXACT_TIMEOUT_S = 60.0
EXACT = True
WORKERS = 10
DUCKDB_THREADS = 4
POLY_MAX_VARS = 10

_cache: dict[str, list[tuple[str, str]]] = {}


def verify_registration(check_data: bool = True) -> dict:  # type: ignore[type-arg]
    expected = DIGEST.read_text().split()[0]
    actual = hashlib.sha256(SUITE.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"REFUSING TO RUN: {SUITE.name} does not match its hash")
    reg = json.loads(SUITE.read_text())
    for rel, sha in reg["data_sha256"].items() if check_data else ():
        text = (DATA / f"{rel}.tsv").read_bytes()
        if hashlib.sha256(text).hexdigest() != sha:
            raise SystemExit(f"REFUSING TO RUN: data/imdb_cycles/{rel}.tsv changed")
    print(f"  registration and data verified (sha256 {actual[:16]}...)")
    return reg  # type: ignore[no-any-return]


def load(rel: str) -> list[tuple[str, str]]:
    if rel not in _cache:
        lines = (DATA / f"{rel}.tsv").read_text().splitlines()[1:]
        _cache[rel] = [tuple(x.split("\t")) for x in lines]  # type: ignore[misc]
    return _cache[rel]


def build_imdb_query(q: dict) -> tuple[QueryHypergraph, RelationDataMap]:  # type: ignore[type-arg]
    """Relation i joins v_i to v_{i+1 mod k}, transposed as the types require."""
    names, var_types = q["relations"], q["var_types"]
    k = len(names)
    rels: list[Relation] = []
    data: RelationDataMap = {}
    for i, rname in enumerate(names):
        native = IMDB_RELATIONS[rname]
        want = (var_types[i], var_types[(i + 1) % k])
        rows = load(rname)
        oriented = rows if native == want else [(b, a) for a, b in rows]
        u, v = f"v{i}", f"v{(i + 1) % k}"
        attrs = tuple(sorted((u, v)))
        name = f"R{i}_{rname}"
        rels.append(Relation(name, frozenset({u, v}), len(oriented)))
        data[name] = (
            attrs,
            [(y, x) for x, y in oriented] if attrs != (u, v) else oriented,
        )
    return QueryHypergraph(relations=tuple(rels), name=q["query_id"]), data


def exact_count(
    query: QueryHypergraph, data: RelationDataMap, timeout_s: float
) -> int | None:
    """`evaluate_long_suite.exact_count` with DuckDB limited to DUCKDB_THREADS.

    WORKERS processes run concurrently, so each connection gets its share
    of the cores; the count itself is unchanged.
    """
    real_connect = els.duckdb.connect

    def connect(*a, **kw):  # type: ignore[no-untyped-def]
        con = real_connect(*a, **kw)
        con.execute(f"SET threads TO {DUCKDB_THREADS}")
        return con

    els.duckdb.connect = connect  # type: ignore[assignment]
    try:
        return els.exact_count(query, data, timeout_s)
    finally:
        els.duckdb.connect = real_connect  # type: ignore[assignment]


def _timed(f, *a, **kw):  # type: ignore[no-untyped-def]
    t0 = time.perf_counter()
    out = f(*a, **kw)
    return out, time.perf_counter() - t0


def evaluate(q: dict) -> dict[str, object]:  # type: ignore[type-arg]
    query, data = build_imdb_query(q)
    attrs = sorted(query.all_attributes)
    md, t_md = _timed(
        lambda: estimate_pessimistic(query, data, max_block_size=1).estimate
    )
    c1, t_c1 = _timed(
        lambda: (
            estimate_lph_ceg(query, data, max_block_size=1, p_values=P_VALUES).estimate
        )
    )
    ent, t_ent = _timed(
        lambda: estimate_lpbound_entropic(query, data, p_values=P_VALUES)
    )
    c2, t_c2 = _timed(
        lambda: (
            estimate_lph_ceg(query, data, max_block_size=2, p_values=P_VALUES).estimate
        )
    )
    stats, t_stats = _timed(
        lambda: compute_all_lp_block_extensions(
            query, enumerate_blocks(query, max_block_size=2), data, p_values=P_VALUES
        )
    )
    cons1 = block_constraints(
        query, data, max_block_size=1, p_values=P_VALUES, lp_stats=stats
    )
    cons2 = block_constraints(
        query, data, max_block_size=2, p_values=P_VALUES, lp_stats=stats
    )
    orders = cycle_candidate_orders(_walk_order(query))
    f1 = flow_bound(attrs, cons1).bound
    (f2res, _), t_f2 = _timed(lambda: flow_bound_over_orders(attrs, cons2, orders))
    f2 = f2res.bound
    poly2 = (
        2.0 ** polymatroid_bound(attrs, cons2)
        if len(attrs) <= POLY_MAX_VARS
        else math.nan
    )
    known = set(cons1)
    sizes = [c for c in cons2 if c not in known and not c.conditioning and c.p == 1]
    fsz = flow_bound(attrs, cons1 + sizes).bound
    free, _, _, _ = split_stats(query, stats)
    cons_free = block_constraints(
        query, data, max_block_size=2, p_values=P_VALUES, lp_stats=free
    )
    free_set = set(cons_free)
    l2join = [c for c in cons2 if c not in free_set and c.p == 2.0]
    linfjoin = [c for c in cons2 if c not in free_set and math.isinf(c.p)]
    ffree = flow_bound_over_orders(attrs, cons_free, orders)[0].bound
    ffree_l2 = flow_bound_over_orders(attrs, cons_free + l2join, orders)[0].bound
    ffree_linf = flow_bound_over_orders(attrs, cons_free + linfjoin, orders)[0].bound
    truth = exact_count(query, data, EXACT_TIMEOUT_S) if EXACT else None

    def valid(b: float) -> str:
        return "" if truth is None else str(b >= truth - 1e-6)

    return {
        "query_id": q["query_id"],
        "n_vars": q["n_vars"],
        "relations": "|".join(q["relations"]),
        "maxdeg": f"{md:.6e}",
        "cover_bs1": f"{c1:.6e}",
        "entropic": f"{ent:.6e}",
        "cover_bs2": f"{c2:.6e}",
        "flow_bs1": f"{f1:.6e}",
        "flow_bs2": f"{f2:.6e}",
        "poly_bs2": f"{poly2:.6e}",
        "flow_bs1_sizes": f"{fsz:.6e}",
        "flow_bs2_free": f"{ffree:.6e}",
        "flow_bs2_free_l2": f"{ffree_l2:.6e}",
        "flow_bs2_free_linf": f"{ffree_linf:.6e}",
        "true_count": "" if truth is None else truth,
        "improvement_maxdeg_over_bs1": f"{md / c1:.4f}",
        "bs1_over_entropic": f"{c1 / ent:.4f}",
        "bs2_over_entropic": f"{c2 / ent:.4e}",
        "bs2_beats_entropic": str(c2 < ent),
        "flow_bs1_eq_entropic": str(abs(f1 / ent - 1) < 1e-4),
        "cover_bs2_over_flow_bs2": f"{c2 / f2:.4f}",
        "flow_bs2_over_poly_bs2": f"{f2 / poly2:.4f}",
        "entropic_over_sizes": f"{ent / fsz:.4f}",
        "sizes_over_flow_bs2": f"{fsz / f2:.4f}",
        "free_over_full": f"{ffree / f2:.4f}",
        "free_l2_over_full": f"{ffree_l2 / f2:.4f}",
        "free_linf_over_full": f"{ffree_linf / f2:.4f}",
        "valid_cover_bs1": valid(c1),
        "valid_cover_bs2": valid(c2),
        "valid_entropic": valid(ent),
        "valid_flow_bs2": valid(f2),
        "valid_sizes": valid(fsz),
        "maxdeg_s": f"{t_md:.3f}",
        "cover_bs1_s": f"{t_c1:.3f}",
        "entropic_s": f"{t_ent:.3f}",
        "cover_bs2_s": f"{t_c2:.3f}",
        "stats_s": f"{t_stats:.3f}",
        "flow_bs2_s": f"{t_f2:.3f}",
        "censored": ""
        if truth is not None
        else ("exact_timeout" if EXACT else "no_exact"),
        "peak_rss_mb": f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f}",
    }


def _evaluate_timed(q: dict) -> tuple[dict[str, object], float]:  # type: ignore[type-arg]
    t0 = time.perf_counter()
    r = evaluate(q)
    return r, time.perf_counter() - t0


def summarise(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    def col(rs: list[dict[str, object]], c: str) -> list[float]:
        return [float(str(r[c])) for r in rs if str(r[c]) not in ("", "nan")]

    def med(rs: list[dict[str, object]], c: str) -> str:
        v = col(rs, c)
        return f"{statistics.median(v):.6g}" if v else ""

    def ratio(rs: list[dict[str, object]], a: str, b: str) -> str:
        v = [float(str(r[a])) / float(str(r[b])) for r in rs]
        return f"{statistics.median(v):.6g}"

    def over_true(rs: list[dict[str, object]], a: str) -> str:
        v = [float(str(r[a])) / int(str(r["true_count"])) for r in rs]
        return f"{statistics.median(v):.6g}" if v else ""

    out: list[dict[str, object]] = []
    for nv in ("all", 6, 8, 10):
        rs = rows if nv == "all" else [r for r in rows if int(str(r["n_vars"])) == nv]
        chk = [r for r in rs if r["true_count"] != ""]
        if not rs:
            continue
        pos = [r for r in chk if int(str(r["true_count"])) > 0]
        out.append(
            {
                "n_vars": nv,
                "n": len(rs),
                "n_checked": len(chk),
                "n_invalid": sum(
                    1
                    for r in chk
                    for c in (
                        "valid_cover_bs1",
                        "valid_cover_bs2",
                        "valid_entropic",
                        "valid_flow_bs2",
                        "valid_sizes",
                    )
                    if r[c] == "False"
                ),
                "median_improvement_maxdeg_over_bs1": med(
                    rs, "improvement_maxdeg_over_bs1"
                ),
                "median_bs1_over_entropic": med(rs, "bs1_over_entropic"),
                "max_bs1_over_entropic": f"{max(col(rs, 'bs1_over_entropic')):.4f}",
                "median_bs2_over_entropic": ratio(rs, "cover_bs2", "entropic"),
                "median_flow_bs2_over_entropic": ratio(rs, "flow_bs2", "entropic"),
                "median_sizes_over_entropic": ratio(rs, "flow_bs1_sizes", "entropic"),
                "median_true_count": (
                    f"{statistics.median(int(str(r['true_count'])) for r in chk):.6g}"
                    if chk
                    else ""
                ),
                "n_true_zero": len(chk) - len(pos),
                "median_entropic_over_true": over_true(pos, "entropic"),
                "median_cover_bs2_over_true": over_true(pos, "cover_bs2"),
                "median_flow_bs2_over_true": over_true(pos, "flow_bs2"),
                "min_cover_bs2_over_flow_bs2": f"{min(col(rs, 'cover_bs2_over_flow_bs2')):.4f}",
                "max_cover_bs2_over_flow_bs2": f"{max(col(rs, 'cover_bs2_over_flow_bs2')):.4f}",
                "n_bs2_beats_entropic": sum(
                    r["bs2_beats_entropic"] == "True" for r in rs
                ),
                "n_flow_bs1_eq_entropic": sum(
                    r["flow_bs1_eq_entropic"] == "True" for r in rs
                ),
                "median_cover_bs2_over_flow_bs2": med(rs, "cover_bs2_over_flow_bs2"),
                "n_flow_eq_poly": sum(
                    1
                    for r in rs
                    if str(r["flow_bs2_over_poly_bs2"]) != "nan"
                    and float(str(r["flow_bs2_over_poly_bs2"])) < 1.0001
                ),
                "max_flow_over_poly": f"{max(col(rs, 'flow_bs2_over_poly_bs2')):.4f}",
                "median_entropic_over_sizes": med(rs, "entropic_over_sizes"),
                "median_sizes_over_flow_bs2": med(rs, "sizes_over_flow_bs2"),
                "n_flow_bs2_below_sizes": sum(
                    float(str(r["sizes_over_flow_bs2"])) > 1.0001 for r in rs
                ),
                "median_free_over_full": med(rs, "free_over_full"),
                "median_free_l2_over_full": med(rs, "free_l2_over_full"),
                "median_free_linf_over_full": med(rs, "free_linf_over_full"),
                "n_free_l2_eq_full": sum(
                    float(str(r["free_l2_over_full"])) < 1.0001 for r in rs
                ),
                "median_cover_bs1_s": med(rs, "cover_bs1_s"),
                "median_entropic_s": med(rs, "entropic_s"),
                "median_cover_bs2_s": med(rs, "cover_bs2_s"),
                "median_flow_bs2_s": med(rs, "flow_bs2_s"),
                "median_stats_s": med(rs, "stats_s"),
                "max_stats_s": f"{max(col(rs, 'stats_s')):.1f}",
                "median_peak_rss_mb": med(rs, "peak_rss_mb")
                if "peak_rss_mb" in rs[0]
                else "",
                "max_peak_rss_mb": (
                    f"{max(col(rs, 'peak_rss_mb')):.0f}"
                    if "peak_rss_mb" in rs[0]
                    else ""
                ),
            }
        )
    return out


def _write(name: str, rows: list[dict[str, object]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    global DATA, EXACT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--tag", default="imdb_suite", help="output file prefix")
    ap.add_argument("--no-exact", action="store_true")
    ap.add_argument("--lengths", type=int, nargs="*", default=None)
    args = ap.parse_args()
    if args.summary_only:  # re-summarise an existing evaluation
        with open(OUT / f"{args.tag}_eval.csv") as fh:
            rows_in: list[dict[str, object]] = list(csv.DictReader(fh))
        _write(f"{args.tag}_summary.csv", summarise(rows_in))
        return
    DATA, EXACT = args.data_dir, not args.no_exact
    # a scale run reuses the registered queries on other data: only the
    # registry hash is checked, the data is pinned by its own MANIFEST.json
    reg = verify_registration(check_data=args.data_dir == ROOT / "data" / "imdb_cycles")
    queries = [
        q for q in reg["queries"] if args.lengths is None or q["n_vars"] in args.lengths
    ]
    rows: list[dict[str, object]] = []
    # Queries are independent; WORKERS processes evaluate them concurrently
    # (timings are per query, measured inside each worker).
    # one fresh process per query, so peak_rss_mb is that query's peak
    with multiprocessing.Pool(processes=WORKERS, maxtasksperchild=1) as pool:
        for r, elapsed in pool.imap_unordered(_evaluate_timed, queries):
            rows.append(r)
            rows.sort(key=lambda x: str(x["query_id"]))
            print(
                f"{r['query_id']} k={r['n_vars']:>2} bs1/ent={r['bs1_over_entropic']} "
                f"bs2/ent={r['bs2_over_entropic']} cov2/flow2={r['cover_bs2_over_flow_bs2']} "
                f"flow/poly={r['flow_bs2_over_poly_bs2']} ent/sizes={r['entropic_over_sizes']} "
                f"sizes/flow2={r['sizes_over_flow_bs2']} true={r['true_count']} "
                f"[{elapsed:.0f}s]",
                flush=True,
            )
            _write(f"{args.tag}_eval.csv", rows)  # incremental: survive interruption
    _write(f"{args.tag}_summary.csv", summarise(rows))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
