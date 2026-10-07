"""Evaluate the pre-registered query suite (W6 bound-frontier pilot).

Runs the suite fixed by `register_query_suite.py` and reports the
cover-LP / entropic / max-degree frontier as a *distribution*, per the
plan's rule that no main claim rests on 6-9 hand-picked queries.

Pre-registration is enforced, not assumed: this script refuses to run
unless `benchmarks/query_suite_v1.json` still hashes to the value pinned
at registration time.  Every registered query is evaluated; none may be
dropped for looking unfavourable.

Censoring rules (declared here, applied uniformly, reported not hidden)
----------------------------------------------------------------------
  R1  Each relation is capped at MAX_ROWS rows, taken as the first N in
      file order (deterministic, not sampled).
  R2  A query is censored `no_data` if any of its relation files is
      missing.
  R3  A query is censored `entropic_intractable` if |V| > ENTROPIC_MAX_V,
      where the 2^|V| Shannon-cone LP is not affordable.  Its cover bound
      is still reported; only the gap column is blank.
  R4  A query is censored `timeout` if its bound computation exceeds
      PER_QUERY_TIMEOUT_S.

Censored queries appear in the CSV with their reason, and the summary
prints the censored count alongside every statistic.

Outputs: results/suite/suite_eval.csv, results/suite/suite_summary.csv
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

from src.estimators.lpbound_entropic import estimate_lpbound_entropic
from src.estimators.lph_ceg import estimate_lph_ceg
from src.estimators.pessimistic import estimate_pessimistic
from src.utils.types import QueryHypergraph, Relation

ROOT = Path(__file__).resolve().parent.parent
SUITE = ROOT / "benchmarks" / "query_suite_v1.json"
DIGEST = ROOT / "benchmarks" / "query_suite_v1.sha256"
HETIONET = ROOT / "data" / "hetionet"
EPINIONS = ROOT / "data" / "epinions"

P_VALUES: tuple[float, ...] = (1.0, 2.0, float("inf"))
MAX_ROWS = 20000  # R1
ENTROPIC_MAX_V = 6  # R3
PER_QUERY_TIMEOUT_S = 90.0  # R4

RelData = dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]]

HETIONET_FILES = {
    "AdG": "AdG_edges.tsv",
    "AuG": "AuG_edges.tsv",
    "CdG": "CdG_edges.tsv",
    "CuG": "CuG_edges.tsv",
    "DdG": "DdG_edges.tsv",
    "DlA": "DlA_edges.tsv",
    "DrD": "DrD_edges.tsv",
    "DuG": "DuG_edges.tsv",
    "cbg": "cbg_edges.tsv",
    "crc": "crc_edges.tsv",
    "ctd": "ctd_edges.tsv",
    "dag": "dag_edges.tsv",
    "gig": "gig_edges.tsv",
}


def verify_registration() -> dict:
    """Refuse to run against an edited suite."""
    if not SUITE.exists() or not DIGEST.exists():
        raise SystemExit("query suite not registered; run register_query_suite.py")
    expected = DIGEST.read_text().split()[0]
    actual = hashlib.sha256(SUITE.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(
            "REFUSING TO RUN: query_suite_v1.json does not match its registered "
            f"hash.\n  registered {expected}\n  found      {actual}\n"
            "The suite was edited after pre-registration."
        )
    print(f"  pre-registration verified (sha256 {actual[:16]}...)")
    return json.loads(SUITE.read_text())


_edge_cache: dict[str, list[tuple[str, str]]] = {}


def load_edges(
    fname: str, base: Path, has_header: bool = True
) -> list[tuple[str, str]]:
    key = str(base / fname)
    if key in _edge_cache:
        return _edge_cache[key]
    edges: list[tuple[str, str]] = []
    with open(base / fname) as f:
        if has_header:
            f.readline()
        for line in f:
            parts = line.rstrip("\n").replace("\t", " ").split()
            if len(parts) >= 2:
                edges.append((parts[0], parts[1]))
                if len(edges) >= MAX_ROWS:
                    break
    _edge_cache[key] = edges
    return edges


def build_hetionet_query(q: dict) -> tuple[QueryHypergraph, RelData] | None:
    """Instantiate a registered metagraph cycle on the real data.

    Relation i joins variable i to variable i+1 (mod k); it is transposed
    when the file's native (source, target) types are reversed relative to
    that orientation.
    """
    from scripts.register_query_suite import HETIONET_RELATIONS

    rels_names, var_types = q["relations"], q["var_types"]
    k = len(rels_names)
    rels: list[Relation] = []
    data: RelData = {}

    for i, rname in enumerate(rels_names):
        fname = HETIONET_FILES.get(rname)
        if fname is None or not (HETIONET / fname).exists():
            return None
        native = HETIONET_RELATIONS[rname]
        want = (var_types[i], var_types[(i + 1) % k])
        rows = load_edges(fname, HETIONET)
        if native == want:
            oriented = rows
        elif native == (want[1], want[0]):
            oriented = [(b, a) for a, b in rows]
        else:  # pragma: no cover - registration guarantees typing
            return None

        u, v = f"v{i}", f"v{(i + 1) % k}"
        attrs = tuple(sorted((u, v)))
        swap = attrs != (u, v)
        name = f"R{i}_{rname}"
        rels.append(Relation(name, frozenset({u, v}), len(oriented)))
        data[name] = (attrs, [(y, x) for x, y in oriented] if swap else oriented)

    return QueryHypergraph(relations=tuple(rels), name=q["query_id"]), data


def build_epinions_query(q: dict) -> tuple[QueryHypergraph, RelData] | None:
    fname = "edges.txt"
    if not (EPINIONS / fname).exists():
        return None
    rows = load_edges(fname, EPINIONS, has_header=False)
    rels: list[Relation] = []
    data: RelData = {}
    for i, spec in enumerate(q["relations"]):
        inner = spec[spec.index("[") + 1 : spec.index("]")]
        a, b = inner.split(",")
        u, v = f"v{a}", f"v{b}"
        attrs = tuple(sorted((u, v)))
        swap = attrs != (u, v)
        name = f"R{i}"
        rels.append(Relation(name, frozenset({u, v}), len(rows)))
        data[name] = (attrs, [(y, x) for x, y in rows] if swap else rows)
    return QueryHypergraph(relations=tuple(rels), name=q["query_id"]), data


def main() -> None:
    print("Evaluating the pre-registered query suite")
    print("=" * 78)
    registry = verify_registration()
    queries = registry["queries"]
    print(f"  {len(queries)} registered queries; all will be attempted\n")

    out_rows: list[dict[str, object]] = []
    n_done = 0
    for qi, q in enumerate(queries):
        row: dict[str, object] = {
            "query_id": q["query_id"],
            "dataset": q["dataset"],
            "shape": q["shape"],
            "n_vars": q["n_vars"],
            "relations": "|".join(q["relations"]),
            "ceg_m": "",
            "cover_lp": "",
            "entropic": "",
            "improvement_vs_cegm": "",
            "gap_cover_over_entropic": "",
            "cover_ms": "",
            "censored": "",
        }
        built = (
            build_hetionet_query(q)
            if q["dataset"] == "hetionet"
            else build_epinions_query(q)
        )
        if built is None:
            row["censored"] = "no_data"  # R2
            out_rows.append(row)
            continue

        query, data = built
        try:
            t0 = time.perf_counter()
            cegm = estimate_pessimistic(query, data, max_block_size=1).estimate
            cover = estimate_lph_ceg(
                query, data, max_block_size=1, p_values=P_VALUES
            ).estimate
            cover_ms = (time.perf_counter() - t0) * 1000.0
            if cover_ms / 1000.0 > PER_QUERY_TIMEOUT_S:
                row["censored"] = "timeout"  # R4
                out_rows.append(row)
                continue
            row["ceg_m"] = f"{cegm:.6e}"
            row["cover_lp"] = f"{cover:.6e}"
            row["cover_ms"] = f"{cover_ms:.2f}"
            row["improvement_vs_cegm"] = f"{cegm / max(cover, 1e-9):.4f}"

            if q["n_vars"] <= ENTROPIC_MAX_V:
                ent = estimate_lpbound_entropic(query, data, p_values=P_VALUES)
                row["entropic"] = f"{ent:.6e}"
                row["gap_cover_over_entropic"] = f"{cover / max(ent, 1e-9):.4f}"
            else:
                row["censored"] = "entropic_intractable"  # R3
        except Exception as exc:  # record the failure, never silently drop
            row["censored"] = f"error:{type(exc).__name__}"
        out_rows.append(row)
        n_done += 1
        if (qi + 1) % 25 == 0:
            print(f"  ... {qi + 1}/{len(queries)} attempted")

    out_dir = ROOT / "results" / "suite"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "suite_eval.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    # ---- summary -------------------------------------------------------
    def pct(xs: list[float], p: float) -> float:
        if not xs:
            return float("nan")
        s = sorted(xs)
        i = min(round(p / 100.0 * (len(s) - 1)), len(s) - 1)
        return s[i]

    summary: list[dict[str, object]] = []
    print("\n" + "=" * 78)
    for ds in ["hetionet", "epinions", "ALL"]:
        sel = [r for r in out_rows if ds == "ALL" or r["dataset"] == ds]
        gaps = [
            float(r["gap_cover_over_entropic"])
            for r in sel
            if r["gap_cover_over_entropic"]
        ]
        imps = [
            float(r["improvement_vs_cegm"]) for r in sel if r["improvement_vs_cegm"]
        ]
        cens = [r for r in sel if r["censored"]]
        if not sel:
            continue
        # Distinct-value counts expose degenerate scopes: a workload whose
        # queries all yield the same ratio contributes one observation, not
        # n of them, and its percentiles must not be read as a distribution.
        rec = {
            "scope": ds,
            "n_registered": len(sel),
            "n_with_gap": len(gaps),
            "n_distinct_gaps": len({round(g, 4) for g in gaps}),
            "n_censored": len(cens),
            "gap_median": f"{statistics.median(gaps):.4f}" if gaps else "",
            "gap_p90": f"{pct(gaps, 90):.4f}" if gaps else "",
            "gap_p95": f"{pct(gaps, 95):.4f}" if gaps else "",
            "gap_max": f"{max(gaps):.4f}" if gaps else "",
            "improvement_median": f"{statistics.median(imps):.4f}" if imps else "",
            "improvement_max": f"{max(imps):.4f}" if imps else "",
            "frac_improved": f"{sum(1 for x in imps if x > 1.001) / len(imps):.4f}"
            if imps
            else "",
        }
        summary.append(rec)
        degen = rec["n_distinct_gaps"] <= 1 and len(gaps) > 1
        print(
            f"  {ds:<10} n={len(sel):<4} gap median={rec['gap_median'] or '-':<8} "
            f"P90={rec['gap_p90'] or '-':<8} P95={rec['gap_p95'] or '-':<8} "
            f"max={rec['gap_max'] or '-':<8} censored={len(cens)}"
            f"{'  [DEGENERATE: all gaps identical]' if degen else ''}"
        )
        print(
            f"  {'':<10} improvement over max-degree: median="
            f"{rec['improvement_median'] or '-'}, max={rec['improvement_max'] or '-'}, "
            f"fraction improved={rec['frac_improved'] or '-'}"
        )

    reasons: dict[str, int] = {}
    for r in out_rows:
        if r["censored"]:
            reasons[str(r["censored"])] = reasons.get(str(r["censored"]), 0) + 1
    if reasons:
        print("\n  censoring breakdown (nothing hidden):")
        for k in sorted(reasons):
            print(f"    {k:<24} {reasons[k]}")

    with open(out_dir / "suite_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # ---- pre-registered C2 bar ----------------------------------------
    allrec = next((s for s in summary if s["scope"] == "ALL"), None)
    if allrec and allrec["gap_median"]:
        med, p90 = float(allrec["gap_median"]), float(allrec["gap_p90"])
        met = med <= 3.0 and p90 <= 8.0
        print(
            f"\n  C2 public-suite bar (median <=3x, P90 <=8x): "
            f"median={med:.2f}x, P90={p90:.2f}x -> {'MET' if met else 'NOT MET'}"
        )

    print(f"\nWrote {out_dir / 'suite_eval.csv'} ({len(out_rows)} rows)")
    print(f"Wrote {out_dir / 'suite_summary.csv'}")
    if n_done == 0:
        sys.exit("no query could be evaluated")


if __name__ == "__main__":
    main()
