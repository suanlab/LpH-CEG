"""Evaluate the pre-registered LONG-cycle suite (|V| in {6, 8, 10}) on Hetionet.

Closes the evaluation gap flagged by two simulated reviewers: the paper
motivates |V| = 10-15 but the v1 suite stops at |V| = 5.  This runs the
v2-long registry (`register_long_suite.py`, hash-verified first) with the
four estimators the paper compares, plus an exact join count so every
bound's validity is checked rather than assumed:

    max-degree (H-CEG, bs=1) | cover LP bs=1 | cover LP bs=2 | entropic LpBound

Censoring rules, applied uniformly and reported
----------------------------------------------
  R1  Relations capped at MAX_ROWS rows, first N in file order (as v1).
  R2  `bs2_timeout`: cover bs=2 exceeded BS2_TIMEOUT_S (block statistics
      materialise two-relation joins; the bound is still reported if it
      finished, only flagged).
  R3  `exact_timeout`: DuckDB did not return the exact count within
      EXACT_TIMEOUT_S; validity is then unknown, not assumed.

Outputs: results/suite/long_suite_eval.csv, results/suite/long_suite_summary.csv
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from scripts.evaluate_query_suite import P_VALUES, build_hetionet_query
from src.estimators.lpbound_entropic import estimate_lpbound_entropic
from src.estimators.lph_ceg import estimate_lph_ceg
from src.estimators.pessimistic import estimate_pessimistic

if TYPE_CHECKING:
    from src.utils.types import QueryHypergraph

ROOT = Path(__file__).resolve().parent.parent
SUITE = ROOT / "benchmarks" / "query_suite_v2_long.json"
DIGEST = ROOT / "benchmarks" / "query_suite_v2_long.sha256"
BS2_TIMEOUT_S = 180.0
EXACT_TIMEOUT_S = 60.0


def verify_registration() -> dict:
    expected = DIGEST.read_text().split()[0]
    actual = hashlib.sha256(SUITE.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(
            "REFUSING TO RUN: query_suite_v2_long.json does not match its "
            f"registered hash.\n  registered {expected}\n  found      {actual}"
        )
    print(f"  pre-registration verified (sha256 {actual[:16]}...)")
    return json.loads(SUITE.read_text())


def exact_count(query: QueryHypergraph, data: dict, timeout_s: float) -> int | None:
    """Exact cycle join size via DuckDB; None on timeout."""
    con = duckdb.connect()
    k = len(query.relations)
    for r in query.relations:
        cols, rows = data[r.name]
        con.execute(
            f'CREATE TABLE "{r.name}" ("{cols[0]}" VARCHAR, "{cols[1]}" VARCHAR)'
        )
        con.executemany(f'INSERT INTO "{r.name}" VALUES (?, ?)', rows)
    rel = query.relations
    sql = (
        "SELECT count(*) FROM "
        + ", ".join(f'"{r.name}"' for r in rel)
        + " WHERE "
        + " AND ".join(
            f'"{rel[i].name}"."v{(i + 1) % k}" = "{rel[(i + 1) % k].name}"."v{(i + 1) % k}"'
            for i in range(k)
        )
    )
    out: list[int | None] = [None]

    def run() -> None:
        try:
            out[0] = int(con.execute(sql).fetchone()[0])
        except Exception:
            out[0] = None

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        con.interrupt()
        th.join()
        return None
    return out[0]


def main() -> None:
    print("Evaluating the pre-registered LONG-cycle suite (|V| = 6, 8, 10)")
    print("=" * 78)
    reg = verify_registration()
    queries = reg["queries"]
    print(f"  {len(queries)} registered queries; all will be attempted\n")

    rows_out: list[dict[str, object]] = []
    for q in queries:
        built = build_hetionet_query(q)
        row: dict[str, object] = {
            "query_id": q["query_id"],
            "n_vars": q["n_vars"],
            "relations": "|".join(q["relations"]),
        }
        if built is None:
            row["censored"] = "no_data"
            rows_out.append(row)
            continue
        query, data = built

        t0 = time.perf_counter()
        md = estimate_pessimistic(query, data, max_block_size=1).estimate
        t_md = time.perf_counter() - t0
        t0 = time.perf_counter()
        c1 = estimate_lph_ceg(query, data, max_block_size=1, p_values=P_VALUES).estimate
        t_c1 = time.perf_counter() - t0
        t0 = time.perf_counter()
        ent = estimate_lpbound_entropic(query, data, p_values=P_VALUES)
        t_ent = time.perf_counter() - t0
        t0 = time.perf_counter()
        c2 = estimate_lph_ceg(query, data, max_block_size=2, p_values=P_VALUES).estimate
        t_c2 = time.perf_counter() - t0
        truth = exact_count(query, data, EXACT_TIMEOUT_S)

        censored = []
        if t_c2 > BS2_TIMEOUT_S:
            censored.append("bs2_timeout")
        if truth is None:
            censored.append("exact_timeout")

        def valid(b: float, t: int | None = truth) -> str:
            return "" if t is None else str(b >= t - 1e-6)

        row.update(
            {
                "maxdeg": f"{md:.6e}",
                "cover_bs1": f"{c1:.6e}",
                "cover_bs2": f"{c2:.6e}",
                "entropic": f"{ent:.6e}",
                "true_count": "" if truth is None else truth,
                "bs1_over_entropic": f"{c1 / ent:.4f}",
                "bs2_over_entropic": f"{c2 / ent:.6f}",
                "bs2_beats_entropic": str(c2 < ent),
                "improvement_maxdeg_over_bs1": f"{md / c1:.4f}",
                "bs1_over_true": "" if not truth else f"{c1 / truth:.4e}",
                "bs2_over_true": "" if not truth else f"{c2 / truth:.4e}",
                "entropic_over_true": "" if not truth else f"{ent / truth:.4e}",
                "valid_bs1": valid(c1),
                "valid_bs2": valid(c2),
                "valid_entropic": valid(ent),
                "maxdeg_s": f"{t_md:.3f}",
                "cover_bs1_s": f"{t_c1:.3f}",
                "cover_bs2_s": f"{t_c2:.3f}",
                "entropic_s": f"{t_ent:.3f}",
                "censored": "+".join(censored),
            }
        )
        rows_out.append(row)
        print(
            f"  {q['query_id']} k={q['n_vars']:>2} bs1/ent={c1 / ent:>7.2f}x "
            f"bs2/ent={c2 / ent:>10.2e}x  t(bs1,ent,bs2)="
            f"{t_c1:.1f}/{t_ent:.1f}/{t_c2:.1f}s  true={truth}"
        )

    out = ROOT / "results" / "suite"
    out.mkdir(parents=True, exist_ok=True)
    keys = max(rows_out, key=len).keys()
    with open(out / "long_suite_eval.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(keys))
        w.writeheader()
        w.writerows(rows_out)

    summary: list[dict[str, object]] = []
    print("\n" + "=" * 78)
    for k in sorted({int(r["n_vars"]) for r in rows_out}):
        sel = [r for r in rows_out if int(r["n_vars"]) == k and "cover_bs1" in r]
        g1 = [float(r["bs1_over_entropic"]) for r in sel]
        g2 = [float(r["bs2_over_entropic"]) for r in sel]
        imp = [float(r["improvement_maxdeg_over_bs1"]) for r in sel]
        chk = [r for r in sel if r["true_count"] != ""]
        rec: dict[str, object] = {
            "n_vars": k,
            "n": len(sel),
            "bs1_over_entropic_median": f"{statistics.median(g1):.4f}",
            "bs1_over_entropic_max": f"{max(g1):.4f}",
            "bs2_over_entropic_median": f"{statistics.median(g2):.6f}",
            "bs2_beats_entropic": sum(
                1 for r in sel if r["bs2_beats_entropic"] == "True"
            ),
            "improvement_median": f"{statistics.median(imp):.4f}",
            "n_validity_checked": len(chk),
            "n_invalid": sum(
                1
                for r in chk
                for c in ("valid_bs1", "valid_bs2", "valid_entropic")
                if r[c] == "False"
            ),
            "cover_bs1_s_median": f"{statistics.median(float(r['cover_bs1_s']) for r in sel):.3f}",
            "entropic_s_median": f"{statistics.median(float(r['entropic_s']) for r in sel):.3f}",
            "cover_bs2_s_median": f"{statistics.median(float(r['cover_bs2_s']) for r in sel):.3f}",
            "n_censored": sum(1 for r in sel if r["censored"]),
        }
        summary.append(rec)
        print(
            f"  k={k:>2} n={len(sel)} bs1/ent median={rec['bs1_over_entropic_median']} "
            f"max={rec['bs1_over_entropic_max']} | bs2/ent median="
            f"{rec['bs2_over_entropic_median']} beats={rec['bs2_beats_entropic']}/{len(sel)} "
            f"| invalid={rec['n_invalid']}/{len(chk)} checked | "
            f"t bs1/ent/bs2={rec['cover_bs1_s_median']}/{rec['entropic_s_median']}/"
            f"{rec['cover_bs2_s_median']}s"
        )
    with open(out / "long_suite_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\nWrote {out / 'long_suite_eval.csv'} and long_suite_summary.csv")


if __name__ == "__main__":
    main()
