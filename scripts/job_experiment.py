"""LpH-CEG experiments on JOB (Join Order Benchmark) queries.

Uses JOB-medium data with cyclic join patterns from the IMDb schema:
- title ⋈ cast_info ⋈ movie_keyword ⋈ movie_companies (4-way cycle via title_id)
- title ⋈ cast_info ⋈ movie_info ⋈ movie_keyword (4-way cycle via title_id)
- Various star + cycle patterns

JOB's schema has title_id as the central hub, creating many cyclic
joins when combining multiple fact tables.
"""

from __future__ import annotations

import csv
import os
import time
from collections import defaultdict
from pathlib import Path

from src.estimators.lph_ceg import estimate_lph_ceg
from src.estimators.pessimistic import estimate_pessimistic
from src.utils.types import QueryHypergraph, Relation

# Repo-relative path (data/ holds the dataset or a symlink to it).
# Override the data root with the LPH_DATA_DIR environment variable.
_DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data"
_DATA_ROOT = Path(os.environ.get("LPH_DATA_DIR", _DEFAULT_DATA))
DATA_DIR = _DATA_ROOT / "job_medium"


def load_csv_relation(
    filename: str,
    columns: list[str],
    attr_names: list[str],
    max_rows: int | None = None,
) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
    """Load specific columns from a CSV as a relation."""
    path = DATA_DIR / filename
    tuples: list[tuple[str, ...]] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t = tuple(row[c] for c in columns)
                if all(v for v in t):  # skip nulls
                    tuples.append(t)
            except KeyError:
                continue
            if max_rows and len(tuples) >= max_rows:
                break
    return tuple(attr_names), tuples


def count_join(
    query: QueryHypergraph,
    data: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]],
) -> int:
    """Brute-force join to get true cardinality."""
    rels = list(query.relations)
    if not rels:
        return 0

    first = rels[0]
    attrs_0, tuples_0 = data[first.name]
    current = [dict(zip(attrs_0, t, strict=False)) for t in tuples_0]

    for rel in rels[1:]:
        attrs_r, tuples_r = data[rel.name]
        shared = set(current[0].keys()) & set(attrs_r) if current else set()
        shared_sorted = sorted(shared)

        idx: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
        for t in tuples_r:
            d = dict(zip(attrs_r, t, strict=False))
            key = tuple(d.get(a, "") for a in shared_sorted)
            idx[key].append(d)

        joined: list[dict[str, str]] = []
        for ct in current:
            key = tuple(ct.get(a, "") for a in shared_sorted)
            for nt in idx.get(key, []):
                joined.append({**ct, **nt})
        current = joined

    return len(current)


def run_query(
    name: str,
    query: QueryHypergraph,
    data: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]],
) -> dict[str, object]:
    """Run H-CEG vs LpH-CEG comparison with true cardinality."""
    true_count = count_join(query, data)

    t0 = time.perf_counter()
    std = estimate_pessimistic(query, data, max_block_size=1)
    t_std = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    lph = estimate_lph_ceg(
        query,
        data,
        max_block_size=1,
        p_values=(1.0, 2.0, float("inf")),
    )
    t_lph = (time.perf_counter() - t0) * 1000

    ratio = std.estimate / max(lph.estimate, 1e-10)

    def qerror(est: float, true: int) -> float:
        if true == 0:
            return float("inf") if est > 0 else 1.0
        return max(est / true, true / max(est, 1e-10))

    q_std = qerror(std.estimate, true_count)
    q_lph = qerror(lph.estimate, true_count)

    print(f"\n  [{name}]")
    sizes = {r: len(data[r][1]) for r in data}
    print(f"    Relations: {sizes}")
    print(f"    True count:  {true_count:>12}")
    print(
        f"    H-CEG:       {std.estimate:>12.0f}  q-error: {q_std:>8.1f}x  ({t_std:.0f}ms)"
    )
    print(
        f"    LpH-CEG:     {lph.estimate:>12.0f}  q-error: {q_lph:>8.1f}x  ({t_lph:.0f}ms)"
    )
    print(f"    LP ratio:    {ratio:.2f}x")
    print(f"    LP p-used:   {lph.metadata.get('lp_p_values_used', [])}")

    return {
        "name": name,
        "true": true_count,
        "hceg": std.estimate,
        "lph": lph.estimate,
        "q_hceg": q_std,
        "q_lph": q_lph,
        "ratio": ratio,
    }


def main() -> None:
    print("=" * 70)
    print("LpH-CEG on JOB-medium (IMDb) Queries")
    print("=" * 70)

    # Load relations (capped to avoid OOM on brute-force join)
    max_rows = 500
    ci = load_csv_relation(
        "cast_info.csv",
        ["title_id", "person_id"],
        ["title_id", "person_id"],
        max_rows=max_rows,
    )
    mk = load_csv_relation(
        "movie_keyword.csv",
        ["title_id", "keyword_id"],
        ["title_id", "keyword_id"],
        max_rows=max_rows,
    )
    mc = load_csv_relation(
        "movie_companies.csv",
        ["title_id", "company_id"],
        ["title_id", "company_id"],
        max_rows=max_rows,
    )
    mi = load_csv_relation(
        "movie_info.csv",
        ["title_id", "info_type_id"],
        ["title_id", "info_type_id"],
        max_rows=max_rows,
    )
    t = load_csv_relation(
        "title.csv",
        ["title_id", "production_year"],
        ["title_id", "production_year"],
        max_rows=max_rows,
    )

    results: list[dict[str, object]] = []

    # ---- Query 1: Cycle-4 via title_id ----
    # cast_info(title_id, person_id) ⋈ movie_keyword(title_id, keyword_id)
    # ⋈ movie_companies(title_id, company_id) ⋈ movie_info(title_id, info_type_id)
    # This is actually a star on title_id, not a cycle.
    # To create real cycles, we need different join keys.

    # ---- True cycle: co-actor pattern ----
    # CI1(tid1, pid) ⋈ CI2(tid2, pid) ⋈ MK1(tid1, kid) ⋈ MK2(tid2, kid)
    # Two movies share an actor AND a keyword = cycle on (pid, kid)
    ci_data = ci[1]
    mk_data = mk[1]

    q1 = QueryHypergraph(
        relations=(
            Relation("CI1", frozenset({"tid1", "pid"}), len(ci_data)),
            Relation("CI2", frozenset({"tid2", "pid"}), len(ci_data)),
            Relation("MK1", frozenset({"tid1", "kid"}), len(mk_data)),
            Relation("MK2", frozenset({"tid2", "kid"}), len(mk_data)),
        ),
        name="job_coactor_cokeyword",
    )
    data_q1 = {
        "CI1": (("tid1", "pid"), ci_data),
        "CI2": (("tid2", "pid"), ci_data),
        "MK1": (("tid1", "kid"), mk_data),
        "MK2": (("tid2", "kid"), mk_data),
    }
    results.append(run_query("JOB Q1: Co-actor + Co-keyword (cycle-4)", q1, data_q1))

    # ---- Query 2: Triangle ----
    # CI(tid, pid) ⋈ MK(tid, kid) ⋈ MC(tid, cid)
    # Star on title_id — control (should show no LP improvement)
    q2 = QueryHypergraph(
        relations=(
            Relation("CI", frozenset({"tid", "pid"}), len(ci_data)),
            Relation("MK", frozenset({"tid", "kid"}), len(mk_data)),
            Relation("MC", frozenset({"tid", "cid"}), len(mc[1])),
        ),
        name="job_star3",
    )
    data_q2 = {
        "CI": (("tid", "pid"), ci_data),
        "MK": (("tid", "kid"), mk_data),
        "MC": (("tid", "cid"), mc[1]),
    }
    results.append(run_query("JOB Q2: Star-3 (CI-MK-MC on tid, control)", q2, data_q2))

    # ---- Query 3: Larger cycle ----
    # CI1(tid1, pid) ⋈ CI2(tid2, pid) ⋈ MC1(tid1, cid) ⋈ MC2(tid2, cid)
    # Co-actor + co-company cycle
    mc_data = mc[1]
    q3 = QueryHypergraph(
        relations=(
            Relation("CI1", frozenset({"tid1", "pid"}), len(ci_data)),
            Relation("CI2", frozenset({"tid2", "pid"}), len(ci_data)),
            Relation("MC1", frozenset({"tid1", "cid"}), len(mc_data)),
            Relation("MC2", frozenset({"tid2", "cid"}), len(mc_data)),
        ),
        name="job_coactor_cocompany",
    )
    data_q3 = {
        "CI1": (("tid1", "pid"), ci_data),
        "CI2": (("tid2", "pid"), ci_data),
        "MC1": (("tid1", "cid"), mc_data),
        "MC2": (("tid2", "cid"), mc_data),
    }
    results.append(run_query("JOB Q3: Co-actor + Co-company (cycle-4)", q3, data_q3))

    # ---- Query 4: 5-way cycle ----
    # CI1(tid1, pid) ⋈ CI2(tid2, pid) ⋈ MK1(tid1, kid) ⋈ MK2(tid2, kid) ⋈ T(tid1, year)
    # Adds title info to the co-actor+co-keyword cycle
    t_data = t[1]
    q4 = QueryHypergraph(
        relations=(
            Relation("CI1", frozenset({"tid1", "pid"}), len(ci_data)),
            Relation("CI2", frozenset({"tid2", "pid"}), len(ci_data)),
            Relation("MK1", frozenset({"tid1", "kid"}), len(mk_data)),
            Relation("MK2", frozenset({"tid2", "kid"}), len(mk_data)),
            Relation("T", frozenset({"tid1", "year"}), len(t_data)),
        ),
        name="job_coactor_cokeyword_year",
    )
    data_q4 = {
        "CI1": (("tid1", "pid"), ci_data),
        "CI2": (("tid2", "pid"), ci_data),
        "MK1": (("tid1", "kid"), mk_data),
        "MK2": (("tid2", "kid"), mk_data),
        "T": (("tid1", "year"), t_data),
    }
    results.append(run_query("JOB Q4: Co-actor+Co-keyword+Year (5-way)", q4, data_q4))

    # ---- Query 5: Heterogeneous cycle ----
    # CI(tid, pid) ⋈ MI(tid, itype) ⋈ MK(tid, kid) ⋈ MC(tid, cid)
    # 4-way star — another control
    mi_data = mi[1]
    q5 = QueryHypergraph(
        relations=(
            Relation("CI", frozenset({"tid", "pid"}), len(ci_data)),
            Relation("MI", frozenset({"tid", "itype"}), len(mi_data)),
            Relation("MK", frozenset({"tid", "kid"}), len(mk_data)),
            Relation("MC", frozenset({"tid", "cid"}), len(mc_data)),
        ),
        name="job_star4",
    )
    data_q5 = {
        "CI": (("tid", "pid"), ci_data),
        "MI": (("tid", "itype"), mi_data),
        "MK": (("tid", "kid"), mk_data),
        "MC": (("tid", "cid"), mc_data),
    }
    results.append(
        run_query("JOB Q5: Star-4 (CI-MI-MK-MC on tid, control)", q5, data_q5)
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"\n{'Name':<50} {'True':>8} {'q(H-CEG)':>10} {'q(LpH)':>10} {'LP ratio':>10}"
    )
    print("-" * 90)
    for r in results:
        print(
            f"{r['name']:<50} {r['true']:>8} {r['q_hceg']:>10.1f}x {r['q_lph']:>10.1f}x {r['ratio']:>9.2f}x"
        )


if __name__ == "__main__":
    main()
