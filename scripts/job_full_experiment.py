"""JOB-full experiment: LpH-CEG on full-scale IMDb data.

Uses the full IMDb dataset (1.76M movie-actor edges, 345K movie-director,
50K co-actor edges) to evaluate bound estimation at realistic scale.

Brute-force join is only used for verification on small samples.
The main experiment measures bounds (CEG, Cover-LP(bs1), H-CEG, LpH-CEG)
at multiple data scales without requiring true cardinality.  "Cover-LP(bs1)"
is the fractional block-cover LP proxy (LpH-CEG at bs=1), not the genuine
entropic LpBound (see src/estimators/lpbound_entropic.py).
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

from src.estimators.lpbound_baseline import estimate_lpbound
from src.estimators.lph_ceg import estimate_lph_ceg
from src.estimators.pessimistic import estimate_pessimistic
from src.utils.types import QueryHypergraph, Relation

DATA_DIR = Path("data/imdb")
JOB_DIR = Path("data/job_medium")

P_VALUES: tuple[float, ...] = (1.0, 2.0, float("inf"))


def load_tsv_edges(
    path: Path,
    max_rows: int | None = None,
) -> list[tuple[str, str]]:
    """Load edge list from TSV file (skip header)."""
    edges: list[tuple[str, str]] = []
    with open(path) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                edges.append((parts[0], parts[1]))
                if max_rows and len(edges) >= max_rows:
                    break
    return edges


def load_csv_relation(
    path: Path,
    columns: list[str],
    attr_names: list[str],
    max_rows: int | None = None,
) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
    """Load specific columns from a CSV as a relation."""
    tuples: list[tuple[str, ...]] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t = tuple(row[c] for c in columns)
                if all(v for v in t):
                    tuples.append(t)
            except KeyError:
                continue
            if max_rows and len(tuples) >= max_rows:
                break
    return tuple(attr_names), tuples


def run_4way(
    name: str,
    query: QueryHypergraph,
    data: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]],
    include_bs2: bool = False,
) -> dict[str, object]:
    """Run 4-way comparison (CEG, LpBound, H-CEG, LpH-CEG)."""
    sizes = {r: len(data[r][1]) for r in data}

    t0 = time.perf_counter()
    ceg = estimate_pessimistic(query, data, max_block_size=1)
    t_ceg = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    lpb = estimate_lpbound(query, data, p_values=P_VALUES)
    t_lpb = (time.perf_counter() - t0) * 1000

    lp_ratio = ceg.estimate / max(lpb.estimate, 1e-10)

    print(f"\n  [{name}]")
    print(f"    Relations: {sizes}")
    print(f"    CEG_M  (bs=1):      {ceg.estimate:>15.0f}  ({t_ceg:.0f}ms)")
    print(
        f"    Cover-LP(bs1):      {lpb.estimate:>15.0f}  ({t_lpb:.0f}ms)  LP ratio: {lp_ratio:.2f}x"
    )

    result: dict[str, object] = {
        "name": name,
        "sizes": sizes,
        "ceg": ceg.estimate,
        "lpbound": lpb.estimate,
        "lp_ratio": lp_ratio,
        "t_ceg_ms": t_ceg,
        "t_lpbound_ms": t_lpb,
        "p_used": list(lpb.metadata.get("lp_p_values_used", [])),
    }

    if include_bs2:
        t0 = time.perf_counter()
        hceg = estimate_pessimistic(query, data, max_block_size=2)
        t_hceg = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        lph = estimate_lph_ceg(query, data, max_block_size=2, p_values=P_VALUES)
        t_lph = (time.perf_counter() - t0) * 1000

        hceg_ratio = ceg.estimate / max(hceg.estimate, 1e-10)
        lph_ratio = ceg.estimate / max(lph.estimate, 1e-10)

        print(
            f"    H-CEG  (bs=2):      {hceg.estimate:>15.0f}  ({t_hceg:.0f}ms)  ratio: {hceg_ratio:.2f}x"
        )
        print(
            f"    LpH-CEG(bs=2,LP):   {lph.estimate:>15.0f}  ({t_lph:.0f}ms)  ratio: {lph_ratio:.2f}x"
        )

        result["hceg"] = hceg.estimate
        result["lph_ceg"] = lph.estimate
        result["hceg_ratio"] = hceg_ratio
        result["lph_ratio"] = lph_ratio

    return result


def main() -> None:
    print("=" * 70)
    print("JOB-Full Experiment: LpH-CEG on Full-Scale IMDb Data")
    print("=" * 70)

    results: list[dict[str, object]] = []

    # =================================================================
    # Part 1: Co-actor cycle patterns (full IMDb)
    # =================================================================
    print("\n" + "=" * 70)
    print("PART 1: Co-Actor Cycle Patterns (Full IMDb)")
    print("=" * 70)

    # movie_actor: (movie_id, actor_id) — 1.76M rows
    # Co-actor cycle-4: MA1(m1,a) ⋈ MA2(m2,a) ⋈ MD1(m1,d) ⋈ MD2(m2,d)
    # Two movies share an actor AND a director

    for max_rows in [5000, 20000, 50000, 100000]:
        ma = load_tsv_edges(DATA_DIR / "movie_actor.tsv", max_rows=max_rows)
        md = load_tsv_edges(DATA_DIR / "movie_director.tsv", max_rows=max_rows)

        q_cycle4 = QueryHypergraph(
            relations=(
                Relation("MA1", frozenset({"m1", "actor"}), len(ma)),
                Relation("MA2", frozenset({"m2", "actor"}), len(ma)),
                Relation("MD1", frozenset({"m1", "director"}), len(md)),
                Relation("MD2", frozenset({"m2", "director"}), len(md)),
            ),
            name="coactor_codirector_c4",
        )
        data_c4 = {
            "MA1": (("m1", "actor"), ma),
            "MA2": (("m2", "actor"), ma),
            "MD1": (("m1", "director"), md),
            "MD2": (("m2", "director"), md),
        }
        results.append(
            run_4way(
                f"Co-actor+Co-director cycle-4 ({max_rows} rows)",
                q_cycle4,
                data_c4,
                include_bs2=(max_rows <= 10000),
            )
        )

    # =================================================================
    # Part 2: Co-actor + keyword cycle (full IMDb style)
    # =================================================================
    print("\n" + "=" * 70)
    print("PART 2: Co-Actor + Co-Keyword Cycles (JOB-medium)")
    print("=" * 70)

    # Use JOB-medium data for queries with true cardinality
    ci = load_csv_relation(
        JOB_DIR / "cast_info.csv",
        ["title_id", "person_id"],
        ["tid", "pid"],
    )
    mk = load_csv_relation(
        JOB_DIR / "movie_keyword.csv",
        ["title_id", "keyword_id"],
        ["tid", "kid"],
    )
    mc = load_csv_relation(
        JOB_DIR / "movie_companies.csv",
        ["title_id", "company_id"],
        ["tid", "cid"],
    )

    # Q1: Co-actor + co-keyword cycle-4
    q1 = QueryHypergraph(
        relations=(
            Relation("CI1", frozenset({"tid1", "pid"}), len(ci[1])),
            Relation("CI2", frozenset({"tid2", "pid"}), len(ci[1])),
            Relation("MK1", frozenset({"tid1", "kid"}), len(mk[1])),
            Relation("MK2", frozenset({"tid2", "kid"}), len(mk[1])),
        ),
        name="job_coactor_cokeyword",
    )
    data_q1 = {
        "CI1": (("tid1", "pid"), ci[1]),
        "CI2": (("tid2", "pid"), ci[1]),
        "MK1": (("tid1", "kid"), mk[1]),
        "MK2": (("tid2", "kid"), mk[1]),
    }
    results.append(run_4way("JOB Co-actor+Co-keyword (full JOB-medium)", q1, data_q1))

    # Q2: Co-actor + co-company cycle-4
    q2 = QueryHypergraph(
        relations=(
            Relation("CI1", frozenset({"tid1", "pid"}), len(ci[1])),
            Relation("CI2", frozenset({"tid2", "pid"}), len(ci[1])),
            Relation("MC1", frozenset({"tid1", "cid"}), len(mc[1])),
            Relation("MC2", frozenset({"tid2", "cid"}), len(mc[1])),
        ),
        name="job_coactor_cocompany",
    )
    data_q2 = {
        "CI1": (("tid1", "pid"), ci[1]),
        "CI2": (("tid2", "pid"), ci[1]),
        "MC1": (("tid1", "cid"), mc[1]),
        "MC2": (("tid2", "cid"), mc[1]),
    }
    results.append(run_4way("JOB Co-actor+Co-company (full JOB-medium)", q2, data_q2))

    # Q3: Star control — CI ⋈ MK ⋈ MC on tid (no cycle)
    q3 = QueryHypergraph(
        relations=(
            Relation("CI", frozenset({"tid", "pid"}), len(ci[1])),
            Relation("MK", frozenset({"tid", "kid"}), len(mk[1])),
            Relation("MC", frozenset({"tid", "cid"}), len(mc[1])),
        ),
        name="job_star3_control",
    )
    data_q3 = {
        "CI": (("tid", "pid"), ci[1]),
        "MK": (("tid", "kid"), mk[1]),
        "MC": (("tid", "cid"), mc[1]),
    }
    results.append(run_4way("JOB Star-3 control (no cycle)", q3, data_q3))

    # =================================================================
    # Part 3: Full IMDb co-actor patterns at scale
    # =================================================================
    print("\n" + "=" * 70)
    print("PART 3: Full IMDb Scale (movie_actor 50K-1M)")
    print("=" * 70)

    for max_rows in [50000, 200000, 500000, 1000000]:
        ma = load_tsv_edges(DATA_DIR / "movie_actor.tsv", max_rows=max_rows)

        # Replicated-edge cycle-4 (same relation on all edges)
        q_rep = QueryHypergraph(
            relations=(
                Relation("R1", frozenset({"a1", "a2"}), len(ma)),
                Relation("R2", frozenset({"a2", "a3"}), len(ma)),
                Relation("R3", frozenset({"a3", "a4"}), len(ma)),
                Relation("R4", frozenset({"a4", "a1"}), len(ma)),
            ),
            name="imdb_replicated_c4",
        )
        data_rep = {
            "R1": (("a1", "a2"), ma),
            "R2": (("a2", "a3"), ma),
            "R3": (("a3", "a4"), ma),
            "R4": (("a4", "a1"), ma),
        }
        results.append(
            run_4way(
                f"IMDb replicated cycle-4 ({max_rows // 1000}K MA rows)",
                q_rep,
                data_rep,
            )
        )

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Name':<55} {'CEG':>12} {'Cover-LP(bs1)':>14} {'LP ratio':>10}")
    print("-" * 92)
    for r in results:
        print(
            f"  {r['name']:<53} {r['ceg']:>12.0f} {r['lpbound']:>12.0f} {r['lp_ratio']:>9.2f}x"
        )

    # Export CSV
    out = Path("results/large_scale/job_full.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["name", "ceg", "cover_lp_bs1", "lp_ratio", "t_ceg_ms", "t_cover_lp_bs1_ms"]
        )
        for r in results:
            writer.writerow(
                [
                    r["name"],
                    f"{r['ceg']:.0f}",
                    f"{r['lpbound']:.0f}",
                    f"{r['lp_ratio']:.4f}",
                    f"{r['t_ceg_ms']:.1f}",
                    f"{r['t_lpbound_ms']:.1f}",
                ]
            )
    print(f"\nResults exported to {out}")


if __name__ == "__main__":
    main()
