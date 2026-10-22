"""Does the block frontier that brackets LpBound survive on real data?

`statistics_hierarchy.py` found, on synthetic instances, that bs>=2 block
statistics produce bounds TIGHTER than the entropic LpBound -- because a
block statistic summarises a materialised multi-relation join, which no
single-relation degree sequence contains.  That inverts the paper's
defensive framing, so it has to be checked where the paper actually lives:
the pre-registered Hetionet suite.

Subset rule, fixed before measuring: every registered Hetionet query with
n_vars <= 4, i.e. all 28 cycle-3 and all 81 cycle-4 queries.  Nothing is
hand-picked, and the registry hash is verified first.

Scale caveat, declared up front: bs=2 materialises two-relation joins, so
relations are capped at ROW_CAP rows rather than the suite's 20,000.  The
bounds here are therefore NOT comparable to `suite_eval.csv`; only the
*relationships between estimators on identical data* are.

Result: bs=1 sits at a median 2.58x the entropic bound (looser, as the
paper says), while bs=2 sits at 0.057x -- tighter on 109/109 queries.  The
N_VALIDATE tightest bounds are then checked against exact join sizes, since
a 17x improvement is exactly the kind of number that should be a bug; none
is violated.

Read this with the same correction as the synthetic result
(`statistics_vs_inference.py`): the comparison is cover-on-blocks against
entropic-on-single-relations, so it says the statistics matter more than
the inference, NOT that we beat LpBound's method.  Note also that these
bounds remain far from the truth in absolute terms -- true counts here are
0-70 while bs=2 returns 291-827 -- so what moved is the ordering, not the
arrival at exactness.

Output: results/ablation/real_frontier.csv (and real_frontier_skipped.csv)
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from src.estimators.lpbound_entropic import estimate_lpbound_entropic
from src.estimators.lph_ceg import estimate_lph_ceg
from src.utils.types import QueryHypergraph, Relation

ROOT = Path(__file__).resolve().parent.parent
SUITE = ROOT / "benchmarks" / "query_suite_v1.json"
DIGEST = ROOT / "benchmarks" / "query_suite_v1.sha256"
HETIONET = ROOT / "data" / "hetionet"
P_VALUES: tuple[float, ...] = (1.0, 2.0, float("inf"))
ROW_CAP = 3000
MAX_VARS = 4
N_VALIDATE = 20  # tightest bounds to check against exact join sizes

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

_cache: dict[str, list[tuple[str, str]]] = {}


def load(rel: str) -> list[tuple[str, str]]:
    if rel in _cache:
        return _cache[rel]
    rows: list[tuple[str, str]] = []
    with open(HETIONET / HETIONET_FILES[rel]) as fh:
        fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                rows.append((parts[0], parts[1]))
                if len(rows) >= ROW_CAP:
                    break
    _cache[rel] = rows
    return rows


def build(q: dict):
    from scripts.register_query_suite import HETIONET_RELATIONS

    rels_names, var_types = q["relations"], q["var_types"]
    k = len(rels_names)
    rels, data = [], {}
    for i, rname in enumerate(rels_names):
        native = HETIONET_RELATIONS[rname]
        want = (var_types[i], var_types[(i + 1) % k])
        rows = load(rname)
        if native == want:
            oriented = rows
        elif native == (want[1], want[0]):
            oriented = [(b, a) for a, b in rows]
        else:
            return None
        u, v = f"v{i}", f"v{(i + 1) % k}"
        attrs = tuple(sorted((u, v)))
        swap = attrs != (u, v)
        name = f"R{i}_{rname}"
        rels.append(Relation(name, frozenset({u, v}), len(oriented)))
        data[name] = (attrs, [(y, x) for x, y in oriented] if swap else oriented)
    return QueryHypergraph(relations=tuple(rels), name=q["query_id"]), data


def exact_count(query: QueryHypergraph, data: dict, cap: int = 5_000_000) -> int:
    """Exact join size via hash joins; -1 if it exceeds `cap`."""
    import pandas as pd

    df = None
    for r in query.relations:
        cols, rows = data[r.name]
        d = pd.DataFrame(rows, columns=list(cols)).drop_duplicates()
        if df is None:
            df = d
        else:
            shared = [c for c in d.columns if c in df.columns]
            df = df.merge(d, on=shared) if shared else df.merge(d, how="cross")
        if len(df) > cap:
            return -1
    return 0 if df is None else len(df)


def main() -> None:
    print("Does the block frontier bracket LpBound on real Hetionet queries?")
    print("=" * 78)
    expected = DIGEST.read_text().split()[0]
    actual = hashlib.sha256(SUITE.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit("REFUSING: query suite does not match its registered hash")
    print(f"  registry verified ({actual[:16]}...); rows capped at {ROW_CAP}")

    reg = json.loads(SUITE.read_text())
    qs = [
        q
        for q in reg["queries"]
        if q["dataset"] == "hetionet" and q["n_vars"] <= MAX_VARS
    ]
    print(f"  {len(qs)} queries with n_vars <= {MAX_VARS} (rule fixed in advance)\n")

    out: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for i, q in enumerate(qs):
        built = build(q)
        if built is None:
            skipped.append({"query_id": q["query_id"], "reason": "no_data"})
            continue
        query, data = built
        try:
            b1 = estimate_lph_ceg(
                query, data, max_block_size=1, p_values=P_VALUES
            ).estimate
            b2 = estimate_lph_ceg(
                query, data, max_block_size=2, p_values=P_VALUES
            ).estimate
            ent = estimate_lpbound_entropic(query, data, p_values=P_VALUES)
        except Exception as exc:
            print(f"  {q['query_id']}: skipped ({type(exc).__name__})")
            skipped.append({"query_id": q["query_id"], "reason": type(exc).__name__})
            continue
        out.append(
            {
                "query_id": q["query_id"],
                "shape": q["shape"],
                "relations": "|".join(q["relations"]),
                "bs1": f"{b1:.6e}",
                "bs2": f"{b2:.6e}",
                "entropic": f"{ent:.6e}",
                "bs1_over_entropic": f"{b1 / max(ent, 1e-9):.4f}",
                "bs2_over_entropic": f"{b2 / max(ent, 1e-9):.4f}",
                "bs2_beats_entropic": b2 < ent,
            }
        )
        if (i + 1) % 25 == 0:
            print(f"  ... {i + 1}/{len(qs)}")

    import statistics as st

    r1 = [float(r["bs1_over_entropic"]) for r in out]
    r2 = [float(r["bs2_over_entropic"]) for r in out]
    beat = sum(1 for r in out if r["bs2_beats_entropic"])
    print(f"\n  evaluated {len(out)} queries, skipped {len(skipped)}")
    print(f"  bs=1 / entropic : median {st.median(r1):.3f}x   (>1 = looser)")
    print(f"  bs=2 / entropic : median {st.median(r2):.3f}x   min {min(r2):.3f}x")
    print(
        f"  bs=2 strictly tighter than entropic on {beat}/{len(out)} "
        f"({beat / len(out) * 100:.1f}%)"
    )

    # Validity: check the tightest bounds, where a bug would show first.
    order = sorted(out, key=lambda r: float(r["bs2"]))[:N_VALIDATE]
    reg_by_id = {q["query_id"]: q for q in reg["queries"]}
    checked = violations = 0
    for r in order:
        built = build(reg_by_id[str(r["query_id"])])
        if built is None:
            continue
        t = exact_count(*built)
        if t < 0:
            continue
        checked += 1
        if float(r["bs2"]) < t - 1e-6 or float(r["bs1"]) < t - 1e-6:
            violations += 1
            print(f"  *** VIOLATION on {r['query_id']}: true={t} bs2={r['bs2']}")
    print(f"  validity: {checked} tightest bounds checked, {violations} violations")

    d = ROOT / "results" / "ablation"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "real_frontier.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    with open(d / "real_frontier_skipped.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "reason"])
        w.writeheader()
        w.writerows(skipped)
    print(f"\nWrote {d / 'real_frontier.csv'} and real_frontier_skipped.csv")


if __name__ == "__main__":
    main()
