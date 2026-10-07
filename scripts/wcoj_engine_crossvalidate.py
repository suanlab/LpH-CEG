"""C3 support: cross-validate the WCOJ simulator against a real engine (Kuzu).

Our WCOJ end-to-end results (Section wcoj) run in a GenericJoin *simulator*.
A fair reviewer question is whether that simulator computes the right thing.
Off-the-shelf embedded engines (DuckDB, Kuzu) do not expose the per-variable
ordering hook we would need to *inject* our ordering, so a full real-engine
ordering experiment remains future work. But we can still validate the
simulator's *correctness* against an independent, WCOJ-capable engine:
Kuzu (an embedded graph DBMS) evaluates the same cyclic patterns, and its
result counts must match ours exactly.

This does not claim a runtime win (Kuzu is a compiled C++ engine; our
simulator is Python). It establishes that the tuple-level result our
simulator counts is the same relation the engine materializes---so the
deterministic iteration metric we report is measuring real join work.

Requires: kuzu (pip install kuzu). Output: results/wcoj/engine_crossval.csv
"""

from __future__ import annotations

import csv
import os
import shutil
import tempfile
from pathlib import Path

from scripts.wcoj_e2e_experiment import generate_power_law_edges, generic_join
from src.utils.query import triangle_query
from src.wcoj.evaluation import lexicographic_ordering


def kuzu_triangle_count(edges: list[tuple[str, str]]) -> int:
    """Count directed triangles a->b->c->a in Kuzu on the given edge set."""
    import kuzu

    d = tempfile.mkdtemp()
    try:
        con = kuzu.Connection(kuzu.Database(os.path.join(d, "g")))
        con.execute("CREATE NODE TABLE N(id INT64, PRIMARY KEY(id))")
        con.execute("CREATE REL TABLE E(FROM N TO N)")
        ids = {int(x) for e in edges for x in e}
        for i in ids:
            con.execute(f"CREATE (:N {{id:{i}}})")
        for a, b in edges:
            con.execute(
                f"MATCH (u:N),(v:N) WHERE u.id={int(a)} AND v.id={int(b)} "
                f"CREATE (u)-[:E]->(v)"
            )
        q = "MATCH (a:N)-[:E]->(b:N)-[:E]->(c:N)-[:E]->(a) RETURN count(*)"
        return int(con.execute(q).get_as_df().iloc[0, 0])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def simulator_triangle_count(edges: list[tuple[str, str]]) -> int:
    q = triangle_query()
    data = {
        "R": (("a", "b"), edges),
        "S": (("b", "c"), edges),
        "T": (("a", "c"), [(y, x) for x, y in edges]),
    }
    return generic_join(q, data, lexicographic_ordering(q)).result_count


def main() -> None:
    try:
        import kuzu  # noqa: F401
    except ImportError:
        print("[kuzu not installed; skipping engine cross-validation]")
        return

    print("C3: WCOJ simulator vs Kuzu (real engine) -- triangle count agreement")
    print("=" * 68)
    print(
        f"  {'nodes':>6} {'edges':>6} {'seed':>5} {'simulator':>10} "
        f"{'kuzu':>8} {'match':>6}"
    )
    print(f"  {'-' * 52}")

    rows: list[dict[str, object]] = []
    all_match = True
    for n_v, n_e, seed in [
        (40, 120, 1),
        (50, 200, 2),
        (60, 300, 3),
        (80, 400, 7),
    ]:
        edges = generate_power_law_edges(n_v, n_e, seed=seed)
        sim = simulator_triangle_count(edges)
        kz = kuzu_triangle_count(edges)
        match = sim == kz
        all_match = all_match and match
        print(f"  {n_v:>6} {n_e:>6} {seed:>5} {sim:>10} {kz:>8} {match!s:>6}")
        rows.append(
            {
                "n_vertices": n_v,
                "n_edges": n_e,
                "seed": seed,
                "simulator_count": sim,
                "kuzu_count": kz,
                "match": match,
            }
        )

    out = (
        Path(__file__).resolve().parent.parent
        / "results"
        / "wcoj"
        / "engine_crossval.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "n_vertices",
                "n_edges",
                "seed",
                "simulator_count",
                "kuzu_count",
                "match",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(
        f"\n  {'ALL MATCH' if all_match else 'MISMATCH DETECTED'} ({len(rows)} configs)"
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
