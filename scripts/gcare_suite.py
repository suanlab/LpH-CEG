"""G-CARE cyclic queries: the paper's comparisons on an external benchmark.

G-CARE (Park et al., SIGMOD 2020) is the standard benchmark for subgraph
cardinality estimation.  Its query sets were fixed by its authors, so there
is no selection on our side.  The rule: every AIDS and YAGO query whose
undirected query graph contains a cycle (|E| >= |V| or a repeated vertex
pair) and has at least three vertices -- 241 AIDS and 366 YAGO queries.

Each query edge (a, b, l) becomes a binary relation: the data edges with
label l whose endpoints carry the query vertices' labels (and fixed ids,
where given).  Statistics are exact on these relations.  Queries with a
relation above MAX_ROWS rows are reported as ``too_large`` -- the block
statistics are collected by a Python prototype -- not dropped.

Per query: the entropic bound (bs=1 flow LP = the 2^|V| program), the cover
LP at bs=1 and bs=2 and the flow LP over the bs<=2 catalog, each with ten
norms ({1..10, inf}) and three ({1, 2, inf}); the exact homomorphism count
(DuckDB, EXACT_TIMEOUT_S); G-CARE's own true cardinality.

Outputs: results/gcare/gcare_eval.csv and gcare_summary.csv
"""

from __future__ import annotations

import collections
import csv
import itertools
import math
import multiprocessing
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import duckdb

from src.core.block import enumerate_blocks
from src.estimators.flow_bound import (
    block_constraints,
    flow_bound,
    flow_bound_over_orders,
)
from src.estimators.lph_ceg import estimate_lph_ceg
from src.statistics.lp_stats import compute_all_lp_block_extensions
from src.utils.types import QueryHypergraph, Relation

ROOT = Path(__file__).resolve().parent.parent
GCARE = Path(os.environ.get("LPH_GCARE_DIR", str(ROOT / "data" / "gcare")))
OUT = ROOT / "results" / "gcare"
DATASETS = ("aids", "yago")
MAX_ROWS = 300_000
#: Largest two-relation join the Python prototype materializes for block
#: statistics; above it the query is reported as ``pair_join_too_large``.
MAX_PAIR_JOIN = 20_000_000
#: Address-space limit per worker process.
MEMORY_LIMIT_BYTES = 24 * 2**30
EXACT_TIMEOUT_S = 60.0
WORKERS = 10
DUCKDB_THREADS = 4
P3 = (1.0, 2.0, math.inf)
P11 = (*(float(p) for p in range(1, 11)), math.inf)


def parse(path: Path) -> tuple[dict[int, tuple[int, int]], list[tuple[int, int, int]]]:
    verts: dict[int, tuple[int, int]] = {}
    edges: list[tuple[int, int, int]] = []
    for line in path.read_text().splitlines():
        t = line.split()
        if t and t[0] == "v":
            verts[int(t[1])] = (int(t[2]), int(t[3]))
        elif t and t[0] == "e":
            edges.append((int(t[1]), int(t[2]), int(t[3])))
    return verts, edges


def is_cyclic(verts: dict, edges: list) -> bool:  # type: ignore[type-arg]
    pairs = collections.Counter(frozenset((a, b)) for a, b, _ in edges)
    return len(verts) >= 3 and (
        len(edges) >= len(verts) or any(c > 1 for c in pairs.values())
    )


def registry() -> list[dict[str, object]]:
    """All cyclic AIDS/YAGO queries, in a fixed (path-sorted) order."""
    out: list[dict[str, object]] = []
    for ds in DATASETS:
        for path in sorted((GCARE / "queries" / ds).rglob("*.txt")):
            verts, edges = parse(path)
            if is_cyclic(verts, edges):
                rel = path.relative_to(GCARE / "queries")
                out.append(
                    {
                        "dataset": ds,
                        "path": str(rel),
                        "topology": path.parent.name,
                        "n_vars": len(verts),
                        "n_edges": len(edges),
                    }
                )
    return out


def rel_sql(ds: str, verts: dict, a: int, b: int, label: int) -> str:  # type: ignore[type-arg]
    joins, where = "", [f"e.label = {label}"]
    (la, da), (lb, db) = verts[a], verts[b]
    if la != -1:
        joins += f" JOIN {ds}_vlabel va ON va.vid = e.src AND va.label = {la}"
    if lb != -1:
        joins += f" JOIN {ds}_vlabel vb ON vb.vid = e.dst AND vb.label = {lb}"
    if da != -1:
        where.append(f"e.src = {da}")
    if db != -1:
        where.append(f"e.dst = {db}")
    return (
        f"SELECT DISTINCT e.src AS x, e.dst AS y FROM {ds}_edges e{joins} WHERE "
        + " AND ".join(where)
    )


def walk_orders(
    attrs: list[str], edges: list[tuple[str, str]]
) -> list[tuple[str, ...]]:
    """BFS and DFS orders from every start vertex, and their reverses."""
    adj: dict[str, list[str]] = collections.defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    orders: set[tuple[str, ...]] = set()
    for start in attrs:
        for mode in ("bfs", "dfs"):
            seen, order, frontier = {start}, [start], [start]
            while frontier:
                v = frontier.pop(0) if mode == "bfs" else frontier.pop()
                for w in sorted(adj[v]):
                    if w not in seen:
                        seen.add(w)
                        order.append(w)
                        frontier.append(w)
            order += [a for a in attrs if a not in seen]
            orders.add(tuple(order))
            orders.add(tuple(reversed(order)))
    return sorted(orders)


def max_pair_join(data: dict, vedges: list[tuple[str, str]], names: list[str]) -> int:  # type: ignore[type-arg]
    """Largest size of a two-relation join on a shared variable (sum of degree products)."""
    deg: dict[tuple[str, str], collections.Counter[str]] = {}
    for name, (u, v) in zip(names, vedges, strict=True):
        attrs, rows = data[name]
        for var in (u, v):
            i = attrs.index(var)
            deg[(name, var)] = collections.Counter(r[i] for r in rows)
    best = 0
    for (n1, (a1, b1)), (n2, (a2, b2)) in itertools.combinations(
        zip(names, vedges, strict=True), 2
    ):
        for var in {a1, b1} & {a2, b2}:
            d1, d2 = deg[(n1, var)], deg[(n2, var)]
            small, large = (d1, d2) if len(d1) < len(d2) else (d2, d1)
            best = max(best, sum(c * large.get(x, 0) for x, c in small.items()))
    return best


def _limit_memory() -> None:
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))


def _evaluate_safe(q: dict[str, object]) -> dict[str, object]:
    """Run one query; failures become censored rows instead of ending the run."""
    try:
        return evaluate(q)
    except (MemoryError, duckdb.OutOfMemoryException):
        return {**q, "censored": "memory_limit"}
    except Exception as exc:  # recorded, never silently dropped
        return {**q, "censored": f"error:{type(exc).__name__}"}


def exact_count(
    con: duckdb.DuckDBPyConnection, names: list[str], edges: list[tuple[str, str]]
) -> int | None:
    conds = []
    owner: dict[str, str] = {}
    for name, (a, b) in zip(names, edges, strict=True):
        for col, var in (("x", a), ("y", b)):
            if var in owner:
                conds.append(f"{name}.{col} = {owner[var]}")
            else:
                owner[var] = f"{name}.{col}"
    sql = f"SELECT count(*) FROM {', '.join(names)}" + (
        " WHERE " + " AND ".join(conds) if conds else ""
    )
    out: list[int | None] = [None]
    err: list[BaseException | None] = [None]

    def go() -> None:
        try:
            out[0] = int(con.execute(sql).fetchone()[0])  # type: ignore[index]
        except duckdb.InterruptException:
            out[0] = None  # our own timeout below
        except Exception as exc:
            err[0] = exc

    th = threading.Thread(target=go, daemon=True)
    th.start()
    th.join(EXACT_TIMEOUT_S)
    if th.is_alive():
        con.interrupt()
        th.join()
        return None
    if err[0] is not None:
        raise err[0]  # e.g. out of memory: censored by _evaluate_safe
    return out[0]


def evaluate(q: dict[str, object]) -> dict[str, object]:
    t0 = time.perf_counter()
    ds = str(q["dataset"])
    verts, edges = parse(GCARE / "queries" / str(q["path"]))
    truth_path = GCARE / "truth" / str(q["path"])
    row: dict[str, object] = {
        **q,
        "gcare_true": truth_path.read_text().split()[0] if truth_path.exists() else "",
    }
    con = duckdb.connect(str(GCARE / "gcare.duckdb"), read_only=True)
    con.execute(f"SET threads TO {DUCKDB_THREADS}")
    names, vedges, sizes = [], [], []
    for i, (a, b, lab) in enumerate(edges):
        name = f"R{i}"
        con.execute(f"CREATE TEMP TABLE {name} AS {rel_sql(ds, verts, a, b, lab)}")
        sizes.append(con.execute(f"SELECT count(*) FROM {name}").fetchone()[0])  # type: ignore[index]
        names.append(name)
        vedges.append((f"v{a}", f"v{b}"))
    row["max_rows"] = max(sizes)
    row["exact"] = exact_count(con, names, vedges)
    if max(sizes) > MAX_ROWS:
        row["censored"] = "too_large"
        con.close()
        return row
    rels, data = [], {}
    for name, (u, v) in zip(names, vedges, strict=True):
        attrs = tuple(sorted((u, v)))
        rows = con.execute(f"SELECT x, y FROM {name}").fetchall()
        rows = (
            [(str(x), str(y)) for x, y in rows]
            if attrs == (u, v)
            else [(str(y), str(x)) for x, y in rows]
        )
        rels.append(Relation(name, frozenset({u, v}), len(rows)))
        data[name] = (attrs, rows)
    con.close()
    row["max_pair_join"] = max_pair_join(data, vedges, names)
    if int(str(row["max_pair_join"])) > MAX_PAIR_JOIN:
        row["censored"] = "pair_join_too_large"
        return row
    query = QueryHypergraph(relations=tuple(rels), name=str(q["path"]))
    attrs = sorted(query.all_attributes)
    stats = compute_all_lp_block_extensions(
        query, enumerate_blocks(query, max_block_size=2), data, p_values=P11
    )
    orders = walk_orders(attrs, vedges)
    for tag, ps in (("p3", P3), ("p11", P11)):
        cons1 = block_constraints(
            query, data, max_block_size=1, p_values=ps, lp_stats=stats
        )
        cons2 = block_constraints(
            query, data, max_block_size=2, p_values=ps, lp_stats=stats
        )
        row[f"entropic_{tag}"] = f"{flow_bound(attrs, cons1).bound:.6e}"
        row[f"cover_bs1_{tag}"] = (
            f"{estimate_lph_ceg(query, data, max_block_size=1, p_values=ps, lp_stats=stats).estimate:.6e}"
        )
        row[f"cover_bs2_{tag}"] = (
            f"{estimate_lph_ceg(query, data, max_block_size=2, p_values=ps, lp_stats=stats).estimate:.6e}"
        )
        row[f"flow_bs2_{tag}"] = (
            f"{flow_bound_over_orders(attrs, cons2, orders)[0].bound:.6e}"
        )
    row["censored"] = "" if row["exact"] is not None else "exact_timeout"
    row["seconds"] = f"{time.perf_counter() - t0:.1f}"
    return row


def _write(name: str, rows: list[dict[str, object]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(OUT / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def summarise(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    def f(r: dict[str, object], c: str) -> float:
        return float(str(r[c]))

    out: list[dict[str, object]] = []
    for ds in DATASETS:
        allq = [r for r in rows if r["dataset"] == ds]
        ev = [r for r in allq if str(r.get("cover_bs1_p11") or "") not in ("", "None")]
        if not ev:
            continue
        chk = [r for r in ev if r.get("exact") not in (None, "")]
        pos = [r for r in chk if int(str(r["exact"])) > 0]
        rec: dict[str, object] = {
            "dataset": ds,
            "n_cyclic": len(allq),
            "n_too_large": sum(r.get("censored") == "too_large" for r in allq),
            "n_pair_join_too_large": sum(
                r.get("censored") == "pair_join_too_large" for r in allq
            ),
            "n_memory_limit": sum(r.get("censored") == "memory_limit" for r in allq),
            "n_evaluated": len(ev),
            "n_exact": len(chk),
            "n_gcare_true_eq_exact": sum(
                str(r["gcare_true"]) == str(r["exact"])
                for r in allq
                if r.get("exact") not in (None, "")
            ),
        }
        for tag in ("p11", "p3"):
            ent = f"entropic_{tag}"
            rec[f"median_cover_bs1_over_entropic_{tag}"] = (
                f"{statistics.median(f(r, f'cover_bs1_{tag}') / f(r, ent) for r in ev):.6g}"
            )
            rec[f"median_cover_bs2_over_entropic_{tag}"] = (
                f"{statistics.median(f(r, f'cover_bs2_{tag}') / f(r, ent) for r in ev):.6g}"
            )
            rec[f"median_flow_bs2_over_entropic_{tag}"] = (
                f"{statistics.median(f(r, f'flow_bs2_{tag}') / f(r, ent) for r in ev):.6g}"
            )
            rec[f"median_cover_bs2_over_flow_bs2_{tag}"] = (
                f"{statistics.median(f(r, f'cover_bs2_{tag}') / f(r, f'flow_bs2_{tag}') for r in ev):.6g}"
            )
            rec[f"n_cover_bs2_below_entropic_{tag}"] = sum(
                f(r, f"cover_bs2_{tag}") < f(r, ent) for r in ev
            )
            rec[f"n_flow_bs2_below_entropic_{tag}"] = sum(
                f(r, f"flow_bs2_{tag}") < f(r, ent) for r in ev
            )
            rec[f"n_invalid_{tag}"] = sum(
                f(r, f"{c}_{tag}") < int(str(r["exact"])) * (1 - 1e-9)
                for r in chk
                for c in ("entropic", "cover_bs1", "cover_bs2", "flow_bs2")
            )
        rec["median_entropic_over_exact_p11"] = (
            f"{statistics.median(f(r, 'entropic_p11') / int(str(r['exact'])) for r in pos):.6g}"
            if pos
            else ""
        )
        rec["median_flow_bs2_over_exact_p11"] = (
            f"{statistics.median(f(r, 'flow_bs2_p11') / int(str(r['exact'])) for r in pos):.6g}"
            if pos
            else ""
        )
        rec["median_seconds"] = f"{statistics.median(f(r, 'seconds') for r in ev):.1f}"
        out.append(rec)
    return out


def main() -> None:
    if "--summary-only" in sys.argv:
        with open(OUT / "gcare_eval.csv") as fh:
            _write("gcare_summary.csv", summarise(list(csv.DictReader(fh))))
        return
    qs = registry()
    rows: list[dict[str, object]] = []
    if "--resume" in sys.argv and (OUT / "gcare_eval.csv").exists():
        with open(OUT / "gcare_eval.csv") as fh:
            rows = list(csv.DictReader(fh))
        done = {r["path"] for r in rows}
        qs = [q for q in qs if q["path"] not in done]
    if "--limit" in sys.argv:
        qs = qs[: int(sys.argv[sys.argv.index("--limit") + 1])]
    print(f"{len(qs)} cyclic queries to evaluate", flush=True)
    with multiprocessing.Pool(
        processes=WORKERS, maxtasksperchild=1, initializer=_limit_memory
    ) as pool:
        for r in pool.imap_unordered(_evaluate_safe, qs):
            rows.append(r)
            rows.sort(key=lambda x: str(x["path"]))
            _write("gcare_eval.csv", rows)
            print(r["path"], r.get("censored"), r.get("seconds"), flush=True)
    _write("gcare_summary.csv", summarise(rows))
    print("done", flush=True)


if __name__ == "__main__":
    main()
