"""Where is the headroom?  Split a bound's gap into program and information.

For statistics S of a database D and any valid bound B computed from S,

    B / truth = (B / OPT_S) * (OPT_S / truth),
    OPT_S = max { |Q(D')| : every statistic of D' is <= its value in S },

the program gap (what the inference leaves on the table) times the
information gap (what S cannot distinguish).  OPT_S is hard to compute, but
any feasible D' gives a lower bound LB <= OPT_S, hence a safe bracket

    program gap <= B / LB      information gap >= LB / truth.

This searches for large feasible D' by simulated annealing (single-tuple
flips, starting from D) on small random cycle instances, for the bs=1 and
the bs<=2 catalogs (three norms), with B the 2^|V| polymatroid bound over
the same catalog.  Exact counts are traces of 0/1 matrix products.

Output: results/characterization/opt_stats_gap.csv
"""

from __future__ import annotations

import csv
import math
import multiprocessing
import random
import statistics
from pathlib import Path

import numpy as np

from src.core.block import enumerate_blocks
from src.estimators.flow_bound import block_constraints, polymatroid_bound
from src.statistics.lp_stats import compute_all_lp_block_extensions
from src.utils.query import cycle_query

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "characterization" / "opt_stats_gap.csv"
P = (1.0, 2.0, math.inf)
DOMAIN = 6
STEPS = 4000
SEEDS = range(12)
SHAPES = (3, 4)


def _instance(k: int, seed: int):  # type: ignore[no-untyped-def]
    rng = random.Random(1000 * k + seed)
    q = cycle_query(k)
    mats = []
    for _ in q.relations:
        m = np.zeros((DOMAIN, DOMAIN), dtype=bool)
        for _ in range(rng.randrange(6, 16)):
            m[
                min(int(rng.paretovariate(1.1)) - 1, DOMAIN - 1), rng.randrange(DOMAIN)
            ] = True
        mats.append(m)
    return q, mats


def _data(q, mats):  # type: ignore[no-untyped-def]
    """Relation i joins attributes a_{i+1} -> a_{i+2} (cycle order)."""
    data = {}
    rels = sorted(q.relations, key=lambda r: int(r.name.lstrip("R")))
    k = len(rels)
    for i, rel in enumerate(rels):
        u, v = f"a{i + 1}", f"a{(i + 1) % k + 1}"
        attrs = tuple(sorted(rel.attributes))
        assert set(attrs) == {u, v}, (attrs, u, v)
        rows = []
        for x, y in zip(*np.nonzero(mats[i]), strict=True):
            row = {u: str(x), v: str(y)}
            rows.append(tuple(row[a] for a in attrs))
        data[rel.name] = (attrs, rows)
    return data


def _count(mats) -> int:  # type: ignore[no-untyped-def]
    prod = mats[0].astype(np.int64)
    for m in mats[1:]:
        prod = prod @ m.astype(np.int64)
    return int(np.trace(prod))


def _stats(q, mats, bs: int):  # type: ignore[no-untyped-def]
    data = _data(q, mats)
    st = compute_all_lp_block_extensions(
        q, enumerate_blocks(q, max_block_size=bs), data, p_values=P
    )
    cons = block_constraints(q, data, max_block_size=bs, p_values=P, lp_stats=st)
    # Several blocks can share (interface, target, p) -- on a triangle every
    # two-relation block covers all attributes -- so key by occurrence too.
    # Block enumeration depends on the query only, so occurrences align.
    keyed: dict[tuple[object, ...], float] = {}
    seen: dict[tuple[object, ...], int] = {}
    for c in cons:
        base = (c.conditioning, c.target, c.p)
        n = seen.get(base, 0)
        seen[base] = n + 1
        keyed[(*base, n)] = c.log_norm
    return keyed, cons


def _feasible(cand: dict, ref: dict) -> bool:  # type: ignore[type-arg]
    return all(v <= ref.get(key, math.inf) + 1e-9 for key, v in cand.items())


def evaluate(item: tuple[int, int]) -> dict[str, object]:
    k, seed = item
    q, mats = _instance(k, seed)
    attrs = sorted(q.all_attributes)
    truth = _count(mats)
    row: dict[str, object] = {"k": k, "seed": seed, "truth": truth}
    if truth == 0:  # nothing to decompose; dropped by main()
        return row
    for bs in (1, 2):
        ref, cons = _stats(q, mats, bs)
        poly = 2.0 ** polymatroid_bound(attrs, cons)
        rng = random.Random(seed * 7 + bs)
        cur = [m.copy() for m in mats]
        cur_c = best = truth
        temp = 1.0
        for step in range(STEPS):
            temp = max(0.05, 1.0 - step / STEPS)
            i = rng.randrange(k)
            x, y = rng.randrange(DOMAIN), rng.randrange(DOMAIN)
            cur[i][x, y] = not cur[i][x, y]
            c = _count(cur)
            ok = c > 0 and _feasible(_stats(q, cur, bs)[0], ref)
            if ok and (
                c >= cur_c
                or rng.random() < math.exp((math.log2(c) - math.log2(cur_c)) / temp)
            ):
                cur_c = c
                best = max(best, c)
            else:
                cur[i][x, y] = not cur[i][x, y]
        row[f"poly_bs{bs}"] = f"{poly:.6g}"
        row[f"lb_bs{bs}"] = best
        row[f"program_gap_ub_bs{bs}"] = f"{poly / best:.6g}"
        row[f"info_gap_lb_bs{bs}"] = f"{best / truth:.6g}" if truth else ""
    return row


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    items = [(k, s) for k in SHAPES for s in SEEDS]
    with multiprocessing.Pool(processes=8) as pool:
        rows = [r for r in pool.map(evaluate, items) if r["truth"]]
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary: list[dict[str, object]] = []
    for k in SHAPES:
        rs = [r for r in rows if r["k"] == k]

        def med(f, rs=rs):  # type: ignore[no-untyped-def]
            return statistics.median(f(r) for r in rs)

        rec: dict[str, object] = {"k": k, "n": len(rs)}
        for bs in (1, 2):
            rec[f"median_bound_over_truth_bs{bs}"] = (
                f"{med(lambda r, b=bs: float(r[f'poly_bs{b}']) / r['truth']):.4g}"
            )
            rec[f"median_program_gap_ub_bs{bs}"] = (
                f"{med(lambda r, b=bs: float(r[f'program_gap_ub_bs{b}'])):.4g}"
            )
            rec[f"median_info_gap_lb_bs{bs}"] = (
                f"{med(lambda r, b=bs: float(r[f'info_gap_lb_bs{b}'])):.4g}"
            )
        rec["n_lb_above_bound"] = sum(
            float(r[f"lb_bs{b}"]) > float(r[f"poly_bs{b}"]) * (1 + 1e-6)
            for r in rs
            for b in (1, 2)
        )
        summary.append(rec)
        print(rec)
    with open(OUT.with_name("opt_stats_gap_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)


if __name__ == "__main__":
    main()
