"""Leave-one-out and add-one ablation of the bs=2 statistic groups.

``ablation_p11_poly.py`` adds groups in one order (join-free, then the
common-neighbour l2).  An attribution read off one order can depend on it,
so this removes each group from the full catalogue and adds each group alone
to the single-relation one, in the headline configuration (eleven norms, the
2^|V| polymatroid program, pair statistics from the sparse collector that
the cost table times).  The groups partition the interfaces of a pair block
R(a, b), S(b, c):

  size      the empty interface (the pair-join size)
  shared    interfaces containing the shared attributes ({b}, {a,b}, {b,c})
  one_side  the other interfaces inside one schema ({a}, {c})
  codeg     the join-required interface {a, c}, every norm
  codeg_l2  its l2 norm alone (the statistic ``ablation_p11_poly`` isolates)

Catalogues: ``bs1``, ``full``, ``full-<g>`` and ``bs1+<g>`` for every group.
Queries: the Hetionet and IMDb long suites (|V| = 6, 8, 10), under a
wall-clock cap; queries over the cap are recorded, not dropped.

Output: results/pushdown/ablation_loo.csv and _summary.csv
"""

from __future__ import annotations

import csv
import json
import multiprocessing
import signal
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.pset_fair import P11, SUITES
from scripts.pushdown_stats_experiment import _l2_only
from src.core.block import enumerate_blocks
from src.estimators.flow_bound import block_constraints, polymatroid_bound
from src.statistics.lp_stats import compute_all_lp_block_extensions
from src.statistics.pushdown_stats import (
    compute_sparse_block_extensions,
    join_free_interfaces,
)

if TYPE_CHECKING:
    from src.statistics.lp_stats import LpBlockExtensionStats
    from src.utils.types import Block, Interface, QueryHypergraph

    StatsMap = dict[tuple[Block, Interface], LpBlockExtensionStats]

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "pushdown"
WORKERS = 8
TASK_TIMEOUT_S = 3600
EQ = 1.0001
GROUPS = ("size", "shared", "one_side", "codeg", "codeg_l2")


class _TaskTimeoutError(Exception):
    pass


def _alarm(signum, frame):  # type: ignore[no-untyped-def]
    raise _TaskTimeoutError


def group_of(query: QueryHypergraph, block: Block, iface: Interface) -> str:
    """The group of a pair-block statistic (see the module docstring)."""
    if not iface:
        return "size"
    kind = join_free_interfaces(query, block)[iface]
    return "codeg" if kind is None else str(kind)


def catalogues(query: QueryHypergraph, stats: StatsMap) -> dict[str, StatsMap]:
    """Every ablation catalogue, as a sub-map of ``stats``."""
    singles: StatsMap = {k: v for k, v in stats.items() if len(k[0].relations) == 1}
    by_group: dict[str, StatsMap] = {g: {} for g in GROUPS}
    for (block, iface), st in stats.items():
        if len(block.relations) != 2:
            continue
        g = group_of(query, block, iface)
        by_group[g][(block, iface)] = st
        if g == "codeg" and 2.0 in st.lp_norms:
            by_group["codeg_l2"][(block, iface)] = _l2_only(st.lp_norms[2.0])
    out: dict[str, StatsMap] = {"bs1": singles, "full": dict(stats)}
    for g in GROUPS:
        out[f"bs1+{g}"] = {**singles, **by_group[g]}
        if g == "codeg_l2":
            # full with the codegree reduced to its l2 norm
            out["full-codeg_but_l2"] = {
                **{k: v for k, v in stats.items() if k not in by_group["codeg"]},
                **by_group["codeg_l2"],
            }
        else:
            out[f"full-{g}"] = {k: v for k, v in stats.items() if k not in by_group[g]}
    return out


def evaluate(item: tuple[str, dict]) -> dict[str, object]:  # type: ignore[type-arg]
    suite, q = item
    row: dict[str, object] = {
        "suite": suite,
        "query_id": q["query_id"],
        "n_vars": q["n_vars"],
    }
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(TASK_TIMEOUT_S)
    t0 = time.perf_counter()
    try:
        built = SUITES[suite][1](q)
        assert built is not None
        query, data = built
        attrs = sorted(query.all_attributes)
        blocks = enumerate_blocks(query, max_block_size=2)
        singles = [b for b in blocks if len(b.relations) == 1]
        stats: StatsMap = dict(
            compute_all_lp_block_extensions(query, singles, data, p_values=P11)
        )
        for b in blocks:
            if len(b.relations) == 2:
                stats.update(
                    compute_sparse_block_extensions(query, b, data, p_values=P11)
                )
        for name, st in catalogues(query, stats).items():
            cons = block_constraints(
                query,
                data,
                max_block_size=1 if name == "bs1" else 2,
                p_values=P11,
                lp_stats=st,
            )
            row[f"log2_{name}"] = f"{polymatroid_bound(attrs, cons):.6f}"
        row["status"] = "ok"
    except _TaskTimeoutError:
        row["status"] = f"over {TASK_TIMEOUT_S} s"
    finally:
        signal.alarm(0)
    row["seconds"] = f"{time.perf_counter() - t0:.1f}"
    return row


def summarise(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    def lg(r: dict[str, object], c: str) -> float:
        return float(str(r[f"log2_{c}"]))

    names = [
        *(f"full-{g}" for g in GROUPS if g != "codeg_l2"),
        "full-codeg_but_l2",
        *(f"bs1+{g}" for g in GROUPS),
    ]
    out: list[dict[str, object]] = []
    for suite in ("hetionet", "imdb"):
        rs = [r for r in rows if r["suite"] == suite and r["status"] == "ok"]
        n_all = sum(1 for r in rows if r["suite"] == suite)
        if not rs:
            continue

        def share(r: dict[str, object], c: str) -> float:
            gain = lg(r, "bs1") - lg(r, "full")
            return (lg(r, "bs1") - lg(r, c)) / gain if gain > 1e-9 else 1.0

        for c in names:
            out.append(
                {
                    "suite": suite,
                    "catalogue": c,
                    "n": n_all,
                    "n_ok": len(rs),
                    "median_full_gain_x": f"{statistics.median(2 ** (lg(r, 'bs1') - lg(r, 'full')) for r in rs):.4g}",
                    "n_eq_full": sum(2 ** (lg(r, c) - lg(r, "full")) < EQ for r in rs),
                    "median_over_full": f"{statistics.median(2 ** (lg(r, c) - lg(r, 'full')) for r in rs):.4g}",
                    "max_over_full": f"{max(2 ** (lg(r, c) - lg(r, 'full')) for r in rs):.4g}",
                    "median_gain_share": f"{statistics.median(share(r, c) for r in rs):.4g}",
                }
            )
    return out


def _write(name: str, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(OUT / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    items = [
        (suite, q)
        for suite in ("hetionet", "imdb")
        for q in json.loads(SUITES[suite][0].read_text())["queries"]
    ]
    with multiprocessing.Pool(processes=WORKERS, maxtasksperchild=1) as pool:
        rows = pool.map(evaluate, items, chunksize=1)
    rows.sort(key=lambda r: (str(r["suite"]), str(r["query_id"])))
    _write("ablation_loo.csv", rows)
    summary = summarise(rows)
    _write("ablation_loo_summary.csv", summary)
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
