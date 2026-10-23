"""G-CARE results by query topology: how many run, agree, and gain.

The YAGO suite is filtered by a size cap (a relation over 300K rows or a
pair join over 20M is reported, not run) and our counts differ from
G-CARE's on part of it, so the evaluated set is not a random subset.  This
breaks ``gcare_eval.csv`` and ``gcare_poly_certify.csv`` down by topology:
cyclic queries, evaluated, counts agreeing with G-CARE's, and the median
LpBound / polymatroid-over-blocks gain (eleven norms) overall, on the
agreeing queries and on the others.

Output: results/gcare/gcare_topology.csv
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GC = ROOT / "results" / "gcare"


def _med(xs: list[float]) -> str:
    return f"{statistics.median(xs):.4g}" if xs else ""


def main() -> None:
    with open(GC / "gcare_eval.csv") as fh:
        ev = list(csv.DictReader(fh))
    with open(GC / "gcare_poly_certify.csv") as fh:
        poly = {r["path"]: r for r in csv.DictReader(fh)}
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for r in ev:
        groups[(r["dataset"], r["topology"])].append(r)
        groups[(r["dataset"], "all")].append(r)
    rows = []
    for (ds, topo), rs in sorted(groups.items()):
        run = [
            r
            for r in rs
            if r.get("cover_bs1_p11") not in ("", None) and r["path"] in poly
        ]

        def gain(r: dict[str, str]) -> float:
            return float(r["entropic_p11"]) / float(poly[r["path"]]["poly_bs2_p11"])

        agree = [r for r in run if r["exact"] == r["gcare_true"]]
        other = [r for r in run if r["exact"] != r["gcare_true"]]
        rows.append(
            {
                "dataset": ds,
                "topology": topo,
                "n_cyclic": len(rs),
                "n_evaluated": len(run),
                "n_count_agrees": len(agree),
                "median_gain": _med([gain(r) for r in run]),
                "median_gain_agree": _med([gain(r) for r in agree]),
                "median_gain_disagree": _med([gain(r) for r in other]),
            }
        )
    with open(GC / "gcare_topology.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
