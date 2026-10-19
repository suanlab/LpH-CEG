"""Certify the cycle-sweep flow bounds beyond k=10 by the polymatroid bound.

``flow_bound_blocks.py part_scale`` reports the bs<=2 flow bound on the
synthetic cycle sweep up to k=16 but runs the ``2**|V|`` polymatroid
program only to k=10.  This rebuilds the same instances and statistics
(three norms) for the larger k and solves the polymatroid bound over the
unrelaxed bs<=2 catalog, so the reported flow bounds are certified
(``flow >= poly``) where the program finishes.  Rows are written after
each k; a k that does not finish within the time cap is recorded as such.

Output: results/flow/sweep_poly_certify.csv
"""

from __future__ import annotations

import csv
import multiprocessing
import sys
import time
from pathlib import Path

from scripts.evaluate_query_suite import P_VALUES
from scripts.scalability_vs_entropic import _build_cycle
from src.core.block import enumerate_blocks
from src.estimators.flow_bound import block_constraints, polymatroid_bound
from src.statistics.lp_stats import compute_all_lp_block_extensions

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "flow" / "sweep_poly_certify.csv"
K_VALUES = (12, 14, 16)
CAP_S = 6 * 3600


def _poly(k: int, conn) -> None:  # type: ignore[no-untyped-def]
    q, data = _build_cycle(k, 100, 400)
    attrs = sorted(q.all_attributes)
    stats = compute_all_lp_block_extensions(
        q, enumerate_blocks(q, max_block_size=2), data, p_values=P_VALUES
    )
    cons2 = block_constraints(
        q, data, max_block_size=2, p_values=P_VALUES, lp_stats=stats
    )
    t0 = time.perf_counter()
    conn.send((2.0 ** polymatroid_bound(attrs, cons2), time.perf_counter() - t0))


def main() -> None:
    flow = {
        int(r["k"]): float(r["flow_bs2_bound"])
        for r in csv.DictReader(
            (ROOT / "results" / "flow" / "flow_scale.csv").read_text().splitlines()
        )
    }
    ks = [int(a) for a in sys.argv[1:]] or list(K_VALUES)
    rows: list[dict[str, object]] = []
    if OUT.exists():
        rows = [r for r in csv.DictReader(OUT.read_text().splitlines())]
        rows = [r for r in rows if int(str(r["k"])) not in ks]
    for k in ks:
        parent, child = multiprocessing.Pipe()
        p = multiprocessing.Process(target=_poly, args=(k, child))
        p.start()
        p.join(CAP_S)
        if p.is_alive():
            p.terminate()
            row: dict[str, object] = {"k": k, "status": f"not finished in {CAP_S} s"}
        else:
            poly, secs = parent.recv()
            row = {
                "k": k,
                "status": "ok",
                "flow_bs2_bound": f"{flow[k]:.6e}",
                "poly_bs2_bound": f"{poly:.6e}",
                "flow_over_poly": f"{flow[k] / poly:.6f}",
                "poly_s": f"{secs:.1f}",
            }
        rows.append(row)
        rows.sort(key=lambda r: int(str(r["k"])))
        keys = [
            "k",
            "status",
            "flow_bs2_bound",
            "poly_bs2_bound",
            "flow_over_poly",
            "poly_s",
        ]
        with open(OUT, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(row, flush=True)


if __name__ == "__main__":
    main()
