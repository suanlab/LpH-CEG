"""Certify the slice-robustness flow bounds by the polymatroid bound.

``slice_robustness.py`` reruns the |V|=6 Hetionet queries on two random
20K-row samples per relation.  This recomputes, on the same samples, the
ten-norm flow bound and the ``2**|V|`` polymatroid bound over the
unrelaxed ``bs<=2`` catalog (``poly_certify.evaluate``), so that every flow
bound the paper reports is certified.  The flow values must reproduce
``slice_robustness.csv``.

Outputs: results/suite/slice_poly_certify.csv and _summary.csv
"""

from __future__ import annotations

import csv
import json
import multiprocessing
import sys
from pathlib import Path

import scripts.evaluate_query_suite as eqs
from scripts import poly_certify, pset_fair
from scripts.slice_robustness import N_VARS, SEEDS, _sampled_loader

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "suite"
WORKERS = 8


def evaluate(item: tuple[int, dict]) -> dict[str, object] | None:  # type: ignore[type-arg]
    seed, q = item
    eqs.load_edges = _sampled_loader(seed)
    row = poly_certify.evaluate(("hetionet", q))
    return None if row is None else {"seed": seed, **row}


def _write(name: str, rows: list[dict[str, object]]) -> None:
    with open(OUT / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarise(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    ref = {
        (r["seed"], r["query_id"]): float(r["flow_bs2_p11"])
        for r in csv.DictReader((OUT / "slice_robustness.csv").read_text().splitlines())
    }
    ratio = [float(str(r["flow_over_poly_p11"])) for r in rows]
    repro = [
        float(str(r["flow_bs2_p11"])) / ref[(str(r["seed"]), str(r["query_id"]))]
        for r in rows
    ]
    return [
        {
            "n": len(rows),
            "n_flow_below_poly": sum(x < 1 - 1e-6 for x in ratio),
            "n_flow_eq_poly": sum(abs(x - 1) <= 1e-6 for x in ratio),
            "max_flow_vs_slice_robustness": f"{max(abs(x - 1) for x in repro):.3g}",
        }
    ]


def main() -> None:
    path = OUT / "slice_poly_certify.csv"
    rows: list[dict[str, object]] = []
    if "--resume" in sys.argv and path.exists():
        rows = list(csv.DictReader(path.read_text().splitlines()))
    if "--summary-only" not in sys.argv:
        done = {(str(r["seed"]), str(r["query_id"])) for r in rows}
        qs = json.loads(pset_fair.SUITES["hetionet"][0].read_text())["queries"]
        items = [
            (seed, q)
            for seed in SEEDS
            for q in qs
            if q["n_vars"] == N_VARS and (str(seed), q["query_id"]) not in done
        ]
        with multiprocessing.Pool(processes=WORKERS, maxtasksperchild=1) as pool:
            for r in pool.imap_unordered(evaluate, items):
                if r is None:
                    continue
                rows.append(r)
                rows.sort(key=lambda x: (str(x["seed"]), str(x["query_id"])))
                _write("slice_poly_certify.csv", rows)
                print(r["seed"], r["query_id"], r["flow_over_poly_p11"], flush=True)
    _write("slice_poly_certify_summary.csv", summarise(rows))
    print("done")


if __name__ == "__main__":
    main()
