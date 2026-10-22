"""Do the Hetionet long-suite margins depend on the file-order prefix?

The long suite caps each relation at its first 20K rows in file order
(``evaluate_query_suite.MAX_ROWS``).  This reruns the |V|=6 queries with
each relation replaced by a uniform random sample of 20K rows of the full
file (two seeds, fixed in advance) and recomputes, with eleven norms on both
sides as in ``pset_fair.py``, the entropic bound and the cover and flow
LPs.  The question is whether the block-statistics margins over LpBound
(``cover_bs2 / entropic``, ``flow_bs2 / entropic``) move with the slice.

Outputs: results/suite/slice_robustness.csv and _summary.csv
"""

from __future__ import annotations

import csv
import json
import multiprocessing
import random
import statistics
import sys
from pathlib import Path

import scripts.evaluate_query_suite as eqs
from scripts import pset_fair

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "suite"
SEEDS = (1, 2)
N_VARS = 6
WORKERS = 8


def _sampled_loader(seed: int):  # type: ignore[no-untyped-def]
    def load(fname: str, base: Path, has_header: bool = True) -> list[tuple[str, str]]:
        with open(base / fname) as f:
            if has_header:
                f.readline()
            rows = [
                (p[0], p[1])
                for p in (ln.rstrip("\n").replace("\t", " ").split() for ln in f)
                if len(p) >= 2
            ]
        if len(rows) <= eqs.MAX_ROWS:
            return rows
        # per-file seed so the same relation gets the same sample in every query
        return random.Random(f"{seed}:{fname}").sample(rows, eqs.MAX_ROWS)

    return load


def evaluate(item: tuple[int, dict]) -> dict[str, object] | None:  # type: ignore[type-arg]
    seed, q = item
    eqs.load_edges = _sampled_loader(seed)
    row = pset_fair.evaluate(("hetionet", q))
    if row is None:
        return None
    return {"seed": seed, **row}


def summarise(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    base = {
        r["query_id"]: r
        for r in csv.DictReader((OUT / "pset_fair.csv").read_text().splitlines())
        if r["suite"] == "hetionet"
    }

    def f(r: dict[str, object], c: str) -> float:
        return float(str(r[c]))

    out: list[dict[str, object]] = []
    for seed in ("prefix", *SEEDS):
        rs = (
            [base[str(r["query_id"])] for r in rows if str(r["seed"]) == str(SEEDS[0])]
            if seed == "prefix"
            else [r for r in rows if str(r["seed"]) == str(seed)]
        )
        if not rs:
            continue
        ent = "entropic_p11"
        out.append(
            {
                "slice": seed,
                "n": len(rs),
                "median_cover_bs1_over_entropic_p11": f"{statistics.median(f(r, 'cover_bs1_p11') / f(r, ent) for r in rs):.6g}",
                "median_cover_bs2_over_entropic_p11": f"{statistics.median(f(r, 'cover_bs2_p11') / f(r, ent) for r in rs):.6g}",
                "median_flow_bs2_over_entropic_p11": f"{statistics.median(f(r, 'flow_bs2_p11') / f(r, ent) for r in rs):.6g}",
                "n_cover_bs2_below_entropic_p11": sum(
                    f(r, "cover_bs2_p11") < f(r, ent) for r in rs
                ),
                "n_flow_bs2_below_entropic_p11": sum(
                    f(r, "flow_bs2_p11") < f(r, ent) for r in rs
                ),
            }
        )
    return out


def _write(name: str, rows: list[dict[str, object]]) -> None:
    with open(OUT / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    path = OUT / "slice_robustness.csv"
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
                _write("slice_robustness.csv", rows)
                print(r["seed"], r["query_id"], r["seconds"], flush=True)
    _write("slice_robustness_summary.csv", summarise(rows))
    print("done")


if __name__ == "__main__":
    main()
