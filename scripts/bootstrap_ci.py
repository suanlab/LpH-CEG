"""Query-level bootstrap intervals for the headline medians (eleven norms).

The real-data results are medians of per-query ratios over 40-241 queries.
This resamples queries with replacement (10,000 draws, fixed seed) and
reports the 95% percentile interval of each median ratio to the entropic
LpBound: cover bs=1, cover bs=2 and flow bs=2, per suite and |V| (G-CARE
per dataset).  The question is whether any headline crosses 1, i.e.
whether the sign of a comparison could be a sampling accident.

Output: results/suite/bootstrap_ci.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DRAWS = 10_000
SEED = 0
RATIOS = ("cover_bs1_p11", "cover_bs2_p11", "flow_bs2_p11", "poly_bs2_p11")


def _rows(rel: str) -> list[dict[str, str]]:
    return list(csv.DictReader((ROOT / rel).read_text().splitlines()))


def _with_poly(
    rows: list[dict[str, str]], poly: dict[str, str], key: str
) -> list[dict[str, str]]:
    return [{**r, "poly_bs2_p11": poly[r[key]]} for r in rows]


def groups() -> list[tuple[str, str, list[dict[str, str]]]]:
    out: list[tuple[str, str, list[dict[str, str]]]] = []
    lp = {
        r["query_id"]: r["poly_bs2_p11"]
        for r in _rows("results/suite/poly_certify.csv")
    }
    fair = _with_poly(_rows("results/suite/pset_fair.csv"), lp, "query_id")
    for suite in ("hetionet", "imdb"):
        for nv in ("6", "8", "10"):
            rs = [r for r in fair if r["suite"] == suite and r["n_vars"] == nv]
            out.append((suite, nv, rs))
    sp = {
        r["query_id"]: r["poly_bs2_p11"]
        for r in _rows("results/suite/poly_certify_short.csv")
    }
    out.append(
        (
            "hetionet_short",
            "3-4",
            _with_poly(_rows("results/suite/p11_short.csv"), sp, "query_id"),
        )
    )
    gp = {
        r["path"]: r["poly_bs2_p11"]
        for r in _rows("results/gcare/gcare_poly_certify.csv")
    }
    gc = [
        r
        for r in _with_poly(
            [r for r in _rows("results/gcare/gcare_eval.csv") if r["path"] in gp],
            gp,
            "path",
        )
    ]
    for ds in ("aids", "yago"):
        rs = [r for r in gc if r["dataset"] == ds and r["cover_bs1_p11"]]
        out.append((f"gcare_{ds}", "all", rs))
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    for suite, nv, rs in groups():
        ent = np.array([float(r["entropic_p11"]) for r in rs])
        for col in RATIOS:
            x = np.array([float(r[col]) for r in rs]) / ent
            idx = rng.integers(0, len(x), size=(DRAWS, len(x)))
            meds = np.median(x[idx], axis=1)
            lo, hi = np.percentile(meds, [2.5, 97.5])
            rows.append(
                {
                    "suite": suite,
                    "n_vars": nv,
                    "n": len(x),
                    "ratio": f"{col}/entropic_p11",
                    "median": f"{np.median(x):.6g}",
                    "ci_lo": f"{lo:.6g}",
                    "ci_hi": f"{hi:.6g}",
                    "crosses_1": bool(lo <= 1.0 <= hi),
                }
            )
    out = ROOT / "results" / "suite" / "bootstrap_ci.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(
            f"{r['suite']:<15} {r['n_vars']:>4} {r['ratio']:<28} "
            f"{float(str(r['median'])):>10.4g} [{float(str(r['ci_lo'])):.4g}, "
            f"{float(str(r['ci_hi'])):.4g}] {'CROSSES 1' if r['crosses_1'] else ''}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
