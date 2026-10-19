"""Headline numbers with the bs<=2 polymatroid bound as the program.

The ``2**|V|`` polymatroid bound over the unrelaxed bs<=2 catalog is valid
for every p without further argument, at least as tight as the flow LP,
and fast at the real-data sizes (|V| <= 12).  This joins the stored
ten-norm values -- the polymatroid and flow bounds (``poly_certify*.csv``,
``gcare_poly_certify.csv``, ``slice_poly_certify.csv``), the entropic and
cover bounds (``pset_fair.csv``, ``p11_short.csv``, ``gcare_eval.csv``,
``slice_robustness.csv``) and the exact counts -- and reports, per suite
and |V|, the medians the paper quotes.  No bound is recomputed.

Output: results/suite/poly_headline.csv
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _rows(rel: str) -> list[dict[str, str]]:
    return list(csv.DictReader((ROOT / rel).read_text().splitlines()))


def _f(x: str | None) -> float | None:
    return float(x) if x not in (None, "", "None", "nan") else None


def _joined() -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    fair = {
        (r["suite"], r["query_id"]): r for r in _rows("results/suite/pset_fair.csv")
    }
    truth: dict[tuple[str, str], float | None] = {}
    for r in _rows("results/suite/long_suite_eval.csv"):
        truth[("hetionet", r["query_id"])] = _f(r["true_count"])
    for r in _rows("results/imdb/imdb_suite_eval.csv"):
        truth[("imdb", r["query_id"])] = _f(r["true_count"])
    for r in _rows("results/suite/poly_certify.csv"):
        b = fair[(r["suite"], r["query_id"])]
        out.append(
            {
                "suite": r["suite"],
                "group": r["n_vars"],
                "poly": float(r["poly_bs2_p11"]),
                "flow": float(r["flow_bs2_p11"]),
                "ent": float(b["entropic_p11"]),
                "cover1": float(b["cover_bs1_p11"]),
                "cover2": float(b["cover_bs2_p11"]),
                "exact": truth.get((r["suite"], r["query_id"])),
            }
        )
    short = {r["query_id"]: r for r in _rows("results/suite/p11_short.csv")}
    st = {
        r["query_id"]: _f(r["true_count"])
        for r in _rows("results/ablation/real_frontier_truth.csv")
    }
    for r in _rows("results/suite/poly_certify_short.csv"):
        b = short[r["query_id"]]
        out.append(
            {
                "suite": "short",
                "group": "3-4",
                "poly": float(r["poly_bs2_p11"]),
                "flow": float(r["flow_bs2_p11"]),
                "ent": float(b["entropic_p11"]),
                "cover1": float(b["cover_bs1_p11"]),
                "cover2": float(b["cover_bs2_p11"]),
                "exact": st.get(r["query_id"]),
            }
        )
    ge = {r["path"]: r for r in _rows("results/gcare/gcare_eval.csv")}
    for r in _rows("results/gcare/gcare_poly_certify.csv"):
        b = ge[r["path"]]
        out.append(
            {
                "suite": f"gcare_{r['dataset']}",
                "group": "all",
                "poly": float(r["poly_bs2_p11"]),
                "flow": float(r["flow_bs2_p11"]),
                "ent": float(b["entropic_p11"]),
                "cover1": float(b["cover_bs1_p11"]),
                "cover2": float(b["cover_bs2_p11"]),
                "exact": _f(b["exact"]),
            }
        )
    sl = {
        (r["seed"], r["query_id"]): r
        for r in _rows("results/suite/slice_robustness.csv")
    }
    for r in _rows("results/suite/slice_poly_certify.csv"):
        b = sl[(r["seed"], r["query_id"])]
        out.append(
            {
                "suite": f"slice_seed{r['seed']}",
                "group": "6",
                "poly": float(r["poly_bs2_p11"]),
                "flow": float(r["flow_bs2_p11"]),
                "ent": float(b["entropic_p11"]),
                "cover1": float(b["cover_bs1_p11"]),
                "cover2": float(b["cover_bs2_p11"]),
                "exact": None,
            }
        )
    return out


def main() -> None:
    rows = _joined()
    keys = sorted({(str(r["suite"]), str(r["group"])) for r in rows})
    out: list[dict[str, object]] = []
    for suite, group in keys:
        rs = [r for r in rows if r["suite"] == suite and r["group"] == group]

        def med(f, rs=rs):  # type: ignore[no-untyped-def]
            return statistics.median(f(r) for r in rs)

        chk = [r for r in rs if r["exact"] is not None and float(str(r["exact"])) > 0]
        rec: dict[str, object] = {
            "suite": suite,
            "group": group,
            "n": len(rs),
            "median_poly_over_entropic": f"{med(lambda r: r['poly'] / r['ent']):.6g}",
            "median_flow_over_entropic": f"{med(lambda r: r['flow'] / r['ent']):.6g}",
            "median_cover2_over_entropic": f"{med(lambda r: r['cover2'] / r['ent']):.6g}",
            "median_cover1_over_entropic": f"{med(lambda r: r['cover1'] / r['ent']):.6g}",
            "median_cover2_over_poly": f"{med(lambda r: r['cover2'] / r['poly']):.6g}",
            "median_flow_over_poly": f"{med(lambda r: r['flow'] / r['poly']):.6g}",
            "n_poly_below_entropic": sum(r["poly"] < r["ent"] * (1 - 1e-9) for r in rs),
            "n_poly_eq_entropic": sum(
                abs(r["poly"] / r["ent"] - 1) <= 1e-9 for r in rs
            ),
            "n_exact": len(chk),
            "median_entropic_over_exact": (
                f"{statistics.median(r['ent'] / r['exact'] for r in chk):.6g}"
                if chk
                else ""
            ),
            "median_poly_over_exact": (
                f"{statistics.median(r['poly'] / r['exact'] for r in chk):.6g}"
                if chk
                else ""
            ),
            "n_poly_below_exact": sum(
                r["poly"] < float(str(r["exact"])) * (1 - 1e-9) for r in chk
            ),
        }
        out.append(rec)
        print(
            f"{suite:<14} {group:>4} n={len(rs):>3} poly/ent={rec['median_poly_over_entropic']:>10} "
            f"flow/ent={rec['median_flow_over_entropic']:>10} cover2/poly={rec['median_cover2_over_poly']:>8} "
            f"below={rec['n_poly_below_entropic']} poly/exact={rec['median_poly_over_exact']} "
            f"invalid={rec['n_poly_below_exact']}/{len(chk)}"
        )
    with open(ROOT / "results" / "suite" / "poly_headline.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    main()
