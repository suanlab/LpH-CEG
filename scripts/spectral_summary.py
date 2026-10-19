"""Summarise spectral_long_suite.csv against the polymatroid bound and truth.

Output: results/spectral/spectral_summary.csv
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _rows(rel: str) -> list[dict[str, str]]:
    return list(csv.DictReader((ROOT / rel).read_text().splitlines()))


def main() -> None:
    poly = {
        (r["suite"], r["query_id"]): float(r["poly_bs2_p11"])
        for r in _rows("results/suite/poly_certify.csv")
    }
    truth: dict[str, float] = {}
    for rel in (
        "results/suite/long_suite_eval.csv",
        "results/imdb/imdb_suite_eval.csv",
    ):
        for r in _rows(rel):
            if r["true_count"] not in ("", "None"):
                truth[r["query_id"]] = float(r["true_count"])
    sp = _rows("results/spectral/spectral_long_suite.csv")
    out: list[dict[str, object]] = []
    groups = [*sorted({(r["suite"], r["n_vars"]) for r in sp}), ("all", "all")]
    for suite, nv in groups:
        rs = [
            r
            for r in sp
            if suite == "all" or (r["suite"] == suite and r["n_vars"] == nv)
        ]
        p = [poly[(r["suite"], r["query_id"])] for r in rs]
        cert = [float(r["holder_certified"]) for r in rs]
        lan = [float(r["holder_sm"]) for r in rs]
        chk = [
            (truth[r["query_id"]], pv, c, la)
            for r, pv, c, la in zip(rs, p, cert, lan, strict=True)
            if truth.get(r["query_id"], 0) > 0
        ]
        out.append(
            {
                "suite": suite,
                "n_vars": nv,
                "n": len(rs),
                "median_certified_over_poly": f"{statistics.median(c / q for c, q in zip(cert, p, strict=True)):.6g}",
                "n_certified_below_poly": sum(
                    c < q for c, q in zip(cert, p, strict=True)
                ),
                "median_lanczos_over_poly": f"{statistics.median(a / q for a, q in zip(lan, p, strict=True)):.6g}",
                "n_lanczos_below_poly": sum(a < q for a, q in zip(lan, p, strict=True)),
                "median_certified_gain_where_tighter": (
                    f"{statistics.median(q / c for c, q in zip(cert, p, strict=True) if c < q):.6g}"
                    if any(c < q for c, q in zip(cert, p, strict=True))
                    else ""
                ),
                "n_checked": len(chk),
                "median_poly_over_true": f"{statistics.median(q / t for t, q, _, _ in chk):.6g}",
                "median_min_certified_poly_over_true": f"{statistics.median(min(c, q) / t for t, q, c, _ in chk):.6g}",
                "median_min_lanczos_poly_over_true": f"{statistics.median(min(a, q) / t for t, q, _, a in chk):.6g}",
                "n_invalid_certified": sum(c < t * (1 - 1e-9) for t, _, c, _ in chk),
                "n_invalid_lanczos": sum(a < t * (1 - 1e-9) for t, _, _, a in chk),
            }
        )
    with open(
        ROOT / "results" / "spectral" / "spectral_summary.csv", "w", newline=""
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(r)


if __name__ == "__main__":
    main()
