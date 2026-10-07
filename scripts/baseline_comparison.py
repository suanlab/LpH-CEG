"""X4: Unified comparison against all implemented upper-bound baselines.

A common reviewer concern is comparison breadth. This script puts every
*guaranteed upper bound* method we implement side by side on one query
suite, so the bound ordering is explicit and reproducible:

* **AGM**~-- fractional edge cover, $\\prod |R_i|^{w_i}$ (cardinality only,
  no degree information).
* **CEG_M**~-- binary max-degree shortest-path bound ($bs{=}1$); equivalent
  to PANDA / MOLP restricted to binary queries.
* **Cover-LP (ours)**~-- the polynomial-size $\\ell_p$ block-cover LP at
  $bs{=}1$.
* **Entropic LpBound**~-- the genuine $2^{|V|}$ polymatroid LP (Zhang et
  al.), the tightest bound these statistics admit.

The expected (and verified) ordering on cyclic queries is
``CEG_M >= Cover-LP >= Entropic >= true``, with AGM incomparable in
general (it ignores degrees). We report each bound and the ratios that
matter: how much tighter our cover-LP is than CEG_M and AGM, and how
close it stays to the entropic optimum.

Learned / sampling estimators (FactorJoin, GNCE, COLOR, CorrBound) are
*point* estimators without guaranteed upper bounds, so they are not
directly comparable on this axis; we position them in the related-work
discussion rather than in this guaranteed-bound table.

Output: ``results/baselines/baseline_comparison.csv``.
"""

from __future__ import annotations

import csv
import importlib.util
import random
from pathlib import Path

from src.estimators.agm_bound import estimate_agm
from src.estimators.lpbound_entropic import MAX_ATTRIBUTES, estimate_lpbound_entropic
from src.estimators.lph_ceg import estimate_lph_ceg
from src.estimators.pessimistic import estimate_pessimistic
from src.utils.query import cycle_query

P_VALUES: tuple[float, ...] = (1.0, 2.0, float("inf"))
RelData = dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]]


def gen_edges(n_v: int, n_e: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    weights = [1.0 / (i + 1) for i in range(n_v)]
    total = sum(weights)
    probs = [w / total for w in weights]

    def pick() -> str:
        r = rng.random()
        c = 0.0
        for i, p in enumerate(probs):
            c += p
            if r <= c:
                return str(i)
        return str(n_v - 1)

    tuples: set[tuple[str, str]] = set()
    while len(tuples) < n_e:
        u, v = pick(), pick()
        if u != v:
            tuples.add((u, v))
    return sorted(tuples)


def _all_bounds(query: object, data: RelData) -> dict[str, float | None]:
    n_attr = len(query.all_attributes)  # type: ignore[attr-defined]
    entropic: float | None
    try:
        entropic = (
            estimate_lpbound_entropic(query, data, p_values=P_VALUES)
            if n_attr <= MAX_ATTRIBUTES
            else None
        )
    except ValueError:
        entropic = None
    return {
        "agm": estimate_agm(query, data),
        "ceg_m": estimate_pessimistic(query, data, max_block_size=1).estimate,
        "cover": estimate_lph_ceg(
            query, data, max_block_size=1, p_values=P_VALUES
        ).estimate,
        "entropic": entropic,
    }


def main() -> None:
    print("X4: Unified upper-bound baseline comparison")
    print("=" * 88)
    header = (
        f"  {'query':<22} {'AGM':>12} {'CEG_M':>12} {'Cover(ours)':>12} "
        f"{'Entropic':>12} {'vs CEG_M':>9} {'vs Ent':>7}"
    )
    print(header)
    print(f"  {'-' * 86}")

    rows_out: list[dict[str, object]] = []

    def record(label: str, query: object, data: RelData) -> None:
        b = _all_bounds(query, data)
        cover = b["cover"] or 1e-9
        vs_ceg = (b["ceg_m"] or 0) / cover
        vs_ent = cover / b["entropic"] if b["entropic"] else None

        def fmt(x: float | None) -> str:
            return f"{x:.3e}" if x is not None else "----"

        vs_ent_s = f"{vs_ent:.2f}x" if vs_ent is not None else "----"
        print(
            f"  {label:<22} {fmt(b['agm']):>12} {fmt(b['ceg_m']):>12} "
            f"{fmt(b['cover']):>12} {fmt(b['entropic']):>12} "
            f"{vs_ceg:>8.2f}x {vs_ent_s:>7}"
        )
        rows_out.append(
            {
                "query": label,
                "agm": fmt(b["agm"]),
                "ceg_m": fmt(b["ceg_m"]),
                "cover_lp": fmt(b["cover"]),
                "entropic": fmt(b["entropic"]),
                "cover_vs_cegm": f"{vs_ceg:.4f}",
                "cover_vs_entropic": f"{vs_ent:.4f}" if vs_ent is not None else "",
            }
        )

    # --- Synthetic cycles (deterministic power-law, 100V/400E) ---
    for k in (3, 4, 5, 6):
        q = cycle_query(k)
        data: RelData = {}
        for i, rel in enumerate(q.relations):
            data[rel.name] = (
                tuple(sorted(rel.attributes)),
                gen_edges(100, 400, 42 + i * 7),
            )
        record(f"Cycle-{k} (100V/400E)", q, data)

    # --- Real Hetionet cycles (reuse the X3 builder) ---
    crb_path = Path(__file__).resolve().parent / "cyclic_real_benchmark.py"
    if crb_path.exists() and (crb_path.parent.parent / "data" / "hetionet").exists():
        spec = importlib.util.spec_from_file_location("crb", crb_path)
        crb = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(crb)  # type: ignore[union-attr]
        for label, _sig, cyc in crb.CYCLES:
            q2, data2 = crb.build(cyc)
            record(f"Hetionet {label}", q2, data2)
    else:
        print("  [Hetionet data not found; synthetic-only comparison]")

    out_path = (
        Path(__file__).resolve().parent.parent
        / "results"
        / "baselines"
        / "baseline_comparison.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query",
                "agm",
                "ceg_m",
                "cover_lp",
                "entropic",
                "cover_vs_cegm",
                "cover_vs_entropic",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nWrote {out_path} ({len(rows_out)} rows)")


if __name__ == "__main__":
    main()
