"""H3: can we tell, *before* collecting degree sequences, that the LP will not help?

Motivation.  Statistics collection dominates estimation cost -- 22-27 ms
against ~1.9 ms to solve the LP (`results/cost/stats_cost.csv`), i.e. the
LP is only ~7% of the bill.  And on the pre-registered suite 112 of 370
real queries (30%) improve by exactly 1.00x, so for those the entire
statistics bill buys nothing.

If a *catalog-only* signal can identify them up front, that 30% turns
from a negative result into a saving: skip the statistics, return the
max-degree bound, and spend nothing.

The predictor is deliberately restricted to what a query optimizer knows
without touching the data: query shape, and per-relation row counts of
the kind any catalog already stores.  Degree sequences, norms and LP duals
are all withheld -- using them would defeat the purpose, since computing
them *is* the cost we are trying to avoid.

Protocol: stratified 70/30 split, logistic regression, and the decision
threshold chosen on TRAIN only.  We report held-out precision/recall for
the "skip" decision plus the resulting cost saving, and compare against
the majority-class baseline so the number means something.

Output: results/cost/skip_predictor.csv
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUITE_EVAL = ROOT / "results" / "suite" / "suite_eval.csv"
HETIONET = ROOT / "data" / "hetionet"

# Fraction of total estimation cost that statistics collection represents
# (measured in results/cost/stats_cost.csv: ~22-27 ms of ~24-29 ms).
STATS_COST_SHARE = 0.93

HETIONET_FILES = {
    "AdG": "AdG_edges.tsv",
    "AuG": "AuG_edges.tsv",
    "CdG": "CdG_edges.tsv",
    "CuG": "CuG_edges.tsv",
    "DdG": "DdG_edges.tsv",
    "DlA": "DlA_edges.tsv",
    "DrD": "DrD_edges.tsv",
    "DuG": "DuG_edges.tsv",
    "cbg": "cbg_edges.tsv",
    "crc": "crc_edges.tsv",
    "ctd": "ctd_edges.tsv",
    "dag": "dag_edges.tsv",
    "gig": "gig_edges.tsv",
}
#: Relations whose two endpoints have the same node type.  A same-type
#: relation can close a cycle without introducing a new variable, which is
#: the structural situation Theorem 3 says yields no improvement.
SAME_TYPE = {"gig", "crc", "DrD"}

MAX_ROWS = 20000  # the cap the suite evaluation used


def relation_sizes() -> dict[str, int]:
    """Row count per relation, as a catalog would already know it."""
    sizes: dict[str, int] = {}
    for rel, fname in HETIONET_FILES.items():
        path = HETIONET / fname
        if not path.exists():
            continue
        with open(path) as fh:
            fh.readline()  # header
            n = sum(1 for _ in fh)
        sizes[rel] = min(n, MAX_ROWS)
    return sizes


def relation_maxdeg() -> dict[str, tuple[float, float]]:
    """Max out/in degree per relation.

    These are NOT extra work: the max-degree bound we compare against needs
    them already, so a predictor built on them can still skip the expensive
    part -- the full l_p degree sequences.
    """
    out: dict[str, tuple[float, float]] = {}
    for rel, fname in HETIONET_FILES.items():
        path = HETIONET / fname
        if not path.exists():
            continue
        du: dict[str, int] = {}
        dv: dict[str, int] = {}
        with open(path) as fh:
            fh.readline()
            for i, line in enumerate(fh):
                if i >= MAX_ROWS:
                    break
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    du[parts[0]] = du.get(parts[0], 0) + 1
                    dv[parts[1]] = dv.get(parts[1], 0) + 1
        out[rel] = (
            float(max(du.values(), default=1)),
            float(max(dv.values(), default=1)),
        )
    return out


def features(
    rels: list[str],
    n_vars: int,
    sizes: dict[str, int],
    md: dict[str, tuple[float, float]] | None = None,
) -> list[float]:
    """Shape + row counts; optionally max-degrees (free, see relation_maxdeg)."""
    ns = [sizes.get(r, 1) for r in rels]
    logs = [math.log10(max(n, 1)) for n in ns]
    mx, mn = max(logs), min(logs)
    mean = sum(logs) / len(logs)
    var = sum((x - mean) ** 2 for x in logs) / len(logs)
    n_same = sum(1 for r in rels if r in SAME_TYPE)
    return [
        1.0,  # bias
        float(n_vars),
        mx,
        mn,
        mx - mn,  # size spread, the catalog's crude proxy for skew
        mean,
        math.sqrt(var),
        float(n_same),
        float(len(set(rels))) / len(rels),  # relation repetition
        float(sum(1 for x in logs if x > mean)),  # count of "large" relations
        *_maxdeg_features(rels, sizes, md),
    ]


def _maxdeg_features(
    rels: list[str],
    sizes: dict[str, int],
    md: dict[str, tuple[float, float]] | None,
) -> list[float]:
    """Density features from max-degree: rows/maxdeg is a cheap skew proxy."""
    if md is None:
        return []
    mds = [max(md.get(r, (1.0, 1.0))) for r in rels]
    lm = [math.log10(max(m, 1.0)) for m in mds]
    # rows / maxdeg ~ number of distinct interface values: low means one hub
    ratios = [
        math.log10(max(sizes.get(r, 1), 1) / max(max(md.get(r, (1.0, 1.0))), 1.0))
        for r in rels
    ]
    return [
        max(lm),
        min(lm),
        max(lm) - min(lm),
        sum(lm) / len(lm),
        max(ratios),
        min(ratios),
        max(ratios) - min(ratios),
    ]


def fit_logistic(
    xs: list[list[float]], ys: list[int], epochs: int = 4000, lr: float = 0.08
) -> list[float]:
    d = len(xs[0])
    w = [0.0] * d
    n = len(xs)
    # standardise (except bias) so a single learning rate works
    for j in range(1, d):
        col = [x[j] for x in xs]
        mu = sum(col) / n
        sd = (sum((c - mu) ** 2 for c in col) / n) ** 0.5 or 1.0
        for x in xs:
            x[j] = (x[j] - mu) / sd
    for _ in range(epochs):
        g = [0.0] * d
        for x, y in zip(xs, ys, strict=True):
            z = sum(wi * xi for wi, xi in zip(w, x, strict=True))
            pred = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            err = pred - y
            for j in range(d):
                g[j] += err * x[j]
        for j in range(d):
            w[j] -= lr * g[j] / n
    return w


def predict(w: list[float], x: list[float]) -> float:
    z = sum(wi * xi for wi, xi in zip(w, x, strict=True))
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def standardise_like(
    train: list[list[float]], test: list[list[float]]
) -> None:  # pragma: no cover - helper
    """Apply train statistics to test rows, in place."""
    n, d = len(train), len(train[0])
    for j in range(1, d):
        col = [x[j] for x in train]
        mu = sum(col) / n
        sd = (sum((c - mu) ** 2 for c in col) / n) ** 0.5 or 1.0
        for x in test:
            x[j] = (x[j] - mu) / sd


def evaluate(data: list[tuple[list[float], int, str]], label: str) -> dict[str, object]:
    """Stratified 70/30 fit + held-out report for one feature set."""
    rng = random.Random(20260817)
    pos = [d for d in data if d[1] == 1]
    neg = [d for d in data if d[1] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    cut_p, cut_n = int(0.7 * len(pos)), int(0.7 * len(neg))
    train = pos[:cut_p] + neg[:cut_n]
    test = pos[cut_p:] + neg[cut_n:]
    rng.shuffle(train)

    xs_tr = [list(d[0]) for d in train]
    ys_tr = [d[1] for d in train]
    xs_te = [list(d[0]) for d in test]
    standardise_like([list(d[0]) for d in train], xs_te)
    w = fit_logistic(xs_tr, ys_tr)

    # Threshold on TRAIN only.  Skipping a query that WOULD have improved is
    # the costly error, so precision is what we protect.
    best_t, best_rate = 0.99, 0.0
    for t in [i / 100 for i in range(50, 100)]:
        tp = sum(
            1
            for x, y in zip(xs_tr, ys_tr, strict=True)
            if predict(w, x) >= t and y == 1
        )
        fp = sum(
            1
            for x, y in zip(xs_tr, ys_tr, strict=True)
            if predict(w, x) >= t and y == 0
        )
        if tp + fp == 0:
            continue
        if tp / (tp + fp) >= 0.95 and (tp + fp) / len(ys_tr) > best_rate:
            best_t, best_rate = t, (tp + fp) / len(ys_tr)

    ys_te = [d[1] for d in test]
    preds = [predict(w, x) for x in xs_te]
    tp = sum(1 for q, y in zip(preds, ys_te, strict=True) if q >= best_t and y == 1)
    fp = sum(1 for q, y in zip(preds, ys_te, strict=True) if q >= best_t and y == 0)
    fn = sum(1 for q, y in zip(preds, ys_te, strict=True) if q < best_t and y == 1)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    skip_rate = (tp + fp) / len(ys_te)
    saving = skip_rate * STATS_COST_SHARE
    bar = prec >= 0.95 and saving >= 0.30

    print(f"  {label}")
    print(f"    threshold {best_t:.2f} (train-only)  held-out n={len(ys_te)}")
    print(f"    skip-rate={skip_rate:.3f}  precision={prec:.3f}  recall={rec:.3f}")
    print(
        f"    statistics cost avoided: {saving * 100:.1f}%  -> bar {'MET' if bar else 'NOT MET'}"
    )
    return {
        "feature_set": label,
        "threshold": f"{best_t:.2f}",
        "heldout_n": len(ys_te),
        "skip_rate": f"{skip_rate:.4f}",
        "precision": f"{prec:.4f}",
        "recall": f"{rec:.4f}",
        "stats_cost_avoided": f"{saving:.4f}",
        "bar_met": bar,
    }


def main() -> None:
    print("H3: when can we skip the expensive l_p statistics?")
    print("=" * 74)

    sizes = relation_sizes()
    if not sizes:
        raise SystemExit("Hetionet relations not found; cannot run H3")
    maxdeg = relation_maxdeg()

    with open(SUITE_EVAL) as fh:
        rows = [
            r
            for r in csv.DictReader(fh)
            if r["dataset"] == "hetionet" and r["improvement_vs_cegm"]
        ]

    n_pos = sum(1 for r in rows if float(r["improvement_vs_cegm"]) <= 1.001)
    print(f"  {len(rows)} queries; {n_pos} gain nothing from the LP\n")

    out_rows = []
    for label, md in (
        ("A. catalog only (row counts + shape)", None),
        ("B. + max-degree (already computed by the baseline)", maxdeg),
    ):
        data = [
            (
                features(r["relations"].split("|"), int(r["n_vars"]), sizes, md),
                1 if float(r["improvement_vs_cegm"]) <= 1.001 else 0,
                r["query_id"],
            )
            for r in rows
        ]
        out_rows.append(evaluate(data, label))

    print("\n  Reading: if B clears the bar and A does not, the decisive signal is")
    print("  degree skew, not relation size -- and it is available for free.")

    out = ROOT / "results" / "cost"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "skip_predictor.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(out_rows)
    print(f"\nWrote {out / 'skip_predictor.csv'}")


if __name__ == "__main__":
    main()
