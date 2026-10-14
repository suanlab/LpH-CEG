"""Exact counts behind the real-frontier validity check.

`real_frontier.py` checks only its N_VALIDATE tightest bs=2 bounds against
exact join sizes and only prints them.  This companion counts ALL of its
queries (at the same 3K-row cap counting is cheap) and writes the exact
counts, flagging the N_VALIDATE tightest, so the paper's validity and
"true counts vs bounds" sentences are pinned to a CSV.

Output: results/ablation/real_frontier_truth.csv
"""

from __future__ import annotations

import csv
import json

from scripts.real_frontier import N_VALIDATE, ROOT, SUITE, build, exact_count


def main() -> None:
    reg = {q["query_id"]: q for q in json.loads(SUITE.read_text())["queries"]}
    path = ROOT / "results" / "ablation"
    with open(path / "real_frontier.csv") as fh:
        rows = sorted(csv.DictReader(fh), key=lambda r: float(r["bs2"]))
    out: list[dict[str, object]] = []
    for rank, r in enumerate(rows):
        built = build(reg[r["query_id"]])
        if built is None:
            continue
        t = exact_count(*built)
        out.append(
            {
                "query_id": r["query_id"],
                "bs1": r["bs1"],
                "bs2": r["bs2"],
                "true_count": t if t >= 0 else "",
                # t < 0: the join exceeded the counting cap -- unknown, not invalid
                "valid": ""
                if t < 0
                else (float(r["bs2"]) >= t and float(r["bs1"]) >= t),
                "tightest_20": rank < N_VALIDATE,
            }
        )
        print(r["query_id"], r["bs2"], t)
    with open(path / "real_frontier_truth.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {path / 'real_frontier_truth.csv'}")


if __name__ == "__main__":
    main()
