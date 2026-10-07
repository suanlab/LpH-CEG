"""Exact counts behind the real-frontier validity check.

`real_frontier.py` checks its N_VALIDATE tightest bs=2 bounds against exact
join sizes but only prints them.  This companion re-derives the same set
(the N_VALIDATE smallest ``bs2`` values in ``real_frontier.csv``) and writes
the exact counts, so the paper's "true counts vs bounds" sentence is pinned
to a CSV rather than to a docstring.

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
    for r in rows[:N_VALIDATE]:
        built = build(reg[r["query_id"]])
        if built is None:
            continue
        t = exact_count(*built)
        out.append(
            {
                "query_id": r["query_id"],
                "bs1": r["bs1"],
                "bs2": r["bs2"],
                "true_count": t,
                "valid": t >= 0 and float(r["bs2"]) >= t and float(r["bs1"]) >= t,
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
