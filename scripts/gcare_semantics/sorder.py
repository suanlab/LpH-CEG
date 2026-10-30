from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys

import pandas as pd

R = str(ROOT / "data" / "gcare") + "/"
ev = pd.read_csv(str(ROOT / "results" / "gcare" / "gcare_eval.csv"))
ev = ev[ev.exact.notna()].copy()


def s_of(p):
    return int(open(R + "queries/" + p).readline().split()[3])


ev["s"] = ev.path.map(s_of)
m = ev.path.str.extract(r"Q_(\d+)_(\d+)\.txt")
ev["b"] = m[0].astype(int)
ev["j"] = m[1].astype(int)
ev["agree"] = ev.exact == ev.gcare_true


# within-bucket monotonicity of s in j
def inv(g):
    g = g.sort_values("j")
    s = g.s.tolist()
    # query is 'in order' if its s rank among bucket equals its j rank
    return pd.Series(
        (pd.Series(s).rank().values == range(1, len(s) + 1)), index=g.index
    )


ev["rank_ok"] = ev.groupby(["dataset", "topology", "b"], group_keys=False).apply(inv)
t = ev.groupby(["dataset", "topology"]).agg(
    n=("agree", "size"), agree=("agree", "sum"), rank_ok=("rank_ok", "sum")
)
print(t.to_string())
print(pd.crosstab([ev.dataset, ev.rank_ok], ev.agree))
if len(sys.argv) > 1:
    top = sys.argv[1]
    g = ev[(ev.dataset == "yago") & (ev.topology == top)].sort_values(["b", "s"])
    print(g[["path", "b", "j", "s", "exact", "gcare_true", "agree"]].to_string())
