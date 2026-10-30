from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import pandas as pd

ev = pd.read_csv(str(ROOT / "results" / "gcare" / "gcare_eval.csv"))
y = ev[(ev.dataset == "yago") & ev.exact.notna()].copy()
y["ex"] = y.exact.astype(int)
y["tr"] = y.gcare_true.astype(int)
for lo in [100, 1000, 10000]:
    for lab, sub in [("dis", y[y.ex != y.tr]), ("agree", y[y.ex == y.tr])]:
        s = sub[sub.tr >= lo]
        k = 0
        pairs = []
        for r in s.itertuples():
            same = y[(y.topology == r.topology) & (y.path != r.path) & (y.ex == r.tr)]
            if len(same):
                k += 1
                pairs.append(
                    (
                        r.path.split("/")[-2] + "/" + r.path.split("/")[-1],
                        list(same.path.str.split("/").str[-1]),
                    )
                )
        print(lo, lab, k, "/", len(s), pairs if lab == "agree" else "")
