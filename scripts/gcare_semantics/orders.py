from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import pandas as pd

R = str(ROOT / "data" / "gcare") + "/"
ev = pd.read_csv(str(ROOT / "results" / "gcare" / "gcare_eval.csv"))
for top in ["Petal_9", "Petal_12"]:
    g = ev[(ev.dataset == "yago") & (ev.topology == top)].copy()
    g["f"] = g.path.str.split("/").str[-1]
    g["s"] = g.path.map(lambda p: int(open(R + "queries/" + p).readline().split()[3]))
    m = g.f.str.extract(r"Q_(\d+)_(\d+)")
    g["b"] = m[0].astype(int)
    g["j"] = m[1].astype(int)
    orders = {
        "numeric": g.sort_values(["b", "j"]),
        "lex": g.sort_values("f"),
        "s": g.sort_values("s"),
        "bs": g.sort_values(["b", "s"]),
    }
    for oa, A in orders.items():
        for ob, B in orders.items():
            tr = A.gcare_true.tolist()
            ex = B.exact.tolist()
            for off in range(-3, 4):
                hits = sum(
                    1
                    for i in range(len(tr))
                    if 0 <= i + off < len(ex) and ex[i + off] == tr[i]
                )
                if hits >= 6:
                    print(
                        top,
                        "truth order",
                        oa,
                        "count order",
                        ob,
                        "offset",
                        off,
                        "hits",
                        hits,
                        "/",
                        len(tr),
                    )
