"""Spectral (trace-Holder) bounds for the cycle suites.

For a k-cycle of binary relations R_1(v_1,v_2), ..., R_k(v_k,v_1) the
homomorphism count is a trace, |Q| = Tr(A_1 A_2 ... A_k), with A_i the 0/1
incidence matrix of R_i (rows v_i, columns v_{i+1}).  Grouping consecutive
pairs into blocks B_j = A_{2j-1} A_{2j} (two pairings, offset 0 or 1) and
applying Holder's inequality for Schatten norms with m = k/2 factors,

    |Q| = Tr(B_1 ... B_m) <= prod_j ||B_j||_{S_m}.

||B_j||_{S_2} = ||B_j||_F is the paper's l2 norm of the block's
common-neighbour (a-c path) counts.  We bound ||B||_{S_m}^m = sum_i s_i^m
without forming B^T B: the top r singular values by Lanczos (``svds``), the
rest by the remaining Frobenius mass,

    sum_i s_i^m <= sum_{i<=r} s_i^m + s_{r+1}^(m-2) (||B||_F^2 - sum_{i<=r} s_i^2),

inflated by a relative 1e-6 against Lanczos error.  This is a valid upper
bound on the true count whenever the singular values are accurate, and it
is compared with the bs<=2 polymatroid bound (eleven norms) of
``poly_certify.csv``, the entropic LpBound and the exact count.

Exploratory: the statistic (top singular values of block products) is
richer than the bs=2 catalog, so this measures a further statistics step,
not a better program on the same statistics.

Output: results/spectral/spectral_long_suite.csv
"""

from __future__ import annotations

import csv
import json
import math
import multiprocessing
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from scripts.pset_fair import SUITES

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "spectral"
TOP_R = 24
SAFETY = 1 + 1e-6
WORKERS = 8


def _schatten_m_pow(b: csr_matrix, m: int) -> float:
    """Upper bound on ||B||_{S_m}^m = sum of s_i^m (m >= 2)."""
    fro2 = float(b.multiply(b).sum())
    if fro2 == 0.0:
        return 0.0
    if m == 2:
        return fro2
    n = min(b.shape)
    if n <= TOP_R + 2:  # small: exact singular values
        s_all = np.linalg.svd(b.toarray(), compute_uv=False)
        return float(np.sum(s_all**m)) * SAFETY
    r = TOP_R
    s = np.sort(
        svds(b.astype(float), k=r + 1, return_singular_vectors=False, random_state=0)
    )[::-1]
    s = s * SAFETY
    head = float(np.sum(s[:r] ** m))
    rest = max(fro2 - float(np.sum(s[:r] ** 2)), 0.0)
    return head + float(s[r]) ** (m - 2) * rest


def _certified_sm_pow(b: csr_matrix, m: int) -> float:
    """Certified upper bound sigma_ub^(m-2) ||B||_F^2 >= sum_i s_i^m.

    sigma_ub^2 is a Collatz-Wielandt bound on lambda_max(B^T B): for the
    nonnegative matrix M = B^T B and any positive x,
    lambda_max(M) <= max_i (M x)_i / x_i.  x is the (abs) top right singular
    vector from Lanczos plus a small positive floor; the bound holds for any
    such x, so Lanczos accuracy does not affect validity.
    """
    fro2 = float(b.multiply(b).sum())
    if fro2 == 0.0:
        return 0.0
    if m == 2:
        return fro2
    if min(b.shape) > 2:
        _, _, vt = svds(b.astype(float), k=1, random_state=0)
        x = np.abs(vt[0]) + 1e-9
    else:
        x = np.ones(b.shape[1])
    mx = b.T @ (b @ x)
    lam_ub = float(np.max(mx / x))
    return lam_ub ** ((m - 2) / 2) * fro2


def evaluate(item: tuple[str, dict]) -> dict[str, object] | None:  # type: ignore[type-arg]
    suite, q = item
    built = SUITES[suite][1](q)
    if built is None:
        return None
    query, data = built
    k = len(query.relations)
    idx: dict[str, dict[str, int]] = {}

    def ix(var: str, val: str) -> int:
        d = idx.setdefault(var, {})
        return d.setdefault(val, len(d))

    edges = []
    for rel in query.relations:
        i = int(rel.name.split("_")[0][1:])
        u, v = f"v{i}", f"v{(i + 1) % k}"
        attrs, rows = data[rel.name]
        edges.append((i, u, v, {(r[attrs.index(u)], r[attrs.index(v)]) for r in rows}))
    edges.sort()
    for _, u, v, pairs in edges:
        for a, b in pairs:
            ix(u, a)
            ix(v, b)
    mats = []
    for _, u, v, pairs in edges:
        r = [idx[u][a] for a, _ in pairs]
        c = [idx[v][b] for _, b in pairs]
        mats.append(
            csr_matrix((np.ones(len(r)), (r, c)), shape=(len(idx[u]), len(idx[v])))
        )
    m = k // 2
    best_s2 = best_sm = best_cert = math.inf
    for off in (0, 1):
        log_s2 = log_sm = log_cert = 0.0
        for j in range(m):
            b = mats[(off + 2 * j) % k] @ mats[(off + 2 * j + 1) % k]
            fro2 = float(b.multiply(b).sum())
            if fro2 == 0.0:
                log_s2 = log_sm = log_cert = -math.inf
                break
            log_s2 += 0.5 * math.log2(fro2)
            log_sm += math.log2(_schatten_m_pow(b, m)) / m
            log_cert += math.log2(_certified_sm_pow(b, m)) / m
        best_s2 = min(best_s2, log_s2)
        best_sm = min(best_sm, log_sm)
        best_cert = min(best_cert, log_cert)
    return {
        "suite": suite,
        "query_id": q["query_id"],
        "n_vars": q["n_vars"],
        "holder_s2": f"{2.0**best_s2:.6e}",
        "holder_sm": f"{2.0**best_sm:.6e}",
        "holder_certified": f"{2.0**best_cert:.6e}",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    items = [
        (suite, q)
        for suite, (qpath, _) in SUITES.items()
        for q in json.loads(qpath.read_text())["queries"]
        if len(sys.argv) < 2 or suite in sys.argv[1:]
    ]
    rows: list[dict[str, object]] = []
    with multiprocessing.Pool(processes=WORKERS, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(evaluate, items):
            if r is None:
                continue
            rows.append(r)
            rows.sort(key=lambda x: (str(x["suite"]), str(x["query_id"])))
            with open(OUT / "spectral_long_suite.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            print(r["suite"], r["query_id"], flush=True)
    print("done")


if __name__ == "__main__":
    main()
