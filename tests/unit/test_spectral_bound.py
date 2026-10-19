"""The trace-Holder spectral bounds never fall below the cycle count."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from scripts.spectral_bound import _certified_sm_pow, _schatten_m_pow


def _holder(mats: list[csr_matrix], fn) -> float:  # type: ignore[no-untyped-def]
    k = len(mats)
    m = k // 2
    best = math.inf
    for off in (0, 1):
        log = 0.0
        for j in range(m):
            b = mats[(off + 2 * j) % k] @ mats[(off + 2 * j + 1) % k]
            v = fn(b, m)
            log += math.log2(v) / m if v > 0 else -math.inf
        best = min(best, log)
    return 2.0**best


@pytest.mark.parametrize("k", [4, 6, 8, 10])
def test_spectral_bounds_dominate_the_trace(k: int) -> None:
    rng = np.random.default_rng(k)
    for _ in range(15):
        n = int(rng.integers(20, 45))
        mats = [
            csr_matrix((rng.random((n, n)) < rng.uniform(0.03, 0.2)).astype(float))
            for _ in range(k)
        ]
        prod = mats[0]
        for a in mats[1:]:
            prod = prod @ a
        trace = float(prod.diagonal().sum())
        cert = _holder(mats, _certified_sm_pow)
        lanczos = _holder(mats, _schatten_m_pow)
        assert cert >= trace * (1 - 1e-9)
        assert lanczos >= trace * (1 - 1e-9)


def test_certified_schatten_bound_dominates_exact() -> None:
    rng = np.random.default_rng(0)
    for m in (3, 4, 5):
        b = csr_matrix(rng.integers(0, 4, size=(30, 25)).astype(float))
        s = np.linalg.svd(b.toarray(), compute_uv=False)
        assert _certified_sm_pow(b, m) >= float((s**m).sum()) * (1 - 1e-9)
