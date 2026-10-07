"""The pre-registered long-cycle suite (|V| in {6, 8, 10}) must stay frozen.

Same guarantee as `test_query_suite.py`, for the v2-long registry: the
file must hash to its pinned digest, the hash-order selection rule must
be reproducible from scratch, and every admitted cycle must be
type-consistent on the Hetionet metagraph.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
SUITE = ROOT / "benchmarks" / "query_suite_v2_long.json"
DIGEST = ROOT / "benchmarks" / "query_suite_v2_long.sha256"


@pytest.fixture(scope="module")
def registry() -> dict:
    return json.loads(SUITE.read_text())


def test_long_suite_matches_recorded_hash() -> None:
    expected = DIGEST.read_text().split()[0]
    actual = hashlib.sha256(SUITE.read_bytes()).hexdigest()
    assert actual == expected, "query_suite_v2_long.json was edited after registration"


def test_long_suite_shape(registry: dict) -> None:
    from scripts.register_long_suite import LONG_LENGTHS, N_PER_LENGTH

    assert registry["n_queries"] == len(LONG_LENGTHS) * N_PER_LENGTH
    for k in LONG_LENGTHS:
        assert sum(1 for q in registry["queries"] if q["n_vars"] == k) == N_PER_LENGTH
    ids = [q["query_id"] for q in registry["queries"]]
    assert len(ids) == len(set(ids))


def test_hash_order_selection_is_reproducible(registry: dict) -> None:
    """Re-running the rule at k=6 must yield exactly the registered queries."""
    from scripts.register_long_suite import (
        N_PER_LENGTH,
        enumerate_long_cycles,
        select_by_hash,
    )

    fresh = select_by_hash(enumerate_long_cycles(6), N_PER_LENGTH)
    assert [q["relations"] for q in fresh] == [
        q["relations"] for q in registry["queries"] if q["n_vars"] == 6
    ]
    assert registry["enumerated_per_length"]["cycle6"] == len(enumerate_long_cycles(6))


def test_long_cycles_are_type_consistent(registry: dict) -> None:
    from scripts.register_query_suite import HETIONET_RELATIONS

    for q in registry["queries"]:
        rels, types = q["relations"], q["var_types"]
        k = len(rels)
        assert len(set(rels)) == k
        for i, rel in enumerate(rels):
            want = (types[i], types[(i + 1) % k])
            native = HETIONET_RELATIONS[rel]
            assert native in (want, (want[1], want[0])), (q["query_id"], i)
