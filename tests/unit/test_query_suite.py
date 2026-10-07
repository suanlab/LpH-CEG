"""The pre-registered query suite must stay frozen.

The suite's anti-cherry-picking guarantee is only worth something if a
later edit is detectable.  These tests fail if the registry drifts from
its recorded hash, if the enumeration rule stops being deterministic, or
if the suite falls below the size the plan's 8/21 gate requires.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
SUITE = ROOT / "benchmarks" / "query_suite_v1.json"
DIGEST = ROOT / "benchmarks" / "query_suite_v1.sha256"

MIN_QUERIES = 30  # plan gate: >=30 cyclic queries over >=2 public workloads


@pytest.fixture(scope="module")
def registry() -> dict:
    return json.loads(SUITE.read_text())


def test_suite_matches_recorded_hash() -> None:
    """The registry file must hash to the value pinned at registration."""
    expected = DIGEST.read_text().split()[0]
    actual = hashlib.sha256(SUITE.read_bytes()).hexdigest()
    assert actual == expected, (
        "query_suite_v1.json no longer matches its registered hash; "
        "the suite was edited after pre-registration"
    )


def test_gate_size_and_workloads(registry: dict) -> None:
    assert registry["n_queries"] >= MIN_QUERIES
    assert len(registry["workloads"]) >= 2
    assert len(registry["queries"]) == registry["n_queries"]


def test_query_ids_unique(registry: dict) -> None:
    ids = [q["query_id"] for q in registry["queries"]]
    assert len(ids) == len(set(ids))


def test_hetionet_queries_are_type_consistent(registry: dict) -> None:
    """Every cycle must chain: each relation's endpoints are its two vars."""
    from scripts.register_query_suite import HETIONET_RELATIONS

    checked = 0
    for q in registry["queries"]:
        if q["dataset"] != "hetionet":
            continue
        rels, var_types = q["relations"], q["var_types"]
        k = len(rels)
        for i, rel in enumerate(rels):
            src, tgt = HETIONET_RELATIONS[rel]
            endpoints = {var_types[i], var_types[(i + 1) % k]}
            assert {src, tgt} == endpoints, f"{q['query_id']}: {rel} breaks typing"
        checked += 1
    assert checked > 0


def test_enumeration_is_deterministic(registry: dict) -> None:
    """Re-running the rule must reproduce the registry exactly."""
    from scripts.register_query_suite import (
        enumerate_epinions_shapes,
        enumerate_hetionet_cycles,
    )

    regenerated = enumerate_hetionet_cycles() + enumerate_epinions_shapes()
    assert regenerated == registry["queries"]
