"""The G-CARE cyclic-query list must stay frozen; helpers behave as stated."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
REG = ROOT / "benchmarks" / "gcare_cyclic.json"
DIGEST = ROOT / "benchmarks" / "gcare_cyclic.sha256"
QUERIES = ROOT / "data" / "gcare" / "queries"


def test_gcare_registry_matches_recorded_hash() -> None:
    expected = DIGEST.read_text().split()[0]
    assert hashlib.sha256(REG.read_bytes()).hexdigest() == expected


def test_gcare_registry_shape() -> None:
    reg = json.loads(REG.read_text())
    qs = reg["queries"]
    assert reg["n_queries"] == len(qs) == 607
    assert {q["dataset"] for q in qs} == {"aids", "yago"}
    assert all(q["n_vars"] >= 3 for q in qs)


def test_is_cyclic() -> None:
    from scripts.gcare_suite import is_cyclic

    tri = {0: (-1, -1), 1: (-1, -1), 2: (-1, -1)}
    assert is_cyclic(tri, [(0, 1, 0), (1, 2, 0), (2, 0, 0)])
    assert not is_cyclic(tri, [(0, 1, 0), (1, 2, 0)])
    # a repeated vertex pair (two edge labels between the same vertices)
    assert is_cyclic(tri, [(0, 1, 0), (1, 0, 1), (1, 2, 0)])


def test_walk_orders_are_permutations() -> None:
    from scripts.gcare_suite import walk_orders

    attrs = ["v0", "v1", "v2", "v3"]
    orders = walk_orders(
        attrs, [("v0", "v1"), ("v1", "v2"), ("v2", "v3"), ("v3", "v0")]
    )
    assert orders
    assert all(sorted(o) == attrs for o in orders)


@pytest.mark.skipif(not QUERIES.exists(), reason="G-CARE queries not downloaded")
def test_registry_rederives_from_rule() -> None:
    from scripts.gcare_suite import registry

    assert registry() == json.loads(REG.read_text())["queries"]
