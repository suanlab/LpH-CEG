"""The pre-registered IMDb cycle suite must stay frozen and build correctly.

Same guarantees as `test_long_suite.py`, for the v3-imdb registry: the file
hashes to its pinned digest, the hash-order rule re-derives the admitted
cycles, every cycle is type-consistent on the IMDb relations, and the query
builder orients each relation so that consecutive relations share exactly
their common variable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
SUITE = ROOT / "benchmarks" / "query_suite_v3_imdb.json"
DIGEST = ROOT / "benchmarks" / "query_suite_v3_imdb.sha256"
DATA = ROOT / "data" / "imdb_cycles"


@pytest.fixture(scope="module")
def registry() -> dict:
    return json.loads(SUITE.read_text())


def test_imdb_suite_matches_recorded_hash() -> None:
    expected = DIGEST.read_text().split()[0]
    actual = hashlib.sha256(SUITE.read_bytes()).hexdigest()
    assert actual == expected, "query_suite_v3_imdb.json was edited after registration"


def test_imdb_suite_shape(registry: dict) -> None:
    from scripts.register_imdb_suite import LENGTHS, N_PER_LENGTH

    assert registry["n_queries"] == len(LENGTHS) * N_PER_LENGTH
    for k in LENGTHS:
        assert sum(1 for q in registry["queries"] if q["n_vars"] == k) == N_PER_LENGTH
    ids = [q["query_id"] for q in registry["queries"]]
    assert len(ids) == len(set(ids))


def test_hash_order_selection_is_reproducible(registry: dict) -> None:
    """Re-running the rule at k=6 must yield exactly the registered queries."""
    from scripts.register_imdb_suite import N_PER_LENGTH, enumerate_cycles
    from scripts.register_long_suite import select_by_hash

    allk = enumerate_cycles(6)
    fresh = select_by_hash(allk, N_PER_LENGTH)
    assert [q["relations"] for q in fresh] == [
        q["relations"] for q in registry["queries"] if q["n_vars"] == 6
    ]
    assert registry["enumerated_per_length"]["cycle6"] == len(allk)


def test_imdb_cycles_are_type_consistent(registry: dict) -> None:
    from scripts.prepare_imdb_cycles import IMDB_RELATIONS

    for q in registry["queries"]:
        rels, types = q["relations"], q["var_types"]
        k = len(rels)
        # movies alternate with entities, each entity reached by one relation
        assert types[0::2] == ["M"] * (k // 2)
        assert rels[0::2] == rels[1::2]
        for i, rel in enumerate(rels):
            want = (types[i], types[(i + 1) % k])
            native = IMDB_RELATIONS[rel]
            assert native in (want, (want[1], want[0])), (q["query_id"], i)


@pytest.mark.skipif(not DATA.exists(), reason="derived IMDb data not prepared")
def test_builder_orients_relations_by_type(registry: dict) -> None:
    """Movie variables must hold movie ids, entity variables never."""
    from scripts.evaluate_imdb_suite import build_imdb_query

    for q in registry["queries"][:: len(registry["queries"]) // 6]:
        query, data = build_imdb_query(q)
        k = len(q["relations"])
        for i, rel in enumerate(query.relations):
            cols, rows = data[rel.name]
            for col_idx, var in enumerate(cols):
                j = int(var[1:])
                assert j in (i, (i + 1) % k)
                is_movie = q["var_types"][j] == "M"
                assert all((row[col_idx][0] == "M") == is_movie for row in rows[:50])
