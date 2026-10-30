"""Regression checks for upper-bound preconditions found in the paper audit."""

from __future__ import annotations

import pytest

from src.core.block import enumerate_blocks
from src.estimators.lp_bound import optimize_hyperpath_weights
from src.estimators.lph_ceg import estimate_lph_ceg
from src.estimators.pessimistic import estimate_pessimistic
from src.statistics.lp_stats import compute_all_lp_block_extensions
from src.utils.types import QueryHypergraph, Relation


def _path() -> tuple[
    QueryHypergraph, dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]]
]:
    data = {
        "R": (("a", "b"), [("a0", "b0"), ("a1", "b1")]),
        "S": (("b", "c"), [("b0", "c0"), ("b1", "c1")]),
    }
    query = QueryHypergraph(
        relations=(
            Relation("R", frozenset({"a", "b"}), 2, selectivity=0.5),
            Relation("S", frozenset({"b", "c"}), 2, selectivity=0.5),
        )
    )
    return query, data


def test_missing_statistics_cannot_become_a_free_fallback() -> None:
    query, data = _path()
    hp = estimate_pessimistic(
        query, data, max_block_size=1, apply_selectivity=False
    ).hyperpath
    with pytest.raises(KeyError, match="Missing Lp statistics"):
        optimize_hyperpath_weights(query, hp, {})


def test_correlated_selectivity_product_is_rejected() -> None:
    query, data = _path()
    # Selecting the shared b0 row in both relations leaves one join tuple.
    # Multiplying two marginal 0.5 rates into the unfiltered bound gives 0.5.
    with pytest.raises(ValueError, match="selectivities"):
        estimate_lph_ceg(query, data, max_block_size=1, apply_selectivity=True)


def test_explicitly_filtered_tuples_preserve_the_bound() -> None:
    query, data = _path()
    filtered = {name: (schema, rows[:1]) for name, (schema, rows) in data.items()}
    result = estimate_lph_ceg(query, filtered, max_block_size=1)
    assert result.estimate >= 1.0
    stats = compute_all_lp_block_extensions(query, enumerate_blocks(query, 1), filtered)
    hp = estimate_pessimistic(
        query, filtered, max_block_size=1, apply_selectivity=False
    ).hyperpath
    assert optimize_hyperpath_weights(query, hp, stats).bound >= 1.0


def test_full_molp_is_tighter_than_the_chain_on_the_audit_example() -> None:
    """The chain baseline (CEG_M label) is a restriction of MOLP: 24 vs 6."""
    from scripts.molp_full import full_molp

    query = QueryHypergraph(
        relations=(
            Relation("R", frozenset("ab"), 2),
            Relation("S", frozenset("bc"), 17),
            Relation("T", frozenset("cd"), 3),
        )
    )
    data = {
        "R": (("a", "b"), [("a0", "b0"), ("a1", "b0")]),
        "S": (
            ("b", "c"),
            [("b0", "c0")]
            + [(f"b{i}", f"c{j}") for i in range(1, 5) for j in range(1, 5)],
        ),
        "T": (("c", "d"), [("c0", f"d{j}") for j in range(3)]),
    }
    chain = estimate_pessimistic(query, data, max_block_size=1).estimate
    assert chain == pytest.approx(24.0)
    assert 2.0 ** full_molp(query, data) == pytest.approx(6.0)


def test_full_molp_matches_the_shared_molp_module() -> None:
    """scripts/molp_full.py agrees with src.core.molp (synced from H-CEG)."""
    import math

    from scripts.molp_full import full_molp
    from scripts.revision_experiments import make_cycle_data
    from src.core.molp import estimate_molp

    for k in (4, 5):
        query, data = make_cycle_data(k, 60, 200, alpha=1.0, seed=1000)
        ours = full_molp(query, data)
        assert ours == pytest.approx(math.log2(estimate_molp(data).bound), abs=1e-9)


def test_empty_relation_gives_zero_everywhere() -> None:
    """An empty relation: every estimator returns the exact count, 0."""
    from src.estimators.lpbound_entropic import estimate_lpbound_entropic
    from src.utils.query import triangle_query

    query = triangle_query()
    data = {
        rel.name: (
            tuple(sorted(rel.attributes)),
            [] if rel.name == "R" else [("1", "2")],
        )
        for rel in query.relations
    }
    assert estimate_pessimistic(query, data, max_block_size=1).estimate == 0.0
    assert estimate_lph_ceg(query, data, max_block_size=1).estimate == 0.0
    assert estimate_lpbound_entropic(query, data) == 0.0


def test_exact_count_uses_set_semantics() -> None:
    """Duplicate input rows do not inflate the SQL truth (estimators use sets)."""
    from scripts.evaluate_long_suite import exact_count
    from src.utils.query import cycle_query

    query = cycle_query(3)
    data = {}
    for i, rel in enumerate(query.relations):
        cols = (f"v{i}", f"v{(i + 1) % 3}")
        data[rel.name] = (cols, [("1", "1"), ("1", "1")])
    assert exact_count(query, data, timeout_s=30) == 1


def test_resume_refuses_a_changed_fingerprint(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """F15: a resumed run must use the same code, inputs and parameters."""
    import pytest as _pytest

    from scripts.run_fingerprint import guard

    out = tmp_path / "rows.csv"
    inp = tmp_path / "registry.json"
    inp.write_text("{}")
    script = "scripts/run_fingerprint.py"
    guard(out, script, [inp], {"P": [1, 2]}, resume=False)
    out.write_text("a\n1\n")
    guard(out, script, [inp], {"P": [1, 2]}, resume=True)  # unchanged: ok
    with _pytest.raises(SystemExit, match="params"):
        guard(out, script, [inp], {"P": [1, 2, 3]}, resume=True)
    inp.write_text('{"q": 1}')
    with _pytest.raises(SystemExit, match=r"registry\.json"):
        guard(out, script, [inp], {"P": [1, 2]}, resume=True)
    (tmp_path / "rows.csv.fingerprint.json").unlink()
    with _pytest.raises(SystemExit, match="no fingerprint"):
        guard(out, script, [inp], {"P": [1, 2]}, resume=True)
