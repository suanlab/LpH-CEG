"""Hyperpath-structure enumeration behind the widened Step-5 certificate."""

from __future__ import annotations

from scripts.shannon_check_wide import structures
from scripts.shannon_implication_check import FAMILIES


def _idx(edges: list[tuple[str, str]]) -> tuple[list[str], dict[str, int]]:
    attrs = sorted({a for e in edges for a in e})
    return attrs, {a: i for i, a in enumerate(attrs)}


def test_triangle_bs1_is_all_orderings() -> None:
    _, idx = _idx(FAMILIES["triangle"])
    out, exhausted = structures(FAMILIES["triangle"], idx, 1)
    assert exhausted
    assert len(out) == 6  # 3! orderings, all connected, all distinct structures


def test_path4_bs1_excludes_disconnected_orderings() -> None:
    _, idx = _idx(FAMILIES["path4"])
    out, exhausted = structures(FAMILIES["path4"], idx, 1)
    assert exhausted
    assert len(out) == 4  # R0 R2 as first two steps is not a hyperpath


def test_larger_blocks_only_add_structures() -> None:
    edges = FAMILIES["cycle4"]
    _, idx = _idx(edges)
    counts = [len(structures(edges, idx, bs)[0]) for bs in (1, 2, 3)]
    assert counts[0] < counts[1] < counts[2]
    for bs in (1, 2, 3):
        for seq in structures(edges, idx, bs)[0]:
            # every block after the first has a non-empty interface
            assert all(st.iface_mask for st in seq[1:])
            assert seq[0].iface_mask == 0


def test_first_block_covers_only_new_attributes() -> None:
    edges = FAMILIES["k4"]
    _, idx = _idx(edges)
    for seq in structures(edges, idx, 2)[0]:
        covered = 0
        for st in seq:
            assert st.iface_mask == st.block_mask & covered
            covered |= st.block_mask
        assert covered == (1 << 4) - 1
