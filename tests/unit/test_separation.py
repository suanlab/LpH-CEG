"""The separation example of the extended version, checked by enumeration.

R has incidence matrix M and S has M^T, for M = A (two disjoint K_{2,2}) and
M = B (the bipartite 8-cycle).  The two databases agree on every listed
statistic -- single-relation degrees, pair-join size, shared-attribute
products, one-side interface norms, and every tree-homomorphism count into
one relation (stars, claw pairs, caterpillars) -- but not on the squared
common-neighbour l2 norm (32 against 24) or the 4-cycle count.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

A = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]])
B = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]])


def _tree_homs(m: np.ndarray, tree: list[tuple[int, int]], root_side: int) -> int:
    """Homomorphisms of a bipartite tree (edges (u, v), u on side 0) into the
    bipartite graph of m, by enumeration."""
    nodes = sorted({x for e in tree for x in e})
    side = {nodes[0]: root_side}
    changed = True
    while changed:  # 2-colour the tree from its first node
        changed = False
        for u, v in tree:
            for x, y in ((u, v), (v, u)):
                if x in side and y not in side:
                    side[y] = 1 - side[x]
                    changed = True
    count = 0
    for img in itertools.product(range(4), repeat=len(nodes)):
        f = dict(zip(nodes, img, strict=True))
        ok = True
        for u, v in tree:
            a, b = (u, v) if side[u] == 0 else (v, u)
            if not m[f[a], f[b]]:
                ok = False
                break
        count += ok
    return count


TREES = {
    "edge": [(0, 1)],
    "star3": [(0, 1), (0, 2), (0, 3)],
    "claw_pair": [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)],
    "caterpillar": [(0, 1), (1, 2), (2, 3), (0, 4), (1, 5), (2, 6)],
    "path4": [(0, 1), (1, 2), (2, 3), (3, 4)],
}


@pytest.mark.parametrize("name", sorted(TREES))
@pytest.mark.parametrize("root_side", [0, 1])
def test_tree_homomorphism_counts_agree(name: str, root_side: int) -> None:
    tree = TREES[name]
    a = _tree_homs(A, tree, root_side)
    b = _tree_homs(B, tree, root_side)
    assert a == b == 4 * 2 ** len(tree)


def test_pairwise_statistics_agree_but_common_neighbour_l2_differs() -> None:
    for m in (A, B):
        assert (m.sum(0) == 2).all() and (m.sum(1) == 2).all()
    # pair join R(a,b) S(b,c) with S = M^T: deg_R(b) = deg_S(b) = 2
    for p, q in itertools.product(range(4), repeat=2):
        assert sum(A.sum(0) ** p * A.sum(0) ** q) == sum(B.sum(0) ** p * B.sum(0) ** q)
    # one-side interface degrees: sum over b in N_R(a) of deg_S(b)
    assert (A @ A.sum(0) == B @ B.sum(0)).all()
    ca, cb = A @ A.T, B @ B.T
    assert int((ca**2).sum()) == 32
    assert int((cb**2).sum()) == 24
    assert ca.max() == cb.max() == 2
    # 4-cycles R S R' S' = tr((M M^T)^2)
    assert int(np.trace(ca @ ca)) == 32
    assert int(np.trace(cb @ cb)) == 24
