"""The independent G-CARE recounter equals brute-force homomorphism counts."""

from __future__ import annotations

import itertools
import random

import pandas as pd

from scripts.gcare_recount import count


def _brute(verts, edges, e, vl, nodes):  # type: ignore[no-untyped-def]
    es = set(zip(e.s, e.d, e.l, strict=True))
    total = 0
    for img in itertools.product(nodes, repeat=len(verts)):
        m = dict(zip(sorted(verts), img, strict=True))
        if any(
            (lab != -1 and m[v] not in vl[lab]) or (dv != -1 and m[v] != dv)
            for v, (lab, dv) in verts.items()
        ):
            continue
        total += all((m[a], m[b], lab) in es for a, b, lab in edges)
    return total


def test_recount_matches_brute_force_on_random_graphs() -> None:
    rng = random.Random(0)
    nodes = list(range(7))
    for _ in range(25):
        e = pd.DataFrame(
            [
                (rng.choice(nodes), rng.choice(nodes), rng.choice((1, 2)))
                for _ in range(18)
            ],
            columns=["s", "d", "l"],
        )
        vl = {7: {n for n in nodes if rng.random() < 0.6}}
        verts = {
            0: (rng.choice((-1, 7)), -1),
            1: (-1, -1),
            2: (-1, rng.choice((-1, 3))),
            3: (rng.choice((-1, 7)), -1),
        }
        edges = [(0, 1, 1), (1, 2, rng.choice((1, 2))), (2, 3, 2), (3, 0, 1), (1, 3, 2)]
        assert count(verts, edges, e, vl) == _brute(verts, edges, e, vl, nodes)
