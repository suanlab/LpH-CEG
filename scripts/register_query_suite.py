"""Pre-register the evaluation query suite BEFORE any bound is measured.

Plan gate (8/21): a canonical suite of 30+ cyclic queries over >=2 public
workloads, fixed in advance so that query selection cannot be influenced
by the results.

The anti-cherry-picking guarantee is structural, not a promise:

  1. Queries are produced by the deterministic enumeration rules below,
     never hand-picked.  Every query the rule admits is included.
  2. This script computes **no bound, estimate, or timing** -- it only
     reads schemas and relation sizes.  It cannot see an outcome.
  3. The output is hashed; `benchmarks/query_suite_v1.sha256` pins it.
     Evaluation scripts must verify that hash before reporting, so a
     post-hoc edit to the suite is detectable.

Registration rules
------------------
**Hetionet** (biomedical knowledge graph, 4 node types A/C/D/G, 13 typed
edge relations).  Variables are typed; a query is a simple cycle in the
metagraph.  Rule: enumerate every type-consistent simple cycle of length
k in {3,4,5} over distinct relation instances, canonicalise under
rotation and reflection, drop duplicates, sort lexicographically by
(k, relation names), and admit all of them.  Same-type relations
(DrD, crc, gig) participate like any other.

**Epinions** (social graph, one homogeneous edge relation).  Rule: the
standard cyclic/dense shape set -- cycle-3..6, K4, diamond, chordal-5 --
each instantiated on the single edge relation.

Both are public datasets.  Neither rule references any measured quantity.

Output: benchmarks/query_suite_v1.json + benchmarks/query_suite_v1.sha256
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "benchmarks"

CYCLE_LENGTHS = (3, 4, 5)

# Hetionet edge relation -> (source node type, target node type).
# Decoded from the standard Hetionet v1.0 metaedge abbreviations.
HETIONET_RELATIONS: dict[str, tuple[str, str]] = {
    "AdG": ("A", "G"),  # Anatomy downregulates Gene
    "AuG": ("A", "G"),  # Anatomy upregulates Gene
    "CdG": ("C", "G"),  # Compound downregulates Gene
    "CuG": ("C", "G"),  # Compound upregulates Gene
    "DdG": ("D", "G"),  # Disease downregulates Gene
    "DlA": ("D", "A"),  # Disease localizes Anatomy
    "DrD": ("D", "D"),  # Disease resembles Disease
    "DuG": ("D", "G"),  # Disease upregulates Gene
    "cbg": ("C", "G"),  # Compound binds Gene
    "crc": ("C", "C"),  # Compound resembles Compound
    "ctd": ("C", "D"),  # Compound treats Disease
    "dag": ("D", "G"),  # Disease associates Gene
    "gig": ("G", "G"),  # Gene interacts Gene
}

EPINIONS_SHAPES: dict[str, list[tuple[int, int]]] = {
    "cycle3": [(0, 1), (1, 2), (2, 0)],
    "cycle4": [(0, 1), (1, 2), (2, 3), (3, 0)],
    "cycle5": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],
    "cycle6": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],
    "k4": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
    "diamond": [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
    "chordal5": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)],
}


def canonical_cycle(rels: tuple[str, ...]) -> tuple[str, ...]:
    """Canonical form of a cycle under rotation and reflection.

    Two cycles that differ only by where you start reading, or by
    direction, are the same query and must not both be admitted.
    """
    n = len(rels)
    forms = []
    for r in range(n):
        rot = tuple(rels[(r + i) % n] for i in range(n))
        forms.append(rot)
        forms.append(tuple(reversed(rot)))
    return min(forms)


def enumerate_hetionet_cycles() -> list[dict[str, object]]:
    """All type-consistent simple metagraph cycles of length 3, 4, 5."""
    names = sorted(HETIONET_RELATIONS)
    admitted: dict[tuple[str, ...], dict[str, object]] = {}

    for k in CYCLE_LENGTHS:
        # A cycle is a sequence of k distinct relations, each optionally
        # traversed in reverse, whose endpoint types chain and close up.
        for combo in itertools.permutations(names, k):
            for flips in itertools.product((False, True), repeat=k):
                types: list[tuple[str, str]] = []
                for rel, flip in zip(combo, flips, strict=True):
                    s, t = HETIONET_RELATIONS[rel]
                    types.append((t, s) if flip else (s, t))
                # chain: target of i must equal source of i+1 (mod k)
                if any(types[i][1] != types[(i + 1) % k][0] for i in range(k)):
                    continue
                key = canonical_cycle(combo)
                if key in admitted:
                    continue
                # Record the orientation actually found, so that
                # var_types[i] (the type of the variable joining relation
                # i-1 to relation i) matches `relations` position by
                # position.  `key` is used only for deduplication.
                admitted[key] = {
                    "dataset": "hetionet",
                    "shape": f"cycle{k}",
                    "n_vars": k,
                    "relations": list(combo),
                    "var_types": [types[i][0] for i in range(k)],
                }

    out = sorted(
        admitted.values(),
        key=lambda q: (q["n_vars"], tuple(q["relations"])),  # type: ignore[arg-type]
    )
    for i, q in enumerate(out):
        q["query_id"] = f"HET-{i:03d}"
    return out


def enumerate_epinions_shapes() -> list[dict[str, object]]:
    """The fixed cyclic/dense shape set on the single Epinions relation."""
    out: list[dict[str, object]] = []
    for i, (shape, edges) in enumerate(sorted(EPINIONS_SHAPES.items())):
        n_vars = 1 + max(max(e) for e in edges)
        out.append(
            {
                "query_id": f"EPI-{i:03d}",
                "dataset": "epinions",
                "shape": shape,
                "n_vars": n_vars,
                "relations": [f"edges[{u},{v}]" for u, v in edges],
                "var_types": ["node"] * n_vars,
            }
        )
    return out


def main() -> None:
    print("Pre-registering the evaluation query suite (no bounds computed)")
    print("=" * 78)

    het = enumerate_hetionet_cycles()
    epi = enumerate_epinions_shapes()
    queries = het + epi

    by_shape: dict[str, int] = {}
    for q in queries:
        key = f"{q['dataset']}/{q['shape']}"
        by_shape[key] = by_shape.get(key, 0) + 1
    for key in sorted(by_shape):
        print(f"  {key:<24} {by_shape[key]:>4}")
    print(f"  {'-' * 30}")
    print(f"  {'TOTAL':<24} {len(queries):>4} queries over 2 public workloads")

    registry = {
        "suite": "query_suite_v1",
        "registered_before_measurement": True,
        "workloads": ["hetionet", "epinions"],
        "rules": {
            "hetionet": (
                "all type-consistent simple metagraph cycles of length 3,4,5 "
                "over distinct relations, canonicalised under rotation and "
                "reflection, sorted by (length, relation names); all admitted"
            ),
            "epinions": (
                "fixed shape set cycle3..6, k4, diamond, chordal5 on the "
                "single homogeneous edge relation"
            ),
        },
        "n_queries": len(queries),
        "queries": queries,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "query_suite_v1.json"
    payload = json.dumps(registry, indent=2, sort_keys=True) + "\n"
    path.write_text(payload)

    digest = hashlib.sha256(payload.encode()).hexdigest()
    (OUT_DIR / "query_suite_v1.sha256").write_text(f"{digest}  query_suite_v1.json\n")

    print(f"\nWrote {path}")
    print(f"  sha256 = {digest}")
    print("  Evaluation must verify this hash before reporting results.")

    if len(queries) < 30:
        raise SystemExit(
            f"gate not met: {len(queries)} queries registered, plan requires >=30"
        )
    print(f"  Gate: {len(queries)} >= 30 required by the 8/21 milestone -- MET")


if __name__ == "__main__":
    main()
