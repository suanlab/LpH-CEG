"""Pre-register the LONG-cycle suite (|V| in {6, 8, 10}) before any bound is measured.

Reviewers R2 and R4 both flagged the same gap: the paper motivates cyclic
queries with 10-15 variables, where the 2^|V| entropic program becomes
expensive, but the pre-registered suite (`register_query_suite.py`) stops
at |V| = 5, where the cover LP and the entropic LP cost about the same.
This registry closes that gap on the same public dataset.

The enumeration rule is the v1 rule extended to longer cycles, with one
addition forced by scale:

  * Hetionet type-consistent simple metagraph cycles of length k in
    LONG_LENGTHS, canonicalised under rotation and reflection.  There are
    777 / 4,380 / 12,960 such cycles at k = 6 / 8 / 10 -- far too many to
    evaluate against the entropic program, so
  * for each k the first N_PER_LENGTH cycles in the order of the SHA-256
    of their canonical relation sequence are admitted.  Hash order is
    deterministic, depends only on the query's name, and is blind to any
    data or bound, so the selection cannot be steered by results.

This script computes no bound, estimate or timing.  Its output is hashed
and pinned by `benchmarks/query_suite_v2_long.sha256`; evaluation refuses
to run on a mismatch, exactly as for v1.

Output: benchmarks/query_suite_v2_long.json + benchmarks/query_suite_v2_long.sha256
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.register_query_suite import HETIONET_RELATIONS, canonical_cycle

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "benchmarks"
LONG_LENGTHS = (6, 8, 10)
N_PER_LENGTH = 40


def enumerate_long_cycles(k: int) -> list[dict[str, object]]:
    """All type-consistent simple metagraph cycles of length k (DFS)."""
    names = sorted(HETIONET_RELATIONS)
    admitted: dict[tuple[str, ...], dict[str, object]] = {}

    def dfs(seq: list[str], types: list[tuple[str, str]]) -> None:
        if len(seq) == k:
            if types[-1][1] == types[0][0]:
                key = canonical_cycle(tuple(seq))
                if key not in admitted:
                    admitted[key] = {
                        "dataset": "hetionet",
                        "shape": f"cycle{k}",
                        "n_vars": k,
                        "relations": list(seq),
                        "var_types": [t[0] for t in types],
                    }
            return
        used = set(seq)
        for r in names:
            if r in used:
                continue
            s, t = HETIONET_RELATIONS[r]
            for st in ((s, t), (t, s)):
                if seq and types[-1][1] != st[0]:
                    continue
                dfs([*seq, r], [*types, st])

    dfs([], [])
    return list(admitted.values())


def select_by_hash(qs: list[dict[str, object]], n: int) -> list[dict[str, object]]:
    """First n queries in SHA-256 order of their canonical relation sequence."""

    def h(q: dict[str, object]) -> str:
        key = "|".join(canonical_cycle(tuple(q["relations"])))  # type: ignore[arg-type]
        return hashlib.sha256(key.encode()).hexdigest()

    return sorted(qs, key=h)[:n]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    for k in LONG_LENGTHS:
        allk = enumerate_long_cycles(k)
        counts[f"cycle{k}"] = len(allk)
        out.extend(select_by_hash(allk, N_PER_LENGTH))
    for i, q in enumerate(out):
        q["query_id"] = f"HETL-{i:03d}"

    registry = {
        "version": "v2-long",
        "rule": (
            "Hetionet type-consistent simple metagraph cycles of length k in "
            f"{list(LONG_LENGTHS)}; per k, the first {N_PER_LENGTH} in SHA-256 "
            "order of the canonical relation sequence. No bound computed here."
        ),
        "enumerated_per_length": counts,
        "n_queries": len(out),
        "queries": out,
    }
    path = OUT_DIR / "query_suite_v2_long.json"
    path.write_text(json.dumps(registry, indent=1, sort_keys=True) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT_DIR / "query_suite_v2_long.sha256").write_text(
        f"{digest}  query_suite_v2_long.json\n"
    )
    print("Long-cycle suite registered (no bounds computed)")
    for k in LONG_LENGTHS:
        print(
            f"  cycle{k}: {counts[f'cycle{k}']} enumerated -> {N_PER_LENGTH} admitted"
        )
    print(f"  total {len(out)} queries; sha256 {digest}")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
