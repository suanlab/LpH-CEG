"""Pre-register the IMDb cycle suite before any bound is measured.

The second real-data suite answers the reviewer's breadth objection (every
real block result was Hetionet).  IMDb is a star schema: every relation of
`prepare_imdb_cycles.py` joins movies to one entity type of its own (people
as actors, as directors, ...; companies as distributors, ...; keywords), so
a cycle alternates movie variables with entity variables and each entity
variable is reached through its one relation on both sides -- two films
sharing an actor, a director, a distributor.  A cycle of length k is
therefore a cyclic sequence of k/2 relations, relations may repeat (as JOB
joins ``cast_info`` and ``movie_companies`` more than once), and the rule
is otherwise the Hetionet long-suite rule verbatim:

  * every such cycle of length k in LENGTHS, canonicalised under rotation
    and reflection;
  * per k, the first N_PER_LENGTH in SHA-256 order of the canonical
    relation sequence.  Hash order depends only on the query's name, so
    the selection is blind to data and bounds.

Two earlier IMDb designs were dropped before this registry and are recorded
in `prepare_imdb_cycles.py` and results/imdb/superseded_sparse_2005/: one
person type with distinct relations forces every person on a cycle to hold
two different roles, which almost never happens, and most joins were empty.

This script computes no bound, estimate or timing.  Output:
benchmarks/query_suite_v3_imdb.json + .sha256 (evaluation refuses to run on
a mismatch).
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

from scripts.prepare_imdb_cycles import IMDB_RELATIONS
from scripts.register_long_suite import select_by_hash
from scripts.register_query_suite import canonical_cycle

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "benchmarks"
LENGTHS = (6, 8, 10)
N_PER_LENGTH = 40


def enumerate_cycles(k: int) -> list[dict[str, object]]:
    """All length-k cycles: cyclic sequences of k/2 relations, each used twice."""
    names = sorted(IMDB_RELATIONS)
    admitted: dict[tuple[str, ...], dict[str, object]] = {}
    for combo in itertools.product(names, repeat=k // 2):
        seq: list[str] = []
        types: list[str] = []
        for r in combo:
            entity = next(t for t in IMDB_RELATIONS[r] if t != "M")
            seq += [r, r]
            types += ["M", entity]
        key = canonical_cycle(tuple(seq))
        if key not in admitted:
            admitted[key] = {
                "dataset": "imdb",
                "shape": f"cycle{k}",
                "n_vars": k,
                "relations": seq,
                "var_types": types,
            }
    return list(admitted.values())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    for k in LENGTHS:
        allk = enumerate_cycles(k)
        counts[f"cycle{k}"] = len(allk)
        out.extend(select_by_hash(allk, N_PER_LENGTH))
    for i, q in enumerate(out):
        q["query_id"] = f"IMDB-{i:03d}"
    manifest = json.loads((ROOT / "data" / "imdb_cycles" / "MANIFEST.json").read_text())
    registry = {
        "version": "v3-imdb",
        "rule": (
            "IMDb (JOB snapshot) typed relations of prepare_imdb_cycles.py; "
            f"cycles of length k in {list(LENGTHS)} alternating movies with "
            "entity variables, each reached through its one relation on both "
            f"sides; per k, the first {N_PER_LENGTH} in SHA-256 order of the "
            "canonical relation sequence. No bound computed here."
        ),
        "data_sha256": {r: v["sha256"] for r, v in manifest["relations"].items()},
        "enumerated_per_length": counts,
        "n_queries": len(out),
        "queries": out,
    }
    path = OUT_DIR / "query_suite_v3_imdb.json"
    path.write_text(json.dumps(registry, indent=1, sort_keys=True) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT_DIR / "query_suite_v3_imdb.sha256").write_text(
        f"{digest}  query_suite_v3_imdb.json\n"
    )
    print("IMDb cycle suite registered (no bounds computed)")
    for k in LENGTHS:
        print(
            f"  cycle{k}: {counts[f'cycle{k}']} enumerated -> "
            f"{min(N_PER_LENGTH, counts[f'cycle{k}'])} admitted"
        )
    print(f"  total {len(out)} queries; sha256 {digest}")


if __name__ == "__main__":
    main()
