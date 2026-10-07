"""LpH-CEG vs H-CEG on real Epinions data (cycle queries).

DEPRECATED / EXPLORATORY — not a paper-result source. Canonical real-data
results come from scripts/lph_ceg_full_experiments.py and
scripts/job_full_experiment.py.

Loads a sample of the Epinions graph and runs cycle-4 and cycle-5
queries, comparing standard H-CEG and LpH-CEG bounds.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

from src.estimators.lph_ceg import estimate_lph_ceg
from src.estimators.pessimistic import estimate_pessimistic
from src.utils.query import cycle_query, triangle_query


def load_epinions_sample(
    path: Path, max_edges: int = 2000, *, seed: int = 42
) -> list[tuple[str, str]]:
    """Load a random sample of Epinions edges."""
    all_edges: list[tuple[str, str]] = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                all_edges.append((parts[0], parts[1]))

    rng = random.Random(seed)
    if len(all_edges) > max_edges:
        sample = rng.sample(all_edges, max_edges)
    else:
        sample = all_edges
    return sorted(set(sample))


def make_replicated_relation(
    edges: list[tuple[str, str]],
    name: str,
    attr_a: str,
    attr_b: str,
) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
    """Create a relation from edge list with given attribute names."""
    return (attr_a, attr_b), [(u, v) for u, v in edges]


def run_cycle_experiment(
    edges: list[tuple[str, str]],
    k: int,
    label: str,
) -> None:
    """Run cycle-k experiment."""
    q = cycle_query(k)
    data: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]] = {}
    for rel in q.relations:
        attrs = tuple(sorted(rel.attributes))
        data[rel.name] = (attrs, edges)

    print(f"\n--- {label}: Cycle-{k} ---")
    print(f"  |E| = {len(edges)}")

    # H-CEG (bs=1)
    t0 = time.perf_counter()
    std = estimate_pessimistic(q, data, max_block_size=1)
    t_std = time.perf_counter() - t0

    # LpH-CEG (bs=1)
    t0 = time.perf_counter()
    lph = estimate_lph_ceg(
        q,
        data,
        max_block_size=1,
        p_values=(1.0, 2.0, float("inf")),
    )
    t_lph = time.perf_counter() - t0

    ratio = std.estimate / max(lph.estimate, 1e-10)
    variant = lph.metadata.get("method_variant", "?")
    p_used = lph.metadata.get("lp_p_values_used", [])
    w_per_step = lph.metadata.get("lp_weights_per_step", [])

    print(f"  H-CEG  (bs=1):  {std.estimate:>15.0f}  ({t_std * 1000:.0f}ms)")
    print(f"  LpH-CEG(bs=1):  {lph.estimate:>15.0f}  ({t_lph * 1000:.0f}ms)")
    print(f"  LP improvement:  {ratio:.2f}x  [{variant}]")
    print(f"  LP p-values:     {p_used}")
    print(f"  LP weights:      {[f'{w:.3f}' for w in w_per_step]}")


def main() -> None:
    print("=" * 65)
    print("LpH-CEG on Real Epinions Data (Cycle Queries)")
    print("=" * 65)

    epinions_path = Path("data/epinions/edges.txt")
    if not epinions_path.exists():
        print(f"ERROR: {epinions_path} not found")
        return

    for sample_size in [1000, 2000, 5000]:
        edges = load_epinions_sample(epinions_path, max_edges=sample_size)
        label = f"Epinions ({len(edges)} edges)"

        for k in [4, 5]:
            run_cycle_experiment(edges, k, label)

    # Triangle for reference
    edges_tri = load_epinions_sample(epinions_path, max_edges=2000)
    q_tri = triangle_query()
    tri_data = {
        "R": (("a", "b"), edges_tri),
        "S": (("b", "c"), edges_tri),
        "T": (("c", "a"), edges_tri),
    }
    print(f"\n--- Epinions Triangle ({len(edges_tri)} edges) ---")
    std = estimate_pessimistic(q_tri, tri_data, max_block_size=1)
    lph = estimate_lph_ceg(
        q_tri, tri_data, max_block_size=1, p_values=(1.0, 2.0, float("inf"))
    )
    ratio = std.estimate / max(lph.estimate, 1e-10)
    print(f"  H-CEG:   {std.estimate:.0f}")
    print(f"  LpH-CEG: {lph.estimate:.0f}")
    print(f"  Improvement: {ratio:.2f}x")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
