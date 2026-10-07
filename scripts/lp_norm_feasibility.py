"""Quick feasibility check: Do Lp-norms give tighter bounds than max-degree?

This script runs on synthetic power-law data and compares:
- μ_∞(F|I) = max degree (current H-CEG)
- μ_2(F|I) = ℓ2-norm of degree sequence
- μ_1(F|I) = sum of degrees = |Q_F| (cardinality)

For a triangle query R(a,b) ⋈ S(b,c) ⋈ T(c,a), checks whether
Lp-norm block extensions produce tighter upper bounds.
"""

from __future__ import annotations

import random
from collections import defaultdict


def generate_power_law_edges(
    n_vertices: int, n_edges: int, *, seed: int = 42
) -> list[tuple[str, str]]:
    """Generate edges with power-law degree distribution."""
    rng = random.Random(seed)
    # Zipf-like weight: vertex i has weight proportional to 1/(i+1)
    weights = [1.0 / (i + 1) for i in range(n_vertices)]
    total = sum(weights)
    probs = [w / total for w in weights]

    edges: set[tuple[str, str]] = set()
    while len(edges) < n_edges:
        u = _weighted_choice(probs, rng)
        v = _weighted_choice(probs, rng)
        if u != v:
            edges.add((str(u), str(v)))
    return sorted(edges)


def _weighted_choice(probs: list[float], rng: random.Random) -> int:
    r = rng.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r <= cumulative:
            return i
    return len(probs) - 1


def compute_degree_sequence(
    tuples: list[tuple[str, ...]],
    key_indices: list[int],
    val_indices: list[int],
) -> list[int]:
    """Compute degree sequence: for each key value, count distinct val values."""
    groups: dict[tuple[str, ...], set[tuple[str, ...]]] = defaultdict(set)
    for row in tuples:
        key = tuple(row[i] for i in key_indices)
        val = tuple(row[i] for i in val_indices)
        groups[key].add(val)
    return sorted((len(v) for v in groups.values()), reverse=True)


def lp_norm(seq: list[int], p: float) -> float:
    """Compute ℓp-norm of a sequence."""
    if p == float("inf"):
        return float(max(seq)) if seq else 0.0
    if not seq:
        return 0.0
    return sum(d**p for d in seq) ** (1.0 / p)


def main() -> None:
    print("=" * 70)
    print("LpH-CEG Feasibility Check: Lp-norm vs max-degree bounds")
    print("=" * 70)

    n_vertices = 500
    n_edges = 2000
    seed = 42

    # Generate 3 relations for triangle query R(a,b) ⋈ S(b,c) ⋈ T(c,a)
    edges_r = generate_power_law_edges(n_vertices, n_edges, seed=seed)
    edges_s = generate_power_law_edges(n_vertices, n_edges, seed=seed + 1)
    edges_t = generate_power_law_edges(n_vertices, n_edges, seed=seed + 2)

    print(f"\nData: {n_vertices} vertices, {n_edges} edges per relation (power-law)")
    print(f"|R| = {len(edges_r)}, |S| = {len(edges_s)}, |T| = {len(edges_t)}")

    # --- Binary extension analysis ---
    print("\n" + "-" * 70)
    print("1. Binary Extension: R(a,b) extended by S(b,c), interface I={b}")
    print("-" * 70)

    # S viewed as tuples (b, c)
    s_tuples = edges_s  # (b, c)
    deg_seq_s_b = compute_degree_sequence(s_tuples, key_indices=[0], val_indices=[1])

    for p in [1.0, 2.0, 3.0, 5.0, float("inf")]:
        norm = lp_norm(deg_seq_s_b, p)
        p_str = "∞" if p == float("inf") else str(int(p))
        print(f"  ||deg_S(c|b)||_{p_str} = {norm:.2f}")

    print("\n  Degree sequence stats:")
    print(f"    count = {len(deg_seq_s_b)}")
    print(f"    max = {max(deg_seq_s_b)}")
    print(f"    mean = {sum(deg_seq_s_b) / len(deg_seq_s_b):.2f}")
    print(f"    median = {sorted(deg_seq_s_b)[len(deg_seq_s_b) // 2]}")
    print(f"    top-5 = {deg_seq_s_b[:5]}")

    # --- Multi-way block analysis (triangle as single block) ---
    print("\n" + "-" * 70)
    print("2. Block Extension: F={S,T} as single block, interface I={a} from R")
    print("-" * 70)

    # Join S(b,c) ⋈ T(c,a) → tuples (b, c, a)
    s_index: dict[str, list[str]] = defaultdict(list)
    for b, c in edges_s:
        s_index[c].append(b)

    st_join: list[tuple[str, str, str]] = []
    for c_val, a_val in edges_t:
        for b_val in s_index.get(c_val, []):
            st_join.append((b_val, c_val, a_val))

    print(f"  |S ⋈ T| = {len(st_join)}")

    # Degree sequence: for each a_val (interface), count distinct (b,c) extensions
    deg_seq_st_a = compute_degree_sequence(st_join, key_indices=[2], val_indices=[0, 1])

    if not deg_seq_st_a:
        print("  Empty join — skipping block analysis")
    else:
        for p in [1.0, 2.0, 3.0, 5.0, float("inf")]:
            norm = lp_norm(deg_seq_st_a, p)
            p_str = "∞" if p == float("inf") else str(int(p))
            print(f"  ||deg_{{S⋈T}}(bc|a)||_{p_str} = {norm:.2f}")

        print("\n  Block degree sequence stats:")
        print(f"    count = {len(deg_seq_st_a)}")
        print(f"    max = {max(deg_seq_st_a)}")
        print(f"    mean = {sum(deg_seq_st_a) / len(deg_seq_st_a):.2f}")
        print(f"    median = {sorted(deg_seq_st_a)[len(deg_seq_st_a) // 2]}")
        print(f"    top-5 = {deg_seq_st_a[:5]}")

    # --- Bound comparison ---
    print("\n" + "-" * 70)
    print("3. Bound Comparison: Triangle R(a,b) ⋈ S(b,c) ⋈ T(c,a)")
    print("-" * 70)

    # Compute actual triangle count
    r_index_a: dict[str, set[str]] = defaultdict(set)
    for a, b in edges_r:
        r_index_a[a].add(b)

    triangle_count = 0
    for b_val, _c_val, a_val in st_join:
        if b_val in r_index_a.get(a_val, set()):
            triangle_count += 1

    print(f"  True triangle count: {triangle_count}")

    # Binary path: R → S → T
    compute_degree_sequence(edges_r, [1], [0])  # deg_R(a|b)
    deg_s_b = compute_degree_sequence(edges_s, [0], [1])  # deg_S(c|b)
    deg_t_c = compute_degree_sequence(edges_t, [0], [1])  # deg_T(a|c)

    for p in [2.0, float("inf")]:
        p_str = "∞" if p == float("inf") else str(int(p))
        # Binary bound: |R| * ||deg_S(c|b)||_p * ||deg_T(a|c)||_p
        # (simplified — actual bound depends on LP weights)
        bound_binary = len(edges_r) * lp_norm(deg_s_b, p) * lp_norm(deg_t_c, p)
        # This isn't the correct LP bound, but shows the scale
        print(f"\n  Naive binary product (p={p_str}): {bound_binary:.0f}")

    # H-CEG max-degree bound
    mu_inf_s = lp_norm(deg_s_b, float("inf"))
    mu_inf_t = lp_norm(deg_t_c, float("inf"))
    hceg_bound = len(edges_r) * mu_inf_s * mu_inf_t
    print(f"\n  H-CEG bound (ℓ∞): |R| × μ_∞(S|b) × μ_∞(T|c) = {hceg_bound:.0f}")

    # Block bound: |R| * μ_p({S,T} | {a})
    if deg_seq_st_a:
        for p in [1.0, 2.0, 3.0, float("inf")]:
            p_str = "∞" if p == float("inf") else str(int(p))
            mu_p = lp_norm(deg_seq_st_a, p)
            # Block bound (for p=∞): |R| * μ_∞({S,T}|a)
            # For p<∞, need LP weights, but μ_p is informative
            print(f"  Block μ_{p_str}({{S,T}}|a) = {mu_p:.2f}")

        block_inf = lp_norm(deg_seq_st_a, float("inf"))
        block_2 = lp_norm(deg_seq_st_a, 2.0)

        # Compare H-CEG bound vs block bound
        print("\n  --- Comparison ---")
        print(f"  True count:              {triangle_count}")
        print(
            f"  Binary H-CEG (ℓ∞):      {hceg_bound:.0f}  (ratio: {hceg_bound / max(triangle_count, 1):.1f}x)"
        )

        # Simple block bound: block cardinality (p=1) if available
        lp_norm(deg_seq_st_a, 1.0)
        # For block with p=∞: each R tuple matches ≤ μ_∞ block tuples
        block_bound_inf = len(edges_r) * block_inf
        block_bound_2 = (
            len(edges_r) * block_2
        )  # not a valid bound as-is, but illustrative

        print(
            f"  Block H-CEG (ℓ∞):       {block_bound_inf:.0f}  (ratio: {block_bound_inf / max(triangle_count, 1):.1f}x)"
        )
        print(
            f"  Block LpH-CEG (ℓ2):     {block_bound_2:.0f}  (ratio: {block_bound_2 / max(triangle_count, 1):.1f}x)"
        )

        improvement = hceg_bound / max(block_bound_inf, 1)
        print(f"\n  Block vs binary (ℓ∞):   {improvement:.2f}x tighter")
        if block_bound_2 < block_bound_inf:
            lp_improvement = block_bound_inf / max(block_bound_2, 1)
            print(f"  ℓ2 vs ℓ∞ (block):       {lp_improvement:.2f}x tighter")

    print("\n" + "=" * 70)
    print("Conclusion: Check if block + Lp-norm gives tighter bounds")
    print("=" * 70)


if __name__ == "__main__":
    main()
