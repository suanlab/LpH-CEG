"""Full LpH-CEG experiments on Hetionet, IMDb, and Epinions.

Runs cycle-k (k=3,4,5) queries on real datasets and compares
H-CEG vs LpH-CEG bounds.
"""

from __future__ import annotations

import time
from pathlib import Path

from src.estimators.lph_ceg import estimate_lph_ceg
from src.estimators.pessimistic import estimate_pessimistic
from src.utils.query import cycle_query, triangle_query
from src.utils.types import QueryHypergraph, Relation


def load_tsv_edges(path: Path, max_rows: int | None = None) -> list[tuple[str, str]]:
    """Load edge list from TSV file (skip header)."""
    edges: list[tuple[str, str]] = []
    with open(path) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                edges.append((parts[0], parts[1]))
                if max_rows and len(edges) >= max_rows:
                    break
    return edges


def run_experiment(
    name: str,
    query: QueryHypergraph,
    data: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]],
) -> dict[str, object]:
    """Run H-CEG vs LpH-CEG comparison."""
    sizes = {r: len(data[r][1]) for r in data}

    t0 = time.perf_counter()
    std = estimate_pessimistic(query, data, max_block_size=1)
    t_std = time.perf_counter() - t0

    t0 = time.perf_counter()
    lph = estimate_lph_ceg(
        query,
        data,
        max_block_size=1,
        p_values=(1.0, 2.0, float("inf")),
    )
    t_lph = time.perf_counter() - t0

    ratio = std.estimate / max(lph.estimate, 1e-10)
    variant = lph.metadata.get("method_variant", "?")
    p_used = lph.metadata.get("lp_p_values_used", [])
    w_step = lph.metadata.get("lp_weights_per_step", [])

    print(f"\n  [{name}]")
    print(f"    Relations: {sizes}")
    print(f"    H-CEG  (bs=1):  {std.estimate:>15.0f}  ({t_std * 1000:.0f}ms)")
    print(f"    LpH-CEG(bs=1):  {lph.estimate:>15.0f}  ({t_lph * 1000:.0f}ms)")
    print(f"    Improvement:     {ratio:.2f}x  [{variant}]")
    print(f"    LP p-values:     {p_used}")
    print(f"    LP weights:      {[f'{w:.3f}' for w in w_step]}")

    return {
        "name": name,
        "hceg": std.estimate,
        "lph_ceg": lph.estimate,
        "ratio": ratio,
        "variant": variant,
    }


def main() -> None:
    print("=" * 70)
    print("LpH-CEG Full Experiments: Hetionet + IMDb + Epinions")
    print("=" * 70)

    results: list[dict[str, object]] = []

    # =====================================================================
    # Hetionet: heterogeneous biomedical graph
    # =====================================================================
    print("\n" + "=" * 70)
    print("HETIONET")
    print("=" * 70)

    hetio = Path("data/hetionet")
    ctd = load_tsv_edges(hetio / "ctd_edges.tsv")  # Compound-treats-Disease
    dag = load_tsv_edges(hetio / "dag_edges.tsv")  # Disease-associates-Gene
    cbg = load_tsv_edges(hetio / "cbg_edges.tsv")  # Compound-binds-Gene
    gig = load_tsv_edges(hetio / "gig_edges.tsv", max_rows=5000)  # Gene-interacts-Gene

    # Hetionet Cycle-4: CtD ⋈ DaG ⋈ GiG ⋈ CbG
    # Compound -[treats]-> Disease -[associates]-> Gene -[interacts]-> Gene2 -[binds]<- Compound
    q_h4 = QueryHypergraph(
        relations=(
            Relation("CtD", frozenset({"compound", "disease"}), len(ctd)),
            Relation("DaG", frozenset({"disease", "gene1"}), len(dag)),
            Relation("GiG", frozenset({"gene1", "gene2"}), len(gig)),
            Relation("CbG", frozenset({"compound", "gene2"}), len(cbg)),
        ),
        name="hetionet_cycle4",
    )
    data_h4 = {
        "CtD": (("compound", "disease"), ctd),
        "DaG": (("disease", "gene1"), dag),
        "GiG": (("gene1", "gene2"), gig),
        "CbG": (("compound", "gene2"), cbg),
    }
    results.append(run_experiment("Hetionet Cycle-4 (CtD-DaG-GiG-CbG)", q_h4, data_h4))

    # Hetionet Triangle: CtD ⋈ DaG ⋈ CbG (compound-disease-gene)
    q_h3 = QueryHypergraph(
        relations=(
            Relation("CtD", frozenset({"compound", "disease"}), len(ctd)),
            Relation("DaG", frozenset({"disease", "gene"}), len(dag)),
            Relation("CbG", frozenset({"compound", "gene"}), len(cbg)),
        ),
        name="hetionet_triangle",
    )
    data_h3 = {
        "CtD": (("compound", "disease"), ctd),
        "DaG": (("disease", "gene"), dag),
        "CbG": (("compound", "gene"), cbg),
    }
    results.append(run_experiment("Hetionet Triangle (CtD-DaG-CbG)", q_h3, data_h3))

    # =====================================================================
    # IMDb: movie database
    # =====================================================================
    print("\n" + "=" * 70)
    print("IMDb")
    print("=" * 70)

    imdb = Path("data/imdb")
    ma = load_tsv_edges(imdb / "movie_actor_10k.tsv", max_rows=5000)
    load_tsv_edges(imdb / "movie_director_10k.tsv")
    load_tsv_edges(imdb / "movie_genre_10k.tsv", max_rows=5000)

    # IMDb Cycle-4: MA ⋈ MD ⋈ MG ⋈ MA2
    # movie -[actor]-> actor, movie -[director]-> director,
    # movie -[genre]-> genre, movie2 -[actor]-> actor
    # But this requires shared attributes. Let's do star + cycle:
    # movie_actor(movie,actor) ⋈ movie_director(movie,director) ⋈
    # movie_genre(movie,genre) — this is a star, not cycle.

    # For cycle: use co-actor pattern
    # MA(movie, actor1), MA2(movie, actor2) — 2 actors in same movie
    # Then actor1 appears in another movie with actor2
    # This is complex; let's stick with replicated-edge cycle pattern
    q_imdb_c4 = cycle_query(4)
    data_imdb_c4: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]] = {}
    for rel in q_imdb_c4.relations:
        attrs = tuple(sorted(rel.attributes))
        data_imdb_c4[rel.name] = (attrs, ma)  # replicated edges

    results.append(
        run_experiment(
            f"IMDb Cycle-4 (replicated MA, {len(ma)} edges)", q_imdb_c4, data_imdb_c4
        )
    )

    # IMDb Triangle (replicated)
    q_imdb_tri = triangle_query()
    data_imdb_tri = {
        "R": (("a", "b"), ma),
        "S": (("b", "c"), ma),
        "T": (("c", "a"), ma),
    }
    results.append(
        run_experiment(
            f"IMDb Triangle (replicated MA, {len(ma)} edges)", q_imdb_tri, data_imdb_tri
        )
    )

    # =====================================================================
    # Epinions: social network (from previous experiments)
    # =====================================================================
    print("\n" + "=" * 70)
    print("EPINIONS")
    print("=" * 70)

    epinions_path = Path("data/epinions/edges.txt")
    ep_edges: list[tuple[str, str]] = []
    with open(epinions_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                ep_edges.append((parts[0], parts[1]))
                if len(ep_edges) >= 5000:
                    break

    for k in [3, 4, 5]:
        if k == 3:
            q = triangle_query()
            data_ep: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]] = {
                "R": (("a", "b"), ep_edges),
                "S": (("b", "c"), ep_edges),
                "T": (("c", "a"), ep_edges),
            }
        else:
            q = cycle_query(k)
            data_ep = {}
            for rel in q.relations:
                attrs = tuple(sorted(rel.attributes))
                data_ep[rel.name] = (attrs, ep_edges)

        results.append(
            run_experiment(f"Epinions Cycle-{k} ({len(ep_edges)} edges)", q, data_ep)
        )

    # =====================================================================
    # Summary Table
    # =====================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Name':<50} {'H-CEG':>15} {'LpH-CEG':>15} {'Ratio':>8}")
    print("-" * 90)
    for r in results:
        print(
            f"{r['name']:<50} {r['hceg']:>15.0f} {r['lph_ceg']:>15.0f} {r['ratio']:>7.2f}x"
        )


if __name__ == "__main__":
    main()
