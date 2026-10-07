"""LpH-CEG experiments on TPC-H-like cyclic queries.

Generates small-scale TPC-H data and constructs cyclic join patterns:
- Q1: lineitem-orders-customer-supplier cycle (via nation)
- Q2: lineitem-partsupp-part-supplier cycle
- Q3: orders-lineitem-partsupp-supplier-nation cycle (5-way)

TPC-H's snowflake schema naturally creates cycles when joining
through dimension tables.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict

from src.estimators.lph_ceg import estimate_lph_ceg
from src.estimators.pessimistic import estimate_pessimistic
from src.utils.types import QueryHypergraph, Relation


def generate_tpch_data(
    sf: float = 0.01, *, seed: int = 42
) -> dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]]:
    """Generate small TPC-H-like data with power-law skew.

    sf=0.01 gives roughly:
    - 25 nations, 5 regions
    - 150 customers, 150 suppliers
    - 1500 orders, 200 parts
    - 6000 lineitem, 800 partsupp
    """
    rng = random.Random(seed)

    n_nations = 25
    n_regions = 5
    n_customers = int(150 * sf * 100)
    n_suppliers = int(150 * sf * 100)
    n_orders = int(1500 * sf * 100)
    n_parts = int(200 * sf * 100)
    n_lineitem = int(6000 * sf * 100)
    n_partsupp = int(800 * sf * 100)

    # Generate dimension tables
    nations = [(str(i), str(i % n_regions)) for i in range(n_nations)]
    [(str(i),) for i in range(n_regions)]

    # Customers with skewed nation distribution
    def skewed_pick(n: int) -> str:
        weights = [1.0 / (i + 1) for i in range(n)]
        total = sum(weights)
        r = rng.random() * total
        c = 0.0
        for i, w in enumerate(weights):
            c += w
            if r <= c:
                return str(i)
        return str(n - 1)

    customers = [(str(i), skewed_pick(n_nations)) for i in range(n_customers)]

    suppliers = [(str(i), skewed_pick(n_nations)) for i in range(n_suppliers)]

    orders = [(str(i), str(rng.randint(0, n_customers - 1))) for i in range(n_orders)]

    [(str(i),) for i in range(n_parts)]

    lineitem: list[tuple[str, str, str]] = []
    for _i in range(n_lineitem):
        oid = str(rng.randint(0, n_orders - 1))
        pid = skewed_pick(n_parts)
        sid = skewed_pick(n_suppliers)
        lineitem.append((oid, pid, sid))

    partsupp: list[tuple[str, str]] = []
    for _i in range(n_partsupp):
        pid = skewed_pick(n_parts)
        sid = skewed_pick(n_suppliers)
        partsupp.append((pid, sid))

    return {
        "nation": (("n_id", "r_id"), nations),
        "customer": (("c_id", "c_nation"), customers),
        "supplier": (("s_id", "s_nation"), suppliers),
        "orders": (("o_id", "o_custkey"), orders),
        "lineitem_os": (
            ("l_orderkey", "l_suppkey"),
            [(row[0], row[2]) for row in lineitem],
        ),
        "lineitem_op": (
            ("l_orderkey", "l_partkey"),
            [(row[0], row[1]) for row in lineitem],
        ),
        "lineitem_ps": (
            ("l_partkey", "l_suppkey"),
            [(row[1], row[2]) for row in lineitem],
        ),
        "partsupp": (("ps_partkey", "ps_suppkey"), partsupp),
    }


def count_join(
    query: QueryHypergraph,
    data: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]],
) -> int:
    """Brute-force join."""
    rels = list(query.relations)
    if not rels:
        return 0

    first = rels[0]
    attrs_0, tuples_0 = data[first.name]
    current = [dict(zip(attrs_0, t, strict=False)) for t in tuples_0]

    for rel in rels[1:]:
        attrs_r, tuples_r = data[rel.name]
        shared = set(current[0].keys()) & set(attrs_r) if current else set()
        shared_sorted = sorted(shared)

        idx: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
        for t in tuples_r:
            d = dict(zip(attrs_r, t, strict=False))
            key = tuple(d.get(a, "") for a in shared_sorted)
            idx[key].append(d)

        joined: list[dict[str, str]] = []
        for ct in current:
            key = tuple(ct.get(a, "") for a in shared_sorted)
            for nt in idx.get(key, []):
                joined.append({**ct, **nt})
        current = joined

    return len(current)


def run_query(
    name: str,
    query: QueryHypergraph,
    data: dict[str, tuple[tuple[str, ...], list[tuple[str, ...]]]],
) -> dict[str, object]:
    """Compare H-CEG vs LpH-CEG."""
    true_count = count_join(query, data)

    t0 = time.perf_counter()
    std = estimate_pessimistic(query, data, max_block_size=1)
    t_std = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    lph = estimate_lph_ceg(
        query,
        data,
        max_block_size=1,
        p_values=(1.0, 2.0, float("inf")),
    )
    t_lph = (time.perf_counter() - t0) * 1000

    ratio = std.estimate / max(lph.estimate, 1e-10)

    def qerror(est: float, true: int) -> float:
        if true == 0:
            return float("inf") if est > 0 else 1.0
        return max(est / true, true / max(est, 1e-10))

    q_std = qerror(std.estimate, true_count)
    q_lph = qerror(lph.estimate, true_count)

    print(f"\n  [{name}]")
    sizes = {
        r: len(data[r][1]) for r in data if r in [rel.name for rel in query.relations]
    }
    print(f"    Relations: {sizes}")
    print(f"    True count:  {true_count:>12}")
    print(
        f"    H-CEG:       {std.estimate:>12.0f}  q-error: {q_std:>8.1f}x  ({t_std:.0f}ms)"
    )
    print(
        f"    LpH-CEG:     {lph.estimate:>12.0f}  q-error: {q_lph:>8.1f}x  ({t_lph:.0f}ms)"
    )
    print(f"    LP ratio:    {ratio:.2f}x")

    return {
        "name": name,
        "true": true_count,
        "hceg": std.estimate,
        "lph": lph.estimate,
        "q_hceg": q_std,
        "q_lph": q_lph,
        "ratio": ratio,
    }


def main() -> None:
    print("=" * 70)
    print("LpH-CEG on TPC-H-like Cyclic Queries")
    print("=" * 70)

    data = generate_tpch_data(sf=0.01)
    print("\nGenerated TPC-H data:")
    for name, (attrs, tuples) in data.items():
        print(f"  {name}: {attrs}, {len(tuples)} rows")

    results: list[dict[str, object]] = []

    # ---- Q1: Customer-Order-Lineitem-Supplier cycle via nation ----
    # customer(c_id, c_nation) ⋈ orders(o_id, o_custkey=c_id)
    # ⋈ lineitem(l_orderkey=o_id, l_suppkey=s_id) ⋈ supplier(s_id, s_nation=c_nation)
    # Cycle: c_nation appears in both customer and supplier
    q1 = QueryHypergraph(
        relations=(
            Relation(
                "customer", frozenset({"c_id", "c_nation"}), len(data["customer"][1])
            ),
            Relation("orders", frozenset({"o_id", "c_id"}), len(data["orders"][1])),
            Relation(
                "lineitem_os", frozenset({"o_id", "s_id"}), len(data["lineitem_os"][1])
            ),
            Relation(
                "supplier", frozenset({"s_id", "c_nation"}), len(data["supplier"][1])
            ),
        ),
        name="tpch_q1_cycle",
    )
    # Rename orders columns to match
    orders_renamed = [t for t in data["orders"][1]]
    data_q1 = {
        "customer": data["customer"],
        "orders": (("o_id", "c_id"), orders_renamed),
        "lineitem_os": (("o_id", "s_id"), data["lineitem_os"][1]),
        "supplier": (("s_id", "c_nation"), data["supplier"][1]),
    }
    results.append(
        run_query("TPC-H Q1: Cust-Order-LI-Supp cycle (nation)", q1, data_q1)
    )

    # ---- Q2: Lineitem-PartSupp cycle ----
    # lineitem(l_partkey, l_suppkey) ⋈ partsupp(ps_partkey=l_partkey, ps_suppkey=l_suppkey)
    # This is actually a 2-way equijoin, not a cycle.
    # Better: lineitem_op(orderkey, partkey) ⋈ lineitem_ps(partkey, suppkey)
    #       ⋈ partsupp(ps_partkey, ps_suppkey) — cycle on partkey+suppkey

    q2 = QueryHypergraph(
        relations=(
            Relation(
                "li_op", frozenset({"orderkey", "partkey"}), len(data["lineitem_op"][1])
            ),
            Relation(
                "li_ps", frozenset({"partkey", "suppkey"}), len(data["lineitem_ps"][1])
            ),
            Relation(
                "partsupp", frozenset({"partkey", "suppkey"}), len(data["partsupp"][1])
            ),
        ),
        name="tpch_q2_triangle",
    )
    data_q2 = {
        "li_op": (("orderkey", "partkey"), data["lineitem_op"][1]),
        "li_ps": (("partkey", "suppkey"), data["lineitem_ps"][1]),
        "partsupp": (("partkey", "suppkey"), data["partsupp"][1]),
    }
    results.append(
        run_query("TPC-H Q2: LI-LI-PartSupp triangle (part+supp)", q2, data_q2)
    )

    # ---- Q3: 5-way cycle ----
    # customer(c_id, c_nation) ⋈ orders(o_id, c_id)
    # ⋈ lineitem_os(o_id, s_id) ⋈ supplier(s_id, s_nation)
    # ⋈ nation(n_id=c_nation, r_id) — with s_nation also joining nation
    # Actually: nation used twice makes it cyclic
    q3 = QueryHypergraph(
        relations=(
            Relation(
                "customer", frozenset({"c_id", "nation"}), len(data["customer"][1])
            ),
            Relation("orders", frozenset({"o_id", "c_id"}), len(data["orders"][1])),
            Relation(
                "lineitem_os", frozenset({"o_id", "s_id"}), len(data["lineitem_os"][1])
            ),
            Relation(
                "supplier", frozenset({"s_id", "nation"}), len(data["supplier"][1])
            ),
        ),
        name="tpch_q3_same_nation",
    )
    data_q3 = {
        "customer": (("c_id", "nation"), data["customer"][1]),
        "orders": (("o_id", "c_id"), data["orders"][1]),
        "lineitem_os": (("o_id", "s_id"), data["lineitem_os"][1]),
        "supplier": (("s_id", "nation"), data["supplier"][1]),
    }
    results.append(run_query("TPC-H Q3: Same-nation Cust-Supp cycle-4", q3, data_q3))

    # ---- Q4: Star (control) ----
    # lineitem(orderkey, partkey, suppkey) as 3 binary relations on shared orderkey
    q4 = QueryHypergraph(
        relations=(
            Relation(
                "li_op", frozenset({"orderkey", "partkey"}), len(data["lineitem_op"][1])
            ),
            Relation(
                "li_os", frozenset({"orderkey", "suppkey"}), len(data["lineitem_os"][1])
            ),
            Relation("orders", frozenset({"orderkey", "c_id"}), len(data["orders"][1])),
        ),
        name="tpch_q4_star",
    )
    data_q4 = {
        "li_op": (("orderkey", "partkey"), data["lineitem_op"][1]),
        "li_os": (("orderkey", "suppkey"), data["lineitem_os"][1]),
        "orders": (("orderkey", "c_id"), [(t[0], t[1]) for t in data["orders"][1]]),
    }
    results.append(run_query("TPC-H Q4: Star-3 on orderkey (control)", q4, data_q4))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Name':<55} {'True':>8} {'q(H-CEG)':>10} {'q(LpH)':>10} {'LP':>8}")
    print("-" * 95)
    for r in results:
        print(
            f"{r['name']:<55} {r['true']:>8} {r['q_hceg']:>10.1f}x {r['q_lph']:>10.1f}x {r['ratio']:>7.2f}x"
        )


if __name__ == "__main__":
    main()
