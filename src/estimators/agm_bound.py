"""True AGM / fractional-edge-cover cardinality bound.

The AGM (Atserias-Grohe-Marx) bound is the classical worst-case-optimal
upper bound on the output cardinality of a natural join.  It is the optimum
of the fractional-edge-cover linear program over the query hypergraph::

    min   prod_e |R_e| ** x_e
    s.t.  sum_{e : v in attr(e)} x_e >= 1   for every attribute v
          x_e >= 0                          for every relation e

Taking ``log2`` turns the product objective into a linear one, so the LP is
solved in ``log2`` units with :func:`scipy.optimize.linprog` (``method="highs"``)::

    min   sum_e x_e * log2(|R_e|)
    s.t.  the cover constraints above.

The returned bound is ``2 ** (optimal objective)``.  Here ``|R_e|`` is the
number of tuples in relation ``e``.

For a simple cycle (or any query where the optimal cover assigns weight
``1/2`` to every edge) with all relations of equal size ``N``, the bound is
the well-known closed form ``N ** (k/2)`` (e.g. ``N ** 1.5`` for a triangle).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linprog

if TYPE_CHECKING:
    from src.statistics.block_stats import RelationDataMap
    from src.utils.types import QueryHypergraph


def estimate_agm(query: QueryHypergraph, relation_data: RelationDataMap) -> float:
    """Estimate join cardinality via the AGM fractional-edge-cover bound.

    Solves the fractional-edge-cover LP in ``log2`` units and returns
    ``2 ** (optimal objective)``.

    Args:
        query: Query hypergraph whose relations are the hyperedges and whose
            attributes are the vertices.
        relation_data: Mapping from relation name to ``(schema, rows)``;
            ``|R_e|`` is the number of tuples in the relation.

    Returns:
        The AGM cardinality upper bound (a positive float).  An empty query
        returns ``1.0``; a query containing an empty relation returns ``0.0``.

    Raises:
        ValueError: If the LP fails to solve.
    """
    relations = query.relations
    if not relations:
        return 1.0

    # |R_e| = number of tuples in relation e.
    cardinalities: list[int] = []
    for relation in relations:
        rows = relation_data[relation.name][1]
        cardinalities.append(len(rows))

    # An empty relation forces an empty join result.
    if any(card == 0 for card in cardinalities):
        return 0.0

    # Objective (log2 units): minimize sum_e x_e * log2(|R_e|).
    c = [math.log2(card) for card in cardinalities]

    # Cover constraints, expressed as A_ub x <= b_ub:
    #   sum_{e : v in e} x_e >= 1   <->   -sum_{e : v in e} x_e <= -1.
    attributes = sorted(query.all_attributes)
    a_ub_rows: list[list[float]] = []
    b_ub: list[float] = []
    for attr in attributes:
        row = [-1.0 if attr in relation.attributes else 0.0 for relation in relations]
        a_ub_rows.append(row)
        b_ub.append(-1.0)

    a_ub = np.asarray(a_ub_rows, dtype=float)
    b_ub_arr = np.asarray(b_ub, dtype=float)
    bounds: list[tuple[float, float | None]] = [(0.0, None)] * len(relations)

    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub_arr,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        msg = f"AGM LP failed to solve: {result.message}"
        raise ValueError(msg)

    return float(2.0 ** float(result.fun))
