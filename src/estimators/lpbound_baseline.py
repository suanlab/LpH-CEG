"""LpBound baseline estimator for comparison with LpH-CEG.

Implements the LpBound approach (SIGMOD 2025) within our framework:
binary extensions only (block_size=1) with LP-optimized Lp-norm weights.

This serves as the direct comparison target to demonstrate LpH-CEG's
advantage from multi-way block extensions.  The LP formulation is
identical to LpH-CEG but restricted to singleton blocks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.estimators.lph_ceg import estimate_lph_ceg

if TYPE_CHECKING:
    from src.statistics.block_stats import RelationDataMap
    from src.utils.types import EstimationResult, QueryHypergraph

LOGGER = logging.getLogger(__name__)

DEFAULT_P_VALUES: tuple[float, ...] = (1.0, 2.0, float("inf"))


def estimate_lpbound(
    query: QueryHypergraph,
    relation_data: RelationDataMap,
    *,
    p_values: tuple[float, ...] = DEFAULT_P_VALUES,
    apply_selectivity: bool = True,
) -> EstimationResult:
    """Estimate cardinality using the LpBound approach (binary LP only).

    This is equivalent to LpH-CEG with max_block_size=1: only singleton
    blocks (individual relations) are used, but LP optimization selects
    optimal (p, w) weights across the fractional cover.

    This baseline isolates the contribution of LP weight optimization
    from the contribution of multi-way block extensions, enabling
    a clean 3-way comparison:

        CEG_M (bs=1, no LP) vs LpBound (bs=1, LP) vs LpH-CEG (bs>=2, LP)

    Args:
        query: Query hypergraph.
        relation_data: Exact relation tuples.
        p_values: Lp-norm exponents for the LP.
        apply_selectivity: Scale bound by relation selectivities.

    Returns:
        Estimation result with LpBound-style LP-optimized bound.
    """
    result = estimate_lph_ceg(
        query,
        relation_data,
        max_block_size=1,
        p_values=p_values,
        apply_selectivity=apply_selectivity,
    )

    # Re-tag metadata for clarity in comparison scripts
    metadata = dict(result.metadata) if result.metadata else {}
    metadata["baseline_method"] = "lpbound"
    metadata["description"] = (
        "Binary (bs=1) fractional cover-LP; a polynomial-size relaxation "
        "of the entropic LpBound, not the entropic bound itself"
    )

    from src.utils.types import EstimationResult

    return EstimationResult(
        estimate=result.estimate,
        hyperpath=result.hyperpath,
        method="lpbound",
        metadata=metadata,
    )
