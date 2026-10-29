"""The bs=1 fractional cover LP (single-relation statistics).

This is LpH-CEG restricted to singleton blocks: a polynomial-size cover LP
over LpBound's statistics.  It is **not** LpBound's bound, which is the
entropic (polymatroid) program over the same statistics
(:mod:`src.estimators.lpbound_entropic`, or the flow LP in
:mod:`src.estimators.flow_bound`) and is at least as tight.  Every
comparison against LpBound in the paper uses those, not this module.

``estimate_cover_bs1`` is the name to use; ``estimate_lpbound`` is a
deprecated alias kept for old scripts.
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


def estimate_cover_bs1(
    query: QueryHypergraph,
    relation_data: RelationDataMap,
    *,
    p_values: tuple[float, ...] = DEFAULT_P_VALUES,
    apply_selectivity: bool = False,
) -> EstimationResult:
    """The bs=1 cover-LP bound (LpH-CEG with singleton blocks).

    This is equivalent to LpH-CEG with max_block_size=1: only singleton
    blocks (individual relations) are used, but LP optimization selects
    optimal (p, w) weights across the fractional cover.

    This baseline isolates the contribution of LP weight optimization
    from the contribution of multi-way block extensions, enabling
    a clean 3-way comparison:

        chain (bs=1, no LP) vs cover LP (bs=1) vs LpH-CEG (bs>=2, LP)

    Args:
        query: Query hypergraph.
        relation_data: Exact relation tuples.
        p_values: Lp-norm exponents for the LP.
        apply_selectivity: Scale bound by relation selectivities.

    Returns:
        Estimation result of the bs=1 cover LP.
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
    metadata["baseline_method"] = "cover_bs1"
    metadata["description"] = (
        "Binary (bs=1) fractional cover-LP; a polynomial-size relaxation "
        "of the entropic LpBound, not the entropic bound itself"
    )

    from src.utils.types import EstimationResult

    return EstimationResult(
        estimate=result.estimate,
        hyperpath=result.hyperpath,
        method="cover_bs1",
        metadata=metadata,
    )


# Deprecated: the name suggested LpBound's (entropic) bound, which it is not.
estimate_lpbound = estimate_cover_bs1
