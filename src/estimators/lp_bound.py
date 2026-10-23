"""LP-based optimal weight selection for LpH-CEG hyperpaths.

Given a hyperpath and Lp-norm statistics for each block/interface pair,
solves a linear program to find the optimal weights (w_j) that minimize
the cardinality upper bound:

    min  sum_j  w_j * log mu_{p_j}(F_j | I_j)
    s.t. fractional block cover constraints
         w_j >= 0

The LP is small (variables = number of hyperpath steps * number of
p-values) and solvable in sub-millisecond time.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linprog

if TYPE_CHECKING:
    from src.statistics.lp_stats import LpBlockExtensionStats
    from src.utils.types import (
        AttributeName,
        Block,
        Hyperpath,
        Interface,
        QueryHypergraph,
    )

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LpWeightResult:
    """Result of LP-based weight optimization for a single hyperpath.

    Attributes:
        log_bound: log2 of the optimized upper bound.
        bound: The optimized upper bound (2^log_bound).
        weights: Optimal weight for each (step, p) pair.
        p_values_used: p-value chosen for each hyperpath step.
        weights_per_step: Optimal weight for each step (summed over p).
        feasible: Whether the LP found a feasible solution.
    """

    log_bound: float
    bound: float
    weights: dict[tuple[int, float], float]
    p_values_used: tuple[float, ...]
    weights_per_step: tuple[float, ...]
    feasible: bool


def optimize_hyperpath_weights(
    query: QueryHypergraph,
    hyperpath: Hyperpath,
    lp_stats: dict[tuple[Block, Interface], LpBlockExtensionStats],
    *,
    p_values: tuple[float, ...] = (1.0, 2.0, float("inf")),
) -> LpWeightResult:
    """Solve LP to find optimal weights for a hyperpath.

    For each step j in the hyperpath and each p-value, we have a
    variable w_{j,p} >= 0.  The objective minimizes the log-bound:

        min  sum_{j,p}  w_{j,p} * log2(mu_p(F_j | I_j))

    Subject to the fractional block cover constraint: for each attribute
    a in A(Q), the weighted coverage must be >= 1.

    Args:
        query: Query hypergraph.
        hyperpath: A valid hyperpath from empty to full query.
        lp_stats: Lp-norm statistics for block/interface pairs.
        p_values: p-values to consider in the LP.

    Returns:
        LP weight optimization result.
    """
    m = hyperpath.num_steps
    n_p = len(p_values)
    n_vars = m * n_p

    if n_vars == 0:
        return LpWeightResult(
            log_bound=0.0,
            bound=1.0,
            weights={},
            p_values_used=(),
            weights_per_step=(),
            feasible=True,
        )

    # Collect block info for each step
    steps = _collect_step_info(query, hyperpath)

    # Build objective: min sum w_{j,p} * log2(mu_p)
    c = np.zeros(n_vars)
    for j, step in enumerate(steps):
        key = (step.block, step.interface)
        if key not in lp_stats:
            LOGGER.warning(
                "Missing Lp stats for block=%s interface=%s, using inf weight",
                sorted(step.block.relations),
                sorted(step.interface),
            )
            for pi, _p in enumerate(p_values):
                c[j * n_p + pi] = 1e6  # large penalty
            continue

        stats = lp_stats[key]
        for pi, p in enumerate(p_values):
            norm_val = stats.norm(p)
            if norm_val <= 1.0:
                c[j * n_p + pi] = 0.0
            else:
                c[j * n_p + pi] = math.log2(norm_val)

    # Build cover constraints: for each attribute a, coverage >= 1
    all_attrs = sorted(query.all_attributes)
    n_constraints = len(all_attrs)
    a_ub = np.zeros((n_constraints, n_vars))
    b_ub = np.full(n_constraints, -1.0)  # -coverage <= -1 => coverage >= 1

    for ai, attr in enumerate(all_attrs):
        for j, step in enumerate(steps):
            for pi, p in enumerate(p_values):
                var_idx = j * n_p + pi
                coverage = _attribute_coverage(attr, step, p)
                # A_ub * x <= b_ub, so we negate: -coverage * w <= -1
                a_ub[ai, var_idx] = -coverage

    # Variable bounds: w >= 0
    bounds = [(0.0, None) for _ in range(n_vars)]

    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        LOGGER.warning("LP infeasible for hyperpath, falling back to uniform weights")
        return _uniform_fallback(hyperpath, lp_stats, steps, p_values)

    # The solver meets the cover constraints only to its feasibility tolerance
    # (about 1e-7).  Scale the weights so every attribute is covered exactly,
    # as the Composition Inequality requires; this raises the bound by the
    # factor B^(eps/(1-eps)) for a shortfall eps (extended version).
    x = np.maximum(np.asarray(result.x, dtype=float), 0.0)
    min_cov = float(np.min(-(a_ub @ x))) if n_constraints else 1.0
    if 0.0 < min_cov < 1.0:
        x = x / min_cov

    # Extract results
    weights: dict[tuple[int, float], float] = {}
    weights_per_step_list: list[float] = []
    p_used_list: list[float] = []

    for j in range(m):
        step_total = 0.0
        best_p = p_values[0]
        best_w = 0.0
        for pi, p in enumerate(p_values):
            w = float(x[j * n_p + pi])
            weights[(j, p)] = w
            step_total += w
            if w > best_w:
                best_w = w
                best_p = p
        weights_per_step_list.append(step_total)
        p_used_list.append(best_p)

    log_bound = float(c @ x)

    return LpWeightResult(
        log_bound=log_bound,
        bound=2.0**log_bound,
        weights=weights,
        p_values_used=tuple(p_used_list),
        weights_per_step=tuple(weights_per_step_list),
        feasible=True,
    )


@dataclass(frozen=True)
class _StepInfo:
    """Info for a single hyperpath step."""

    block: Block
    interface: Interface
    extension_attrs: frozenset[AttributeName]
    interface_attrs: frozenset[AttributeName]


def _collect_step_info(
    query: QueryHypergraph,
    hyperpath: Hyperpath,
) -> list[_StepInfo]:
    """Collect block, interface, and attribute info for each step."""
    steps: list[_StepInfo] = []
    for j, block in enumerate(hyperpath.blocks):
        state = hyperpath.states[j]
        interface = query.interface(state, block)
        block_attrs = query.attributes_of(block)
        extension_attrs = block_attrs - interface
        steps.append(
            _StepInfo(
                block=block,
                interface=interface,
                extension_attrs=frozenset(extension_attrs),
                interface_attrs=frozenset(interface),
            )
        )
    return steps


def _attribute_coverage(
    attr: AttributeName,
    step: _StepInfo,
    p: float,
) -> float:
    """Coverage contribution of variable w_{j,p} for attribute a.

    From the information-theoretic derivation:
    - If a in extension attrs of step j: contributes w_j
    - If a in interface attrs of step j: contributes w_j / p
    - Otherwise: contributes 0
    """
    if attr in step.extension_attrs:
        return 1.0
    if attr in step.interface_attrs:
        if p == float("inf"):
            return 0.0  # 1/inf = 0
        return 1.0 / p
    return 0.0


def _uniform_fallback(
    hyperpath: Hyperpath,
    lp_stats: dict[tuple[Block, Interface], LpBlockExtensionStats],
    steps: list[_StepInfo],
    p_values: tuple[float, ...],
) -> LpWeightResult:
    """Fallback: use w=1, p=inf (equivalent to standard H-CEG)."""
    log_bound = 0.0
    weights: dict[tuple[int, float], float] = {}
    p_used: list[float] = []
    w_per_step: list[float] = []

    for j, step in enumerate(steps):
        key = (step.block, step.interface)
        for _pi, p in enumerate(p_values):
            weights[(j, p)] = 1.0 if p == float("inf") else 0.0

        log_max = lp_stats[key].log_norm(float("inf")) if key in lp_stats else 0.0
        log_bound += log_max
        p_used.append(float("inf"))
        w_per_step.append(1.0)

    return LpWeightResult(
        log_bound=log_bound,
        bound=2.0**log_bound,
        weights=weights,
        p_values_used=tuple(p_used),
        weights_per_step=tuple(w_per_step),
        feasible=False,
    )
