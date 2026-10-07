"""WCOJ evaluation: variable ordering quality and residual bound accuracy.

Provides quantitative metrics for comparing H-CEG-guided variable orderings
against baseline strategies (lexicographic, min-degree, random).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.core.hceg import DEFAULT_MAX_STATES
from src.wcoj.residual_bound import residual_bound
from src.wcoj.variable_order import variable_ordering

if TYPE_CHECKING:
    from src.statistics.block_stats import RelationDataMap
    from src.utils.types import AttributeName, QueryHypergraph

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrderingEvaluation:
    """Evaluation result for a single variable ordering strategy.

    Attributes:
        strategy_name: Name of the ordering strategy.
        ordering: The variable ordering produced.
        prefix_residual_bounds: Residual bounds at each prefix length.
        total_residual_product: Product of all prefix residual bounds
            (lower is better — tighter overall bound).
        max_residual_bound: Maximum single-step residual bound
            (lower is better — avoids large intermediate blowups).
    """

    strategy_name: str
    ordering: list[AttributeName]
    prefix_residual_bounds: list[float]
    total_residual_product: float
    max_residual_bound: float


@dataclass
class WCOJEvaluationResult:
    """Comparative evaluation of multiple ordering strategies.

    Attributes:
        query_name: Name of the evaluated query.
        evaluations: Per-strategy evaluation results.
        best_strategy: Name of the strategy with the smallest total
            residual product.
        improvement_over_lexicographic: Ratio of lexicographic total
            product to best strategy total product.
    """

    query_name: str
    evaluations: dict[str, OrderingEvaluation] = field(default_factory=dict)
    best_strategy: str = ""
    improvement_over_lexicographic: float = 1.0


def lexicographic_ordering(query: QueryHypergraph) -> list[AttributeName]:
    """Return attributes in lexicographic order."""
    return sorted(query.all_attributes)


def min_degree_ordering(query: QueryHypergraph) -> list[AttributeName]:
    """Return attributes ordered by ascending relation coverage (min-degree).

    Attributes appearing in fewer relations come first, breaking ties
    lexicographically.
    """
    coverage: dict[AttributeName, int] = {
        attr: 0 for attr in query.all_attributes
    }
    for relation in query.relations:
        for attr in relation.attributes:
            coverage[attr] += 1

    return sorted(
        query.all_attributes,
        key=lambda attr: (coverage[attr], attr),
    )


def max_degree_ordering(query: QueryHypergraph) -> list[AttributeName]:
    """Return attributes ordered by descending relation coverage (max-degree).

    Attributes appearing in more relations come first, breaking ties
    lexicographically.
    """
    coverage: dict[AttributeName, int] = {
        attr: 0 for attr in query.all_attributes
    }
    for relation in query.relations:
        for attr in relation.attributes:
            coverage[attr] += 1

    return sorted(
        query.all_attributes,
        key=lambda attr: (-coverage[attr], attr),
    )


def _evaluate_ordering(
    query: QueryHypergraph,
    relation_data: RelationDataMap,
    ordering: list[AttributeName],
    strategy_name: str,
    max_block_size: int,
    max_states: int,
) -> OrderingEvaluation:
    """Evaluate a single ordering by computing residual bounds at each prefix."""
    prefix_bounds: list[float] = []

    for i in range(len(ordering)):
        prefix = ordering[:i]
        bound = residual_bound(
            query,
            relation_data,
            prefix,
            max_block_size=max_block_size,
            max_states=max_states,
        )
        prefix_bounds.append(bound)

    total_product = 1.0
    for b in prefix_bounds:
        total_product *= max(b, 1.0)

    max_bound = max(prefix_bounds) if prefix_bounds else 1.0

    return OrderingEvaluation(
        strategy_name=strategy_name,
        ordering=ordering,
        prefix_residual_bounds=prefix_bounds,
        total_residual_product=total_product,
        max_residual_bound=max_bound,
    )


def evaluate_wcoj_orderings(
    query: QueryHypergraph,
    relation_data: RelationDataMap,
    *,
    max_block_size: int = 3,
    max_states: int = DEFAULT_MAX_STATES,
    include_strategies: tuple[str, ...] = (
        "hceg",
        "lexicographic",
        "min_degree",
        "max_degree",
    ),
) -> WCOJEvaluationResult:
    """Evaluate H-CEG variable ordering against baseline strategies.

    Args:
        query: Query hypergraph to evaluate.
        relation_data: Exact relation tuples.
        max_block_size: Block size for H-CEG ordering and residual bounds.
        max_states: Max H-CEG DAG states.
        include_strategies: Tuple of strategy names to evaluate.

    Returns:
        Comparative evaluation result with per-strategy metrics.
    """
    strategy_fns: dict[str, list[AttributeName]] = {}

    if "hceg" in include_strategies:
        strategy_fns["hceg"] = variable_ordering(
            query, relation_data,
            max_block_size=max_block_size,
            max_states=max_states,
        )

    if "lexicographic" in include_strategies:
        strategy_fns["lexicographic"] = lexicographic_ordering(query)

    if "min_degree" in include_strategies:
        strategy_fns["min_degree"] = min_degree_ordering(query)

    if "max_degree" in include_strategies:
        strategy_fns["max_degree"] = max_degree_ordering(query)

    result = WCOJEvaluationResult(query_name=query.name)

    for name, ordering in strategy_fns.items():
        evaluation = _evaluate_ordering(
            query, relation_data, ordering, name,
            max_block_size, max_states,
        )
        result.evaluations[name] = evaluation

    # Determine best strategy
    if result.evaluations:
        best_name = min(
            result.evaluations,
            key=lambda name: result.evaluations[name].total_residual_product,
        )
        result.best_strategy = best_name

        # Improvement over lexicographic
        if "lexicographic" in result.evaluations:
            lex_product = result.evaluations["lexicographic"].total_residual_product
            best_product = result.evaluations[best_name].total_residual_product
            if best_product > 0:
                result.improvement_over_lexicographic = lex_product / best_product

    LOGGER.info(
        "WCOJ evaluation for %r: best=%s, improvement=%.2fx over lexicographic",
        query.name,
        result.best_strategy,
        result.improvement_over_lexicographic,
    )

    return result
