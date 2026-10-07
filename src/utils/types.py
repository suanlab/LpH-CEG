"""Shared type definitions for H-CEG."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Iterator

# A relation is identified by name
RelationName: TypeAlias = str

# An attribute is identified by name
AttributeName: TypeAlias = str

# Interface attributes between current state and a block
Interface: TypeAlias = frozenset[AttributeName]


@dataclass(frozen=True)
class Block:
    """A block of relations added in one extension step."""

    relations: frozenset[RelationName]

    def __iter__(self) -> Iterator[RelationName]:
        return iter(self.relations)

    def __len__(self) -> int:
        return len(self.relations)

    def isdisjoint(self, other: State) -> bool:
        """Return whether the block shares no relations with a state."""
        return self.relations.isdisjoint(other.relations)


@dataclass(frozen=True)
class State:
    """A join state represented by the relations already covered."""

    relations: frozenset[RelationName]

    def __iter__(self) -> Iterator[RelationName]:
        return iter(self.relations)

    def __len__(self) -> int:
        return len(self.relations)

    def union(self, block: Block) -> State:
        """Return the state reached after adding a block."""
        return State(self.relations | block.relations)


@dataclass(frozen=True)
class Relation:
    """A relation (table) in the query.

    Attributes:
        name: Unique name identifying the relation.
        attributes: Set of attribute names in the relation schema.
        cardinality: Number of tuples in the relation.
        selectivity: Fraction of tuples surviving filter predicates
            (1.0 = no filter, 0.5 = half filtered). Must be in (0, 1].
    """

    name: RelationName
    attributes: frozenset[AttributeName]
    cardinality: int
    selectivity: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.selectivity <= 1.0:
            msg = (
                f"Relation {self.name!r} selectivity must be in (0, 1], "
                f"got {self.selectivity}"
            )
            raise ValueError(msg)

    @property
    def effective_cardinality(self) -> float:
        """Cardinality after applying filter selectivity."""
        return self.cardinality * self.selectivity


@dataclass(frozen=True)
class QueryHypergraph:
    """A query represented as a hypergraph.

    Vertices are attributes, hyperedges are relations.
    """

    relations: tuple[Relation, ...]
    name: str = ""

    def __post_init__(self) -> None:
        relation_names = [relation.name for relation in self.relations]
        if len(set(relation_names)) != len(relation_names):
            msg = "Query hypergraph relation names must be unique"
            raise ValueError(msg)
        if len(self.relations) > 1 and not self._is_relation_subset_connected(
            frozenset(self.relation_names)
        ):
            msg = "Query hypergraph must be connected"
            raise ValueError(msg)

    @property
    def all_attributes(self) -> frozenset[AttributeName]:
        """All attributes across all relations."""
        result: set[AttributeName] = set()
        for rel in self.relations:
            result |= rel.attributes
        return frozenset(result)

    @property
    def relation_names(self) -> frozenset[RelationName]:
        """All relation names."""
        return frozenset(r.name for r in self.relations)

    def get_relation(self, name: RelationName) -> Relation:
        """Get relation by name."""
        for rel in self.relations:
            if rel.name == name:
                return rel
        msg = f"Relation {name!r} not found"
        raise KeyError(msg)

    def _attributes_of_relations(
        self, relation_names: frozenset[RelationName]
    ) -> frozenset[AttributeName]:
        result: set[AttributeName] = set()
        for relation_name in relation_names:
            result |= self.get_relation(relation_name).attributes
        return frozenset(result)

    def _relation_neighbors(self, name: RelationName) -> frozenset[RelationName]:
        source_attributes = self.get_relation(name).attributes
        neighbors: set[RelationName] = set()
        for relation in self.relations:
            if relation.name == name:
                continue
            if source_attributes & relation.attributes:
                neighbors.add(relation.name)
        return frozenset(neighbors)

    def _is_relation_subset_connected(
        self, relation_names: frozenset[RelationName]
    ) -> bool:
        if not relation_names:
            return False
        if len(relation_names) == 1:
            return True

        start = next(iter(relation_names))
        visited: set[RelationName] = {start}
        queue: deque[RelationName] = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in self._relation_neighbors(current):
                if neighbor not in relation_names or neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
        return visited == set(relation_names)

    def attributes_of(self, block: Block) -> frozenset[AttributeName]:
        """Get union of attributes for a block of relations."""
        return self._attributes_of_relations(block.relations)

    def attributes_of_state(self, state: State) -> frozenset[AttributeName]:
        """Get union of attributes for a state of relations."""
        return self._attributes_of_relations(state.relations)

    def interface(self, state: State, block: Block) -> Interface:
        """Compute interface I = A(state) ∩ A(block)."""
        state_attrs = self.attributes_of_state(state)
        block_attrs = self.attributes_of(block)
        return frozenset(state_attrs & block_attrs)

    def is_valid_block(self, block: Block) -> bool:
        """Return whether a block is non-empty, in-query, and connected."""
        if not block.relations:
            return False
        if not block.relations <= self.relation_names:
            return False
        return self._is_relation_subset_connected(block.relations)

    def remaining_attributes(self, state: State) -> frozenset[AttributeName]:
        """Return attributes not yet covered by the current state."""
        if not state.relations <= self.relation_names:
            msg = "State contains relations outside the query hypergraph"
            raise ValueError(msg)
        return frozenset(self.all_attributes - self.attributes_of_state(state))

    def is_complete(self, state: State) -> bool:
        """Return whether the state covers all relations in the query."""
        if not state.relations <= self.relation_names:
            msg = "State contains relations outside the query hypergraph"
            raise ValueError(msg)
        return state.relations == self.relation_names

    def valid_next_blocks(self, state: State, max_block_size: int) -> tuple[Block, ...]:
        """Enumerate valid extension blocks from the current state."""
        if max_block_size < 1:
            msg = "max_block_size must be at least 1"
            raise ValueError(msg)
        if not state.relations <= self.relation_names:
            msg = "State contains relations outside the query hypergraph"
            raise ValueError(msg)

        remaining_relations = tuple(sorted(self.relation_names - state.relations))
        blocks: list[Block] = []
        for size in range(1, min(max_block_size, len(remaining_relations)) + 1):
            for combination in combinations(remaining_relations, size):
                block = Block(frozenset(combination))
                if not self.is_valid_block(block):
                    continue
                if state.relations and not self.interface(state, block):
                    continue
                blocks.append(block)
        return tuple(blocks)


@dataclass(frozen=True)
class OptimisticEdgeWeight:
    """Weight for H-CEG_O using multiplicative extension rates."""

    source: State
    target: State
    block: Block
    interface: Interface
    weight: float


@dataclass(frozen=True)
class PessimisticEdgeWeight:
    """Weight for H-CEG_M using multiplicative factors in log-space."""

    source: State
    target: State
    block: Block
    interface: Interface
    weight: float


@dataclass(frozen=True)
class Hyperpath:
    """A hyperpath from empty set to full query in H-CEG."""

    states: tuple[State, ...]
    blocks: tuple[Block, ...]
    weights: tuple[float, ...]

    @property
    def num_steps(self) -> int:
        """Number of extension steps."""
        return len(self.blocks)

    @property
    def product_estimate(self) -> float:
        """Optimistic estimate: product of weights."""
        result = 1.0
        for w in self.weights:
            result *= w
        return result

    @property
    def sum_log_estimate(self) -> float:
        """Pessimistic estimate: sum of log weights."""
        if any(weight == 0 for weight in self.weights):
            return -math.inf
        if any(weight < 0 for weight in self.weights):
            msg = "Hyperpath weights must be non-negative"
            raise ValueError(msg)
        return sum(math.log(weight) for weight in self.weights)


@dataclass
class EstimationResult:
    """Result of a cardinality estimation."""

    estimate: float
    hyperpath: Hyperpath
    method: str  # "optimistic", "pessimistic", "hybrid"
    metadata: dict[str, object] = field(default_factory=dict)
