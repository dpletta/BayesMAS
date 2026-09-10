"""Puzzle protocol shared by the Apple-style environments."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Protocol, TypeVar

Move = Hashable
P = TypeVar("P", bound="Puzzle")


class Puzzle(Protocol):
    """Immutable, fully observable planning environment."""

    @property
    def name(self) -> str: ...

    @property
    def complexity(self) -> int: ...

    def is_solved(self) -> bool: ...

    def legal_moves(self) -> tuple[Move, ...]: ...

    def apply(self, move: Move) -> Puzzle: ...

    def encode(self) -> Hashable: ...

    def goal(self) -> Puzzle: ...
