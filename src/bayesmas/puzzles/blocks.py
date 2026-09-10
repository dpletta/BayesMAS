"""Blocks world on three stacks. Complexity N is the block count.

Start: blocks ``1..n`` stacked on peg 0 (``n`` on top). Goal: the reverse
order on peg 2. Any top block may move onto any other stack.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BlocksWorld:
    stacks: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    n: int

    @property
    def name(self) -> str:
        return "blocks"

    @property
    def complexity(self) -> int:
        return self.n

    @classmethod
    def start(cls, n: int) -> BlocksWorld:
        if n < 1:
            raise ValueError("n must be >= 1")
        return cls(stacks=(tuple(range(1, n + 1)), (), ()), n=n)

    def goal(self) -> BlocksWorld:
        return BlocksWorld(stacks=((), (), tuple(range(self.n, 0, -1))), n=self.n)

    def is_solved(self) -> bool:
        return self.stacks == self.goal().stacks

    def encode(self) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return self.stacks

    def legal_moves(self) -> tuple[tuple[int, int], ...]:
        moves: list[tuple[int, int]] = []
        for i in range(3):
            if not self.stacks[i]:
                continue
            for j in range(3):
                if i != j:
                    moves.append((i, j))
        return tuple(moves)

    def apply(self, move: object) -> BlocksWorld:
        if move not in set(self.legal_moves()):
            raise ValueError(f"illegal blocks move: {move!r}")
        src, dest = move
        block = self.stacks[src][-1]
        new_stacks = list(self.stacks)
        new_stacks[src] = self.stacks[src][:-1]
        new_stacks[dest] = (*self.stacks[dest], block)
        return BlocksWorld(stacks=(new_stacks[0], new_stacks[1], new_stacks[2]), n=self.n)
