"""One-dimensional checker jumping (Apple Illusion-of-Thinking).

``n`` red checkers, ``n`` blue, one empty. Reds move only right; blues only
left. Slide into an adjacent empty or jump one opposite-color checker.
Optimal length with ``2n`` checkers is (n+1)² − 1.
"""

from __future__ import annotations

from dataclasses import dataclass

RED = 1
BLUE = -1
EMPTY = 0


@dataclass(frozen=True, slots=True)
class CheckerJumping:
    cells: tuple[int, ...]
    n: int

    @property
    def name(self) -> str:
        return "checkers"

    @property
    def complexity(self) -> int:
        return self.n

    @classmethod
    def start(cls, n: int) -> CheckerJumping:
        if n < 1:
            raise ValueError("n must be >= 1")
        cells = (RED,) * n + (EMPTY,) + (BLUE,) * n
        return cls(cells=cells, n=n)

    def goal(self) -> CheckerJumping:
        return CheckerJumping(cells=(BLUE,) * self.n + (EMPTY,) + (RED,) * self.n, n=self.n)

    def is_solved(self) -> bool:
        return self.cells == self.goal().cells

    def encode(self) -> tuple[int, ...]:
        return self.cells

    def legal_moves(self) -> tuple[tuple[int, int], ...]:
        cells = self.cells
        empty = cells.index(EMPTY)
        moves: list[tuple[int, int]] = []
        for i, piece in enumerate(cells):
            if piece == RED:
                if i + 1 == empty:
                    moves.append((i, i + 1))
                elif i + 2 == empty and i + 1 < len(cells) and cells[i + 1] == BLUE:
                    moves.append((i, i + 2))
            elif piece == BLUE:
                if i - 1 == empty:
                    moves.append((i, i - 1))
                elif i - 2 == empty and i - 1 >= 0 and cells[i - 1] == RED:
                    moves.append((i, i - 2))
        return tuple(moves)

    def apply(self, move: object) -> CheckerJumping:
        if move not in set(self.legal_moves()):
            raise ValueError(f"illegal checker move: {move!r}")
        src, dest = move
        cells = list(self.cells)
        cells[dest] = cells[src]
        cells[src] = EMPTY
        return CheckerJumping(cells=tuple(cells), n=self.n)
