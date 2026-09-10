"""Tower of Hanoi. Complexity N is the disk count; optimal length is 2^N − 1."""

from __future__ import annotations

from dataclasses import dataclass

_PEGS = ("A", "B", "C")


@dataclass(frozen=True, slots=True)
class TowerOfHanoi:
    """Disks ``1..n`` with ``1`` smallest. Peg stacks are bottom-to-top."""

    pegs: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    n: int

    @property
    def name(self) -> str:
        return "hanoi"

    @property
    def complexity(self) -> int:
        return self.n

    @classmethod
    def start(cls, n: int) -> TowerOfHanoi:
        if n < 1:
            raise ValueError("n must be >= 1")
        return cls(pegs=(tuple(range(n, 0, -1)), (), ()), n=n)

    def goal(self) -> TowerOfHanoi:
        return TowerOfHanoi(pegs=((), (), tuple(range(self.n, 0, -1))), n=self.n)

    def is_solved(self) -> bool:
        return self.pegs == self.goal().pegs

    def encode(self) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return self.pegs

    def legal_moves(self) -> tuple[tuple[str, str], ...]:
        moves: list[tuple[str, str]] = []
        for i, src in enumerate(_PEGS):
            if not self.pegs[i]:
                continue
            moving = self.pegs[i][-1]
            for j, dest in enumerate(_PEGS):
                if i == j:
                    continue
                top = self.pegs[j][-1] if self.pegs[j] else None
                if top is None or moving < top:
                    moves.append((src, dest))
        return tuple(moves)

    def apply(self, move: object) -> TowerOfHanoi:
        if move not in set(self.legal_moves()):
            raise ValueError(f"illegal Hanoi move: {move!r}")
        src, dest = move
        i, j = _PEGS.index(src), _PEGS.index(dest)
        disk = self.pegs[i][-1]
        new_pegs = list(self.pegs)
        new_pegs[i] = self.pegs[i][:-1]
        new_pegs[j] = (*self.pegs[j], disk)
        return TowerOfHanoi(pegs=(new_pegs[0], new_pegs[1], new_pegs[2]), n=self.n)
