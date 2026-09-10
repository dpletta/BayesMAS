"""Actors-and-agents river crossing (Apple Illusion-of-Thinking).

``n`` actors and ``n`` agents. The boat holds at most ``k`` people and cannot
travel empty. An actor may not share a bank with a foreign agent unless their
own agent is also present.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

Person = tuple[str, int]


def _everyone(n_pairs: int) -> frozenset[Person]:
    people = [(kind, i) for i in range(1, n_pairs + 1) for kind in ("A", "G")]
    return frozenset(people)


def _side_ok(people: frozenset[Person]) -> bool:
    agents = {i for kind, i in people if kind == "G"}
    actors = {i for kind, i in people if kind == "A"}
    for actor in actors:
        foreign = [j for j in agents if j != actor]
        if foreign and actor not in agents:
            return False
    return True


@dataclass(frozen=True, slots=True)
class RiverCrossing:
    n_pairs: int
    boat_capacity: int
    left: frozenset[Person]
    boat_on_left: bool

    @property
    def name(self) -> str:
        return "river"

    @property
    def complexity(self) -> int:
        return self.n_pairs

    @classmethod
    def start(cls, n_pairs: int, boat_capacity: int = 2) -> RiverCrossing:
        if n_pairs < 1:
            raise ValueError("n_pairs must be >= 1")
        if boat_capacity < 1:
            raise ValueError("boat_capacity must be >= 1")
        return cls(
            n_pairs=n_pairs,
            boat_capacity=boat_capacity,
            left=_everyone(n_pairs),
            boat_on_left=True,
        )

    def goal(self) -> RiverCrossing:
        return RiverCrossing(
            n_pairs=self.n_pairs,
            boat_capacity=self.boat_capacity,
            left=frozenset(),
            boat_on_left=False,
        )

    def is_solved(self) -> bool:
        return len(self.left) == 0 and not self.boat_on_left

    def encode(self) -> tuple[frozenset[Person], bool]:
        return (self.left, self.boat_on_left)

    def _right(self) -> frozenset[Person]:
        return _everyone(self.n_pairs) - self.left

    def _result(self, passengers: frozenset[Person]) -> RiverCrossing:
        if self.boat_on_left:
            new_left = self.left - passengers
            boat_on_left = False
        else:
            new_left = self.left | passengers
            boat_on_left = True
        return RiverCrossing(
            n_pairs=self.n_pairs,
            boat_capacity=self.boat_capacity,
            left=new_left,
            boat_on_left=boat_on_left,
        )

    def _move_ok(self, passengers: frozenset[Person]) -> bool:
        if not 1 <= len(passengers) <= self.boat_capacity:
            return False
        side = self.left if self.boat_on_left else self._right()
        if not passengers <= side:
            return False
        nxt = self._result(passengers)
        return _side_ok(nxt.left) and _side_ok(nxt._right())

    def legal_moves(self) -> tuple[frozenset[Person], ...]:
        side = self.left if self.boat_on_left else self._right()
        ordered = tuple(sorted(side))
        moves: list[frozenset[Person]] = []
        for r in range(1, self.boat_capacity + 1):
            for combo in combinations(ordered, r):
                passengers = frozenset(combo)
                if self._move_ok(passengers):
                    moves.append(passengers)
        return tuple(moves)

    def apply(self, move: object) -> RiverCrossing:
        if not isinstance(move, frozenset) or not self._move_ok(move):
            raise ValueError(f"illegal river move: {move!r}")
        return self._result(move)
