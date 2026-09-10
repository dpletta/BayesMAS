"""Apple Illusion-of-Thinking puzzle simulators."""

from __future__ import annotations

import pytest

from bayesmas.puzzles import BlocksWorld, CheckerJumping, RiverCrossing, TowerOfHanoi
from bayesmas.puzzles.solve import shortest_length


def test_hanoi_start_is_unsolved_and_goal_is_solved() -> None:
    start = TowerOfHanoi.start(3)
    assert start.complexity == 3
    assert start.is_solved() is False
    assert start.goal().is_solved() is True
    assert shortest_length(start) == 2**3 - 1


def test_hanoi_rejects_larger_on_smaller() -> None:
    start = TowerOfHanoi.start(2)
    # Disk 2 (larger) cannot move onto disk 1.
    legal = set(start.legal_moves())
    assert ("A", "B") in legal  # smallest disk may move
    applied = start.apply(("A", "C"))
    # After moving smallest to C, moving the large disk onto C is illegal.
    assert ("A", "C") not in set(applied.legal_moves())


def test_hanoi_apply_unknown_move_raises() -> None:
    start = TowerOfHanoi.start(2)
    with pytest.raises(ValueError, match="illegal"):
        start.apply(("B", "A"))


def test_checkers_mirrors_the_line_and_has_known_optimal_length() -> None:
    start = CheckerJumping.start(2)
    assert start.is_solved() is False
    assert start.goal().is_solved() is True
    assert shortest_length(start) == (2 + 1) ** 2 - 1
    # Red slides right into the empty slot.
    nxt = start.apply(start.legal_moves()[0])
    assert nxt.encode() != start.encode()


def test_checkers_forbid_backward_moves() -> None:
    start = CheckerJumping.start(1)
    # n=1: R _ B. Red can slide right; blue can slide left. No backward.
    encodings_after = {start.apply(m).encode() for m in start.legal_moves()}
    # Neither piece can move away from the empty slot.
    assert len(encodings_after) == 2


def test_river_n1_solves_in_one_crossing() -> None:
    start = RiverCrossing.start(n_pairs=1, boat_capacity=2)
    assert start.is_solved() is False
    length = shortest_length(start)
    assert length == 1
    assert start.goal().is_solved() is True


def test_river_forbids_unescorted_actor_with_foreign_agent() -> None:
    start = RiverCrossing.start(n_pairs=2, boat_capacity=2)
    # Sending actor-1 alone with agent-2 is illegal (jealous-husbands constraint).
    illegal = frozenset({("A", 1), ("G", 2)})
    with pytest.raises(ValueError, match="illegal"):
        start.apply(illegal)


def test_blocks_start_is_not_the_goal_and_optimal_is_positive() -> None:
    start = BlocksWorld.start(3)
    assert start.is_solved() is False
    assert start.goal().is_solved() is True
    assert shortest_length(start) > 0
    assert start.legal_moves()


def test_puzzle_registry_covers_the_four_apple_environments() -> None:
    from bayesmas.puzzles import PUZZLES

    assert set(PUZZLES) == {"hanoi", "checkers", "river", "blocks"}
