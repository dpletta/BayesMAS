"""Apple-style controllable puzzle environments and noisy planners."""

from __future__ import annotations

from bayesmas.puzzles.blocks import BlocksWorld
from bayesmas.puzzles.checkers import CheckerJumping
from bayesmas.puzzles.hanoi import TowerOfHanoi
from bayesmas.puzzles.planner import (
    PUZZLE_FACTORIES,
    PuzzleTrialResult,
    run_puzzle_sweep,
    run_puzzle_trial,
)
from bayesmas.puzzles.river import RiverCrossing

PUZZLES = PUZZLE_FACTORIES

__all__ = [
    "BlocksWorld",
    "CheckerJumping",
    "PUZZLES",
    "PuzzleTrialResult",
    "RiverCrossing",
    "TowerOfHanoi",
    "run_puzzle_sweep",
    "run_puzzle_trial",
]
