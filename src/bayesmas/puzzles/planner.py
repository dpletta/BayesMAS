"""Noisy multi-agent planners over the puzzle simulators.

Each honest agent scores legal moves as ``−remaining_length + N(0, noise)``.
A Byzantine agent (when present) assigns a huge score to the *worst* move
and claims vanishing variance. BayesMAS-F drops that agent with SAC on the
scalar remaining-length reports before pooling move scores.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from bayesmas.filter import sac_retain_mask
from bayesmas.pooling import consensus_mean
from bayesmas.puzzles.base import Move, Puzzle
from bayesmas.puzzles.blocks import BlocksWorld
from bayesmas.puzzles.checkers import CheckerJumping
from bayesmas.puzzles.hanoi import TowerOfHanoi
from bayesmas.puzzles.river import RiverCrossing
from bayesmas.puzzles.solve import bfs_distance, shortest_length
from bayesmas.types import ProtocolName

Factory = Callable[[int], Puzzle]

# Remaining-length assigned to a state from which the goal is unreachable.
# Large enough to dominate any real distance in these small puzzles, so a move
# that deadlocks the puzzle is scored as the worst possible option.
UNSOLVABLE_REMAIN = 1e6


def _remaining_length(state: Puzzle) -> float:
    """Distance to the goal, or ``UNSOLVABLE_REMAIN`` for deadlocked states."""
    dist = bfs_distance(state)
    return float(dist) if dist is not None else UNSOLVABLE_REMAIN


PUZZLE_FACTORIES: dict[str, Factory] = {
    "hanoi": TowerOfHanoi.start,
    "checkers": CheckerJumping.start,
    "river": lambda n: RiverCrossing.start(n_pairs=n, boat_capacity=2),
    "blocks": BlocksWorld.start,
}


@dataclass(frozen=True, slots=True)
class PuzzleTrialResult:
    puzzle: str
    complexity: int
    protocol: str
    solved: bool
    steps: int
    optimal_steps: int


def _choose_move(
    moves: tuple[Move, ...],
    scores: np.ndarray,
    sigma2: np.ndarray,
    y_scalar: np.ndarray,
    protocol: ProtocolName,
    f_discard: int,
    sigma_eval: float,
) -> Move:
    n_agents, n_moves = scores.shape
    if protocol is ProtocolName.DYLAN:
        votes = np.argmax(scores, axis=1)
        counts = np.bincount(votes, minlength=n_moves)
        return moves[int(np.argmax(counts))]

    if protocol is ProtocolName.BAYESMAS_F:
        # Receiver = agent with median remaining-length report. Byzantine
        # reports are large negative outliers; with an honest majority the
        # median lands on an honest agent.
        order = np.argsort(y_scalar)
        recv = int(order[n_agents // 2])
        retain = sac_retain_mask(y_scalar, recv, f_discard=f_discard, sigma_eval=sigma_eval)
        if not np.any(retain):
            retain = np.ones(n_agents, dtype=bool)
    else:
        retain = np.ones(n_agents, dtype=bool)

    pooled = np.array(
        [_claimed_mean(scores[retain, m], sigma2[retain]) for m in range(n_moves)],
        dtype=np.float64,
    )
    return moves[int(np.argmax(pooled))]


def _claimed_mean(y: np.ndarray, sigma2: np.ndarray) -> float:
    return consensus_mean(y, sigma2, tau2=0.0)


def run_puzzle_trial(
    *,
    puzzle: str,
    n: int,
    protocol: ProtocolName,
    n_agents: int = 4,
    n_byzantine: int = 0,
    noise: float = 0.0,
    seed: int = 0,
    max_steps: int | None = None,
    sigma_eval: float = 2.0,
) -> PuzzleTrialResult:
    """One seeded planning trial. ``noise=0`` is an exact oracle ensemble."""
    if puzzle not in PUZZLE_FACTORIES:
        raise ValueError(f"unknown puzzle {puzzle!r}")
    if n_byzantine >= n_agents:
        raise ValueError("need at least one honest agent")

    rng = np.random.default_rng(seed)
    state = PUZZLE_FACTORIES[puzzle](n)
    optimal = shortest_length(state)
    cap = max_steps if max_steps is not None else optimal * 3 + 8
    n_honest = n_agents - n_byzantine
    steps = 0

    for steps in range(1, cap + 1):
        if state.is_solved():
            return PuzzleTrialResult(
                puzzle=puzzle,
                complexity=n,
                protocol=protocol.value,
                solved=True,
                steps=steps - 1,
                optimal_steps=optimal,
            )
        moves = state.legal_moves()
        if not moves:
            break
        remain = np.array(
            [_remaining_length(state.apply(move)) for move in moves], dtype=np.float64
        )
        quality = -remain
        scores = np.zeros((n_agents, len(moves)), dtype=np.float64)
        sigma2 = np.empty(n_agents, dtype=np.float64)
        y_scalar = np.empty(n_agents, dtype=np.float64)
        true_remain = _remaining_length(state)

        for i in range(n_honest):
            jitter = 0.0 if noise == 0.0 else rng.normal(0.0, noise, size=len(moves))
            scores[i] = quality + jitter
            sigma2[i] = 1e-6 if noise == 0.0 else noise**2
            y_scalar[i] = true_remain + (0.0 if noise == 0.0 else float(rng.normal(0.0, noise)))

        worst = int(np.argmin(quality))
        for j in range(n_honest, n_agents):
            scores[j] = -100.0
            scores[j, worst] = 100.0
            sigma2[j] = 1e-6
            y_scalar[j] = -1.0

        chosen = _choose_move(
            moves,
            scores,
            sigma2,
            y_scalar,
            protocol,
            f_discard=n_byzantine,
            sigma_eval=sigma_eval,
        )
        state = state.apply(chosen)

    return PuzzleTrialResult(
        puzzle=puzzle,
        complexity=n,
        protocol=protocol.value,
        solved=state.is_solved(),
        steps=steps,
        optimal_steps=optimal,
    )


def run_puzzle_sweep(
    *,
    puzzles: tuple[str, ...] = ("hanoi", "checkers", "river", "blocks"),
    sizes: tuple[int, ...] = (2, 3),
    protocols: tuple[ProtocolName, ...] = (
        ProtocolName.FIXED,
        ProtocolName.DYLAN,
        ProtocolName.BAYESMAS,
        ProtocolName.BAYESMAS_F,
    ),
    trials: int = 10,
    seed: int = 0,
    n_agents: int = 5,
    n_byzantine: int = 1,
    noise: float = 1.0,
) -> list[dict[str, float | str | int]]:
    """Small complexity sweep for the CLI. Returns one row per cell mean."""
    rows: list[dict[str, float | str | int]] = []
    for puzzle in puzzles:
        for n in sizes:
            for protocol in protocols:
                results = [
                    run_puzzle_trial(
                        puzzle=puzzle,
                        n=n,
                        protocol=protocol,
                        n_agents=n_agents,
                        n_byzantine=n_byzantine if protocol is not ProtocolName.DYLAN else 0,
                        noise=noise,
                        seed=seed + t,
                    )
                    for t in range(trials)
                ]
                rows.append(
                    {
                        "puzzle": puzzle,
                        "n": n,
                        "protocol": protocol.value,
                        "solve_rate": float(np.mean([r.solved for r in results])),
                        "mean_steps": float(np.mean([r.steps for r in results])),
                        "optimal_steps": results[0].optimal_steps,
                        "trials": trials,
                    }
                )
    return rows
