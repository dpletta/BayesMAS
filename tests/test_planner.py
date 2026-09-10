"""Noisy multi-agent planners on the puzzle simulators."""

from __future__ import annotations

from bayesmas.puzzles.planner import run_puzzle_trial
from bayesmas.puzzles.solve import shortest_length
from bayesmas.types import ProtocolName


def test_zero_noise_oracle_solves_hanoi_optimally() -> None:
    result = run_puzzle_trial(
        puzzle="hanoi",
        n=3,
        protocol=ProtocolName.BAYESMAS,
        n_agents=3,
        noise=0.0,
        seed=0,
    )
    assert result.solved is True
    assert result.steps == 7
    assert result.optimal_steps == 7


def test_byzantine_filter_improves_solve_rate_on_noisy_hanoi() -> None:
    kwargs = dict(
        puzzle="hanoi",
        n=3,
        n_agents=5,
        n_byzantine=1,
        noise=1.5,
        max_steps=20,
    )
    raw = [run_puzzle_trial(protocol=ProtocolName.BAYESMAS, seed=s, **kwargs) for s in range(12)]
    filtered = [
        run_puzzle_trial(protocol=ProtocolName.BAYESMAS_F, seed=s, **kwargs) for s in range(12)
    ]
    assert sum(r.solved for r in filtered) > sum(r.solved for r in raw)


def test_optimal_steps_match_the_solver() -> None:
    from bayesmas.puzzles import TowerOfHanoi

    result = run_puzzle_trial(
        puzzle="hanoi",
        n=2,
        protocol=ProtocolName.FIXED,
        n_agents=2,
        noise=0.0,
        seed=0,
        max_steps=10,
    )
    assert result.optimal_steps == shortest_length(TowerOfHanoi.start(2))
    assert result.solved is True
