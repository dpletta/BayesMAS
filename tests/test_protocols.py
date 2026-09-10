"""Four deliberation protocols on scalar opinion reports."""

from __future__ import annotations

import numpy as np
import pytest

from bayesmas.protocols import ProtocolName, dylan_agreed, run_protocol
from bayesmas.simulation import AgentPopulation, ComplexitySetting
from bayesmas.stopping import StabilityConfig


def test_dylan_agrees_when_more_than_two_thirds_cluster() -> None:
    assert dylan_agreed(np.array([0.0, 0.01, 0.02, 9.0]), tol=0.1) is True
    assert dylan_agreed(np.array([0.0, 5.0, 10.0]), tol=0.1) is False


def test_fixed_always_runs_the_step_cap() -> None:
    pop = AgentPopulation.from_setting(
        ComplexitySetting.low(),
        rng=np.random.default_rng(1),
        n_agents=5,
    )
    result = run_protocol(
        ProtocolName.FIXED,
        pop,
        max_steps=15,
        stability=StabilityConfig(min_steps=2),
    )
    assert result.steps == 15
    assert result.stopped_early is False
    assert result.name == "fixed"


def test_dylan_stops_early_when_agents_already_cluster() -> None:
    pop = AgentPopulation(
        y=np.array([0.0, 0.01, 0.02, 0.01, 0.0]),
        sigma2=np.full(5, 0.25),
        honest=np.ones(5, dtype=bool),
        truth=0.0,
    )
    result = run_protocol(
        ProtocolName.DYLAN,
        pop,
        max_steps=15,
        dylan_tol=0.1,
    )
    assert result.stopped_early is True
    assert result.steps < 15
    assert result.name == "dylan"


def test_bayesmas_stops_when_posterior_is_stable() -> None:
    pop = AgentPopulation(
        y=np.array([0.0, 0.05, -0.04, 0.02]),
        sigma2=np.full(4, 0.05),
        honest=np.ones(4, dtype=bool),
        truth=0.0,
    )
    result = run_protocol(
        ProtocolName.BAYESMAS,
        pop,
        max_steps=15,
        stability=StabilityConfig(abs_threshold=0.05, delta_threshold=0.01, min_steps=2),
    )
    assert result.stopped_early is True
    assert 2 <= result.steps < 15
    assert result.name == "bayesmas"
    assert result.final_mu == pytest.approx(0.0, abs=0.1)


def test_bayesmas_f_drops_byzantine_pull_toward_the_attack_target() -> None:
    # Four honest agents near 0; one Byzantine locked on 75 with fake certainty.
    pop = AgentPopulation(
        y=np.array([0.0, 0.3, -0.2, 0.1, 75.0]),
        sigma2=np.array([1.0, 1.0, 1.0, 1.0, 1e-6]),
        honest=np.array([True, True, True, True, False]),
        truth=0.0,
    )
    raw = run_protocol(ProtocolName.BAYESMAS, pop, max_steps=8, f_discard=1)
    filtered = run_protocol(ProtocolName.BAYESMAS_F, pop, max_steps=8, f_discard=1)
    assert filtered.honest_error < raw.honest_error
    assert filtered.final_mu < 20.0
    assert raw.final_mu > filtered.final_mu
    assert filtered.name == "bayesmas-f"
