"""Empirical Bayes consensus and partial-pooling identities."""

from __future__ import annotations

import numpy as np
import pytest

from bayesmas.pooling import consensus_mean, estimate_tau_squared, partial_pool, sufficient_stats


def test_consensus_mean_is_precision_weighted_average() -> None:
    y = np.array([0.0, 10.0])
    sigma2 = np.array([1.0, 4.0])
    tau2 = 0.0
    # weights 1/(σ²+τ²) = [1, 0.25]; μ = (0 + 2.5) / 1.25 = 2.0
    assert consensus_mean(y, sigma2, tau2) == pytest.approx(2.0)


def test_partial_pool_is_precision_weighted_mix_of_local_and_group() -> None:
    theta = partial_pool(y=4.0, sigma2=1.0, mu=0.0, tau2=1.0)
    # (y/σ² + μ/τ²) / (1/σ² + 1/τ²) = (4 + 0) / 2 = 2
    assert theta == pytest.approx(2.0)


def test_partial_pool_collapses_to_consensus_when_group_is_unified() -> None:
    assert partial_pool(y=9.0, sigma2=1.0, mu=3.0, tau2=0.0) == pytest.approx(3.0)


def test_partial_pool_keeps_local_anchor_when_group_is_chaotic() -> None:
    theta = partial_pool(y=9.0, sigma2=1.0, mu=0.0, tau2=1e12)
    assert theta == pytest.approx(9.0, rel=0.0, abs=1e-6)


def test_estimate_tau_squared_is_zero_when_agents_already_agree() -> None:
    y = np.array([1.0, 1.0, 1.0])
    sigma2 = np.array([0.25, 0.25, 0.25])
    assert estimate_tau_squared(y, sigma2) == pytest.approx(0.0)


def test_estimate_tau_squared_recovers_extra_between_agent_spread() -> None:
    y = np.array([0.0, 10.0])
    sigma2 = np.array([1.0, 1.0])
    tau2 = estimate_tau_squared(y, sigma2)
    # sample var of [0, 10] is 50; minus mean σ² = 1 → 49
    assert tau2 == pytest.approx(49.0)


def test_sufficient_stats_bundle_matches_components() -> None:
    y = np.array([0.0, 2.0, 4.0])
    sigma2 = np.array([1.0, 1.0, 1.0])
    stats = sufficient_stats(y, sigma2)
    assert stats.mu == pytest.approx(consensus_mean(y, sigma2, stats.tau2))
    assert stats.tau2 == pytest.approx(estimate_tau_squared(y, sigma2))
    assert stats.sigma2_bar == pytest.approx(1.0)
    assert stats.post_var_mu > 0.0
