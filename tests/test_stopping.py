"""Asymptotic posterior-stability stopping rule."""

from __future__ import annotations

import numpy as np
import pytest

from bayesmas.pooling import posterior_variance_of_mean
from bayesmas.stopping import StabilityConfig, should_stop


def test_posterior_variance_shrinks_when_more_precise_agents_arrive() -> None:
    one = posterior_variance_of_mean(np.array([1.0]), tau2=1.0)
    two = posterior_variance_of_mean(np.array([1.0, 1.0]), tau2=1.0)
    # Var(μ) = 1 / Σ 1/(σ²+τ²); one agent → 1/0.5 = 2, two agents → 1.
    assert one == pytest.approx(2.0)
    assert two == pytest.approx(1.0)
    assert two < one


def test_does_not_stop_before_minimum_steps() -> None:
    cfg = StabilityConfig(abs_threshold=1e-9, delta_threshold=1e-9, min_steps=3)
    assert should_stop([0.0], cfg) is False
    assert should_stop([0.1, 0.0], cfg) is False


def test_stops_when_posterior_variance_crosses_absolute_floor() -> None:
    cfg = StabilityConfig(abs_threshold=0.05, delta_threshold=1e-12, min_steps=2)
    assert should_stop([0.4, 0.04], cfg) is True


def test_stops_when_variance_change_is_below_delta() -> None:
    cfg = StabilityConfig(abs_threshold=1e-12, delta_threshold=0.01, min_steps=2)
    assert should_stop([0.50, 0.495], cfg) is True


def test_continues_while_variance_is_still_moving() -> None:
    cfg = StabilityConfig(abs_threshold=1e-6, delta_threshold=0.01, min_steps=2)
    assert should_stop([0.50, 0.30], cfg) is False
