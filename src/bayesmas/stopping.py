"""Asymptotic posterior-stability stopping rule."""

from __future__ import annotations

from collections.abc import Sequence

from bayesmas.types import StabilityConfig

__all__ = ["StabilityConfig", "should_stop"]


def should_stop(var_history: Sequence[float], config: StabilityConfig) -> bool:
    """Return True once the posterior variance of μ has stabilized.

    Two complementary triggers, both gated on ``min_steps``:

    * absolute floor — Var(μ) has crossed ``abs_threshold`` (easy tasks
      collapse quickly and should not overthink);
    * diminishing returns — the last step changed Var(μ) by at most
      ``delta_threshold`` (hard tasks halt when extra rounds stop informing).
    """
    if len(var_history) < config.min_steps:
        return False
    current = float(var_history[-1])
    if current <= config.abs_threshold:
        return True
    if len(var_history) >= 2:
        delta = abs(current - float(var_history[-2]))
        if delta <= config.delta_threshold:
            return True
    return False
