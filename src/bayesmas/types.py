"""Shared dataclasses for beliefs and sufficient statistics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


@dataclass(frozen=True, slots=True)
class SufficientStats:
    """Compact group state passed between agents.

    ``mu`` is the precision-weighted consensus. ``tau2`` is between-agent
    disagreement. ``sigma2_bar`` is mean local observation variance.
    ``post_var_mu`` is the posterior variance of the consensus mean.
    """

    mu: float
    tau2: float
    sigma2_bar: float
    post_var_mu: float


@dataclass(frozen=True, slots=True)
class StabilityConfig:
    """Stop when posterior variance is small, or when it stops moving."""

    abs_threshold: float = 1e-3
    delta_threshold: float = 1e-4
    min_steps: int = 2


class ProtocolName(StrEnum):
    FIXED = "fixed"
    DYLAN = "dylan"
    BAYESMAS = "bayesmas"
    BAYESMAS_F = "bayesmas-f"


@dataclass
class ReportBundle:
    """Mutable scalar reports for one deliberation trial."""

    y: np.ndarray
    sigma2: np.ndarray
    honest: np.ndarray
    truth: float

    def copy(self) -> ReportBundle:
        return ReportBundle(
            y=self.y.copy(),
            sigma2=self.sigma2.copy(),
            honest=self.honest.copy(),
            truth=self.truth,
        )
