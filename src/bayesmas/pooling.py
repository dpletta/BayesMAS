"""Hierarchical-normal empirical Bayes updates.

The group is modeled as

    θ_j | μ, τ  ~  N(μ, τ²)
    y_j | θ_j   ~  N(θ_j, σ_j²)

Agents never store raw message history. They reshape the running sufficient
statistics (μ̂, σ̄², τ²) and a partial-pooling pull of each local anchor y_j
toward the group mean.
"""

from __future__ import annotations

import numpy as np

from bayesmas.types import SufficientStats

EPS = 1e-12


def _as_1d(y: np.ndarray, sigma2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    s_arr = np.asarray(sigma2, dtype=np.float64).reshape(-1)
    if y_arr.shape != s_arr.shape:
        raise ValueError("y and sigma2 must have the same shape")
    if y_arr.size == 0:
        raise ValueError("y must be non-empty")
    if np.any(s_arr <= 0.0):
        raise ValueError("sigma2 must be strictly positive")
    return y_arr, s_arr


def consensus_mean(y: np.ndarray, sigma2: np.ndarray, tau2: float) -> float:
    """Precision-weighted consensus μ̂ = Σ w_j y_j / Σ w_j, w_j = 1/(σ_j²+τ²)."""
    y_arr, s_arr = _as_1d(y, sigma2)
    if tau2 < 0.0:
        raise ValueError("tau2 must be non-negative")
    weights = 1.0 / (s_arr + tau2)
    return float(np.sum(weights * y_arr) / np.sum(weights))


def partial_pool(y: float, sigma2: float, mu: float, tau2: float) -> float:
    """Shrink local report y toward μ: (y/σ² + μ/τ²) / (1/σ² + 1/τ²).

    τ² → 0 pulls the agent fully onto the consensus (unified group).
    τ² → ∞ leaves the agent on its hidden anchor y (chaotic group).
    """
    if sigma2 <= 0.0:
        raise ValueError("sigma2 must be strictly positive")
    if tau2 < 0.0:
        raise ValueError("tau2 must be non-negative")
    if tau2 <= EPS:
        return float(mu)
    prec_local = 1.0 / sigma2
    prec_group = 1.0 / tau2
    return float((prec_local * y + prec_group * mu) / (prec_local + prec_group))


def estimate_tau_squared(y: np.ndarray, sigma2: np.ndarray) -> float:
    """Method-of-moments between-agent variance, truncated at zero.

    For J ≥ 2: τ̂² = max(0, s²(y) − mean(σ²)) with the unbiased sample
    variance s². A single report cannot identify disagreement.
    """
    y_arr, s_arr = _as_1d(y, sigma2)
    if y_arr.size < 2:
        return 0.0
    extra = float(np.var(y_arr, ddof=1) - np.mean(s_arr))
    return max(0.0, extra)


def posterior_variance_of_mean(sigma2: np.ndarray, tau2: float) -> float:
    """Var(μ | y) ≈ 1 / Σ 1/(σ_j² + τ²) under the hierarchical normal."""
    s_arr = np.asarray(sigma2, dtype=np.float64).reshape(-1)
    if s_arr.size == 0:
        raise ValueError("sigma2 must be non-empty")
    if np.any(s_arr <= 0.0):
        raise ValueError("sigma2 must be strictly positive")
    if tau2 < 0.0:
        raise ValueError("tau2 must be non-negative")
    precision = np.sum(1.0 / (s_arr + tau2))
    return float(1.0 / precision)


def sufficient_stats(y: np.ndarray, sigma2: np.ndarray) -> SufficientStats:
    """Estimate (μ̂, τ̂², σ̄², posterior Var(μ)) from the current reports."""
    y_arr, s_arr = _as_1d(y, sigma2)
    tau2 = estimate_tau_squared(y_arr, s_arr)
    mu = consensus_mean(y_arr, s_arr, tau2)
    return SufficientStats(
        mu=mu,
        tau2=tau2,
        sigma2_bar=float(np.mean(s_arr)),
        post_var_mu=posterior_variance_of_mean(s_arr, tau2),
    )
