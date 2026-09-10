"""BayesMAS: empirical Bayes multi-agent collaboration."""

from __future__ import annotations

from bayesmas.filter import gaussian_kernel_score, sac_retain_mask
from bayesmas.pooling import (
    consensus_mean,
    estimate_tau_squared,
    partial_pool,
    posterior_variance_of_mean,
    sufficient_stats,
)
from bayesmas.protocols import ProtocolResult, dylan_agreed, run_protocol
from bayesmas.simulation import (
    AgentPopulation,
    ComplexitySetting,
    ExperimentSummary,
    run_experiment,
)
from bayesmas.stopping import should_stop
from bayesmas.types import ProtocolName, StabilityConfig, SufficientStats

__version__ = "0.1.0"

__all__ = [
    "AgentPopulation",
    "ComplexitySetting",
    "ExperimentSummary",
    "ProtocolName",
    "ProtocolResult",
    "StabilityConfig",
    "SufficientStats",
    "__version__",
    "consensus_mean",
    "dylan_agreed",
    "estimate_tau_squared",
    "gaussian_kernel_score",
    "partial_pool",
    "posterior_variance_of_mean",
    "run_experiment",
    "run_protocol",
    "sac_retain_mask",
    "should_stop",
    "sufficient_stats",
]
