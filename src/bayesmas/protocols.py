"""Deliberation protocols: Fixed, DyLAN, BayesMAS, BayesMAS-F.

All four start from the same one-shot reports. They differ in how they
form the group mean and when they halt.

* **Fixed** — claimed-precision partial pooling, always ``max_steps``.
* **DyLAN** — DeGroot pull toward the unweighted mean; halt when more than
  two-thirds of reports cluster (fractional consensus).
* **BayesMAS** — claimed-precision consensus + partial pooling; halt on
  posterior-stability of μ. Unfiltered, so a Byzantine agent that claims
  tiny σ² can dominate μ (dishonest-confidence attack).
* **BayesMAS-F** — same update after a receiver-side SAC trim of the
  bottom-F outliers.

The τ used *inside* partial pooling is an annealing group-trust term
(``pool_tau2`` decayed each step). Residual MoM τ is still recorded on
``SufficientStats`` as a disagreement diagnostic. This is what produces
the herd-effect collapse in the unfiltered high-complexity regime: as
trust in μ grows, honest agents are pulled toward a contaminated mean.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bayesmas.filter import sac_retain_mask
from bayesmas.pooling import consensus_mean, estimate_tau_squared, partial_pool, sufficient_stats
from bayesmas.stopping import StabilityConfig, should_stop
from bayesmas.types import ProtocolName, ReportBundle, SufficientStats

# Re-export so callers can write `from bayesmas.protocols import ProtocolName`.
__all__ = [
    "ProtocolName",
    "ProtocolResult",
    "dylan_agreed",
    "run_protocol",
]


@dataclass
class ProtocolResult:
    name: str
    steps: int
    stopped_early: bool
    final_mu: float
    final_tau2: float
    honest_error: float
    trajectory: list[SufficientStats] = field(default_factory=list)


def dylan_agreed(values: np.ndarray, tol: float) -> bool:
    """True when the largest |·|-ball of radius ``tol`` holds more than 2/3 of agents."""
    reports = np.asarray(values, dtype=np.float64).reshape(-1)
    n = reports.size
    if n == 0:
        return False
    threshold = (2.0 / 3.0) * n
    best = 0
    for value in reports:
        size = int(np.sum(np.abs(reports - value) <= tol))
        if size > best:
            best = size
    return best > threshold


def _honest_error(bundle: ReportBundle) -> float:
    honest_y = bundle.y[bundle.honest]
    if honest_y.size == 0:
        return float("nan")
    return float(np.mean(np.abs(honest_y - bundle.truth)))


def _active_stats(bundle: ReportBundle, mask: np.ndarray) -> SufficientStats:
    return sufficient_stats(bundle.y[mask], bundle.sigma2[mask])


def _claimed_mean(y: np.ndarray, sigma2: np.ndarray) -> float:
    return consensus_mean(y, sigma2, tau2=0.0)


def _bayes_update(
    bundle: ReportBundle,
    *,
    filtered: bool,
    f_discard: int,
    sigma_eval: float,
    pool_tau2: float,
) -> None:
    """In-place honest-agent update. Byzantine reports stay pinned."""
    n = bundle.y.size
    new_y = bundle.y.copy()
    for i in range(n):
        if not bool(bundle.honest[i]):
            continue
        if filtered:
            retain = sac_retain_mask(
                bundle.y, self_index=i, f_discard=f_discard, sigma_eval=sigma_eval
            )
        else:
            retain = np.ones(n, dtype=bool)
        mu = _claimed_mean(bundle.y[retain], bundle.sigma2[retain])
        new_y[i] = partial_pool(
            float(bundle.y[i]),
            float(bundle.sigma2[i]),
            mu,
            pool_tau2,
        )
    bundle.y = new_y


def _degroot_update(bundle: ReportBundle) -> None:
    mu = float(np.mean(bundle.y))
    new_y = bundle.y.copy()
    honest = bundle.honest
    new_y[honest] = 0.5 * bundle.y[honest] + 0.5 * mu
    bundle.y = new_y


def run_protocol(
    name: ProtocolName,
    population: ReportBundle,
    *,
    max_steps: int = 15,
    stability: StabilityConfig | None = None,
    dylan_tol: float = 0.25,
    f_discard: int = 1,
    sigma_eval: float = 3.0,
    pool_tau2: float = 1.0,
    pool_tau_decay: float = 0.75,
) -> ProtocolResult:
    """Run one protocol on a copy of ``population`` and return metrics."""
    bundle = population.copy()
    cfg = stability if stability is not None else StabilityConfig()
    history: list[float] = []
    trajectory: list[SufficientStats] = []
    tau_k = pool_tau2
    stopped_early = False
    steps = 0

    for step in range(1, max_steps + 1):
        steps = step
        if name is ProtocolName.DYLAN and dylan_agreed(bundle.y, dylan_tol):
            stopped_early = True
            break

        if name is ProtocolName.DYLAN:
            _degroot_update(bundle)
        elif name is ProtocolName.BAYESMAS_F:
            _bayes_update(
                bundle,
                filtered=True,
                f_discard=f_discard,
                sigma_eval=sigma_eval,
                pool_tau2=tau_k,
            )
        else:
            # FIXED and unfiltered BayesMAS share the same update; they differ
            # only in the stopping rule.
            _bayes_update(
                bundle,
                filtered=False,
                f_discard=f_discard,
                sigma_eval=sigma_eval,
                pool_tau2=tau_k,
            )

        stats = _active_stats(bundle, bundle.honest)
        trajectory.append(stats)
        history.append(stats.post_var_mu)
        tau_k = max(1e-6, tau_k * pool_tau_decay)

        if name is ProtocolName.FIXED:
            continue
        if name is ProtocolName.DYLAN and dylan_agreed(bundle.y, dylan_tol):
            stopped_early = True
            break
        if name in (ProtocolName.BAYESMAS, ProtocolName.BAYESMAS_F) and should_stop(history, cfg):
            stopped_early = True
            break

    if trajectory:
        final_mu = trajectory[-1].mu
        final_tau2 = trajectory[-1].tau2
    else:
        final_mu = float(np.mean(bundle.y[bundle.honest])) if np.any(bundle.honest) else 0.0
        final_tau2 = estimate_tau_squared(bundle.y, bundle.sigma2)

    return ProtocolResult(
        name=name.value,
        steps=steps,
        stopped_early=stopped_early,
        final_mu=final_mu,
        final_tau2=final_tau2,
        honest_error=_honest_error(bundle),
        trajectory=trajectory,
    )
