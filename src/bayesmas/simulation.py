"""Opinion-dynamics population and complexity regimes.

Honest agents hold a one-shot noisy observation of ``truth``. A Byzantine
agent (high-complexity regime only) reports ``byzantine_target`` with a
vanishing claimed variance — the dishonest-confidence attack.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bayesmas.protocols import ProtocolResult, run_protocol
from bayesmas.stopping import StabilityConfig
from bayesmas.types import ProtocolName, ReportBundle


@dataclass(frozen=True, slots=True)
class ComplexitySetting:
    """Named noise / adversary regime used by the Monte Carlo experiment."""

    name: str
    obs_sd: float
    bias_sd: float
    n_byzantine: int
    byzantine_target: float = 75.0
    n_agents: int = 5
    truth: float = 0.0
    sigma_eval: float = 3.0

    @classmethod
    def low(cls) -> ComplexitySetting:
        return cls(name="low", obs_sd=0.35, bias_sd=0.15, n_byzantine=0)

    @classmethod
    def medium(cls) -> ComplexitySetting:
        return cls(name="medium", obs_sd=1.25, bias_sd=2.0, n_byzantine=0)

    @classmethod
    def high(cls) -> ComplexitySetting:
        return cls(
            name="high",
            obs_sd=4.0,
            bias_sd=12.0,
            n_byzantine=1,
            byzantine_target=75.0,
        )

    @classmethod
    def all_regimes(cls) -> tuple[ComplexitySetting, ...]:
        return (cls.low(), cls.medium(), cls.high())


@dataclass
class AgentPopulation(ReportBundle):
    """Current scalar reports for one trial."""

    @classmethod
    def from_setting(
        cls,
        setting: ComplexitySetting,
        rng: np.random.Generator,
        n_agents: int | None = None,
    ) -> AgentPopulation:
        n = setting.n_agents if n_agents is None else n_agents
        n_byz = min(setting.n_byzantine, n)
        n_honest = n - n_byz
        honest = np.zeros(n, dtype=bool)
        honest[:n_honest] = True
        y = np.empty(n, dtype=np.float64)
        sigma2 = np.empty(n, dtype=np.float64)
        y[:n_honest] = (
            setting.truth
            + rng.normal(0.0, setting.bias_sd, size=n_honest)
            + rng.normal(0.0, setting.obs_sd, size=n_honest)
        )
        sigma2[:n_honest] = setting.obs_sd**2
        if n_byz:
            y[n_honest:] = setting.byzantine_target
            sigma2[n_honest:] = 1e-6
        return cls(y=y, sigma2=sigma2, honest=honest, truth=setting.truth)

    def copy(self) -> AgentPopulation:
        return AgentPopulation(
            y=self.y.copy(),
            sigma2=self.sigma2.copy(),
            honest=self.honest.copy(),
            truth=self.truth,
        )


@dataclass(frozen=True, slots=True)
class TrialMetrics:
    mean_error: float
    mean_steps: float
    n_trials: int


@dataclass
class ExperimentSummary:
    by_setting: dict[str, dict[ProtocolName, TrialMetrics]] = field(default_factory=dict)

    def rows(self) -> list[dict[str, float | str | int]]:
        out: list[dict[str, float | str | int]] = []
        for setting_name, protocols in self.by_setting.items():
            for protocol, metrics in protocols.items():
                out.append(
                    {
                        "setting": setting_name,
                        "protocol": protocol.value,
                        "mean_error": metrics.mean_error,
                        "mean_steps": metrics.mean_steps,
                        "n_trials": metrics.n_trials,
                    }
                )
        return out


def run_experiment(
    settings: list[ComplexitySetting] | None = None,
    protocols: list[ProtocolName] | None = None,
    *,
    trials: int = 100,
    seed: int = 0,
    max_steps: int = 15,
    stability: StabilityConfig | None = None,
) -> ExperimentSummary:
    """Seeded Monte Carlo over settings × protocols."""
    if settings is None:
        chosen_settings = list(ComplexitySetting.all_regimes())
    else:
        chosen_settings = list(settings)
    chosen_protocols = (
        list(protocols)
        if protocols is not None
        else [
            ProtocolName.FIXED,
            ProtocolName.DYLAN,
            ProtocolName.BAYESMAS,
            ProtocolName.BAYESMAS_F,
        ]
    )
    rng = np.random.default_rng(seed)
    summary = ExperimentSummary()
    for setting in chosen_settings:
        bucket: dict[ProtocolName, TrialMetrics] = {}
        for protocol in chosen_protocols:
            errors: list[float] = []
            steps: list[int] = []
            for _ in range(trials):
                pop = AgentPopulation.from_setting(setting, rng)
                result: ProtocolResult = run_protocol(
                    protocol,
                    pop,
                    max_steps=max_steps,
                    stability=stability,
                    f_discard=setting.n_byzantine,
                    sigma_eval=setting.sigma_eval,
                )
                errors.append(result.honest_error)
                steps.append(result.steps)
            bucket[protocol] = TrialMetrics(
                mean_error=float(np.mean(errors)),
                mean_steps=float(np.mean(steps)),
                n_trials=trials,
            )
        summary.by_setting[setting.name] = bucket
    return summary
