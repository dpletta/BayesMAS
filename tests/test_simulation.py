"""Monte Carlo experiment: qualitative claims from the Notion spec."""

from __future__ import annotations

from bayesmas.protocols import ProtocolName
from bayesmas.simulation import ComplexitySetting, run_experiment


def test_complexity_settings_match_documented_regimes() -> None:
    low = ComplexitySetting.low()
    medium = ComplexitySetting.medium()
    high = ComplexitySetting.high()
    assert low.n_byzantine == 0
    assert medium.n_byzantine == 0
    assert high.n_byzantine == 1
    assert high.byzantine_target == 75.0
    assert high.bias_sd == 12.0
    assert low.obs_sd < medium.obs_sd < high.obs_sd


def test_high_complexity_filter_beats_unfiltered_and_beats_fixed_step_count() -> None:
    summary = run_experiment(
        settings=[ComplexitySetting.high()],
        protocols=[ProtocolName.FIXED, ProtocolName.BAYESMAS, ProtocolName.BAYESMAS_F],
        trials=40,
        seed=0,
        max_steps=15,
    )
    high = summary.by_setting["high"]
    assert high[ProtocolName.BAYESMAS_F].mean_error < high[ProtocolName.BAYESMAS].mean_error
    assert high[ProtocolName.BAYESMAS_F].mean_steps < high[ProtocolName.FIXED].mean_steps
    assert high[ProtocolName.FIXED].mean_steps == 15.0


def test_low_complexity_bayesmas_stops_well_before_the_cap() -> None:
    summary = run_experiment(
        settings=[ComplexitySetting.low()],
        protocols=[ProtocolName.FIXED, ProtocolName.DYLAN, ProtocolName.BAYESMAS],
        trials=30,
        seed=1,
        max_steps=15,
    )
    low = summary.by_setting["low"]
    assert low[ProtocolName.BAYESMAS].mean_steps < 8.0
    assert low[ProtocolName.DYLAN].mean_steps < 8.0
    assert low[ProtocolName.FIXED].mean_steps == 15.0
