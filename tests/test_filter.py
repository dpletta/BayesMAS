"""Receiver-side Self-Anchored Consensus (SAC) Byzantine filter."""

from __future__ import annotations

import numpy as np
import pytest

from bayesmas.filter import gaussian_kernel_score, sac_retain_mask


def test_self_score_is_one() -> None:
    assert gaussian_kernel_score(3.0, 3.0, sigma_eval=1.0) == pytest.approx(1.0)


def test_score_falls_as_predictions_diverge() -> None:
    near = gaussian_kernel_score(0.0, 1.0, sigma_eval=1.0)
    far = gaussian_kernel_score(0.0, 8.0, sigma_eval=1.0)
    assert 0.0 < far < near < 1.0
    assert near == pytest.approx(np.exp(-0.5))


def test_sac_always_keeps_the_receiver() -> None:
    reports = np.array([0.0, 0.2, 75.0])
    mask = sac_retain_mask(reports, self_index=0, f_discard=1, sigma_eval=1.0)
    assert mask[0] is True or mask[0] == True  # noqa: E712 — bool or np.bool_
    assert bool(mask[0])


def test_sac_drops_the_bottom_f_outliers_from_the_receiver() -> None:
    reports = np.array([0.0, 0.1, 75.0])
    mask = sac_retain_mask(reports, self_index=0, f_discard=1, sigma_eval=1.0)
    assert np.array_equal(mask, np.array([True, True, False]))


def test_sac_with_f_zero_keeps_everyone() -> None:
    reports = np.array([0.0, 75.0])
    mask = sac_retain_mask(reports, self_index=0, f_discard=0, sigma_eval=1.0)
    assert np.all(mask)
