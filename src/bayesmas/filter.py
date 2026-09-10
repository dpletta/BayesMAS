"""Receiver-side Self-Anchored Consensus (SAC) filter.

Neighbors are graded against the receiver's own report with a Gaussian kernel.
The bottom-F scores are discarded (W-MSR style). The receiver always keeps
itself. This is immune to dishonest self-reported confidence: a Byzantine
agent that claims tiny σ² is still dropped if its *prediction* is an outlier
relative to the receiver.
"""

from __future__ import annotations

import numpy as np


def gaussian_kernel_score(r_self: float, r_other: float, sigma_eval: float) -> float:
    """s = exp( − (r_self − r_other)² / (2 σ_eval²) )."""
    if sigma_eval <= 0.0:
        raise ValueError("sigma_eval must be strictly positive")
    delta = float(r_self) - float(r_other)
    return float(np.exp(-0.5 * (delta**2) / (sigma_eval**2)))


def sac_retain_mask(
    reports: np.ndarray,
    self_index: int,
    f_discard: int,
    sigma_eval: float,
) -> np.ndarray:
    """Boolean mask of reports agent ``self_index`` should trust.

    ``f_discard`` is the assumed number of Byzantine neighbors. Self is never
    discarded. If ``f_discard`` exceeds the number of neighbors it is clipped.
    """
    values = np.asarray(reports, dtype=np.float64).reshape(-1)
    n = values.size
    if n == 0:
        raise ValueError("reports must be non-empty")
    if not 0 <= self_index < n:
        raise ValueError("self_index out of range")
    if f_discard < 0:
        raise ValueError("f_discard must be non-negative")

    keep = np.ones(n, dtype=bool)
    neighbor_idx = [i for i in range(n) if i != self_index]
    if not neighbor_idx or f_discard == 0:
        return keep

    r_self = float(values[self_index])
    scored = [
        (gaussian_kernel_score(r_self, float(values[i]), sigma_eval), i) for i in neighbor_idx
    ]
    scored.sort(key=lambda pair: pair[0])  # lowest agreement first
    drop_n = min(f_discard, len(scored))
    for _, idx in scored[:drop_n]:
        keep[idx] = False
    return keep
