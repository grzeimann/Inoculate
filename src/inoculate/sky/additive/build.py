"""Builders for additive amplifier components (e.g., shared polynomials).

This module provides production-grade, typed helpers to estimate additive
polynomial trends per amplifier using the same robust linear least-squares
(IRLS with Huber/Tukey) used elsewhere in the workflow.

Functions here are designed to operate on already-computed stage artifacts
(BW_amp, BW_full, multiplicative scales) and respect the modeling wavelength
mask used across stages.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ...robust import robust_linear_least_squares


def build_amp_poly(
    bw_amp: np.ndarray,
    bw_full: np.ndarray,
    mult: np.ndarray,
    wave_mask: np.ndarray,
    order: int,
    *,
    loss: str = "huber",
    huber_delta: float = 1.0,
    tukey_c: float = 4.685,
) -> np.ndarray:
    """Estimate per-amplifier additive polynomial coefficients with robust LS.

    Constructs residual_1 on the modeling wavelength mask as::

        residual_1 = (bw_amp / mult) - bw_full

    and fits a wavelength-polynomial shared across exposures for each amplifier
    using robust IRLS (Huber or Tukey). The fit is performed to the exposure-
    averaged residual over the mask (simple mean), ensuring numerical stability
    and consistency with later stages.

    Args:
        bw_amp: Array with shape (n_amp, n_exp, n_wave) of biweight amplifier spectra.
        bw_full: Array with shape (n_exp, n_wave) of biweight full-shot spectra.
        mult: Array with shape (n_amp, n_exp) of multiplicative scales.
        wave_mask: Boolean array (n_wave,) specifying wavelengths to use for the fit.
        order: Polynomial order (inclusive). An order of ``p`` yields ``p+1`` coefficients.
        loss: Robust loss to use for IRLS ('huber' or 'tukey').
        huber_delta: Huber tuning constant.
        tukey_c: Tukey biweight tuning constant.

    Returns:
        Array of shape (n_amp, n_poly) with polynomial coefficients per amplifier,
        where ``n_poly = order + 1``.
    """
    if order < 0:
        raise ValueError("order must be >= 0")

    n_amp, n_exp, n_wave = bw_amp.shape
    if bw_full.shape != (n_exp, n_wave):  # pragma: no cover - sanity
        raise ValueError("bw_full shape mismatch")
    if mult.shape != (n_amp, n_exp):  # pragma: no cover - sanity
        raise ValueError("mult shape mismatch")
    if wave_mask.shape[0] != n_wave:  # pragma: no cover - sanity
        raise ValueError("wave_mask length mismatch")

    wm = wave_mask.astype(bool)
    # Normalized wavelength grid in [-1, 1]
    x = np.linspace(-1.0, 1.0, n_wave)[wm]
    n_poly = int(order) + 1
    # Design matrix P: columns [x^0, x^1, ..., x^order]
    P = np.vstack([x ** i for i in range(n_poly)]).T  # (nw_mask, n_poly)

    beta_all = np.zeros((n_amp, n_poly), dtype=float)

    for a in range(n_amp):
        # Build residual_1 across exposures on mask, then average across exposures
        Y_list = []
        for e in range(n_exp):
            with np.errstate(all="ignore"):
                y = (
                    bw_amp[a, e, wm]
                    / np.clip(float(mult[a, e]), 1e-6, np.inf)
                    - bw_full[e, wm]
                )
            Y_list.append(y)
        with np.errstate(all="ignore"):
            Ystack = np.vstack(Y_list) if len(Y_list) else np.empty((0, P.shape[0]))
            ymean = (
                np.nanmean(Ystack, axis=0)
                if (Ystack.size and np.isfinite(Ystack).any())
                else np.zeros(P.shape[0], dtype=float)
            )
        # Robust linear least squares on (P, ymean)
        beta, _w = robust_linear_least_squares(
            P,
            np.nan_to_num(ymean, nan=0.0),
            loss=loss,  # type: ignore[arg-type]
            delta=huber_delta,
            c=tukey_c,
        )
        beta_all[a, :] = np.asarray(beta).reshape(-1)[:n_poly]

    return beta_all
