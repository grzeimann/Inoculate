"""2D polynomial model for multiplicative scales vs sky position.

Fits, per exposure, a 2D polynomial in RA/Dec (arcmin offsets) to explain the
amplifier multiplicative scales (mult). RA/Dec per amplifier/exposure are
computed as robust (biweight) centers over the 112 fibers in that amp/exp.

Google-style docstrings and full type hints are used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from ...io.h5 import H5VIRUS, amp_exposure_slice
from ...robust import biweight_location, robust_linear_least_squares


@dataclass
class Poly2DResult:
    """Result of the 2D polynomial fit.

    Attributes:
        degree: Polynomial degree used (non-negative integer).
        coeffs: Coefficients per exposure with shape (n_exp, n_terms).
        ra_amp: RA biweight per (amp, exp) in degrees, shape (n_amp, n_exp).
        dec_amp: Dec biweight per (amp, exp) in degrees, shape (n_amp, n_exp).
        x: RA arcmin offsets from global mean, shape (n_amp, n_exp).
        y: Dec arcmin offsets from global mean, shape (n_amp, n_exp).
        pred: Predicted mult values per (amp, exp) from the fitted model,
            shape (n_amp, n_exp).
    """

    degree: int
    coeffs: np.ndarray
    ra_amp: np.ndarray
    dec_amp: np.ndarray
    x: np.ndarray
    y: np.ndarray
    pred: np.ndarray


def _design_matrix_xy(x: np.ndarray, y: np.ndarray, degree: int) -> Tuple[np.ndarray, list[str]]:
    """Build a 2D polynomial design matrix for coordinates (x, y).

    Generates all monomials x^i y^j such that i + j <= degree, ordered by total
    degree (0, 1, 2, ...), and within each degree in descending x-power
    (graded lexicographic: x^d, x^{d-1}y, ..., y^d).

    Args:
        x: Array of x coordinates (any shape). Interpreted element-wise.
        y: Array of y coordinates with the same shape as ``x``.
        degree: Non-negative polynomial degree (0, 1, 2, ...).

    Returns:
        Tuple (A, names) where A has shape (x.size, n_terms) and names lists the
        column names in order.
    """
    if int(degree) != degree or degree < 0 or degree > 10:
        raise ValueError("degree must be a non-negative integer < 11 for the 2D polynomial model")
    d = int(degree)
    x1 = np.asarray(x, dtype=float).ravel()
    y1 = np.asarray(y, dtype=float).ravel()

    cols: list[np.ndarray] = []
    names: list[str] = []
    # Degree 0 term
    cols.append(np.ones_like(x1))
    names.append("1")
    # Higher degree terms
    for total in range(1, d + 1):
        for i in range(total, -1, -1):  # i = total..0, j = total - i
            j = total - i
            term = (x1 ** i) * (y1 ** j)
            cols.append(term)
            if i == 0 and j == 0:
                names.append("1")
            elif i == 0:
                names.append(f"y{j}")
            elif j == 0:
                names.append(f"x{i}")
            else:
                names.append(f"x{i}y{j}")

    A = np.vstack(cols).T  # (N, n_terms)
    return A, names


def build_mult_poly2d(
    h5: H5VIRUS,
    mult: np.ndarray,
    *,
    degree: int = 2,
    loss: str = "huber",
    huber_delta: float = 1.0,
    tukey_c: float = 4.685,
) -> Poly2DResult:
    """Fit a per-exposure 2D polynomial to multiplicative scales using RA/Dec.

    The coordinates for each amplifier/exposure are computed as the robust
    (biweight) mean of the per-fiber RA, Dec columns over that amp/exp block.

    Args:
        h5: Open VIRUS reader for the shot file.
        mult: Array of multiplicative scales with shape (n_amp, n_exp).
        degree: Polynomial degree (non-negative integer). Default is 2.
        loss: Robust loss to use in the IRLS solver ("huber" or "tukey").
        huber_delta: Huber tuning constant.
        tukey_c: Tukey tuning constant.

    Returns:
        Poly2DResult object containing fitted coefficients and predictions.
    """
    info = h5.read_info()
    n_amp = int(info["n_amp"])  # type: ignore[arg-type]
    n_exp = int(info["exposures"])  # type: ignore[arg-type]
    fibers_per_amp = int(info["fibers_per_amp"])  # type: ignore[arg-type]

    ra_amp = np.full((n_amp, n_exp), np.nan, dtype=float)
    dec_amp = np.full((n_amp, n_exp), np.nan, dtype=float)

    # Read RA/Dec from /Info table; align slices using amp_exposure_slice
    h5._require_tables()  # may raise ImportError if pytables missing
    with h5._open() as fh:  # type: ignore[attr-defined]
        try:
            info_tbl = fh.root.Info
        except Exception as exc:  # pragma: no cover (file dependent)
            raise RuntimeError(f"Missing /Info table required for RA/Dec: {exc}") from exc

        if not (hasattr(info_tbl.cols, "ra") and hasattr(info_tbl.cols, "dec")):
            raise RuntimeError("/Info table must contain 'ra' and 'dec' columns for mult poly2d")

        for a in range(n_amp):
            for e in range(n_exp):
                s = amp_exposure_slice(a, e, fibers_per_amp, n_exp)
                ra_block = info_tbl.cols._f_col("ra")[s]  # type: ignore[attr-defined]
                dec_block = info_tbl.cols._f_col("dec")[s]  # type: ignore[attr-defined]
                ra_amp[a, e] = float(biweight_location(ra_block, axis=0))
                dec_amp[a, e] = float(biweight_location(dec_block, axis=0))

    # Center to arcmin offsets
    ra_mean = float(np.nanmean(ra_amp))
    dec_mean = float(np.nanmean(dec_amp))
    x = (ra_amp - ra_mean) * 60.0  # arcmin
    y = (dec_amp - dec_mean) * 60.0  # arcmin

    # Prepare outputs
    if int(degree) != degree or degree < 0:
        raise ValueError("degree must be a non-negative integer")
    n_terms = (int(degree) + 1) * (int(degree) + 2) // 2
    coeffs = np.zeros((n_exp, n_terms), dtype=float)
    pred = np.full_like(mult, np.nan, dtype=float)

    # Fit per exposure
    for e in range(n_exp):
        xe = x[:, e]
        ye = y[:, e]
        me = mult[:, e]
        A, _ = _design_matrix_xy(xe, ye, degree)
        # Robust fit; mask finite
        mask = np.isfinite(xe) & np.isfinite(ye) & np.isfinite(me)
        beta, _ = robust_linear_least_squares(
            A,
            me.ravel(),
            loss="huber" if loss == "huber" else "tukey",
            delta=huber_delta,
            c=tukey_c,
            mask=mask.ravel(),
        )
        if beta.ndim > 1:
            beta = beta[:, 0]
        coeffs[e, : beta.shape[0]] = beta
        # Predictions
        pred[:, e] = (A @ beta).reshape((-1,))

    return Poly2DResult(
        degree=degree,
        coeffs=coeffs,
        ra_amp=ra_amp,
        dec_amp=dec_amp,
        x=x,
        y=y,
        pred=pred,
    )
