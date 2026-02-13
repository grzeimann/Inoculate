"""Robust loss functions and weighting utilities.

Provides Huber and Tukey biweight functions and a simple IRLS solver for
robust linear least-squares fits used in PCA projection and related tasks.

All functions use Google-style docstrings.
"""
from __future__ import annotations

from typing import Literal, Tuple

import numpy as np


def huber_weights(residuals: np.ndarray, delta: float = 1.0) -> np.ndarray:
    """Compute Huber weights for residuals.

    The Huber psi function is linear for large residuals and quadratic near 0.
    Weights are defined as w = 1 for |r| <= delta, and w = delta/|r| otherwise.

    Args:
        residuals: Residual array-like (any shape), interpreted element-wise.
        delta: Huber tuning constant (default 1.0). Larger delta is less robust.

    Returns:
        Weights array with the same shape as ``residuals`` in [0, 1].
    """
    r = np.asanyarray(residuals, dtype=float)
    with np.errstate(all="ignore"):
        absr = np.abs(r)
        w = np.ones_like(r, dtype=float)
        mask = absr > delta
        # avoid division by zero
        w[mask] = delta / np.clip(absr[mask], 1e-12, np.inf)
    w[~np.isfinite(w)] = 0.0
    return w


def tukey_biweight(residuals: np.ndarray, c: float = 4.685) -> np.ndarray:
    """Compute Tukey's biweight rho for residuals (not used directly)."""
    r = np.asanyarray(residuals, dtype=float) / float(c)
    u2 = r * r
    mask = u2 < 1.0
    out = np.zeros_like(r)
    r2 = u2[mask]
    out[mask] = (1 - r2) ** 2
    return out


def tukey_weights(residuals: np.ndarray, c: float = 4.685) -> np.ndarray:
    """Compute Tukey biweight weights for residuals.

    For |r| >= c, weight is 0 (outliers rejected). For |r| < c, weight is
    (1 - (r/c)^2)^2.

    Args:
        residuals: Residual array-like (any shape), interpreted element-wise.
        c: Tukey tuning constant (default 4.685 ~ 95% efficiency for Normal).

    Returns:
        Weights array with the same shape as ``residuals`` in [0, 1].
    """
    r = np.asanyarray(residuals, dtype=float)
    with np.errstate(all="ignore"):
        u = r / float(c)
        w = (1.0 - u * u) ** 2
        w[np.abs(u) >= 1.0] = 0.0
    w[~np.isfinite(w)] = 0.0
    return w


def robust_linear_least_squares(
    A: np.ndarray,
    y: np.ndarray,
    *,
    loss: Literal["huber", "tukey"] = "huber",
    delta: float = 1.0,
    c: float = 4.685,
    max_iter: int = 20,
    tol: float = 1e-6,
    ridge: float = 0.0,
    mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve min_x sum rho(residuals) via IRLS for y ≈ A @ x.

    Args:
        A: Design matrix of shape (n_samples, n_params).
        y: Target vector of shape (n_samples,) or (n_samples, n_targets).
        loss: Robust loss to use: 'huber' or 'tukey'.
        delta: Huber tuning constant.
        c: Tukey tuning constant.
        max_iter: Maximum IRLS iterations.
        tol: Convergence tolerance on parameter change (L2 norm).
        ridge: Optional L2 regularization added to normal equations.
        mask: Optional boolean mask for valid samples (n_samples,). False rows are ignored.

    Returns:
        Tuple (x, weights) where x has shape (n_params,) for 1D y, or
        (n_params, n_targets) for 2D y; weights are the final per-sample weights
        of shape (n_samples,).
    """
    A = np.asanyarray(A, dtype=float)
    y = np.asanyarray(y, dtype=float)

    if y.ndim == 1:
        y = y[:, None]
    n_samples, n_params = A.shape
    if y.shape[0] != n_samples:
        raise ValueError("y length does not match A rows")

    valid = np.isfinite(A).all(axis=1) & np.isfinite(y).all(axis=1)
    if mask is not None:
        valid &= mask.astype(bool)
    if not np.any(valid):
        return np.zeros((n_params, y.shape[1])), np.zeros(n_samples)

    A = A[valid]
    y = y[valid]
    n_samples_eff = A.shape[0]

    # Initialize with ordinary least squares
    try:
        if ridge > 0:
            # Solve (A^T W A + ridge I) x = A^T W y with W=I initially
            ATA = A.T @ A + ridge * np.eye(n_params)
            ATy = A.T @ y
            x = np.linalg.solve(ATA, ATy)
        else:
            x, *_ = np.linalg.lstsq(A, y, rcond=None)
    except Exception:
        x = np.zeros((n_params, y.shape[1]))

    w = np.ones(n_samples_eff, dtype=float)

    for _ in range(max_iter):
        yhat = A @ x
        r = y - yhat  # (n_samples_eff, n_targets)
        # Reduce to scalar residual per sample via robust scale (L2 over targets)
        rs = np.sqrt(np.nanmean(r * r, axis=1))
        # Estimate robust scale s using MAD of rs
        with np.errstate(all="ignore"):
            med = np.nanmedian(rs)
            mad = np.nanmedian(np.abs(rs - med))
            s = max(1e-12, 1.4826 * mad)
            z = (rs - med) / s
        if loss == "huber":
            w_new = huber_weights(z, delta=delta)
        else:
            w_new = tukey_weights(z, c=c)
        # Guard against degenerate all-zero weights
        if not np.any(w_new > 0):
            break
        # Form weighted least squares
        sw = np.sqrt(w_new)[:, None]
        Aw = A * sw
        yw = y * sw
        try:
            if ridge > 0:
                ATA = Aw.T @ Aw + ridge * np.eye(n_params)
                ATy = Aw.T @ yw
                x_new = np.linalg.solve(ATA, ATy)
            else:
                x_new, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
        except Exception:
            break
        # Convergence check
        dx = np.linalg.norm(x_new - x)
        x = x_new
        w = w_new
        if not np.isfinite(dx) or dx < tol:
            break

    # Map weights back to full length
    w_full = np.zeros(mask.shape[0] if mask is not None else n_samples, dtype=float)
    w_full_idx = np.where(valid)[0]
    w_full[w_full_idx] = w

    return (x if x.shape[1] != 1 else x[:, 0]), w_full
