"""Diagnostic plotting utilities for the single-shot workflow.

This module provides convenience functions to visualize key stage artifacts
produced by the single-file shot pipeline. Plots are designed to help assess
quality quickly without loading the entire HDF5 into memory.

All functions read stage outputs written in an ``outdir`` produced by the
CLI command:

    inoculate-shot <h5file> --outdir <dir> --resume

Functions use Google-style docstrings and full type hints per the developer
standards. Matplotlib is used for rendering.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as d:  # type: ignore[no-untyped-call]
        return {k: d[k] for k in d.files}


def _stage_paths(outdir: str | Path) -> Dict[str, Path]:
    out = Path(outdir)
    return {
        "bw_amp": out / "stage_01_bw_amp.npz",
        "bw_full": out / "stage_02_bw_full.npz",
        "qc": out / "stage_03_amp_qc.parquet",
        "mult": out / "stage_04_mult.npz",
        "mult_poly2d": out / "stage_0425_mult_poly2d.npz",
        "pca": out / "stage_05_pca.npz",
        "fits": out / "stage_06_amp_fits.parquet",
        "info": out / "stage_00_info.json",
    }


def _assert_exists(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Required stage artifact not found: {p}")


def _poly_curve(beta: np.ndarray, n_wave: int) -> np.ndarray:
    """Evaluate polynomial with coefficients ``beta`` on full wavelength grid.

    The fitting used a normalized grid in [-1, 1]. We apply the same here over
    the full wavelength range of length ``n_wave``.

    Args:
        beta: Polynomial coefficients (n_poly,).
        n_wave: Total wavelength bins.

    Returns:
        Array of shape (n_wave,) with the polynomial evaluated across the grid.
    """
    x_full = np.linspace(-1.0, 1.0, n_wave)
    P_full = np.vstack([x_full ** i for i in range(len(beta))]).T
    return P_full @ beta


def plot_bw_amp_vs_full(
    outdir: str | Path,
    amp: int,
    exp: int,
    *,
    ax: Optional[plt.Axes] = None,
    save: Optional[str | Path] = None,
    show: bool = False,
) -> plt.Axes:
    """Plot BW_amp for one amplifier/exposure vs. scaled BW_full reference.

    Ensures figures are not shared across calls to avoid cluttered overlays.

    Args:
        outdir: Directory containing stage artifacts.
        amp: Amplifier index (0-based).
        exp: Exposure index (0-based).
        ax: Optional Matplotlib axes to draw on. If None, a new figure is created.
        save: Optional path to save the figure (PNG recommended).
        show: If True, call ``plt.show()`` at the end.

    Returns:
        The Matplotlib axes used for the plot.
    """
    paths = _stage_paths(outdir)
    for key in ("bw_amp", "bw_full", "mult"):
        _assert_exists(paths[key])

    bw_amp = _load_npz(paths["bw_amp"])['bw_amp']  # (n_amp, n_exp, n_wave)
    bw_full = _load_npz(paths["bw_full"])['bw_full']  # (n_exp, n_wave)
    mult = _load_npz(paths["mult"])['mult_scale']  # (n_amp, n_exp)

    n_amp, n_exp, n_wave = bw_amp.shape
    if not (0 <= amp < n_amp and 0 <= exp < n_exp):
        raise IndexError("amp or exp out of range for artifacts in outdir")

    y_amp = bw_amp[amp, exp]
    y_ref = bw_full[exp] * float(np.clip(mult[amp, exp], 1e-6, np.inf))

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
        created_fig = True
    ax.plot(y_amp, label=f"BW_amp a{amp} e{exp}")
    ax.plot(y_ref, label="scaled BW_full", alpha=0.7)
    ax.set_title(f"BW_amp vs scaled BW_full (a{amp}, e{exp})")
    ax.set_xlabel("wavelength index")
    ax.set_ylabel("flux (arb)")
    ax.legend()
    ax.grid(True, alpha=0.2)

    if save is not None:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    if created_fig:
        plt.close(ax.figure)
    return ax


def plot_amp_ratio(
    outdir: str | Path,
    amp: int,
    exp: int,
    *,
    ax: Optional[plt.Axes] = None,
    save: Optional[str | Path] = None,
    show: bool = False,
) -> plt.Axes:
    """Deprecated: kept for backward-compatibility; draws single-amp ratio.

    Prefer ``plot_mult_by_amp`` for the intended multiplicative summary across
    all amplifiers. This function now ensures it creates and closes its own
    figure when ``ax`` is None to avoid plot clutter.
    """
    paths = _stage_paths(outdir)
    for key in ("bw_amp", "bw_full", "mult"):
        _assert_exists(paths[key])

    bw_amp = _load_npz(paths["bw_amp"])['bw_amp']
    bw_full = _load_npz(paths["bw_full"])['bw_full']
    mult = _load_npz(paths["mult"])['mult_scale']

    n_amp, n_exp, _ = bw_amp.shape
    if not (0 <= amp < n_amp and 0 <= exp < n_exp):
        raise IndexError("amp or exp out of range for artifacts in outdir")

    denom = bw_full[exp] * float(np.clip(mult[amp, exp], 1e-6, np.inf))
    ratio = np.divide(bw_amp[amp, exp], denom, out=np.full_like(denom, np.nan), where=np.isfinite(denom) & (np.abs(denom) > 0))

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
        created_fig = True
    ax.plot(ratio, label=f"ratio a{amp} e{exp}", color="tab:purple")
    ax.axhline(1.0, color="black", lw=1, alpha=0.5)
    ax.set_title(f"Amp ratio (wavelength) a{amp} e{exp}")
    ax.set_xlabel("wavelength index")
    ax.set_ylabel("BW_amp / (mult * BW_full)")
    ax.legend()
    ax.grid(True, alpha=0.2)

    if save is not None:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    if created_fig:
        plt.close(ax.figure)
    return ax


def plot_amp_fit(
    outdir: str | Path,
    amp: int,
    exp: int,
    *,
    components: int | None = 3,
    ax: Optional[plt.Axes] = None,
    save: Optional[str | Path] = None,
    show: bool = False,
) -> plt.Axes:
    """Plot observed (scaled) amp spectrum vs model with poly+PCA components.

    The model is reconstructed as:
        y_model = mu[e] + poly_curve + sum_k c_e{k} * evec[e,k]
    where coefficients are read from ``stage_06_amp_fits.parquet``.

    Ensures a fresh figure is used when ``ax`` is None to avoid cross-plot
    overlays from lingering global state.
    """
    paths = _stage_paths(outdir)
    for key in ("bw_amp", "bw_full", "mult", "pca", "fits"):
        _assert_exists(paths[key])

    bw_amp = _load_npz(paths["bw_amp"])['bw_amp']
    bw_full = _load_npz(paths["bw_full"])['bw_full']
    mult = _load_npz(paths["mult"])['mult_scale']
    pca = _load_npz(paths["pca"])  # pca_mean, pca_evecs
    mu = pca["pca_mean"]  # (n_exp, n_wave)
    Vfull = pca["pca_evecs"]  # (n_exp, n_comp, n_wave)
    n_exp, n_comp, n_wave = Vfull.shape

    df = pd.read_parquet(paths["fits"])  # one row per amp
    row = df.loc[df["amp"] == int(amp)].iloc[0]

    # Poly coefficients
    beta_cols = [c for c in df.columns if c.startswith("poly_c")]
    beta = np.array([row[c] for c in sorted(beta_cols, key=lambda s: int(s.split('poly_c')[1]))], dtype=float)
    poly = _poly_curve(beta, n_wave)

    # Observed (scaled by mult to remove multiplicative factor)
    y_obs = bw_amp[amp, exp] / float(np.clip(mult[amp, exp], 1e-6, np.inf))

    # Reconstruct PCA part using stored coefficients for this amp/exp
    V = Vfull[exp, :, :]  # (n_comp, n_wave)
    # Some entries outside the training wave mask may be zeros; that is fine.
    k_use = n_comp if components is None else min(int(components), n_comp)
    coeffs = np.array([row[f"c_e{exp}_k{k}"] for k in range(n_comp)], dtype=float)

    pca_part = (coeffs[:k_use, None] * V[:k_use, :]).sum(axis=0)

    # Final model to compare to y_obs = bw_amp/mult
    y_model = bw_full[exp] + mu[exp] + poly + pca_part

    # Reference BW_full for context (absolute scale)
    ref = bw_full[exp]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
        created_fig = True
    ax.plot(y_obs, label="observed (scaled)", color="tab:blue", lw=1.2)
    ax.plot(y_model, label="model (mu + poly + PCA)", color="tab:orange", lw=1.2)
    if ref is not None:
        ax.plot(ref, label="BW_full (ref)", color="tab:green", alpha=0.6)
    ax.plot(poly, label="poly (additive)", color="tab:red", alpha=0.7, ls="--")
    if k_use > 0:
        for k in range(k_use):
            ax.plot(coeffs[k] * V[k, :], label=f"PCA{k}", lw=0.9, alpha=0.5)
    ax.set_title(f"Amp {amp}, Exp {exp}: fit components")
    ax.set_xlabel("wavelength index")
    ax.set_ylabel("flux (arb)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.2)

    if save is not None:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    if created_fig:
        plt.close(ax.figure)
    return ax



def plot_mult_by_amp(
    outdir: str | Path,
    *,
    highlight_outliers: bool = True,
    show_labels: bool = True,
    save: Optional[str | Path] = None,
    show: bool = False,
) -> plt.Axes:
    """Summarize multiplicative scale by amplifier across exposures.

    Plots mult_scale[amp, exp] as a scatter vs amp index for each exposure.
    Optionally highlights non-"good" amps based on QC labels.

    Args:
        outdir: Directory containing stage artifacts.
        highlight_outliers: If True, emphasize non-good amps using red edges.
        show_labels: If True, add a small table of amp classifications.
        save: Optional path to save the resulting figure.
        show: If True, call plt.show().

    Returns:
        The Matplotlib axes used for the plot.
    """
    paths = _stage_paths(outdir)
    for key in ("mult", "qc", "info"):
        _assert_exists(paths[key])

    mult = _load_npz(paths["mult"])['mult_scale']  # (n_amp, n_exp)
    df_qc = pd.read_parquet(paths["qc"])  # columns: amp, exp, features, label merged in pipeline
    # Build per-amp label (worst-case across exps)
    labels = df_qc.groupby("amp")["label"].first() if "label" in df_qc.columns else pd.Series({i: "good" for i in range(mult.shape[0])})

    n_amp, n_exp = mult.shape
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    markers = ["o", "s", "^", "D", "v", "P"]

    fig, ax = plt.subplots(figsize=(10, 4))
    xs = np.arange(n_amp, dtype=float)
    dx = np.linspace(-0.15, 0.15, num=n_exp) if n_exp > 1 else np.array([0.0])

    # Optional overlay: 2D poly model predictions if available
    pred = None
    if paths["mult_poly2d"].exists():
        try:
            pred = _load_npz(paths["mult_poly2d"])['pred']  # (n_amp, n_exp)
        except Exception:
            pred = None

    for e in range(n_exp):
        x = xs + (dx[e] if e < len(dx) else 0.0)
        y = mult[:, e]
        if highlight_outliers and "label" in df_qc.columns:
            bad_mask = labels.reindex(range(n_amp)).fillna("good").values != "good"
            ax.scatter(x[~bad_mask], y[~bad_mask], c=colors[e % len(colors)], marker=markers[e % len(markers)], s=22, alpha=0.8, label=f"exp {e}")
            ax.scatter(x[bad_mask], y[bad_mask], facecolors="none", edgecolors="red", marker=markers[e % len(markers)], s=60, linewidths=1.0)
        else:
            ax.scatter(x, y, c=colors[e % len(colors)], marker=markers[e % len(markers)], s=22, alpha=0.8, label=f"exp {e}")
        # Overlay model as a line across amp index ordering
        if pred is not None:
            ax.plot(xs, pred[:, e], color=colors[e % len(colors)], lw=3.2, ls='-', alpha=0.7)

    ax.set_xlabel("amplifier index")
    ax.set_ylabel("multiplicative scale (mult)")
    ax.set_title("Multiplicative scale by amplifier (per exposure)")
    ax.grid(True, alpha=0.2)

    # Legend entries: exposures + outlier marker
    handles = [Line2D([0], [0], color=colors[e % len(colors)], marker=markers[e % len(markers)], linestyle="None", label=f"exp {e}") for e in range(n_exp)]
    if highlight_outliers:
        handles.append(Line2D([0], [0], color="red", marker="o", fillstyle="none", linestyle="None", label="non-good amp"))
    ax.legend(handles=handles, ncol=min(4, len(handles)))

    if show_labels and "label" in df_qc.columns:
        # Build a compact label summary at the bottom
        counts = labels.value_counts().to_dict()
        text = " | ".join([f"{k}: {v}" for k, v in counts.items()])
        ax.text(0.01, -0.18, f"QC labels — {text}", transform=ax.transAxes, fontsize=9, va="top")
        plt.subplots_adjust(bottom=0.22)

    if save is not None:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return ax


def plot_fit_example(
    outdir: str | Path,
    amp: Optional[int] = None,
    exp: Optional[int] = None,
    *,
    save: Optional[str | Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Create a 2-panel example showing residuals and model components.
 
    Panel A: Initial residual ((bw_amp/mult) - bw_full) with model components overlaid
      (net poly and net PCA contributions) over the modeling mask.
    Panel B: Final residual after applying multiplicative scaling, mu, poly, PCA
      over the same modeling mask.

    Amp/exp selection defaults to the first 'good' amplifier from QC labels
    (exposure 0) if not specified.
    """
    paths = _stage_paths(outdir)
    for key in ("bw_amp", "bw_full", "mult", "pca", "fits", "qc"):
        _assert_exists(paths[key])

    bw_amp = _load_npz(paths["bw_amp"])['bw_amp']  # (n_amp, n_exp, n_wave)
    bw_full = _load_npz(paths["bw_full"])['bw_full']  # (n_exp, n_wave)
    mult = _load_npz(paths["mult"])['mult_scale']  # (n_amp, n_exp)
    pca = _load_npz(paths["pca"])  # pca_mean, pca_evecs
    mu = pca["pca_mean"]
    V = pca["pca_evecs"]
    df = pd.read_parquet(paths["fits"])  # per-amp row
    df_qc = pd.read_parquet(paths["qc"])  # per-amp/exp rows with labels merged

    n_amp, n_exp, n_wave = bw_amp.shape

    # Choose amp/exp if not provided
    if amp is None:
        if "label" in df_qc.columns:
            good_list = sorted(df_qc.loc[df_qc["label"] == "good", "amp"].unique().tolist())
            amp = int(good_list[0]) if good_list else 0
        else:
            amp = 0
    if exp is None:
        exp = 0

    # Extract coefficients and build components
    row = df.loc[df["amp"] == int(amp)].iloc[0]
    beta_cols = [c for c in df.columns if c.startswith("poly_c")]
    beta = np.array([row[c] for c in sorted(beta_cols, key=lambda s: int(s.split('poly_c')[1]))], dtype=float)
    poly = _poly_curve(beta, n_wave)

    coeff_cols = [c for c in df.columns if c.startswith(f"c_e{exp}_k")]
    coeffs = np.array([row[c] for c in sorted(coeff_cols, key=lambda s: int(s.split('_k')[1]))], dtype=float)
    k_use = min(coeffs.shape[0], V.shape[1])
    pca_part = (coeffs[:k_use, None] * V[exp, :k_use, :]).sum(axis=0)

    # Build residuals on full grid (modeling was trained on a mask)
    y_scaled_full = bw_amp[amp, exp] / float(np.clip(mult[amp, exp], 1e-6, np.inf))
    resid0_full = y_scaled_full - bw_full[exp]
    model_full = bw_full[exp] + mu[exp] + poly + pca_part
    resid1_full = y_scaled_full - model_full

    # For the diagnostic figure, restrict to the modeling mask [40 : n-25]
    if n_wave <= 65:
        sl = slice(0, n_wave)
        mask_note = "(full range)"
    else:
        sl = slice(40, n_wave - 25)
        mask_note = "(mask: 40:-25)"

    resid0 = resid0_full[sl]
    resid1 = resid1_full[sl]
    poly_m = poly[sl]
    pca_part_m = pca_part[sl]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Panel A: initial residual with model components overlaid
    axes[0].plot(resid0, color="tab:gray", label="initial residual")
    axes[0].plot(poly_m, label="net poly", color="tab:red", ls="--", alpha=0.8)
    axes[0].plot(pca_part_m, label="net PCA", color="tab:orange", alpha=0.8)
    axes[0].axhline(0.0, color="black", lw=0.8, alpha=0.5)
    axes[0].set_title(f"Initial residual with model components {mask_note} (a{amp}, e{exp})")
    axes[0].set_ylabel("flux (arb)")
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].grid(True, alpha=0.2)

    # Panel B: final residual after full model
    axes[1].plot(resid1, color="tab:blue", label="final residual")
    axes[1].axhline(0.0, color="black", lw=0.8, alpha=0.5)
    axes[1].set_title(f"Final residual after full model {mask_note}")
    axes[1].set_xlabel("wavelength index (masked)")
    axes[1].set_ylabel("flux (arb)")
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout()

    if save is not None:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig
