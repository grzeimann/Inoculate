"""Plotting helpers for Inoculate diagnostics.

Public API:
- plot_bw_amp_vs_full
- plot_amp_ratio (deprecated single-amp ratio vs wavelength)
- plot_amp_fit
- plot_mult_by_amp (multiplicative summary across amps)
- plot_fit_example (two-panel: initial residual + model components; final residual)
"""
from .diagnostics import (
    plot_bw_amp_vs_full,
    plot_amp_ratio,
    plot_amp_fit,
    plot_mult_by_amp,
    plot_fit_example,
)

__all__ = [
    "plot_bw_amp_vs_full",
    "plot_amp_ratio",
    "plot_amp_fit",
    "plot_mult_by_amp",
    "plot_fit_example",
]
