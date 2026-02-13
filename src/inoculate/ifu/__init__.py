"""IFU-level (fiber-level) sky subtraction refinement.

This package provides a minimal, modular pipeline to compute per-fiber
multiplicative tweaks (delta_mult) and source masks for each IFU, exposure,
and amplifier, using the products of the shot-level pipeline.

Design goals:
- Consume existing stage artifacts from inoculate.shot.pipeline (bw_amp, bw_full,
  mult, poly, PCA, labels) without modifying their generation.
- Iterate fibers via H5VIRUS.iter_amp_blocks lazily to avoid loading full HDF5.
- Respect labeling invariants: non-good amps use minimal corrections; zero amps
  remain zero (no computation).
- Save compact per-IFU outputs to enable later application/visualization.

Public API:
- run_ifu: main entry-point for computing fiber-level refinements from a shot.
"""
from __future__ import annotations

from .pipeline import run_ifu

__all__ = ["run_ifu"]
