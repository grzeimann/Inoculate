"""IFU plan and standardized path resolution for fiber-level artifacts.

Provides a single source of truth for IFU pipeline output locations and
filenames, avoiding hardcoded patterns in the pipeline implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IFUPlan:
    """Resolver for IFU-level (fiber-level) artifact paths.

    Attributes:
        shot_outdir: Base directory where the shot pipeline wrote its artifacts.
    """

    shot_outdir: Path

    @property
    def ifu_dir(self) -> Path:
        return Path(self.shot_outdir) / "ifu"

    @property
    def plots_dir(self) -> Path:
        return self.ifu_dir / "plots"

    def ensure_dirs(self) -> None:
        self.ifu_dir.mkdir(parents=True, exist_ok=True)
        # plots dir created on demand

    def fiber_model_path(self, amp: int, exp: int) -> Path:
        """Return the path for the per-amp/per-exp fiber model NPZ file."""
        return self.ifu_dir / f"a{int(amp):02d}_e{int(exp):02d}_fiber_model.npz"

    def ifu_model_path(self, ifu: int, exp: int) -> Path:
        """Return the path for the per-IFU/per-exp aggregated fiber model NPZ file."""
        return self.ifu_dir / f"ifu{int(ifu):03d}_e{int(exp):02d}_fiber_model.npz"

    def fiber_plot_path(self, amp: int, exp: int) -> Path:
        """Path for diagnostic plot of delta_mult for a given (amp, exp)."""
        return self.plots_dir / f"a{int(amp):02d}_e{int(exp):02d}_delta_mult.png"

    def ifu_plot_path(self, ifu: int) -> Path:
        """Path for per-IFU diagnostic plot (delta_mult vs fiber, lines per exposure)."""
        return self.plots_dir / f"ifu{int(ifu):03d}_delta_mult.png"

    def manifest_path(self) -> Path:
        return self.ifu_dir / "manifest.json"
