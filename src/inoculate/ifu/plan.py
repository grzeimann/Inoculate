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

    def ensure_dirs(self) -> None:
        self.ifu_dir.mkdir(parents=True, exist_ok=True)

    def fiber_model_path(self, amp: int, exp: int) -> Path:
        """Return the path for the per-amp/per-exp fiber model NPZ file."""
        return self.ifu_dir / f"a{int(amp):02d}_e{int(exp):02d}_fiber_model.npz"

    def manifest_path(self) -> Path:
        return self.ifu_dir / "manifest.json"
