"""Shot plan and standardized stage path resolution.

This module centralizes the naming conventions for shot-level pipeline
artifacts and provides a small helper to compute a plan if/when we add more
advanced scheduling. For now, it focuses on de-hardcoding file names so that
both the pipeline and plotting utilities use a single source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class ShotPlan:
    """Centralized resolver for shot-stage artifact paths.

    Attributes:
        outdir: Base directory for all shot artifacts.
    """

    outdir: Path

    def paths(self) -> Dict[str, Path]:
        """Return the canonical paths for all standard shot artifacts.

        Keys are stable identifiers used throughout the codebase. Values are
        absolute Paths under ``outdir``.
        """
        out = Path(self.outdir)
        return {
            "stage_00_info": out / "stage_00_info.json",
            "stage_01_bw_amp": out / "stage_01_bw_amp.npz",
            "stage_02_bw_full": out / "stage_02_bw_full.npz",
            "stage_03_qc": out / "stage_03_amp_qc.parquet",
            "stage_03_labels": out / "stage_03_labels.json",
            "stage_04_mult": out / "stage_04_mult.npz",
            "stage_04_labels": out / "stage_04_labels.json",
            "stage_0425_mult_poly2d": out / "stage_0425_mult_poly2d.npz",
            "stage_0425_labels": out / "stage_0425_labels.json",
            "stage_045_poly": out / "stage_045_poly.npz",
            "stage_05_pca": out / "stage_05_pca.npz",
            "stage_06_amp_fits": out / "stage_06_amp_fits.parquet",
            "stage_07_manifest": out / "stage_07_model_start_manifest.json",
        }

    def path(self, key: str) -> Path:
        p = self.paths().get(key)
        if p is None:
            raise KeyError(f"Unknown shot stage key: {key}")
        return p


def compute_plan(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the processing plan for a shot (placeholder).

    Currently returns a simple structure; reserved for future dynamic planning.
    """
    return {"steps": [], "notes": ["not-implemented"]}
