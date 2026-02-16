"""Survey-level orchestrator for aggregating many shots and building IFU priors.

This module provides a minimal, resumable pipeline that:
- Discovers shot output directories under a root (or from an explicit list),
- Loads per-IFU fiber-level products written by the IFU pipeline, and
- Aggregates statistics across shots to form a registry of per-IFU behaviors.

Outputs are written under a survey root directory using SurveyPlan. The key
artifact is a per-IFU registry JSON that captures average delta_mult profiles
(length 448; 4 amps × 112 fibers) and robust dispersions (MAD-based), which can
be consumed in a second IFU pass as priors.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import json
import logging
import numpy as np

from .plan import SurveyPlan

logger = logging.getLogger(__name__)


@dataclass
class SurveyOptions:
    """Tuning options for the survey orchestrator.

    Attributes:
        include_partial_ifu: If False, skip IFUs missing any exposure files in a shot.
        max_shots: Optional limit for development/testing to cap number of shots.
    """

    include_partial_ifu: bool = True
    max_shots: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"include_partial_ifu": bool(self.include_partial_ifu), "max_shots": self.max_shots}


# ---- Stage-like helpers ----

def _discover_shot_outdirs(root: Path) -> List[Path]:
    """Return shot outdirs under root by locating the shot manifest.

    We consider a directory to be a shot outdir if it contains
    stage_07_model_start_manifest.json. This matches ShotPlan's manifest path.
    """
    shots: List[Path] = []
    for p in root.rglob("stage_07_model_start_manifest.json"):
        shots.append(p.parent)
    shots_sorted = sorted(set(shots))
    logger.info("Discovered %d shot outdirs under %s", len(shots_sorted), root)
    return shots_sorted


def _iter_ifu_npz(shot_outdir: Path) -> Iterable[Tuple[int, int, Path]]:
    """Yield (ifu_idx, exp_idx, npz_path) for per-IFU files in a shot outdir."""
    ifu_dir = shot_outdir / "ifu"
    if not ifu_dir.exists():
        return []  # type: ignore[return-value]
    for p in sorted(ifu_dir.glob("ifu*_e*_fiber_model.npz")):
        try:
            stem = p.stem  # e.g., ifu012_e01_fiber_model
            parts = stem.split("_")
            ifu = int(parts[0].replace("ifu", ""))
            e = int(parts[1].replace("e", ""))
        except Exception:
            continue
        yield (ifu, e, p)


def _aggregate_ifu_across_shots(shots: List[Path], *, include_partial_ifu: bool = True) -> Dict[int, Dict[str, Any]]:
    """Aggregate per-IFU delta_mult across all provided shots.

    For each IFU index, load all available (exp) NPZ files across shots and build:
      - mean_profile: nanmean over samples of delta_mult[448]
      - mad_profile: 1.4826 * median(|x - median(x)|) per fiber across samples
      - n_samples: number of contributing vectors
      - frac_masked: mean of source_mask across samples per fiber (optional)

    Returns a mapping ifu_idx -> stats dict suitable for JSON.
    """
    # First pass: collect lists per IFU
    buckets: Dict[int, Dict[str, List[np.ndarray]]] = {}
    counts: Dict[int, int] = {}
    mask_lists: Dict[int, List[np.ndarray]] = {}
    for shot_dir in shots:
        # Index IFU files by ifu and exposure for this shot
        by_ifu: Dict[int, List[Path]] = {}
        for ifu, e, path in _iter_ifu_npz(shot_dir):
            by_ifu.setdefault(ifu, []).append(path)
        for ifu, files in by_ifu.items():
            # If require complete exposures per shot, enforce at least 1 file
            if not files:
                continue
            # Load each file's delta_mult and mask
            for p in files:
                try:
                    with np.load(p) as d:  # type: ignore
                        dm = np.array(d.get("delta_mult"), dtype=float)
                        sm = np.array(d.get("source_mask", np.zeros_like(dm, dtype=bool)), dtype=bool)
                except Exception:
                    continue
                if dm.size == 0:
                    continue
                if dm.ndim != 1:
                    dm = dm.reshape(-1)
                buckets.setdefault(ifu, {}).setdefault("delta", []).append(dm)
                mask_lists.setdefault(ifu, []).append(sm)
                counts[ifu] = counts.get(ifu, 0) + 1
    # Second pass: compute aggregates
    reg: Dict[int, Dict[str, Any]] = {}
    for ifu, dct in buckets.items():
        arrs = dct.get("delta", [])
        if not arrs:
            continue
        # Stack to (n_samples, 448)
        X = np.vstack([a.astype(float) for a in arrs])
        # Mean profile (nanmean)
        with np.errstate(all="ignore"):
            mean_profile = np.nanmean(X, axis=0)
            med = np.nanmedian(X, axis=0, keepdims=True)
            mad_profile = 1.4826 * np.nanmedian(np.abs(X - med), axis=0)
        # Mask fraction
        if mask_lists.get(ifu):
            M = np.vstack([m.astype(bool) for m in mask_lists[ifu]])
            frac_masked = np.mean(M, axis=0).astype(float)
        else:
            frac_masked = np.zeros(mean_profile.shape, dtype=float)
        reg[ifu] = {
            "ifu": int(ifu),
            "n_samples": int(counts.get(ifu, 0)),
            "mean_delta_mult": mean_profile.astype(float).tolist(),
            "mad_delta_mult": mad_profile.astype(float).tolist(),
            "frac_masked": frac_masked.astype(float).tolist(),
        }
    return reg


def run_survey(
    shots_root: str | Path,
    survey_root: str | Path,
    *,
    resume: bool = True,
    options: SurveyOptions | None = None,
) -> Dict[str, Any]:
    """Run the survey-level aggregation to build per-IFU priors registry.

    Args:
        shots_root: Root directory under which individual shot outdirs reside.
        survey_root: Output directory for survey artifacts (registry, manifests).
        resume: Placeholder for future per-stage resume; not used heavily yet.
        options: Optional SurveyOptions for tuning.

    Returns:
        Dict with manifest and basic counts.
    """
    shots_root = Path(shots_root)
    survey_root = Path(survey_root)
    plan = SurveyPlan(survey_root)
    plan.ensure_dirs()
    opts = options or SurveyOptions()

    # Discover shots
    shots = _discover_shot_outdirs(shots_root)
    if opts.max_shots is not None:
        shots = shots[: int(opts.max_shots)]

    # Write shot index
    idx_path = plan.paths()["Shot_Index"]
    with idx_path.open("w", encoding="utf-8") as f:
        json.dump({"shots": [str(p) for p in shots]}, f, indent=2)

    # Aggregate per-IFU across shots
    reg = _aggregate_ifu_across_shots(shots, include_partial_ifu=opts.include_partial_ifu)

    # Persist per-IFU registry entries
    n_ifu_written = 0
    for ifu, d in reg.items():
        dest = plan.ifu_registry_path(int(ifu))
        with dest.open("w", encoding="utf-8") as f:
            json.dump(d, f)
        n_ifu_written += 1

    # Global survey stats
    stats_path = plan.paths()["Survey_Stats"]
    survey_stats = {
        "n_shots": len(shots),
        "n_ifu_with_entries": n_ifu_written,
        "options": opts.to_dict(),
    }
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(survey_stats, f, indent=2)

    # Manifest
    manifest = {
        "shots_root": str(shots_root.resolve()),
        "survey_root": str(survey_root.resolve()),
        "shot_index": str(idx_path),
        "registry_dir": str(plan.registry_dir),
        "survey_stats": str(stats_path),
        "n_shots": len(shots),
        "n_ifu": n_ifu_written,
    }
    man_path = plan.paths()["Survey_Manifest"]
    with man_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Survey complete: %s (IFUs: %d, shots: %d)", man_path, n_ifu_written, len(shots))
    return manifest
