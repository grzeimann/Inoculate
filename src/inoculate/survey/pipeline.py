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
from ..robust import biweight_location

logger = logging.getLogger(__name__)


@dataclass
class SurveyOptions:
    """Tuning options for the survey orchestrator.

    Attributes:
        include_partial_ifu: If False, skip IFUs missing any exposure files in a shot.
        max_shots: Optional cap on number of shots (or H5 files) to process.
        build_from_h5: If True, discover .h5 under shots_root and run per-shot pipelines
            into survey_root/shots/<shotname>/ before aggregation. If False, treat
            shots_root as a directory tree of existing shot outdirs (with manifests).
        shot_resume: Pass through to run_shot(resume=...).
        ifu_resume: Pass through to run_ifu(resume=...).
        suppress_warnings: Suppress runtime warnings in run_shot.
    """

    include_partial_ifu: bool = True
    max_shots: Optional[int] = None
    build_from_h5: bool = False
    shot_resume: bool = True
    ifu_resume: bool = True
    suppress_warnings: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "include_partial_ifu": bool(self.include_partial_ifu),
            "max_shots": self.max_shots,
            "build_from_h5": bool(self.build_from_h5),
            "shot_resume": bool(self.shot_resume),
            "ifu_resume": bool(self.ifu_resume),
            "suppress_warnings": bool(self.suppress_warnings),
        }


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
      - mean_profile: biweight_location over samples of delta_mult[448]
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
        # Robust mean profile (biweight location) and MAD-based dispersion
        with np.errstate(all="ignore"):
            mean_profile = biweight_location(X, axis=0)
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


def _write_ifu_profile_plots(shots: List[Path], reg: Dict[int, Dict[str, Any]], plan: SurveyPlan) -> int:
    """Write per-IFU overlay plots: individual profiles and robust mean.

    Returns the number of plots successfully written. Missing matplotlib is handled
    gracefully (returns 0).
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return 0

    n_written = 0
    # Build a quick in-memory index of profiles per IFU
    profiles_by_ifu: Dict[int, List[np.ndarray]] = {}
    for shot_dir in shots:
        for ifu, _e, p in _iter_ifu_npz(shot_dir):
            try:
                with np.load(p) as d:  # type: ignore[no-untyped-call]
                    dm = np.array(d.get("delta_mult"), dtype=float)
            except Exception:
                continue
            if dm.ndim != 1:
                dm = dm.reshape(-1)
            if dm.size == 0:
                continue
            profiles_by_ifu.setdefault(int(ifu), []).append(dm)

    for ifu, entry in reg.items():
        plot_path = plan.ifu_profiles_plot_path(int(ifu))
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        mean_prof = np.array(entry.get("mean_delta_mult", []), dtype=float)
        Xs = profiles_by_ifu.get(int(ifu), [])
        if not Xs and mean_prof.size == 0:
            continue
        try:
            fig, ax = plt.subplots(figsize=(10, 3.0))
            x = np.arange(mean_prof.size if mean_prof.size else (Xs[0].size if Xs else 448))
            # Plot individual profiles lightly
            for y in Xs:
                if y.size != x.size:
                    continue
                ax.plot(x, y, color="#1f77b4", alpha=0.05, lw=0.8)
            # Overlay robust mean in a stronger color
            if mean_prof.size == x.size:
                ax.plot(x, mean_prof, color="tab:red", lw=1.6, label="biweight mean")
            ax.set_title(f"IFU {int(ifu):03d} — delta_mult profiles across shots")
            ax.set_xlabel("fiber index (4 amps × 112)")
            ax.set_ylabel("delta_mult")
            ax.grid(True, alpha=0.2)
            if mean_prof.size == x.size:
                ax.legend(loc="upper right", fontsize=8)
            # Ensure visible y-range even if constant
            ydata = []
            if mean_prof.size:
                ydata.append(mean_prof)
            if Xs:
                ydata.append(np.concatenate([y for y in Xs if y.size == x.size]))
            if ydata:
                vals = np.concatenate(ydata)
                if np.isfinite(vals).any():
                    vmin = float(np.nanmin(vals))
                    vmax = float(np.nanmax(vals))
                    if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                        pad = 0.05 * (vmax - vmin)
                        ax.set_ylim(vmin - pad, vmax + pad)
            ax.set_ylim([0.8, 1.2])
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            n_written += 1
        except Exception:
            try:
                plt.close(fig)  # type: ignore[name-defined]
            except Exception:
                pass
            continue
    return n_written


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

    # Determine shot outdirs: either discover existing, or build from H5 files
    shots: List[Path]
    if opts.build_from_h5:
        # Build per-shot outputs under survey_root/shots from H5 inputs
        from .aggregate import find_h5_files
        from ..shot.pipeline import run_shot
        from ..ifu.pipeline import run_ifu
        h5_files = find_h5_files(shots_root)
        if opts.max_shots is not None:
            h5_files = h5_files[: int(opts.max_shots)]
        built_shots: List[Path] = []
        shots_dir = plan.shots_dir
        shots_dir.mkdir(parents=True, exist_ok=True)
        for h5 in h5_files:
            try:
                stem = h5.stem
                out = shots_dir / stem
                out.mkdir(parents=True, exist_ok=True)
                # Run shot then IFU
                run_shot(str(h5), outdir=str(out), resume=bool(opts.shot_resume), make_plots=False, suppress_warnings=bool(opts.suppress_warnings))
                run_ifu(str(h5), outdir=str(out), resume=bool(opts.ifu_resume), suppress_warnings=bool(opts.suppress_warnings))
                built_shots.append(out)
            except Exception:
                logger.exception("Failed processing shot H5: %s", h5)
                continue
        shots = built_shots
    else:
        # Discover pre-existing shot outputs by manifest
        shots = _discover_shot_outdirs(shots_root)
        if opts.max_shots is not None:
            shots = shots[: int(opts.max_shots)]

    # Write shot index
    idx_path = plan.paths()["Shot_Index"]
    with idx_path.open("w", encoding="utf-8") as f:
        json.dump({"shots": [str(p) for p in shots]}, f, indent=2)

    # Aggregate per-IFU across shots
    import warnings
    if opts.suppress_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.filterwarnings("ignore", category=UserWarning, module=r"astropy\\.stats")
            warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"astropy\\.stats")
            reg = _aggregate_ifu_across_shots(shots, include_partial_ifu=opts.include_partial_ifu)
    else:
        reg = _aggregate_ifu_across_shots(shots, include_partial_ifu=opts.include_partial_ifu)

    # Persist per-IFU registry entries
    n_ifu_written = 0
    for ifu, d in reg.items():
        dest = plan.ifu_registry_path(int(ifu))
        with dest.open("w", encoding="utf-8") as f:
            json.dump(d, f)
        n_ifu_written += 1

    # Write per-IFU overlay plots (individual profiles + robust mean)
    try:
        if opts.suppress_warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                warnings.filterwarnings("ignore", category=UserWarning, module=r"matplotlib")
                n_plots = _write_ifu_profile_plots(shots, reg, plan)
        else:
            n_plots = _write_ifu_profile_plots(shots, reg, plan)
        logger.info("Wrote %d IFU profile plots to %s", int(n_plots), plan.plots_dir)
    except Exception:
        logger.warning("Failed to write IFU profile plots", exc_info=True)

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
