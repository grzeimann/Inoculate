"""IFU (fiber-level) refinement pipeline.

Builds per-fiber multiplicative tweaks (delta_mult) and source masks by
consuming shot-level artifacts. Outputs compact NPZ files per amplifier and
exposure, grouped under outdir/ifu/.

This is a minimal implementation intended to match the Junie instructions:
- ZERO amps remain zero; we skip computations and record the invariant.
- Non-good amps (per labels) are treated as DEFECTIVE: we record minimal model
  usage and compute diagnostics but do not apply refined corrections.
- GOOD amps: compute per-fiber provisional delta_mult against BW_full using a
  robust wavelength window; then identify likely source-affected fibers using a
  MAD threshold across fibers within an amp/exposure.

The pipeline reads only what it needs from the HDF5 using H5VIRUS.iter_amp_blocks
and the /Info table for IFU slot annotations.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from ..io.h5 import H5VIRUS, amp_exposure_slice
from ..qc.labels import load_labelset
from ..robust import biweight_location, mad


@dataclass
class IFUOptions:
    wave_mask_frac: float = 0.8  # central fraction of wavelengths to use
    k_source: float = 5.0        # MAD threshold for source masking across fibers


def _wave_mask(n_wave: int, frac: float) -> np.ndarray:
    frac = float(np.clip(frac, 0.05, 1.0))
    n = int(max(1, round(frac * int(n_wave))))
    start = (int(n_wave) - n) // 2
    end = start + n
    w = np.zeros(int(n_wave), dtype=bool)
    w[start:end] = True
    return w


def _latest_labelset(outdir: Path) -> Any | None:
    # Prefer centralized ShotPlan label paths with graceful fallback
    try:
        from ..shot.plan import ShotPlan
        sp = ShotPlan(outdir).paths()
        candidates = [
            sp.get("stage_0425_labels"),
            sp.get("stage_04_labels"),
            sp.get("stage_03_labels"),
        ]
    except Exception:
        candidates = [
            outdir / "stage_0425_labels.json",
            outdir / "stage_04_labels.json",
            outdir / "stage_03_labels.json",
        ]
    for c in candidates:
        if c is not None and c.exists():
            try:
                return load_labelset(c)
            except Exception:
                continue
    return None


def _load_shot_artifacts(outdir: Path) -> Dict[str, np.ndarray]:
    def _load_npz(p: Path) -> Dict[str, np.ndarray]:
        with np.load(p) as d:  # type: ignore[no-untyped-call]
            return {k: d[k] for k in d.files}

    bw_amp = _load_npz(outdir / "stage_01_bw_amp.npz").get("bw_amp")
    bw_full = _load_npz(outdir / "stage_02_bw_full.npz").get("bw_full")
    mult = _load_npz(outdir / "stage_04_mult.npz").get("mult_scale")
    if bw_amp is None or bw_full is None or mult is None:
        raise FileNotFoundError("Required shot artifacts (bw_amp, bw_full, mult) are missing in outdir")
    artifacts: Dict[str, np.ndarray] = {
        "bw_amp": bw_amp,
        "bw_full": bw_full,
        "mult": mult,
    }
    # Optional poly2d predictions and PCA could be added later if needed
    return artifacts


def _amp_label(ls: Any | None, a: int, e: int, zero_frac: Optional[float]) -> str:
    # ZERO invariant shortcut if parquet features said many zeros
    if zero_frac is not None and np.isfinite(zero_frac) and zero_frac > 0.5:
        return "ZERO"
    if ls is None:
        return "GOOD"  # default optimistic
    try:
        use = bool(ls.mask[a, e])
    except Exception:
        # fallback to per-amp mask if available
        try:
            use = bool(ls.good_mask()[a])
        except Exception:
            use = True
    return "GOOD" if use else "DEFECTIVE_OR_BRIGHT"


def _read_zero_frac_table(outdir: Path) -> Optional[np.ndarray]:
    # Try to read stage_03_amp_qc.parquet for zero_frac by (amp, exp)
    p = outdir / "stage_03_amp_qc.parquet"
    if not p.exists():
        return None
    try:
        import pandas as pd  # local import to avoid mandatory dep otherwise
        df = pd.read_parquet(p)
        n_amp = int(df["amp"].max()) + 1
        n_exp = int(df["exp"].max()) + 1
        z = np.full((n_amp, n_exp), np.nan, dtype=float)
        for _, r in df.iterrows():
            z[int(r["amp"]), int(r["exp"]) ] = float(r.get("zero_frac", np.nan))
        return z
    except Exception:
        return None


def run_ifu(
    h5file: str | Path,
    outdir: str | Path,
    *,
    ifu_indices: Optional[Iterable[int]] = None,
    resume: bool = True,
    options: IFUOptions | None = None,
) -> Dict[str, Any]:
    """Run the IFU-level refinement and write per-amplifier outputs.

    Args:
        h5file: Path to the VIRUS HDF5 shot file.
        outdir: Directory containing shot-level artifacts (same as for inoculate-shot).
        ifu_indices: Optional iterable of amplifier indices to process; if None, process all.
        resume: Skip writing files that already exist.
        options: Tuning parameters for the refinement.

    Returns:
        Mapping with summary counts and output directory for IFU artifacts.
    """
    out = Path(outdir)
    from .plan import IFUPlan
    ifu_plan = IFUPlan(out)
    ifu_plan.ensure_dirs()
    ifu_out = ifu_plan.ifu_dir
    opts = options or IFUOptions()

    # Load artifacts and labels
    artifacts = _load_shot_artifacts(out)
    bw_full = artifacts["bw_full"]  # (n_exp, n_wave)
    n_exp, n_wave = int(bw_full.shape[0]), int(bw_full.shape[1])
    wmask = _wave_mask(n_wave, opts.wave_mask_frac)
    ls = _latest_labelset(out)
    zero_table = _read_zero_frac_table(out)

    # HDF5 iteration
    h5 = H5VIRUS(h5file)
    info = h5.read_info()
    n_amp = int(info["n_amp"])  # type: ignore[arg-type]
    exposures = int(info["exposures"])  # type: ignore[arg-type]
    fibers_per_amp = int(info["fibers_per_amp"])  # type: ignore[arg-type]

    if ifu_indices is None:
        amps_to_do = set(range(n_amp))
    else:
        amps_to_do = {int(i) for i in ifu_indices}

    # Helper to read IFU slot label for a given slice start
    def _ifuslot_for_slice_start(row_index: int) -> str:
        try:
            h5._require_tables()
            with h5._open() as fh:  # type: ignore[attr-defined]
                t = fh.root.Info
                raw = t.cols.ifuslot[row_index]
                if isinstance(raw, (bytes, bytearray)):
                    return raw.decode("utf-8", errors="ignore")
                return str(raw)
        except Exception:
            return f"amp{row_index // fibers_per_amp // exposures:02d}"

    n_written = 0
    n_skipped = 0

    for a, e, s, arrays in h5.iter_amp_blocks(["spectrum"]):
        if a not in amps_to_do:
            continue
        # Only proceed if exposure matches our expectations
        if e < 0 or e >= n_exp:
            continue
        save_path = ifu_plan.fiber_model_path(a, e)
        if resume and save_path.exists():
            n_skipped += 1
            continue

        Y = arrays["spectrum"].astype(float)  # (fibers_per_amp, n_wave)
        if Y.shape[0] != fibers_per_amp or Y.shape[1] != n_wave:
            # shape mismatch; skip gracefully
            n_skipped += 1
            continue

        # Determine amp label (ZERO/DEFECTIVE/GOOD)
        zf = float(zero_table[a, e]) if (zero_table is not None and np.isfinite(zero_table[a, e])) else None
        label = _amp_label(ls, a, e, zf)

        # ZERO invariant: leave outputs as zeros and continue
        if label == "ZERO":
            delta_mult = np.zeros(fibers_per_amp, dtype=float)
            source_mask = np.zeros(fibers_per_amp, dtype=bool)
            notes = "ZERO amp: outputs left at zero; no computation performed"
        else:
            # Provisional per-fiber delta_mult via robust ratio to BW_full
            ref = bw_full[e, wmask]
            with np.errstate(all="ignore"):
                ratio = Y[:, wmask] / np.where(np.abs(ref) > 0, ref, np.nan)
            # Use biweight location across wavelength per fiber; fallback to nanmedian if needed
            delta_mult = np.full(fibers_per_amp, np.nan, dtype=float)
            for f in range(fibers_per_amp):
                r = ratio[f]
                if np.isfinite(r).any():
                    try:
                        delta_mult[f] = float(biweight_location(r, axis=0))
                    except Exception:
                        delta_mult[f] = float(np.nanmedian(r))
                else:
                    delta_mult[f] = np.nan

            # Source masking across fibers within this amp/exp using MAD
            finite = np.isfinite(delta_mult)
            if not np.any(finite):
                med = np.nan
                scale = np.nan
                source_mask = np.zeros(fibers_per_amp, dtype=bool)
            else:
                med = float(np.nanmedian(delta_mult[finite]))
                try:
                    scale = float(1.4826 * mad(delta_mult[finite]))
                except Exception:
                    # fallback MAD
                    scale = float(1.4826 * np.nanmedian(np.abs(delta_mult[finite] - med)))
                thr = max(1e-12, opts.k_source * (scale if np.isfinite(scale) and scale > 0 else 0.0))
                source_mask = np.zeros(fibers_per_amp, dtype=bool)
                if np.isfinite(med) and thr > 0:
                    source_mask = np.abs(delta_mult - med) > thr
                # Replace outlier deltas with median to avoid contaminating downstream application
                delta_mult = np.where(source_mask, med, delta_mult)
                # Ensure finite
                delta_mult = np.where(np.isfinite(delta_mult), delta_mult, 1.0)

            if label == "DEFECTIVE_OR_BRIGHT":
                notes = (
                    "DEFECTIVE_OR_BRIGHT amp: minimal model recommended; "
                    "fiber-level deltas provided for diagnostics only"
                )
            else:
                notes = "GOOD amp: fiber-level deltas computed with source masking"

        # Annotate IFU slot (best-effort)
        ifuslot = _ifuslot_for_slice_start(int(s.start))

        # Persist compact output
        np.savez_compressed(
            save_path,
            ifuslot=np.array(ifuslot),
            amp=np.array(a, dtype=np.int16),
            exp=np.array(e, dtype=np.int16),
            delta_mult=delta_mult.astype(np.float32),
            source_mask=source_mask.astype(np.bool_),
            amp_label=np.array(label),
            notes=np.array(notes),
        )
        n_written += 1

    # Write a small manifest
    manifest = {
        "inputs": [str(h5file)],
        "shot_dir": str(out.resolve()),
        "ifu_dir": str(ifu_out.resolve()),
        "n_written": int(n_written),
        "n_skipped": int(n_skipped),
        "options": {
            "wave_mask_frac": float(opts.wave_mask_frac),
            "k_source": float(opts.k_source),
        },
    }
    with ifu_plan.manifest_path().open("w", encoding="utf-8") as f:
        import json
        json.dump(manifest, f, indent=2)

    return manifest
