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
    make_plots: bool = False     # write example IFU diagnostic plots
    max_plots: int = 6           # maximum number of example plots to write

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wave_mask_frac": float(self.wave_mask_frac),
            "k_source": float(self.k_source),
            "make_plots": bool(self.make_plots),
            "max_plots": int(self.max_plots),
        }

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IFUOptions":
        keys = {"wave_mask_frac", "k_source", "make_plots", "max_plots"}
        data = {k: d[k] for k in d if k in keys}
        obj = cls(**data)  # type: ignore[arg-type]
        obj.validate()
        return obj

    @classmethod
    def from_json(cls, s: str) -> "IFUOptions":
        import json
        return cls.from_dict(json.loads(s))

    def validate(self) -> None:
        if not (0.0 < float(self.wave_mask_frac) <= 1.0):
            raise ValueError("wave_mask_frac must be in (0, 1]")
        if not (float(self.k_source) > 0):
            raise ValueError("k_source must be positive")
        if not (isinstance(self.max_plots, int) and self.max_plots >= 0):
            raise ValueError("max_plots must be a non-negative integer")


def _wave_mask(n_wave: int, frac: float) -> np.ndarray:
    frac = float(np.clip(frac, 0.05, 1.0))
    n = int(max(1, round(frac * int(n_wave))))
    start = (int(n_wave) - n) // 2
    end = start + n
    w = np.zeros(int(n_wave), dtype=bool)
    w[start:end] = True
    return w


def _latest_labelset(outdir: Path) -> Any | None:
    """Deprecated internal: delegate to qc.labels.discover_latest_snapshot."""
    try:
        from ..qc.labels import discover_latest_snapshot
        return discover_latest_snapshot(outdir)
    except Exception:
        return None


def _load_shot_artifacts(outdir: Path) -> Dict[str, np.ndarray]:
    def _load_npz(p: Path) -> Dict[str, np.ndarray]:
        with np.load(p) as d:  # type: ignore[no-untyped-call]
            return {k: d[k] for k in d.files}

    # Resolve paths via ShotPlan with semantic keys; fall back to legacy if needed
    try:
        from ..shot.plan import ShotPlan
        sp = ShotPlan(outdir).paths()
        p_bw_amp = sp.get("Build_Amplifier_Robust_Spectra") or sp.get("stage_01_bw_amp")
        p_bw_full = sp.get("Build_Full_Exposure_Sky") or sp.get("stage_02_bw_full")
        p_mult = sp.get("Fit_Multiplicative_Scale") or sp.get("stage_04_mult")
    except Exception:
        p_bw_amp = outdir / "stage_01_bw_amp.npz"
        p_bw_full = outdir / "stage_02_bw_full.npz"
        p_mult = outdir / "stage_04_mult.npz"

    bw_amp = _load_npz(p_bw_amp).get("bw_amp") if p_bw_amp and p_bw_amp.exists() else None
    bw_full = _load_npz(p_bw_full).get("bw_full") if p_bw_full and p_bw_full.exists() else None
    mult = _load_npz(p_mult).get("mult_scale") if p_mult and p_mult.exists() else None
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
    bw_amp = artifacts["bw_amp"]    # (n_amp, n_exp, n_wave)
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
    plots_written = 0

    for a, e, s, arrays in h5.iter_amp_blocks(["spectrum", "skyspectrum"]):
        if a not in amps_to_do:
            continue
        # Only proceed if exposure matches our expectations
        if e < 0 or e >= n_exp:
            continue
        save_path = ifu_plan.fiber_model_path(a, e)
        if resume and save_path.exists():
            n_skipped += 1
            continue

        # Construct total flux per fiber: spectrum + skyspectrum
        spec = arrays["spectrum"].astype(float)
        sky = arrays["skyspectrum"].astype(float)
        Y = spec + sky  # (fibers_per_amp, n_wave)
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
            # Provisional per-fiber delta_mult via robust ratio to the amp's expected sky
            # Use exposure-level baseline scaled by the amp's multiplicative factor to avoid
            # trivial self-division that can collapse to ones.
            mult = artifacts["mult"][a, e]
            ref = bw_full[e, wmask] * (mult if np.isfinite(mult) else np.nan)
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
                # No finite per-fiber estimates; fall back to unity so plots are not blank
                delta_mult = np.ones(fibers_per_amp, dtype=float)
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

    # Aggregate per-IFU (4 amps per IFU) outputs into 448-length vectors per exposure
    # We will load any missing per-amp results from disk (e.g., when --resume skipped computation).
    def _load_amp_npz(a_idx: int, e_idx: int) -> Optional[dict]:
        p = ifu_plan.fiber_model_path(a_idx, e_idx)
        if not p.exists():
            return None
        try:
            with np.load(p) as d:  # type: ignore[no-untyped-call]
                return {k: d[k] for k in d.files}
        except Exception:
            return None

    ifu_written = 0
    n_ifu = (n_amp + 3) // 4  # assume 4 amps per IFU in order
    for ifu_idx in range(n_ifu):
        for e_idx in range(n_exp):
            # collect four amps for this IFU
            parts_delta: list[np.ndarray] = []
            parts_mask: list[np.ndarray] = []
            amp_labels: list[str] = []
            amp_ifuslots: list[str] = []
            notes_list: list[str] = []
            complete = True
            for k in range(4):
                a_idx = ifu_idx * 4 + k
                if a_idx >= n_amp:
                    complete = False
                    break
                data = _load_amp_npz(a_idx, e_idx)
                if data is None:
                    complete = False
                    break
                dm = np.array(data.get("delta_mult", np.full((fibers_per_amp,), np.nan)), dtype=float)
                sm = np.array(data.get("source_mask", np.zeros((fibers_per_amp,), dtype=bool)), dtype=bool)
                parts_delta.append(dm)
                parts_mask.append(sm)
                amp_labels.append(str(np.array(data.get("amp_label", ""))))
                amp_ifuslots.append(str(np.array(data.get("ifuslot", f"a{a_idx:02d}"))))
                notes_list.append(str(np.array(data.get("notes", ""))))
            if not complete:
                continue
            # Concatenate into 448-length vectors: [amp0 fibers | amp1 | amp2 | amp3]
            delta_448 = np.concatenate(parts_delta, axis=0).astype(np.float32)
            mask_448 = np.concatenate(parts_mask, axis=0).astype(np.bool_)
            # Save per-IFU file
            save_ifu = ifu_plan.ifu_model_path(ifu_idx, e_idx)
            np.savez_compressed(
                save_ifu,
                ifu=np.array(ifu_idx, dtype=np.int16),
                exp=np.array(e_idx, dtype=np.int16),
                delta_mult=delta_448,
                source_mask=mask_448,
                amp_labels=np.array(amp_labels),
                amp_ifuslots=np.array(amp_ifuslots),
                notes=np.array(" | ".join([n for n in notes_list if n])),
            )
            ifu_written += 1

        # After assembling all exposures for this IFU, optionally create a per-IFU plot with one line per exposure
        if opts.make_plots and plots_written < int(opts.max_plots):
            try:
                import matplotlib.pyplot as plt  # type: ignore
                colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
                plot_path = ifu_plan.ifu_plot_path(ifu_idx)
                plot_path.parent.mkdir(parents=True, exist_ok=True)
                fig, ax = plt.subplots(figsize=(10, 3.2))
                x = np.arange(fibers_per_amp * 4)
                have_any = False
                for e_idx in range(n_exp):
                    p = ifu_plan.ifu_model_path(ifu_idx, e_idx)
                    if not p.exists():
                        continue
                    with np.load(p) as d:  # type: ignore[no-untyped-call]
                        y = np.array(d.get("delta_mult", np.full((fibers_per_amp*4,), np.nan)), dtype=float)
                        m = np.array(d.get("source_mask", np.zeros((fibers_per_amp*4,), dtype=bool)), dtype=bool)
                    if not np.isfinite(y).any():
                        continue
                    have_any = True
                    ax.plot(x, y, color=colors[e_idx % len(colors)], lw=1.2, alpha=0.9, label=f"exp {e_idx}")
                    if np.any(m):
                        # Mark masked fibers lightly on top
                        ax.scatter(x[m], y[m], facecolors="none", edgecolors=colors[e_idx % len(colors)], s=12, linewidths=0.8, alpha=0.7)
                title = f"IFU {ifu_idx:03d} — delta_mult per fiber (lines per exposure)"
                ax.set_title(title)
                ax.set_xlabel("fiber index (4 amps × 112)")
                ax.set_ylabel("delta_mult")
                ax.grid(True, alpha=0.2)
                if have_any:
                    ax.legend(loc="upper right", fontsize=8, ncol=min(4, n_exp))
                    # Ensure y-limits non-degenerate
                    try:
                        ymins = [] ; ymaxs = []
                        for line in ax.get_lines():
                            yd = line.get_ydata()
                            yfin = yd[np.isfinite(yd)]
                            if yfin.size:
                                ymins.append(float(np.nanmin(yfin)))
                                ymaxs.append(float(np.nanmax(yfin)))
                        if ymins and ymaxs:
                            ymin, ymax = min(ymins), max(ymaxs)
                            if not (ymax > ymin):
                                pad = 0.05 * max(1.0, abs(ymax) if np.isfinite(ymax) else 1.0)
                                ax.set_ylim(ymin - pad, ymax + pad)
                    except Exception:
                        pass
                fig.savefig(plot_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                plots_written += 1
            except Exception:
                pass

    # Write a small manifest
    manifest = {
        "inputs": [str(h5file)],
        "shot_dir": str(out.resolve()),
        "ifu_dir": str(ifu_out.resolve()),
        "plots_dir": str(getattr(ifu_plan, "plots_dir", ifu_out / "plots")),
        "n_written": int(n_written),
        "n_skipped": int(n_skipped),
        "ifu_written": int(ifu_written),
        "plots_written": int(plots_written),
        "options": {
            "wave_mask_frac": float(opts.wave_mask_frac),
            "k_source": float(opts.k_source),
            "make_plots": bool(opts.make_plots),
            "max_plots": int(opts.max_plots),
        },
    }
    with ifu_plan.manifest_path().open("w", encoding="utf-8") as f:
        import json
        json.dump(manifest, f, indent=2)

    return manifest
