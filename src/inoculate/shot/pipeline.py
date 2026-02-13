"""High-level pipeline entry point for processing a single shot.

Implements the staged SingleShot workflow with resumable artifacts.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from ..io.h5 import H5VIRUS
from ..io.write import write_parquet
from ..model.base import ModelSpec
from ..qc.features import compute_amp_features
from ..qc.labels import label_amps, LabelSet, save_labelset, load_labelset
from ..robust import biweight_location
from ..sky.mult.build import build_mult_scale
from ..sky.mult import build_mult_poly2d
from ..sky.pca.build import build_shot_pca
from ..sky.additive import build_amp_poly
from ..robust import robust_linear_least_squares
from ..stats.aggregate import compute_bw_amp, compute_bw_full
from ..utils import SchemaError

logger = logging.getLogger(__name__)


def _wave_mask(n_wave: int, frac: float) -> np.ndarray:
    """Return a fixed-index wavelength mask using [40 : n-25].

    This overrides the previous central-fraction behavior. The mask is used for
    modeling (multiplicative estimation, PCA, and additive polynomial fits) but
    residuals may still be computed and reported over the full wavelength grid.

    Args:
        n_wave: Number of wavelength samples.
        frac: Unused (kept for signature compatibility).

    Returns:
        Boolean array of shape (n_wave,), True for indices in [40, n_wave-25).
        If ``n_wave <= 65`` the function falls back to an all-True mask.
    """
    # Guard for very short spectra
    if n_wave <= 65:
        return np.ones(n_wave, dtype=bool)
    start = 40
    stop = max(start, n_wave - 25)
    mask = np.zeros(n_wave, dtype=bool)
    mask[start:stop] = True
    return mask


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}


def run_shot(
    h5file: str,
    outdir: str | Path,
    modelspec: ModelSpec | None = None,
    *,
    resume: bool = True,
    make_plots: bool = False,
    max_plots: int = 12,
) -> Dict[str, Any]:
    """Run the Inoculate pipeline for a single shot.

    Args:
        h5file: Path to the VIRUS spectral HDF5 file.
        outdir: Directory to write stage artifacts.
        modelspec: Model specification; if None, uses defaults.
        resume: If True, skip stages whose artifacts already exist.

    Returns:
        Mapping with high-level outputs including the manifest path.
    """
    # Avoid noisy NumPy warnings from intentional NaN propagation in robust logic
    np.seterr(invalid="ignore", divide="ignore")
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    ms = modelspec or ModelSpec()

    # Centralized stage paths via ShotPlan
    from .plan import ShotPlan
    plan = ShotPlan(out)
    paths = plan.paths()

    # Stage 00 — Info + schema validation
    stage00 = paths["stage_00_info"]
    if not (resume and stage00.exists()):
        logger.info("[Stage 00] Info + schema validation")
        h5 = H5VIRUS(h5file)
        info = h5.read_info()  # may raise SchemaError
        with stage00.open("w", encoding="utf-8") as f:
            json.dump({k: (int(v) if isinstance(v, np.integer) else (v.tolist() if hasattr(v, 'dtype') else v)) for k, v in info.items()}, f, indent=2)
    else:
        with stage00.open("r", encoding="utf-8") as f:
            info = json.load(f)

    n_amp = int(info["n_amp"])  # type: ignore[index]
    n_exp = int(info["exposures"])  # type: ignore[index]
    n_wave = int(info["n_wave"])  # type: ignore[index]

    # Construct wavelength mask
    wmask = _wave_mask(n_wave, ms.wave_mask_frac)

    # Stage 01 — Compute BW_amp
    stage01 = paths["stage_01_bw_amp"]
    if not (resume and stage01.exists()):
        logger.info("[Stage 01] Compute BW_amp")
        h5 = H5VIRUS(h5file)
        bw_amp = compute_bw_amp(h5)
        np.savez_compressed(stage01, bw_amp=bw_amp)
    else:
        bw_amp = _load_npz(stage01)["bw_amp"]

    # Stage 02 — Compute BW_full
    stage02 = paths["stage_02_bw_full"]
    if not (resume and stage02.exists()):
        logger.info("[Stage 02] Compute BW_full")
        bw_full = compute_bw_full(bw_amp)
        np.savez_compressed(stage02, bw_full=bw_full)
    else:
        bw_full = _load_npz(stage02)["bw_full"]

    # Stage 03 — QC features + labeling (initialize LabelSet)
    stage03 = paths["stage_03_qc"]
    stage03_labels = paths["stage_03_labels"]
    if not (resume and stage03.exists()):
        logger.info("[Stage 03] QC features + labeling")
        df_feat = compute_amp_features(bw_amp, bw_full, wmask)
        # Derive labels (per-amp summary) for visibility in the parquet
        df_labels, _good_mask_tmp = label_amps(df_feat)
        # Merge per-amp label back to rows for visibility
        df = df_feat.merge(df_labels, on="amp", how="left")
        write_parquet(df, stage03)
        # Initialize iterative LabelSet and save snapshot
        ls = LabelSet.from_features(df_feat)
        save_labelset(stage03_labels, ls)
    else:
        # On resume, build or load LabelSet from existing parquet
        import pandas as pd  # local import to avoid mandatory dep if unused
        df = pd.read_parquet(stage03)
        if stage03_labels.exists():
            ls = load_labelset(stage03_labels)
        else:
            ls = LabelSet.from_features(df)
            save_labelset(stage03_labels, ls)
    # Use LabelSet to derive conservative good_mask for subsequent stages
    good_mask = ls.good_mask()

    # Stage 04 — Build mult_scale
    stage04 = paths["stage_04_mult"]
    stage04_labels = paths["stage_04_labels"]
    if not (resume and stage04.exists()):
        logger.info("[Stage 04] Build multiplicative mult_scale")
        mult = build_mult_scale(bw_amp, bw_full, good_mask, wmask, bounds=ms.mult_bounds)
        np.savez_compressed(stage04, mult_scale=mult)
    else:
        mult = _load_npz(stage04)["mult_scale"]
    # Update labels with mult diagnostics and save snapshot (idempotent on resume)
    if not stage04_labels.exists():
        ls.update_with_mult(mult, bounds=ms.mult_bounds)
        save_labelset(stage04_labels, ls)

    # Stage 04.25 — Fit 2D polynomial model to mult vs. sky position (per exposure)
    stage0425 = paths["stage_0425_mult_poly2d"]
    stage0425_labels = paths["stage_0425_labels"]
    if not (resume and stage0425.exists()):
        logger.info("[Stage 04.25] Fit 2D polynomial to mult (per exposure)")
        h5 = H5VIRUS(h5file)
        poly2d = build_mult_poly2d(
            h5,
            mult,
            degree=3,
            loss=ms.robust_loss,
            huber_delta=ms.huber_delta,
            tukey_c=ms.tukey_c,
        )
        np.savez_compressed(
            stage0425,
            degree=int(poly2d.degree),
            coeffs=poly2d.coeffs,
            ra_amp=poly2d.ra_amp,
            dec_amp=poly2d.dec_amp,
            x=poly2d.x,
            y=poly2d.y,
            pred=poly2d.pred,
        )
        # Update labels with poly2d residuals and save snapshot
        ls.update_with_poly2d(mult, poly2d.pred)
        if not stage0425_labels.exists():
            save_labelset(stage0425_labels, ls)
    else:
        # On resume, if labels snapshot missing, load poly2d outputs and update
        if not stage0425_labels.exists():
            data0425 = _load_npz(stage0425)
            if "pred" in data0425:
                ls.update_with_poly2d(mult, data0425["pred"])
                save_labelset(stage0425_labels, ls)

    # Stage 04.5 — Fit shared additive polynomial per amp on residual_1
    stage045 = paths["stage_045_poly"]
    if not (resume and stage045.exists()):
        logger.info("[Stage 04.5] Fit shared additive polynomial per amplifier")
        beta_all = build_amp_poly(
            bw_amp,
            bw_full,
            mult,
            wmask,
            order=ms.poly_order,
            loss=ms.robust_loss,
            huber_delta=ms.huber_delta,
            tukey_c=ms.tukey_c,
        )
        np.savez_compressed(stage045, poly_beta=beta_all)
    else:
        beta_all = _load_npz(stage045)["poly_beta"]

    # Stage 05 — Build PCA basis
    stage05 = paths["stage_05_pca"]
    if not (resume and stage05.exists()):
        logger.info("[Stage 05] Build shot PCA basis")
        pca = build_shot_pca(
            bw_amp,
            bw_full,
            mult,
            good_mask,
            wmask,
            beta_all,
            n_components=ms.n_pca,
            loss=ms.robust_loss,
            huber_delta=ms.huber_delta,
            tukey_c=ms.tukey_c,
        )
        np.savez_compressed(stage05, **pca)
    else:
        pca = _load_npz(stage05)

    # Stage 06 — Per-amp fits (use Stage 04.5 poly + per-exp PCA coeffs)
    stage06 = paths["stage_06_amp_fits"]
    if not (resume and stage06.exists()):
        logger.info("[Stage 06] Fit amplifiers: shared poly + PCA coeffs")
        df_fits = _fit_all_amps(
            bw_amp,
            bw_full,
            mult,
            pca,
            wmask,
            beta_all,
            loss=ms.robust_loss,
            huber_delta=ms.huber_delta,
            tukey_c=ms.tukey_c,
        )
        write_parquet(df_fits, stage06)

    # Stage 07 — Manifest
    stage07 = paths["stage_07_manifest"]
    if not (resume and stage07.exists()):
        logger.info("[Stage 07] Write model-start manifest")
        manifest = {
            "inputs": [str(h5file)],
            "outputs": [
                str(p)
                for p in [
                    stage00,
                    stage01,
                    stage02,
                    stage03,
                    stage04,
                    stage0425,
                    stage045,
                    stage05,
                    stage06,
                    stage07,
                ]
            ],
            "modelspec": ms.to_dict(),
            "work_dir": str(out.resolve()),
        }
        with stage07.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    # Optional diagnostics plotting
    if make_plots:
        try:
            from ..plot import plot_bw_amp_vs_full, plot_mult_by_amp, plot_fit_example  # type: ignore
            plots_dir = out / "plots"
            plots_dir.mkdir(exist_ok=True)
            # 1) Multiplicative summary across all amps
            plot_mult_by_amp(out, highlight_outliers=True, show_labels=True, save=plots_dir / "mult_by_amp.png")
            # 2) Choose a representative amp (first good, else 0) and exposure 0
            a_sel = int(np.where(good_mask)[0][0]) if np.any(good_mask) else 0
            e_sel = 0
            # 3) Single-amp comparison and fit example
            plot_bw_amp_vs_full(out, amp=a_sel, exp=e_sel, save=plots_dir / f"bw_amp_vs_full_a{a_sel}_e{e_sel}.png")
            plot_fit_example(out, amp=a_sel, exp=e_sel, save=plots_dir / f"fit_example_a{a_sel}_e{e_sel}.png")
        except Exception as exc:  # pragma: no cover
            logger.warning("Plot generation failed: %s", exc)

    return {
        "status": "ok",
        "outdir": str(out),
        "manifest": str(stage07),
    }


def _fit_all_amps(
    bw_amp: np.ndarray,
    bw_full: np.ndarray,
    mult: np.ndarray,
    pca: Dict[str, np.ndarray],
    wave_mask: np.ndarray,
    poly_beta: np.ndarray,
    *,
    loss: str = "huber",
    huber_delta: float = 1.0,
    tukey_c: float = 4.685,
) -> "pd.DataFrame":
    """Fit per-exp PCA coefficients for all amps using prefit additive poly with robust loss.

    Forms residual_2 on the modeling mask as:
        residual_2 = (bw_amp/mult) - bw_full - poly(amp) - mu
    where mu is the PCA mean computed on the same masked grid. Projects this
    centered residual onto the PCA eigenvectors to obtain coefficients. The
    per-amp polynomial coefficients from Stage 04.5 are passed through to the
    output parquet.

    Args:
        bw_amp: (n_amp, n_exp, n_wave) biweight amplifier spectra.
        bw_full: (n_exp, n_wave) biweight full-shot spectra.
        mult: (n_amp, n_exp) multiplicative scales.
        pca: Mapping with keys 'pca_mean' and 'pca_evecs'.
        wave_mask: Boolean mask (n_wave,) used for modeling.
        poly_beta: (n_amp, n_poly) per-amp polynomial coefficients from Stage 04.5.

    Returns:
        DataFrame: one row per amplifier with columns poly_c*, c_e{e}_k{k}.
    """
    import pandas as pd

    n_amp, n_exp, n_wave = bw_amp.shape
    wm = wave_mask.astype(bool)
    # Build wavelength grid normalized to [-1,1] for numerical stability
    x = np.linspace(-1.0, 1.0, n_wave)[wm]
    n_poly = poly_beta.shape[1] if poly_beta.ndim == 2 else 0
    P = np.vstack([x ** i for i in range(n_poly)]).T if n_poly > 0 else np.zeros((wm.sum(), 0))

    evecs = pca["pca_evecs"]  # (n_exp, n_comp, n_wave)
    mu = pca["pca_mean"]
    n_comp = evecs.shape[1]

    rows = []
    for a in range(n_amp):
        coeffs_pca = np.zeros((n_exp, n_comp), dtype=float)
        # Precompute poly curve on mask for this amp
        poly_curve = (P @ poly_beta[a]) if n_poly > 0 else np.zeros(P.shape[0])
        for e in range(n_exp):
            with np.errstate(all="ignore"):
                # residual_2 centered: (bw_amp/mult) - bw_full - poly - mu
                y = (
                    bw_amp[a, e, wm] / np.clip(mult[a, e], 1e-6, np.inf)
                    - bw_full[e, wm]
                    - poly_curve
                    - mu[e, wm]
                )
            V_full = evecs[e, :, :]
            V = V_full[:, wm]  # (n_comp, nw_mask)
            # Design matrix for masked wavelengths is A=(V^T), so that A @ c ≈ y
            A = V.T  # (nw_mask, n_comp)
            c, _ = robust_linear_least_squares(
                A,
                np.nan_to_num(y, nan=0.0),
                loss=loss,
                delta=huber_delta,
                c=tukey_c,
            )
            # Ensure shape (n_comp,)
            c = np.asarray(c).reshape(-1)
            coeffs_pca[e, :] = c[:n_comp]
        # Build row
        row: Dict[str, float | int] = {"amp": a}
        for i in range(n_poly):
            row[f"poly_c{i}"] = float(poly_beta[a, i])
        for e in range(n_exp):
            for k in range(n_comp):
                row[f"c_e{e}_k{k}"] = float(coeffs_pca[e, k])
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("amp").reset_index(drop=True)
    return df
