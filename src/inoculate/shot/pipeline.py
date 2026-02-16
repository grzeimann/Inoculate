"""High-level pipeline entry point for processing a single shot.

Mathematical overview
---------------------
We model the per-amplifier, per-exposure sky signal using shared exposure-level
references and low-parameter corrections. Key arrays:
  - BW_amp[a,e,w]: robust per-amplifier spectra (biweight over 112 fibers).
  - BW_full[e,w]: robust per-exposure spectra (biweight over amplifiers).
  - mult[a,e]: per-(amp,exp) multiplicative scale from "Fit_Multiplicative_Scale".
  - mult_hat[a,e]: 2D polynomial field prediction of mult vs. (x,y) from
    "Fit_Poly2D_Field_Model" (stored as key "pred" with shape (n_amp, n_exp)).

Conceptually, the modeled amplifier spectrum is
  Model_amp[a,e,w] = mult_hat[a,e] * BW_full[e,w] + Poly1D_a[a](w) + PCA_e[e](w; c[a,e,:])
where Poly1D_a is a per-amp additive polynomial (shared across exposures) and
PCA_e are exposure-specific additive components with per-amp coefficients c.

Amp-fit residuals (Stage 06) are formed on the wavelength mask W as
  residual[a,e,W] = (BW_amp[a,e,W] / mult_hat[a,e]) - BW_full[e,W] - Poly1D_a[a](W) - mu_e[W]
We then solve for c[a,e,:] by projecting these centered residuals onto the PCA
basis V_e[:,W] with a robust linear least-squares. (Note: the implementation may
fall back to using mult[a,e] for the division step; the Poly2D intent is shown
here to reflect the target formulation.)

Pipeline structure
------------------
This module implements the staged SingleShot workflow with resumable artifacts.
Artifacts are addressed via ShotPlan using semantic keys (and legacy fallbacks).
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


# ---- Minimal Stage abstraction (Phase B) ----
from dataclasses import dataclass


@dataclass
class _Context:
    h5file: str
    outdir: Path
    ms: ModelSpec
    plan: Any
    # ephemeral/stateful fields propagated between stages
    info: Dict[str, Any] | None = None
    wmask: np.ndarray | None = None
    bw_amp: np.ndarray | None = None
    bw_full: np.ndarray | None = None
    ls: Any | None = None
    mult: np.ndarray | None = None
    poly2d_pred: np.ndarray | None = None


class _Stage:
    key: str = ""
    produces: tuple[str, ...] = ()

    def should_run(self, ctx: _Context, resume: bool) -> bool:
        # Run if any produced artifact is missing, else skip when resume=True
        paths = ctx.plan.paths()
        if not self.produces:
            return True
        missing = [k for k in self.produces if (paths.get(k) is None or not paths[k].exists())]
        return (len(missing) > 0) or (not resume)

    def run(self, ctx: _Context) -> None:  # pragma: no cover - overridden per stage
        raise NotImplementedError


def _execute_stages(stages: list[_Stage], ctx: _Context, *, resume: bool = True) -> None:
    for st in stages:
        try:
            if st.should_run(ctx, resume):
                logger.info("[%s] start", st.key or st.__class__.__name__)
                st.run(ctx)
            else:
                logger.info("[%s] resume: artifacts present, skipping", st.key or st.__class__.__name__)
        except Exception:
            logger.exception("Stage failed: %s", st.key or st.__class__.__name__)
            raise


class _Stage00Info(_Stage):
    key = "Validate_Input_Shot_Info"
    produces = ("Validate_Input_Shot_Info",)

    def run(self, ctx: _Context) -> None:
        from .plan import ShotPlan
        paths = ctx.plan.paths() if hasattr(ctx.plan, "paths") else ShotPlan(ctx.outdir).paths()
        p = paths["Validate_Input_Shot_Info"]
        h5 = H5VIRUS(ctx.h5file)
        info = h5.read_info()
        with p.open("w", encoding="utf-8") as f:
            json.dump({k: (int(v) if isinstance(v, np.integer) else (v.tolist() if hasattr(v, 'dtype') else v)) for k, v in info.items()}, f, indent=2)
        ctx.info = info
        # build wavelength mask
        n_wave = int(info["n_wave"])  # type: ignore[index]
        ctx.wmask = _wave_mask(n_wave, ctx.ms.wave_mask_frac)
        try:
            idx = np.flatnonzero(ctx.wmask)
            if idx.size > 0:
                logger.info("[Stage 00] Wavelength mask frac=%.2f -> [%d:%d] (%.3f)", float(ctx.ms.wave_mask_frac), int(idx[0]), int(idx[-1])+1, float(ctx.wmask.mean()))
        except Exception:
            pass


class _Stage01BwAmp(_Stage):
    key = "Build_Amplifier_Robust_Spectra"
    produces = ("Build_Amplifier_Robust_Spectra",)

    def run(self, ctx: _Context) -> None:
        from .plan import ShotPlan
        paths = ctx.plan.paths() if hasattr(ctx.plan, "paths") else ShotPlan(ctx.outdir).paths()
        p = paths["Build_Amplifier_Robust_Spectra"]
        h5 = H5VIRUS(ctx.h5file)
        bw_amp = compute_bw_amp(h5)
        np.savez_compressed(p, bw_amp=bw_amp)
        ctx.bw_amp = bw_amp


class _Stage02BwFull(_Stage):
    key = "Build_Full_Exposure_Sky"
    produces = ("Build_Full_Exposure_Sky",)

    def run(self, ctx: _Context) -> None:
        from .plan import ShotPlan
        paths = ctx.plan.paths() if hasattr(ctx.plan, "paths") else ShotPlan(ctx.outdir).paths()
        p_in = paths["Build_Amplifier_Robust_Spectra"]
        p_out = paths["Build_Full_Exposure_Sky"]
        bw_amp = ctx.bw_amp if ctx.bw_amp is not None else _load_npz(p_in)["bw_amp"]
        bw_full = compute_bw_full(bw_amp)
        np.savez_compressed(p_out, bw_full=bw_full)
        ctx.bw_full = bw_full


class _Stage03QCInitLabels(_Stage):
    key = "Compute_QC_Features"
    produces = ("Compute_QC_Features", "Initialize_Iterative_Labels")

    def run(self, ctx: _Context) -> None:
        from .plan import ShotPlan
        paths = ctx.plan.paths() if hasattr(ctx.plan, "paths") else ShotPlan(ctx.outdir).paths()
        p_bw_amp = paths["Build_Amplifier_Robust_Spectra"]
        p_bw_full = paths["Build_Full_Exposure_Sky"]
        p_qc = paths["Compute_QC_Features"]
        p_labels = paths["Initialize_Iterative_Labels"]
        bw_amp = ctx.bw_amp if ctx.bw_amp is not None else _load_npz(p_bw_amp)["bw_amp"]
        bw_full = ctx.bw_full if ctx.bw_full is not None else _load_npz(p_bw_full)["bw_full"]
        wmask = ctx.wmask if ctx.wmask is not None else _wave_mask(int(bw_full.shape[1]), ctx.ms.wave_mask_frac)
        df_feat = compute_amp_features(bw_amp, bw_full, wmask)
        df_labels, _gm = label_amps(df_feat)
        df = df_feat.merge(df_labels, on="amp", how="left")
        write_parquet(df, p_qc)
        ls = LabelSet.from_features(df_feat)
        save_labelset(p_labels, ls)
        ctx.ls = ls


class _Stage04Mult(_Stage):
    key = "Fit_Multiplicative_Scale"
    produces = ("Fit_Multiplicative_Scale", "Labels_After_Mult")

    def run(self, ctx: _Context) -> None:
        from .plan import ShotPlan
        paths = ctx.plan.paths() if hasattr(ctx.plan, "paths") else ShotPlan(ctx.outdir).paths()
        p_bw_amp = paths["Build_Amplifier_Robust_Spectra"]
        p_bw_full = paths["Build_Full_Exposure_Sky"]
        p_out = paths["Fit_Multiplicative_Scale"]
        p_labels = paths["Labels_After_Mult"]
        bw_amp = ctx.bw_amp if ctx.bw_amp is not None else _load_npz(p_bw_amp)["bw_amp"]
        bw_full = ctx.bw_full if ctx.bw_full is not None else _load_npz(p_bw_full)["bw_full"]
        wmask = ctx.wmask if ctx.wmask is not None else _wave_mask(int(bw_full.shape[1]), ctx.ms.wave_mask_frac)
        ls = ctx.ls if ctx.ls is not None else None
        good_mask = (ls.good_mask() if ls is not None else np.ones(bw_amp.shape[0], dtype=bool))
        mult = build_mult_scale(bw_amp, bw_full, good_mask, wmask, bounds=ctx.ms.mult_bounds)
        np.savez_compressed(p_out, mult_scale=mult)
        if ls is None:
            # lazy init from parquet if needed
            import pandas as pd
            df = pd.read_parquet(paths["Compute_QC_Features"])
            ls = LabelSet.from_features(df)
        ls.update_with_mult(mult, bounds=ctx.ms.mult_bounds)
        if not p_labels.exists():
            save_labelset(p_labels, ls)
        ctx.ls = ls
        ctx.mult = mult


class _Stage0425Poly2D(_Stage):
    key = "Fit_Poly2D_Field_Model"
    produces = ("Fit_Poly2D_Field_Model", "Labels_After_Poly2D")

    def run(self, ctx: _Context) -> None:
        from .plan import ShotPlan
        paths = ctx.plan.paths() if hasattr(ctx.plan, "paths") else ShotPlan(ctx.outdir).paths()
        p_out = paths["Fit_Poly2D_Field_Model"]
        p_labels = paths["Labels_After_Poly2D"]
        # ensure mult is available
        if ctx.mult is None:
            mult = _load_npz(paths["Fit_Multiplicative_Scale"]) ["mult_scale"]
        else:
            mult = ctx.mult
        h5 = H5VIRUS(ctx.h5file)
        poly2d = build_mult_poly2d(
            h5,
            mult,
            degree=int(ctx.ms.poly_degree),
            loss=ctx.ms.robust_loss,
            huber_delta=ctx.ms.huber_delta,
            tukey_c=ctx.ms.tukey_c,
        )
        np.savez_compressed(
            p_out,
            degree=int(poly2d.degree),
            coeffs=poly2d.coeffs,
            ra_amp=poly2d.ra_amp,
            dec_amp=poly2d.dec_amp,
            x=poly2d.x,
            y=poly2d.y,
            pred=poly2d.pred,
        )
        # update labels
        ls = ctx.ls if ctx.ls is not None else None
        if ls is None:
            # load latest labels if not in ctx
            from ..qc.labels import discover_latest_snapshot
            ls = discover_latest_snapshot(ctx.outdir) or LabelSet.initialize(mult.shape[0], mult.shape[1])
        ls.update_with_poly2d(mult, poly2d.pred)
        if not p_labels.exists():
            save_labelset(p_labels, ls)
        ctx.ls = ls
        ctx.poly2d_pred = poly2d.pred


class _Stage045Poly(_Stage):
    key = "Build_Additive_Polynomial"
    produces = ("Build_Additive_Polynomial",)

    def run(self, ctx: _Context) -> None:
        from .plan import ShotPlan
        paths = ctx.plan.paths() if hasattr(ctx.plan, "paths") else ShotPlan(ctx.outdir).paths()
        p_out = paths["Build_Additive_Polynomial"]
        # Load required inputs
        bw_amp = ctx.bw_amp if ctx.bw_amp is not None else _load_npz(paths["Build_Amplifier_Robust_Spectra"])["bw_amp"]
        bw_full = ctx.bw_full if ctx.bw_full is not None else _load_npz(paths["Build_Full_Exposure_Sky"])["bw_full"]
        mult = ctx.mult if ctx.mult is not None else _load_npz(paths["Fit_Multiplicative_Scale"])["mult_scale"]
        wmask = ctx.wmask if ctx.wmask is not None else _wave_mask(int(bw_full.shape[1]), ctx.ms.wave_mask_frac)
        logger.info("[Build_Additive_Polynomial] Fit shared additive polynomial per amplifier")
        beta_all = build_amp_poly(
            bw_amp,
            bw_full,
            mult,
            wmask,
            order=ctx.ms.poly_order,
            loss=ctx.ms.robust_loss,
            huber_delta=ctx.ms.huber_delta,
            tukey_c=ctx.ms.tukey_c,
        )
        np.savez_compressed(p_out, poly_beta=beta_all)


class _Stage05PCA(_Stage):
    key = "Build_PCA_Components"
    produces = ("Build_PCA_Components",)

    def run(self, ctx: _Context) -> None:
        from .plan import ShotPlan
        paths = ctx.plan.paths() if hasattr(ctx.plan, "paths") else ShotPlan(ctx.outdir).paths()
        p_out = paths["Build_PCA_Components"]
        # Inputs
        bw_amp = ctx.bw_amp if ctx.bw_amp is not None else _load_npz(paths["Build_Amplifier_Robust_Spectra"])["bw_amp"]
        bw_full = ctx.bw_full if ctx.bw_full is not None else _load_npz(paths["Build_Full_Exposure_Sky"])["bw_full"]
        mult = ctx.mult if ctx.mult is not None else _load_npz(paths["Fit_Multiplicative_Scale"])["mult_scale"]
        wmask = ctx.wmask if ctx.wmask is not None else _wave_mask(int(bw_full.shape[1]), ctx.ms.wave_mask_frac)
        beta_all = _load_npz(paths["Build_Additive_Polynomial"]) ["poly_beta"]
        # good_mask from labels if available
        from ..qc.labels import discover_latest_snapshot
        ls = ctx.ls if ctx.ls is not None else (discover_latest_snapshot(ctx.outdir) or None)
        good_mask = (ls.good_mask() if ls is not None else np.ones(bw_amp.shape[0], dtype=bool))
        logger.info("[Build_PCA_Components] Build shot PCA basis")
        pca = build_shot_pca(
            bw_amp,
            bw_full,
            mult,
            good_mask,
            wmask,
            beta_all,
            n_components=ctx.ms.n_pca,
            loss=ctx.ms.robust_loss,
            huber_delta=ctx.ms.huber_delta,
            tukey_c=ctx.ms.tukey_c,
        )
        np.savez_compressed(p_out, **pca)


class _Stage06AmpFits(_Stage):
    key = "Write_Amp_Fits"
    produces = ("Write_Amp_Fits",)

    def run(self, ctx: _Context) -> None:
        from .plan import ShotPlan
        paths = ctx.plan.paths() if hasattr(ctx.plan, "paths") else ShotPlan(ctx.outdir).paths()
        p_out = paths["Write_Amp_Fits"]
        bw_amp = ctx.bw_amp if ctx.bw_amp is not None else _load_npz(paths["Build_Amplifier_Robust_Spectra"])["bw_amp"]
        bw_full = ctx.bw_full if ctx.bw_full is not None else _load_npz(paths["Build_Full_Exposure_Sky"])["bw_full"]
        mult = ctx.mult if ctx.mult is not None else _load_npz(paths["Fit_Multiplicative_Scale"])["mult_scale"]
        wmask = ctx.wmask if ctx.wmask is not None else _wave_mask(int(bw_full.shape[1]), ctx.ms.wave_mask_frac)
        beta_all = _load_npz(paths["Build_Additive_Polynomial"]) ["poly_beta"]
        pca = _load_npz(paths["Build_PCA_Components"])  # contains 'pca_mean' and 'pca_evecs'
        logger.info("[Write_Amp_Fits] Fit amplifiers: shared poly + PCA coeffs")
        df_fits = _fit_all_amps(
            bw_amp,
            bw_full,
            mult,
            pca,
            wmask,
            beta_all,
            loss=ctx.ms.robust_loss,
            huber_delta=ctx.ms.huber_delta,
            tukey_c=ctx.ms.tukey_c,
        )
        write_parquet(df_fits, p_out)


class _Stage07Manifest(_Stage):
    key = "Write_Model_Start_Manifest"
    produces = ("Write_Model_Start_Manifest", "Stage_Stats")

    def run(self, ctx: _Context) -> None:
        from .plan import ShotPlan
        paths = ctx.plan.paths() if hasattr(ctx.plan, "paths") else ShotPlan(ctx.outdir).paths()
        # Try to read info for stats
        try:
            with paths["Validate_Input_Shot_Info"].open("r", encoding="utf-8") as f:
                info = json.load(f)
        except Exception:
            info = ctx.info or {}
        n_amp = int(info.get("n_amp", 0)) if info else 0
        n_exp = int(info.get("exposures", 0)) if info else 0
        n_wave = int(info.get("n_wave", 0)) if info else 0
        # Inputs for stats
        bw_amp = ctx.bw_amp if ctx.bw_amp is not None else _load_npz(paths["Build_Amplifier_Robust_Spectra"]) ["bw_amp"]
        bw_full = ctx.bw_full if ctx.bw_full is not None else _load_npz(paths["Build_Full_Exposure_Sky"]) ["bw_full"]
        wmask = ctx.wmask if ctx.wmask is not None else _wave_mask(int(bw_full.shape[1]), ctx.ms.wave_mask_frac)
        # Resolve latest labels for good_mask
        from ..qc.labels import discover_latest_snapshot
        ls = ctx.ls if ctx.ls is not None else discover_latest_snapshot(ctx.outdir)
        good_mask = ls.good_mask() if (ls is not None and hasattr(ls, "good_mask")) else np.ones(bw_amp.shape[0], dtype=bool)
        # Stage stats
        stage_stats_path = paths.get("Stage_Stats")
        try:
            stats: Dict[str, Any] = {
                "mask_frac": float(np.mean(wmask)) if wmask.size else 0.0,
                "bw_amp_nonfinite_frac": float((~np.isfinite(bw_amp)).mean()) if isinstance(bw_amp, np.ndarray) else None,
                "bw_full_nonfinite_frac": float((~np.isfinite(bw_full)).mean()) if isinstance(bw_full, np.ndarray) else None,
                "n_amp": int(n_amp),
                "n_exp": int(n_exp),
                "n_wave": int(n_wave),
                "n_good_amps": int(np.sum(good_mask)) if isinstance(good_mask, np.ndarray) and good_mask.size else 0,
                "n_bad_amps": int(n_amp - int(np.sum(good_mask))) if isinstance(good_mask, np.ndarray) and good_mask.size else int(n_amp),
            }
            if stage_stats_path is not None:
                with stage_stats_path.open("w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to write stage_stats: %s", exc)
        # Manifest
        p_manifest = paths["Write_Model_Start_Manifest"]
        logger.info("[Write_Model_Start_Manifest] Write model-start manifest")
        manifest = {
            "inputs": [str(ctx.h5file)],
            "outputs": [
                str(paths[k]) for k in (
                    "Validate_Input_Shot_Info",
                    "Build_Amplifier_Robust_Spectra",
                    "Build_Full_Exposure_Sky",
                    "Compute_QC_Features",
                    "Fit_Multiplicative_Scale",
                    "Fit_Poly2D_Field_Model",
                    "Build_Additive_Polynomial",
                    "Build_PCA_Components",
                    "Write_Amp_Fits",
                    "Write_Model_Start_Manifest",
                )
            ],
            "modelspec": ctx.ms.to_dict(),
            "work_dir": str(Path(ctx.outdir).resolve()),
            "stage_stats": str(stage_stats_path) if stage_stats_path else None,
        }
        with p_manifest.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)


# ---- Existing helpers and pipeline entry point ----

def _wave_mask(n_wave: int, frac: float) -> np.ndarray:
    """Return a central-fraction wavelength mask, as used previously.

    The mask selects a contiguous central window covering ``frac`` of the
    wavelength grid. This behavior matches the earlier pipeline and ensures
    consistent scaling for BW_full in diagnostics like bw_amp_vs_full.

    Args:
        n_wave: Number of wavelength samples.
        frac: Fraction (0..1] of central wavelengths to keep.

    Returns:
        Boolean array of shape (n_wave,), True within the central window.
    """
    # Clip fraction to a safe range and compute central window
    frac = float(np.clip(frac, 0.05, 1.0))
    n = int(max(1, round(frac * int(n_wave))))
    start = (int(n_wave) - n) // 2
    end = start + n
    mask = np.zeros(int(n_wave), dtype=bool)
    mask[start:end] = True
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
    suppress_warnings: bool = False,
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

    import warnings
    ctx = _Context(h5file=h5file, outdir=out, ms=ms, plan=plan)
    def _run_all() -> None:
        stages: list[_Stage] = [
            _Stage00Info(),
            _Stage01BwAmp(),
            _Stage02BwFull(),
            _Stage03QCInitLabels(),
            _Stage04Mult(),
            _Stage0425Poly2D(),
            _Stage045Poly(),
            _Stage05PCA(),
            _Stage06AmpFits(),
            _Stage07Manifest(),
        ]
        _execute_stages(stages, ctx, resume=resume)
        # Optional diagnostics plotting (post-run; load latest labels for good_mask)
        if make_plots:
            try:
                from ..qc.labels import discover_latest_snapshot
                from ..plot import plot_bw_amp_vs_full, plot_mult_by_amp, plot_fit_example  # type: ignore
                plots_dir = out / "plots"
                plots_dir.mkdir(exist_ok=True)
                # 1) Multiplicative summary across all amps
                plot_mult_by_amp(out, highlight_outliers=True, show_labels=True, save=plots_dir / "mult_by_amp.png")
                # 2) Choose a representative amp (first good, else 0) and exposure 0
                ls_latest = discover_latest_snapshot(out)
                bw_amp_arr = _load_npz(paths["Build_Amplifier_Robust_Spectra"]) ["bw_amp"]
                good_mask = ls_latest.good_mask() if (ls_latest is not None) else np.ones(bw_amp_arr.shape[0], dtype=bool)
                a_sel = int(np.where(good_mask)[0][0]) if np.any(good_mask) else 0
                e_sel = 0
                # 3) Single-amp comparison and fit example
                plot_bw_amp_vs_full(out, amp=a_sel, exp=e_sel, save=plots_dir / f"bw_amp_vs_full_a{a_sel}_e{e_sel}.png")
                plot_fit_example(out, amp=a_sel, exp=e_sel, save=plots_dir / f"fit_example_a{a_sel}_e{e_sel}.png")
            except Exception as exc:  # pragma: no cover
                logger.warning("Plot generation failed: %s", exc)

    if suppress_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=UserWarning, module=r"astropy\\.stats")
            warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"astropy\\.stats")
            _run_all()
    else:
        _run_all()

    return {
        "status": "ok",
        "outdir": str(out),
        "manifest": str(paths["Write_Model_Start_Manifest"]),
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
