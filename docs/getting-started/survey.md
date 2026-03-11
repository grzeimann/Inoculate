# Survey orchestrator (inoculate-survey)

This page explains what scientific problem the survey pipeline addresses, how the algorithm works step‑by‑step, and how the code pieces map to the scientific ideas. It also shows example commands.

## What scientific problem does this pipeline solve?

VIRUS sky subtraction is limited by small, systematic multiplicative variations in fiber and amplifier response that change with time and observing conditions. Per‑exposure ("shot") modeling can estimate and remove these variations, but the estimate is noisy if taken from a single visit, especially when bright sources contaminate some fibers.

The survey pipeline addresses this by aggregating many shots to learn stable, instrument‑anchored priors for each IFU’s fiber×amplifier multiplicative behavior (delta_mult). These survey‑level priors summarize, for every IFU fiber across the instrument, the typical offset and dispersion of the multiplicative scale seen across many nights and conditions. They can then be fed back into the per‑shot IFU modeling to stabilize fits, down‑weight outliers, and reduce residuals in faint‑line science.

In short: the pipeline turns many per‑IFU, per‑exposure measurements into robust profiles (location and dispersion) that act as informative priors for subsequent sky modeling.

## How does the algorithm work step‑by‑step?

The survey orchestrator follows the same step‑by‑step flow shown in the README’s Mermaid diagram, explicitly connecting the products it locates to how they are made.

1. Entry and discovery
   - If you pass raw H5 files and request `--build-from-h5`, the orchestrator first runs the per‑shot pipelines into `survey_out/shots/<shotname>/` (see steps 2–3 below), then proceeds to aggregation.
   - If you pass a root containing existing per‑shot outdirs (identified by `stage_07_model_start_manifest.json`), it discovers them directly and skips building.

2. Shot‑level pipeline (per discovered shot; how IFU inputs are produced)
   The following stages are executed when `--build-from-h5` is used; these create the prerequisites that the IFU modeling and the survey aggregator will later consume.
   - Validate_Input_Shot_Info
   - Stage 00: Build wavelength mask (typ. central 96%)
   - Build_Amplifier_Robust_Spectra
   - Build_Full_Exposure_Sky
   - Compute_QC_Features
   - Fit_Multiplicative_Scale
   - Fit_Poly2D_Field_Model
   - Build_Additive_Polynomial
   - Build_PCA_Components
   - Write_Amp_Fits
   - Write_Model_Start_Manifest (produces the manifest used for discovery)

3. IFU modeling within each shot (exact artifacts and where they come from)
   These stages turn shot‑level products into the per‑amp and per‑IFU fiber models that the survey step locates under each shot’s `ifu/` directory.
   - IFU_Compute_PerAmp_Fiber_Model
     - For every amplifier a and exposure e, writes: `ifu/aAA_eEE_fiber_model.npz`
     - Contains keys: `delta_mult` (length 112), `source_mask` (bool[112]), `ifuslot`, `amp_label`, `notes`.
   - IFU_Aggregate_PerIFU
     - Concatenates the four amps of an IFU to length 448 (4×112) and writes: `ifu/ifuIII_eEE_fiber_model.npz`
     - This is the exact file pattern the survey step scans: `ifu*_e*_fiber_model.npz`.
   - IFU_Write_PerIFU_Plots (optional diagnostics)
     - Writes plots like `ifu/plots/ifuNNN_profiles.png` (one line per exposure), which can help validate inputs to aggregation.

4. Locate per‑IFU fiber‑level products (what the survey reads)
   - For each discovered shot outdir, the survey scans the `ifu/` directory for `ifu*_e*_fiber_model.npz` made by IFU_Aggregate_PerIFU.
   - From each NPZ it extracts the per‑fiber multiplicative residual vector `delta_mult` (length 448) and, if present, the boolean `source_mask`.

5. Aggregate across shots (per IFU)
   - For each IFU index, collect all available `delta_mult` vectors across shots (and exposures).
   - Compute per‑fiber robust statistics across samples:
     - Robust mean profile via biweight location (high‑breakdown, outlier‑resistant).
     - Robust dispersion via MAD (median absolute deviation) scaled by 1.4826 to approximate σ.
   - Also track `n_samples` (how many vectors contributed) and `frac_masked` (mean of `source_mask`, i.e., fraction of samples flagged as source‑contaminated for each fiber).

6. Write the survey registry
   - The aggregated per‑IFU statistics are written to a registry JSON under the survey output directory. These constitute survey‑level priors that the IFU modeling can use in a second pass.

Key properties:
- Robustness: uses robust estimators (biweight, MAD) to mitigate outliers and partial contamination.
- Resumability: can be re‑run without recomputing everything (and can build per‑shot outputs on the fly from H5 files when requested).
- Modularity: per‑shot IFU modeling is independent and produces standard artifacts that the survey job can consume.

## How the code maps to the scientific ideas

Scientific idea: Learn stable, instrument‑anchored priors for multiplicative fiber×amp behavior from many shots.

Code components:
- CLI frontend
  - File: `src/inoculate/cli/inoculate_survey.py`
  - Responsible for parsing options such as `--build-from-h5`, `--max-shots`, resume flags, and invoking the survey pipeline via `run_survey`.

- Survey aggregation pipeline
  - File: `src/inoculate/survey/pipeline.py`
  - `run_survey(...)`: Orchestrates discovery of shots, optional per‑shot building (when `build_from_h5=True`), aggregation, and writing the registry via `SurveyPlan`.
  - `_discover_shot_outdirs(...)`: Locates shot outdirs by finding the per‑shot manifest.
  - `_iter_ifu_npz(...)`: Iterates over `ifu*_e*_fiber_model.npz` artifacts inside each shot outdir.
  - `_aggregate_ifu_across_shots(...)`: Core aggregation; stacks `delta_mult` vectors across shots and computes per‑fiber robust mean (biweight) and dispersion (MAD), plus `frac_masked`.

- Per‑shot and IFU context
  - File: `src/inoculate/shot/pipeline.py` and `src/inoculate/ifu/pipeline.py`
  - These components build the per‑shot fiber‑level products that contain `delta_mult` and `source_mask`. In IFU modeling, the profiles are 448‑length vectors (4 amplifiers × 112 fibers) that describe multiplicative residuals after sky modeling; those are the inputs to survey aggregation.

- Robust statistics utilities
  - File: `src/inoculate/robust/__init__.py` (and related)
  - Provides `biweight_location` (used for robust mean) and helpers used throughout the pipelines.

- Priors for multiplicative sky model
  - File: `src/inoculate/sky/mult/priors.py`
  - Consumes the registry profiles (mean and dispersion) to regularize per‑shot multiplicative fits during IFU modeling, closing the survey→shot feedback loop.

## Example commands

- Aggregate existing shot outputs under a root and write the survey registry:

  inoculate-survey /path/to/shots_root --outdir survey_out

- Limit to the first N shots for a quick development test:

  inoculate-survey /path/to/shots_root --outdir survey_out --max-shots 5

- Build from H5 files (each H5 corresponds to a shot), cap the number of inputs, and suppress warnings in per‑shot runs (example path and options taken from the issue):

  inoculate-survey /Users/grz85/work/lukas_virus/ --build-from-h5 --max-shots 50 --suppress-warnings

Common options:
- `--build-from-h5`: Discover .h5 files under `shots_root` and first run per‑shot pipelines into `survey_out/shots/<name>/` before aggregation.
- `--max-shots N`: Cap the number of shots (or H5 files) processed.
- `--suppress-warnings`: Suppress runtime warnings during per‑shot building.
- `--shot-resume`, `--ifu-resume`: Pass resume flags into the per‑shot and IFU pipelines when building from H5.
- `--outdir survey_out`: Set the survey output directory (default: `survey_out`).

## Outputs and how to use them

The primary artifact is a survey registry (JSON) under `survey_out/registry/` that, for each IFU index, contains:
- `mean_delta_mult`: Robust per‑fiber mean profile (length 448).
- `mad_delta_mult`: Robust per‑fiber dispersion (MAD × 1.4826).
- `n_samples`: Number of `delta_mult` vectors contributing for that IFU.
- `frac_masked`: Average fraction of samples flagged as source‑contaminated for each fiber.

These profiles can be supplied back to the IFU modeling as priors (see `sky/mult/priors.py` and the IFU pipeline) to stabilize multiplicative fits in subsequent passes, reducing residual systematics.

## Notes and caveats

- The aggregator tolerates partial IFU coverage by default (`include_partial_ifu=True`) but you can tighten this behavior if needed.
- The robustness of the priors improves with the diversity and number of shots; consider re‑running the survey aggregation periodically as the dataset grows.
- The NPZ schema must include `delta_mult`; `source_mask` is optional and will be treated as all‑False if absent.
