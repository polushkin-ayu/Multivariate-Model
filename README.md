# The monetary cost of calibration error for an additional score in customer acquisition decisions

Reproducibility package for the current synthetic full-model results. At N=64 the expected-profit loss is 2.0612% (−8,711 currency units per 100,000 eligible users); cached gains are +981 at N=512 and +2,966 at N=4,096. No customer data are included.

## Environment

The package was checked with the local project interpreter (Python 3.9.6, NumPy 2.0.2, SciPy 1.13.1, Matplotlib 3.9.4). Use Python 3.9 or newer with dependencies from `requirements.txt`:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
export MPLCONFIGDIR=.mplconfig
```

## Rebuild the eight figures from saved results

Run from this directory:

```sh
python3 full_model_example/refined/render_ru.py
python3 full_model_example/refined/render_en.py
python3 full_model_example/build/build_assets.py
python3 full_model_example/sensitivity/render_ru.py
python3 full_model_example/sensitivity/render_en.py
python3 full_model_example/theorem_training/make_figure_ru.py
python3 full_model_example/theorem_training/make_figure_en.py
```

This produces 32 files: PNG and PDF in Russian and English for each of eight logical figures: process, calibration mechanism, theorem mean normalized L1 residual, profit decomposition, learning/reference quality, profit distribution, sensitivity profit map, and theorem validation. Outputs are stored in the corresponding `full_model_example/build/`, `refined/`, `sensitivity/`, and `theorem_training/` directories. Both refined renderers read `refined/profit_histogram.json`; they do not replay draws. That cache was produced once from the original fixed-seed sequence (`seed=2026101347`, `N` order recorded in the JSON, stopping at N=64, 200,000 replications). Before serialization, its mean was asserted equal to `primary.mean_gain × flow` (absolute difference 0.0). The remaining generators read saved JSON/NPZ data. The build manifest maps every file to its generator and source inputs and records SHA-256 checksums.

## Full scientific recalculation

These commands rerun the numerical analyses and can take substantial time. They are provided for authors who want to regenerate the saved result data; they were not run during this packaging task.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 full_model_example/refined/run.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 full_model_example/refined/validate.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 full_model_example/sensitivity/run_sensitivity.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 full_model_example/sensitivity/run_refinement.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 full_model_example/theorem_training/run_validation.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 full_model_example/theorem_check/run.py
```

The selected run scripts depend on the included `GaussianCAC` model (`cac_calibration.py`), `WeakBoundaryEngine` (`structural_weak_score.py`), decision-boundary utilities (`cac_decision.py`), and shared model implementation (`verify_structural_weak_score.py`). The related root-level `gaussian_cac.py` and `gaussian_risk_integrals.py` tools are also included with their NumPy/SciPy/Matplotlib dependencies. They write results and checkpoints inside their scenario directories. Keep the fixed scenario specifications and do not reuse checkpoints after changing a scenario.

## License

No license is assigned, as chosen by the author for this GitHub release.
Zenodo publication is deferred; this version has no archival DOI.

## Package scope and checks

This repository contains model and experiment code, eight figures with Russian and English labels, and the saved synthetic results needed to reproduce them. Manuscripts and manuscript build files are not included. No customer data are included.

Run `python3 scripts/build_numbers.py` to regenerate analytic summaries from saved results, then `python3 scripts/check_release.py` to check analytic controls and the 32 figure asset hashes. PDF metadata can change after a rebuild. The tested dependency versions are recorded in `requirements-verified.txt`.
