# CLAUDE.md — Project Rules (read every turn)

Supervised **regression** experiment predicting house rent, as a clean modular repo.
Full orientation: `README.md`. These rules are always in force.

## Architecture (single source of truth = src/)
- All data-prep logic lives ONCE in `src/` and is imported, never copy-pasted. Every model
  notebook is thin: it calls `src.preprocessing.prepare_data()` and `src.models.make_model()`.
- Layout: `data/raw` (CSV), `notebooks/` (one .ipynb per model + model_comparison), `src/`
  (config, data_loading, data_cleaning, feature_engineering, preprocessing, models, evaluation,
  prediction), `models/`, `reports/{metrics,predictions}`, `visualizations/`, `main.py`.
- The original phase1–phase10 work is archived in `_legacy_phases/` (reference only).

## Data
- `data/raw/House_Rent_Dataset.csv` (~4,746 rows). Target `Rent` (continuous), modelled on
  `log1p`; predictions/metrics reported in rupees. Glossary in `data/raw/`.
- Each model notebook reads the CSV and runs the full `src` pipeline itself (identical by
  construction). No prepared .joblib bundle is required to run a model.

## Non-negotiable rules
1. **Identical data for every model.** Same rows/features/split/seed (`RANDOM_STATE=42`). The
   fingerprint `reference_id` (currently `70560d1bd70af7cd`) is written by the first model to
   `reports/_run_reference.json`; every other model asserts it matches via
   `assert_run_consistency(data)`. Mismatch -> STOP.
2. **No hardcoding.** Cleaning is stateless RULES (conditions, never row indices) or fitted
   TRANSFORMERS in sklearn Pipelines. No manual per-row edits.
3. **No leakage.** Split before anything fitted; impute/encode/scale/target-transform fit on
   TRAIN only; CV re-fits transforms per fold; validation is for selection only; test untouched
   until `EVALUATE_TEST=True` (or `main.py --test`) at the final stage.
4. **Fair comparison.** Only the estimator differs between notebooks; scaling is applied only
   where it matters (linear/distance models) and never changes the underlying prepared matrix.
5. **Regression** — no SMOTE / resampling. Target bins are for stratified splitting only.

## Workflow
- Change prep ONLY in `src/` (one place). Delete `reports/_run_reference.json` to re-baseline
  the fingerprint if the pipeline is changed intentionally.
- Defaults: `python main.py --test`. Tuning: `python main.py --tune --test` (search spaces and
  budgets in `src/tuning.py`; CV on the TRAIN split only), then `python scripts/robustness_check.py`
  (repeated 5x3 CV + paired test bootstrap), then `notebooks/final_selection.ipynb`.
- Model choice uses training-split evidence only (lowest repeated-CV RMSE). Never pick a model
  or hyperparameters by test score; test gaps among the top boosters are within bootstrap noise.
- Final configs: `python main.py --decide` keeps tuned params only where they beat the default on
  test RMSE (user's rule) -> `reports/final_configs.json`; `python main.py --final` re-evaluates all
  models with those configs into `reports/final/` (never overwrites reports/metrics or the main
  leaderboard). Since this uses test for selection, treat validation + repeated CV as the unbiased checks.
- Final model is chosen by the user and locked in via `CHOICE` in `notebooks/final_selection.ipynb`.
- Results: `reports/metrics/metrics_<key>[_tuned].{json,csv}`, `reports/predictions/`,
  `reports/model_comparison.csv`, `reports/tuning/`, `reports/tuned_best_params.json`,
  `reports/tuning_report.md`.

## Environment
- scikit-learn models are CPU-only (train <1s on this data); there is no Apple-GPU path for them.
- xgboost is required (final model); lightgbm/catboost for the full comparison. macOS needs
  `brew install libomp`. CatBoost runs with `allow_writing_files=False` (no catboost_info/).
- Untuned Decision Tree / Random Forest can differ slightly across CPU architectures (fully grown
  trees flip near-tied splits); everything else reproduces exactly.
