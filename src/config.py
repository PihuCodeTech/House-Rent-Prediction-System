"""Experiment settings shared by every notebook, script and test. Change nothing here mid-experiment."""

from pathlib import Path

# --- reproducibility ---
RANDOM_STATE = 42  # default seed (diagnostic plots, feature importance)
SEEDS = (42, 43, 44)  # every model runs on all three; results are reported as mean ± std
TARGET = "Rent"
TARGET_TRANSFORM = "log1p"  # models learn log1p(rent); predictions are converted back to rupees

# --- split: 70 / 15 / 15, stratified on rent deciles ---
TEST_SIZE = 0.15
VAL_SIZE_OF_TEMP = 0.1765  # 0.1765 × 0.85 ≈ 0.15
STRAT_Q = 10  # rent bins used for stratified splitting only, never as a feature
CV_FOLDS = 5

# --- cleaning ---
FAR_OUT_K = 3.0  # train-only filter: Tukey far-out fences on log(rent per sq ft)
CLIP_K = 1.5  # lower clip of Size at Tukey fence of log1p(Size), learned on train

# --- integrity references for seed 42 (a mismatch usually means a pandas/scikit-learn version change) ---
EXPECTED_SPLIT_HASHES = {"train": "67cd67c1e7b86380", "val": "7f4c4f69e012f488", "test": "8b6ab3d03e11fffa"}
EXPECTED_REFERENCE_ID = "e9c8bab776f26dc9"

# --- paths ---
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw" / "House_Rent_Dataset.csv"
REPORTS = ROOT / "reports"
RESULTS_CSV = REPORTS / "results.csv"  # the single results file (mean ± std over SEEDS)
FINAL_ERRORS_CSV = REPORTS / "final_model_errors.csv"  # final model: test error by city and rent band
FINAL_RESIDUALS_CSV = REPORTS / "final_model_residuals.csv"  # final model: histogram of test residuals
RUN_REFERENCE = REPORTS / ".run_reference.json"  # same-data tripwire across notebooks (one id per seed)
MODELS_DIR = ROOT / "models"
VIZ_DIR = ROOT / "visualizations"
