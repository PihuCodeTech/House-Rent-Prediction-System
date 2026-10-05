"""Central configuration — the frozen experiment settings (do not change mid-experiment)."""
from pathlib import Path

RANDOM_STATE = 42
TARGET = "Rent"
TARGET_TRANSFORM = "log1p"          # models wrapped in TransformedTargetRegressor(log1p, expm1) -> predictions in rupees

# --- split ---
TEST_SIZE = 0.15
VAL_SIZE_OF_TEMP = 0.1765           # 0.1765 * 0.85 ~= 0.15  ->  70 / 15 / 15
STRAT_Q = 10                        # qcut bins for stratified splitting ONLY (never a feature)

# --- cleaning / features (frozen in the original Phase 2-4) ---
FAR_OUT_K = 3.0                     # train-only rent/sqft far-out filter (Tukey, k=3)
FEATURE_GROUPS = ("log", "floor", "city_x_size")
LOCALITY_ENCODING = "target"       # {"target", "count", None}
CLIP_K = 1.5                        # LogIQR lower-clip multiplier for Size

# --- integrity references (from the original frozen Phase 3 / Phase 5) ---
# Pre-filter split index hashes (deterministic given seed+data).
EXPECTED_SPLIT_HASHES = {"train": "82bf32b8437a934e", "val": "55e1909c244731ec", "test": "1874e7d6b55e66dd"}
# Post-prep fingerprint id. Recomputed at runtime; a change here usually means a sklearn/pandas version shift.
EXPECTED_REFERENCE_ID = "70560d1bd70af7cd"

# --- paths ---
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw" / "House_Rent_Dataset.csv"
DATA_RAW_FALLBACK = ROOT / "House_Rent_Dataset.csv"     # notebooks run with repo root as cwd read this
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
METRICS_DIR = REPORTS / "metrics"
PREDICTIONS_DIR = REPORTS / "predictions"
MODELS_DIR = ROOT / "models"
VIZ_DIR = ROOT / "visualizations"
RUN_REFERENCE = REPORTS / "_run_reference.json"        # cross-model fairness tripwire

CV_FOLDS = 5
