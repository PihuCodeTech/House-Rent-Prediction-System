"""Data cleaning.

Stateless rules learn nothing from the data distribution, so they run before the split and are written as
conditions (never as hand-picked row indices). Anything learned from the data (the Size clipper, the far-out
filter) is fitted on the training split only.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TEXT_COLS = [
    "Posted On",
    "Floor",
    "Area Type",
    "Area Locality",
    "City",
    "Furnishing Status",
    "Tenant Preferred",
    "Point of Contact",
]
FLOOR_LEVEL_MAP = {"Ground": 0, "Upper Basement": -1, "Lower Basement": -2}
MAX_RENT_PER_SQFT = 500  # Rs per sq ft per month; prime Delhi/Mumbai listings reach about Rs 500


# ----------------------------- stateless rules (run before the split) -----------------------------
def rule_strip_text(df):
    """Trim whitespace in text columns; normalise locality names (lower case, punctuation removed)."""
    df = df.copy()
    for c in TEXT_COLS:
        if c in df:
            df[c] = df[c].astype(str).str.strip()
    df["Area Locality"] = df["Area Locality"].str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    return df


def rule_validity(df):
    """Keep physically valid rows: Rent > 0, Size > 0, BHK >= 1, Bathroom >= 1."""
    ok = (df["Rent"] > 0) & (df["Size"] > 0) & (df["BHK"] >= 1) & (df["Bathroom"] >= 1)
    return df[ok]


def rule_rent_per_sqft(df):
    """Drop implausible listings: monthly rent above MAX_RENT_PER_SQFT per sq ft (unit or typing errors,
    e.g. a 3BHK of 10 sq ft, or Rs 35 lakh for a 2,500 sq ft flat)."""
    return df[(df["Rent"] / df["Size"]) <= MAX_RENT_PER_SQFT]


def rule_dedup_relisted(df):
    """Drop re-listings: rows identical in every column except the posting date."""
    return df.drop_duplicates(subset=[c for c in df.columns if c != "Posted On"], keep="first")


def rule_parse_floor(df):
    """'3 out of 5' -> current_floor = 3, total_floors = 5 (Ground = 0, basements negative)."""
    df = df.copy()
    parts = df["Floor"].str.split(" out of ", n=1, expand=True).reindex(columns=[0, 1])
    current = parts[0].replace(FLOOR_LEVEL_MAP)
    df["current_floor"] = pd.to_numeric(current, errors="coerce").astype("float64")
    df["total_floors"] = pd.to_numeric(parts[1], errors="coerce").astype("float64")
    above_total = df["current_floor"] > df["total_floors"]  # NaN comparisons are False
    df.loc[above_total, "total_floors"] = df.loc[above_total, "current_floor"]
    return df


STATELESS_RULES = [rule_strip_text, rule_validity, rule_rent_per_sqft, rule_dedup_relisted, rule_parse_floor]


def apply_stateless_rules(df, verbose=False):
    """Apply every stateless rule in order. Returns (cleaned DataFrame, per-rule row log)."""
    log = []
    for rule in STATELESS_RULES:
        n_before = len(df)
        df = rule(df)
        log.append(
            {"rule": rule.__name__, "rows_before": n_before, "rows_after": len(df), "rows_removed": n_before - len(df)}
        )
    log = pd.DataFrame(log)
    if verbose:
        print(log.to_string(index=False))
    return df.reset_index(drop=True), log


# ----------------------------- learned on the training split only -----------------------------
class LogIQRClipper(BaseEstimator, TransformerMixin):
    """Clip values from below at the Tukey lower fence of log1p(x), learned on the training rows.
    Never drops rows, so validation/test rows are only clipped with the training fence."""

    def __init__(self, k=1.5):
        self.k = k

    def fit(self, X, y=None):
        """Learn the lower fence from the training rows."""
        logged = np.log1p(np.asarray(X, dtype="float64"))
        q1, q3 = np.nanpercentile(logged, 25, axis=0), np.nanpercentile(logged, 75, axis=0)
        self.lower_ = np.expm1(q1 - self.k * (q3 - q1))
        self.n_features_in_ = logged.shape[1]
        return self

    def transform(self, X):
        """Raise values below the learned fence to the fence."""
        return np.maximum(np.asarray(X, dtype="float64"), self.lower_)


def train_only_far_out_mask(X_train, y_train, k=3.0):
    """Keep-mask for TRAINING rows: drop rows whose log(rent per sq ft) is beyond Tukey far-out fences
    computed on the training rows themselves. Never applied to validation/test rows."""
    rps = np.log(np.asarray(y_train, dtype="float64") / X_train["Size"].to_numpy(dtype="float64"))
    q1, q3 = np.percentile(rps, [25, 75])
    lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    keep = (rps >= lo) & (rps <= hi)
    info = {
        "k": k,
        "rent_per_sqft_lower": float(np.exp(lo)),
        "rent_per_sqft_upper": float(np.exp(hi)),
        "n_dropped": int((~keep).sum()),
    }
    return keep, info
