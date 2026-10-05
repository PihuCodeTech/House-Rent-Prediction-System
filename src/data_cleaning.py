"""Stateless, distribution-free cleaning RULES + leakage-safe learned cleaners.

Rules learn nothing from the data distribution, so they run pre-split and are expressed as
conditions (never hand-picked row indices). Learned cleaners (LogIQRClipper) are fit on TRAIN only.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

TEXT_COLS = ["Posted On", "Floor", "Area Type", "Area Locality", "City",
             "Furnishing Status", "Tenant Preferred", "Point of Contact"]
FLOOR_LEVEL_MAP = {"Ground": 0, "Upper Basement": -1, "Lower Basement": -2}


def rule_strip_text(df):
    df = df.copy()
    for c in TEXT_COLS:
        df[c] = df[c].astype(str).str.strip()
    df["Area Locality"] = (df["Area Locality"].str.lower()
                           .str.replace(r"[^a-z0-9]+", " ", regex=True)
                           .str.strip())
    return df


def rule_validity(df):
    """Keep only physically valid rows (a rule, not an index list)."""
    ok = (df["Rent"] > 0) & (df["Size"] > 0) & (df["BHK"] >= 1) & (df["Bathroom"] >= 1)
    return df[ok]


def rule_dedup_relisted(df):
    """Drop exact re-listings (same everything except the posting date)."""
    return df.drop_duplicates(subset=[c for c in df.columns if c != "Posted On"], keep="first")


def rule_parse_floor(df):
    df = df.copy()
    parts = df["Floor"].str.split(" out of ", n=1, expand=True).reindex(columns=[0, 1])
    cur = parts[0].replace(FLOOR_LEVEL_MAP)
    df["current_floor"] = pd.to_numeric(cur, errors="coerce").astype("float64")
    df["total_floors"] = pd.to_numeric(parts[1], errors="coerce").astype("float64")
    bad = df["current_floor"] > df["total_floors"]          # NaN comparisons are False
    df.loc[bad, "total_floors"] = df.loc[bad, "current_floor"]
    return df


def rule_parse_date(df):
    df = df.copy()
    df["posted_date"] = pd.to_datetime(df["Posted On"], format="%Y-%m-%d", errors="coerce")
    return df


STATELESS_RULES = [rule_strip_text, rule_validity, rule_dedup_relisted, rule_parse_floor, rule_parse_date]


def apply_stateless_rules(df, verbose=False):
    """Apply all distribution-free rules in a fixed order. Returns (clean_df, log_df)."""
    log = []
    for rule in STATELESS_RULES:
        n_before = len(df)
        df = rule(df)
        log.append({"rule": rule.__name__, "rows_before": n_before, "rows_after": len(df),
                    "rows_removed": n_before - len(df)})
    log = pd.DataFrame(log)
    if verbose:
        print(log.to_string(index=False))
    return df.reset_index(drop=True), log


class LogIQRClipper(BaseEstimator, TransformerMixin):
    """Clip numeric columns to Tukey fences learned on log1p(x) of the TRAIN data.

    transform() never drops rows (only clips), so val/test are transformed with train-learned fences.
    """
    def __init__(self, k=1.5, side="lower"):
        self.k = k
        self.side = side

    def fit(self, X, y=None):
        L = np.log1p(np.asarray(X, dtype="float64"))
        q1, q3 = np.nanpercentile(L, 25, axis=0), np.nanpercentile(L, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = np.expm1(q1 - self.k * iqr) if self.side in ("lower", "both") else np.full(L.shape[1], -np.inf)
        self.upper_ = np.expm1(q3 + self.k * iqr) if self.side in ("upper", "both") else np.full(L.shape[1], np.inf)
        self.n_features_in_ = L.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        A = np.asarray(X, dtype="float64")
        return np.clip(A, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return getattr(self, "feature_names_in_",
                       np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object))


def train_only_far_out_mask(X_train, y_train, k=3.0):
    """Keep-mask for TRAIN rows only: drop rows whose log(Rent/Size) is beyond Tukey far-out fences
    computed on the train rows themselves. NEVER call this on val/test."""
    rps = np.log(np.asarray(y_train, dtype="float64") / X_train["Size"].to_numpy(dtype="float64"))
    q1, q3 = np.percentile(rps, [25, 75]); iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    keep = (rps >= lo) & (rps <= hi)
    info = {"k": k, "rent_per_sqft_lower": float(np.exp(lo)), "rent_per_sqft_upper": float(np.exp(hi)),
            "n_dropped": int((~keep).sum())}
    return keep, info


NUM_IMPUTER = SimpleImputer(strategy="median")          # fit on X_train
CAT_IMPUTER = SimpleImputer(strategy="most_frequent")   # safety net
