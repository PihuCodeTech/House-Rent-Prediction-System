"""Feature engineering: stateless derivations (pre-split) + the fitted FeatureBuilder (train only)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from .config import FEATURE_GROUPS
from .data_cleaning import LogIQRClipper


def add_stateless_features(df):
    """Derivations that use no cross-row statistics -> safe before the split."""
    df = df.copy()
    df["locality_key"] = df["City"] + " | " + df["Area Locality"]
    df["posted_month"] = df["posted_date"].dt.month.astype("float64")
    df["posted_dow"] = df["posted_date"].dt.dayofweek.astype("float64")
    return df


FB_INPUT_COLS = ["BHK", "Bathroom", "Size", "current_floor", "total_floors", "posted_date",
                 "posted_month", "posted_dow", "City"]


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Numeric feature construction; parameters learned on the fit data (train / CV fold-train only)."""
    def __init__(self, groups=FEATURE_GROUPS, clip_k=1.5):
        self.groups = groups
        self.clip_k = clip_k

    def fit(self, X, y=None):
        X = pd.DataFrame(X, columns=FB_INPUT_COLS) if not hasattr(X, "columns") else X
        self.clipper_ = LogIQRClipper(k=self.clip_k, side="lower").fit(X[["Size"]])
        self.floor_imputer_ = SimpleImputer(strategy="median").fit(X[["current_floor", "total_floors"]])
        self.start_date_ = pd.to_datetime(X["posted_date"]).min()
        self.cities_ = sorted(X["City"].unique())
        self.feature_names_out_ = np.asarray(list(self._build(X.head(2)).columns), dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def _build(self, X):
        g = set(self.groups)
        out = pd.DataFrame(index=X.index)
        size = self.clipper_.transform(X[["Size"]]).ravel()
        bhk, bath = X["BHK"].to_numpy("float64"), X["Bathroom"].to_numpy("float64")
        out["BHK"], out["Bathroom"] = bhk, bath
        if "size_raw" in g:
            out["size"] = size
        if "log" in g:
            out["log_size"], out["log_bhk"], out["log_bath"] = np.log1p(size), np.log1p(bhk), np.log1p(bath)
        if "floor" in g:
            fl = self.floor_imputer_.transform(X[["current_floor", "total_floors"]])
            cur, tot = fl[:, 0], np.maximum(fl[:, 1], fl[:, 0])
            out["current_floor"], out["total_floors"] = cur, tot
            out["floor_ratio"] = np.clip(cur, 0, None) / np.maximum(tot, 1)
            out["is_basement"] = (cur < 0).astype("float64")
            out["log_total_floors"] = np.log1p(tot)
        if "ratios" in g:
            out["size_per_bhk"] = size / bhk
            out["log_size_per_bhk"] = np.log1p(size / bhk)
            out["bath_per_bhk"] = bath / bhk
        if "date" in g:
            out["days_since_start"] = (pd.to_datetime(X["posted_date"]) - self.start_date_).dt.days.to_numpy("float64")
            out["posted_month"] = X["posted_month"].to_numpy("float64")
        if "dow" in g:
            out["posted_dow"] = X["posted_dow"].to_numpy("float64")
        if "city_x_size" in g:
            ls = np.log1p(size)
            for c in self.cities_:
                out[f"logsize_x_{c}"] = ls * (X["City"].to_numpy() == c)
        return out

    def transform(self, X):
        X = pd.DataFrame(X, columns=FB_INPUT_COLS) if not hasattr(X, "columns") else X
        return self._build(X).to_numpy("float64")

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


class LocalityCountEncoder(BaseEstimator, TransformerMixin):
    """log1p(number of TRAIN rows sharing the locality); unseen -> 0. Uses no target."""
    def fit(self, X, y=None):
        s = pd.Series(np.asarray(X).ravel())
        self.counts_ = s.value_counts().to_dict()
        self.n_features_in_ = 1
        return self

    def transform(self, X):
        s = pd.Series(np.asarray(X).ravel())
        return np.log1p(s.map(self.counts_).fillna(0).to_numpy("float64")).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(["locality_count_log"], dtype=object)


CAT_COLS = ["City", "Furnishing Status", "Area Type", "Tenant Preferred", "Point of Contact"]
