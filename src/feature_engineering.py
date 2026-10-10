"""Feature engineering: stateless derivations (safe before the split) and the fitted FeatureBuilder.

Numeric features (29 in total with the encoders in preprocessing.py):
    BHK, Bathroom, log1p of size / BHK / bathrooms, and log1p(size) crossed with each city.
Floor and building height are not used: together they changed the error by about Rs 600 when shuffled (versus
Rs 25,800 for size), and their effect had no consistent direction, so they were dropped for a simpler model.
The posting date is not used: listings span only a few months of 2022, and it is not a property attribute.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .data_cleaning import LogIQRClipper

FB_INPUT_COLS = ["BHK", "Bathroom", "Size", "City"]
CAT_COLS = ["City", "Furnishing Status", "Area Type", "Tenant Preferred", "Point of Contact"]


def add_stateless_features(df):
    """Row-wise derivations that use no statistics across rows (safe before the split)."""
    df = df.copy()
    df["locality_key"] = df["City"] + " | " + df["Area Locality"]  # same locality name can exist in two cities
    return df


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Numeric features. Learns (on the fit rows only): the Size lower clip and the list of cities for the
    city × size interactions."""

    def __init__(self, clip_k=1.5):
        self.clip_k = clip_k

    def fit(self, X, y=None):
        """Learn the Size clip and city list from the fit rows."""
        X = self._frame(X)
        self.clipper_ = LogIQRClipper(k=self.clip_k).fit(X[["Size"]])
        self.cities_ = sorted(X["City"].unique())
        self.feature_names_out_ = np.asarray(list(self._build(X.head(2)).columns), dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """Build the numeric feature matrix (float64)."""
        return self._build(self._frame(X)).to_numpy("float64")

    def get_feature_names_out(self, input_features=None):
        """Names of the numeric features, in column order."""
        return self.feature_names_out_

    @staticmethod
    def _frame(X):
        return X if hasattr(X, "columns") else pd.DataFrame(X, columns=FB_INPUT_COLS)

    def _build(self, X):
        out = pd.DataFrame(index=X.index)
        size = self.clipper_.transform(X[["Size"]]).ravel()
        bhk, bath = X["BHK"].to_numpy("float64"), X["Bathroom"].to_numpy("float64")
        out["BHK"], out["Bathroom"] = bhk, bath
        out["log_size"], out["log_bhk"], out["log_bath"] = np.log1p(size), np.log1p(bhk), np.log1p(bath)

        log_size = np.log1p(size)
        for city in self.cities_:
            out[f"logsize_x_{city}"] = log_size * (X["City"].to_numpy() == city)
        return out
