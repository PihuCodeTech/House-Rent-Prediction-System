"""Model registry. Each entry returns an unfitted estimator wrapped in the common Pipeline
(+ model-appropriate scaler) and a TransformedTargetRegressor for the log1p target, so predictions
come back in rupees. The prepared feature matrix is identical for all models; only the estimator and
whether it is scaled differ.
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor

from .config import RANDOM_STATE, TARGET_TRANSFORM

RS = RANDOM_STATE

# Scaling matters for linear / distance models; trees are scale-invariant.
_SCALED = True
_UNSCALED = False

# name -> (display name, factory() -> estimator, needs_scaling, needs_extra_package)
REGISTRY = {
    "baseline_mean":       ("Baseline (mean rent)",       lambda: DummyRegressor(strategy="mean"),                      _UNSCALED, None),
    "baseline_median":     ("Baseline (median rent)",     lambda: DummyRegressor(strategy="median"),                    _UNSCALED, None),
    "linear_regression":   ("Linear Regression",          lambda: LinearRegression(),                                  _SCALED,   None),
    "ridge_regression":    ("Ridge Regression",           lambda: Ridge(random_state=RS),                               _SCALED,   None),
    "lasso_regression":    ("Lasso Regression",           lambda: Lasso(random_state=RS, max_iter=10000),               _SCALED,   None),
    "elastic_net":         ("Elastic Net",                lambda: ElasticNet(random_state=RS, max_iter=10000),          _SCALED,   None),
    "decision_tree":       ("Decision Tree Regressor",    lambda: DecisionTreeRegressor(random_state=RS),               _UNSCALED, None),
    "random_forest":       ("Random Forest Regressor",    lambda: RandomForestRegressor(random_state=RS, n_jobs=-1),    _UNSCALED, None),
    "gradient_boosting":   ("Gradient Boosting Regressor",lambda: GradientBoostingRegressor(random_state=RS),           _UNSCALED, None),
    "xgboost_model":       ("XGBoost Regressor",          lambda: _xgb(),                                               _UNSCALED, "xgboost"),
    "lightgbm_model":      ("LightGBM Regressor",         lambda: _lgbm(),                                              _UNSCALED, "lightgbm"),
    "catboost_model":      ("CatBoost Regressor",         lambda: _catboost(),                                          _UNSCALED, "catboost"),
}


try:
    from xgboost import XGBRegressor as _XGBBase
except ImportError:                       # xgboost is optional for everything except XGBoost models
    _XGBBase = None

if _XGBBase is not None:
    class XGBRegressorMedianInit(_XGBBase):
        """XGBRegressor that, for the pseudo-Huber objective, starts boosting from the median of the
        targets it is fitted on (bounded Huber gradients cannot climb from XGBoost's default start to
        a log-rent scale of ~9-12). The median comes from the fit data only -> no leakage across CV
        folds. Behaviour for every other objective is unchanged. Module-level so it pickles."""
        def fit(self, X, y, **kw):
            if self.get_params().get("objective") == "reg:pseudohubererror" and self.get_params().get("base_score") is None:
                self.set_params(base_score=float(np.median(np.asarray(y, dtype="float64"))))
                try:
                    return super().fit(X, y, **kw)
                finally:
                    self.set_params(base_score=None)   # keep get_params()/clone() identical to the input
            return super().fit(X, y, **kw)


def _xgb():
    if _XGBBase is None:
        raise ImportError("xgboost is not installed (pip install xgboost)")
    return XGBRegressorMedianInit(random_state=RS, n_jobs=-1, tree_method="hist")


def _lgbm():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(random_state=RS, n_jobs=-1, verbose=-1)


def _catboost():
    from catboost import CatBoostRegressor
    return CatBoostRegressor(random_state=RS, verbose=0, allow_writing_files=False)


def make_model(key, target_transform=TARGET_TRANSFORM, estimator=None):
    """Build the full pipeline for a registry key (or a custom estimator).

    Returns a fitted-ready model whose .predict() yields rupees.
    """
    if key not in REGISTRY:
        raise KeyError(f"unknown model key {key!r}; choices: {list(REGISTRY)}")
    display, factory, needs_scaling, _ = REGISTRY[key]
    est = estimator if estimator is not None else factory()
    steps = ([("scaler", StandardScaler())] if needs_scaling else []) + [("model", est)]
    pipe = Pipeline(steps)
    if key.startswith("baseline"):
        return pipe            # a constant predictor; the target transform is irrelevant
    if target_transform == "log1p":
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
    return pipe


def display_name(key):
    return REGISTRY[key][0]


def needs_package(key):
    return REGISTRY[key][3]
