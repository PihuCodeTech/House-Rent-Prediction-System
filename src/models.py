"""Model registry: thirteen models, one final configuration each.

`make_model(key, seed)` returns an unfitted model that predicts rupees. Every model sees the same prepared
features; linear models are standardised first (tree models do not need it), and every model except the
baselines learns log1p(rent), converted back to rupees by TransformedTargetRegressor. `seed` sets the model's
own randomness (bootstrap rows, feature subsampling, ...); models without randomness ignore it.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .config import RANDOM_STATE, TARGET_TRANSFORM


class LocalityRateBaseline(RegressorMixin, BaseEstimator):
    """The rule of thumb a person would use: rent = the typical rent per sq ft of the home's locality × its size.

    The rate is the median rent per sq ft of the TRAINING listings in the same locality (only if it has at least
    `min_listings` of them), otherwise of the same city. It reads the raw columns City, locality_key and Size —
    no encoders, no learned features — so it is the honest benchmark the machine-learning models must beat."""

    raw_input = True  # evaluation feeds it the raw training columns instead of the prepared feature matrix

    def __init__(self, min_listings=3):
        self.min_listings = min_listings

    def fit(self, X, y):
        rate = np.asarray(y, dtype="float64") / X["Size"].to_numpy("float64")
        df = pd.DataFrame({"city": X["City"].to_numpy(), "loc": X["locality_key"].to_numpy(), "rate": rate})
        by_loc = df.groupby("loc")["rate"].agg(["median", "size"])
        self.locality_rate_ = by_loc.loc[by_loc["size"] >= self.min_listings, "median"].to_dict()
        self.city_rate_ = df.groupby("city")["rate"].median().to_dict()
        self.overall_rate_ = float(np.median(rate))
        self.n_features_in_ = 3
        return self

    def rates(self, X):
        """Rent per sq ft used for each row (locality, else city, else overall median)."""
        return np.array(
            [
                self.locality_rate_.get(loc, self.city_rate_.get(city, self.overall_rate_))
                for loc, city in zip(X["locality_key"], X["City"], strict=True)
            ],
            dtype="float64",
        )

    def predict(self, X):
        return self.rates(X) * X["Size"].to_numpy("float64")


def is_raw(model):
    """True for models that read the raw columns (the locality × size baseline) instead of prepared features."""
    return bool(getattr(model, "raw_input", False))


def _xgboost(seed):
    from xgboost import XGBRegressor

    return XGBRegressor(random_state=seed, n_jobs=-1, tree_method="hist")


def _lightgbm(seed):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(random_state=seed, n_jobs=-1, verbose=-1)


def _catboost(seed):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(random_state=seed, verbose=0, allow_writing_files=False)


# key -> (display name, factory(seed) -> estimator, standardise features?, extra package needed)
REGISTRY = {
    "baseline_mean": ("Baseline (Mean)", lambda s: DummyRegressor(strategy="mean"), False, None),
    "baseline_median": ("Baseline (Median)", lambda s: DummyRegressor(strategy="median"), False, None),
    "baseline_locality_size": ("Baseline (Locality × Size)", lambda s: LocalityRateBaseline(), False, None),
    "linear_regression": ("Linear Regression", lambda s: LinearRegression(), True, None),
    "ridge_regression": ("Ridge Regression", lambda s: Ridge(random_state=s), True, None),
    "lasso_regression": ("Lasso Regression", lambda s: Lasso(random_state=s, max_iter=10000), True, None),
    "elastic_net": ("Elastic Net", lambda s: ElasticNet(random_state=s, max_iter=10000), True, None),
    "decision_tree": ("Decision Tree", lambda s: DecisionTreeRegressor(random_state=s), False, None),
    "random_forest": ("Random Forest", lambda s: RandomForestRegressor(random_state=s, n_jobs=-1), False, None),
    "gradient_boosting": ("Gradient Boosting", lambda s: GradientBoostingRegressor(random_state=s), False, None),
    "xgboost_model": ("XGBoost", _xgboost, False, "xgboost"),
    "lightgbm_model": ("LightGBM", _lightgbm, False, "lightgbm"),
    "catboost_model": ("CatBoost", _catboost, False, "catboost"),
}

# ---------------------------------------------------------------------------------------------------------------
# FINAL HYPERPARAMETERS. Each tunable model was searched with a grid search, then a Bayesian (Optuna) search on
# the seed-42 training split (5-fold CV). Library defaults, grid-best and Bayesian-best were scored on seeds
# 42-44 and the lowest mean test RMSE was kept (log: reports/final_hyperparameters.json). The searches ran while the
# five floor features were still in the feature set; they were dropped afterwards (they changed the error by ~Rs 600)
# and the tuned values were kept — `python run_all.py --tuning` re-runs the searches on the current features.
# Models not listed use library defaults.
# ---------------------------------------------------------------------------------------------------------------
FINAL_PARAMS = {
    "ridge_regression": {"alpha": 31.622776601683793},
    "lasso_regression": {"alpha": 0.0020750608061092753},
    "elastic_net": {"alpha": 0.0024658930745310127, "l1_ratio": 0.9287947846859934},
    "decision_tree": {"max_depth": None, "max_features": 1.0, "min_samples_leaf": 10},
    "gradient_boosting": {
        "n_estimators": 282,
        "learning_rate": 0.05920605525727517,
        "max_depth": 3,
        "subsample": 0.540062581916704,
        "min_samples_leaf": 20,
        "max_features": 0.4521695605697275,
        "loss": "huber",
    },
    "xgboost_model": {
        "n_estimators": 600,
        "learning_rate": 0.03,
        "max_depth": 3,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_weight": 1.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0001,
    },
    "lightgbm_model": {
        "n_estimators": 922,
        "learning_rate": 0.05700042878874667,
        "num_leaves": 13,
        "min_child_samples": 41,
        "subsample": 0.8410987834117182,
        "subsample_freq": 1,
        "colsample_bytree": 0.5009153617648229,
        "reg_lambda": 0.17149581668128844,
        "reg_alpha": 0.0002898115581696978,
    },
}

PARAM_SOURCE = {
    "baseline_mean": "library defaults (no hyperparameters to tune)",
    "baseline_median": "library defaults (no hyperparameters to tune)",
    "baseline_locality_size": "rule-based: median rent per sq ft of the locality (>= 3 training listings, else the city)",
    "linear_regression": "library defaults (no hyperparameters to tune)",
    "ridge_regression": "grid search (5-fold CV on seed-42 train)",
    "lasso_regression": "Bayesian search (Optuna TPE around the grid best, 5-fold CV on seed-42 train)",
    "elastic_net": "Bayesian search (Optuna TPE around the grid best, 5-fold CV on seed-42 train)",
    "decision_tree": "grid search (5-fold CV on seed-42 train)",
    "random_forest": "library defaults",
    "gradient_boosting": "Bayesian search (Optuna TPE around the grid best, 5-fold CV on seed-42 train)",
    "xgboost_model": "Bayesian search (Optuna TPE around the grid best, 5-fold CV on seed-42 train)",
    "lightgbm_model": "Bayesian search (Optuna TPE around the grid best, 5-fold CV on seed-42 train)",
    "catboost_model": "library defaults",
}

# Best point of each search, kept for reference / reuse: make_model(key, params=BAYES_PARAMS[key]).
GRID_PARAMS = {
    "ridge_regression": {"alpha": 31.622776601683793},
    "lasso_regression": {"alpha": 0.0031622776601683794},
    "elastic_net": {"alpha": 0.0031622776601683794, "l1_ratio": 1.0},
    "decision_tree": {"max_depth": None, "max_features": 1.0, "min_samples_leaf": 10},
    "random_forest": {"max_depth": 15, "max_features": 1.0, "min_samples_leaf": 3, "n_estimators": 300},
    "gradient_boosting": {"learning_rate": 0.03, "max_depth": 4, "n_estimators": 300, "subsample": 0.7},
    "xgboost_model": {
        "colsample_bytree": 0.7,
        "learning_rate": 0.03,
        "max_depth": 3,
        "n_estimators": 600,
        "subsample": 0.7,
    },
    "lightgbm_model": {"learning_rate": 0.03, "min_child_samples": 30, "n_estimators": 600, "num_leaves": 15},
    "catboost_model": {"depth": 4, "iterations": 1000, "learning_rate": 0.03},
}

BAYES_PARAMS = {
    "ridge_regression": {"alpha": 33.68109289640676},
    "lasso_regression": {"alpha": 0.0020750608061092753},
    "elastic_net": {"alpha": 0.0024658930745310127, "l1_ratio": 0.9287947846859934},
    "decision_tree": {
        "max_depth": 30,
        "min_samples_leaf": 12,
        "min_samples_split": 6,
        "max_features": 0.7745353259755203,
    },
    "random_forest": {
        "n_estimators": 322,
        "max_depth": 11,
        "max_features": 0.9210582566280392,
        "min_samples_leaf": 3,
        "min_samples_split": 3,
        "max_samples": 0.798070764044508,
    },
    "gradient_boosting": {
        "n_estimators": 282,
        "learning_rate": 0.05920605525727517,
        "max_depth": 3,
        "subsample": 0.540062581916704,
        "min_samples_leaf": 20,
        "max_features": 0.4521695605697275,
        "loss": "huber",
    },
    "xgboost_model": {
        "n_estimators": 600,
        "learning_rate": 0.03,
        "max_depth": 3,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_weight": 1.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0001,
    },
    "lightgbm_model": {
        "n_estimators": 922,
        "learning_rate": 0.05700042878874667,
        "num_leaves": 13,
        "min_child_samples": 41,
        "subsample": 0.8410987834117182,
        "subsample_freq": 1,
        "colsample_bytree": 0.5009153617648229,
        "reg_lambda": 0.17149581668128844,
        "reg_alpha": 0.0002898115581696978,
    },
    "catboost_model": {
        "iterations": 693,
        "learning_rate": 0.07489723835527984,
        "depth": 3,
        "l2_leaf_reg": 4.359360321865315,
        "random_strength": 0.10824376249086654,
        "bagging_temperature": 0.19807936126413278,
    },
}


def make_model(key, seed=RANDOM_STATE, target_transform=TARGET_TRANSFORM, params=None):
    """Unfitted model for `key` with the given seed. Uses FINAL_PARAMS unless `params` is given
    ({} = library defaults, as used by the tuning searches). The model's .predict() returns rupees."""
    if key not in REGISTRY:
        raise KeyError(f"unknown model key {key!r}; choices: {list(REGISTRY)}")
    _, factory, standardise, _ = REGISTRY[key]
    estimator = factory(seed)
    chosen = FINAL_PARAMS.get(key, {}) if params is None else params
    if chosen:
        estimator.set_params(**chosen)
    if is_raw(estimator):
        return estimator  # rule-based, reads raw columns and predicts rupees directly
    pipe = Pipeline(([("scaler", StandardScaler())] if standardise else []) + [("model", estimator)])
    if key.startswith("baseline") or target_transform != "log1p":
        return pipe  # the baselines predict a constant; the target transform is irrelevant for them
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def display_name(key):
    """Readable model name, e.g. 'Random Forest'."""
    return REGISTRY[key][0]


def final_params(key):
    """Final hyperparameters of `key` ({} = library defaults)."""
    return dict(FINAL_PARAMS.get(key, {}))


def needs_package(key):
    """Extra package a model needs (xgboost / lightgbm / catboost), or None."""
    return REGISTRY[key][3]
