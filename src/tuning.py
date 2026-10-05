"""Hyperparameter tuning — every search is CV on the TRAIN split only (5-fold, same folds as the
leaderboard), with scaling + log1p target transform re-fit inside each fold. The validation and test
sets are never seen during a search. Scoring is RMSE in rupees.

- Ridge / Lasso / Elastic Net: exhaustive GridSearchCV over alpha (+ l1_ratio).
- Trees / boosting: RandomizedSearchCV over regularisation + capacity parameters.
- Mean/median baselines and plain Linear Regression have no hyperparameters to tune.
"""
import json, time
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, RepeatedKFold, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error

from . import config
from .models import make_model, REGISTRY

RS = config.RANDOM_STATE
_RMSE = make_scorer(lambda a, p: np.sqrt(mean_squared_error(a, p)), greater_is_better=False)
TUNED_PARAMS_FILE = config.REPORTS / "tuned_best_params.json"
UNTUNABLE = ("baseline_mean", "baseline_median", "linear_regression")


def _cv():
    return KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=RS)


def _param(grid):
    """Prefix parameter names for the estimator inside TransformedTargetRegressor -> Pipeline."""
    return {f"regressor__model__{k}": v for k, v in grid.items()}


# ---- exhaustive grids for the regularised linear models ----
REG_GRIDS = {
    "ridge_regression": {"alpha": np.logspace(-3, 3, 31)},
    "lasso_regression": {"alpha": np.logspace(-5, 0, 31)},
    "elastic_net":      {"alpha": np.logspace(-5, 0, 21), "l1_ratio": [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]},
}

# ---- randomized spaces (capacity + regularisation) ----
RANDOM_SPACES = {
    "decision_tree": {"max_depth": [None, 4, 6, 8, 10, 12, 15, 20], "min_samples_split": randint(2, 40),
                      "min_samples_leaf": randint(1, 30), "max_features": [None, "sqrt", 0.5, 0.7],
                      "ccp_alpha": [0.0, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3]},
    "random_forest": {"n_estimators": randint(200, 700), "max_depth": [None, 10, 15, 20, 30],
                      "min_samples_split": randint(2, 15), "min_samples_leaf": randint(1, 8),
                      "max_features": [0.3, 0.5, 0.7, 1.0, "sqrt"], "max_samples": [None, 0.6, 0.8]},
    "gradient_boosting": {"loss": ["squared_error", "huber"],
                          "n_estimators": randint(200, 900), "learning_rate": loguniform(0.01, 0.2),
                          "max_depth": randint(2, 7), "min_samples_leaf": randint(1, 20),
                          "subsample": uniform(0.6, 0.4), "max_features": [None, "sqrt", 0.5, 0.8]},
    "xgboost_model": {"n_estimators": randint(200, 1200), "learning_rate": loguniform(0.01, 0.2),
                      "max_depth": randint(2, 9), "min_child_weight": loguniform(0.5, 20),
                      "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.5, 0.5),
                      "reg_alpha": loguniform(1e-4, 1), "reg_lambda": loguniform(1e-2, 10),
                      "gamma": [0.0, 0.001, 0.01, 0.05]},
    "lightgbm_model": {"n_estimators": randint(200, 1200), "learning_rate": loguniform(0.01, 0.2),
                       "num_leaves": randint(8, 80), "max_depth": [-1, 4, 6, 8, 10],
                       "min_child_samples": randint(5, 60), "subsample": uniform(0.6, 0.4),
                       "subsample_freq": [1], "colsample_bytree": uniform(0.5, 0.5),
                       "reg_alpha": loguniform(1e-4, 1), "reg_lambda": loguniform(1e-3, 10)},
    "catboost_model": {"iterations": randint(300, 1200), "learning_rate": loguniform(0.02, 0.2),
                       "depth": randint(4, 9), "l2_leaf_reg": loguniform(1, 20),
                       "random_strength": loguniform(0.1, 5), "bagging_temperature": uniform(0, 1)},
}

# search budget per model (fits = n_iter x 5 folds)
N_ITER = {"decision_tree": 100, "random_forest": 40, "gradient_boosting": 60,
          "xgboost_model": 100, "lightgbm_model": 100, "catboost_model": 40}

# single-thread the estimator while the search parallelises over candidates (avoids oversubscription)
_THREAD_PARAM = {"random_forest": "n_jobs", "xgboost_model": "n_jobs", "lightgbm_model": "n_jobs",
                 "catboost_model": "thread_count"}


def tune(key, data, n_iter=None):
    """Run the CV search for one model on the training split. Returns the fitted search object."""
    model = make_model(key, data["target_transform"])
    if key in _THREAD_PARAM:
        model.set_params(**{f"regressor__model__{_THREAD_PARAM[key]}": 1})
    if key in REG_GRIDS:
        search = GridSearchCV(model, _param(REG_GRIDS[key]), scoring=_RMSE, cv=_cv(), n_jobs=-1)
    elif key in RANDOM_SPACES:
        search = RandomizedSearchCV(model, _param(RANDOM_SPACES[key]), n_iter=n_iter or N_ITER[key],
                                    scoring=_RMSE, cv=_cv(), random_state=RS, n_jobs=-1)
    else:
        raise ValueError(f"{key} has no hyperparameters to tune")
    search.fit(data["X_train"], data["y_train"])
    return search


def best_params(search):
    """Strip the pipeline prefix and convert numpy scalars to plain Python for JSON."""
    out = {}
    for k, v in search.best_params_.items():
        v = v.item() if hasattr(v, "item") else v
        out[k.split("__")[-1]] = v
    return out


def best_cv_rmse(search):
    return -search.best_score_


def save_tuned_params(params_by_key):
    existing = load_tuned_params()
    existing.update(params_by_key)
    TUNED_PARAMS_FILE.write_text(json.dumps(existing, indent=2, default=str))


def load_tuned_params():
    return json.loads(TUNED_PARAMS_FILE.read_text()) if TUNED_PARAMS_FILE.exists() else {}


def build_tuned_model(key, target_transform, params=None):
    """Full-thread pipeline for `key` with tuned hyperparameters (from file if params is None)."""
    params = params if params is not None else load_tuned_params().get(key, {})
    est = REGISTRY[key][1]()
    if params:
        est.set_params(**params)
    return make_model(key, target_transform, estimator=est)


def repeated_cv_rmse(model, data, n_splits=5, n_repeats=3):
    """Repeated K-fold RMSE on train — a more stable ranking signal than one 5-fold run."""
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RS)
    s = -cross_val_score(model, data["X_train"], data["y_train"], cv=cv, scoring=_RMSE, n_jobs=-1)
    return float(s.mean()), float(s.std())


# --- backward-compatible helpers used by notebooks/regularization.ipynb ---
def regularization_search(key, data):
    return tune(key, data)


def randomized_search(key, data, n_iter=15):
    return tune(key, data, n_iter=n_iter)
