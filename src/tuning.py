"""Hyperparameter tuning: a coarse grid search, then a Bayesian (Optuna TPE) search centred on the best grid point.

Both searches minimise 5-fold CV RMSE on the seed-42 TRAINING split only. The whole pipeline, preprocessing
included, is re-fit inside every fold, so validation and test rows are never seen. The best grid point is
the first Bayesian trial, so the Bayesian result is never worse than the grid result on CV.
`compare_on_seeds` then scores library defaults, grid-best and Bayesian-best on every seed.
"""

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

from . import config
from .evaluation import evaluate, with_preprocessor
from .models import make_model
from .preprocessing import prepare_data

TUNABLE = [
    "ridge_regression",
    "lasso_regression",
    "elastic_net",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "xgboost_model",
    "lightgbm_model",
    "catboost_model",
]
# Linear Regression and the two baselines have no hyperparameters to tune.

GRIDS = {
    "ridge_regression": {"alpha": list(np.logspace(-3, 3, 13))},
    "lasso_regression": {"alpha": list(np.logspace(-5, 0, 11))},
    "elastic_net": {"alpha": list(np.logspace(-5, 0, 11)), "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]},
    "decision_tree": {"max_depth": [4, 6, 8, 10, None], "min_samples_leaf": [1, 5, 10, 20], "max_features": [0.5, 1.0]},
    "random_forest": {
        "n_estimators": [300],
        "max_depth": [None, 15],
        "max_features": [0.33, 0.6, 1.0],
        "min_samples_leaf": [1, 3, 5],
    },
    "gradient_boosting": {
        "n_estimators": [300, 600],
        "learning_rate": [0.03, 0.1],
        "max_depth": [2, 3, 4],
        "subsample": [0.7, 1.0],
    },
    "xgboost_model": {
        "n_estimators": [300, 600],
        "learning_rate": [0.03, 0.1],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0],
    },
    "lightgbm_model": {
        "n_estimators": [300, 600],
        "learning_rate": [0.03, 0.1],
        "num_leaves": [15, 31, 63],
        "min_child_samples": [10, 30],
    },
    "catboost_model": {"iterations": [500, 1000], "learning_rate": [0.03, 0.1], "depth": [4, 6, 8]},
}
N_TRIALS = {
    "ridge_regression": 30,
    "lasso_regression": 30,
    "elastic_net": 30,
    "decision_tree": 40,
    "random_forest": 25,
    "gradient_boosting": 30,
    "xgboost_model": 40,
    "lightgbm_model": 40,
    "catboost_model": 25,
}
_SINGLE_THREAD = {"ridge_regression", "lasso_regression", "elastic_net", "decision_tree", "gradient_boosting"}


def _prefix(key):
    return "regressor__model__"  # every tunable model is TransformedTargetRegressor(Pipeline(..., model))


def _cv(seed):
    return KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=seed)


def cv_rmse(key, params, data):
    """5-fold CV RMSE on the training split, preprocessing re-fit inside every fold."""
    seed = data["seed"]
    model = with_preprocessor(make_model(key, seed=seed, params=params), seed)
    s = cross_val_score(
        model,
        data["X_train_raw"],
        data["y_train"],
        cv=_cv(seed),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1 if key in _SINGLE_THREAD else 1,
    )
    return float(-s.mean())


def grid_search(key, data):
    """Exhaustive grid search (5-fold CV RMSE). Returns (best params, best CV RMSE, number of candidates)."""
    seed = data["seed"]
    est = with_preprocessor(make_model(key, seed=seed, params={}), seed)
    grid = {_prefix(key) + k: v for k, v in GRIDS[key].items()}
    gs = GridSearchCV(
        est,
        grid,
        scoring="neg_root_mean_squared_error",
        cv=_cv(seed),
        refit=False,
        n_jobs=-1 if key in _SINGLE_THREAD else 1,
    )
    gs.fit(data["X_train_raw"], data["y_train"])
    best = {k[len(_prefix(key)) :]: _py(v) for k, v in gs.best_params_.items()}
    return best, float(-gs.best_score_), len(gs.cv_results_["params"])


def _py(v):
    return v.item() if hasattr(v, "item") else v


# ---- Bayesian search spaces, centred on the best grid point g ----
def _log(t, name, c, f, lo=None, hi=None):
    a, b = c / f, c * f
    return t.suggest_float(name, max(a, lo) if lo else a, min(b, hi) if hi else b, log=True)


def _int(t, name, lo, hi):
    lo, hi = int(lo), int(max(hi, lo))
    return t.suggest_int(name, lo, hi)


def _flt(t, name, lo, hi):
    return t.suggest_float(name, float(lo), float(max(hi, lo)))


def _space(key, t, g):
    if key in ("ridge_regression", "lasso_regression"):
        return {"alpha": _log(t, "alpha", g["alpha"], 10)}
    if key == "elastic_net":
        return {
            "alpha": _log(t, "alpha", g["alpha"], 10),
            "l1_ratio": _flt(t, "l1_ratio", max(0.05, g["l1_ratio"] - 0.2), min(1.0, g["l1_ratio"] + 0.2)),
        }
    if key == "decision_tree":
        d = g["max_depth"]
        return {
            "max_depth": _int(t, "max_depth", 12, 30) if d is None else _int(t, "max_depth", max(2, d - 2), d + 2),
            "min_samples_leaf": _int(
                t, "min_samples_leaf", max(1, g["min_samples_leaf"] // 2), 2 * g["min_samples_leaf"] + 2
            ),
            "min_samples_split": _int(t, "min_samples_split", 2, 30),
            "max_features": _flt(t, "max_features", 0.3, 1.0),
        }
    if key == "random_forest":
        d = g["max_depth"]
        return {
            "n_estimators": _int(t, "n_estimators", 200, 600),
            "max_depth": _int(t, "max_depth", 15, 50) if d is None else _int(t, "max_depth", max(5, d - 5), d + 10),
            "max_features": _flt(
                t, "max_features", max(0.2, g["max_features"] - 0.25), min(1.0, g["max_features"] + 0.25)
            ),
            "min_samples_leaf": _int(
                t, "min_samples_leaf", max(1, g["min_samples_leaf"] - 2), g["min_samples_leaf"] + 3
            ),
            "min_samples_split": _int(t, "min_samples_split", 2, 12),
            "max_samples": _flt(t, "max_samples", 0.6, 1.0),
        }
    if key == "gradient_boosting":
        return {
            "n_estimators": _int(t, "n_estimators", max(100, g["n_estimators"] // 2), g["n_estimators"] * 2),
            "learning_rate": _log(t, "learning_rate", g["learning_rate"], 3),
            "max_depth": _int(t, "max_depth", max(2, g["max_depth"] - 1), g["max_depth"] + 1),
            "subsample": _flt(t, "subsample", max(0.5, g["subsample"] - 0.2), min(1.0, g["subsample"] + 0.2)),
            "min_samples_leaf": _int(t, "min_samples_leaf", 1, 20),
            "max_features": _flt(t, "max_features", 0.4, 1.0),
            "loss": t.suggest_categorical("loss", ["squared_error", "huber"]),
        }
    if key == "xgboost_model":
        return {
            "n_estimators": _int(t, "n_estimators", g["n_estimators"] // 2, int(g["n_estimators"] * 2.5)),
            "learning_rate": _log(t, "learning_rate", g["learning_rate"], 3),
            "max_depth": _int(t, "max_depth", max(2, g["max_depth"] - 1), g["max_depth"] + 1),
            "subsample": _flt(t, "subsample", max(0.5, g["subsample"] - 0.2), min(1.0, g["subsample"] + 0.2)),
            "colsample_bytree": _flt(
                t, "colsample_bytree", max(0.4, g["colsample_bytree"] - 0.2), min(1.0, g["colsample_bytree"] + 0.2)
            ),
            "min_child_weight": t.suggest_float("min_child_weight", 0.5, 10, log=True),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "reg_alpha": t.suggest_float("reg_alpha", 1e-4, 10, log=True),
        }
    if key == "lightgbm_model":
        return {
            "n_estimators": _int(t, "n_estimators", g["n_estimators"] // 2, int(g["n_estimators"] * 2.5)),
            "learning_rate": _log(t, "learning_rate", g["learning_rate"], 3),
            "num_leaves": _int(t, "num_leaves", max(7, g["num_leaves"] // 2), g["num_leaves"] * 2),
            "min_child_samples": _int(
                t, "min_child_samples", max(5, g["min_child_samples"] // 2), g["min_child_samples"] * 2
            ),
            "subsample": _flt(t, "subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": _flt(t, "colsample_bytree", 0.5, 1.0),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "reg_alpha": t.suggest_float("reg_alpha", 1e-4, 10, log=True),
        }
    if key == "catboost_model":
        return {
            "iterations": _int(t, "iterations", g["iterations"] // 2, g["iterations"] * 2),
            "learning_rate": _log(t, "learning_rate", g["learning_rate"], 3),
            "depth": _int(t, "depth", max(3, g["depth"] - 1), min(10, g["depth"] + 1)),
            "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1, 10, log=True),
            "random_strength": t.suggest_float("random_strength", 0.1, 5, log=True),
            "bagging_temperature": t.suggest_float("bagging_temperature", 0.0, 1.0),
        }
    raise KeyError(key)


# Library-default values of the parameters the Bayesian spaces add on top of the grid, so the first trial is
# exactly the best grid point (0 is mapped to the lower end of a log range).
_EXTRA_DEFAULTS = {
    "decision_tree": {"min_samples_split": 2},
    "random_forest": {"min_samples_split": 2, "max_samples": 1.0},
    "gradient_boosting": {"min_samples_leaf": 1, "max_features": 1.0, "loss": "squared_error"},
    "xgboost_model": {"min_child_weight": 1.0, "reg_lambda": 1.0, "reg_alpha": 1e-4},
    "lightgbm_model": {"subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 1e-3, "reg_alpha": 1e-4},
    "catboost_model": {"l2_leaf_reg": 3.0, "random_strength": 1.0, "bagging_temperature": 1.0},
}


def _grid_as_trial(key, g):
    """The grid best expressed in the Bayesian space (first trial)."""
    t = {**_EXTRA_DEFAULTS.get(key, {}), **g}
    if key == "decision_tree" and g.get("max_depth") is None:
        t["max_depth"] = 30
    if key == "random_forest" and g.get("max_depth") is None:
        t["max_depth"] = 50
    return t


def bayes_search(key, data, grid_best, n_trials=None, seed=config.RANDOM_STATE):
    """Optuna TPE search around the grid best (first trial = grid best). Returns (best params, best CV RMSE, trials)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial(_grid_as_trial(key, grid_best), skip_if_exists=True)

    def objective(t):
        return cv_rmse(key, _space(key, t, grid_best), data)

    study.optimize(objective, n_trials=n_trials or N_TRIALS[key])
    best = _space(key, optuna.trial.FixedTrial(study.best_params), grid_best)
    return {k: _py(v) for k, v in best.items()}, float(study.best_value), len(study.trials)


def tune(key, data=None, verbose=True):
    """Grid search, then Bayesian search around the grid best, on the seed-42 training split."""
    data = data or prepare_data(seed=config.RANDOM_STATE)
    t0 = time.time()
    g, g_cv, n_g = grid_search(key, data)
    t1 = time.time()
    b, b_cv, n_b = bayes_search(key, data, g)
    out = {
        "grid": {"params": g, "cv_rmse": g_cv, "n_candidates": n_g, "time_s": round(t1 - t0, 1)},
        "bayes": {"params": b, "cv_rmse": b_cv, "n_trials": n_b, "time_s": round(time.time() - t1, 1)},
    }
    if verbose:
        print(
            f"{key}: grid CV {g_cv:,.2f} ({n_g} candidates, {t1 - t0:.0f}s) -> "
            f"Bayesian CV {b_cv:,.2f} ({n_b} trials, {time.time() - t1:.0f}s)",
            flush=True,
        )
    return out


def compare_on_seeds(key, candidates, seeds=config.SEEDS, clean=None):
    """Score each {label: params} on every seed (fit on train, CV on train, val, test).
    Returns one row per candidate with mean scores over the seeds."""
    datas = {s: prepare_data(seed=s, clean=clean) for s in seeds}
    rows = []
    for label, params in candidates.items():
        per = []
        for s in seeds:
            sc, _ = evaluate(make_model(key, seed=s, params=params), datas[s])
            per.append(
                {
                    "cv": sc["cv"]["RMSE"],
                    "val": sc["val"]["RMSE"],
                    "test": sc["test"]["RMSE"],
                    "test_mae": sc["test"]["MAE"],
                }
            )
        per = pd.DataFrame(per)
        rows.append(
            {
                "model": key,
                "candidate": label,
                "cv_rmse": per.cv.mean(),
                "val_rmse": per.val.mean(),
                "test_rmse": per.test.mean(),
                "test_rmse_std": per.test.std(ddof=1),
                "test_mae": per.test_mae.mean(),
            }
        )
    return pd.DataFrame(rows)
