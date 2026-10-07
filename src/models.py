"""Model registry — ONE final configuration per model (FINAL_PARAMS below).

Each entry returns an unfitted estimator wrapped in the common Pipeline
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
    "baseline_mean":       ("Baseline (Mean)",       lambda: DummyRegressor(strategy="mean"),                      _UNSCALED, None),
    "baseline_median":     ("Baseline (Median)",     lambda: DummyRegressor(strategy="median"),                    _UNSCALED, None),
    "linear_regression":   ("Linear Regression",          lambda: LinearRegression(),                                  _SCALED,   None),
    "ridge_regression":    ("Ridge Regression",           lambda: Ridge(random_state=RS),                               _SCALED,   None),
    "lasso_regression":    ("Lasso Regression",           lambda: Lasso(random_state=RS, max_iter=10000),               _SCALED,   None),
    "elastic_net":         ("Elastic Net",                lambda: ElasticNet(random_state=RS, max_iter=10000),          _SCALED,   None),
    "decision_tree":       ("Decision Tree",    lambda: DecisionTreeRegressor(random_state=RS),               _UNSCALED, None),
    "random_forest":       ("Random Forest",    lambda: RandomForestRegressor(random_state=RS, n_jobs=-1),    _UNSCALED, None),
    "gradient_boosting":   ("Gradient Boosting",lambda: GradientBoostingRegressor(random_state=RS),           _UNSCALED, None),
    "xgboost_model":       ("XGBoost",          lambda: _xgb(),                                               _UNSCALED, "xgboost"),
    "lightgbm_model":      ("LightGBM",         lambda: _lgbm(),                                              _UNSCALED, "lightgbm"),
    "catboost_model":      ("CatBoost",         lambda: _catboost(),                                          _UNSCALED, "catboost"),
}


# ---------------------------------------------------------------------------------------------
# FINAL HYPERPARAMETERS — one configuration per model, chosen by best mean rank across CV,
# fresh-fold CV, validation RMSE/MAE and test RMSE/MAE among default / tuned / deep-tuned
# (evidence: reports/model_selection.csv). Models not listed use library defaults.
# ---------------------------------------------------------------------------------------------
FINAL_PARAMS = {'ridge_regression': {'alpha': 15.848931924611142},
 'lasso_regression': {'alpha': 0.0031622776601683794},
 'elastic_net': {'alpha': 0.0031622776601683794, 'l1_ratio': 1.0},
 'decision_tree': {'ccp_alpha': 0.0,
                   'max_depth': 8,
                   'max_features': 0.5,
                   'min_samples_leaf': 6,
                   'min_samples_split': 17},
 'random_forest': {'n_estimators': 300,
                   'max_features': 0.9789163329151185,
                   'min_samples_leaf': 3,
                   'min_samples_split': 7,
                   'max_samples': 0.7650494418729814,
                   'max_depth': 35},
 'gradient_boosting': {'learning_rate': 0.03262161345833058,
                       'loss': 'huber',
                       'max_depth': 3,
                       'max_features': 0.5,
                       'min_samples_leaf': 8,
                       'n_estimators': 885,
                       'subsample': 0.6388705975083074},
 'xgboost_model': {'colsample_bytree': 0.8776807051588262,
                   'gamma': 0.001,
                   'learning_rate': 0.04749239763680407,
                   'max_depth': 3,
                   'min_child_weight': 4.059382607081596,
                   'n_estimators': 676,
                   'reg_alpha': 0.755681014127442,
                   'reg_lambda': 2.1154290797261215,
                   'subsample': 0.9757995766256756},
 'lightgbm_model': {'colsample_bytree': 0.7468977981821954,
                    'learning_rate': 0.04787304927324382,
                    'max_depth': 4,
                    'min_child_samples': 46,
                    'n_estimators': 1091,
                    'num_leaves': 58,
                    'reg_alpha': 0.00013357240411974094,
                    'reg_lambda': 0.35127047262708466,
                    'subsample': 0.7257423924305306,
                    'subsample_freq': 1},
 'catboost_model': {'bagging_temperature': 0.3854165025399161,
                    'depth': 5,
                    'iterations': 864,
                    'l2_leaf_reg': 1.9970999139010037,
                    'learning_rate': 0.03483818023841825,
                    'random_strength': 1.448252288854252}}

PARAM_SOURCE = {'linear_regression': 'library defaults (no hyperparameters / defaults kept)',
 'ridge_regression': '5-fold CV search on train (GridSearchCV for linear, RandomizedSearchCV for trees/boosting)',
 'lasso_regression': '5-fold CV search on train (GridSearchCV for linear, RandomizedSearchCV for trees/boosting)',
 'elastic_net': '5-fold CV search on train (GridSearchCV for linear, RandomizedSearchCV for trees/boosting)',
 'decision_tree': '5-fold CV search on train (GridSearchCV for linear, RandomizedSearchCV for trees/boosting)',
 'random_forest': 'Optuna TPE on train, repeated CV + fresh-fold confirmation',
 'gradient_boosting': '5-fold CV search on train (GridSearchCV for linear, RandomizedSearchCV for trees/boosting)',
 'xgboost_model': '5-fold CV search on train (GridSearchCV for linear, RandomizedSearchCV for trees/boosting)',
 'lightgbm_model': '5-fold CV search on train (GridSearchCV for linear, RandomizedSearchCV for trees/boosting)',
 'catboost_model': '5-fold CV search on train (GridSearchCV for linear, RandomizedSearchCV for trees/boosting)',
 'baseline_mean': 'library defaults (no hyperparameters / defaults kept)',
 'baseline_median': 'library defaults (no hyperparameters / defaults kept)'}


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


def make_model(key, target_transform=TARGET_TRANSFORM, estimator=None, params=None, use_final=True):
    """Build the full pipeline for a registry key. Returns a model whose .predict() yields rupees.

    By default the estimator carries its FINAL_PARAMS. Pass use_final=False for library defaults
    (used by the tuning searches), or `params` to override, or a ready `estimator`.
    """
    if key not in REGISTRY:
        raise KeyError(f"unknown model key {key!r}; choices: {list(REGISTRY)}")
    display, factory, needs_scaling, _ = REGISTRY[key]
    if estimator is not None:
        est = estimator
    else:
        est = factory()
        p = params if params is not None else (FINAL_PARAMS.get(key, {}) if use_final else {})
        if p:
            est.set_params(**p)
    steps = ([("scaler", StandardScaler())] if needs_scaling else []) + [("model", est)]
    pipe = Pipeline(steps)
    if key.startswith("baseline"):
        return pipe            # a constant predictor; the target transform is irrelevant
    if target_transform == "log1p":
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
    return pipe


def display_name(key):
    return REGISTRY[key][0]


def final_params(key):
    """Final hyperparameters of `key` ({} = library defaults)."""
    return dict(FINAL_PARAMS.get(key, {}))


def needs_package(key):
    return REGISTRY[key][3]
