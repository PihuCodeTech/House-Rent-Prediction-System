"""Evaluation: rupee-scale metrics, 5-fold CV on the training split (preprocessing re-fit inside every fold),
multi-seed runs, and the single results file reports/results.csv (mean ± std over the seeds)."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

from . import config
from .models import PARAM_SOURCE, REGISTRY, display_name, final_params, is_raw, make_model
from .prediction import SeedAveragedModel, build_error_profile
from .preprocessing import assert_run_consistency, build_preprocessor, prepare_data

RESULT_COLUMNS = [
    "Model",
    "Train RMSE",
    "CV RMSE",
    "Val RMSE",
    "Test RMSE",
    "Test R²",
    "Test MAE",
    "Test Median % Error",
]
_METRIC_KEYS = {
    "Train RMSE": "train_rmse",
    "CV RMSE": "cv_rmse",
    "Val RMSE": "val_rmse",
    "Test RMSE": "test_rmse",
    "Test R²": "test_r2",
    "Test MAE": "test_mae",
    "Test Median % Error": "test_mdape",
}


def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def median_pct_error(y_true, y_pred):
    """Median absolute error as a percentage of the actual rent (MdAPE). Unlike RMSE in rupees, it is not
    dominated by a few very expensive homes: half the predictions are closer than this, half further."""
    y_true, y_pred = np.asarray(y_true, dtype="float64"), np.asarray(y_pred, dtype="float64")
    return float(np.median(np.abs(y_true - y_pred) / y_true) * 100)


def metrics(y_true, y_pred):
    """RMSE, MAE and R² in rupees, and the median % error."""
    return {
        "RMSE": _rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MdAPE": median_pct_error(y_true, y_pred),
    }


# ----------------------------- cross-validation -----------------------------
def with_preprocessor(model, seed):
    """The same model with a fresh (unfitted) preprocessor as its first step, so it can be trained on the
    raw training columns. Inside a TransformedTargetRegressor the preprocessor sees log1p(rent), exactly
    as in prepare_data(); used to re-fit encoders/imputers/clippers inside every CV fold."""
    model = clone(model)
    pre = ("prep", build_preprocessor(seed=seed))
    if isinstance(model, TransformedTargetRegressor):
        return model.set_params(regressor=Pipeline([pre] + list(model.regressor.steps)))
    return Pipeline([pre] + list(model.steps))


def cross_val_rmse(model, data, folds=config.CV_FOLDS):
    """5-fold CV on the TRAINING split only. Each fold re-fits the whole pipeline — preprocessor
    (locality target encoding, size clipping, floor imputation, one-hot), scaler, target transform and
    model — on the fold's training rows, so the held-out fold never influences any fitted step."""
    seed = data["seed"]
    cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
    scoring = {
        "rmse": make_scorer(_rmse, greater_is_better=False),
        "mae": make_scorer(mean_absolute_error, greater_is_better=False),
        "r2": "r2",
    }
    pipeline = clone(model) if is_raw(model) else with_preprocessor(model, seed)
    cvr = cross_validate(pipeline, data["X_train_raw"], data["y_train"], cv=cv, scoring=scoring)
    return {
        "RMSE": float(-cvr["test_rmse"].mean()),
        "RMSE_std": float(cvr["test_rmse"].std()),
        "MAE": float(-cvr["test_mae"].mean()),
        "R2": float(cvr["test_r2"].mean()),
    }


# ----------------------------- one seed -----------------------------
def evaluate(model, data):
    """Fit on train, score train / val / test, run 5-fold CV. Returns (scores, predictions). Rule-based models
    (the locality × size baseline) read the raw columns of the same rows instead of the prepared features."""
    suffix = "_raw" if is_raw(model) else ""
    model.fit(data[f"X_train{suffix}"], data["y_train"])
    pred = {s: model.predict(data[f"X_{s}{suffix}"]) for s in ("train", "val", "test")}
    scores = {s: metrics(data[f"y_{s}"], pred[s]) for s in pred}
    scores["cv"] = cross_val_rmse(model, data)
    return scores, pred


def _flat(seed, scores):
    return {
        "seed": seed,
        "train_rmse": scores["train"]["RMSE"],
        "cv_rmse": scores["cv"]["RMSE"],
        "val_rmse": scores["val"]["RMSE"],
        "test_rmse": scores["test"]["RMSE"],
        "test_r2": scores["test"]["R2"],
        "test_mae": scores["test"]["MAE"],
        "test_mdape": scores["test"]["MdAPE"],
    }


# ----------------------------- all seeds -----------------------------
def run_seeds(key, clean=None, seeds=config.SEEDS, verbose=True):
    """Run one model on every seed. Each seed changes the train/val/test split, the CV folds, the locality
    encoder's internal folds and the model's own randomness. Returns (per_seed DataFrame, artifacts of the
    first seed: {"model", "pred", "data"} for diagnostics, members: one fitted pipeline per seed, tests: each seed's
    (actual, predicted) test rents — every prediction from the member that never trained on those rows)."""
    rows, first, members, tests = [], None, [], []
    for seed in seeds:
        data = prepare_data(seed=seed, clean=clean)
        assert_run_consistency(data)
        model = make_model(key, seed=seed, target_transform=data["target_transform"])
        scores, pred = evaluate(model, data)
        rows.append(_flat(seed, scores))
        tests.append((data["y_test"].to_numpy(), pred["test"]))
        raw = is_raw(model)
        members.append(
            {
                "seed": seed,
                "model": model,
                "preprocessor": None if raw else data["preprocessor"],  # rule-based: reads raw columns
                "feature_names": None if raw else list(data["feature_names"]),
            }
        )
        if first is None:
            first = {"model": model, "pred": pred, "data": data}
        if verbose:
            r = rows[-1]
            print(
                f"  seed {seed}: train {r['train_rmse']:>10,.2f} | CV {r['cv_rmse']:>10,.2f} | "
                f"val {r['val_rmse']:>10,.2f} | test {r['test_rmse']:>10,.2f} | "
                f"R² {r['test_r2']:.2f} | MAE {r['test_mae']:>9,.2f} | MdAPE {r['test_mdape']:.1f}%",
                flush=True,
            )
    return pd.DataFrame(rows), first, members, tests


def summarize(per_seed, model_name):
    """One results row: every metric as 'mean ± std' (sample std across seeds, 2 decimals)."""
    row = {"Model": model_name}
    for col, k in _METRIC_KEYS.items():
        v = per_seed[k].to_numpy(dtype="float64")
        sd = v.std(ddof=1) if len(v) > 1 else 0.0
        row[col] = f"{round(v.mean(), 2) + 0.0:.2f} ± {round(sd, 2) + 0.0:.2f}"  # +0.0 avoids "-0.00"
    return row


def update_results(row, path=None):
    """Insert/replace this model's row in reports/results.csv (rows kept in registry order)."""
    path = path or config.RESULTS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        table = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
        table = table[table["Model"] != row["Model"]]
    else:
        table = pd.DataFrame(columns=RESULT_COLUMNS)
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)[RESULT_COLUMNS]
    order = {display_name(k): i for i, k in enumerate(REGISTRY)}
    table = table.sort_values("Model", key=lambda s: s.map(order).fillna(len(order))).reset_index(drop=True)
    table.to_csv(path, index=False, encoding="utf-8-sig")  # BOM so Excel shows ± and ² correctly
    return table


def read_results(path=None):
    """reports/results.csv as text, plus numeric mean/std columns for plotting."""
    path = path or config.RESULTS_CSV
    if not path.exists():
        raise FileNotFoundError("reports/results.csv not found — run the model notebooks (or run_all.py) first.")
    table = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    num = pd.DataFrame({"Model": table["Model"]})
    for col in RESULT_COLUMNS[1:]:
        if col not in table.columns or table[col].isna().any():  # e.g. a file written before the column existed
            continue
        parts = table[col].str.split(" ± ", expand=True).astype("float64")
        num[f"{col} mean"], num[f"{col} std"] = parts[0], parts[1]
    return table, num


def run_model(key, clean=None, seeds=config.SEEDS, verbose=True):
    """run_seeds + summarize + update_results. Returns (results row, per_seed, first-seed artifacts,
    averaged model). The averaged model (one fitted pipeline per seed, mean prediction) is saved by the caller
    with src.prediction.save_model."""
    name = display_name(key)
    if verbose:
        print(f"{name} — {len(seeds)} seeds {seeds}", flush=True)
    per_seed, first, members, tests = run_seeds(key, clean=clean, seeds=seeds, verbose=verbose)
    row = summarize(per_seed, name)
    update_results(row)
    averaged = SeedAveragedModel(
        key,
        name,
        members,
        params=final_params(key),
        source=PARAM_SOURCE[key],
        results=row,
        error_profile=build_error_profile(tests),
    )
    if verbose:
        print("  mean ± std  →  " + " | ".join(f"{c} {row[c]}" for c in RESULT_COLUMNS[1:]), flush=True)
    return row, per_seed, first, averaged


# ----------------------------- the final model on its test splits -----------------------------
RENT_BANDS = [
    (0, 10_000, "under Rs 10k"),
    (10_000, 20_000, "Rs 10k–20k"),
    (20_000, 40_000, "Rs 20k–40k"),
    (40_000, 100_000, "Rs 40k–1L"),
    (100_000, np.inf, "Rs 1L and above"),
]


def final_test_predictions(averaged, clean=None):
    """Every seed member of a SeedAveragedModel scored on its own seed's test split (rows it never trained on).
    Returns a DataFrame: seed, City, actual, predicted, residual (actual − predicted, rupees)."""
    if clean is None:
        from .data_cleaning import apply_stateless_rules
        from .data_loading import load_raw

        clean, _ = apply_stateless_rules(load_raw())
    frames = []
    for member in averaged.members:
        data = prepare_data(seed=member["seed"], clean=clean)
        raw = member["preprocessor"] is None  # rule-based baseline reads raw columns
        if not raw and list(data["feature_names"]) != list(member["feature_names"]):
            raise RuntimeError(f"seed {member['seed']}: the saved model was trained on different features")
        X, y = (data["X_test_raw"] if raw else data["X_test"]), data["y_test"]
        city = clean.loc[X.index, "City"].to_numpy()  # split indices are the cleaned frame's row labels
        frames.append(
            pd.DataFrame(
                {"seed": member["seed"], "City": city, "actual": y.to_numpy(), "predicted": member["model"].predict(X)},
                index=X.index,
            )
        )
    out = pd.concat(frames)
    out["residual"] = out["actual"] - out["predicted"]
    return out


def _error_row(group, value, d):
    abs_err = (d["actual"] - d["predicted"]).abs()
    return {
        "Group": group,
        "Value": value,
        "Test predictions": len(d),
        "MAE": round(float(abs_err.mean()), 2),
        "RMSE": round(_rmse(d["actual"], d["predicted"]), 2),
        "Median % error": round(float((abs_err / d["actual"]).median() * 100), 2),
        "Within ±20%": round(float((abs_err / d["actual"] <= 0.20).mean() * 100), 2),
    }


def error_breakdown(preds):
    """Test error overall, by city and by actual-rent band (pooled over the seeds' test splits)."""
    rows = [_error_row("Overall", "All test homes", preds)]
    rows += [_error_row("City", c, d) for c, d in preds.groupby("City", sort=True)]
    for lo, hi, label in RENT_BANDS:
        d = preds[(preds["actual"] >= lo) & (preds["actual"] < hi)]
        if len(d):
            rows.append(_error_row("Rent band", label, d))
    return pd.DataFrame(rows)


def residual_histogram(preds, width=2_500, limit=50_000):
    """Counts of residuals (actual − predicted) in `width`-rupee bins from −limit to +limit; residuals beyond the
    limit are counted in the two end bins (open-ended)."""
    edges = np.arange(-limit, limit + width, width)
    clipped = preds["residual"].clip(-limit, limit - 1e-9)
    counts, _ = np.histogram(clipped, bins=edges)
    return pd.DataFrame(
        {
            "bin_start": edges[:-1],
            "bin_end": edges[1:],
            "count": counts,
            "open_ended": [i in (0, len(counts) - 1) for i in range(len(counts))],
        }
    )


def save_final_diagnostics(averaged, clean=None):
    """Write reports/final_model_errors.csv and reports/final_model_residuals.csv (aggregates only — no listing
    rows) and visualizations/final_actual_vs_predicted.png. Returns (predictions, breakdown, histogram)."""
    from .plotting import plot_actual_vs_predicted

    preds = final_test_predictions(averaged, clean=clean)
    breakdown, hist = error_breakdown(preds), residual_histogram(preds)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    breakdown.to_csv(config.FINAL_ERRORS_CSV, index=False, encoding="utf-8-sig")
    hist.to_csv(config.FINAL_RESIDUALS_CSV, index=False, encoding="utf-8-sig")
    plot_actual_vs_predicted(preds, averaged.name, save_path=config.VIZ_DIR / "final_actual_vs_predicted.png")
    return preds, breakdown, hist
