"""Evaluation: rupee-scale metrics, 5-fold CV on train, and saving results to CSV for comparison."""
import json, time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             mean_absolute_percentage_error, make_scorer)

from . import config


def metrics(y_true, y_pred, n_features=None):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    out = {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse, "RMSE": float(np.sqrt(mse)),
           "R2": r2, "MAPE": float(mean_absolute_percentage_error(y_true, y_pred))}
    n = len(y_true)
    if n_features and n - n_features - 1 > 0:                 # Adjusted R^2 (meaningful when n > p + 1)
        out["Adj_R2"] = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return out


def error_by_rent_range(y_true, y_pred, labels=("low", "medium", "high")):
    """RMSE / MAE / MAPE within rent terciles (low / medium / high)."""
    y_true = np.asarray(y_true, dtype="float64"); y_pred = np.asarray(y_pred, dtype="float64")
    edges = np.quantile(y_true, [0, 1/3, 2/3, 1.0]); edges[-1] += 1
    band = np.digitize(y_true, edges[1:-1])
    rows = []
    for i, lab in enumerate(labels):
        m = band == i
        if m.sum():
            e = y_true[m] - y_pred[m]
            rows.append({"band": lab, "n": int(m.sum()),
                         "rent_min": float(y_true[m].min()), "rent_max": float(y_true[m].max()),
                         "RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e))),
                         "MAPE": float(np.mean(np.abs(e / y_true[m])))})
    return pd.DataFrame(rows)


def cross_val_rmse(model, X_train, y_train, folds=config.CV_FOLDS):
    """5-fold CV on TRAIN only; the whole pipeline (scaler + target transform) is cloned and re-fit per fold."""
    cv = KFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_STATE)
    scoring = {"rmse": make_scorer(lambda a, p: np.sqrt(mean_squared_error(a, p)), greater_is_better=False),
               "mae": make_scorer(mean_absolute_error, greater_is_better=False), "r2": "r2"}
    t0 = time.perf_counter()
    cvr = cross_validate(clone(model), X_train, y_train, cv=cv, scoring=scoring)
    folds_rmse = -cvr["test_rmse"]
    return {"RMSE_mean": float(folds_rmse.mean()), "RMSE_std": float(folds_rmse.std()),
            "MAE_mean": float(-cvr["test_mae"].mean()), "R2_mean": float(cvr["test_r2"].mean()),
            "fold_RMSE": folds_rmse.round(1).tolist(), "time_s": time.perf_counter() - t0}


def evaluate(model, data, model_name, model_key, evaluate_test=False, cv_folds=config.CV_FOLDS):
    """Fit, score train/val (+ test if evaluate_test), run CV, and return (results, predictions, train_time)."""
    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    p = len(data["feature_names"])
    pred = {"train": model.predict(X_train), "val": model.predict(X_val)}
    results = {"model": model_name, "key": model_key,
               "reference_id": data["fingerprint"]["reference"]["reference_id"],
               "evaluate_test": evaluate_test, "train_time_s": train_time,
               "train": metrics(y_train, pred["train"], p), "val": metrics(y_val, pred["val"], p)}
    results["cv"] = cross_val_rmse(model, X_train, y_train, folds=cv_folds)
    if evaluate_test:
        pred["test"] = model.predict(X_test)
        results["test"] = metrics(y_test, pred["test"], p)
        results["error_by_rent_range"] = error_by_rent_range(y_test, pred["test"]).to_dict(orient="records")
    return results, pred, train_time


def results_table(results):
    splits = [s for s in ("train", "val", "test") if s in results]
    cols = [c for c in ["MAE", "MSE", "RMSE", "R2", "Adj_R2", "MAPE"] if c in results[splits[0]]]
    table = pd.DataFrame({s: results[s] for s in splits}).T[cols]
    table["CV RMSE"] = results["cv"]["RMSE_mean"]
    table["Train time (s)"] = results["train_time_s"]
    return table


def print_summary(results, n_rows, n_features, cv_folds=config.CV_FOLDS):
    r = results
    print("=" * 40); print(f"MODEL: {r['model']}"); print("=" * 40)
    print(f"Dataset:  Rows: {n_rows}  Features: {n_features}   Fingerprint: OK ({r['reference_id']})")
    line = f"Train RMSE: {r['train']['RMSE']:,.2f}   Validation RMSE: {r['val']['RMSE']:,.2f}"
    if "test" in r:
        line += f"   Test RMSE: {r['test']['RMSE']:,.2f}"
    print(line)
    for s in [x for x in ("val", "test") if x in r]:
        lab = "Validation" if s == "val" else "Test"
        print(f"{lab} MAE: {r[s]['MAE']:,.2f}   {lab} MSE: {r[s]['MSE']:,.2f}   {lab} R2: {r[s]['R2']:.4f}")
    print(f"CV RMSE: {r['cv']['RMSE_mean']:,.2f} (+/- {r['cv']['RMSE_std']:,.2f}, {cv_folds}-fold on train)")
    print(f"Training Time: {r['train_time_s']:.2f} s   (CV time: {r['cv']['time_s']:.2f} s)")


def save_results(results, pred, data, metrics_dir=None, predictions_dir=None):
    """Write per-model metrics (JSON + a flat CSV row) and predictions CSV (default: reports/metrics, reports/predictions)."""
    metrics_dir = metrics_dir or config.METRICS_DIR
    predictions_dir = predictions_dir or config.PREDICTIONS_DIR
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    key = results["key"]

    # full metrics JSON
    (metrics_dir / f"metrics_{key}.json").write_text(json.dumps(results, indent=2))

    # flat one-row CSV for easy leaderboard concatenation
    row = {"key": key, "model": results["model"], "reference_id": results["reference_id"],
           "train_time_s": round(results["train_time_s"], 3),
           "cv_rmse": results["cv"]["RMSE_mean"], "cv_rmse_std": results["cv"]["RMSE_std"]}
    for split in ("train", "val", "test"):
        if split in results:
            for m in ("MAE", "MSE", "RMSE", "R2", "Adj_R2", "MAPE"):
                if m in results[split]:
                    row[f"{split}_{m.lower()}"] = results[split][m]
    pd.DataFrame([row]).to_csv(metrics_dir / f"metrics_{key}.csv", index=False)
    if "error_by_rent_range" in results:
        pd.DataFrame(results["error_by_rent_range"]).to_csv(
            metrics_dir / f"error_by_band_{key}.csv", index=False)

    # predictions (actual / predicted / residual) in rupees
    y = {"train": data["y_train"], "val": data["y_val"], "test": data["y_test"]}
    X = {"train": data["X_train"], "val": data["X_val"], "test": data["X_test"]}
    frames = []
    for s in ("train", "val", "test"):
        if s in pred:
            frames.append(pd.DataFrame({"row_id": X[s].index, "split": s, "actual_rent": y[s].to_numpy(),
                                        "predicted_rent": pred[s], "residual": y[s].to_numpy() - pred[s]}))
    pred_df = pd.concat(frames, ignore_index=True)
    pred_df.to_csv(predictions_dir / f"predictions_{key}.csv", index=False)
    return pred_df


def build_leaderboard(metrics_dir=None, out_path=None):
    """Concatenate metrics_*.csv (default reports/metrics) into one leaderboard sorted by CV RMSE."""
    files = sorted((metrics_dir or config.METRICS_DIR).glob("metrics_*.csv"))
    if not files:
        raise FileNotFoundError("no metrics_*.csv yet — run the model notebooks first.")
    board = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    sort_col = "cv_rmse" if "cv_rmse" in board else "val_rmse"
    board = board.sort_values(sort_col).reset_index(drop=True)
    board.to_csv(out_path or (config.REPORTS / "model_comparison.csv"), index=False)
    return board


def plot_diagnostics(results, pred, data, model, model_name, save_path=None):
    """3-panel diagnostic: Actual vs Predicted, Residuals vs Predicted, feature importance/coefficients."""
    import matplotlib
    matplotlib.use("Agg") if save_path else None
    import matplotlib.pyplot as plt
    feature_names = data["feature_names"]
    inner = (model.regressor_ if hasattr(model, "regressor_") else model).named_steps["model"]
    if hasattr(inner, "coef_"):
        imp = pd.Series(np.ravel(inner.coef_), index=feature_names); imp_title = "Std. coefficients (log-rent)"
    elif hasattr(inner, "feature_importances_"):
        imp = pd.Series(inner.feature_importances_, index=feature_names); imp_title = "Feature importances"
    else:
        imp = None
    s = "test" if "test" in pred else "val"
    yt = (data["y_test"] if s == "test" else data["y_val"]).to_numpy(); yp = pred[s]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].scatter(yt, yp, s=10, alpha=.5); lim = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    ax[0].plot(lim, lim, "k--", lw=1); ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("Actual rent (Rs)"); ax[0].set_ylabel("Predicted rent (Rs)"); ax[0].set_title(f"Actual vs Predicted ({s})")
    ax[1].scatter(yp, yt - yp, s=10, alpha=.5); ax[1].axhline(0, color="k", lw=1); ax[1].set_xscale("log")
    ax[1].set_xlabel("Predicted rent (Rs)"); ax[1].set_ylabel("Residual (Rs)"); ax[1].set_title(f"Residuals vs Predicted ({s})")
    if imp is not None and np.any(imp != 0):
        top = imp.reindex(imp.abs().sort_values(ascending=False).index)[:15][::-1]
        ax[2].barh(top.index, top.values); ax[2].set_title(imp_title + " - top 15")
    else:
        ax[2].text(.5, .5, "n/a", ha="center"); ax[2].set_axis_off()
    fig.suptitle(model_name); plt.tight_layout()
    if save_path:
        config.VIZ_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=110, bbox_inches="tight"); plt.close(fig)
    return fig


def bootstrap_rmse(y_true, preds, n_boot=2000, seed=42):
    """Paired bootstrap of test RMSE for several models on the SAME resamples.

    preds: {name: y_pred}. Returns (ci_table, win_matrix) where win_matrix[a][b] is the share of
    resamples in which model a has a lower RMSE than model b.
    """
    y = np.asarray(y_true, dtype="float64"); rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    rm = {k: np.sqrt(((y[idx] - np.asarray(p)[idx]) ** 2).mean(axis=1)) for k, p in preds.items()}
    ci = pd.DataFrame([{"model": k, "test_rmse": float(np.sqrt(((y - np.asarray(preds[k])) ** 2).mean())),
                        "ci_low_2.5%": float(np.percentile(v, 2.5)), "ci_high_97.5%": float(np.percentile(v, 97.5))}
                       for k, v in rm.items()]).sort_values("test_rmse")
    names = list(preds)
    win = pd.DataFrame([[float((rm[a] < rm[b]).mean()) if a != b else np.nan for b in names] for a in names],
                       index=names, columns=names)
    return ci.reset_index(drop=True), win


def bootstrap_std(y_true, y_pred, n_boot=2000, seed=42):
    """Bootstrap standard deviation of RMSE / MAE / R2 over row resamples of one split."""
    y = np.asarray(y_true, dtype="float64"); p = np.asarray(y_pred, dtype="float64")
    idx = np.random.default_rng(seed).integers(0, len(y), size=(n_boot, len(y)))
    yt, yp = y[idx], p[idx]
    rmse = np.sqrt(((yt - yp) ** 2).mean(1)); mae = np.abs(yt - yp).mean(1)
    r2 = 1 - ((yt - yp) ** 2).sum(1) / ((yt - yt.mean(1, keepdims=True)) ** 2).sum(1)
    return {"RMSE_std": float(rmse.std()), "MAE_std": float(mae.std()), "R2_std": float(r2.std())}
