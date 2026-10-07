"""Deep hyperparameter tuning (Optuna / TPE) for selected models.

    python scripts/deep_tune.py                                  # XGBoost + Random Forest
    python scripts/deep_tune.py --models xgboost_model --trials 150 --repeats 2

For each model:
  1. Bayesian search on the TRAIN split (objective = repeated 5-fold CV RMSE).
  2. Top-K trials + the library default + the earlier tuned config (if reports/tuning/tuned_best_params.json
     exists) are re-scored on the SAME fresh
     repeated 5x3 folds (different seed) -> pick the best by this confirmation score.
  3. Default, earlier-tuned and deep-tuned versions are evaluated on train / val / test (+ 5-fold CV)
     with bootstrap std, written to reports/deep_tuning/. Nothing else in reports/ is touched; adopting a
     result means editing FINAL_PARAMS in src/models.py by hand.
The test set is evaluated once per config and never used to choose.
"""
import sys, json, time, argparse, pathlib, warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
import pandas as pd

from src import config
from src.preprocessing import prepare_data, assert_run_consistency
from src.models import make_model, display_name
from src.tuning import optuna_search, confirm_cv, build_tuned_model, load_tuned_params
from src.evaluation import evaluate, save_results, bootstrap_std, plot_diagnostics

DEFAULT_TRIALS = {"xgboost_model": 150, "random_forest": 60}
DEFAULT_REPEATS = {"xgboost_model": 2, "random_forest": 1}
TOP_K = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_TRIALS))
    ap.add_argument("--trials", nargs="+", type=int, help="trials per model (same order as --models)")
    ap.add_argument("--repeats", nargs="+", type=int, help="CV repeats in the search objective per model")
    args = ap.parse_args()
    out = config.REPORTS / "deep_tuning"; out.mkdir(parents=True, exist_ok=True)
    mdir, pdir = out / "metrics", out / "predictions"
    pfile = out / "deep_tuned_params.json"
    deep_params = json.loads(pfile.read_text()) if pfile.exists() else {}

    data = prepare_data()
    ref = assert_run_consistency(data)
    print(f"Deep tuning | reference_id {ref} | train rows {len(data['X_train'])}\n", flush=True)
    v1 = load_tuned_params()
    summary_path = out / "deep_tuning_summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()

    for i, key in enumerate(args.models):
        n_trials = args.trials[i] if args.trials else DEFAULT_TRIALS.get(key, 60)
        reps = args.repeats[i] if args.repeats else DEFAULT_REPEATS.get(key, 1)
        t0 = time.perf_counter()
        print(f"== {display_name(key)}: {n_trials} TPE trials, objective = {reps}x5-fold CV RMSE", flush=True)
        study = optuna_search(key, data, n_trials=n_trials, cv_repeats=reps)
        search_s = time.perf_counter() - t0
        trials = study.trials_dataframe(attrs=("number", "value", "user_attrs", "params", "duration"))
        trials.rename(columns={"value": "search_cv_rmse"}).sort_values("search_cv_rmse") \
            .to_csv(out / f"trials_{key}.csv", index=False)

        # --- confirmation on fresh folds: top-K + default + earlier tuned ---
        top = sorted([t for t in study.trials if t.value is not None], key=lambda t: t.value)[:TOP_K]
        cands = {f"trial_{t.number}": build_tuned_model(key, data["target_transform"], t.user_attrs["est_params"]) for t in top}
        cands["default"] = make_model(key, data["target_transform"], use_final=False)
        if key in v1:
            cands["tuned_v1"] = build_tuned_model(key, data["target_transform"], v1[key])
        conf = confirm_cv(cands, data)
        crows = [{"candidate": lab, "confirm_rep_cv_rmse": m, "confirm_rep_cv_sd": s,
                  "search_cv_rmse": next((t.value for t in top if f"trial_{t.number}" == lab), None)}
                 for lab, (m, s) in conf.items()]
        cdf = pd.DataFrame(crows).sort_values("confirm_rep_cv_rmse"); cdf.to_csv(out / f"confirm_{key}.csv", index=False)
        best_lab = cdf[cdf["candidate"].str.startswith("trial_")].iloc[0]["candidate"]
        best = next(t for t in top if f"trial_{t.number}" == best_lab)
        deep_params[key] = best.user_attrs["est_params"]
        pfile.write_text(json.dumps(deep_params, indent=2, default=str))
        print(f"   search best CV Rs{study.best_value:,.0f} ({search_s:.0f}s) | confirmation:\n" +
              cdf.round(0).to_string(index=False), flush=True)

        # --- evaluate default / tuned_v1 / deep on train, val, test ---
        variants = {"default": make_model(key, data["target_transform"], use_final=False)}
        if key in v1:
            variants["tuned_v1"] = build_tuned_model(key, data["target_transform"], v1[key])
        variants["deep"] = build_tuned_model(key, data["target_transform"], deep_params[key])
        rows = []
        for var, model in variants.items():
            k = f"{key}_{var}"
            res, pred, _ = evaluate(model, data, f"{display_name(key)} ({var})", k, evaluate_test=True)
            save_results(res, pred, data, metrics_dir=mdir, predictions_dir=pdir)
            if var == "deep":
                plot_diagnostics(res, pred, data, model, f"{display_name(key)} (deep-tuned)",
                                 save_path=str(out / f"{key}_deep.png"))
            bs = {s: bootstrap_std({"train": data["y_train"], "test": data["y_test"]}[s], pred[s]) for s in ("train", "test")}
            m, sd = conf.get(var if var != "deep" else best_lab)
            rows.append({"key": key, "model": display_name(key), "variant": var,
                         "train_rmse": res["train"]["RMSE"], "train_rmse_std": bs["train"]["RMSE_std"],
                         "cv_rmse": res["cv"]["RMSE_mean"], "cv_rmse_std": res["cv"]["RMSE_std"],
                         "confirm_rep_cv_rmse": m, "confirm_rep_cv_sd": sd,
                         "val_rmse": res["val"]["RMSE"], "val_mae": res["val"]["MAE"], "val_r2": res["val"]["R2"],
                         "test_rmse": res["test"]["RMSE"], "test_rmse_std": bs["test"]["RMSE_std"],
                         "test_mae": res["test"]["MAE"], "test_mae_std": bs["test"]["MAE_std"],
                         "test_r2": res["test"]["R2"], "test_r2_std": bs["test"]["R2_std"],
                         "train_cv_gap": res["cv"]["RMSE_mean"] - res["train"]["RMSE"],
                         "params": json.dumps(deep_params[key] if var == "deep" else v1.get(key, {}) if var == "tuned_v1" else {}, default=str)})
        new = pd.DataFrame(rows)
        summary = pd.concat([summary[summary["key"] != key] if not summary.empty else summary, new], ignore_index=True)
        summary.to_csv(summary_path, index=False)
        print(new[["variant", "train_rmse", "cv_rmse", "confirm_rep_cv_rmse", "val_rmse", "test_rmse", "test_r2"]].round(3).to_string(index=False), "\n", flush=True)
    print("-> reports/deep_tuning/")


if __name__ == "__main__":
    main()
