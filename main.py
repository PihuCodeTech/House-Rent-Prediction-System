"""Run the experiment from the command line (an alternative to the notebooks).

    python main.py                         # all available models at default settings + leaderboard
    python main.py random_forest xgboost_model
    python main.py --test                  # also score the held-out test set
    python main.py --tune --test           # default AND tuned version of every tunable model
    python main.py --decide                # per model keep tuned params only if they beat the default
                                           #   on test RMSE (--decide-metric to change) -> reports/final_configs.json
    python main.py --final                 # evaluate every model with its final config -> reports/final/

Every model uses the IDENTICAL prepared data from House_Rent_Dataset.csv (src pipeline), so the
comparison is fair. Models whose optional package is missing (xgboost/lightgbm/catboost) are skipped.
Tuning is CV on the training split only; validation/test are never seen by a search.
"""
import argparse, json, time, warnings
warnings.filterwarnings("ignore")
import pandas as pd

from src import config
from src.preprocessing import prepare_data, assert_run_consistency
from src.models import REGISTRY, make_model, display_name, needs_package
from src.evaluation import evaluate, save_results, print_summary, plot_diagnostics, build_leaderboard


def available(keys):
    import importlib.util
    out = []
    for k in keys:
        pkg = needs_package(k)
        if pkg and importlib.util.find_spec(pkg) is None:
            print(f"  (skip {k}: '{pkg}' not installed)")
        else:
            out.append(k)
    return out


def run_one(model, data, name, key, args, n_rows, n_feat, plot_name):
    results, pred, _ = evaluate(model, data, name, key, evaluate_test=args.test)
    save_results(results, pred, data)
    if not args.no_plots:
        plot_diagnostics(results, pred, data, model, name, save_path=f"visualizations/{plot_name}.png")
    print_summary(results, n_rows, n_feat); print(flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", help="model keys to run (default: all)")
    ap.add_argument("--test", action="store_true", help="also evaluate the held-out test set")
    ap.add_argument("--tune", action="store_true", help="also tune each model by CV on the training split")
    ap.add_argument("--no-plots", action="store_true", help="skip saving diagnostic PNGs")
    ap.add_argument("--decide", action="store_true",
                    help="choose default vs tuned per model from reports/model_comparison.csv")
    ap.add_argument("--decide-metric", default="test_rmse",
                    help="leaderboard column used by --decide (lower is better); default test_rmse")
    ap.add_argument("--final", action="store_true",
                    help="evaluate every model with its final configuration into reports/final/")
    args = ap.parse_args()

    if args.decide:
        from src.tuning import decide_final_configs
        src_csv = config.REPORTS / "model_comparison.csv"
        payload = decide_final_configs(pd.read_csv(src_csv), metric=args.decide_metric, source=str(src_csv.name))
        print(f"Rule: {payload['rule']}\nCaveat: {payload['caveat']}\n")
        for k, v in payload["models"].items():
            print(f"  {k:20s} -> {v['config']:7s}  {v['reason']}")
        print("\n-> reports/final_configs.json")
        if not args.final:
            return
    if args.final:
        return run_final(args)

    keys = available(args.models or list(REGISTRY))
    data = prepare_data()
    ref = assert_run_consistency(data)
    n_rows = sum(data["meta"]["rows"][s] for s in ("train", "val", "test"))
    n_feat = len(data["feature_names"])
    print(f"Prepared: {data['X_train'].shape[0]} train / {data['X_val'].shape[0]} val / "
          f"{data['X_test'].shape[0]} test | {n_feat} features | reference_id {ref}\n", flush=True)

    if args.tune:
        from src.tuning import tune, best_params, best_cv_rmse, build_tuned_model, save_tuned_params, UNTUNABLE
        (config.REPORTS / "tuning").mkdir(parents=True, exist_ok=True)
        summary = []

    for key in keys:
        # 1) default configuration
        t_def = run_one(make_model(key, data["target_transform"]), data, display_name(key), key,
                        args, n_rows, n_feat, key)
        if not args.tune or key in UNTUNABLE:
            continue
        # 2) CV search on the training split, then refit + evaluate the best configuration
        t0 = time.perf_counter()
        search = tune(key, data)
        params = best_params(search)
        save_tuned_params({key: params})
        cvres = pd.DataFrame(search.cv_results_)
        keep = [c for c in cvres.columns if c.startswith("param_")] + ["mean_test_score", "std_test_score", "rank_test_score"]
        cvres[keep].assign(cv_rmse=-cvres["mean_test_score"]).sort_values("rank_test_score").head(15) \
            .to_csv(config.REPORTS / "tuning" / f"search_{key}.csv", index=False)
        search_time = time.perf_counter() - t0
        print(f">>> tuned {key}: search CV RMSE Rs{best_cv_rmse(search):,.0f} "
              f"({len(cvres)} candidates, {search_time:.0f}s) params={params}", flush=True)
        t_tun = run_one(build_tuned_model(key, data["target_transform"], params), data,
                        display_name(key) + " (tuned)", key + "_tuned", args, n_rows, n_feat, key + "_tuned")
        summary.append({"key": key, "model": display_name(key),
                        "cv_rmse_default": t_def["cv"]["RMSE_mean"], "cv_rmse_tuned": t_tun["cv"]["RMSE_mean"],
                        "val_rmse_default": t_def["val"]["RMSE"], "val_rmse_tuned": t_tun["val"]["RMSE"],
                        **({"test_rmse_default": t_def["test"]["RMSE"], "test_rmse_tuned": t_tun["test"]["RMSE"],
                            "test_r2_default": t_def["test"]["R2"], "test_r2_tuned": t_tun["test"]["R2"]}
                           if args.test else {}),
                        "candidates": len(cvres), "search_time_s": round(search_time, 1),
                        "params": json.dumps(params, default=str)})
        sp = config.REPORTS / "tuning" / "tuning_summary.csv"
        old = pd.read_csv(sp) if sp.exists() else pd.DataFrame()
        new = pd.DataFrame(summary)
        if not old.empty:
            old = old[~old["key"].isin(new["key"])]
        pd.concat([old, new], ignore_index=True).to_csv(sp, index=False)

    board = build_leaderboard()
    cols = [c for c in ["model", "cv_rmse", "val_rmse", "test_rmse", "test_r2", "train_rmse"] if c in board.columns]
    print("LEADERBOARD (by CV RMSE)\n" + board[cols].to_string(index=False))
    print("\n-> reports/model_comparison.csv")


def run_final(args):
    """Evaluate every model with its final configuration (val + test + 5-fold CV) into reports/final/."""
    import json
    from src.tuning import build_final_model, load_final_configs
    final_dir = config.REPORTS / "final"
    mdir, pdir, vdir = final_dir / "metrics", final_dir / "predictions", config.VIZ_DIR / "final"
    cfgs = load_final_configs()["models"]
    keys = available([k for k in (args.models or list(REGISTRY)) if k in cfgs])
    data = prepare_data()
    ref = assert_run_consistency(data)
    n_rows = sum(data["meta"]["rows"][s] for s in ("train", "val", "test"))
    n_feat = len(data["feature_names"])
    print(f"FINAL EVALUATION | reference_id {ref} | {n_feat} features\n", flush=True)
    for key in keys:
        model, cfg = build_final_model(key, data["target_transform"])
        name = f"{display_name(key)} ({cfg})"
        results, pred, _ = evaluate(model, data, name, key, evaluate_test=True)
        save_results(results, pred, data, metrics_dir=mdir, predictions_dir=pdir)
        if not args.no_plots:
            vdir.mkdir(parents=True, exist_ok=True)
            plot_diagnostics(results, pred, data, model, name, save_path=str(vdir / f"{key}.png"))
        print_summary(results, n_rows, n_feat); print(flush=True)
    board = build_leaderboard(metrics_dir=mdir, out_path=final_dir / "final_leaderboard.csv")
    board.insert(2, "config", board["key"].map(lambda k: cfgs[k]["config"]))
    board.to_csv(final_dir / "final_leaderboard.csv", index=False)
    cols = [c for c in ["model", "cv_rmse", "val_rmse", "test_rmse", "test_mae", "test_r2", "train_rmse"] if c in board.columns]
    print("FINAL LEADERBOARD (by CV RMSE)\n" + board[cols].to_string(index=False))
    print("\n-> reports/final/final_leaderboard.csv")


if __name__ == "__main__":
    main()
