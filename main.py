"""Run the experiment from the command line (an alternative to the notebooks).

    python main.py --test                  # every model with its final configuration + leaderboard
    python main.py random_forest xgboost_model --test
    python main.py --test --no-plots

Every model uses the IDENTICAL prepared data from House_Rent_Dataset.csv (src pipeline) and its single
final configuration (src/models.py -> FINAL_PARAMS), so the comparison is fair. Models whose optional
package is missing (xgboost/lightgbm/catboost) are skipped. Outputs: reports/metrics/, reports/predictions/,
reports/model_comparison.csv, visualizations/<model>.png.
"""
import argparse, warnings
warnings.filterwarnings("ignore")

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", help="model keys to run (default: all)")
    ap.add_argument("--test", action="store_true", help="also evaluate the held-out test set")
    ap.add_argument("--no-plots", action="store_true", help="skip saving diagnostic PNGs")
    args = ap.parse_args()

    keys = available(args.models or list(REGISTRY))
    data = prepare_data()
    ref = assert_run_consistency(data)
    n_rows = sum(data["meta"]["rows"][s] for s in ("train", "val", "test"))
    n_feat = len(data["feature_names"])
    print(f"Prepared: {data['X_train'].shape[0]} train / {data['X_val'].shape[0]} val / "
          f"{data['X_test'].shape[0]} test | {n_feat} features | reference_id {ref}\n", flush=True)

    for key in keys:
        model = make_model(key, data["target_transform"])
        results, pred, _ = evaluate(model, data, display_name(key), key, evaluate_test=args.test)
        save_results(results, pred, data)
        if not args.no_plots:
            plot_diagnostics(results, pred, data, model, display_name(key), save_path=f"visualizations/{key}.png")
        print_summary(results, n_rows, n_feat); print(flush=True)

    board = build_leaderboard()
    cols = [c for c in ["model", "cv_rmse", "val_rmse", "test_rmse", "test_r2", "train_rmse"] if c in board.columns]
    print("LEADERBOARD (by CV RMSE)\n" + board[cols].to_string(index=False))
    print("\n-> reports/model_comparison.csv")


if __name__ == "__main__":
    main()
