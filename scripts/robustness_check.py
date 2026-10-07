"""Robustness check for the final models (run after `python main.py --test`).

    python scripts/robustness_check.py

1) Repeated 5x3 CV RMSE on the training split for every non-baseline model.
2) Paired bootstrap (2,000 resamples) of validation and test RMSE -> 95% CIs + pairwise win rates.
Writes reports/robustness/ and then rebuilds reports/final_evaluation_report.md (scripts/final_report.py).
"""
import sys, pathlib, warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
import pandas as pd

from src import config
from src.preprocessing import prepare_data
from src.models import make_model, display_name
from src.tuning import repeated_cv_rmse
from src.evaluation import bootstrap_rmse

OUT = config.REPORTS / "robustness"


def load_preds(keys, split):
    preds, y = {}, None
    for k in keys:
        p = config.PREDICTIONS_DIR / f"predictions_{k}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p); d = d[d["split"] == split].sort_values("row_id")
        if d.empty:
            continue
        y = d["actual_rent"].to_numpy() if y is None else y
        preds[k] = d["predicted_rent"].to_numpy()
    return y, preds


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    board = pd.read_csv(config.REPORTS / "model_comparison.csv")
    keys = [k for k in board["key"] if not k.startswith("baseline")]
    data = prepare_data()
    rows = []
    for k in keys:
        mean, sd = repeated_cv_rmse(make_model(k, data["target_transform"]), data)
        rows.append({"key": k, "model": display_name(k), "rep_cv_rmse_5x3": mean, "rep_cv_sd": sd})
        print(f"  {display_name(k):20s} repeated CV {mean:,.0f} (sd {sd:,.0f})", flush=True)
    rep = pd.DataFrame(rows).sort_values("rep_cv_rmse_5x3")
    rep.to_csv(OUT / "repeated_cv.csv", index=False)
    for split in ("val", "test"):
        y, preds = load_preds(list(rep["key"]), split)
        if not preds:
            print(f"(no {split} predictions — run `python main.py --test` first)"); continue
        ci, win = bootstrap_rmse(y, preds)
        ci.insert(1, "display", ci["model"].map(display_name))
        ci.to_csv(OUT / f"bootstrap_{split}_ci.csv", index=False)
        win.to_csv(OUT / f"bootstrap_{split}_winrate.csv")
        print(f"\n{split} RMSE 95% CI:\n" + ci.round(0).to_string(index=False))
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import final_report
    final_report.main()


if __name__ == "__main__":
    main()
