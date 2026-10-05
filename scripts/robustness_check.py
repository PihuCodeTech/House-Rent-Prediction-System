"""Post-tuning robustness check (run after `python main.py --tune --test`).

1) Repeated 5x3 CV RMSE on the training split for the strongest tuned models (stabler ranking
   than a single 5-fold run, and less prone to winning by search luck).
2) Paired bootstrap of test RMSE on identical resamples -> 95% CIs + pairwise win rates, to see
   whether test-set gaps between models are real or noise.
"""
import sys, json, pathlib, warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
import pandas as pd

from src import config
from src.preprocessing import prepare_data
from src.tuning import build_tuned_model, repeated_cv_rmse, load_tuned_params
from src.models import display_name, make_model
from src.evaluation import bootstrap_rmse

TOP_N = 6
out = config.REPORTS / "tuning"; out.mkdir(parents=True, exist_ok=True)
board = pd.read_csv(config.REPORTS / "model_comparison.csv")
tuned = board[board["key"].str.endswith("_tuned")].sort_values("cv_rmse").head(TOP_N)
print("Top tuned by 5-fold CV:", list(tuned["key"]))

data = prepare_data()
rows = []
for k in tuned["key"]:
    base = k[:-len("_tuned")]
    m = build_tuned_model(base, data["target_transform"])
    mean, sd = repeated_cv_rmse(m, data)
    r = tuned.set_index("key").loc[k]
    rows.append({"key": k, "model": display_name(base) + " (tuned)", "cv5_rmse": r["cv_rmse"],
                 "rep_cv_rmse_5x3": mean, "rep_cv_sd": sd, "val_rmse": r["val_rmse"],
                 "test_rmse": r.get("test_rmse"), "test_mae": r.get("test_mae"),
                 "test_r2": r.get("test_r2"), "test_mape": r.get("test_mape")})
    print(f"  {k:26s} rep-CV {mean:,.0f} (sd {sd:,.0f})", flush=True)
rep = pd.DataFrame(rows).sort_values("rep_cv_rmse_5x3")
rep.to_csv(out / "repeated_cv_top_tuned.csv", index=False)

# paired bootstrap on test predictions (top tuned + their defaults for reference)
keys = list(rep["key"]) + [k[:-len("_tuned")] for k in rep["key"]]
preds, y = {}, None
for k in keys:
    p = config.PREDICTIONS_DIR / f"predictions_{k}.csv"
    if not p.exists():
        continue
    d = pd.read_csv(p); d = d[d["split"] == "test"].sort_values("row_id")
    y = d["actual_rent"].to_numpy() if y is None else y
    preds[k] = d["predicted_rent"].to_numpy()
ci, win = bootstrap_rmse(y, preds)
ci.to_csv(out / "test_bootstrap_ci.csv", index=False)
win.to_csv(out / "test_bootstrap_winrate.csv")
print("\nRepeated CV:\n", rep.round(4).to_string(index=False))
print("\nTest bootstrap 95% CI:\n", ci.round(0).to_string(index=False))
