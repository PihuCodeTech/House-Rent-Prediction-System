"""Robustness check.

    python scripts/robustness_check.py          # after `main.py --tune --test`: top tuned models
    python scripts/robustness_check.py --final  # after `main.py --final`: every final configuration

1) Repeated 5x3 CV RMSE on the training split (stabler ranking than a single 5-fold run).
2) Paired bootstrap of RMSE on identical resamples -> 95% CIs + pairwise win rates (test; plus
   validation in --final mode, the one split no tuning/revert decision has used).
"""
import sys, pathlib, warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
import pandas as pd

from src import config
from src.preprocessing import prepare_data
from src.tuning import build_tuned_model, repeated_cv_rmse
from src.models import display_name
from src.evaluation import bootstrap_rmse

FINAL = "--final" in sys.argv
data = prepare_data()


def load_preds(pred_dir, keys, split):
    preds, y = {}, None
    for k in keys:
        p = pred_dir / f"predictions_{k}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p); d = d[d["split"] == split].sort_values("row_id")
        y = d["actual_rent"].to_numpy() if y is None else y
        preds[k] = d["predicted_rent"].to_numpy()
    return y, preds


if FINAL:
    from src.tuning import build_final_model
    out = config.REPORTS / "final"
    board = pd.read_csv(out / "final_leaderboard.csv")
    board = board[~board["key"].str.startswith("baseline")]
    rows = []
    for _, r in board.iterrows():
        model, cfg = build_final_model(r["key"], data["target_transform"])
        mean, sd = repeated_cv_rmse(model, data)
        rows.append({"key": r["key"], "model": r["model"], "config": cfg, "cv5_rmse": r["cv_rmse"],
                     "rep_cv_rmse_5x3": mean, "rep_cv_sd": sd,
                     "val_rmse": r["val_rmse"], "val_mae": r["val_mae"], "val_r2": r["val_r2"], "val_mape": r["val_mape"],
                     "test_rmse": r["test_rmse"], "test_mae": r["test_mae"], "test_r2": r["test_r2"], "test_mape": r["test_mape"]})
        print(f"  {r['key']:20s} {cfg:7s} rep-CV {mean:,.0f} (sd {sd:,.0f})", flush=True)
    rep = pd.DataFrame(rows).sort_values("rep_cv_rmse_5x3")
    rep.to_csv(out / "final_repeated_cv.csv", index=False)
    for split in ("val", "test"):
        y, preds = load_preds(out / "predictions", list(rep["key"]), split)
        ci, win = bootstrap_rmse(y, preds)
        ci.to_csv(out / f"bootstrap_{split}_ci.csv", index=False)
        win.to_csv(out / f"bootstrap_{split}_winrate.csv")
        print(f"\n{split} bootstrap 95% CI:\n", ci.round(0).to_string(index=False))
    print("\nRepeated CV (final configs):\n", rep.round(4).to_string(index=False))
    sys.exit(0)

# ---- default mode: strongest tuned models ----
TOP_N = 6
out = config.REPORTS / "tuning"; out.mkdir(parents=True, exist_ok=True)
board = pd.read_csv(config.REPORTS / "model_comparison.csv")
tuned = board[board["key"].str.endswith("_tuned")].sort_values("cv_rmse").head(TOP_N)
print("Top tuned by 5-fold CV:", list(tuned["key"]))
rows = []
for k in tuned["key"]:
    base = k[:-len("_tuned")]
    mean, sd = repeated_cv_rmse(build_tuned_model(base, data["target_transform"]), data)
    r = tuned.set_index("key").loc[k]
    rows.append({"key": k, "model": display_name(base) + " (tuned)", "cv5_rmse": r["cv_rmse"],
                 "rep_cv_rmse_5x3": mean, "rep_cv_sd": sd, "val_rmse": r["val_rmse"],
                 "test_rmse": r.get("test_rmse"), "test_mae": r.get("test_mae"),
                 "test_r2": r.get("test_r2"), "test_mape": r.get("test_mape")})
    print(f"  {k:26s} rep-CV {mean:,.0f} (sd {sd:,.0f})", flush=True)
rep = pd.DataFrame(rows).sort_values("rep_cv_rmse_5x3")
rep.to_csv(out / "repeated_cv_top_tuned.csv", index=False)
keys = list(rep["key"]) + [k[:-len("_tuned")] for k in rep["key"]]
y, preds = load_preds(config.PREDICTIONS_DIR, keys, "test")
ci, win = bootstrap_rmse(y, preds)
ci.to_csv(out / "test_bootstrap_ci.csv", index=False)
win.to_csv(out / "test_bootstrap_winrate.csv")
print("\nRepeated CV:\n", rep.round(4).to_string(index=False))
print("\nTest bootstrap 95% CI:\n", ci.round(0).to_string(index=False))
