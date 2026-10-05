# Housing Rent Prediction — Supervised Learning Project

A complete, reproducible, leakage-free machine-learning pipeline that predicts monthly house/apartment
**rent** (in ₹) for the India House Rent dataset. Every model is trained and evaluated on the **identical**
prepared data, so model comparison is scientifically fair.

## 1. Project overview
End-to-end regression project: inspection → cleaning → EDA → feature engineering → preprocessing →
baselines → regularization → tuning → evaluation → final model → interpretability → inference. Data
preparation lives once in `src/` and is imported by every notebook (identical by construction).

## 2. Problem statement
Given property attributes (size, bedrooms, bathrooms, floor, locality, city, furnishing, tenant type,
contact), predict the monthly **Rent**. This is a **regression** problem with a strongly right-skewed target.

## 3. Dataset description
- `data/raw/House_Rent_Dataset.csv` — **4,746 rows × 12 columns** (see `Dataset Glossary.txt`).
- Target `Rent`: median ₹15,000, mean ₹35,000, max ₹3,500,000, skew ≈ 4.4 (heavy right tail).
- Features: `BHK`, `Size`, `Bathroom` (numeric); `Floor` (text), `Area Type`, `Area Locality`
  (~2,235 values), `City` (6), `Furnishing Status`, `Tenant Preferred`, `Point of Contact`, `Posted On`.

## 4. Data cleaning  (`src/data_cleaning.py`, `reports/phase2_data_cleaning.md`)
Stateless rules applied before any split: strip/normalise text, validity filter (`Rent>0, Size>0, BHK≥1,
Bathroom≥1`), drop re-listings (9 rows), parse `Floor` → current/total floors, parse `Posted On`. No missing
values (median/most-frequent imputers kept as a train-fit safety net). No hand-picked row deletions.

## 5. Exploratory data analysis  (`reports/phase3_eda.md`, `visualizations/`)
Target is right-skewed (skew 4.4) → `log1p` makes it near-symmetric (skew 0.17). `Size`, `Bathroom`, `BHK`
correlate most with rent; Mumbai/Delhi have the highest medians. IQR flags ~520 "outliers" but most are
legitimate luxury units → kept; only extreme train-split rent/sqft points are trimmed.

## 6. Feature engineering  (`src/feature_engineering.py`)
Log-size/BHK/bath, floor features (current, total, ratio, is_basement), city × log-size interactions,
leakage-safe **target encoding** for `Area Locality` (cross-fitted inside CV), one-hot for low-cardinality
categoricals. **34 features** total. No "property age" (post date ≠ build date); no target-derived features.

## 7. Preprocessing  (`src/preprocessing.py`)
`ColumnTransformer` + `Pipeline`, **fit on the training set only**. Scaling (`StandardScaler`) applied only
to linear/distance models; tree/boosting models use the unscaled matrix. Target modelled on `log1p`,
predictions back-transformed with `expm1` via `TransformedTargetRegressor` → metrics in rupees.

## 8. Models tested  (`src/models.py`, `notebooks/`)
Mean & median baselines; Linear, Ridge, Lasso, Elastic Net; Decision Tree, Random Forest, Gradient Boosting;
XGBoost, LightGBM, CatBoost. Each has a default and a tuned version.

## 9. Regularization  (`notebooks/regularization.ipynb`, `reports/regularization_summary.csv`)
Ridge/Lasso/Elastic Net `alpha` (+ `l1_ratio`) tuned by 5-fold CV on train. Tuned CV RMSE ≈ ₹29.6–29.8k vs
unregularized Linear ₹30.1k. Lasso/Elastic Net zero out ~16/34 coefficients (feature selection) with no CV-RMSE
loss → the linear feature set is somewhat over-parameterised, but linear models remain well behind the trees.

## 10. Hyperparameter tuning  (`src/tuning.py`, `reports/tuning_report.md`, `reports/tuning/`)
Every tunable model is tuned by 5-fold CV on the training split only (scaling + log1p re-fit inside each fold;
validation/test never seen): exhaustive grids for Ridge/Lasso/Elastic Net, `RandomizedSearchCV` (40–100
candidates) for Decision Tree, Random Forest, Gradient Boosting (incl. Huber loss), XGBoost, LightGBM, CatBoost.
CV RMSE improved for every model, e.g. XGBoost ₹28.4k → **₹26.5k**, Gradient Boosting ₹28.0k → ₹26.3k,
LightGBM ₹28.2k → ₹26.5k, Decision Tree ₹36.9k → ₹29.8k, Lasso ₹59.3k → ₹29.6k. The top six are re-ranked on
repeated 5x3 CV and their test RMSEs compared with a paired bootstrap (`scripts/robustness_check.py`).
Mean/median baselines and plain Linear Regression have no hyperparameters.

## 11. Evaluation metrics  (`src/evaluation.py`)
MAE, MSE, RMSE, R², Adjusted R², MAPE — all in rupees — plus 5-fold CV RMSE, error-by-rent-range
(low/medium/high terciles), residual and actual-vs-predicted plots.

## 12. Final model  (`notebooks/final_selection.ipynb`, `models/`)
**Tuned XGBoost Regressor** — lowest repeated 5x3 CV RMSE (₹27,025) among tuned models; the choice uses
training-split evidence only and the test set is evaluated once afterwards. Tuned Gradient Boosting (Huber loss)
is statistically tied (₹27,067). Saved as `models/final_model.pkl` + `models/preprocessor.pkl`
(+ `feature_names.json`, `final_model_meta.json` with the hyperparameters).

## 13. Results  (full tables: `reports/tuning_report.md`)

| Model (tuned unless noted) | CV RMSE | Repeated CV | Test RMSE | Test MAE | Test R² |
|---|--:|--:|--:|--:|--:|
| **XGBoost** | 26,517 | **27,025** | 23,517 | 9,629 | 0.809 |
| Gradient Boosting (Huber) | 26,254 | 27,067 | 24,387 | 9,720 | 0.794 |
| LightGBM | 26,541 | 27,432 | 23,726 | 9,775 | 0.805 |
| CatBoost | 26,985 | 27,870 | 23,300 | 9,526 | 0.812 |
| Random Forest | 28,128 | 28,624 | 21,798 | 9,146 | 0.836 |
| Lasso / Elastic Net | 29,644 | 29,717 | 28,715 | 10,561 | 0.715 |
| Ridge | 29,787 | — | 28,547 | 10,584 | 0.718 |
| Decision Tree | 29,802 | — | 23,957 | 10,652 | 0.801 |
| Linear (no hyperparameters) | 30,056 | — | 28,487 | 10,671 | 0.719 |
| Mean baseline | 57,520 | — | 53,781 | 28,779 | −0.000 |

Test-set differences among the tree/boosting models are within noise (95% bootstrap intervals ≈ ±₹6k; a few
luxury listings dominate single-split RMSE), so the model is chosen on repeated CV. Results are in line with
earlier tuning in this project (Ridge/Lasso/Elastic Net identical, RF/GBR within ₹200 repeated CV) and with
published models on this dataset (e.g. R² 0.71 / MAE ₹10.1k for a log1p Gradient Boosting with an untouched test set).
Top drivers (permutation importance): city×size (Mumbai), log-size, Point-of-Contact (agent), Bathroom, total floors.

## 14. How to run
```bash
pip install -r requirements.txt          # or: pip install --user -r requirements.txt
# boosters (macOS): brew install libomp && pip install xgboost lightgbm catboost

python main.py --test                    # every model at default settings + leaderboard
python main.py --tune --test             # default AND tuned version of every tunable model (~20 min)
python scripts/robustness_check.py       # repeated CV + test bootstrap for the top tuned models
# then in Jupyter: notebooks/final_selection → feature_importance → predict_demo
# (notebooks/hp_tuning.ipynb runs the same searches as --tune)
```

## 15. Project structure
```
housing-rent-prediction/
├── data/{raw,processed}/        raw CSV + cleaned_dataset.csv
├── notebooks/                   baselines, regularization, hp_tuning, final_selection,
│                                feature_importance, predict_demo, model_comparison
├── src/                         data_loading, data_cleaning, feature_engineering,
│                                preprocessing, models, tuning, evaluation, prediction, config
├── models/                      final_model.pkl, preprocessor.pkl, feature_names.json
├── reports/{metrics,predictions}/  per-model JSON/CSV, phase1–3 .md, comparison & tuning summaries
├── visualizations/             diagnostic + importance PNGs
├── _legacy_phases/             original phase1–10 exploration (archived)
├── scripts/robustness_check.py  repeated CV + test bootstrap after tuning
├── main.py, requirements.txt, README.md
```

## 16. Future improvements
- Stacking / averaging the tied top boosters (XGBoost, Gradient Boosting, LightGBM).
- SHAP values for local explanations (hook provided in `feature_importance.ipynb`).
- Quantile/robust losses for the high-rent tail; monotonic constraints.
- Spatial features from locality geocoding; recency-aware validation.
- Package as an API (FastAPI) around `src/prediction.py` and add CI + tests.

---
**Reproducibility & no leakage:** fixed 70/15/15 split (`random_state=42`, stratified on rent deciles);
all transforms fit on train only, re-fit inside every CV fold; test set untouched until final evaluation; a
fingerprint `reference_id` (`70560d1bd70af7cd`) asserts every model used identical data.

*Note: scikit-learn models are CPU-only (train <1s on ~4.7k rows); there is no Apple-GPU path for them.*
