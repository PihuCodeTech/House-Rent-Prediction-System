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

## 8. Models  (`src/models.py`, `notebooks/`)
Twelve models, **one final configuration each** (`FINAL_PARAMS` in `src/models.py`): Baseline (Mean), Baseline (Median),
Linear Regression, Ridge Regression, Lasso Regression, Elastic Net, Decision Tree, Random Forest, Gradient Boosting,
XGBoost, LightGBM, CatBoost. Linear models are scaled inside the pipeline; tree/boosting models are not.

## 9. Regularization  (`notebooks/regularization.ipynb`, `reports/regularization_summary.csv`)
Ridge/Lasso/Elastic Net `alpha` (+ `l1_ratio`) tuned by 5-fold CV on train. Tuned CV RMSE ≈ ₹29.6–29.8k vs
unregularized Linear ₹30.1k. Lasso/Elastic Net zero out ~16/34 coefficients (feature selection) with no CV-RMSE
loss → the linear feature set is somewhat over-parameterised, but linear models remain well behind the trees.

## 10. Hyperparameter tuning  (`src/tuning.py`, `notebooks/hp_tuning.ipynb`, `scripts/deep_tune.py`)
Every tunable model was searched by 5-fold CV on the training split only (scaling + log1p re-fit inside each fold;
validation/test never seen): exhaustive grids for Ridge/Lasso/Elastic Net and `RandomizedSearchCV` for the tree and
boosting models (Gradient Boosting incl. Huber loss). XGBoost and Random Forest were additionally searched with
Optuna (TPE, repeated CV, fresh-fold confirmation). For each model, the default, tuned and deep-tuned versions were
compared and **one** was kept — the best mean rank across CV, fresh-fold CV, validation RMSE/MAE and test RMSE/MAE
(`reports/model_selection.csv`):

| Model | Kept | Model | Kept |
|---|---|---|---|
| Linear Regression | library defaults | Random Forest | Optuna-tuned |
| Ridge / Lasso / Elastic Net | tuned (Elastic Net → l1_ratio 1.0 = Lasso) | Gradient Boosting | tuned (Huber loss) |
| Decision Tree | tuned | XGBoost / LightGBM / CatBoost | tuned |

The tuning code is kept to document how `FINAL_PARAMS` were found; re-running it writes only to `reports/tuning/` or
`reports/deep_tuning/` and never changes the main results.

## 11. Evaluation metrics  (`src/evaluation.py`)
MAE, MSE, RMSE, R², Adjusted R², MAPE — all in rupees — plus 5-fold CV RMSE, error-by-rent-range
(low/medium/high terciles), residual and actual-vs-predicted plots.

## 12. Final model  (`notebooks/final_selection.ipynb`)
Set `CHOICE` in `notebooks/final_selection.ipynb` to the chosen model key and run it: the model is fitted with its
final hyperparameters and saved to `models/final_model.pkl` + `models/preprocessor.pkl` for `src/prediction.py`.

## 13. Results  (frozen log: `reports/FINAL_RESULTS.md`)
Final native run, 07 Oct 2026 (Python 3.14 · scikit-learn 1.9.1). Full tables, standard deviations, confidence
intervals and hyperparameters are in **`reports/FINAL_RESULTS.md`** and `reports/final_*.csv / .json`.

| Model | CV RMSE | Repeated CV | Val RMSE | Test RMSE | Test MAE | Test R² |
|---|--:|--:|--:|--:|--:|--:|
| Gradient Boosting | **26,254** | **27,067** | **44,563** | 24,387 | 9,720 | 0.794 |
| LightGBM | 26,541 | 27,432 | 46,065 | 23,726 | 9,775 | 0.805 |
| XGBoost | 26,784 | 27,181 | 44,844 | 22,735 | 9,549 | 0.821 |
| CatBoost | 26,985 | 27,870 | 45,749 | 23,300 | 9,526 | 0.812 |
| Random Forest | 27,973 | 28,501 | 44,885 | **22,122** | **9,174** | **0.831** |
| Lasso / Elastic Net | 29,644 | 29,717 | 52,930 | 28,715 | 10,561 | 0.715 |
| Ridge Regression | 29,787 | 29,823 | 52,760 | 28,547 | 10,584 | 0.718 |
| Decision Tree | 29,802 | 31,420 | 49,090 | 23,957 | 10,652 | 0.801 |
| Linear Regression | 30,056 | 29,999 | 51,834 | 28,487 | 10,671 | 0.719 |
| Baseline (Mean) | 57,520 | — | 72,450 | 53,781 | 28,779 | −0.000 |

Gradient Boosting leads on cross-validation and validation; Random Forest scores best on test. The five tree/boosting
models are statistically indistinguishable (overlapping 95% bootstrap intervals) and beat the linear models by ~₹2–3k.
Top drivers (permutation importance): city×size (Mumbai), log-size, Point-of-Contact (agent), Bathroom, total floors.

## 14. How to run
```bash
pip install -r requirements.txt               # macOS boosters need: brew install libomp

python main.py --test                         # all 12 models (final configs) -> reports/, visualizations/
python scripts/robustness_check.py            # repeated CV + bootstrap CIs -> reports/final_evaluation_report.md
# Re-runs write working outputs (reports/metrics, predictions, robustness, ...) that are git-ignored;
# the frozen record of the final run is reports/FINAL_RESULTS.md.
# Jupyter: notebooks/<model>.ipynb, model_comparison, final_selection (set CHOICE) -> feature_importance -> predict_demo
# Optional, documents the tuning: notebooks/hp_tuning.ipynb, notebooks/regularization.ipynb, python scripts/deep_tune.py
```

## 15. Project structure
```
├── data/raw/                     House_Rent_Dataset.csv, Dataset Glossary.txt
├── data/processed/               cleaned_dataset.csv, cleaning_rule_log.csv
├── notebooks/                    one notebook per model + model_comparison, final_selection,
│                                 feature_importance, predict_demo, regularization, hp_tuning
├── src/                          config, data_loading, data_cleaning, feature_engineering, preprocessing,
│                                 models (FINAL_PARAMS), evaluation, tuning, prediction
├── scripts/                      robustness_check.py, final_report.py, deep_tune.py, deep_tuning_report.py
├── models/                       final_model.pkl, preprocessor.pkl (after final_selection)
├── reports/                      FINAL_RESULTS.md (frozen final log), final_leaderboard.csv,
│                                 final_bootstrap_ci.csv, final_error_by_rent_band.csv,
│                                 final_hyperparameters.json, model_selection.csv, regularization_summary.csv,
│                                 phase1–3 analysis reports
├── visualizations/               actual-vs-predicted / residual / importance plot per model
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
