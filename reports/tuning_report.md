# Hyperparameter Tuning Report

All numbers come from one environment (Python 3.13, scikit-learn 1.9.1, XGBoost 3.4.1, LightGBM 4.7.0, CatBoost 1.2.10, matching the project's macOS venv). Same prepared data for every model (fingerprint `70560d1bd70af7cd`), same 70/15/15 split, rupee-scale metrics.

**Protocol.** Each search is 5-fold CV on the training split only (scaling + log1p target re-fit inside every fold); validation and test are never seen by a search. Ridge/Lasso/Elastic Net: exhaustive grid. Trees and boosting: `RandomizedSearchCV` over capacity + regularisation parameters. Final choice = lowest repeated 5x3 CV RMSE (training-split evidence only); the test set is reported, not used to choose.

## 1. Default vs tuned

| Model | CV RMSE default → tuned | Δ CV | Val RMSE default → tuned | Test RMSE default → tuned | Test R² tuned | Candidates |
|---|--:|--:|--:|--:|--:|--:|
| Gradient Boosting Regressor | 28,000 → **26,254** | -6.2% | 45,882 → 44,563 | 22,639 → 24,387 | 0.794 | 60 |
| XGBoost Regressor | 28,419 → **26,517** | -6.7% | 45,217 → 44,801 | 21,313 → 23,517 | 0.809 | 100 |
| LightGBM Regressor | 28,168 → **26,541** | -5.8% | 46,328 → 46,065 | 24,082 → 23,726 | 0.805 | 100 |
| CatBoost Regressor | 27,795 → **26,985** | -2.9% | 45,388 → 45,749 | 23,760 → 23,300 | 0.812 | 40 |
| Random Forest Regressor | 28,896 → **28,128** | -2.7% | 45,742 → 44,710 | 21,931 → 21,798 | 0.836 | 40 |
| Elastic Net | 57,448 → **29,644** | -48.4% | 72,717 → 52,930 | 53,760 → 28,715 | 0.715 | 168 |
| Lasso Regression | 59,321 → **29,644** | -50.0% | 74,205 → 52,930 | 55,473 → 28,715 | 0.715 | 31 |
| Ridge Regression | 29,992 → **29,787** | -0.7% | 51,941 → 52,760 | 28,481 → 28,547 | 0.718 | 31 |
| Decision Tree Regressor | 36,915 → **29,802** | -19.3% | 53,581 → 49,090 | 29,508 → 23,957 | 0.801 | 100 |
| Linear Regression | 30,056 (no hyperparameters) | — | 51,834 | 28,487 | 0.719 | — |

Mean/median baselines have no hyperparameters (CV RMSE 57,520 / 60,326).

## 2. Ranking the strongest tuned models (repeated 5x3 CV on train)

| Rank | Model | 5-fold CV | Repeated 5x3 CV | Val RMSE | Test RMSE | Test MAE | Test R² | Test MAPE |
|--:|---|--:|--:|--:|--:|--:|--:|--:|
| 1 | XGBoost Regressor (tuned) | 26,517 | **27,025** | 44,801 | 23,517 | 9,629 | 0.809 | 29.3% |
| 2 | Gradient Boosting Regressor (tuned) | 26,254 | **27,067** | 44,563 | 24,387 | 9,720 | 0.794 | 29.2% |
| 3 | LightGBM Regressor (tuned) | 26,541 | **27,432** | 46,065 | 23,726 | 9,775 | 0.805 | 29.6% |
| 4 | CatBoost Regressor (tuned) | 26,985 | **27,870** | 45,749 | 23,300 | 9,526 | 0.812 | 29.6% |
| 5 | Random Forest Regressor (tuned) | 28,128 | **28,624** | 44,710 | 21,798 | 9,146 | 0.836 | 29.3% |
| 6 | Lasso Regression (tuned) | 29,644 | **29,717** | 52,930 | 28,715 | 10,561 | 0.715 | 31.5% |

## 3. Is the test-set ranking meaningful? (paired bootstrap, 2,000 resamples)

| Model | Test RMSE | 95% CI |
|---|--:|--:|
| xgboost_model | 21,313 | 17,164 – 25,209 |
| random_forest_tuned | 21,798 | 16,080 – 27,514 |
| random_forest | 21,931 | 15,949 – 27,596 |
| gradient_boosting | 22,639 | 17,308 – 28,015 |
| catboost_model_tuned | 23,300 | 17,549 – 29,023 |
| xgboost_model_tuned | 23,517 | 17,737 – 29,317 |
| lightgbm_model_tuned | 23,726 | 18,021 – 29,623 |
| catboost_model | 23,760 | 17,810 – 29,517 |
| lightgbm_model | 24,082 | 17,919 – 30,961 |
| gradient_boosting_tuned | 24,387 | 17,917 – 31,152 |
| lasso_regression_tuned | 28,715 | 19,139 – 39,281 |
| lasso_regression | 55,473 | 42,431 – 69,758 |

All tree/boosting intervals overlap heavily. A handful of luxury listings dominate single-split RMSE (in validation, 5 of 711 rows carry ~80% of the squared error), so test-set gaps of ₹1–3k between top models are not evidence of a better model. Repeated CV on 3,313 training rows is the more reliable signal.
## 4. Comparison with this project's earlier tuning (legacy phases 7–9, same pipeline and split)

| Model | Legacy tuned CV | Now tuned CV | Legacy repeated CV | Now repeated CV | Legacy test RMSE | Now test RMSE |
|---|--:|--:|--:|--:|--:|--:|
| Ridge | 29,787 (α=15.85) | 29,787 (α=15.85) | — | — | 28,547 | 28,547 |
| Lasso | 29,651 (α=0.0025) | 29,644 (α=0.0032) | — | 29,717 | 28,689 | 28,715 |
| Elastic Net | 29,645 (α=0.0032, l1=0.99) | 29,644 (α=0.0032, l1=1.0) | — | — | 28,714 | 28,715 |
| Random Forest | 28,095 | 28,128 | 28,427 | 28,624 | 22,564 | 21,798 |
| Gradient Boosting | 26,242 (Huber) | 26,254 (Huber) | 27,086 | 27,067 | 24,588 | 24,387 |
| XGBoost / LightGBM / CatBoost | not tuned before | 26,517 / 26,541 / 26,985 | — | 27,025 / 27,432 / 27,870 | — | 23,517 / 23,726 / 23,300 |

The tuned linear models reproduce the legacy results exactly, and Random Forest and Gradient Boosting land within
₹200 of them on repeated CV. Gradient Boosting even converged to nearly the same configuration (learning rate
0.033, Huber loss, depth 3, max_features 0.5, min_samples_leaf 8, subsample 0.64). The legacy run showed the same
test-set pattern seen here: tuned Gradient Boosting has the best CV but a *worse* test RMSE than its default.
Elastic Net's best `l1_ratio` is 1.0, i.e. it collapses to Lasso.

## 5. Comparison with published models on the same Kaggle dataset

Different splits and preprocessing, so this is a sanity range, not a like-for-like ranking.

| Source | Model / setup | R² | RMSE (₹) | MAE (₹) | MAPE |
|---|---|--:|--:|--:|--:|
| [sujalwarke28/Residential_rent_pred_model](https://github.com/sujalwarke28/Residential_rent_pred_model) | Gradient Boosting on log1p(Rent), 80/20 split stratified by city, untouched test | 0.714 | 32,716 | 10,135 | 26.7% |
| [RohitSuwalka/house-rent-prediction](https://github.com/RohitSuwalka/house-rent-prediction) | Gradient Boosting (300 trees, lr 0.08, depth 5), 80/20, no target transform reported | 0.41 | 48,417 | 13,814 | — |
| [saherliaqat227-cell/House-Rent-Prediction-ML](https://github.com/saherliaqat227-cell/House-Rent-Prediction-ML) | Linear Regression, 80/20 | 0.72 | — | — | — |
| **This project — tuned XGBoost** | 5-fold CV on train (split-independent) | 0.787 | 26,517 | 9,625 | — |
| | validation (711 rows, includes a ₹1.2M listing) | 0.617 | 44,801 | 11,958 | 29.1% |
| | test (711 rows) | 0.809 | 23,517 | 9,629 | 29.3% |

Our results sit inside the published range, at the stronger end: the closest methodological match (log1p target,
untouched test) reports R² 0.71 / RMSE ₹32.7k / MAE ₹10.1k, which falls between our validation and test figures,
with our MAE slightly lower and our MAPE slightly higher (29% vs 27%). Nothing suggests leakage (our numbers are
not implausibly better) or a broken pipeline (not worse).

## 6. Conclusion

- **Best model: tuned XGBoost** (lowest repeated-CV RMSE ₹27,025; 5-fold CV ₹26,517, down from ₹28,419 at
  defaults). Saved as `models/final_model.pkl`. Test: RMSE ₹23,517, MAE ₹9,629, R² 0.809, MAPE 29.3%.
- **Tuned Gradient Boosting (Huber loss) is statistically tied** (repeated CV ₹27,067); LightGBM is close behind.
- **Biggest tuning gains:** Lasso / Elastic Net (default α=1 wiped out every coefficient, worse than the mean
  baseline; tuned α≈0.003 → CV ₹29.6k) and the Decision Tree (fully grown → depth 8, min_samples_leaf 6: CV
  ₹36.9k → ₹29.8k, test R² 0.70 → 0.80). Boosters gained 3–7% CV RMSE; Ridge was already near-optimal.
- **On the test set**, tuning helped some models and hurt others (XGBoost, Gradient Boosting). With 95% bootstrap
  intervals of roughly ±₹6k and a few luxury listings dominating RMSE, these test gaps are noise; the defaults'
  better test scores in the pre-tuning table do not make them better models.

Reproduce on your machine: `python main.py --tune --test` → `python scripts/robustness_check.py` →
`notebooks/final_selection.ipynb`. Untuned Decision Tree / Random Forest numbers can differ slightly across CPU
architectures (fully grown trees flip near-tied splits); all other models reproduce to the rupee.
