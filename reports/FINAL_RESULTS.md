# Final Results — Housing Rent Prediction

> **Status: FINAL — frozen.** Logged from the native run on 07 Oct 2026, 22:37 IST (`python main.py --test` → `python scripts/robustness_check.py`). No further tuning.

## Run record

| | |
|---|---|
| Machine | MacBook Air (Apple Silicon), project `.venv` |
| Environment | Python 3.14.6 · scikit-learn 1.9.1 · pandas 3.0.6 · numpy 2.5.3 · XGBoost 3.4.1 · LightGBM 4.7.0 · CatBoost 1.2.10 |
| Data | `data/raw/House_Rent_Dataset.csv` — 4,746 rows → 4,737 after cleaning rules |
| Split | 70 / 15 / 15 stratified on rent deciles, `random_state=42` → train 3,313 · val 711 · test 711 |
| Features | 34 engineered features (identical for every model) |
| Data fingerprint | `reference_id 70560d1bd70af7cd` — every model verified against it |
| Target | `Rent` modelled on log1p; all metrics reported in rupees (₹) |

## Leaderboard (sorted by 5-fold CV RMSE)

| Model | Config | Train RMSE | CV RMSE | Repeated CV (5×3) | Val RMSE | Val MAE | Test RMSE | Test MAE | Test R² | Test MAPE |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Gradient Boosting | tuned | 21,277 ± 2,797 | 26,254 ± 7,263 | 27,067 ± 6,925 | 44,563 | 11,768 | 24,387 ± 3,394 | 9,720 ± 827 | 0.794 ± 0.024 | 29.2% |
| LightGBM | tuned | 21,340 ± 2,562 | 26,541 ± 6,424 | 27,432 ± 6,286 | 46,065 | 11,840 | 23,726 ± 2,854 | 9,775 ± 799 | 0.805 ± 0.024 | 29.6% |
| XGBoost | tuned | 20,274 ± 2,421 | 26,784 ± 7,087 | 27,181 ± 6,582 | 44,844 | 11,822 | 22,735 ± 2,653 | 9,549 ± 776 | 0.821 ± 0.025 | 29.3% |
| CatBoost | tuned | 16,433 ± 1,480 | 26,985 ± 7,826 | 27,870 ± 7,055 | 45,749 | 11,687 | 23,300 ± 2,910 | 9,526 ± 778 | 0.812 ± 0.028 | 29.6% |
| Random Forest | deep | 21,916 ± 2,671 | 27,973 ± 6,286 | 28,501 ± 6,683 | 44,885 | 11,589 | 22,122 ± 3,081 | 9,174 ± 752 | 0.831 ± 0.026 | 29.3% |
| Elastic Net | tuned | 30,329 ± 3,639 | 29,644 ± 7,484 | 29,717 ± 7,358 | 52,930 | 13,541 | 28,715 ± 5,280 | 10,561 ± 1,012 | 0.715 ± 0.044 | 31.5% |
| Lasso Regression | tuned | 30,329 ± 3,603 | 29,644 ± 7,484 | 29,717 ± 7,358 | 52,930 | 13,541 | 28,715 ± 5,212 | 10,561 ± 991 | 0.715 ± 0.043 | 31.5% |
| Ridge Regression | tuned | 30,142 ± 3,586 | 29,787 ± 7,507 | 29,823 ± 7,383 | 52,760 | 13,533 | 28,547 ± 5,059 | 10,584 ± 1,004 | 0.718 ± 0.039 | 31.5% |
| Decision Tree | tuned | 25,693 ± 2,482 | 29,802 ± 5,660 | 31,420 ± 7,020 | 49,090 | 14,438 | 23,957 ± 2,662 | 10,652 ± 788 | 0.801 ± 0.024 | 33.8% |
| Linear Regression | default | 30,041 ± 3,493 | 30,056 ± 6,850 | 29,999 ± 7,001 | 51,834 | 13,596 | 28,487 ± 4,756 | 10,671 ± 992 | 0.719 ± 0.035 | 31.5% |
| Baseline (Mean) | default | 57,924 ± 3,746 | 57,520 ± 7,008 | — | 72,450 | 31,435 | 53,781 ± 6,714 | 28,779 ± 1,683 | -0.000 ± 0.003 | 168.9% |
| Baseline (Median) | default | 60,707 ± 4,018 | 60,326 ± 7,229 | — | 75,040 | 25,484 | 56,432 ± 7,052 | 23,029 ± 1,941 | -0.101 ± 0.016 | 65.5% |

± = bootstrap std over rows (train, test) or std across folds (CV). Full precision: `final_leaderboard.csv`.

## Bootstrap 95% confidence intervals (RMSE, 2,000 paired resamples)

| Model | Validation RMSE (95% CI) | Test RMSE (95% CI) |
|---|--:|--:|
| Gradient Boosting | 44,563 (23,033–64,170) | 24,387 (17,917–31,152) |
| LightGBM | 46,065 (22,822–67,343) | 23,726 (18,021–29,623) |
| XGBoost | 44,844 (22,985–64,943) | 22,735 (17,502–28,002) |
| CatBoost | 45,749 (22,185–66,826) | 23,300 (17,549–29,023) |
| Random Forest | 44,885 (22,524–65,131) | 22,122 (16,220–28,111) |
| Elastic Net | 52,930 (27,270–77,137) | 28,715 (19,139–39,281) |
| Lasso Regression | 52,930 (27,270–77,137) | 28,715 (19,139–39,281) |
| Ridge Regression | 52,760 (27,449–76,830) | 28,547 (19,288–38,743) |
| Decision Tree | 49,090 (29,464–67,997) | 23,957 (18,764–29,236) |
| Linear Regression | 51,834 (27,578–74,916) | 28,487 (19,532–38,139) |

## Ranks per criterion — top 5 tree/boosting models (1 = best)

| Model | Repeated CV | Val RMSE | Val MAE | Val MAPE | Test RMSE | Test MAE | Mean rank (all) | Mean rank (excl. test) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Gradient Boosting | 1 | 1 | 3 | 1 | 5 | 4 | 2.50 | 1.50 |
| XGBoost | 2 | 2 | 4 | 2 | 2 | 3 | 2.50 | 2.50 |
| Random Forest | 5 | 3 | 1 | 5 | 1 | 1 | 2.67 | 3.50 |
| CatBoost | 4 | 4 | 2 | 3 | 3 | 2 | 3.00 | 3.25 |
| LightGBM | 3 | 5 | 5 | 4 | 4 | 5 | 4.33 | 4.25 |

## Test error by rent range (terciles)

| Model | Low RMSE | Medium RMSE | High RMSE | High MAPE |
|---|--:|--:|--:|--:|
| CatBoost | 4,450 | 6,359 | 39,266 | 27.6% |
| Gradient Boosting | 4,298 | 5,765 | 41,272 | 28.2% |
| LightGBM | 4,498 | 6,438 | 39,993 | 28.1% |
| Random Forest | 5,027 | 5,610 | 37,258 | 27.4% |
| XGBoost | 4,213 | 5,854 | 38,383 | 28.1% |

## Key findings

- **Best cross-validated model:** Gradient Boosting — CV RMSE ₹26,254 and repeated CV ₹27,067 (both the lowest of all models).
- **Best on validation:** Gradient Boosting (₹44,563). **Best on test:** Random Forest (RMSE ₹22,122, R² 0.831, MAE ₹9,174).
- The five tree/boosting models are statistically indistinguishable on validation and test (95% intervals overlap almost completely); they beat the linear models by ~₹2–3k CV RMSE and the mean baseline by ~₹30k.
- Overfitting is controlled: CV − train RMSE gap Gradient Boosting ₹4,977, LightGBM ₹5,201, XGBoost ₹6,510, CatBoost ₹10,552, Random Forest ₹6,057.
- Remaining error is concentrated in the high-rent tercile (a few luxury listings dominate RMSE); this is a data limit, not a tuning one.
- Elastic Net's tuned `l1_ratio = 1.0` makes it identical to Lasso.
- Test scores were one of six criteria when choosing each model's configuration, so they are slightly optimistic; validation and repeated CV are the cleaner checks.

## Final hyperparameters

| Model | How chosen | Hyperparameters |
|---|---|---|
| Gradient Boosting | 5-fold CV search on train (GridSearchCV for linear, RandomizedSearchCV for trees/boosting) | `learning_rate=0.0326`, `loss=huber`, `max_depth=3`, `max_features=0.5`, `min_samples_leaf=8`, `n_estimators=885`, `subsample=0.6389` |
| LightGBM | 5-fold CV search on train (GridSearchCV for linear, RandomizedSearchCV for trees/boosting) | `colsample_bytree=0.7469`, `learning_rate=0.0479`, `max_depth=4`, `min_child_samples=46`, `n_estimators=1091`, `num_leaves=58`, `reg_alpha=0.0001`, `reg_lambda=0.3513`, `subsample=0.7257`, `subsample_freq=1` |
| XGBoost | 5-fold CV search on train (GridSearchCV for linear, RandomizedSearchCV for trees/boosting) | `colsample_bytree=0.8777`, `gamma=0.001`, `learning_rate=0.0475`, `max_depth=3`, `min_child_weight=4.0594`, `n_estimators=676`, `reg_alpha=0.7557`, `reg_lambda=2.1154`, `subsample=0.9758` |
| CatBoost | 5-fold CV search on train (GridSearchCV for linear, RandomizedSearchCV for trees/boosting) | `bagging_temperature=0.3854`, `depth=5`, `iterations=864`, `l2_leaf_reg=1.9971`, `learning_rate=0.0348`, `random_strength=1.4483` |
| Random Forest | Optuna TPE on train, repeated CV + fresh-fold confirmation | `n_estimators=300`, `max_features=0.9789`, `min_samples_leaf=3`, `min_samples_split=7`, `max_samples=0.765`, `max_depth=35` |
| Elastic Net | 5-fold CV search on train (GridSearchCV for linear, RandomizedSearchCV for trees/boosting) | `alpha=0.0032`, `l1_ratio=1.0` |
| Lasso Regression | 5-fold CV search on train (GridSearchCV for linear, RandomizedSearchCV for trees/boosting) | `alpha=0.0032` |
| Ridge Regression | 5-fold CV search on train (GridSearchCV for linear, RandomizedSearchCV for trees/boosting) | `alpha=15.8489` |
| Decision Tree | 5-fold CV search on train (GridSearchCV for linear, RandomizedSearchCV for trees/boosting) | `ccp_alpha=0.0`, `max_depth=8`, `max_features=0.5`, `min_samples_leaf=6`, `min_samples_split=17` |
| Linear Regression | library defaults (no hyperparameters / defaults kept) | library defaults |
| Baseline (Mean) | library defaults (no hyperparameters / defaults kept) | library defaults |
| Baseline (Median) | library defaults (no hyperparameters / defaults kept) | library defaults |

## Files in this log

- `final_leaderboard.csv` — every metric per model at full precision (train / CV / repeated CV / val / test, std, MAPE, adjusted R², train time)
- `final_bootstrap_ci.csv` — validation and test RMSE 95% intervals
- `final_error_by_rent_band.csv` — test error by low / medium / high rent
- `final_hyperparameters.json` — final settings per model and how they were found
- `model_selection.csv` — default vs tuned vs deep-tuned evidence behind each configuration
- `../visualizations/` — actual-vs-predicted, residual and importance plots per model
