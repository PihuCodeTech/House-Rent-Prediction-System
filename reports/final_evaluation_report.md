# Final Evaluation Report

## 1. Rule applied

**keep tuned hyperparameters only where they beat the default on test_rmse** (decided on the native Mac leaderboard, `reports/model_comparison.csv`). The tuned settings stay on record in `reports/tuned_best_params.json`; `reports/final_configs.json` records which configuration each model uses and why.

> **Caveat.** Choosing default-vs-tuned by test RMSE uses the test set for model selection, so the test scores of the chosen configurations are optimistically biased. The **validation set has not been used by any tuning or revert decision** and is the cleanest held-out check; repeated CV on the training split is the most stable one.

| Model | Test RMSE default | Test RMSE tuned | Final config |
|---|--:|--:|---|
| Ridge Regression | 28,481 | 28,547 | **default** |
| Lasso Regression | 55,473 | 28,715 | **tuned** |
| Elastic Net | 53,760 | 28,715 | **tuned** |
| Decision Tree Regressor | 32,251 | 23,957 | **tuned** |
| Random Forest Regressor | 21,639 | 21,798 | **default** |
| Gradient Boosting Regressor | 22,639 | 24,387 | **default** |
| XGBoost Regressor | 21,313 | 23,517 | **default** |
| LightGBM Regressor | 24,082 | 23,726 | **tuned** |
| CatBoost Regressor | 23,760 | 23,300 | **tuned** |

Linear Regression and the mean/median baselines have no hyperparameters (default).

## 2. Final leaderboard (every model with its final configuration)

| Model | CV RMSE | Repeated 5x3 CV | Val RMSE | Val MAE | Val MAPE | Test RMSE* | Test MAE* | Test R²* | Train RMSE |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| LightGBM Regressor (tuned) | 26,541 | 27,432 | 46,065 | 11,840 | 29.5% | 23,726 | 9,775 | 0.805 | 21,340 |
| CatBoost Regressor (tuned) | 26,985 | 27,870 | 45,749 | 11,687 | 29.1% | 23,300 | 9,526 | 0.812 | 16,433 |
| Gradient Boosting Regressor (default) | 28,000 | 28,122 | 45,882 | 11,896 | 29.4% | 22,639 | 9,443 | 0.823 | 21,265 |
| Random Forest Regressor (default) | 28,896 | 29,361 | 45,742 | 11,668 | 30.5% | 21,931 | 9,094 | 0.834 | 13,573 |
| XGBoost Regressor (default) | 28,419 | 29,387 | 45,217 | 12,165 | 30.6% | 21,313 | 9,670 | 0.843 | 6,785 |
| Elastic Net (tuned) | 29,644 | 29,717 | 52,930 | 13,541 | 31.8% | 28,715 | 10,561 | 0.715 | 30,329 |
| Lasso Regression (tuned) | 29,644 | 29,717 | 52,930 | 13,541 | 31.8% | 28,715 | 10,561 | 0.715 | 30,329 |
| Ridge Regression (default) | 29,992 | 29,952 | 51,941 | 13,589 | 31.8% | 28,481 | 10,659 | 0.719 | 30,035 |
| Linear Regression (default) | 30,056 | 29,999 | 51,834 | 13,596 | 31.8% | 28,487 | 10,671 | 0.719 | 30,041 |
| Decision Tree Regressor (tuned) | 29,802 | 31,420 | 49,090 | 14,438 | 35.6% | 23,957 | 10,652 | 0.801 | 25,693 |
| Baseline (mean rent) (default) | 57,520 | — | 72,450 | 31,435 | 170.4% | 53,781 | 28,779 | -0.000 | 57,924 |
| Baseline (median rent) (default) | 60,326 | — | 75,040 | 25,484 | 65.9% | 56,432 | 23,029 | -0.101 | 60,707 |

\* Test metrics are optimistic for configurations that were chosen by test RMSE.

**Random Forest (default) is machine-dependent.** Fully grown trees flip near-tied splits across CPU architectures. On your Mac it scores CV 28,924 / val 45,296 / test 21,639 (R² 0.838); in the run above, 28,896 / 45,742 / 21,931. Its revert decision (default 21,639 vs tuned 21,798 on the Mac) only holds on the Mac; here, tuned would win. Every other model reproduces to the rupee across machines.

## 3. How certain are these rankings? (paired bootstrap, 2,000 resamples)

| Model | Val RMSE (95% CI) | Test RMSE (95% CI) |
|---|--:|--:|
| LightGBM Regressor (tuned) | 46,065 (22,822–67,343) | 23,726 (18,021–29,623) |
| CatBoost Regressor (tuned) | 45,749 (22,185–66,826) | 23,300 (17,549–29,023) |
| Gradient Boosting Regressor (default) | 45,882 (22,782–66,519) | 22,639 (17,308–28,015) |
| Random Forest Regressor (default) | 45,742 (21,810–67,426) | 21,931 (15,949–27,596) |
| XGBoost Regressor (default) | 45,217 (22,322–66,256) | 21,313 (17,164–25,209) |
| Elastic Net (tuned) | 52,930 (27,270–77,137) | 28,715 (19,139–39,281) |
| Lasso Regression (tuned) | 52,930 (27,270–77,137) | 28,715 (19,139–39,281) |
| Ridge Regression (default) | 51,941 (27,549–75,160) | 28,481 (19,476–38,120) |
| Linear Regression (default) | 51,834 (27,578–74,916) | 28,487 (19,532–38,139) |
| Decision Tree Regressor (tuned) | 49,090 (29,464–67,997) | 23,957 (18,764–29,236) |

All tree/boosting intervals overlap almost completely on both held-out sets: no model is separated from the others by RMSE on a single split. A handful of luxury listings (up to ₹1.2M) dominate squared error.

## 4. Ranks per criterion (top five models; 1 = best)

| Model | Rep. CV | Val RMSE | Val MAE | Val MAPE | Test RMSE* | Test MAE* | Mean rank (all) | Mean rank (excl. test) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| CatBoost Regressor (tuned) | 2 | 3 | 2 | 1 | 4 | 3 | 2.50 | **2.00** |
| Random Forest Regressor (default) | 4 | 2 | 1 | 4 | 2 | 1 | 2.33 | **2.75** |
| LightGBM Regressor (tuned) | 1 | 5 | 3 | 3 | 5 | 5 | 3.67 | **3.00** |
| Gradient Boosting Regressor (default) | 3 | 4 | 4 | 2 | 3 | 2 | 3.00 | **3.25** |
| XGBoost Regressor (default) | 5 | 1 | 5 | 5 | 1 | 4 | 3.50 | **4.00** |

## 5. Reading the evidence

- **CatBoost (tuned)** is the most consistent: best mean rank on the evidence no decision has touched (2.0), 2nd on repeated CV, best validation MAPE, 2nd validation MAE, and never worse than 4th anywhere.
- **Random Forest (default)** has the lowest MAE on validation and test and the best mean rank when test is included (2.33), but ranks 4th on repeated CV, overfits (train ₹13.6k vs CV ₹28.9k) and its numbers change across machines.
- **XGBoost (default)** has the best headline validation and test RMSE/R², but ranks last on repeated CV, validation MAE and MAPE, and overfits most (train ₹6.8k vs CV ₹28.4k): its RMSE advantage comes from a few extreme listings.
- **LightGBM (tuned)** has the best repeated CV (₹27,432) but the weakest validation and test scores of the five.
- Linear models (Lasso/Elastic Net tuned, Ridge, Linear) trail the tree models by ~₹2–3k CV RMSE; the tuned Decision Tree is the weakest tree model on repeated CV.

## 6. Next step

Pick the model, then set `CHOICE` in `notebooks/final_selection.ipynb` (e.g. `CHOICE = 'catboost_model'`) and run it to save `models/final_model.pkl` with that model's final configuration. Until then, `final_model.pkl` still holds the tuned XGBoost from the previous step (no longer in the final configuration set).

Reproduce: `python main.py --decide` → `python main.py --final` → `python scripts/robustness_check.py --final`.