# ACTION PLAN — Housing Rent Prediction (Supervised Learning Experiment)

> **For Claude Code.** This is the execution playbook. Read it fully before starting.
> Execute **one phase at a time**, show results, explain decisions, then **STOP and wait**
> for my instruction before the next phase. Do not auto-proceed.
> Always-on guardrails live in `CLAUDE.md`; this file holds the phase detail.

---

## 0. CONTEXT

- **Dataset:** `House_Rent_Dataset.csv` (~4,746 rows, 12 columns). `Dataset Glossary.txt`
  defines every column — use it in Phase 1.
- **Target:** `Rent` (continuous) -> **regression**, not classification.
- **Columns:** `Posted On` (listing-post date), `BHK` (int), `Rent` (target), `Size` (sqft),
  `Floor` (text, e.g. "Ground out of 2"), `Area Type`, `Area Locality` (high-cardinality),
  `City`, `Furnishing Status`, `Tenant Preferred`, `Bathroom` (int), `Point of Contact`.
- **Where models run:** Google Colab. **Each model = one standalone Jupyter notebook
  (`.ipynb`)** that opens directly in Colab and runs top-to-bottom.
- **Data load:** `House_Rent_Dataset.csv` is uploaded into the Colab working directory (read via
  `pd.read_csv("House_Rent_Dataset.csv")`, Phase 1/5 only). The Phase-5 output is saved as a
  single bundle **`prepared_data.joblib`**, uploaded the same way; model notebooks load that.
  No `files.upload()`, no Drive mount, no path variable.
- **Paths:** `./phase1` ... `./phase10` are folders inside THIS project, not filesystem root.

---

## 1. THE NON-NEGOTIABLE RULE

> **Every model trains and evaluates on the EXACT same prepared data and the EXACT same
> train/validation/test split.** Same rows, same features, same target, same seed
> (`RANDOM_STATE = 42`), same evaluation.

If any model's data fingerprint differs from the Phase-5 reference -> **stop and report.
Never silently continue.**

---

## 2. STANDARD ML WORKFLOW — PIPELINES, NOT HARDCODING  *(read before writing any code)*

The data preparation must be done the standard, reproducible way. **No manual, eyeballed,
row-by-row surgery.** Every cleaning action is either a stateless *rule* or a *fitted
transformer* — never a hand-picked index.

### 2.1 EDA is look-only
Phase 1 inspection informs decisions but **never modifies or saves data and never drops a
row**. Its only output is a documented list of decisions.

### 2.2 Two kinds of operations
- **Stateless rule-based cleaning** (learns nothing from the distribution): expressed as
  conditions, applied identically, safe before the split.
  - OK: `df = df[df["Rent"] > 0]`, `df.drop_duplicates()`, parse `Floor`, parse dates.
  - FORBIDDEN: dropping specific indices (`df.drop([13, 4213])`), any per-row manual edit,
    any threshold a human read off a chart and typed in by hand without deriving it in code.
- **Learned transforms** (compute a parameter from data: median, category map, mean/std,
  feature ranking): **must be `fit` on TRAIN only**, then applied to val/test. Assemble them
  in a scikit-learn `Pipeline` + `ColumnTransformer` so `.fit()` structurally cannot see
  val/test.

### 2.3 Never clean or drop validation/test rows on distribution
Val/test stand in for unseen data. Outliers are **never hand-deleted** from them. For outliers:
either keep them (and use `RobustScaler` / tree models), or apply an IQR/threshold rule whose
cutoff is **computed on train and applied to train only** — val/test kept intact. Dropping
test rows to improve metrics is forbidden.

### 2.4 Prepare once, consume everywhere
Phase 5 builds and `fit`s the preparation **once on train**, transforms all three splits, and
**saves them to disk**. Model notebooks **load the saved splits** and do not re-derive any
cleaning. This is what guarantees identical data and removes all per-notebook drift.

---

## 3. SPLIT & LEAKAGE POLICY

### 70 / 15 / 15 split (done once, in Phase 5)
```python
from sklearn.model_selection import train_test_split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=bins)          # 15% test
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42, stratify=bins_temp)  # 0.1765*0.85≈0.15
# -> 70 / 15 / 15
```
Stratify on target bins if Phase 3 says so; **bins are never input features.**

### Zero leakage (train -> validation -> test)
1. Split happens **before** any fitted step; only stateless rules run pre-split.
2. Impute / encode / scale / feature-select / the `log1p` decision are **fit on train only**,
   via `Pipeline` + `ColumnTransformer`, then applied to val/test.
3. In tuning, every fitted transform lives **inside** the Pipeline passed to the CV search, so
   it is re-fit within each fold.
4. The 15% validation set is for model comparison only — never fitting.
5. The 15% test set is untouched until Phase 9 (`EVALUATE_TEST = False` until then).
6. No target-derived input features.

---

## 4. RESOLVED DESIGN DECISIONS (apply throughout)
1. Test metrics gated by `EVALUATE_TEST` (False P6-8, True only P9).
2. Split precedes any fitted step; prepare-once/consume-everywhere (Section 2.4).
3. No `property age` (post date ≠ build date); only posting month / dow / days-since-earliest,
   if Phase 1 shows signal.
4. `Area Locality`, if target-encoded, is encoded inside the train-fitted `ColumnTransformer`
   (and re-fit per fold during tuning).
5. Each notebook **loads Phase-5 prepared splits** and verifies the fingerprint before training.
6. CV on the 70% train fills "CV RMSE" for every model; the 15% val set is the selection check.
7. If Rent is `log1p`-transformed, back-transform with `np.expm1`; report all metrics in rupees.
8. Ridge / Lasso / ElasticNet at default alpha in P6; tuned in P7.

---

## 5. WORKING RULES
1. **Phase-gated.** Finish -> report findings, decisions, effect on next phase -> **STOP**.
2. **Evidence-driven.** No preprocessing decision before Phase 1.
3. **No repo structure yet.** Flat `./phase1` ... `./phase10`.
4. **Freeze at Phase 5.** Prep methodology fixed thereafter.

---

## 6. PHASES

### PHASE 1 - Dataset Inspection *(look-only; no data modified)*  -> `phase1_inspection.ipynb`
Report: shape, dtypes, missing values, duplicates, unique counts, categorical vs numerical,
target distribution (min/max/mean/median/std/skew), potential outliers, constant / near-
constant / suspicious / ID columns, possible leakage, and **cardinality of each categorical**
(flag `Area Locality`). Cross-check `Dataset Glossary.txt`. Assess posting-date signal.
Visualize (rent histogram, boxplots, correlation). Output = documented decisions only. **STOP.**

### PHASE 2 - Data Cleaning & Filtering *(as reusable rules + transformers)*
Translate Phase 1 findings into code, not manual edits:
- **Stateless rules** (pre-split): drop_duplicates; validity filters (Rent>0, Size>0, sane
  BHK/Bathroom) written as conditions; parse `Floor` and dates; fix formatting. Document each
  rule and how many rows it affects — but express it as a rule, never a hardcoded index.
- **Learned cleaning** (deferred to the train-fitted pipeline): imputation strategy chosen now,
  **fit later on train only**.
- **Outliers:** investigate (IQR / z-score / distribution); **do not hand-delete**. Decide:
  keep + robust methods, OR a train-derived IQR rule applied to train only. Document choice.
Column removals: only demonstrable IDs / indexes / duplicates / leakage, each documented. **STOP.**

### PHASE 3 - Target Distribution & Regression Balancing
Regression -> **no SMOTE / over- / under-sampling**. Analyze skew / sparse ranges. Decide on
`np.log1p(Rent)` (back-transform with `np.expm1`, report in rupees). Decide on target-bin
**stratified splitting** (`pd.qcut(y, q=5, duplicates="drop")`). **Bins for splitting only.** **STOP.**

### PHASE 4 - Feature Engineering
Only what the columns support, as code: `Floor` -> current/total floor (+ratio); size-per-BHK;
bathroom-to-BHK ratio; posting-date features; furnishing/area-type/city encodings;
log-transformed numerics; sensible interactions. **No invented features** (no property age).
Stateless derivations run pre-split; fitted encodings go in the train-fitted pipeline.
Avoid target leakage. **STOP.**

### PHASE 5 - FREEZE & BUILD THE PIPELINE *(critical — prepare once)*  -> `phase5_prepare_data.ipynb`
Do it once, the standard way:
1. Load raw CSV -> apply **stateless rules** (Section 2.2) + feature engineering.
2. **70/15/15 split** with `random_state=42` (+ stratify bins if chosen).
3. Build a `ColumnTransformer`/`Pipeline` for impute + encode (+ feature-select). **`fit` on
   X_train only**; `transform` train, val, test.
4. **Save ONE bundle to disk** (uploaded to Colab next to the CSV) — default
   **`prepared_data.joblib`** via `joblib.dump({...}, "prepared_data.joblib")` containing:
   - `X_train, X_val, X_test, y_train, y_val, y_test` (exact dtypes preserved)
   - `feature_names` (ordered list)
   - `fingerprint` = {row counts, ordered feature names, split-index hashes, target hash}
   - `preprocessor` (the fitted ColumnTransformer, for reference)
   - `target_transform` flag (e.g. "log1p" or "none")
   (Optional: also dump the six split CSVs for eyeballing — the joblib bundle is the source of truth.)
5. Scaling is **not** baked in here (it is model-specific); the saved X's are the common,
   encoded, leakage-free feature matrices. After saving, **STOP.**

### PHASE 6 - Baseline & Multiple Models (one notebook each)
Notebooks: `linear_regression.ipynb`, `ridge_regression.ipynb`, `lasso_regression.ipynb`,
`elastic_net.ipynb`, `decision_tree.ipynb`, `random_forest.ipynb`, `gradient_boosting.ipynb`,
and if practical `xgboost_model.ipynb`, `lightgbm_model.ipynb`, `catboost_model.ipynb`.
Regularized linear models use default alpha here.

**Each notebook does NO data cleaning.** It: loads the Phase-5 prepared splits -> **verifies the
fingerprint** (stop on mismatch) -> applies only a model-appropriate scaler *inside a Pipeline,
fit on X_train* (StandardScaler/MinMax/Robust for linear/KNN; none for trees) -> trains ->
validation + CV RMSE -> (test only if `EVALUATE_TEST`) -> metrics -> save predictions -> plots
-> summary. Only the model differs between notebooks. **STOP** after the baseline run.

### PHASE 7 - Regularization Experiments  -> `phase7_regularization.ipynb`
Ridge/Lasso/ElasticNet `alpha` (+ `l1_ratio`) via CV on **train only** (scaler inside the
Pipeline, re-fit per fold). Report under/overfitting, coefficient shrinkage, # coefs -> 0,
val RMSE/MAE/R². Decide from results. **STOP.**

### PHASE 8 - Hyperparameter Optimization  -> `phase8_tuning.ipynb`
Tune **only the strongest candidates** (GridSearchCV / RandomizedSearchCV / Optuna). Same CV
methodology; transforms re-fit inside folds. Test untouched. **STOP.**

### PHASE 9 - Final Evaluation & Comparison  -> `phase9_final_eval.ipynb`
Set `EVALUATE_TEST = True`. Table `| Model | MAE | MSE | RMSE | R² | CV RMSE |` (rupees). Also
train/val/test RMSE, generalization gap, training time. Plots: Actual vs Predicted; Residuals
vs Predicted; RMSE/MAE across models; error by rent range (low/med/high). **STOP.**

### PHASE 10 - Final Results  -> `phase10_conclusions.ipynb`
Pick the strongest model on combined evidence (RMSE, MAE, R², CV, generalization, residuals,
compute cost, interpretability). Explain why models differed. **STOP.**

---

## 7. NOTEBOOK STANDARD (every model `.ipynb`, Colab-ready)

Fixed cell order — only the model cell changes:
```
1.  [markdown] Title + what this notebook does
2.  [code] !pip install ...          # only if needed (e.g. xgboost)
3.  [code] Imports + RANDOM_STATE = 42 + EVALUATE_TEST = False
4.  [code] Load Phase-5 bundle:
            d = joblib.load("prepared_data.joblib")
            X_train, X_val, X_test = d["X_train"], d["X_val"], d["X_test"]
            y_train, y_val, y_test = d["y_train"], d["y_val"], d["y_test"]
5.  [code] FINGERPRINT VERIFY vs d["fingerprint"] (row counts, feature names, hashes) -> stop on mismatch
6.  [code] Build Pipeline([scaler(if applicable), model])   # scaler fit on X_train only
7.  [code] Train
8.  [code] Validation metrics + CV RMSE (CV on train; scaler re-fit per fold)
9.  [code] Final test evaluation  -> runs ONLY if EVALUATE_TEST == True
10. [code] Metrics (rupees; expm1 if log1p used)
11. [code] Save predictions: actual_rent, predicted_rent, residual
12. [code] Visualization
13. [markdown] Results summary
```
No notebook reads the raw CSV or cleans data — that is Phase 5's job. Printed summary:
```
========================================
MODEL: <name>
========================================
Dataset:  Rows: XXXX  Features: XX   Fingerprint: OK
Train RMSE: XX.XX   Validation RMSE: XX.XX   [Test RMSE: XX.XX]
Test MAE: XX.XX   Test MSE: XX.XX   Test R²: X.XXXX   CV RMSE: XX.XX
Training Time: XX.XX s
```

---

## 8. FINAL DELIVERABLES
1. Cleaning methodology (rules + transformers)  2. Preprocessing pipeline (inside `prepared_data.joblib`)
3. Feature-engineering methodology  4. Fixed 70/15/15 split + fingerprint + `prepared_data.joblib`
5. One standalone Colab `.ipynb` per model  6. Standardized evaluation results
7. Regularization comparison  8. Hyperparameter optimization results
9. Model comparison table  10. Prediction outputs  11. Visualizations
12. Final experimental conclusions

*(Repo organization, reusable modules, README, API/deployment come later.)*

---

### >> START WITH PHASE 1. Inspect the real dataset (and the glossary), look-only, before any
### modeling or preprocessing decision. Report findings, then STOP and wait.
