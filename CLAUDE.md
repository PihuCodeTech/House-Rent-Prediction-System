# CLAUDE.md — Project Rules (read every turn)

Supervised **regression** experiment predicting house rent. Full playbook: `ACTION_PLAN.md`.

## Dataset
- `House_Rent_Dataset.csv` (~4,746 rows, 12 cols). Target: `Rent` (continuous).
- Cols: Posted On, BHK, Rent, Size, Floor ("Ground out of 2"), Area Type, Area Locality
  (high-cardinality), City, Furnishing Status, Tenant Preferred, Bathroom, Point of Contact.
- `Dataset Glossary.txt` defines columns — use it in Phase 1.

## STANDARD ML WORKFLOW — pipelines, not hardcoding (most important)
- **No manual/eyeballed data surgery.** Never drop specific row indices
  (`df.drop([13, 4213])`), never per-row hand edits, never a chart-read threshold typed in
  without deriving it in code. Cleaning is either a stateless RULE or a fitted TRANSFORMER.
- **Stateless rules** (learn nothing from the distribution) are expressed as conditions and
  may run pre-split: `df[df["Rent"]>0]`, `drop_duplicates()`, parsing Floor/dates.
- **Learned transforms** (median impute, encoders, scalers, feature selection, the log1p
  decision) are assembled in sklearn `Pipeline`+`ColumnTransformer` and `fit` on TRAIN ONLY,
  then applied to val/test. `.fit()` must never see val/test.
- **EDA is look-only.** Phase 1 modifies/saves nothing and drops no rows; it outputs decisions.
- **Never clean/drop validation or test rows** on distribution. Outliers: keep + robust
  methods, OR an IQR cutoff computed on train and applied to train only. Val/test stay intact.
- **Prepare once, consume everywhere.** Phase 5 builds + fits the pipeline once on train,
  transforms all splits, and SAVES them. Model notebooks LOAD the saved splits and do NO
  cleaning themselves — this is what guarantees identical data and kills per-notebook drift.

## Output format
- One standalone Jupyter notebook (`.ipynb`) per model, runs top-to-bottom in Colab.
- Phase 5 saves ONE bundle `prepared_data.joblib` (dtype-exact: X/y splits, feature_names,
  fingerprint, fitted preprocessor, target_transform flag). Model notebooks load it via
  `joblib.load("prepared_data.joblib")`, uploaded into Colab cwd. No files.upload()/Drive.

## Split — 70/15/15 (fixed, done once in Phase 5)
- Train 70% / Val 15% / Test 15%, `random_state=42`, two-step (0.15 test, then 0.1765 of the
  remainder = val). Reused identically everywhere. Stratify on bins if Phase 3 says so; bins
  are never features.

## Zero leakage (train -> val -> test)
Split before anything fitted; fit on train only; CV re-fits transforms per fold; val is for
comparison not fitting; test sacred until Phase 9 (`EVALUATE_TEST=False` until then); no
target-derived features.

## Core rules
1. Identical data for every model; mismatch vs Phase-5 fingerprint (rows, feature names,
   split indices, target hash) -> STOP and report.
2. Phase-gated (./phase1..../phase10, relative). Report + STOP each phase; never auto-proceed.
3. Evidence-driven; no preprocessing decision before Phase 1.
4. Freeze at Phase 5. 5. Regression -> no SMOTE. 6. No repo structure yet.

## Metrics & transform
- log1p Rent -> back-transform with `np.expm1`; report ALL metrics (incl. CV RMSE) in rupees.
- CV on 70% train fills "CV RMSE" for every model; 15% val is the model-selection check.

## Each model notebook
- Loads `prepared_data.joblib` -> verifies fingerprint BEFORE training (stop on mismatch) ->
  applies only a model-appropriate scaler inside a Pipeline (fit on X_train; none for trees).
  Linear/Ridge/Lasso/ElasticNet default alpha in P6; tuned in P7. Does NO data cleaning.
- `!pip install` only when needed; standardized summary (Train/Val in P6-8, +Test in P9;
  MAE, MSE, R², CV RMSE, training time).
