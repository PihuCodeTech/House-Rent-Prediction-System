# ACTION PLAN — superseded by the modular repo

The original phase-by-phase, Colab-oriented plan has been realised and then **restructured**
into a clean, standard ML repository. The completed phase1–phase10 notebooks and their outputs
are archived in `_legacy_phases/` for reference.

**Current state of the project:** see `README.md` and `CLAUDE.md`.

- Data preparation is frozen once in `src/` (imported by every model; identical by construction).
- One standalone notebook per model in `notebooks/`, each reading `House_Rent_Dataset.csv` and
  running the full `src` pipeline, then writing results to `reports/`.
- `main.py` runs the whole baseline experiment and builds `reports/model_comparison.csv`.

## Remaining stages (not yet done)
1. **Hyperparameter tuning** — on the strongest baselines (lowest CV RMSE), same CV methodology,
   test set untouched. Add a `notebooks/hp_tuning.ipynb` (and/or `src` helpers) when ready.
2. **Final evaluation** — set `EVALUATE_TEST=True` (or `python main.py --test`) to score the
   held-out test set once, on the selected candidates only.
3. **Fine-tune + finalize** the chosen model; then consider packaging/serving.

The fair-comparison guarantees (identical data, no leakage, fingerprint tripwire) carry into
every later stage because the `src/` pipeline and the fixed 70/15/15 split do not change.
