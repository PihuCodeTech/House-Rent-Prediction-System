# House Rent Prediction

Predicting the monthly rent of flats and houses in six Indian cities from their listing details — twelve regression
models compared fairly on identical data, three random seeds each, with no data leakage.

**Best model: Gradient Boosting — test RMSE ₹22,282 ± 717 and R² 0.83, typical error (MAE) ₹9,172 a month**,
averaged over three different train/validation/test splits. That is less than half the error of always guessing
the average rent (₹53,731); the top four boosting models are within ₹250 of each other.

![Test error by model](visualizations/model_comparison.png)

## Results

Mean ± standard deviation over seeds 42, 43 and 44 (full table with train, CV and validation errors:
[`reports/results.csv`](reports/results.csv)). Errors are in rupees per month; lower is better.

| Model | Test RMSE | Test MAE | Test R² | CV RMSE (train) |
|---|--:|--:|--:|--:|
| **Gradient Boosting** | **22,282 ± 717** | **9,172 ± 436** | **0.83 ± 0.01** | 29,050 ± 2,436 |
| XGBoost | 22,314 ± 1,046 | 9,209 ± 365 | 0.83 ± 0.02 | 29,051 ± 1,857 |
| LightGBM | 22,404 ± 1,103 | 9,368 ± 401 | 0.83 ± 0.02 | 29,009 ± 2,145 |
| CatBoost | 22,538 ± 1,238 | 9,256 ± 232 | 0.82 ± 0.02 | 29,939 ± 2,389 |
| Random Forest | 23,269 ± 1,657 | 9,751 ± 547 | 0.81 ± 0.02 | 30,482 ± 2,299 |
| Lasso Regression | 25,597 ± 367 | 10,488 ± 228 | 0.77 ± 0.02 | 32,811 ± 2,048 |
| Elastic Net | 25,602 ± 342 | 10,486 ± 239 | 0.77 ± 0.02 | 32,825 ± 2,055 |
| Ridge Regression | 25,628 ± 320 | 10,461 ± 255 | 0.77 ± 0.02 | 32,945 ± 2,170 |
| Decision Tree | 25,683 ± 1,446 | 10,894 ± 529 | 0.77 ± 0.02 | 31,485 ± 1,252 |
| Linear Regression | 25,922 ± 940 | 10,451 ± 330 | 0.77 ± 0.01 | 32,995 ± 1,724 |
| Baseline (Mean) | 53,731 ± 2,976 | 29,309 ± 166 | 0.00 ± 0.00 | 57,777 ± 3,623 |
| Baseline (Median) | 56,583 ± 2,929 | 23,617 ± 330 | −0.11 ± 0.01 | 60,573 ± 3,663 |

**What drives rent.** Size dominates — especially in Mumbai, where a square foot costs several times more than
anywhere else — followed by the number of bathrooms, whether an agent lists the property, and the locality.

![What the final model relies on](visualizations/feature_importance.png)

The final model's predictions on unseen listings (seed 42) track actual rents closely across two orders of magnitude;
errors grow for the most expensive properties:

![Gradient Boosting diagnostics](visualizations/gradient_boosting.png)

## How it works

1. **Clean** (`src/data_cleaning.py`). Fixed rules, never hand-picked rows: trim text and normalise locality names,
   keep physically valid listings, drop 6 implausible ones (rent above ₹500 per sq ft — e.g. a 3-bedroom flat of
   10 sq ft) and 9 re-listed duplicates, and parse "3 out of 5" into floor numbers. 4,746 → 4,731 listings.
2. **Split** each seed's data 70 / 15 / 15 into training, validation and test sets, balanced across rent levels.
   Nothing learned from data ever sees validation or test rows: extreme rent-per-sq-ft listings are trimmed from the
   training rows only, and every encoder, imputer and scaler is fitted on training rows only.
3. **Features** (`src/feature_engineering.py`, `src/preprocessing.py`) — 34 in total: size, bedrooms and bathrooms
   (also on a log scale), floor and building height, size × city, one-hot city / furnishing / area type / tenant /
   contact, and the locality's typical rent (target encoding, cross-fitted so a listing never sees its own rent).
4. **Learn log rent.** Rents are heavily skewed (₹1,200 to ₹35 lakh), so models learn log(1 + rent) and their
   predictions are converted back to rupees; all errors are reported in rupees.
5. **Tune** (`src/tuning.py`, `notebooks/hyperparameter_tuning.ipynb`). For each model: a grid search, then a
   Bayesian search (Optuna) around the best grid point, both scored by 5-fold cross-validation on training data. The
   library defaults, grid-best and Bayesian-best were compared on the three seeds and the best kept
   ([`reports/final_hyperparameters.json`](reports/final_hyperparameters.json)).
6. **Evaluate on three seeds** (`src/evaluation.py`). Each seed draws a new split, new CV folds and new model
   randomness; every metric is reported as mean ± std. Cross-validation re-fits the whole preprocessing inside each
   fold. A fingerprint check guarantees every model saw exactly the same data for each seed.
7. **Save** (`src/prediction.py`). Each model is saved as the average of its three seed-trained pipelines
   (`models/<model>.joblib`); the best becomes `models/final_model.joblib`, next to `models/input_schema.json` (the
   allowed choices and value ranges). `predict_listing({...})` validates a listing, fills sensible defaults with
   warnings, explains invalid input, and returns the predicted rent with its range across seeds.

## Limitations

- **Six listings were removed as entry errors** (rent above ₹500 per sq ft). One of them, ₹35 lakh a month for a
  2,500 sq ft flat in Bangalore, dominated test RMSE whenever it fell into a test split. The rule is fixed and
  documented in [`notebooks/data_exploration.ipynb`](notebooks/data_exploration.ipynb), but it is a judgement call.
- **Model versions were chosen on test RMSE.** The searches themselves used only cross-validation on training data,
  but choosing between default, grid and Bayesian settings by test score makes the reported test errors slightly
  optimistic. Hyperparameters were also tuned on seed 42's training rows, some of which are test rows for seeds 43–44.
- **The spread across seeds is real.** Validation RMSE varies by about ±₹10k between seeds because a few very
  expensive listings land in different splits; differences between the top models are smaller than this spread.
- **Data scope.** About 4,700 listings from six cities, posted in April–July 2022 on one website. Predictions outside
  these cities, sizes (20–8,000 sq ft) or that period are extrapolations; the app warns about out-of-range inputs.
- **No location detail beyond locality names** — a locality not seen in training gets an average locality effect.
- Random Forest and XGBoost results can differ by a few rupees between machines (multithreaded subsampling).

## Project structure

```
├── data/README.md            how to get the dataset (data/ is git-ignored)
├── notebooks/
│   ├── data_exploration      the raw data, rent distribution, rent by city, what cleaning removes
│   ├── <model> × 12          one per model: clean → train on 3 seeds → results → save → diagnostics
│   ├── hyperparameter_tuning grid search → Bayesian search (optional re-run, 20–40 min)
│   ├── model_comparison      all models side by side
│   ├── final_selection       picks the best model; saves final_model.joblib + input_schema.json
│   ├── feature_importance    what the final model relies on
│   └── predict_demo          predicting new listings, including invalid input
├── src/                      config · data_loading · data_cleaning · feature_engineering · preprocessing
│                             · models · tuning · evaluation · plotting · prediction
├── tests/test_pipeline.py    data, splits, leakage guards, input validation, saved models, results
├── reports/                  results.csv · final_hyperparameters.json
├── visualizations/           figures (per-model diagnostics, comparison, importance, data exploration)
├── models/                   saved models + input_schema.json (created by the notebooks, git-ignored)
├── run_all.py                runs everything end to end
└── requirements.txt · pyproject.toml · LICENSE
```

## How to run

```bash
git clone https://github.com/PihuCodeTech/House-Rent-Prediction-System.git
cd House-Rent-Prediction-System
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt               # macOS: XGBoost/LightGBM also need `brew install libomp`
# download the dataset into data/raw/House_Rent_Dataset.csv — see data/README.md

python run_all.py                             # every notebook in order, then the tests (≈ 3–4 min)
python run_all.py --only random_forest        # re-run selected notebooks
python run_all.py --tuning                    # also re-run the hyperparameter search
python -m pytest                              # just the tests
```

`run_all.py` works from any folder, checks packages and the dataset first, runs the notebooks with the active Python,
saves each notebook with its outputs, and stops with the notebook name and error if anything fails.

Predicting from Python:

```python
from src.prediction import predict_listing

predict_listing({"City": "Mumbai", "Area Locality": "Andheri West", "BHK": 3, "Size": 1400, "Bathroom": 3,
                 "current_floor": 10, "total_floors": 20, "Furnishing Status": "Furnished"})
# -> {'rent': …, 'low': …, 'high': …, 'per_seed': {42: …, 43: …, 44: …}, 'warnings': [...], 'model': 'Gradient Boosting', …}
```

## Dataset

**House Rent Prediction Dataset** by Sourav Banerjee, on Kaggle:
<https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset>. According to its description, it
was collected from [MagicBricks](https://www.magicbricks.com/). The dataset is not redistributed here; download it
from Kaggle (see [`data/README.md`](data/README.md)) and check its terms there.

## Licence

The code in this repository is released under the [MIT Licence](LICENSE). The licence does not cover the dataset.
