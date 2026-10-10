"""End-to-end checks of the pipeline, the saved outputs and the prediction layer the app uses.

    python -m pytest -q tests            (run_all.py runs this automatically after the notebooks)

Data, split and leakage checks always run (they need only the dataset). Checks of saved models, the input schema
and reports/results.csv run once the notebooks have produced them, and are skipped before that.
"""

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src import config  # noqa: E402
from src.data_cleaning import MAX_RENT_PER_SQFT, apply_stateless_rules  # noqa: E402
from src.data_loading import load_raw  # noqa: E402
from src.evaluation import RESULT_COLUMNS  # noqa: E402
from src.models import REGISTRY, display_name  # noqa: E402
from src.prediction import (  # noqa: E402
    RAW_COLUMNS,
    InvalidListing,
    build_input_schema,
    load_final,
    load_model,
    model_path,
    predict_listing,
    validate_listing,
)
from src.preprocessing import build_preprocessor, prepare_data  # noqa: E402

pytestmark = pytest.mark.skipif(not config.DATA_RAW.exists(), reason="dataset not found (see data/README.md)")
HAS_FINAL = (config.MODELS_DIR / "final_model.joblib").exists()


@pytest.fixture(scope="module")
def raw():
    return load_raw()


@pytest.fixture(scope="module")
def clean(raw):
    return apply_stateless_rules(raw)[0]


@pytest.fixture(scope="module")
def datas(clean):
    return {s: prepare_data(seed=s, clean=clean) for s in config.SEEDS}


@pytest.fixture(scope="module")
def schema(raw):
    return build_input_schema(raw)


@pytest.fixture(scope="module")
def sample_rows(raw):
    return raw.sample(25, random_state=0)[RAW_COLUMNS].reset_index(drop=True)


GOOD = {
    "City": "Kolkata",
    "Area Locality": "Salt Lake City Sector 2",
    "BHK": 2,
    "Size": 900,
    "Bathroom": 2,
    "Furnishing Status": "Semi-Furnished",
    "Area Type": "Super Area",
    "Tenant Preferred": "Bachelors/Family",
    "Point of Contact": "Contact Owner",
}


# ----------------------------- data, splits, leakage -----------------------------
def test_cleaning(clean):
    assert len(clean) == 4731
    assert (clean["Rent"] / clean["Size"]).max() <= MAX_RENT_PER_SQFT
    assert not clean.drop(columns=["Posted On"]).duplicated().any()


def test_seed42_matches_recorded_reference(datas):
    assert datas[42]["fingerprint"]["reference"]["reference_id"] == config.EXPECTED_REFERENCE_ID


def test_splits_disjoint_and_complete(datas, clean):
    for seed, d in datas.items():
        train, val, test = set(d["X_train_raw"].index), set(d["X_val"].index), set(d["X_test"].index)
        assert not (train & val or train & test or val & test), f"seed {seed}: overlapping splits"
        dropped = d["meta"]["far_out_filter"]["n_dropped"]
        assert len(train) + len(val) + len(test) + dropped == len(clean), f"seed {seed}: rows lost"


def test_seeds_give_different_splits(datas):
    tests = [set(d["X_test"].index) for d in datas.values()]
    assert all(a != b for i, a in enumerate(tests) for b in tests[i + 1 :])


def test_prepare_data_is_deterministic(clean):
    assert prepare_data(seed=43, clean=clean)["fingerprint"] == prepare_data(seed=43, clean=clean)["fingerprint"]


def test_no_target_or_raw_columns_in_features(datas):
    for d in datas.values():
        assert not [c for c in d["feature_names"] if re.fullmatch(r"(?i)rent|posted_on|floor|locality_key", c)]


def test_preprocessor_depends_on_training_rows_only(datas):
    """Re-fitting on the training rows alone reproduces the fitted preprocessor exactly."""
    d = datas[42]
    refit = build_preprocessor(seed=42).fit(d["X_train_raw"], np.log1p(d["y_train"]))
    sample = d["X_train_raw"].head(50)
    np.testing.assert_allclose(d["preprocessor"].transform(sample), refit.transform(sample))


# ----------------------------- input validation (the app's contract) -----------------------------
def test_valid_listing(schema):
    row, warns = validate_listing(GOOD, schema)
    assert list(row.columns) == RAW_COLUMNS and len(row) == 1 and warns == []


def test_floor_fields_are_ignored(schema):
    """Floors are not model inputs any more: old callers may still send them, without effect or error."""
    row, warns = validate_listing({**GOOD, "current_floor": 9, "total_floors": 3}, schema)
    plain, _ = validate_listing(GOOD, schema)
    assert warns == [] and row.equals(plain)


def test_lenient_formats_are_normalised(schema):
    row, _ = validate_listing(
        {**GOOD, "City": "  kolkata ", "BHK": "2", "Size": "1,200", "Furnishing Status": "furnished"}, schema
    )
    r = row.iloc[0]
    assert (r["City"], r["BHK"], r["Size"], r["Furnishing Status"]) == ("Kolkata", 2, 1200.0, "Furnished")


def test_optional_fields_get_defaults_with_warnings(schema):
    row, warns = validate_listing({"City": "Delhi", "BHK": 2, "Size": 1000, "Bathroom": 2}, schema)
    assert len(row) == 1 and len(warns) >= 4  # four listing details defaulted + no locality


@pytest.mark.parametrize(
    "change, message",
    [
        ({"City": "Pune"}, "City must be one of"),
        ({"City": None}, "City is required"),
        ({"BHK": 0}, "BHK must be at least 1"),
        ({"BHK": 2.5}, "whole number"),
        ({"Size": -10}, "greater than 0"),
        ({"Size": "abc"}, "must be a number"),
        ({"Size": float("nan")}, "finite"),
        ({"Bathroom": True}, "must be a number"),
        ({"Furnishing Status": "Luxury"}, "Furnishing Status must be one of"),
    ],
)
def test_invalid_listings_are_explained(schema, change, message):
    with pytest.raises(InvalidListing) as problem:
        validate_listing({**GOOD, **change}, schema)
    assert any(message in e for e in problem.value.errors)


def test_non_dict_listing(schema):
    with pytest.raises(InvalidListing):
        validate_listing(["not", "a", "dict"], schema)


def test_unusual_values_warn_but_still_validate(schema):
    _, warns = validate_listing({**GOOD, "Size": 50000, "Area Locality": "Nowhere Town"}, schema)
    assert any("outside the training data" in w for w in warns) and any("not in the training data" in w for w in warns)


# ----------------------------- saved outputs -----------------------------
SAVED = [k for k in REGISTRY if model_path(k).exists()]


@pytest.mark.parametrize("key", SAVED or [pytest.param(None, marks=pytest.mark.skip(reason="no saved models yet"))])
def test_saved_averaged_model(key, sample_rows):
    model = load_model(key)
    assert model.key == key and model.seeds == list(config.SEEDS)
    each, avg = model.predict_each(sample_rows), model.predict(sample_rows)
    assert each.shape == (len(config.SEEDS), len(sample_rows))
    assert np.all(np.isfinite(avg)) and np.all(avg > 0)
    np.testing.assert_allclose(avg, each.mean(axis=0))


@pytest.mark.skipif(not HAS_FINAL, reason="final model not selected yet")
def test_predict_listing_end_to_end(schema):
    final = load_final()
    result = predict_listing(GOOD, final, schema)
    assert 1_000 < result["rent"] < 1_000_000 and result["low"] <= result["rent"] <= result["high"]
    assert set(result["per_seed"]) == set(config.SEEDS)
    extreme = predict_listing({**GOOD, "BHK": 6, "Size": 8000, "Bathroom": 6}, final, schema)
    assert np.isfinite(extreme["rent"]) and extreme["rent"] > result["rent"]


def test_results_csv():
    if not config.RESULTS_CSV.exists():
        pytest.skip("reports/results.csv not produced yet")
    table = pd.read_csv(config.RESULTS_CSV, encoding="utf-8-sig", dtype=str)
    assert list(table.columns) == RESULT_COLUMNS
    assert set(table["Model"]) <= {display_name(k) for k in REGISTRY}
    for col in RESULT_COLUMNS[1:]:
        assert table[col].str.fullmatch(r"-?\d+\.\d{2} ± \d+\.\d{2}").all(), col


def test_error_profile_handles_constant_predictions():
    """A constant model (the baselines) puts every prediction in one price band; the range must still work."""
    from src.prediction import build_error_profile, likely_range

    rng = np.random.default_rng(0)
    tests = [(rng.lognormal(10, 1, 700), np.full(700, 20_000.0)) for _ in range(3)]
    low, high = likely_range(20_000, build_error_profile(tests))
    assert 0 < low < 20_000 < high


def test_what_if_stays_inside_the_trained_ranges():
    from src.prediction import load_final, load_input_schema, what_if

    if not (config.MODELS_DIR / "final_model.joblib").exists():
        pytest.skip("final model not built yet")
    schema = load_input_schema()
    rows = what_if({"City": "Mumbai", "BHK": 6, "Size": 8000, "Bathroom": 10}, load_final(), schema)
    labels = " ".join(r["label"] for r in rows)
    assert "7 BHK" not in labels and "11 bathrooms" not in labels and "9,600 sq ft" not in labels
    assert "5 BHK" in labels and "9 bathrooms" in labels


def test_range_band_explains_the_likely_range():
    from src.prediction import likely_range, range_band

    profile = {
        "edges": [10_000, 20_000],
        "ratio_low": [0.5, 0.6, 0.7],
        "ratio_high": [1.5, 1.6, 1.7],
        "coverage": 0.8,
        "n_test_predictions": 300,
    }
    band = range_band(15_000, profile)
    assert band == {"low": 10_000, "high": 20_000, "n": 100, "ratio_low": 0.6, "ratio_high": 1.6}
    low, high = likely_range(15_000, profile)
    assert (low, high) == (15_000 * band["ratio_low"], 15_000 * band["ratio_high"])
    assert range_band(5_000, profile)["low"] is None and range_band(50_000, profile)["high"] is None
    assert range_band(1, None) is None


def test_final_model_error_reports_are_aggregates_only():
    """reports/final_model_*.csv must never contain listing rows (the dataset can't be redistributed)."""
    import pandas as pd

    for path in (config.FINAL_ERRORS_CSV, config.FINAL_RESIDUALS_CSV):
        if not path.exists():
            pytest.skip("final diagnostics not generated yet")
        table = pd.read_csv(path, encoding="utf-8-sig")
        assert len(table) <= 50
        assert not {"Rent", "Area Locality", "actual", "predicted"} & set(table.columns)
    errors = pd.read_csv(config.FINAL_ERRORS_CSV, encoding="utf-8-sig")
    overall = errors[errors["Group"] == "Overall"].iloc[0]
    assert errors[errors["Group"] == "City"]["Test predictions"].sum() == overall["Test predictions"]
    assert errors[errors["Group"] == "Rent band"]["Test predictions"].sum() == overall["Test predictions"]


# ----------------------------- the locality × size baseline and the median % error -----------------------------
def test_locality_rate_baseline_uses_training_rows_and_falls_back():
    from src.models import LocalityRateBaseline

    X = pd.DataFrame(
        {
            "City": ["A"] * 4 + ["B"],
            "locality_key": ["A | x"] * 3 + ["A | y", "B | z"],
            "Size": [1000.0, 1000.0, 500.0, 1000.0, 1000.0],
        }
    )
    y = [10_000, 12_000, 7_000, 40_000, 9_000]  # rates 10, 12, 14 (x) · 40 (y, one listing) · 9 (z)
    m = LocalityRateBaseline(min_listings=3).fit(X, y)
    new = pd.DataFrame(
        {"City": ["A", "A", "B", "C"], "locality_key": ["A | x", "A | y", "B | new", "C | q"], "Size": [100.0] * 4}
    )
    np.testing.assert_allclose(m.predict(new), [1200, 1300, 900, 1200])  # x median 12; y → city A median 13;
    # unknown locality → city B 9; unknown city → overall median 12


def test_baseline_rates_come_from_training_rows_only(datas):
    from src.models import LocalityRateBaseline

    d = datas[42]
    m = LocalityRateBaseline().fit(d["X_train_raw"], d["y_train"])
    rate = d["y_train"] / d["X_train_raw"]["Size"]
    expected = rate.groupby(d["X_train_raw"]["City"]).median()
    assert m.city_rate_ == pytest.approx(expected.to_dict())  # nothing from the validation or test rows


def test_median_percent_error():
    from src.evaluation import median_pct_error

    assert median_pct_error([100, 200, 400], [110, 150, 400]) == pytest.approx(10.0)  # errors 10%, 25%, 0%


def test_floor_features_are_gone(datas):
    names = datas[42]["feature_names"]
    assert len(names) == 29 and not [n for n in names if "floor" in n.lower() or "basement" in n.lower()]
