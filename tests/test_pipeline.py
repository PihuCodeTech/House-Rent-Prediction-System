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
    "current_floor": 3,
    "total_floors": 5,
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
    assert row.iloc[0]["Floor"] == "3 out of 5"


def test_lenient_formats_are_normalised(schema):
    row, _ = validate_listing(
        {**GOOD, "City": "  kolkata ", "BHK": "2", "Size": "1,200", "Furnishing Status": "furnished"}, schema
    )
    r = row.iloc[0]
    assert (r["City"], r["BHK"], r["Size"], r["Furnishing Status"]) == ("Kolkata", 2, 1200.0, "Furnished")


def test_optional_fields_get_defaults_with_warnings(schema):
    row, warns = validate_listing({"City": "Delhi", "BHK": 2, "Size": 1000, "Bathroom": 2}, schema)
    assert len(row) == 1 and len(warns) >= 6


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
        ({"current_floor": 9, "total_floors": 3}, "above the building"),
        ({"current_floor": -5}, "below -2"),
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
