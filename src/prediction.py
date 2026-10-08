"""Saved models and predictions for NEW listings — the layer the app (phase 2) calls.

Saved models
    Every model notebook saves a SeedAveragedModel to models/<key>.joblib: the model trained on each seed's
    training split, each with the preprocessing fitted on that split. A new listing's prediction is the mean of
    the per-seed predictions. final_selection.ipynb copies the chosen one to models/final_model.joblib and writes
    models/input_schema.json (the allowed choices and value ranges the app offers).

    Do NOT score a SeedAveragedModel on any seed's validation/test rows: every dataset row is in another seed's
    training data, so such scores would be leaked. The honest scores are the per-seed ones in reports/results.csv.

Validated prediction
    predict_listing({...}) checks and normalises one listing, explains problems in plain words
    (InvalidListing), warns when a value is outside what the model has seen, and returns the averaged
    prediction together with the per-seed spread.
"""

import json
import math
import re
import shutil

import joblib
import numpy as np
import pandas as pd

from . import config
from .data_cleaning import FLOOR_LEVEL_MAP, rule_parse_floor, rule_strip_text
from .feature_engineering import add_stateless_features

RAW_COLUMNS = [
    "Posted On",
    "BHK",
    "Size",
    "Floor",
    "Area Type",
    "Area Locality",
    "City",
    "Furnishing Status",
    "Tenant Preferred",
    "Bathroom",
    "Point of Contact",
]
CHOICE_FIELDS = ["City", "Furnishing Status", "Area Type", "Tenant Preferred", "Point of Contact"]
FINAL_MODEL, FINAL_META, INPUT_SCHEMA = "final_model.joblib", "final_model_meta.json", "input_schema.json"
_PLACEHOLDER_DATE = "2022-06-01"  # the posting date is not a model feature; any valid date works


# ----------------------------- raw-row prediction -----------------------------
def predict_raw(model, preprocessor, feature_names, raw_df):
    """Predict rupees for rows in the ORIGINAL dataset schema (Rent not needed): inference-safe cleaning,
    the fitted preprocessor, then the model."""
    missing = [c for c in RAW_COLUMNS if c not in raw_df.columns and c != "Posted On"]
    if missing:
        raise ValueError(f"raw rows are missing columns: {missing}")
    clean = add_stateless_features(rule_parse_floor(rule_strip_text(raw_df.copy())))
    Z = preprocessor.transform(clean.drop(columns=[config.TARGET], errors="ignore"))
    X = pd.DataFrame(np.asarray(Z, dtype="float64"), columns=feature_names, index=clean.index)
    return model.predict(X)


class SeedAveragedModel:
    """One model trained on several seeds; predicts the mean of the per-seed predictions (rupees).

    members: list of {"seed", "model", "preprocessor", "feature_names"}; each model was fitted on its seed's
    training split together with that seed's fitted preprocessor.
    """

    def __init__(self, key, name, members, params=None, source=None, results=None):
        self.key, self.name, self.members = key, name, list(members)
        self.params, self.source, self.results = dict(params or {}), source, results
        self.seeds = [m["seed"] for m in self.members]

    def predict_each(self, raw_df):
        """Per-seed predictions, shape (n_seeds, n_rows)."""
        return np.vstack([predict_raw(m["model"], m["preprocessor"], m["feature_names"], raw_df) for m in self.members])

    def predict(self, raw_df):
        """Mean of the per-seed predictions, one value per row."""
        return self.predict_each(raw_df).mean(axis=0)

    def member(self, seed):
        """The single-seed member (model, preprocessor, feature names) for `seed`."""
        return next(m for m in self.members if m["seed"] == seed)

    def __repr__(self):
        return f"SeedAveragedModel({self.name}, seeds={self.seeds})"


# ----------------------------- save / load -----------------------------
def model_path(key):
    """models/<key>.joblib"""
    return config.MODELS_DIR / f"{key}.joblib"


def save_model(averaged):
    """Save a SeedAveragedModel to models/<key>.joblib."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = model_path(averaged.key)
    joblib.dump(averaged, path, compress=3)
    return path


def load_model(key):
    """Load models/<key>.joblib (saved by notebooks/<key>.ipynb)."""
    path = model_path(key)
    if not path.exists():
        raise FileNotFoundError(f"models/{path.name} not found — run notebooks/{key}.ipynb (or run_all.py) first.")
    return joblib.load(path)


def save_final(key, meta=None):
    """Copy models/<key>.joblib to models/final_model.joblib (+ final_model_meta.json)."""
    src = model_path(key)
    if not src.exists():
        raise FileNotFoundError(f"models/{src.name} not found — run notebooks/{key}.ipynb first.")
    shutil.copyfile(src, config.MODELS_DIR / FINAL_MODEL)
    if meta is not None:
        (config.MODELS_DIR / FINAL_META).write_text(json.dumps(meta, indent=2, default=str))
    return config.MODELS_DIR / FINAL_MODEL


def load_final():
    """Load models/final_model.joblib (saved by notebooks/final_selection.ipynb)."""
    path = config.MODELS_DIR / FINAL_MODEL
    if not path.exists():
        raise FileNotFoundError("models/final_model.joblib not found — run notebooks/final_selection.ipynb first.")
    return joblib.load(path)


def load_final_meta():
    """The final model's metadata (name, hyperparameters, results), or {} if not saved yet."""
    path = config.MODELS_DIR / FINAL_META
    return json.loads(path.read_text()) if path.exists() else {}


# ----------------------------- input schema (choices & ranges for the app) -----------------------------
def _normalise_locality(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def build_input_schema(raw_df):
    """Allowed choices, locality suggestions per city and value ranges, from the cleaned dataset.
    Saved next to the final model (models/ is git-ignored, like the dataset itself)."""
    from .data_cleaning import apply_stateless_rules

    clean, _ = apply_stateless_rules(raw_df)
    raw = raw_df.assign(
        _key=raw_df["Area Locality"].map(_normalise_locality),
        City=raw_df["City"].astype(str).str.strip(),
        _name=raw_df["Area Locality"].astype(str).str.strip(),
    )
    grouped = raw.groupby(["City", "_key"])["_name"]
    display = (
        pd.DataFrame(
            {  # the most frequent original spelling of each locality; most common localities first
                "name": grouped.agg(lambda s: s.value_counts().index[0]),
                "n": grouped.size(),
            }
        )
        .reset_index()
        .sort_values(["City", "n", "name"], ascending=[True, False, True])
    )
    localities = {city: g["name"].tolist() for city, g in display.groupby("City")}

    def rng(series, integer=False):
        s = series.dropna()
        cast = int if integer else float
        return {"min": cast(s.min()), "max": cast(s.max()), "median": cast(s.median()), "integer": integer}

    choices = {f: clean[f].value_counts().index.tolist() for f in CHOICE_FIELDS}  # most common first
    return {
        "version": 1,
        "choices": choices,
        "localities": localities,
        "numeric": {
            "BHK": rng(clean["BHK"], True),
            "Bathroom": rng(clean["Bathroom"], True),
            "Size": rng(clean["Size"]),
            "current_floor": rng(clean["current_floor"], True),
            "total_floors": rng(clean["total_floors"], True),
        },
        "defaults": {f: choices[f][0] for f in CHOICE_FIELDS},
        "floor_levels": {str(v): k for k, v in FLOOR_LEVEL_MAP.items()},
        "rows_used": int(len(clean)),
    }


def save_input_schema(schema):
    """Write models/input_schema.json."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MODELS_DIR / INPUT_SCHEMA
    path.write_text(json.dumps(schema, indent=1, ensure_ascii=False))
    return path


def load_input_schema():
    """Read models/input_schema.json (written by notebooks/final_selection.ipynb)."""
    path = config.MODELS_DIR / INPUT_SCHEMA
    if not path.exists():
        raise FileNotFoundError("models/input_schema.json not found — run notebooks/final_selection.ipynb first.")
    return json.loads(path.read_text())


# ----------------------------- validation -----------------------------
class InvalidListing(ValueError):
    """Raised when a listing cannot be scored; `.errors` lists every problem in plain words."""

    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def _number(value, name, errors, integer=False, required=True):
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            errors.append(f"{name} is required.")
        return None
    if isinstance(value, bool):
        errors.append(f"{name} must be a number, not {value!r}.")
        return None
    try:
        number = float(str(value).replace(",", "").strip()) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        errors.append(f"{name} must be a number (got {value!r}).")
        return None
    if not math.isfinite(number):
        errors.append(f"{name} must be a finite number.")
        return None
    if integer and number != int(number):
        errors.append(f"{name} must be a whole number (got {value!r}).")
        return None
    return int(number) if integer else number


def _choice(value, name, options, default, errors, warnings):
    if value is None or (isinstance(value, str) and not value.strip()):
        warnings.append(f"{name} not given — assumed '{default}' (the most common value).")
        return default
    match = {o.lower(): o for o in options}.get(str(value).strip().lower())
    if match is None:
        errors.append(f"{name} must be one of: {', '.join(options)} (got {value!r}).")
    return match


def floor_label(level):
    """0 -> 'Ground', -1 -> 'Upper Basement', -2 -> 'Lower Basement', 3 -> '3'."""
    return {v: k for k, v in FLOOR_LEVEL_MAP.items()}.get(level, str(level))


def validate_listing(listing, schema):
    """Check and normalise one listing (a dict). Returns (one-row DataFrame in the raw schema, warnings).
    Raises InvalidListing listing every problem.

    Fields: City, BHK, Size (sq ft), Bathroom (required); Area Locality, current_floor, total_floors,
    Furnishing Status, Area Type, Tenant Preferred, Point of Contact (optional — sensible defaults with a warning).
    Case and surrounding spaces of choices are ignored; numbers may be strings such as "1,200".
    """
    if not isinstance(listing, dict):
        raise InvalidListing([f"listing must be a dict of fields (got {type(listing).__name__})."])
    errors, warnings = [], []
    num, choices = schema["numeric"], schema["choices"]

    values = {}
    for f in CHOICE_FIELDS:
        given = listing.get(f)
        if f == "City" and (given is None or not str(given).strip()):
            errors.append("City is required.")
            values[f] = None
        else:
            values[f] = _choice(given, f, choices[f], schema["defaults"][f], errors, warnings)

    bhk = _number(listing.get("BHK"), "BHK", errors, integer=True)
    bath = _number(listing.get("Bathroom"), "Bathroom", errors, integer=True)
    size = _number(listing.get("Size"), "Size", errors)
    current = _number(listing.get("current_floor"), "current_floor", errors, integer=True, required=False)
    total = _number(listing.get("total_floors"), "total_floors", errors, integer=True, required=False)

    if bhk is not None and bhk < 1:
        errors.append("BHK must be at least 1.")
    if bath is not None and bath < 1:
        errors.append("Bathroom must be at least 1.")
    if size is not None and size <= 0:
        errors.append("Size must be greater than 0 sq ft.")
    if current is not None and current < min(FLOOR_LEVEL_MAP.values()):
        errors.append("current_floor cannot be below -2 (Lower Basement).")
    if total is not None and total < 0:
        errors.append("total_floors cannot be negative.")
    if current is not None and total is not None and current > total:
        errors.append(f"current_floor ({current}) is above the building's total_floors ({total}).")
    if errors:
        raise InvalidListing(errors)

    for name, value in (
        ("BHK", bhk),
        ("Bathroom", bath),
        ("Size", size),
        ("current_floor", current),
        ("total_floors", total),
    ):
        r = num[name]
        if value is not None and not (r["min"] <= value <= r["max"]):
            unit = " sq ft" if name == "Size" else ""
            warnings.append(
                f"{name} = {value:,g}{unit} is outside the training data ({r['min']:,g}–{r['max']:,g}"
                f"{unit}); the estimate is less reliable."
            )
    if size is not None and bhk is not None and size / bhk < 100:
        warnings.append(f"{size:,g} sq ft for {bhk} BHK is unusually small; check the size.")
    if bath is not None and bhk is not None and bath > bhk + 2:
        warnings.append(f"{bath} bathrooms for {bhk} BHK is unusual.")

    if current is None:
        current = int(num["current_floor"]["median"])
        warnings.append(f"current_floor not given — assumed {floor_label(current)}.")
    if total is None:
        total = max(int(num["total_floors"]["median"]), current, 0)
        warnings.append(f"total_floors not given — assumed {total}.")

    locality = str(listing.get("Area Locality") or "").strip()
    known = {_normalise_locality(x) for x in schema["localities"].get(values["City"], [])}
    if not locality:
        warnings.append(
            "Area Locality not given — an average locality effect is used (city, size and the other "
            "details still count)."
        )
    elif _normalise_locality(locality) not in known:
        warnings.append(
            f"Locality '{locality}' is not in the training data for {values['City']} — an average "
            "locality effect is used (city, size and the other details still count)."
        )

    row = {
        "Posted On": _PLACEHOLDER_DATE,
        "BHK": bhk,
        "Size": size,
        "Floor": f"{floor_label(current)} out of {total}",
        "Area Type": values["Area Type"],
        "Area Locality": locality or "unknown",
        "City": values["City"],
        "Furnishing Status": values["Furnishing Status"],
        "Tenant Preferred": values["Tenant Preferred"],
        "Bathroom": bath,
        "Point of Contact": values["Point of Contact"],
    }
    return pd.DataFrame([row], columns=RAW_COLUMNS), warnings


def predict_listing(listing, model=None, schema=None):
    """Validate and score one listing with the final model. Returns a dict:
    rent (averaged prediction, Rs/month), per_seed {seed: rent}, low/high (range across seeds), warnings, model.
    Raises InvalidListing for unusable input; any failure inside the model is re-raised as RuntimeError with a
    readable message."""
    schema = schema or load_input_schema()
    model = model or load_final()
    row, warnings = validate_listing(listing, schema)
    try:
        each = model.predict_each(row)[:, 0]
    except Exception as exc:  # the app shows this message instead of crashing
        raise RuntimeError(f"The model could not score this listing: {exc}") from exc
    if not np.all(np.isfinite(each)) or np.any(each <= 0):
        raise RuntimeError("The model returned an invalid prediction for this listing.")
    return {
        "rent": float(each.mean()),
        "per_seed": {int(s): float(p) for s, p in zip(model.seeds, each, strict=True)},
        "low": float(each.min()),
        "high": float(each.max()),
        "warnings": warnings,
        "model": model.name,
        "listing": row.iloc[0].to_dict(),
    }
