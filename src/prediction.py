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
from .data_cleaning import rule_parse_floor, rule_strip_text
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
_PLACEHOLDER_FLOOR = "1 out of 1"  # likewise the floor (dropped from the features)


# ----------------------------- raw-row prediction -----------------------------
def predict_raw(model, preprocessor, feature_names, raw_df):
    """Predict rupees for rows in the ORIGINAL dataset schema (Rent not needed): inference-safe cleaning,
    the fitted preprocessor, then the model."""
    missing = [c for c in RAW_COLUMNS if c not in raw_df.columns and c != "Posted On"]
    if missing:
        raise ValueError(f"raw rows are missing columns: {missing}")
    clean = add_stateless_features(rule_parse_floor(rule_strip_text(raw_df.copy())))
    if preprocessor is None:  # rule-based model (locality × size baseline): reads the raw columns itself
        return model.predict(clean)
    Z = preprocessor.transform(clean.drop(columns=[config.TARGET], errors="ignore"))
    X = pd.DataFrame(np.asarray(Z, dtype="float64"), columns=feature_names, index=clean.index)
    return model.predict(X)


class SeedAveragedModel:
    """One model trained on several seeds; predicts the mean of the per-seed predictions (rupees).

    members: list of {"seed", "model", "preprocessor", "feature_names"}; each model was fitted on its seed's
    training split together with that seed's fitted preprocessor.
    """

    error_profile = None  # class default, so models saved before the error profile existed still load cleanly

    def __init__(self, key, name, members, params=None, source=None, results=None, error_profile=None):
        self.key, self.name, self.members = key, name, list(members)
        self.params, self.source, self.results = dict(params or {}), source, results
        self.error_profile = error_profile  # see build_error_profile
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


# ----------------------------- how far off is the model? -----------------------------
def build_error_profile(tests, n_bins=5, coverage=0.8):
    """Typical error by price level, measured on the test splits: `tests` is a list of (actual, predicted) arrays,
    one per seed, every prediction made by the seed's member that never trained on those rows. Predictions are
    grouped into `n_bins` equal-size price bands; for each band the central `coverage` share of actual/predicted
    ratios gives the "likely range" multipliers."""
    actual = np.concatenate([np.asarray(a, dtype="float64") for a, _ in tests])
    predicted = np.concatenate([np.asarray(p, dtype="float64") for _, p in tests])
    ratio = actual / predicted
    edges = np.quantile(predicted, np.linspace(0, 1, n_bins + 1))[1:-1]
    band = np.digitize(predicted, edges)
    tail = (1 - coverage) / 2

    def bounds(r):  # bands with too few predictions (e.g. a constant baseline) use all predictions instead
        r = r if len(r) >= 30 else ratio
        return float(np.quantile(r, tail)), float(np.quantile(r, 1 - tail))

    low, high = zip(*(bounds(ratio[band == b]) for b in range(n_bins)), strict=True)
    low, high = list(low), list(high)
    return {
        "edges": edges.tolist(),
        "ratio_low": low,
        "ratio_high": high,
        "coverage": coverage,
        "n_test_predictions": int(len(ratio)),
    }


def likely_range(rent, profile):
    """(low, high) for a predicted rent from an error profile; None if the model has no profile."""
    if not profile:
        return None
    band = int(np.digitize([rent], profile["edges"])[0])
    return rent * profile["ratio_low"][band], rent * profile["ratio_high"][band]


def range_band(rent, profile):
    """The price band whose test errors set the likely range for `rent`: {"low", "high", "n", "ratio_low",
    "ratio_high"} (low/high are predicted-rent bounds, None when open-ended; n is about how many test predictions
    fell in the band). None if the model has no profile."""
    if not profile:
        return None
    edges = profile["edges"]
    band = int(np.digitize([rent], edges)[0])
    n_bands = len(edges) + 1
    return {
        "low": float(edges[band - 1]) if band > 0 else None,
        "high": float(edges[band]) if band < len(edges) else None,
        "n": int(round(profile.get("n_test_predictions", 0) / n_bands)),
        "ratio_low": float(profile["ratio_low"][band]),
        "ratio_high": float(profile["ratio_high"][band]),
    }


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
    display["name"] = display["name"].map(lambda n: n.title() if n.islower() else n)  # 'whitefield' -> 'Whitefield'
    localities = {city: list(dict.fromkeys(g["name"])) for city, g in display.groupby("City")}

    def rng(series, integer=False):
        s = series.dropna()
        cast = int if integer else float
        return {"min": cast(s.min()), "max": cast(s.max()), "median": cast(s.median()), "integer": integer}

    choices = {f: clean[f].value_counts().index.tolist() for f in CHOICE_FIELDS}  # most common first

    priced = clean.assign(rps=clean["Rent"] / clean["Size"])
    by_city = priced.groupby("City").agg(
        median_rent=("Rent", "median"), median_rps=("rps", "median"), n=("Rent", "size")
    )
    by_locality = priced.groupby(["City", "Area Locality"]).agg(
        median_rent=("Rent", "median"), median_rps=("rps", "median"), n=("Rent", "size")
    )
    by_locality = by_locality[by_locality["n"] >= 3]  # medians from fewer listings are too noisy to show
    market = {
        "cities": {c: {k: float(v) for k, v in r.items()} for c, r in by_city.iterrows()},
        "localities": {
            city: {loc: {k: float(v) for k, v in r.items()} for (_, loc), r in g.iterrows()}
            for city, g in by_locality.groupby(level=0)
        },
    }
    return {
        "version": 1,
        "choices": choices,
        "localities": localities,
        "numeric": {
            "BHK": rng(clean["BHK"], True),
            "Bathroom": rng(clean["Bathroom"], True),
            "Size": rng(clean["Size"]),
        },
        "defaults": {f: choices[f][0] for f in CHOICE_FIELDS},
        "market": market,
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


def validate_listing(listing, schema):
    """Check and normalise one listing (a dict). Returns (one-row DataFrame in the raw schema, warnings).
    Raises InvalidListing listing every problem.

    Fields: City, BHK, Size (sq ft), Bathroom (required); Area Locality, Furnishing Status, Area Type,
    Tenant Preferred, Point of Contact (optional — sensible defaults with a warning). Floor fields are ignored: the
    model does not use them.
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

    if bhk is not None and bhk < 1:
        errors.append("BHK must be at least 1.")
    if bath is not None and bath < 1:
        errors.append("Bathroom must be at least 1.")
    if size is not None and size <= 0:
        errors.append("Size must be greater than 0 sq ft.")
    if errors:
        raise InvalidListing(errors)

    for name, value in (
        ("BHK", bhk),
        ("Bathroom", bath),
        ("Size", size),
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
        "Floor": _PLACEHOLDER_FLOOR,  # part of the raw schema; not a model input
        "Area Type": values["Area Type"],
        "Area Locality": locality or "unknown",
        "City": values["City"],
        "Furnishing Status": values["Furnishing Status"],
        "Tenant Preferred": values["Tenant Preferred"],
        "Bathroom": bath,
        "Point of Contact": values["Point of Contact"],
    }
    return pd.DataFrame([row], columns=RAW_COLUMNS), warnings


def market_context(city, locality, schema):
    """Median rent and rent per sq ft for the city and (if it has at least 3 listings) the locality."""
    market = schema.get("market", {})
    city_stats = market.get("cities", {}).get(city)
    loc_stats = market.get("localities", {}).get(city, {}).get(_normalise_locality(locality)) if locality else None
    return {"city": city_stats, "locality": loc_stats}


def predict_listing(listing, model=None, schema=None):
    """Validate and score one listing with the final model. Returns a dict with:
    rent (averaged prediction, Rs/month); low / high and coverage (likely range from the model's test-set errors at
    this price level) and range_band (that price band); seed_low / seed_high and per_seed (spread across the three seed models); rent_per_sqft;
    market (city / locality medians); warnings; model; listing (the normalised raw row).
    Raises InvalidListing for unusable input; any failure inside the model becomes a RuntimeError with a readable
    message."""
    schema = schema or load_input_schema()
    model = model or load_final()
    row, warnings = validate_listing(listing, schema)
    try:
        each = model.predict_each(row)[:, 0]
    except Exception as exc:  # the app shows this message instead of crashing
        raise RuntimeError(f"The model could not score this listing: {exc}") from exc
    if not np.all(np.isfinite(each)) or np.any(each <= 0):
        raise RuntimeError("The model returned an invalid prediction for this listing.")
    rent = float(each.mean())
    profile = getattr(model, "error_profile", None)
    low, high = likely_range(rent, profile) or (float(each.min()), float(each.max()))
    r = row.iloc[0]
    return {
        "rent": rent,
        "low": float(low),
        "high": float(high),
        "coverage": profile["coverage"] if profile else None,
        "range_band": range_band(rent, profile),
        "seed_low": float(each.min()),
        "seed_high": float(each.max()),
        "per_seed": {int(s): float(p) for s, p in zip(model.seeds, each, strict=True)},
        "rent_per_sqft": rent / float(r["Size"]),
        "market": market_context(r["City"], r["Area Locality"], schema),
        "warnings": warnings,
        "model": model.name,
        "listing": r.to_dict(),
    }


def what_if(listing, model=None, schema=None):
    """How the estimate changes when one thing about the listing changes. Returns a list of
    {"group", "label", "rent", "change"} (change = rent minus the listing's own estimate), scored in one batch.
    Variants that would be invalid (e.g. 0 BHK) are skipped."""
    schema = schema or load_input_schema()
    model = model or load_final()
    base_row, _ = validate_listing(listing, schema)
    base = base_row.iloc[0]
    size, bhk, bath = float(base["Size"]), int(base["BHK"]), int(base["Bathroom"])
    variants = []
    for option in schema["choices"]["Furnishing Status"]:
        if option != base["Furnishing Status"]:
            variants.append(("Furnishing", option, {"Furnishing Status": option}))
    num = schema.get("numeric", {})

    def within(field, value):  # only suggest changes inside the range the model was trained on
        r = num.get(field)
        return r is None or r["min"] <= value <= r["max"]

    for label, value in (
        (f"{size * 0.8:,.0f} sq ft (20% smaller)", round(size * 0.8)),
        (f"{size * 1.2:,.0f} sq ft (20% larger)", round(size * 1.2)),
    ):
        if within("Size", value):
            variants.append(("Size", label, {"Size": value}))
    for value in (bhk + 1, bhk - 1):
        if value >= 1 and within("BHK", value):
            variants.append(("Bedrooms", f"{value} BHK, same size", {"BHK": value}))
    for value in (bath + 1, bath - 1):
        if value >= 1 and within("Bathroom", value):
            variants.append(("Bathrooms", f"{value} bathroom{'s' if value > 1 else ''}", {"Bathroom": value}))
    for option in schema["choices"]["Point of Contact"]:
        if option != base["Point of Contact"] and option != "Contact Builder":
            variants.append(
                ("Listed by", option.replace("Contact ", "listed by ").lower(), {"Point of Contact": option})
            )

    rows, kept = [base_row], []
    for group, label, change in variants:
        try:
            row, _ = validate_listing({**listing, **change}, schema)
        except InvalidListing:
            continue
        rows.append(row)
        kept.append((group, label))
    predictions = model.predict(pd.concat(rows, ignore_index=True))
    own = float(predictions[0])
    return [
        {"group": g, "label": lbl, "rent": float(p), "change": float(p) - own}
        for (g, lbl), p in zip(kept, predictions[1:], strict=True)
    ]
