"""The frozen preparation pipeline: raw CSV -> clean -> features -> 70/15/15 split -> fit-on-train encode.

`prepare_data()` is the single entry point every model notebook calls. It is deterministic
(seed + data), so all notebooks obtain identical splits ("same data, different model"), and a
fingerprint plus a cross-model run-reference guard against any drift.
"""
import json, hashlib, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

from . import config
from .data_loading import load_raw
from .data_cleaning import apply_stateless_rules, train_only_far_out_mask
from .feature_engineering import (add_stateless_features, FeatureBuilder, LocalityCountEncoder,
                                  FB_INPUT_COLS, CAT_COLS)

RS = config.RANDOM_STATE


# ----------------------------- split -----------------------------
def _index_hash(idx):
    return hashlib.sha256(np.sort(np.asarray(idx)).tobytes()).hexdigest()[:16]


def make_split(df):
    """70/15/15 two-step split stratified on q=10 target bins (bins for splitting ONLY, never a feature)."""
    y = df[config.TARGET]
    X = df.drop(columns=[config.TARGET])
    bins = pd.qcut(y, q=config.STRAT_Q, labels=False, duplicates="drop")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=RS, stratify=bins)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=config.VAL_SIZE_OF_TEMP, random_state=RS, stratify=bins.loc[X_temp.index])
    got = {"train": _index_hash(X_train.index), "val": _index_hash(X_val.index), "test": _index_hash(X_test.index)}
    if got != config.EXPECTED_SPLIT_HASHES:
        warnings.warn(f"pre-filter split hashes differ from the frozen reference: {got}. "
                      "This can happen with a different pandas/sklearn version; the split is still "
                      "identical across all model notebooks in THIS environment.")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ----------------------------- preprocessor -----------------------------
def build_preprocessor(groups=config.FEATURE_GROUPS, locality=config.LOCALITY_ENCODING, clip_k=config.CLIP_K):
    """Unfitted ColumnTransformer. Fit on X_train ONLY. locality in {"target","count",None}."""
    parts = [("num", FeatureBuilder(groups=groups, clip_k=clip_k), FB_INPUT_COLS),
             ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10,
                                   sparse_output=False), CAT_COLS)]
    if locality == "target":
        parts.append(("loc", TargetEncoder(target_type="continuous", smooth="auto", cv=5,
                                           shuffle=True, random_state=RS), ["locality_key"]))
    elif locality == "count":
        parts.append(("loc", LocalityCountEncoder(), ["locality_key"]))
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=True)


def _clean_name(n):
    import re
    n = n.replace("loc__locality_key", "locality_target_enc").replace("num__", "").replace("cat__", "")
    return re.sub(r"[^0-9A-Za-z_]+", "_", n).strip("_")


# ----------------------------- fingerprint -----------------------------
def _h(b):
    return hashlib.sha256(b).hexdigest()[:16]


def compute_fingerprint(X_train, X_val, X_test, y_train, y_val, y_test, feature_names):
    splits = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}
    ref = {"n_rows": {k: int(len(x)) for k, (x, _) in splits.items()},
           "n_features": len(feature_names),
           "feature_names_hash": _h("|".join(feature_names).encode()),
           "index_hash_sorted": {k: _h(np.sort(x.index.to_numpy("int64")).tobytes()) for k, (x, _) in splits.items()},
           "index_hash_ordered": {k: _h(x.index.to_numpy("int64").tobytes()) for k, (x, _) in splits.items()},
           "target_hash": {k: _h(y.to_numpy("int64").tobytes()) for k, (_, y) in splits.items()}}
    ref["reference_id"] = _h(json.dumps(ref, sort_keys=True).encode())
    content = {k: _h(np.ascontiguousarray(x.to_numpy("float64")).tobytes()) for k, (x, _) in splits.items()}
    return {"reference": ref, "content": content}


def _integrity_checks(X_train, X_val, X_test, y_train, y_val, y_test, feature_names):
    import re
    checks = {
        "no NaN/inf in X": all(np.isfinite(x.to_numpy()).all() for x in (X_train, X_val, X_test)),
        "identical column order": list(X_train.columns) == list(X_val.columns) == list(X_test.columns) == feature_names,
        "all X float64": all((x.dtypes == "float64").all() for x in (X_train, X_val, X_test)),
        "all y int64 rupees > 0": all(s.dtype == "int64" and s.min() > 0 for s in (y_train, y_val, y_test)),
        "X/y indices aligned": all(x.index.equals(s.index) for x, s in ((X_train, y_train), (X_val, y_val), (X_test, y_test))),
        "no split overlap": not (set(X_train.index) & set(X_val.index) or set(X_train.index) & set(X_test.index)
                                 or set(X_val.index) & set(X_test.index)),
        "no leakage/raw columns in X": not any(
            re.fullmatch(r"rent|bins?|strat.*|posted_on|posted_date|floor|area_locality|locality_key", c, re.I)
            for c in feature_names),
    }
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise RuntimeError("INTEGRITY CHECK FAILED — STOP:\n  " + "\n  ".join(failed))
    return checks


# ----------------------------- orchestration -----------------------------
def prepare_data(csv_path=None, verbose=False):
    """Run the whole frozen pipeline from the raw CSV and return a dict with the prepared splits.

    Returns keys: X_train/X_val/X_test (float64 DataFrames), y_train/y_val/y_test (int64 rupees),
    feature_names, preprocessor (fitted), fingerprint, target_transform, meta.
    """
    raw = load_raw(csv_path)
    clean, rule_log = apply_stateless_rules(raw, verbose=verbose)
    clean = add_stateless_features(clean)

    X_tr_raw, X_val_raw, X_te_raw, y_tr_raw, y_val, y_test = make_split(clean)

    keep, far_info = train_only_far_out_mask(X_tr_raw, y_tr_raw, k=config.FAR_OUT_K)
    X_tr_raw, y_train = X_tr_raw.loc[keep], y_tr_raw.loc[keep]

    pre = build_preprocessor()
    Z_tr = pre.fit_transform(X_tr_raw, np.log1p(y_train))     # locality target-encoder is cross-fitted on train
    Z_val = pre.transform(X_val_raw)
    Z_te = pre.transform(X_te_raw)

    feature_names = [_clean_name(n) for n in pre.get_feature_names_out()]
    assert len(set(feature_names)) == len(feature_names), "feature names must be unique"

    def frame(Z, idx):
        return pd.DataFrame(np.asarray(Z, dtype="float64"), columns=feature_names, index=idx)

    X_train, X_val, X_test = frame(Z_tr, X_tr_raw.index), frame(Z_val, X_val_raw.index), frame(Z_te, X_te_raw.index)
    y_train = y_train.astype("int64").rename(config.TARGET)
    y_val = y_val.astype("int64").rename(config.TARGET)
    y_test = y_test.astype("int64").rename(config.TARGET)

    _integrity_checks(X_train, X_val, X_test, y_train, y_val, y_test, feature_names)
    fp = compute_fingerprint(X_train, X_val, X_test, y_train, y_val, y_test, feature_names)

    return {"X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val, "y_test": y_test,
            "feature_names": feature_names, "preprocessor": pre, "fingerprint": fp,
            "target_transform": config.TARGET_TRANSFORM,
            "meta": {"rule_log": json.loads(rule_log.to_json(orient="records")), "far_out_filter": far_info,
                     "rows": {"raw": int(len(raw)), "after_rules": int(len(clean)),
                              "train": int(len(X_train)), "val": int(len(X_val)), "test": int(len(X_test))},
                     "feature_groups": list(config.FEATURE_GROUPS), "locality_encoding": config.LOCALITY_ENCODING,
                     "random_state": RS}}


def assert_run_consistency(data, enforce_expected=False):
    """Cross-model fairness tripwire. The first model notebook to run writes reports/_run_reference.json;
    every later notebook asserts it got the SAME reference_id. Guarantees all models used identical data
    in THIS environment, independent of library versions."""
    ref_id = data["fingerprint"]["reference"]["reference_id"]
    config.RUN_REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    if config.RUN_REFERENCE.exists():
        saved = json.loads(config.RUN_REFERENCE.read_text())
        if saved["reference_id"] != ref_id:
            raise RuntimeError(
                f"FAIRNESS TRIPWIRE — STOP. This run's reference_id {ref_id} != the run reference "
                f"{saved['reference_id']} in {config.RUN_REFERENCE.name}. The prepared data differs "
                "between model notebooks. Delete reports/_run_reference.json to re-baseline if the "
                "pipeline was intentionally changed.")
    else:
        config.RUN_REFERENCE.write_text(json.dumps(
            {"reference_id": ref_id, "n_features": data["fingerprint"]["reference"]["n_features"],
             "rows": data["meta"]["rows"]}, indent=2))
    if enforce_expected and ref_id != config.EXPECTED_REFERENCE_ID:
        warnings.warn(f"reference_id {ref_id} != frozen EXPECTED_REFERENCE_ID {config.EXPECTED_REFERENCE_ID} "
                      "(library-version drift from the original run).")
    return ref_id
