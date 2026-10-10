"""The preparation pipeline: cleaned data -> 70/15/15 split -> train-only filter -> encoders fitted on train.

`prepare_data(seed=...)` is the single entry point every model notebook calls. For a given seed it is
deterministic, so all notebooks get identical splits ("same data, different model"). A fingerprint, plus a
run reference recorded per seed, guards against any drift between notebooks. A different seed changes the
train/validation/test rows and the locality encoder's internal folds.
"""

import hashlib
import json
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

from . import config
from .data_cleaning import apply_stateless_rules, train_only_far_out_mask
from .data_loading import load_raw
from .feature_engineering import CAT_COLS, FB_INPUT_COLS, FeatureBuilder, add_stateless_features


# ----------------------------- split -----------------------------
def _hash(b):
    return hashlib.sha256(b).hexdigest()[:16]


def _index_hash(idx):
    return _hash(np.sort(np.asarray(idx)).tobytes())


def make_split(df, seed=config.RANDOM_STATE):
    """70/15/15 two-step split, stratified on rent deciles (the bins are used for splitting only)."""
    y = df[config.TARGET]
    X = df.drop(columns=[config.TARGET])
    bins = pd.qcut(y, q=config.STRAT_Q, labels=False, duplicates="drop")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=seed, stratify=bins
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=config.VAL_SIZE_OF_TEMP, random_state=seed, stratify=bins.loc[X_temp.index]
    )
    if seed == 42:
        got = {"train": _index_hash(X_train.index), "val": _index_hash(X_val.index), "test": _index_hash(X_test.index)}
        if got != config.EXPECTED_SPLIT_HASHES:
            warnings.warn(
                f"seed-42 split differs from the recorded reference {got}; this can happen with another "
                "pandas/scikit-learn version. All notebooks in this environment still share the same split.",
                stacklevel=2,
            )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ----------------------------- preprocessor -----------------------------
def _target_encoder(seed):
    """Cross-fitted locality target encoder with shuffled 5-fold internal splits. scikit-learn >= 1.9 takes the
    splitter as `cv` (shuffle/random_state are deprecated there and removed in 1.11); older versions take
    shuffle/random_state. Both build the same KFold(5, shuffle=True, random_state=seed), so encodings match."""
    import sklearn

    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 9):
        return TargetEncoder(target_type="continuous", smooth="auto", cv=KFold(5, shuffle=True, random_state=seed))
    return TargetEncoder(target_type="continuous", smooth="auto", cv=5, shuffle=True, random_state=seed)


def build_preprocessor(seed=config.RANDOM_STATE):
    """Unfitted preprocessor: numeric features, one-hot categoricals (rare levels pooled) and the locality
    target encoding. Fit it on training rows only (it is re-fit inside every CV fold)."""
    return ColumnTransformer(
        [
            ("num", FeatureBuilder(clip_k=config.CLIP_K), FB_INPUT_COLS),
            (
                "cat",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False),
                CAT_COLS,
            ),
            ("loc", _target_encoder(seed), ["locality_key"]),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def _clean_name(name):
    name = name.replace("loc__locality_key", "locality_target_enc").replace("num__", "").replace("cat__", "")
    return re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_")


# ----------------------------- fingerprint & checks -----------------------------
def compute_fingerprint(X_train, X_val, X_test, y_train, y_val, y_test, feature_names):
    """Hashes of rows, targets and feature names (`reference`) and of the feature values (`content`)."""
    splits = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}
    ref = {
        "n_rows": {k: int(len(x)) for k, (x, _) in splits.items()},
        "n_features": len(feature_names),
        "feature_names_hash": _hash("|".join(feature_names).encode()),
        "index_hash_sorted": {k: _hash(np.sort(x.index.to_numpy("int64")).tobytes()) for k, (x, _) in splits.items()},
        "index_hash_ordered": {k: _hash(x.index.to_numpy("int64").tobytes()) for k, (x, _) in splits.items()},
        "target_hash": {k: _hash(y.to_numpy("int64").tobytes()) for k, (_, y) in splits.items()},
    }
    ref["reference_id"] = _hash(json.dumps(ref, sort_keys=True).encode())
    content = {k: _hash(np.ascontiguousarray(x.to_numpy("float64")).tobytes()) for k, (x, _) in splits.items()}
    return {"reference": ref, "content": content}


def _integrity_checks(X_train, X_val, X_test, y_train, y_val, y_test, feature_names):
    Xs, ys = (X_train, X_val, X_test), (y_train, y_val, y_test)
    idx = [set(x.index) for x in Xs]
    checks = {
        "no NaN/inf in X": all(np.isfinite(x.to_numpy()).all() for x in Xs),
        "identical column order": all(list(x.columns) == feature_names for x in Xs),
        "all y int64 rupees > 0": all(s.dtype == "int64" and s.min() > 0 for s in ys),
        "X/y indices aligned": all(x.index.equals(s.index) for x, s in zip(Xs, ys, strict=True)),
        "no split overlap": not (idx[0] & idx[1] or idx[0] & idx[2] or idx[1] & idx[2]),
        "no target/raw columns in X": not any(
            re.fullmatch(r"rent|bins?|strat.*|posted_on|floor|area_locality|locality_key", c, re.I)
            for c in feature_names
        ),
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError("Data integrity check failed:\n  " + "\n  ".join(failed))


# ----------------------------- orchestration -----------------------------
def prepare_data(seed=config.RANDOM_STATE, clean=None):
    """Prepare the splits for one seed. Nothing is written to disk.

    clean: the cleaned DataFrame (output of the stateless rules); loaded and cleaned here if None.
    Returns a dict: X_train / X_val / X_test (float64 feature frames), y_train / y_val / y_test (int64 rupees),
    X_train_raw (training rows before encoding, used to re-fit the preprocessor inside each CV fold),
    feature_names, preprocessor (fitted on train), fingerprint, target_transform, seed, meta.
    """
    if clean is None:
        clean, _ = apply_stateless_rules(load_raw())
    n_clean = len(clean)
    clean = add_stateless_features(clean)

    X_tr_raw, X_val_raw, X_te_raw, y_tr_raw, y_val, y_test = make_split(clean, seed=seed)
    keep, far_info = train_only_far_out_mask(X_tr_raw, y_tr_raw, k=config.FAR_OUT_K)
    X_tr_raw, y_train = X_tr_raw.loc[keep], y_tr_raw.loc[keep]

    pre = build_preprocessor(seed=seed)
    Z_tr = pre.fit_transform(X_tr_raw, np.log1p(y_train))  # the locality encoder is cross-fitted on train
    feature_names = [_clean_name(n) for n in pre.get_feature_names_out()]
    if len(set(feature_names)) != len(feature_names):
        raise RuntimeError("feature names are not unique")

    def frame(Z, index):
        return pd.DataFrame(np.asarray(Z, dtype="float64"), columns=feature_names, index=index)

    X_train = frame(Z_tr, X_tr_raw.index)
    X_val = frame(pre.transform(X_val_raw), X_val_raw.index)
    X_test = frame(pre.transform(X_te_raw), X_te_raw.index)
    y_train, y_val, y_test = (s.astype("int64").rename(config.TARGET) for s in (y_train, y_val, y_test))

    _integrity_checks(X_train, X_val, X_test, y_train, y_val, y_test, feature_names)
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "X_train_raw": X_tr_raw,
        "X_val_raw": X_val_raw,  # raw rows, for rule-based models (the locality × size baseline)
        "X_test_raw": X_te_raw,
        "feature_names": feature_names,
        "preprocessor": pre,
        "fingerprint": compute_fingerprint(X_train, X_val, X_test, y_train, y_val, y_test, feature_names),
        "target_transform": config.TARGET_TRANSFORM,
        "seed": seed,
        "meta": {
            "far_out_filter": far_info,
            "rows": {"after_cleaning": int(n_clean), "train": len(X_train), "val": len(X_val), "test": len(X_test)},
        },
    }


def assert_run_consistency(data):
    """Same-data tripwire across notebooks. The first notebook to run a seed records that seed's reference id in
    reports/.run_reference.json; every later notebook must get the same id for the same seed."""
    seed, ref_id = str(data["seed"]), data["fingerprint"]["reference"]["reference_id"]
    path = config.RUN_REFERENCE
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        refs = json.loads(path.read_text()).get("seeds", {}) if path.exists() else {}
    except (ValueError, AttributeError):
        refs = {}
    if seed in refs and refs[seed] != ref_id:
        raise RuntimeError(
            f"Seed {seed}: this run's data (id {ref_id}) differs from the data an earlier notebook used "
            f"(id {refs[seed]}). If you changed the pipeline on purpose, delete reports/{path.name} "
            "(run_all.py does this automatically) and re-run all notebooks."
        )
    if seed not in refs:
        refs[seed] = ref_id
        path.write_text(json.dumps({"seeds": refs}, indent=2))
    if seed == "42" and ref_id != config.EXPECTED_REFERENCE_ID:
        warnings.warn(
            f"seed-42 data id {ref_id} differs from the recorded {config.EXPECTED_REFERENCE_ID} "
            "(probably a pandas/scikit-learn version change).",
            stacklevel=2,
        )
    return ref_id
