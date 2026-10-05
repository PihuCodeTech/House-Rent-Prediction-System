"""Save / load a fitted model and make rupee predictions on new raw rows via the fitted preprocessor."""
import joblib
import numpy as np
import pandas as pd
from . import config
from .data_cleaning import rule_strip_text, rule_parse_floor, rule_parse_date
from .feature_engineering import add_stateless_features


def save_model(model, key):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MODELS_DIR / f"{key}.joblib"
    joblib.dump(model, path)
    return path


def load_model(key):
    return joblib.load(config.MODELS_DIR / f"{key}.joblib")


def predict_raw(model, preprocessor, feature_names, raw_df):
    """raw_df: rows in the ORIGINAL schema. Applies the same stateless cleaning/features, then the
    fitted preprocessor, then the model. Returns predicted rent in rupees."""
    # Inference-safe cleaning only (skip validity/dedup rules that need the Rent column)
    clean = rule_parse_date(rule_parse_floor(rule_strip_text(raw_df.copy())))
    clean = add_stateless_features(clean)
    Z = preprocessor.transform(clean.drop(columns=[config.TARGET], errors="ignore"))
    X = pd.DataFrame(np.asarray(Z, dtype="float64"), columns=feature_names, index=clean.index)
    return model.predict(X)


def save_final(model, preprocessor, feature_names, meta=None):
    """Persist the chosen pipeline for inference: models/final_model.pkl + models/preprocessor.pkl
    (+ feature_names.json). predict_raw() uses both to go from raw rows to rupee predictions."""
    import json
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODELS_DIR / "final_model.pkl")
    joblib.dump(preprocessor, config.MODELS_DIR / "preprocessor.pkl")
    (config.MODELS_DIR / "feature_names.json").write_text(json.dumps(list(feature_names), indent=2))
    if meta is not None:
        (config.MODELS_DIR / "final_model_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return config.MODELS_DIR / "final_model.pkl"


def load_final():
    """Load (final_model, preprocessor, feature_names) for inference."""
    import json
    model = joblib.load(config.MODELS_DIR / "final_model.pkl")
    pre = joblib.load(config.MODELS_DIR / "preprocessor.pkl")
    names = json.loads((config.MODELS_DIR / "feature_names.json").read_text())
    return model, pre, names
