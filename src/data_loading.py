"""Load the raw dataset — the single input of the whole pipeline."""

import pandas as pd

from . import config


def load_raw(path=None):
    """Read House_Rent_Dataset.csv from data/raw/ (or from an explicit `path`)."""
    path = path or config.DATA_RAW
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download it as described in data/README.md "
            "and save it as data/raw/House_Rent_Dataset.csv."
        ) from None
