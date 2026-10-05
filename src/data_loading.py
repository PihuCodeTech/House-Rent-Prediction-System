"""Load the raw dataset. The CSV is the single input to the whole pipeline."""
import pandas as pd
from . import config

RAW_SHAPE = (4746, 12)

def load_raw(path=None):
    """Read House_Rent_Dataset.csv. Tries data/raw/ then the repo-root copy, or an explicit path."""
    if path is not None:
        return pd.read_csv(path)
    for p in (config.DATA_RAW, config.DATA_RAW_FALLBACK):
        if p.exists():
            df = pd.read_csv(p)
            return df
    raise FileNotFoundError(
        "House_Rent_Dataset.csv not found in data/raw/ or repo root. "
        "Pass an explicit path: load_raw('House_Rent_Dataset.csv')."
    )
