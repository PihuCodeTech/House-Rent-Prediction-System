"""Housing-rent-prediction source package.

Single source of truth for the data pipeline so every model notebook runs the
IDENTICAL preparation (same rows, features, split, seed) directly from
House_Rent_Dataset.csv. Import, don't copy-paste.
"""
from . import config          # noqa: F401
