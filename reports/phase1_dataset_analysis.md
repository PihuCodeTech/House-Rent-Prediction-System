# Phase 1 — Dataset Inspection & Understanding

**Dataset:** `data/raw/House_Rent_Dataset.csv` · **Shape:** 4,746 rows × 12 columns · **Target:** `Rent` (continuous → regression)

## Columns & types
|                   | dtype   |   non_null |   unique |
|:------------------|:--------|-----------:|---------:|
| Posted On         | object  |       4746 |       81 |
| BHK               | int64   |       4746 |        6 |
| Rent              | int64   |       4746 |      243 |
| Size              | int64   |       4746 |      615 |
| Floor             | object  |       4746 |      480 |
| Area Type         | object  |       4746 |        3 |
| Area Locality     | object  |       4746 |     2235 |
| City              | object  |       4746 |        6 |
| Furnishing Status | object  |       4746 |        3 |
| Tenant Preferred  | object  |       4746 |        3 |
| Bathroom          | int64   |       4746 |        8 |
| Point of Contact  | object  |       4746 |        3 |

## Missing values & duplicates
- Missing values: **0** total (no column has any missing value).
- Exact duplicate rows: **0**. Near-duplicate re-listings (same property, different `Posted On`) are handled in Phase 2.

## Numerical features
|       |     BHK |        Rent |    Size |   Bathroom |
|:------|--------:|------------:|--------:|-----------:|
| count | 4746    |  4746       | 4746    |    4746    |
| mean  |    2.08 | 34993.4     |  967.49 |       1.97 |
| std   |    0.83 | 78106.4     |  634.2  |       0.88 |
| min   |    1    |  1200       |   10    |       1    |
| 25%   |    2    | 10000       |  550    |       1    |
| 50%   |    2    | 16000       |  850    |       2    |
| 75%   |    3    | 33000       | 1200    |       2    |
| max   |    6    |     3.5e+06 | 8000    |      10    |

## Target: `Rent`
- mean ₹34,993 · median ₹16,000 · std ₹78,106 · min ₹1,200 · max ₹3,500,000
- skewness **21.41** (strong right skew) → motivates a `log1p` transform (Phase 4). log1p(Rent) skew = 0.91.

## Categorical cardinality
|                   |   unique_values |
|:------------------|----------------:|
| Area Locality     |            2235 |
| Size              |             615 |
| Floor             |             480 |
| Rent              |             243 |
| Posted On         |              81 |
| Bathroom          |               8 |
| BHK               |               6 |
| City              |               6 |
| Area Type         |               3 |
| Furnishing Status |               3 |
| Tenant Preferred  |               3 |
| Point of Contact  |               3 |

- **High cardinality:** `Area Locality` (2,235 values) — not usable as plain one-hot; handled with leakage-safe **target encoding** inside CV (Phase 5/6).
- `City` (6 values), `Point of Contact` (3), `Furnishing Status` (3), `Area Type` (3), `Tenant Preferred` (3) are low-cardinality → one-hot.

## Correlation with Rent (numeric)
|          |   corr_with_Rent |
|:---------|-----------------:|
| Bathroom |            0.441 |
| Size     |            0.414 |
| BHK      |            0.37  |

## Suspicious columns / leakage
- `Posted On` is the listing date, **not** build date — no "property age" feature is derivable; only posting month / day-of-week / days-since-first are valid.
- `Floor` is free text ("Ground out of 2") → parsed into `current_floor` / `total_floors`.
- No ID/index column; no constant or near-constant column; **no target-leaking column** (nothing derived from Rent).

## Decisions carried forward
1. Regression with right-skewed target → evaluate `log1p` transform; report metrics in rupees.
2. `Area Locality` → target encoding inside CV; other categoricals → one-hot.
3. Parse `Floor` and `Posted On`; drop exact/near-duplicate re-listings.
4. Investigate outliers (do not blindly delete); treat only on the training split.
