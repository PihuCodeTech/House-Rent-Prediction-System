# Phase 3 — Exploratory Data Analysis & Outlier Analysis

Visualizations are saved under `visualizations/` (per-model diagnostics) and generated inline by the model
notebooks. Key findings below.

## Target distribution
- `Rent` is strongly right-skewed (skew **21.41**; median ₹16,000 vs mean ₹34,993, max ₹3,500,000).
- `log1p(Rent)` is near-symmetric (skew **0.91**) → models are trained on the log scale and
  predictions are back-transformed with `expm1` to rupees.

## Numeric correlations with Rent
|          |   corr_with_Rent |
|:---------|-----------------:|
| Bathroom |            0.441 |
| Size     |            0.414 |
| BHK      |            0.37  |

`Size`, `Bathroom` and `BHK` are the strongest linear signals; interactions (city × size) add more in the models.

## Rent by City (median)
| City      |   median_rent |
|:----------|--------------:|
| Mumbai    |         52000 |
| Delhi     |         17000 |
| Bangalore |         14000 |
| Chennai   |         14000 |
| Hyderabad |         14000 |
| Kolkata   |          8500 |

Mumbai and Delhi command the highest rents — city is a major driver and is one-hot encoded plus crossed with size.

## Outliers
- IQR rule (1.5×) on Rent flags **520** points; |z|>3 flags **66**.
- Inspection shows most are **legitimate luxury properties** (large `Size`, premium `City`), not errors — so they
  are **kept**. Only extreme rent-per-sqft outliers in the **training split** are trimmed (Tukey k=3), never in
  validation/test, which must reflect reality.

## Decisions carried forward
1. Use `log1p(Rent)`; report all metrics in rupees.
2. Keep outliers except train-only far-out rent/sqft rows.
3. Stratify the split on `qcut(Rent, 10)` bins so all rent ranges are represented in train/val/test.
