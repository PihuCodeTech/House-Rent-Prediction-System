# Phase 2 — Data Cleaning

All cleaning is rule-based (stateless, applied before the split) or deferred to transformers fit on the
training set only (no leakage). No row is dropped by hand-picked index.

## Stateless cleaning rules (in order)
| rule                |   rows_before |   rows_after |   rows_removed |
|:--------------------|--------------:|-------------:|---------------:|
| rule_strip_text     |          4746 |         4746 |              0 |
| rule_validity       |          4746 |         4746 |              0 |
| rule_dedup_relisted |          4746 |         4737 |              9 |
| rule_parse_floor    |          4737 |         4737 |              0 |
| rule_parse_date     |          4737 |         4737 |              0 |

- **`rule_strip_text`** — trim whitespace; normalise `Area Locality` casing/punctuation so variants collapse.
- **`rule_validity`** — keep rows with `Rent>0, Size>0, BHK>=1, Bathroom>=1` (a rule, not an index list). 0 rows failed.
- **`rule_dedup_relisted`** — drop exact re-listings (identical except `Posted On`): **9 rows** removed.
- **`rule_parse_floor`** — "Ground out of 2" → `current_floor`, `total_floors` (Ground=0, basements negative).
- **`rule_parse_date`** — `Posted On` → datetime.

## Missing values
No missing values in the raw data. A median (numeric) / most-frequent (categorical) `SimpleImputer` is still
included in the pipeline as a **safety net**, fit on train only.

## Invalid / impossible values
Handled by `rule_validity` above (none present today, but the guard stays).

## Columns
No columns dropped as useless. Raw text columns (`Floor`, `Posted On`, `Area Locality`) are **transformed**, not
kept as-is, in the modelling matrix.

## Outliers
Investigated in Phase 3. Decision: **do not delete from val/test.** A train-only far-out filter on
log(Rent/Size) (Tukey k=3) removes a handful of extreme data-entry-like rows from the training split only.

## Output
- `data/processed/cleaned_dataset.csv` — shape **4,737 × 18** (post rules + stateless features).
- `data/processed/cleaning_rule_log.csv` — the table above.
