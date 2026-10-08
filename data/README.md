# Data

The dataset is **not included in this repository** — it belongs to its author and is not ours to redistribute.

## Get it

1. Download the **House Rent Prediction Dataset** by Sourav Banerjee from Kaggle:
   <https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset>
   (according to its description, collected from [MagicBricks](https://www.magicbricks.com/); check its terms on Kaggle).
2. Save the CSV here, unchanged:

```
data/raw/House_Rent_Dataset.csv      # 4,746 rows × 12 columns, listings posted April–July 2022
```

Everything else is built in memory from that one file: every notebook cleans it with the same rules and prints the
cleaned size (4,731 rows). No cleaned or processed copy is ever written to disk. `data/` is git-ignored apart from
this README.

## Columns

| Column | Meaning |
|---|---|
| Posted On | Date the listing was posted (not used as a feature) |
| BHK | Number of bedrooms, hall, kitchen |
| Rent | Monthly rent in rupees — **the target** |
| Size | Size in square feet |
| Floor | Floor and building height, e.g. "Ground out of 2", "3 out of 5" |
| Area Type | How the size was measured: Super Area, Carpet Area or Built Area |
| Area Locality | Locality of the property |
| City | Bangalore, Chennai, Delhi, Hyderabad, Kolkata or Mumbai |
| Furnishing Status | Furnished, Semi-Furnished or Unfurnished |
| Tenant Preferred | Bachelors, Family, or Bachelors/Family |
| Bathroom | Number of bathrooms |
| Point of Contact | Contact Owner, Contact Agent or Contact Builder |
