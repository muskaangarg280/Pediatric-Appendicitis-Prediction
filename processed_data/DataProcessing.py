# DataProcessing.py
# Cleans the raw pediatric appendicitis tabular data and writes:
#   - processed_data.csv   (full cleaned dataset)
#   - dataset_tabular.csv  (tabular-only view used by the EDA and MLP)

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT.parent / "raw_data" / "tabular_data.csv"

# -- Load raw data once -----------------------------------------------
df = pd.read_csv(RAW_PATH)

# -- Drop sparse (>300 nulls) and unnamed columns ---------------------
null_count = df.isnull().sum()
null_columns = null_count[null_count > 300].index.tolist()
empty_columns = [col for col in df.columns if str(col).startswith("Unnamed")]
dropping_columns = list(set(null_columns + empty_columns))
df = df.drop(columns=dropping_columns)

print("Dropped columns:")
for col in dropping_columns:
    print(f"  {col}: {null_count.get(col, 'NA (empty column)')} nulls")

# -- Drop rows with more than 4 missing values ------------------------
nulls_per_row = df.isnull().sum(axis=1)
print("\nNumber of rows vs number of nulls in each row:")
print(nulls_per_row.value_counts().sort_index())

print(f"\nRows before dropping: {len(df)}")
df = df[nulls_per_row <= 4].reset_index(drop=True)
print(f"Rows after dropping: {len(df)}")

# -- Encode categorical features --------------------------------------
# Gender -> numeric label
df["Sex"] = LabelEncoder().fit_transform(df["Sex"])

# Yes/No symptom & flag columns -> 1/0
yes_no_cols = [
    "Migratory_Pain", "Lower_Right_Abd_Pain", "Contralateral_Rebound_Tenderness",
    "Coughing_Pain", "Nausea", "Loss_of_Appetite", "Dysuria", "Psoas_Sign",
    "Ipsilateral_Rebound_Tenderness", "Appendix_on_US", "Neutrophilia",
    "Free_Fluids", "US_Performed",
]
for col in yes_no_cols:
    df[col] = df[col].astype(str).str.strip().str.lower().apply(lambda x: 1 if x == "yes" else 0)

# -- Save the full cleaned dataset ------------------------------------
df.to_csv(ROOT / "processed_data.csv", index=False)

# -- Build and save the tabular-only view -----------------------------
image_cols = ["Appendix_on_US", "Appendix_Diameter", "US_Performed", "Free_Fluids"]
df_tabular = (
    df.drop(columns=image_cols, errors="ignore")
      .rename(columns={"US_Number": "Img_Ref_Num"})
)
df_tabular.to_csv(ROOT / "dataset_tabular.csv", index=False)

print("\nSaved: processed_data.csv, dataset_tabular.csv")