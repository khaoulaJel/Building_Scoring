# ──────────────────────────────────────────────────────────────
#  scripts/precompute_city_data.py
# ──────────────────────────────────────────────────────────────

import argparse
import pathlib
import pandas as pd
import numpy as np
import os

# ──────────────────────────────────────────────────────────────
# 1 ▸ Parse CLI
# ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Pre‑compute building dataset for one city.")
parser.add_argument("--city",   required=True)
parser.add_argument("--input",  required=True, help="raw CSV path")
parser.add_argument("--output", required=True, help=".parquet to write")
args = parser.parse_args()

city = args.city.lower()

RAW  = pathlib.Path(args.input).expanduser()
DEST = pathlib.Path(args.output).expanduser()
DEST.parent.mkdir(parents=True, exist_ok=True)

print(f"🔍 Reading file: {RAW}")

# ──────────────────────────────────────────────────────────────
# 2 ▸ Load & handle CSV directly
# ──────────────────────────────────────────────────────────────
df = pd.read_csv(RAW)

print("🔍 Original columns:", df.columns.tolist())
print("🔍 First few rows:")
print(df.head(2))

# Rename columns if needed
if "building_id" not in df.columns and "numero_dpe" in df.columns:
    df = df.rename(columns={
        "numero_dpe"                    : "building_id",
        "conso_5 usages_ef"             : "Energy_Consumption",
        "emission_ges_5_usages"         : "CO2_Usage",
        "conso_5 usages_par_m2_ef"      : "Energy_Intensity",
        "emission_ges_5_usages par_m2"  : "CO2_Intensity"
    })

# Add Water_Usage if missing
if "Water_Usage" not in df.columns and "Energy_Consumption" in df.columns:
    df["Water_Usage"] = df["Energy_Consumption"] * 0.30

# Add log-transformed features
log_targets = ["Energy_Consumption", "CO2_Usage", "Energy_Intensity", "CO2_Intensity"]
for col in log_targets:
    df[f"log1p_{col}"] = np.log1p(df[col])

# ──────────────────────────────────────────────────────────────
# 3 ▸ Run every ML / distance classifier OFF‑LINE
# ──────────────────────────────────────────────────────────────
from app.models.euclidean   import classify_euclidean
from app.models.mahalanobis import classify_mahalanobis
from app.models.pca         import classify_pca
from app.models.weighted    import classify_weighted
from app.models.bayesian    import classify_bayesian

print("🔍 Running classification algorithms...")

# ✅ Define the features to use
selected_features = ["log1p_CO2_Usage", "Water_Usage", "log1p_Energy_Consumption"]

# Run classifiers
df = classify_mahalanobis(df, selected_features, return_distance=True)
df["class_mahalanobis"] = df["class_label"]

df["class_euclidean"]   = classify_euclidean(df, selected_features)["class_label"]
df["class_pca"]         = classify_pca(df, selected_features)["class_label"]
df["class_weighted"]    = classify_weighted(df, selected_features)["class_label"]
df["class_bayesian"]    = classify_bayesian(df, selected_features)["class_label"]

print("✅ Classification complete!")
print("🧾 Final columns:", df.columns.tolist())
print("🔎 Preview of classified dataframe:")
print(df.head(3))

# ──────────────────────────────────────────────────────────────
# 4 ▸ Save both parquet and updated CSV
# ──────────────────────────────────────────────────────────────
df.to_parquet(DEST, index=False)
print(f"💾 {city.title()}: saved {len(df):,} rows → {DEST}")

# Overwrite the input CSV with enriched data
output_csv_path = RAW
print(f"📝 Overwriting CSV with enriched data: {output_csv_path}")
df.to_csv(output_csv_path, index=False)
print(f"✅ Updated: {output_csv_path}")

# Optional: create a backup of this file
backup_path = output_csv_path.with_suffix(output_csv_path.suffix + ".backup")
if not backup_path.exists():
    import shutil
    shutil.copy2(output_csv_path, backup_path)
    print(f"📦 Created backup of original file: {backup_path}")
