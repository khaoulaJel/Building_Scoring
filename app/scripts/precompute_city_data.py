# ──────────────────────────────────────────────────────────────
#  scripts/precompute_city_data.py
#  -------------------------------------------------------------
#  Purpose  : heavy lifting – clean, normalize, classify, cache
#  Run when : once per city / once per yearly data refresh
#  Output   : compact <city>_buildings.parquet (all metrics + classes)
#             and updates the source CSV with classifications
# ──────────────────────────────────────────────────────────────

import argparse
import pathlib
import pandas as pd
import os

# ──────────────────────────────────────────────────────────────
# 1 ▸ Parse CLI
# ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Pre‑compute building dataset for one city.")
parser.add_argument("--city",   required=True)
parser.add_argument("--input",  required=True, help="raw CSV path")
parser.add_argument("--output", required=True, help=".parquet to write")
args = parser.parse_args()

RAW  = pathlib.Path(args.input).expanduser()
DEST = pathlib.Path(args.output).expanduser()
DEST.parent.mkdir(parents=True, exist_ok=True)

print(f"🔍 Reading file: {RAW}")

# ──────────────────────────────────────────────────────────────
# 2 ▸ Load & handle CSV directly
# ──────────────────────────────────────────────────────────────
# Read the CSV file
df = pd.read_csv(RAW)

print("🔍 Original columns:", df.columns.tolist())
print("🔍 First few rows:")
print(df.head(2))

# If your CSV is already in the right format, no need to rename
# Check if we need to rename columns based on the format
if "building_id" not in df.columns and "numero_dpe" in df.columns:
    df = df.rename(columns={
        "numero_dpe"                    : "building_id",
        "conso_5 usages_ef"             : "Energy_Consumption",       # kWh/an
        "emission_ges_5_usages"         : "CO2_Usage",                # kg/an
        "conso_5 usages_par_m2_ef"      : "Energy_Intensity",         # kWh/m²/an
        "emission_ges_5_usages par_m2"  : "CO2_Intensity"             # kg/m²/an
    })

# Add water usage if it doesn't exist
if "Water_Usage" not in df.columns and "Energy_Consumption" in df.columns:
    df["Water_Usage"] = df["Energy_Consumption"] * 0.30

# ──────────────────────────────────────────────────────────────
# 3 ▸ Run every ML / distance classifier OFF‑LINE
# ──────────────────────────────────────────────────────────────
from app.models.euclidean   import classify_euclidean
from app.models.mahalanobis import classify_mahalanobis
from app.models.pca         import classify_pca
from app.models.weighted    import classify_weighted
from app.models.bayesian    import classify_bayesian

print("🔍 Running classification algorithms...")

# Apply classifications directly to the dataframe
df["class_euclidean"]   = classify_euclidean(df)["class_label"]
df["class_mahalanobis"] = classify_mahalanobis(df)["class_label"]
df["class_pca"]         = classify_pca(df)["class_label"]
df["class_weighted"]    = classify_weighted(df)["class_label"]
df["class_bayesian"]    = classify_bayesian(df)["class_label"]

# Debug classifier output
print(" Classification complete!")
print(" Final columns:", df.columns.tolist())
print(" Preview of classified dataframe:")
print(df.head(3))

# ──────────────────────────────────────────────────────────────
# 4 ▸ Save both parquet and updated CSV
# ──────────────────────────────────────────────────────────────
# Save parquet file
df.to_parquet(DEST, index=False)
print(f" {args.city}: saved {len(df):,} rows → {DEST}")

# Update the original CSV file
csv_path = r"C:\Users\ZBooK\Documents\projects\Building_Scoring\app\data\reduced_lyon_buildings.csv"
print(f" Updating CSV file: {csv_path}")
df.to_csv(csv_path, index=False)
print(f" Successfully updated classifications in: {csv_path}")

# Optional: Create a backup of the original file
backup_path = csv_path + ".backup"
if not os.path.exists(backup_path):
    import shutil
    shutil.copy2(csv_path, backup_path)
    print(f" Created backup of original file: {backup_path}")