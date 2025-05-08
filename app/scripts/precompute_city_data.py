#!/usr/bin/env python3
"""
Pre-compute building dataset for a single city.

This script:
 1. Reads a raw DPE CSV
 2. Renames key columns and adds derived metrics (Water_Usage, log-transforms)
 3. Runs classification algorithms (Mahalanobis, PCA, Weighted, Bayesian, Manhattan, TOPSIS)
 4. Saves the enriched data to Parquet and overwrites the original CSV (with backup)

Usage:
  python precompute_city_data.py \
    --city Lyon \
    --input raw/Lyon_dataset.csv \
    --output output/Lyon.parquet \
    --features log1p_CO2_Usage Water_Usage log1p_Energy_Consumption \
    --weights 0.5 0.2 0.3
"""
import argparse
import pathlib
import pandas as pd
import numpy as np
import shutil

# ──────────────────────────────────────────────────────────────
# 1 ▸ Parse CLI arguments
# ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Pre-compute building dataset for one city."
)
parser.add_argument("--city", required=True, help="City name (for logging)")
parser.add_argument(
    "--input", required=True, help="Path to raw CSV file (will be overwritten)"
)
parser.add_argument(
    "--output", required=True, help="Path to write Parquet file"
)
parser.add_argument(
    "--features", nargs='+', required=True,
    help="List of feature columns to use for classification"
)
parser.add_argument(
    "--weights", nargs='+', type=float, default=None,
    help=("Optional weights for weighted classifier; must match length of --features. "
          "If omitted, equal weights are used.")
)
args = parser.parse_args()

city = args.city
RAW = pathlib.Path(args.input).expanduser()
DEST = pathlib.Path(args.output).expanduser()
DEST.parent.mkdir(parents=True, exist_ok=True)

print(f"🔍 Reading file: {RAW}")

# ──────────────────────────────────────────────────────────────
# 2 ▸ Backup original CSV
# ──────────────────────────────────────────────────────────────
backup_path = RAW.with_suffix(RAW.suffix + ".orig_backup")
if not backup_path.exists():
    shutil.copy2(RAW, backup_path)
    print(f"📦 Backed up original CSV to {backup_path}")

# ──────────────────────────────────────────────────────────────
# 3 ▸ Load raw data
# ──────────────────────────────────────────────────────────────
df = pd.read_csv(RAW, dtype=str)
print("🔍 Original columns:", df.columns.tolist())

# Rename raw DPE columns to standardized names
if "building_id" not in df.columns and "numero_dpe" in df.columns:
    df = df.rename(columns={
        "numero_dpe": "building_id",
        "conso_5 usages_ef": "Energy_Consumption",
        "emission_ges_5_usages": "CO2_Usage",
        "conso_5 usages_par_m2_ef": "Energy_Intensity",
        "emission_ges_5_usages par_m2": "CO2_Intensity",
    })

# Ensure id is string
df['building_id'] = df['building_id'].astype(str)

# Add Water_Usage if missing
if 'Water_Usage' not in df.columns and 'Energy_Consumption' in df.columns:
    df['Water_Usage'] = df['Energy_Consumption'].astype(float) * 0.30

# Convert numeric columns to float for transforms
numeric_cols = ['Energy_Consumption','CO2_Usage','Energy_Intensity','CO2_Intensity','Water_Usage']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Add log1p-transformed features
log_targets = ['Energy_Consumption','CO2_Usage','Energy_Intensity','CO2_Intensity']
for col in log_targets:
    df[f'log1p_{col}'] = np.log1p(df[col].fillna(0))

# ──────────────────────────────────────────────────────────────
# 4 ▸ Run classification algorithms
# ──────────────────────────────────────────────────────────────
from app.models.weighted   import classify_weighted
from app.models.pca        import classify_pca
from app.models.bayesian   import classify_bayesian
from app.models.mahalanobis import classify_mahalanobis
from app.models.manhattan  import classify_manhattan
from app.models.topsis     import classify_topsis

features = args.features
weights = args.weights
if weights is not None and len(weights) != len(features):
    raise ValueError(
        f"--weights length {len(weights)} does not match --features length {len(features)}"
    )
# Default equal weights if none provided
if weights is None:
    weights = [1.0] * len(features)

print("🔍 Classifying using features:", features)

# Mahalanobis
df_mah = classify_mahalanobis(
    df, features=features, return_distance=True
)
df['class_mahalanobis']     = df_mah['class_label']
df['Mahalanobis_Distance']  = df_mah['Mahalanobis_Distance']

# PCA
df['class_pca'] = classify_pca(
    df, features=features
)['class_label']

# Weighted
df['class_weighted'] = classify_weighted(
    df, features=features, weights=weights
)['class_label']

# Bayesian
df['class_bayesian'] = classify_bayesian(
    df, features=features
)['class_label']

# Manhattan
df['class_manhattan'] = classify_manhattan(
    df, features=features
)['class_label']

# TOPSIS
df = classify_topsis(
    df,
    features=['Energy_Consumption', 'CO2_Usage', 'Water_Usage'],
    weights=[0.5, 0.3, 0.2],        # importance
    benefit=[False, False, False],   # lower-is-better metrics
    n_classes=6                      # A-F
)

# “topsis_score”  — continuous 0-1
# “class_label”   — balanced A–F

# Cleanup any stray column_label
if 'class_label' in df.columns:
    df.drop(columns=['class_label'], inplace=True)

print("✅ Classification complete!")
print("🧾 Final columns:", df.columns.tolist())

# ──────────────────────────────────────────────────────────────
# 5 ▸ Save outputs
# ──────────────────────────────────────────────────────────────
# Parquet
DEST = DEST.with_suffix('.parquet')
df.to_parquet(DEST, index=False)
print(f"💾 {city.title()}: saved {len(df):,} rows to {DEST}")

# Overwrite original CSV with enriched data
print(f"📝 Overwriting original CSV: {RAW}")
df.to_csv(RAW, index=False)
print("✅ CSV updated.")