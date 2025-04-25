#!/usr/bin/env python3
"""
compute_feature_ranges.py

Compute the min/max ranges of specified numeric features for each class label in a city dataset.

Usage:
  python compute_feature_ranges.py \
    --input path/to/enriched_city_data.csv \
    --class-col class_pca \
    --features Energy_Consumption CO2_Usage Water_Usage \
    --output path/to/ranges.csv
"""
import argparse
import pandas as pd
import sys

def compute_ranges(df: pd.DataFrame, features: list, class_col: str) -> pd.DataFrame:
    """
    Group by `class_col` and compute min/max for each feature.

    Returns a DataFrame with columns:
      - class_col
      - {feature}_min, {feature}_max for each feature
    """
    # Aggregate
    agg = df.groupby(class_col)[features].agg(['min', 'max'])
    # Flatten MultiIndex columns
    agg.columns = [f"{feat}_{stat}" for feat, stat in agg.columns]
    return agg.reset_index()


def main():
    parser = argparse.ArgumentParser(
        description="Compute feature ranges by class label"
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help="Path to input CSV or Parquet file containing building data with class labels"
    )
    parser.add_argument(
        '--class-col', '-c',
        required=True,
        help="Name of the column containing class labels"
    )
    parser.add_argument(
        '--features', '-f',
        nargs='+',
        required=True,
        help="List of numeric feature columns to compute ranges for"
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help="Path to output CSV file for saving the ranges"
    )
    args = parser.parse_args()

    # Load data
    if args.input.lower().endswith('.parquet'):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)

    # Validate columns
    missing_feats = [f for f in args.features if f not in df.columns]
    if missing_feats:
        print(f"Error: Missing features in data: {missing_feats}", file=sys.stderr)
        sys.exit(1)
    if args.class_col not in df.columns:
        print(f"Error: Class column '{args.class_col}' not found in data", file=sys.stderr)
        sys.exit(1)

    # Ensure numeric
    df[args.features] = df[args.features].apply(pd.to_numeric, errors='coerce')

    # Compute ranges
    ranges_df = compute_ranges(df, args.features, args.class_col)

    # Save
    ranges_df.to_csv(args.output, index=False)
    print(f"✅ Feature ranges by class saved to {args.output}")


if __name__ == '__main__':
    main()
