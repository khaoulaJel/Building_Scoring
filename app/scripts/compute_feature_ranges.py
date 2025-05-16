import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def compute_ranges(df: pd.DataFrame, features: List[str], class_col: str, mode: str) -> pd.DataFrame:
    """Return an aggregated DataFrame for one class column."""
    if mode == "minmax":
        agg = df.groupby(class_col)[features].agg(["min", "max"])
        agg.columns = [f"{feat}_{stat}" for feat, stat in agg.columns]
    elif mode == "meanstd":
        agg = df.groupby(class_col)[features].agg(["min", "mean", "max", "std"])
        agg.columns = [f"{feat}_{stat}" for feat, stat in agg.columns]
    else:
        raise ValueError("mode must be 'minmax' or 'meanstd'")
    return agg.reset_index()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def _compute(df: pd.DataFrame, features: List[str], class_col: str, agg_mode: str) -> pd.DataFrame:
    """Compute statistical measures for each class and feature."""
    return compute_ranges(df, features, class_col, agg_mode)


def main() -> None:
    p = argparse.ArgumentParser("Compute feature ranges (or mean/std) by class label")
    p.add_argument("--input", "-i", required=True, help="CSV or Parquet file with building data")
    p.add_argument("--class-col", "-c", help="Specific class column; if omitted auto-detect all 'class_*'")
    p.add_argument("--features", "-f", nargs="+", required=True, help="Numeric feature columns")
    p.add_argument("--output", "-o", help="Output CSV file or pattern. If multiple class cols, we'll derive names.")
    p.add_argument("--agg", choices=["minmax", "meanstd"], default="minmax", help="Aggregation type")
    args = p.parse_args()

    path = Path(args.input)
    if not path.exists():
        sys.exit(f"Error: {path} not found")

    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    # Validate features present & numeric
    missing = [f for f in args.features if f not in df.columns]
    if missing:
        sys.exit(f"Missing features: {missing}")
    df[args.features] = df[args.features].apply(pd.to_numeric, errors="coerce")

    # Determine which class columns to process
    if args.class_col:
        class_cols = [args.class_col]
        if args.class_col not in df.columns:
            sys.exit(f"Class column '{args.class_col}' not found in data")
    else:
        class_cols = [c for c in df.columns if c.startswith("class_")]
        if not class_cols:
            sys.exit("No class_* columns found; specify --class-col explicitly")

    # Output pattern handling
    if args.output:
        out_path = Path(args.output)
        if len(class_cols) > 1 and "{col}" not in str(out_path):
            # allow pattern like ranges_{col}.csv
            print("✱ Multiple class columns detected; appending column name to output file.")
            out_pattern = out_path.with_stem(out_path.stem + "_{col}")
        else:
            out_pattern = out_path
    else:
        out_pattern = path.with_stem(path.stem + "_{col}_ranges")
        out_pattern = out_pattern.with_suffix(".csv")

    # Compute + save per class column
    for c in class_cols:
        out_df = _compute(df, args.features, c, args.agg)
        dest = Path(str(out_pattern).format(col=c))
        out_df.to_csv(dest, index=False)
        print(f" Saved ranges for {c} → {dest}")


if __name__ == "__main__":
    main()