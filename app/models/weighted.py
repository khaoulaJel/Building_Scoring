import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def classify_weighted(
    df: pd.DataFrame,
    features: list,
    weights: list,
    class_labels: list = None
) -> pd.DataFrame:
    """
    Weighted-sum classification over any number of features.
    You supply one weight per feature; lower scores → better class.

    Args:
        df (pd.DataFrame): Input data
        features (list of str): Numeric feature column names
        weights (list of float): Weights for each feature (must match len(features))
        class_labels (list of str, optional): Labels for each bin,
            ordered from best (smallest score) to worst. Defaults to ['A','B','C','D','E','F'].

    Returns:
        pd.DataFrame: Copy of df with two new columns:
          - "Global_Score"    : The weighted sum of normalized features
          - "class_label"     : The quantile-based class label
    """
    df = df.copy()

    # 1) Validate inputs
    n = len(features)
    if len(weights) != n:
        raise ValueError(f"Expected {n} weights, got {len(weights)}")
    if class_labels is None:
        class_labels = ['A','B','C','D','E','F']
    m = len(class_labels)

    # 2) Impute missing & normalize each feature to [0,1]
    df[features] = df[features].fillna(df[features].mean())
    scaler = MinMaxScaler()
    norm_cols = [f"{feat}_norm" for feat in features]
    df[norm_cols] = scaler.fit_transform(df[features])

    # 3) Normalize the weights so they sum to 1
    w = np.array(weights, dtype=float)
    w = w / w.sum()

    # 4) Compute the weighted score
    df["Global_Score"] = df[norm_cols].values.dot(w)

    # 5) Bin into quantiles → class_labels
    df["class_label"] = pd.qcut(
        df["Global_Score"],
        q=m,
        labels=class_labels
    ).astype(str)

    # 6) Clean up (keep Global_Score if you like)
    df = df.drop(columns=norm_cols, errors='ignore')

    return df
