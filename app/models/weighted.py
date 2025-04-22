import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
def classify_weighted(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Weighted classification of any feature list.
    Lower 'Global_Score' => better class (A)

    Args:
        df (pd.DataFrame): The dataset
        features (list): List of 3 features to use (e.g., log1p_...)

    Returns:
        pd.DataFrame: With 'class_label' and 'class_weighted' columns added
    """
    assert len(features) == 3, "Expected exactly 3 features"

    df = df.copy()
    
    # Handle NaN values
    df[features] = df[features].fillna(df[features].mean())

    scaler = MinMaxScaler()
    norm_cols = [f"{col}_norm" for col in features]
    df[norm_cols] = scaler.fit_transform(df[features])

    # Define weights
    weights = np.array([0.2, 0.3, 0.5])
    df["Global_Score"] = np.sum(df[norm_cols] * weights, axis=1)

    percentiles = df["Global_Score"].quantile([0.1, 0.3, 0.6, 0.8, 0.9])

    def classify_score(score):
        if score <= percentiles[0.1]: return "A"
        elif score <= percentiles[0.3]: return "B"
        elif score <= percentiles[0.6]: return "C"
        elif score <= percentiles[0.8]: return "D"
        elif score <= percentiles[0.9]: return "E"
        else: return "F"

    df["class_weighted"] = df["Global_Score"].apply(classify_score)
    df["class_label"] = df["class_weighted"]

    # Clean up temporary columns
    df = df.drop(columns=norm_cols + ["Global_Score"], errors='ignore')

    return df
