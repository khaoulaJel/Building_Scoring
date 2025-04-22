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
        pd.DataFrame: With 'class_label' column added
    """
    df = df.copy()
    assert len(features) == 3, "Expected exactly 3 features"

    scaler = MinMaxScaler()
    norm_cols = [f"{col}_norm" for col in features]
    df[norm_cols] = scaler.fit_transform(df[features])

    # Optional: you can define weights based on feature names if needed
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

    df["class_label"] = df["Global_Score"].apply(classify_score)
    return df
