import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def classify_pca(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    PCA-based classification where lower PC1 => better performance.
    Accepts any list of 3 features (original or log-transformed).
    
    Args:
        df (pd.DataFrame): Input dataframe
        features (list): List of 3 features (e.g., log1p_CO2_Usage, Water_Usage, log1p_Energy_Consumption)
    
    Returns:
        pd.DataFrame: Modified dataframe with classification results
    """
    assert len(features) == 3, "You must provide exactly 3 features."

    df = df.copy()
    
    # Handle NaN values
    df[features] = df[features].fillna(df[features].mean())
    
    scaler = MinMaxScaler()
    norm_cols = [f"{col}_norm" for col in features]
    df[norm_cols] = scaler.fit_transform(df[features])

    pca = PCA(n_components=1)
    df["PC1"] = pca.fit_transform(df[norm_cols])

    # Flip sign if negatively correlated with the first feature
    corr_sign = np.corrcoef(df["PC1"], df[norm_cols[0]])[0, 1]
    if corr_sign < 0:
        df["PC1"] = -df["PC1"]

    percentiles = df["PC1"].quantile([0.1, 0.3, 0.6, 0.8, 0.9])

    def classify(score):
        if score <= percentiles[0.1]: return "A"
        elif score <= percentiles[0.3]: return "B"
        elif score <= percentiles[0.6]: return "C"
        elif score <= percentiles[0.8]: return "D"
        elif score <= percentiles[0.9]: return "E"
        else: return "F"

    df["class_pca"] = df["PC1"].apply(classify)
    df["class_label"] = df["class_pca"]

    # Clean up temporary columns
    df = df.drop(columns=norm_cols + ["PC1"], errors='ignore')

    return df
