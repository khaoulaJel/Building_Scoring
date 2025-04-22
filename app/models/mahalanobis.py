import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis

def classify_mahalanobis(df: pd.DataFrame, features: list, return_distance: bool = False) -> pd.DataFrame:
    """
    Mahalanobis-distance-based classification using customizable feature list.
    Produces 'class_label' (A–F). Optionally returns Mahalanobis_Distance.

    Args:
        df (pd.DataFrame): Input dataframe
        features (list): List of 3 numerical features
        return_distance (bool): If True, includes Mahalanobis_Distance column
    
    Returns:
        pd.DataFrame: With 'class_label' and optional Mahalanobis_Distance
    """
    assert len(features) == 3, "You must provide exactly 3 features."

    df = df.copy()
    
    # Step 1: Mahalanobis distance to optimal (min values)
    optimal_point = df[features].min().values
    cov_matrix = df[features].cov().values
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    def mahalanobis_distance(row_vector, optimal_pt, inv_cov):
        return mahalanobis(row_vector, optimal_pt, inv_cov)

    df["Mahalanobis_Distance"] = df[features].apply(
        lambda row: mahalanobis_distance(row.values, optimal_point, inv_cov_matrix),
        axis=1
    )

    # Step 2: Convert distance to A–F class
    df["Global_Class"] = pd.cut(
        df["Mahalanobis_Distance"],
        bins=6,
        labels=["A", "B", "C", "D", "E", "F"]
    ).astype(str)

    # Step 3: Return only what's needed
    df["class_label"] = df["Global_Class"]

    if not return_distance:
        df = df.drop(columns=["Mahalanobis_Distance", "Global_Class"])

    return df
