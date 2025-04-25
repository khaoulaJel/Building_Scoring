import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis

def classify_mahalanobis(
    df: pd.DataFrame,
    features: list,
    class_labels: list = None,
    return_distance: bool = False
) -> pd.DataFrame:
    """
    Mahalanobis-distance-based classification using a customizable feature list.
    Produces 'class_label' (A–F by default), and optionally 'Mahalanobis_Distance'.

    Args:
        df (pd.DataFrame): Input dataframe
        features (list of str): Numeric feature column names
        class_labels (list of str, optional): Labels for each distance-bin,
            ordered from “closest” to “farthest.” Defaults to ['A','B','C','D','E','F'].
        return_distance (bool): If True, keeps the 'Mahalanobis_Distance' column

    Returns:
        pd.DataFrame: Copy of df with:
          - 'class_label' for each row
          - optionally 'Mahalanobis_Distance'
    """
    df = df.copy()
    
    # Default to 6 classes A–F if not provided
    if class_labels is None:
        class_labels = ['A','B','C','D','E','F']
    n_classes = len(class_labels)

    # 1. Compute the “optimal” reference (min on each feature)
    optimal_point = df[features].min().values

    # 2. Compute covariance & its inverse
    cov_matrix = df[features].cov().values
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    # 3. Mahalanobis distance for each row
    df['Mahalanobis_Distance'] = df[features].apply(
        lambda row: mahalanobis(row.values, optimal_point, inv_cov_matrix),
        axis=1
    )

    # 4. Bin distances into n_classes equal-width intervals
    df['bin'] = pd.cut(
        df['Mahalanobis_Distance'],
        bins=n_classes,
        labels=class_labels
    )

    # 5. Final class_label
    df['class_label'] = df['bin'].astype(str)

    # 6. Cleanup
    if return_distance:
        df = df.drop(columns=['bin'])
    else:
        df = df.drop(columns=['bin', 'Mahalanobis_Distance'])

    return df
