import pandas as pd
import numpy as np

def classify_euclidean(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Euclidean distance-based classification using customizable 3 features.
    Produces 'class_label' column with values A–F.

    Args:
        df (pd.DataFrame): Input dataframe
        features (list): List of 3 feature column names (e.g. log1p_CO2_Usage, Water_Usage, log1p_Energy_Consumption)

    Returns:
        pd.DataFrame: Modified dataframe with classification added
    """
    assert len(features) == 3, "You must provide exactly 3 features."

    df = df.copy()

    # Normalize each feature
    norm_cols = [f"{col}_norm" for col in features]
    for i, col in enumerate(features):
        df[norm_cols[i]] = df[col] / df[col].max()

    # Compute Euclidean distance to origin (0,0,0)
    df['distance'] = np.sqrt(sum(df[n]**2 for n in norm_cols))

    # Discretize into 6 equal-width bins
    thresholds = np.linspace(df['distance'].min(), df['distance'].max(), 7)
    df['class'] = np.digitize(df['distance'], thresholds[1:], right=True)

    class_labels = ['A', 'B', 'C', 'D', 'E', 'F']
    df['class_label'] = df['class'].apply(lambda x: class_labels[min(x, len(class_labels)-1)])

    return df
