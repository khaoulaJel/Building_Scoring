import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def classify_pca(
    df: pd.DataFrame,
    features: list,
    class_labels: list = None
) -> pd.DataFrame:
    """
    PCA-based classification where lower PC1 => better performance.
    Works over any list of numeric features and any number of target classes.

    Args:
        df (pd.DataFrame): Input dataframe
        features (list of str): Feature column names (any length ≥1).
        class_labels (list of str, optional): Labels for each quantile-bin,
            ordered from best (smallest PC1) to worst. Defaults to ['A','B','C','D','E','F'].

    Returns:
        pd.DataFrame: Copy of df with a new 'class_label' column.
    """
    df = df.copy()
    
    # 1) Default to 6 classes A–F if none provided
    if class_labels is None:
        class_labels = ['A','B','C','D','E','F']
    n_classes = len(class_labels)

    # 2) Impute any NaNs with feature means
    df[features] = df[features].fillna(df[features].mean())

    # 3) Normalize each feature into [0,1]
    norm_cols = [f"{feat}_norm" for feat in features]
    scaler = MinMaxScaler()
    df[norm_cols] = scaler.fit_transform(df[features])

    # 4) Run PCA → single principal component (PC1)
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(df[norm_cols]).flatten()

    # 5) Flip sign so that higher PC1 always means worse performance
    corr = np.corrcoef(pc1, df[norm_cols[0]])[0,1]
    if corr < 0:
        pc1 = -pc1
    df["PC1"] = pc1

    # 6) Quantile‐bin PC1 into n_classes bins
    df["class_label"] = pd.qcut(
        df["PC1"],
        q=n_classes,
        labels=class_labels
    )

    # 7) Clean up temporary columns
    df = df.drop(columns=norm_cols + ["PC1"], errors='ignore')
    return df
