import pandas as pd
import numpy as np

def classify_manhattan(
    df: pd.DataFrame,
    features: list,
    class_labels: list = None
) -> pd.DataFrame:
    """
    Manhattan distance–based classification into N classes based
    on any number of features.

    Args:
        df (pd.DataFrame): Input dataframe.
        features (list of str): Names of numeric feature columns.
        class_labels (list of str, optional): Labels for each class,
            in order from "best" (smallest distance) to "worst".
            If None, defaults to ['A','B','C','D','E','F'].

    Returns:
        pd.DataFrame: Copy of df with a new 'class_label' column.
    """
    df = df.copy()
    if class_labels is None:
        class_labels = ['A','B','C','D','E','F']
    n_classes = len(class_labels)

    # 1. Normalize each feature to [0,1]
    norm_cols = []
    for feat in features:
        norm_col = f"{feat}_norm"
        max_val = df[feat].max()
        # avoid division by zero
        df[norm_col] = df[feat] / (max_val if max_val != 0 else 1)
        norm_cols.append(norm_col)

    # 2. Compute Manhattan distance to the origin in N-D
    # Manhattan distance is the sum of absolute values of coordinates
    df['distance'] = df[norm_cols].abs().sum(axis=1)

    # 3. Create N equal-frequency bins (quantiles) instead of equal-width bins
    # This handles skewed data better than equal-width binning
    df['class_label'] = pd.qcut(
        df['distance'], 
        q=n_classes, 
        labels=class_labels,
        duplicates='drop'  # Handle case where there are duplicate quantile values
    )
    
    # Handle edge case where qcut fails due to too many duplicates
    if 'class_label' not in df.columns or df['class_label'].isna().any():
        # Fallback to rank-based approach
        df['rank'] = df['distance'].rank(method='first')
        bin_edges = np.linspace(0, len(df), n_classes + 1).astype(int)
        df['bin'] = pd.cut(df['rank'], bins=bin_edges, labels=False, include_lowest=True)
        df['class_label'] = df['bin'].apply(lambda i: class_labels[i])
        df = df.drop(columns=['rank', 'bin'])
    
    # 4. Clean up
    df = df.drop(columns=norm_cols + ['distance'])
    return df