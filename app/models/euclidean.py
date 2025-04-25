import pandas as pd
import numpy as np

def classify_euclidean(
    df: pd.DataFrame,
    features: list,
    class_labels: list = None
) -> pd.DataFrame:
    """
    Euclidean distance–based classification into N classes based
    on any number of features.

    Args:
        df (pd.DataFrame): Input dataframe.
        features (list of str): Names of numeric feature columns.
        class_labels (list of str, optional): Labels for each class,
            in order from “best” (smallest distance) to “worst”.
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

    # 2. Compute Euclidean distance to the origin in N-D
    df['distance'] = np.sqrt(
        sum(df[n]**2 for n in norm_cols)
    )

    # 3. Create N equal-width bins over the range of distances
    edges = np.linspace(df['distance'].min(),
                        df['distance'].max(),
                        n_classes + 1)
    # digitize into 0…n_classes-1
    df['bin'] = np.digitize(df['distance'], edges[1:], right=False)
    df['bin'] = df['bin'].clip(0, n_classes-1)

    # 4. Map bins → labels
    df['class_label'] = df['bin'].apply(lambda i: class_labels[i])

    # 5. Clean up
    df = df.drop(columns=norm_cols + ['distance','bin'])
    return df
