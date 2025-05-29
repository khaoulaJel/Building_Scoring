import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

def classify_cosine(
    df: pd.DataFrame,
    features: list,
    class_labels: list = None
) -> pd.DataFrame:
    """
    Cosine similarity–based classification into N classes based
    on similarity to the 'ideal' (origin) building profile.
    
    Args:
        df: Input dataframe
        features: List of feature columns to use
        class_labels: Optional list of class labels (default A-F)
        
    Returns:
        DataFrame with added classification columns:
        - class_cosine: The cosine similarity-based class
        - class_label: Copy of class_cosine for consistency
    """
    df = df.copy()
    if class_labels is None:
        class_labels = ['A','B','C','D','E','F']
    n_classes = len(class_labels)

    # 1. Normalize features to [0,1]
    scaler = MinMaxScaler()
    norm_features = scaler.fit_transform(df[features])
    df_norm = pd.DataFrame(norm_features, columns=[f"{col}_norm" for col in features])

    # 2. Compute cosine similarity with ideal (all-zero) point
    ideal = np.zeros((1, len(features)))
    similarities = cosine_similarity(df_norm.values, ideal).flatten()

    # 3. Convert to "distance" (1 - similarity)
    df['distance'] = 1 - similarities

    # 4. Try to assign class labels using qcut
    try:
        df['class_cosine'] = pd.qcut(
            df['distance'],
            q=n_classes,
            labels=class_labels,
            duplicates='drop'
        )
    except ValueError:
        # Fallback to rank-based classification
        df['rank'] = df['distance'].rank(method='first')
        bin_edges = np.linspace(0, len(df), n_classes + 1).astype(int)
        df['bin'] = pd.cut(df['rank'], bins=bin_edges, labels=False, include_lowest=True)
        df['class_cosine'] = df['bin'].apply(lambda i: class_labels[i])
        df.drop(columns=['rank', 'bin'], inplace=True)

    # Add class_label for consistency with other classifiers
    df['class_label'] = df['class_cosine']

    # Final cleanup - remove intermediate calculations
    df.drop(columns=['distance'], inplace=True)

    return df
