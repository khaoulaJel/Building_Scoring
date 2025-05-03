import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer


def classify_pca(
    df: pd.DataFrame,
    features: list,
    target_feature: str = None,            # Feature that defines “good” performance
    class_labels: list | None = None,
    return_details: bool = False
) -> pd.DataFrame:
    """
    PCA-based classification for buildings, with robust scaling,
    KNN-imputation and automatic orientation of the first component.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    features : list[str]
        Numeric feature columns to include in the PCA.
    target_feature : str | None, default first feature
        Feature whose *lower* value indicates better performance.
    class_labels : list[str] | None
        Labels for classes from best → worst. Defaults to ['A' …].
    return_details : bool, default False
        If True, returns the DataFrame *with* PC columns and prints
        variance / loadings; otherwise PC columns are dropped.

    Returns
    -------
    pd.DataFrame
        DataFrame with a new column ``class_label``.
    """
    df = df.copy()

    # 1️⃣  default labels A-F
    if class_labels is None:
        class_labels = ['A', 'B', 'C', 'D', 'E', 'F']
    n_classes = len(class_labels)

    # 2️⃣  choose target feature
    if target_feature is None:
        target_feature = features[0]

    # 3️⃣  KNN impute ▸ Standard-scale
    imputed = KNNImputer(n_neighbors=5).fit_transform(df[features])
    X_scaled = StandardScaler().fit_transform(imputed)

    # 4️⃣  PCA (keep up to 3 PCs)
    pca = PCA(n_components=min(len(features), 3))
    pcs = pca.fit_transform(X_scaled)
    df['PC1'] = pcs[:, 0]
    if pcs.shape[1] > 1:
        df['PC2'] = pcs[:, 1]
    if pcs.shape[1] > 2:
        df['PC3'] = pcs[:, 2]

    # 5️⃣  orient PC1 so that *lower* is better
    corr = np.corrcoef(df['PC1'], imputed[:, features.index(target_feature)])[0, 1]
    if corr > 0:           # higher PC1 ↔ higher target (bad) → flip sign
        df['PC1'] = -df['PC1']

    # 6️⃣  equal-frequency binning (quantiles)
    try:
        df['class_label'] = pd.qcut(
            df['PC1'],
            q=n_classes,
            labels=class_labels,
            duplicates='drop'
        )
    except ValueError:     # too few unique values
        uniq = df['PC1'].nunique()
        df['class_label'] = pd.qcut(
            df['PC1'],
            q=min(uniq, n_classes),
            labels=class_labels[:uniq],
            duplicates='drop'
        )

    df['class_label'] = df['class_label'].astype(str)

    if return_details:
        print("Explained variance:", pca.explained_variance_ratio_)
        print("Cumulative var.:  ", np.cumsum(pca.explained_variance_ratio_))
        print("\nComponent loadings:")
        print(pd.DataFrame(pca.components_, columns=features))

        return df
    else:
        df.drop(columns=[c for c in ['PC1', 'PC2', 'PC3'] if c in df.columns],
                inplace=True)
        return df
