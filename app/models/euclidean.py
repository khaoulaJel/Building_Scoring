import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def classify_euclidean(
    df: pd.DataFrame,
    features: list,
    class_labels: list = None,
    binning: str = "quantile",      # "quantile" | "equal_width" | "kmeans"
    n_clusters: int = 6,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Classify buildings by Euclidean distance in feature space.

    Parameters
    ----------
    df : pd.DataFrame
    features : list[str]      – numeric columns to use
    class_labels : list[str]  – default ['A'..] length must match n_clusters
    binning : str             – "quantile" (≈ equal-frequency),
                                 "equal_width" (old behaviour),
                                 "kmeans" (distance clustering)
    n_clusters : int          – number of classes / clusters
    """
    df = df.copy()
    if class_labels is None:
        class_labels = list("ABCDEF")[:n_clusters]
    assert len(class_labels) == n_clusters, "class_labels length ≠ n_clusters"

    # 1) normalise 0-1 per feature (robust to zero-variance)
    norm_cols = []
    for feat in features:
        rng = df[feat].max() - df[feat].min()
        rng = rng if rng else 1  # avoid /0
        col = f"{feat}_norm"
        df[col] = (df[feat] - df[feat].min()) / rng
        norm_cols.append(col)

    # 2) Euclidean distance to origin
    df["_distance"] = np.sqrt(df[norm_cols].pow(2).sum(axis=1))

    # 3) Choose binning strategy
    if binning == "equal_width":
        edges = np.linspace(df["_distance"].min(),
                            df["_distance"].max(),
                            n_clusters + 1)
        # digitize returns 0…n_clusters-1
        bins = np.digitize(df["_distance"], edges[1:], right=False)

    elif binning == "quantile":
        # pd.qcut returns nearly equal counts; duplicates="drop" avoids errors
        bins, edges = pd.qcut(
            df["_distance"],
            q=n_clusters,
            labels=False,
            retbins=True,
            duplicates="drop"
        )
        # qcut can drop a bin if all obs identical; ensure we still map labels
        n_clusters = bins.max() + 1

    elif binning == "kmeans":
        km = KMeans(n_clusters=n_clusters,
                    n_init="auto",
                    random_state=random_state)
        bins = km.fit_predict(df[["_distance"]])
        # order clusters by centroid so that A=best
        order = np.argsort(km.cluster_centers_.ravel())
        rank  = {c: r for r, c in enumerate(order)}
        bins  = bins.map(rank)
    else:
        raise ValueError("binning must be 'equal_width', 'quantile', or 'kmeans'")

    # 4) Assign labels (invert so 0 = best class)
    df["class_label"] = bins.map(lambda i: class_labels[int(i)])

    # 5) tidy up
    return df.drop(columns=norm_cols + ["_distance"])
