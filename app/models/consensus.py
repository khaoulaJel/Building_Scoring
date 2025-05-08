# consensus.py  — memory‑safe version for large datasets

"""Ensemble consensus clustering with safeguards against memory blow‑ups.

Changes vs. previous version:
• Base algorithms:
  – *MiniBatchKMeans* (fast)
  – *GaussianMixture* (diagonal covariance)
  – *Birch* (hierarchical but O(n))
  Agglomerative (Ward) is kept **only** if the dataset is small (< 10 000 rows).
• Consensus step still builds a co‑association matrix on a random subset
  (default 5 000) — that fits in RAM (~100 MB).
• Labels for remaining rows are assigned by nearest consensus cluster centroid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans, Birch, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import sklearn
from packaging import version
from typing import List, Optional


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _agglo_consensus(n_clusters: int, dist: np.ndarray) -> np.ndarray:
    kw = dict(linkage="average", n_clusters=n_clusters)
    if version.parse(sklearn.__version__) >= version.parse("1.4"):
        kw["metric"] = "precomputed"
    else:
        kw["affinity"] = "precomputed"
    return AgglomerativeClustering(**kw).fit_predict(dist)


# ──────────────────────────────────────────────────────────────
# Main API
# ──────────────────────────────────────────────────────────────

def classify_consensus(
    df: pd.DataFrame,
    features: List[str],
    n_clusters: int = 6,
    class_labels: Optional[List[str]] = None,
    random_state: int = 42,
    sample_size: int = 5000,
) -> pd.DataFrame:
    if class_labels is None:
        class_labels = list("ABCDEF")[:n_clusters]
    if len(class_labels) != n_clusters:
        raise ValueError("class_labels length must equal n_clusters")

    # ── 1 ▸ Standardise
    X = df[features].fillna(df[features].mean()).values
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    n = X_std.shape[0]

    # ── 2 ▸ Base cluster labels (fast/light algorithms)

    base_labels = []
    base_labels.append(
        MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=4096)
        .fit_predict(X_std)
    )
    base_labels.append(
        GaussianMixture(
            n_components=n_clusters,
            covariance_type="diag",
            random_state=random_state,
            max_iter=200,
        ).fit_predict(X_std)
    )

    if n < 10_000:
        # Ward only when safe
        base_labels.append(
            AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(X_std)
        )
    else:
        # Birch as lightweight hierarchical stand‑in
        base_labels.append(
            Birch(n_clusters=n_clusters).fit_predict(X_std)
        )

    label_arr = np.stack(base_labels)  # shape (n_algs, n)
    n_algs = label_arr.shape[0]

    # ── 3 ▸ Pick subset for co‑association if necessary
    if n <= sample_size:
        subset_idx = np.arange(n)
    else:
        rng = np.random.default_rng(random_state)
        subset_idx = rng.choice(n, size=sample_size, replace=False)

    # co‑association
    coassoc = np.zeros((subset_idx.size, subset_idx.size), dtype=np.float32)
    for lbl in label_arr:
        s_lbl = lbl[subset_idx]
        for cid in np.unique(s_lbl):
            idx = np.where(s_lbl == cid)[0]
            coassoc[np.ix_(idx, idx)] += 1
    coassoc /= n_algs
    dist = 1.0 - coassoc

    sub_cons = _agglo_consensus(n_clusters, dist)

    # ── 4 ▸ Propagate
    final = np.full(n, -1, dtype=int)
    final[subset_idx] = sub_cons
    if n > sample_size:
        centroids = np.vstack([
            X_std[subset_idx][sub_cons == k].mean(axis=0) for k in range(n_clusters)
        ])
        rest = np.setdiff1d(np.arange(n), subset_idx)
        d2 = ((X_std[rest, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        final[rest] = d2.argmin(axis=1)

    # ── 5 ▸ Map clusters → classes (rank by energy consumption)
    out = df.copy()
    out["_cl"] = final
    key_col = "Energy_Consumption" if "Energy_Consumption" in out.columns else features[0]
    rank = out.groupby("_cl")[key_col].mean().sort_values().reset_index()
    cid2rank = dict(zip(rank["_cl"], range(n_clusters)))
    out["class_label"] = out["_cl"].map(lambda cid: class_labels[cid2rank[cid]])

    return out.drop(columns="_cl")
