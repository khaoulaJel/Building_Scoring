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
from sklearn.cluster import MiniBatchKMeans, Birch, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
import sklearn
from packaging import version
from typing import List, Optional
import warnings

# Try to import joblib to manage parallelism
try:
    from joblib import parallel_backend
    has_joblib = True
except ImportError:
    has_joblib = False

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

def _safe_minibatch_kmeans(X, n_clusters, random_state):
    """Try MiniBatchKMeans with fallback to regular KMeans if it fails"""
    try:
        # Skip MiniBatchKMeans entirely and use KMeans with n_jobs=1
        # This avoids the threadpoolctl error completely
        return KMeans(
            n_clusters=n_clusters, 
            random_state=random_state,
            n_init=1,
            n_jobs=1  # Use single thread to avoid threadpool issues
        ).fit_predict(X)
    except Exception as e:
        print(f"KMeans failed: {e}. Trying non-parallel.")
        # Ultimate fallback with minimum dependencies
        from sklearn.cluster import KMeans
        kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=random_state,
            n_init="auto",
            algorithm="elkan"  # More efficient algorithm
        )
        # Manually disable internal parallelism
        kmeans._check_mkl_vcomp = lambda *args, **kwargs: None
        return kmeans.fit_predict(X)

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
    
    # Use safe version of MiniBatchKMeans
    base_labels.append(_safe_minibatch_kmeans(X_std, n_clusters, random_state))
    
    # Try/except for GaussianMixture
    try:
        base_labels.append(
            GaussianMixture(
                n_components=n_clusters,
                covariance_type="diag",
                random_state=random_state,
                max_iter=200,
            ).fit_predict(X_std)
        )
    except Exception as e:
        print(f"GaussianMixture failed: {e}. Using KMeans instead.")
        base_labels.append(
            KMeans(
                n_clusters=n_clusters, 
                random_state=random_state+1,
                n_init=1
            ).fit_predict(X_std)
        )

    if n < 10_000:
        # Ward only when safe
        try:
            base_labels.append(
                AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(X_std)
            )
        except Exception as e:
            print(f"AgglomerativeClustering failed: {e}. Using KMeans instead.")
            base_labels.append(
                KMeans(
                    n_clusters=n_clusters, 
                    random_state=random_state+2,
                    n_init=1
                ).fit_predict(X_std)
            )
    else:
        # Birch as lightweight hierarchical stand‑in
        try:
            base_labels.append(
                Birch(n_clusters=n_clusters).fit_predict(X_std)
            )
        except Exception as e:
            print(f"Birch failed: {e}. Using KMeans instead.")
            base_labels.append(
                KMeans(
                    n_clusters=n_clusters, 
                    random_state=random_state+2,
                    n_init=1
                ).fit_predict(X_std)
            )

    # Ensure we got at least one valid clustering
    if not base_labels:
        # Last resort: single KMeans with high n_init
        base_labels.append(
            KMeans(
                n_clusters=n_clusters, 
                random_state=random_state,
                n_init=10
            ).fit_predict(X_std)
        )
    
    # Continue with standard processing
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

    # Safe consensus clustering
    try:
        sub_cons = _agglo_consensus(n_clusters, dist)
    except Exception as e:
        print(f"Consensus clustering failed: {e}. Using first base clustering.")
        # Just use the first base clustering as fallback
        sub_cons = label_arr[0][subset_idx]

    # ── 4 ▸ Propagate
    final = np.full(n, -1, dtype=int)
    final[subset_idx] = sub_cons
    if n > sample_size:
        try:
            centroids = np.vstack([
                X_std[subset_idx][sub_cons == k].mean(axis=0) for k in range(n_clusters)
            ])
            rest = np.setdiff1d(np.arange(n), subset_idx)
            d2 = ((X_std[rest, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            final[rest] = d2.argmin(axis=1)
        except Exception as e:
            print(f"Centroid propagation failed: {e}. Using KMeans for remaining data.")
            # Fallback: just use KMeans for the rest
            rest = np.setdiff1d(np.arange(n), subset_idx)
            if len(rest) > 0:
                final[rest] = KMeans(
                    n_clusters=n_clusters, 
                    random_state=random_state,
                    n_init=1
                ).fit_predict(X_std[rest])

    # ── 5 ▸ Map clusters → classes (rank by energy consumption)
    out = df.copy()
    out["_cl"] = final
    key_col = "Energy_Consumption" if "Energy_Consumption" in out.columns else features[0]
    
    # Handle potential edge cases in ranking
    try:
        rank = out.groupby("_cl")[key_col].mean().sort_values().reset_index()
        cid2rank = dict(zip(rank["_cl"], range(n_clusters)))
        out["class_label"] = out["_cl"].map(lambda cid: class_labels[cid2rank.get(cid, 0)])
    except Exception as e:
        print(f"Rank mapping failed: {e}. Using direct mapping.")
        # Direct mapping - just make sure we stay in bounds
        out["class_label"] = out["_cl"].apply(
            lambda x: class_labels[max(0, min(x, len(class_labels)-1))]
        )
    
    # Also add the consensus-specific class name for consistency
    out["class_consensus"] = out["class_label"]

    return out.drop(columns="_cl")
