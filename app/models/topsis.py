# topsis.py
"""Multi‑criteria building scoring via TOPSIS (Technique for Order Preference
by Similarity to Ideal Solution).

Returns a DataFrame with two new columns:
• `topsis_score`  – continuous 0–1 value (higher = better)
• `class_label`   – A–F derived from quantile cut‑points (equal frequency)

Example
-------
>>> from models.topsis import classify_topsis
>>> df = classify_topsis(df, features=[
        'Energy_Consumption', 'CO2_Usage', 'Water_Usage'],
        weights=[0.5,0.3,0.2],
        benefit=[False, False, False])
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import List, Optional


def _topsis(X: np.ndarray, weights: np.ndarray, benefit: np.ndarray) -> np.ndarray:
    """Raw TOPSIS implementation returning closeness coefficient (0–1)."""
    # 1 ▸ Normalise (vector normalisation)
    norm = X / np.linalg.norm(X, axis=0, keepdims=True)
    # 2 ▸ Weight
    V = norm * weights
    # 3 ▸ Ideal & anti‑ideal
    ideal = np.where(benefit, V.max(axis=0), V.min(axis=0))
    nadir = np.where(benefit, V.min(axis=0), V.max(axis=0))
    # 4 ▸ Distances
    d_pos = np.linalg.norm(V - ideal, axis=1)
    d_neg = np.linalg.norm(V - nadir, axis=1)
    # 5 ▸ Closeness coefficient
    return d_neg / (d_pos + d_neg + 1e-12)


def classify_topsis(
    df: pd.DataFrame,
    features: List[str],
    weights: Optional[List[float]] = None,
    benefit: Optional[List[bool]] = None,
    n_classes: int = 6,
    class_labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Add TOPSIS score + A–F class labels to *df*.

    Parameters
    ----------
    features : list
        Names of numeric columns.
    weights : list or None
        Importance weights for each feature; will be normalised to sum=1.
        If None ➜ equal weights.
    benefit : list[bool] or None
        True if the feature is *beneficial* (higher is better), False if
        *cost* (lower is better). If None ➜ all treated as cost.
    n_classes : int
        Number of bins (default 6).
    class_labels : list[str] or None
        Ordered labels best→worst. Defaults to A–F.
    """
    if class_labels is None:
        class_labels = list("ABCDEF")[:n_classes]
    if len(class_labels) != n_classes:
        raise ValueError("class_labels length must equal n_classes")

    X = df[features].values.astype(float)

    # Impute missing with column medians
    col_meds = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_meds, inds[1])

    # Normalise features 0–1 so weights are meaningful
    X = MinMaxScaler().fit_transform(X)

    m = len(features)
    if weights is None:
        weights = np.full(m, 1 / m)
    else:
        weights = np.array(weights, dtype=float)
        if weights.size != m:
            raise ValueError("weights length must match features")
        weights = weights / weights.sum()

    if benefit is None:
        benefit = np.zeros(m, dtype=bool)
    else:
        benefit = np.array(benefit, dtype=bool)
        if benefit.size != m:
            raise ValueError("benefit length must match features")

    score = _topsis(X, weights, benefit)

    out = df.copy()
    out["topsis_score"] = score
    # bin into equal‑frequency quantiles (balanced classes)
    quantiles = np.linspace(0, 1, n_classes + 1)
    bins = np.quantile(score, quantiles)
    labels = class_labels[::-1]  # A=best (highest score) ➜ invert
    out["class_label"] = pd.cut(score, bins=bins, labels=labels, include_lowest=True)
    # duplicate for consistent naming like other methods
    out["class_topsis"] = out["class_label"]

    return out
