from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────
#  Imports for classification models
# ──────────────────────────────────────────────────────────────
from models.mahalanobis import classify_mahalanobis
from models.pca import classify_pca
from models.weighted import classify_weighted
from models.tree_classifier import classify_robust_tree
from models.cosine import classify_cosine
from models.topsis import classify_topsis


# ──────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "building_id", "latitude", "longitude",
    "CO2_Usage", "Water_Usage", "Energy_Consumption",
]
INTENSITY_COLUMNS = ["Energy_Intensity", "CO2_Intensity"]

# ──────────────────────────────────────────────────────────────
#  Validation / Pre‑processing
# ──────────────────────────────────────────────────────────────

def validate_and_preprocess_dataset(df: pd.DataFrame, scoring_basis: str) -> Optional[pd.DataFrame]:
    """Validate columns, coerce numerics & switch to intensity metrics when requested."""

    # 1 ▸ Handle intensity‑based scoring toggle
    if scoring_basis == "Per m² (kWh/m²/year)":
        if set(INTENSITY_COLUMNS).issubset(df.columns):
            df = df.copy()
            df["Energy_Consumption"] = df["Energy_Intensity"]
            df["CO2_Usage"] = df["CO2_Intensity"]
        else:
            st.error("Missing Energy_Intensity/CO2_Intensity for intensity‑based scoring.")
            return None

    # 2 ▸ Check required cols
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return None

    # 3 ▸ Coerce numerics & drop NA rows in required cols
    numeric_cols = ["latitude", "longitude", "CO2_Usage", "Water_Usage", "Energy_Consumption"]
    df = df.copy()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=REQUIRED_COLUMNS)
    if df.empty:
        st.error("Dataset is empty after preprocessing.")
        return None

    return df

# ------------------------------------------------------------------
# Utility: ensure the chosen class columns exist
# ------------------------------------------------------------------
def ensure_classifications(df, features, weights):
    """Call add_classifications() only when at least one class_* column
    is missing.  Returns df unchanged if everything is already there."""
    expected = {
        "class_euclidean",
        "class_cosine",
        "class_mahalanobis",
        "class_pca",
        "class_weighted",
        "class_tree",
        "class_topsis",
    }
    if expected.issubset(df.columns):
        return df          # nothing to do
    return add_classifications(df, features=features, weights=weights)


# ──────────────────────────────────────────────────────────────
#  Classification helper
# ──────────────────────────────────────────────────────────────

def add_classifications(
    df: pd.DataFrame,
    features: List[str],
    weights: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Run *all* classifiers (Mahalanobis, PCA, Weighted, tree_classifier,
    cosine, TOPSIS) and append their `class_*` columns.
    """

    if len(features) < 2:
        st.error("Need at least two features for classification.")
        return df

    out = df.copy()

    # –– Mahalanobis ––
    try:
        mah = classify_mahalanobis(out, features=features, return_distance=True)
        out["class_mahalanobis"] = mah["class_label"]
        if "Mahalanobis_Distance" in mah:
            out["Mahalanobis_Distance"] = mah["Mahalanobis_Distance"]
    except Exception as e:
        st.warning(f"Mahalanobis failed: {e}")
        out["class_mahalanobis"] = "C"

    # –– PCA ––
    try:
        out["class_pca"] = classify_pca(out, features=features)["class_label"]
    except Exception as e:
        st.warning(f"PCA failed: {e}")
        out["class_pca"] = "C"

    # –– Weighted ––
    try:
        if weights is None:
            weights = [1.0] * len(features)
        out["class_weighted"] = classify_weighted(out, features=features, weights=weights)["class_label"]
    except Exception as e:
        st.warning(f"Weighted failed: {e}")
        out["class_weighted"] = "C"    # –– tree_classifier ––    try:
        tree = classify_robust_tree(out, numeric_features=features)  # Note: tree classifier uses numeric_features
        out["class_tree"] = tree["class_tree"]
    except Exception as e:
        st.warning(f"tree classifier failed: {e}")
        out["class_tree"] = "C"

    # –– cosine ––
    try:
        out["class_cosine"] = classify_cosine(out, features=features)["class_label"]
    except Exception as e:
        st.warning(f"cosine failed: {e}")
        out["class_cosine"] = "C"

    # –– TOPSIS ––
    try:
        out["class_topsis"] = classify_topsis(out, features=features, weights=weights)["class_topsis"]
        out["topsis_score"] = classify_topsis(out, features=features, weights=weights)["topsis_score"]
    except Exception as e:
        st.warning(f"TOPSIS failed: {e}")
        out["class_topsis"] = "C"

    return out
