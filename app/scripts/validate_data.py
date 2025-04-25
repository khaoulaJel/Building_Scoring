from datetime import time
from pathlib import Path
import pandas as pd
from pyproj import Transformer
import streamlit as st
import pandas as pd
from models.euclidean import classify_euclidean
from models.mahalanobis import classify_mahalanobis
from models.pca import classify_pca
from models.weighted import classify_weighted
from models.bayesian import classify_bayesian


REQUIRED_COLUMNS = [
    "building_id", "latitude", "longitude",
    "CO2_Usage", "Water_Usage", "Energy_Consumption"
]
INTENSITY_COLUMNS = ["Energy_Intensity", "CO2_Intensity"]


# Function to validate and preprocess dataset
def validate_and_preprocess_dataset(df, scoring_basis):
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        if scoring_basis == "Per m² (kWh/m²/year)":
            if "Energy_Intensity" in df.columns and "CO2_Intensity" in df.columns:
                df["Energy_Consumption"] = df["Energy_Intensity"]
                df["CO2_Usage"] = df["CO2_Intensity"]
            else:
                st.error(f"Missing required columns for intensity-based scoring: {missing_cols} or Energy_Intensity/CO2_Intensity")
                return None
        else:
            st.error(f"Missing required columns: {missing_cols}")
            return None
    
    numeric_cols = ["latitude", "longitude", "CO2_Usage", "Water_Usage", "Energy_Consumption"]
    for col in numeric_cols:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            st.error(f"Column {col} must contain numeric values")
            return None
    
    df = df.dropna(subset=REQUIRED_COLUMNS)
    
    if df.empty:
        st.error("Dataset is empty after preprocessing")
        return None
    
    return df
import pandas as pd
import streamlit as st
@st.cache_data
def add_classifications(df: pd.DataFrame, features: list,    weights: list = None ) -> pd.DataFrame:
    """
    Add classification columns to the DataFrame using all available
    classification models. Assumes each classifier returns a 'class_label'
    column (and Bayesian returns 'Bayesian_Certainty').

    Args:
        df (pd.DataFrame): Input data
        features (list): Exactly 3 feature column names for classification
    Returns:
        pd.DataFrame: df with added class_* columns (and any distance/confidence)
    """
    df = df.copy()

    # Euclidean Classification
    try:
        df_euclid = classify_euclidean(df, features=features)
        df["class_euclidean"] = df_euclid["class_label"]
    except Exception as e:
        st.error(f"Euclidean Classification failed: {e}")
        df["class_euclidean"] = "C"

    # Mahalanobis Classification
    try:
        df_mah = classify_mahalanobis(df, features=features, return_distance=True)
        df["class_mahalanobis"] = df_mah["class_label"]
        if "Mahalanobis_Distance" in df_mah:
            df["Mahalanobis_Distance"] = df_mah["Mahalanobis_Distance"]
    except Exception as e:
        st.error(f"Mahalanobis Classification failed: {e}")
        df["class_mahalanobis"] = "C"

    # PCA Classification
    try:
        df_pca = classify_pca(df, features=features)
        df["class_pca"] = df_pca["class_label"]
    except Exception as e:
        st.error(f"PCA Classification failed: {e}")
        df["class_pca"] = "C"

    # Weighted Classification
    try:
        if weights is not None:
            df_w = classify_weighted(df, features=features, weights=weights)
        else:
            # fallback:  equal weights
            eq_w = [1]*len(features)
            df_w = classify_weighted(df, features=features, weights=eq_w)
        df["class_weighted"] = df_w["class_label"]
    except Exception as e:
        st.error(f"Weighted Classification failed: {e}")
        df["class_weighted"] = "C"
    # Bayesian Classification
    try:
        df_bayes = classify_bayesian(df, features=features)
        df["class_bayesian"] = df_bayes["class_label"]
        if "Bayesian_Certainty" in df_bayes:
            df["Bayesian_Certainty"] = df_bayes["Bayesian_Certainty"]
    except Exception as e:
        st.error(f"Bayesian Classification failed: {e}")
        df["class_bayesian"] = "C"

    return df
