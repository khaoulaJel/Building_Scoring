import os
import pandas as pd
import streamlit as st
from datetime import datetime
from sqlalchemy.engine.url import URL
from sqlalchemy import create_engine
from models.euclidean import classify_euclidean
from models.mahalanobis import classify_mahalanobis
from models.pca import classify_pca
from models.weighted import classify_weighted
from models.bayesian import classify_bayesian

# -------------------------------------------------------------------
# Define required columns and helper functions
# -------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "building_id", "latitude", "longitude",
    "CO2_Usage", "Water_Usage", "Energy_Consumption"
]
INTENSITY_COLUMNS = ["Energy_Intensity", "CO2_Intensity"]

def validate_and_preprocess_dataset(df, scoring_basis):
    """Validate and preprocess the dataset."""
    # First, check if we need to rename lowercase column names from DB to match app expectations
    column_mapping = {
        "energy_consumption": "Energy_Consumption",
        "co2_usage": "CO2_Usage",
        "water_usage": "Water_Usage",
        "energy_intensity": "Energy_Intensity",
        "co2_intensity": "CO2_Intensity"
    }
    
    # Only rename columns that exist in the dataframe
    rename_dict = {old: new for old, new in column_mapping.items() if old in df.columns}
    if rename_dict:
        df = df.rename(columns=rename_dict)
    
    # Check for required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        if scoring_basis == "Per m² (kWh/m²/year)":
            if "Energy_Intensity" in df.columns and "CO2_Intensity" in df.columns:
                df["Energy_Consumption"] = df["Energy_Intensity"]
                df["CO2_Usage"] = df["CO2_Intensity"]
                # Recheck for missing columns after adding these
                missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
                if missing_cols:
                    st.warning(f"Missing required columns: {missing_cols}")
                    return None
            else:
                st.warning(f"Missing required columns for intensity-based scoring: {missing_cols} or Energy_Intensity/CO2_Intensity")
                return None
        else:
            st.warning(f"Missing required columns: {missing_cols}")
            return None
    
    # Convert columns to numeric
    numeric_cols = ["latitude", "longitude", "CO2_Usage", "Water_Usage", "Energy_Consumption"]
    for col in numeric_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                st.warning(f"Error converting {col} to numeric: {e}")
                return None
    
    # Remove rows with missing values in key columns
    existing_columns = [col for col in REQUIRED_COLUMNS if col in df.columns]
    if existing_columns:
        df = df.dropna(subset=existing_columns)
    
    if df.empty:
        st.warning("Dataset is empty after preprocessing")
        return None
    
    # Handle scoring basis
    if scoring_basis == "Per m² (kWh/m²/year)":
        # If we don't already have intensity columns
        if "Energy_Intensity" not in df.columns or "CO2_Intensity" not in df.columns:
            # If we have area info
            if "surface_habitable_immeuble" in df.columns:
                if "Energy_Consumption" in df.columns:
                    df["Energy_Intensity"] = df["Energy_Consumption"] / df["surface_habitable_immeuble"]
                if "CO2_Usage" in df.columns:
                    df["CO2_Intensity"] = df["CO2_Usage"] / df["surface_habitable_immeuble"]
    
    return df

@st.cache_data
def add_classifications(df: pd.DataFrame, features: list, weights: list = None) -> pd.DataFrame:
    """
    Add classification columns to the DataFrame using all available
    classification models. Assumes each classifier returns a 'class_label'
    column (and Bayesian returns 'Bayesian_Certainty').

    Args:
        df (pd.DataFrame): Input data
        features (list): Feature column names for classification
        weights (list, optional): Weights for weighted classification
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
            # fallback: equal weights
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