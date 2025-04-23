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

@st.cache_data
def add_classifications(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add classification columns to the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        pd.DataFrame: DataFrame with classification columns
    """
    df = df.copy()
    
    # Features for classification
    features = ["Energy_Consumption", "CO2_Usage", "Water_Usage"]
    
    # Euclidean Classification
    try:
        df_euclidean = classify_euclidean(df, features=features)
        df["class_euclidean"] = df_euclidean["class_euclidean"]
    except Exception as e:
        st.error(f"Euclidean Classification failed: {str(e)}")
        df["class_euclidean"] = "C"  # Fallback
    
    # Mahalanobis Classification
    try:
        df_mahalanobis = classify_mahalanobis(df, features=features, return_distance=True)
        df["class_mahalanobis"] = df_mahalanobis["class_label"]
        if "Mahalanobis_Distance" in df_mahalanobis.columns:
            df["Mahalanobis_Distance"] = df_mahalanobis["Mahalanobis_Distance"]
    except Exception as e:
        st.error(f"Mahalanobis Classification failed: {str(e)}")
        df["class_mahalanobis"] = "C"  # Fallback
    
    # PCA Classification
    try:
        df_pca = classify_pca(df, features=features)
        df["class_pca"] = df_pca["class_pca"]
    except Exception as e:
        st.error(f"PCA Classification failed: {str(e)}")
        df["class_pca"] = "C"  # Fallback
    
    # Weighted Classification
    try:
        df_weighted = classify_weighted(df, features=features)
        df["class_weighted"] = df_weighted["class_weighted"]
    except Exception as e:
        st.error(f"Weighted Classification failed: {str(e)}")
        df["class_weighted"] = "C"  # Fallback
    
    # Bayesian Classification
    try:
        df_bayesian = classify_bayesian(df, features=features)
        df["class_bayesian"] = df_bayesian["class_bayesian"]
        if "Bayesian_Certainty" in df_bayesian.columns:
            df["Bayesian_Certainty"] = df_bayesian["Bayesian_Certainty"]
    except Exception as e:
        st.error(f"Bayesian Classification failed: {str(e)}")
        df["class_bayesian"] = "C"  # Fallback
    
    return df
