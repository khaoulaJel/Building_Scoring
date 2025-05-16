"""
Module: dataprocessing.py

This module provides data-loading, cleaning, feature-engineering, synthetic-data generation,
clustering, and analysis utilities for building-performance datasets.
It supports both local CSV paths and in-memory file-like objects (e.g., Streamlit uploads).
"""
import os
from datetime import datetime

import pandas as pd
import numpy as np
from pyproj import Transformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


def process_city_data(city_name: str, input_source=None, year: int = None) -> pd.DataFrame:
    """
    Process building data for a specific city with optional year tagging.

    Args:
        city_name (str): Name of the city to process
        input_source (str, file-like): Path to CSV or file-like object. If None, defaults to '<city>.csv'.
        year (int, optional): Year to tag this dataset with if no 'year' column exists.

    Returns:
        pd.DataFrame: Cleaned, feature-engineered DataFrame.
    """
    # 1. Read CSV
    if input_source is None:
        filename = f"{city_name.lower()}.csv"
        df = pd.read_csv(filename)
    else:
        try:
            df = pd.read_csv(input_source)
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file not found: {input_source}")

    # 2. Convert coordinates (if present)
    def _convert_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
        transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
        x = frame.get("coordonnee_cartographique_x_ban")
        y = frame.get("coordonnee_cartographique_y_ban")
        if x is not None and y is not None:
            lon, lat = transformer.transform(x.values, y.values)
            frame["longitude"] = lon
            frame["latitude"] = lat
        return frame

    df = _convert_coordinates(df)

    # 3. Rename & feature engineering
    df = df.rename(columns={
        "numero_dpe": "building_id",
        "conso_5 usages_ef": "Energy_Consumption",
        "emission_ges_5_usages": "CO2_Usage",
        "etiquette_dpe": "true_energy_label",
        "etiquette_ges": "true_ges_label"
    })
    if "Energy_Consumption" in df:
        df["Water_Usage"] = df["Energy_Consumption"] * 0.3
    df["Energy_Intensity"] = df.get("conso_5 usages_par_m2_ef", np.nan)
    df["CO2_Intensity"]    = df.get("emission_ges_5_usages par_m2", np.nan)

    # 4. Select relevant columns & drop rows missing required
    required = [
        "building_id", "Energy_Consumption", "CO2_Usage",
        "Water_Usage", "Energy_Intensity", "CO2_Intensity"
    ]
    if "latitude" in df and "longitude" in df:
        required += ["latitude", "longitude"]
    df = df.dropna(subset=[c for c in required if c in df.columns])

    # 5. Log-transform numeric features
    for col in ["Energy_Consumption", "CO2_Usage", "Energy_Intensity", "CO2_Intensity"]:
        if col in df:
            df[f"log1p_{col}"] = np.log1p(df[col])

    # 6. Tag city & year (preserve existing 'year' if present)
    df["city"] = city_name
    if "year" not in df.columns:
        df["year"] = year or datetime.now().year

    return df


def draw_multipliers(n_buildings: int) -> np.ndarray:
    """
    Generate multipliers for synthetic future data.

    Args:
        n_buildings (int): Number of buildings
    Returns:
        np.ndarray: Array of multipliers
    """
    scenarios = np.random.choice(
        ["normal", "uniform", "lognormal"],
        size=n_buildings,
        p=[0.6, 0.3, 0.1]
    )
    m = np.zeros(n_buildings)
    for i, scen in enumerate(scenarios):
        if scen == "normal":
            m[i] = np.random.normal(1.0, 0.05)
        elif scen == "uniform":
            m[i] = np.random.uniform(0.9, 1.1)
        else:
            m[i] = np.random.lognormal(0, 0.1)
    return m


def generate_future_data(df: pd.DataFrame, city_name: str, target_year: int) -> pd.DataFrame:
    """
    Generate synthetic data for a future year based on existing data.
    """
    future = df.copy()
    future["city"] = city_name
    future["year"] = target_year
    n = len(future)
    per = [("Energy_Consumption", draw_multipliers(n)),
           ("Energy_Intensity", draw_multipliers(n)),
           ("CO2_Usage", draw_multipliers(n)),
           ("CO2_Intensity", draw_multipliers(n)),
           ("Water_Usage", draw_multipliers(n))]
    for col, mul in per:
        if col in future:
            future[col] = future[col] * mul
    # update logs
    for col in ["Energy_Consumption", "CO2_Usage", "Energy_Intensity", "CO2_Intensity"]:
        logc = f"log1p_{col}"
        if col in future and logc in future:
            future[logc] = np.log1p(future[col])
    # optional label improvements
    label_map = {"G": "F", "F": "E", "E": "D", "D": "C", "C": "B", "B": "A", "A": "A"}
    for lab in ["true_energy_label", "true_ges_label"]:
        if lab in future:
            mask = np.random.random(n) < 0.05
            future.loc[mask, lab] = future.loc[mask, lab].map(lambda x: label_map.get(x, x))
    return future


def cluster_future_data(df: pd.DataFrame, city_name: str, year: int) -> pd.DataFrame:
    """
    Apply PCA + KMeans clustering to the data.
    """
    log_cols = [c for c in df if c.startswith("log1p_")]
    if len(log_cols) < 2:
        return df
    X = StandardScaler().fit_transform(df[log_cols].values)
    pcs = PCA(n_components=2).fit_transform(X)
    df["PC1"], df["PC2"] = pcs[:,0], pcs[:,1]
    df["cluster"] = KMeans(n_clusters=4, random_state=42).fit_predict(X)
    return df


def analyze_city_data(df: pd.DataFrame, city_name: str, year: int = None) -> pd.DataFrame:
    """
    Generate visual analyses: histograms, boxplots, heatmaps, pairplots, geo-scatter.
    """
    title = f"{city_name} ({year or df.get('year', '')})"
    base_nums = ["Energy_Consumption","CO2_Usage","Water_Usage","Energy_Intensity","CO2_Intensity"]
    nums = [c for c in base_nums if c in df]
    for col in nums:
        plt.figure(); sns.histplot(df[col], kde=True); plt.title(f"{title} - {col}"); plt.close()
        plt.figure(); sns.boxplot(x=df[col]); plt.title(f"{title} - {col} box"); plt.close()
    corr = df[nums].corr()
    plt.figure(); sns.heatmap(corr, annot=True); plt.title(f"{title} corr"); plt.close()
    sns.pairplot(df[nums], diag_kind="kde", corner=True); plt.close()
    if all(c in df for c in ["longitude","latitude"]):
        plt.figure(); sns.scatterplot(x=df.longitude, y=df.latitude,
                                     size=df.Energy_Consumption if "Energy_Consumption" in df else None);
        plt.close()
    return df


def process_city_with_years(
    city_name: str,
    input_source=None,
    base_year: int = None,
    future_years: list = None
) -> dict:
    """
    Process city data for a base year and optionally generate future-year datasets.

    Returns a dict: {year: DataFrame}.
    """
    base_year = base_year or datetime.now().year
    futs = future_years or []
    base_df = process_city_data(city_name, input_source, base_year)
    result = {base_year: base_df}
    prev = base_df
    for y in futs:
        fut = generate_future_data(prev, city_name, y)
        result[y] = fut
        prev = fut
    return result
