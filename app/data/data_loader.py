import streamlit as st
import pandas as pd
import numpy as np
import osmnx as ox

@st.cache_data
def load_osm_data(city_name):
    """
    Load building data from OpenStreetMap for the specified city
    and generate synthetic metrics for demonstration purposes.
    
    Args:
        city_name (str): Name of the city to load data for
        
    Returns:
        pd.DataFrame: DataFrame containing building data with synthetic metrics
    """
    with st.spinner("Fetching OSM data ..."):
        gdf = ox.features_from_place(city_name, tags={"building": True})
        gdf = gdf[gdf.geometry.type == "Polygon"]

        n = len(gdf)
        gdf["CO2_Usage"] = np.random.uniform(50, 500, n)
        gdf["Water_Usage"] = np.random.uniform(1000, 10000, n)
        gdf["Energy_Consumption"] = np.random.uniform(500, 5000, n)
        gdf["height"] = np.random.uniform(10, 100, n)
        gdf["latitude"] = gdf.geometry.centroid.y
        gdf["longitude"] = gdf.geometry.centroid.x

        gdf["polygon"] = gdf.geometry.apply(
            lambda geom: [[list(coord) for coord in geom.exterior.coords]]
        )

        df_ = pd.DataFrame({
            'building_id': range(n),
            'CO2_Usage': gdf["CO2_Usage"],
            'Water_Usage': gdf["Water_Usage"],
            'Energy_Consumption': gdf["Energy_Consumption"],
            'height': gdf["height"],
            'latitude': gdf["latitude"],
            'longitude': gdf["longitude"],
            'polygon': gdf["polygon"]
        })
        return df_