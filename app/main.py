import streamlit as st
import pandas as pd
import numpy as np

from data.data_loader import load_osm_data
from models.euclidean import classify_euclidean
from models.mahalanobis import classify_mahalanobis
from models.pca import classify_pca
from models.weighted import classify_weighted
from models.bayesian import classify_bayesian
from visualization.map import display_map
from visualization.charts import display_relationship_plot, display_distribution_plot
from visualization.model_specific import display_model_visualization
from utils.metrics import display_metrics_overview
from utils.export import add_export_section, add_benchmark_comparison

# Set page configuration
st.set_page_config(page_title="Building Analytics Dashboard", layout="wide")
st.title("Building Analytics Dashboard")

# Sidebar setup
with st.sidebar:
    st.header("Select a City")
    
    city_name = st.text_input("Enter a city in Morocco", "Casablanca")
    
    if st.button("Load Data"):
        st.session_state["df"] = load_osm_data(city_name)
        if st.session_state["df"].empty:
            st.warning("No data available for the entered city. Try another city.")
    
    if "df" not in st.session_state or st.session_state["df"].empty:
        st.info("Enter a city and click 'Load Data' to fetch building data.")
        st.stop()
        
    # Load data
    df = st.session_state["df"]
    
    # User picks classification
    classification_method = st.radio(
        "Select Classification Method",
        ["Euclidean Distance", "Mahalanobis Distance", "PCA Classification", "Weighted Classification", "Bayesian Classification"]
    )
    
    # Apply classification
    if classification_method == "Euclidean Distance":
        df = classify_euclidean(df)
    elif classification_method == "Mahalanobis Distance":
        df = classify_mahalanobis(df)
    elif classification_method == "PCA Classification":
        df = classify_pca(df)
    elif classification_method == "Weighted Classification":
        df = classify_weighted(df)
    elif classification_method == "Bayesian Classification":
        df = classify_bayesian(df)
    
    # Sliders
    st.subheader("Filters")
    co2_min, co2_max = st.slider(
        "CO2 Usage (kg/month)",
        float(df["CO2_Usage"].min()),
        float(df["CO2_Usage"].max()),
        (float(df["CO2_Usage"].min()), float(df["CO2_Usage"].max()))
    )
    water_min, water_max = st.slider(
        "Water Usage (L/month)",
        float(df["Water_Usage"].min()),
        float(df["Water_Usage"].max()),
        (float(df["Water_Usage"].min()), float(df["Water_Usage"].max()))
    )
    energy_min, energy_max = st.slider(
        "Energy Consumption (kWh/month)",
        float(df["Energy_Consumption"].min()),
        float(df["Energy_Consumption"].max()),
        (float(df["Energy_Consumption"].min()), float(df["Energy_Consumption"].max()))
    )
    height_min, height_max = st.slider(
        "Building Height (m)",
        float(df["height"].min()),
        float(df["height"].max()),
        (float(df["height"].min()), float(df["height"].max()))
    )

    # Class filter
    st.subheader("Class Filter")
    all_classes = ['A', 'B', 'C', 'D', 'E', 'F']
    selected_classes = st.multiselect(
        "Select Building Classes",
        options=all_classes,
        default=all_classes
    )

    color_by = st.selectbox(
        "Color Buildings By",
        ["class_label", "CO2_Usage", "Water_Usage", "Energy_Consumption", "height"]
    )

    # Filter data
    filtered_df = df[
        (df["CO2_Usage"] >= co2_min) & (df["CO2_Usage"] <= co2_max) &
        (df["Water_Usage"] >= water_min) & (df["Water_Usage"] <= water_max) &
        (df["Energy_Consumption"] >= energy_min) & (df["Energy_Consumption"] <= energy_max) &
        (df["height"] >= height_min) & (df["height"] <= height_max) &
        (df["class_label"].isin(selected_classes))
    ]

    st.write(f"Showing {len(filtered_df)} of {len(df)} buildings")

    # Add usage instructions
    st.sidebar.markdown("---")
    st.sidebar.subheader("How to use this dashboard")
    st.sidebar.markdown("""
    1. Use the sliders to filter buildings by metrics  
    2. Select which building classes to display using the Class Filter  
    3. Choose which metric to use for coloring the buildings  
    4. Explore the 3D visualization by dragging, zooming, and rotating  
    5. Analyze the classification plots (PCA, Weighted, etc.)  
    6. Examine relationships between metrics using the scatter plot  
    7. View detailed data in the table below  
    """)

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Building Distribution in {city_name}")
    if not filtered_df.empty:
        display_map(filtered_df, city_name, color_by)
    else:
        st.write("No buildings match the current filters.")
        
    # Model-specific visualization
    st.subheader("Classification-Specific Visualization")
    if not filtered_df.empty:
        display_model_visualization(filtered_df, classification_method)
    
    # Relationship plot
    st.subheader("Relationship Between Metrics")
    if not filtered_df.empty:
        display_relationship_plot(filtered_df, color_by)

with col2:
    st.subheader("Metrics Overview")
    if not filtered_df.empty:
        display_metrics_overview(filtered_df)
    else:
        st.write("No buildings match the current filters.")

# Distribution plot
if not filtered_df.empty:
    display_distribution_plot(filtered_df, color_by)

# Show building data
if not filtered_df.empty:
    st.subheader("Building Data")
    st.dataframe(
        filtered_df[
            [
                "building_id", "class_label", "CO2_Usage", "Water_Usage",
                "Energy_Consumption", "height", "latitude", "longitude"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )
else:
    st.subheader("Building Data")
    st.write("No buildings match the current filters.")

# Export functionality
st.subheader("Export Data")
add_export_section(filtered_df)

# Benchmark comparison
add_benchmark_comparison(filtered_df)