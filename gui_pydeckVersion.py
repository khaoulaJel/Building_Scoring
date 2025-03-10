import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import osmnx as ox
import geopandas as gpd
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

st.set_page_config(page_title="Building Analytics Dashboard", layout="wide")
st.title("Ben Guerir Building Analytics Dashboard")

with st.sidebar:
    st.header("Data and Filters")
    data_source = st.radio("Data Source", ["Sample Data", "Load OSM Data (Slower)"])

    @st.cache_data
    def generate_sample_data(n=100):
        data = {
            'building_id': range(n),
            'CO2_Usage': np.random.uniform(50, 500, n),
            'Water_Usage': np.random.uniform(1000, 10000, n),
            'Energy_Consumption': np.random.uniform(500, 5000, n),
            'height': np.random.uniform(10, 100, n),
            'latitude': np.random.uniform(32.22, 32.24, n),
            'longitude': np.random.uniform(-7.96, -7.94, n)
        }
        polygons = []
        for i in range(n):
            center_lat = data['latitude'][i]
            center_lon = data['longitude'][i]
            size = np.random.uniform(0.0002, 0.0005)
            points = [
                [center_lon - size, center_lat - size],
                [center_lon + size, center_lat - size],
                [center_lon + size, center_lat + size],
                [center_lon - size, center_lat + size],
                [center_lon - size, center_lat - size]
            ]
            polygons.append([points])
        data['polygon'] = polygons
        return pd.DataFrame(data)

    @st.cache_data
    def load_osm_data():
        city_name = "Ben Guerir, Morocco"
        with st.spinner("Fetching OSM data for Ben Guerir..."):
            gdf = ox.features_from_place(city_name, tags={"building": True})
            gdf = gdf[gdf.geometry.type == "Polygon"]
            gdf["CO2_Usage"] = np.random.uniform(50, 500, len(gdf))
            gdf["Water_Usage"] = np.random.uniform(1000, 10000, len(gdf))
            gdf["Energy_Consumption"] = np.random.uniform(500, 5000, len(gdf))
            gdf["height"] = np.random.uniform(10, 100, len(gdf))
            gdf["latitude"] = gdf.geometry.centroid.y
            gdf["longitude"] = gdf.geometry.centroid.x
            gdf["polygon"] = gdf.geometry.apply(
                lambda geom: [[list(coord) for coord in geom.exterior.coords]]
            )
            df = pd.DataFrame({
                'building_id': range(len(gdf)),
                'CO2_Usage': gdf["CO2_Usage"],
                'Water_Usage': gdf["Water_Usage"],
                'Energy_Consumption': gdf["Energy_Consumption"],
                'height': gdf["height"],
                'latitude': gdf["latitude"],
                'longitude': gdf["longitude"],
                'polygon': gdf["polygon"]
            })
            return df

    if data_source == "Sample Data":
        df = generate_sample_data()
    else:
        df = load_osm_data()

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
    color_by = st.selectbox(
        "Color Buildings By", 
        ["CO2_Usage", "Water_Usage", "Energy_Consumption", "height"]
    )
    filtered_df = df[
        (df["CO2_Usage"] >= co2_min) & (df["CO2_Usage"] <= co2_max) &
        (df["Water_Usage"] >= water_min) & (df["Water_Usage"] <= water_max) &
        (df["Energy_Consumption"] >= energy_min) & (df["Energy_Consumption"] <= energy_max) &
        (df["height"] >= height_min) & (df["height"] <= height_max)
    ]
    st.write(f"Showing {len(filtered_df)} of {len(df)} buildings")

col1, col2 = st.columns([2, 1])

def get_color_mapping(values, metric):
    min_val = values.min()
    max_val = values.max()
    def map_to_color(val):
        normalized = (val - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        if normalized > 0.8:
            return [255, 0, 0, 200]
        elif normalized > 0.5:
            return [255, 165, 0, 200]
        elif normalized > 0.3:
            return [255, 255, 0, 200]
        else:
            return [0, 255, 0, 200]
    return filtered_df[metric].apply(map_to_color).tolist()

with col1:
    st.subheader("Building 3D Visualization")
    colors = get_color_mapping(filtered_df[color_by], color_by)
    colored_df = filtered_df.copy()
    colored_df["color"] = colors
    building_layer = pdk.Layer(
        "PolygonLayer",
        colored_df,
        id="buildings",
        get_polygon="polygon",
        get_elevation="height",
        elevation_scale=1,
        get_fill_color="color",
        extruded=True,
        pickable=True,
        auto_highlight=True,
    )
    view_state = pdk.ViewState(
        latitude=filtered_df["latitude"].mean(),
        longitude=filtered_df["longitude"].mean(),
        zoom=14,
        pitch=45,
    )
    deck = pdk.Deck(
        layers=[building_layer],
        initial_view_state=view_state,
        tooltip={
            "text": "Building ID: {building_id}\nHeight: {height}m\nCO₂: {CO2_Usage}kg\nWater: {Water_Usage}L\nEnergy: {Energy_Consumption}kWh"
        }
    )
    st.pydeck_chart(deck)

with col2:
    st.subheader("Metrics Overview")
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric("Average CO₂ Usage", f"{filtered_df['CO2_Usage'].mean():.1f} kg")
        st.metric("Average Water Usage", f"{filtered_df['Water_Usage'].mean():.0f} L")
    with metric_col2:
        st.metric("Average Energy", f"{filtered_df['Energy_Consumption'].mean():.0f} kWh")
        st.metric("Average Height", f"{filtered_df['height'].mean():.1f} m")
    st.subheader(f"Distribution of {color_by}")
    fig = px.histogram(
        filtered_df, 
        x=color_by,
        nbins=20,
        color_discrete_sequence=["#3366CC"]
    )
    fig.update_layout(
        xaxis_title=color_by,
        yaxis_title="Number of Buildings",
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Relationship Between Metrics")
    x_metric = st.selectbox("X-Axis", ["CO2_Usage", "Water_Usage", "Energy_Consumption", "height"], key="x_axis")
    y_metric = st.selectbox("Y-Axis", ["Energy_Consumption", "CO2_Usage", "Water_Usage", "height"], key="y_axis")
    scatter = px.scatter(
        filtered_df,
        x=x_metric,
        y=y_metric,
        color=color_by,
        color_continuous_scale="Viridis",
        hover_data=["building_id", "CO2_Usage", "Water_Usage", "Energy_Consumption", "height"]
    )
    scatter.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        coloraxis_colorbar=dict(title=color_by)
    )
    st.plotly_chart(scatter, use_container_width=True)

st.subheader("Building Data")
st.dataframe(
    filtered_df[["building_id", "CO2_Usage", "Water_Usage", "Energy_Consumption", "height", "latitude", "longitude"]], 
    use_container_width=True,
    hide_index=True
)

st.sidebar.markdown("---")
st.sidebar.subheader("How to use this dashboard")
st.sidebar.markdown("""
1. Use the sliders to filter buildings by metrics
2. Select which metric to use for coloring the buildings
3. Explore the 3D visualization by dragging, zooming, and rotating
4. Analyze relationships between metrics using the scatter plot
5. View detailed data in the table below
""")
