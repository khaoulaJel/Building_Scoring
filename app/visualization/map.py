# map_utils.py

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import geopandas as gpd
import json


def get_color_mapping(filtered_df, color_by):
    """
    Get RGBA color mapping for buildings based on the selected attribute.
    """
    values = filtered_df[color_by]

    if color_by == "class_label":
        # Discrete class colors
        class_colors = {
            'A': [0, 255, 0, 200],
            'B': [144, 238, 144, 200],
            'C': [255, 255, 0, 200],
            'D': [255, 165, 0, 200],
            'E': [255, 0, 0, 200],
            'F': [139, 0, 0, 200]
        }
        return [class_colors.get(cls, [128, 128, 128, 200]) for cls in values]

    # Continuous gradient thresholds
    min_val, max_val = values.min(), values.max()
    colors = []
    for val in values:
        norm = (val - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        if norm > 0.8:
            colors.append([255, 0, 0, 200])      # Red
        elif norm > 0.5:
            colors.append([255, 165, 0, 200])    # Orange
        elif norm > 0.3:
            colors.append([255, 255, 0, 200])    # Yellow
        else:
            colors.append([0, 255, 0, 200])      # Green
    return colors


def display_map(filtered_df, city_name, color_by):
    """
    Display a dark-themed map of buildings colored by the selected attribute.

    Args:
        filtered_df (pd.DataFrame): Filtered dataframe with building data
        city_name (str): Name of the city
        color_by (str): Column name to color by
    """
    if filtered_df.empty:
        st.write("No buildings match the current filters.")
        return

    # Compute RGBA colors
    colors = get_color_mapping(filtered_df, color_by)

    # Determine rendering mode
    if 'geometry' in filtered_df.columns and not filtered_df['geometry'].isna().all():
        # Render polygons
        gdf = gpd.GeoDataFrame(filtered_df, geometry='geometry')
        geojson = json.loads(gdf.to_json())
        for feature, col in zip(geojson['features'], colors):
            feature['properties']['color'] = col

        layer = pdk.Layer(
            'GeoJsonLayer',
            geojson,
            stroked=True,
            filled=True,
            get_fill_color='properties.color',
            get_line_color=[255, 255, 255, 100],
            pickable=True,
            auto_highlight=True
        )
        html = (
            "<b>Building {properties.building_id}</b><br/>"
            "Class: {properties.class_label}<br/>"
            "Height: {properties.height} m<br/>"
            "CO₂: {properties.CO2_Usage} kg<br/>"
            "Water: {properties.Water_Usage} L<br/>"
            "Energy: {properties.Energy_Consumption} kWh"
        )
    else:
        # Fallback: scatter points
        cdf = filtered_df.copy()
        cdf['color'] = colors
        layer = pdk.Layer(
            'ScatterplotLayer',
            cdf,
            id='buildings',
            get_position=['longitude', 'latitude'],
            get_radius=30,
            get_fill_color='color',
            pickable=True
        )
        html = (
            "Building ID: {building_id}<br/>"
            "Class: {class_label}<br/>"
            "Height: {height} m<br/>"
            "CO₂: {CO2_Usage} kg<br/>"
            "Water: {Water_Usage} L<br/>"
            "Energy: {Energy_Consumption} kWh"
        )

    # Center the view
    view_state = pdk.ViewState(
        latitude=float(filtered_df['latitude'].mean()),
        longitude=float(filtered_df['longitude'].mean()),
        zoom=14,
        pitch=0
    )

    # Build deck with dark style and styled tooltip
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v10",
        tooltip={
            "html": html,
            "style": {
                "backgroundColor": "rgba(0, 0, 0, 0.8)",
                "color": "white",
                "padding": "10px",
                "borderRadius": "4px",
                "fontFamily": "Arial"
            }
        }
    )

    # Render full-width
    st.pydeck_chart(deck, use_container_width=True)

    # Legend for discrete classes
    if color_by == 'class_label':
        legend = {
            'A': '#00FF00',
            'B': '#90EE90',
            'C': '#FFFF00',
            'D': '#FFA500',
            'E': '#FF0000',
            'F': '#8B0000'
        }
        st.markdown("**Class Legend:**")
        for cls, hexcol in legend.items():
            st.markdown(
                f"<span style='display:inline-block;width:15px;height:15px;"
                f"background-color:{hexcol};margin-right:5px;'></span>{cls}",
                unsafe_allow_html=True
            )
