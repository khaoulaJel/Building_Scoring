import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

def get_color_mapping(filtered_df, color_by):
    """
    Get color mapping for buildings based on the selected attribute
    
    Args:
        filtered_df (pd.DataFrame): Filtered dataframe with buildings
        color_by (str): Column name to color by
        
    Returns:
        list: List of colors for each building
    """
    values = filtered_df[color_by]
    
    if color_by == "class_label":
        class_colors = {
            'A': [0, 255, 0, 200],
            'B': [144, 238, 144, 200],
            'C': [255, 255, 0, 200],
            'D': [255, 165, 0, 200],
            'E': [255, 0, 0, 200],
            'F': [139, 0, 0, 200]
        }
        return [class_colors[cls] if cls in class_colors else [128, 128, 128, 200] for cls in values]
    else:
        min_val = values.min()
        max_val = values.max()
        colors = []
        for val in values:
            normalized = (val - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            if normalized > 0.8:
                colors.append([255, 0, 0, 200])  # Red
            elif normalized > 0.5:
                colors.append([255, 165, 0, 200])  # Orange
            elif normalized > 0.3:
                colors.append([255, 255, 0, 200])  # Yellow
            else:
                colors.append([0, 255, 0, 200])  # Green
        return colors

def display_map(filtered_df, city_name, color_by):
    """
    Display a map of buildings colored by the selected attribute
    
    Args:
        filtered_df (pd.DataFrame): Filtered dataframe with buildings
        city_name (str): Name of the city
        color_by (str): Column name to color by
    """
    if filtered_df.empty:
        st.write("No buildings match the current filters.")
        return
        
    colors = get_color_mapping(filtered_df, color_by)
    cdf = filtered_df.copy()
    cdf["color"] = colors

    # Define a ScatterplotLayer
    building_layer = pdk.Layer(
        "ScatterplotLayer",
        cdf,
        id="buildings",
        get_position=["longitude", "latitude"],
        get_radius=30,
        get_fill_color="color",
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=filtered_df["latitude"].mean(),
        longitude=filtered_df["longitude"].mean(),
        zoom=14,
        pitch=0,
    )

    deck = pdk.Deck(
        layers=[building_layer],
        initial_view_state=view_state,
        tooltip={
            "text": (
                "Building ID: {building_id}\n"
                "Class: {class_label}\n"
                "Height: {height}m\n"
                "CO₂: {CO2_Usage}kg\n"
                "Water: {Water_Usage}L\n"
                "Energy: {Energy_Consumption}kWh"
            )
        }
    )
    st.pydeck_chart(deck)
