import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

def get_color_mapping(filtered_df: pd.DataFrame, color_by: str) -> list:
    """
    Return a list of RGBA colors (length == len(filtered_df)).
    """
    # 1) If we're coloring by class_label (A–F), use our dict + a default
    if color_by == "class_label":
        class_colors = {
            "A": [0,   255,   0, 200],
            "B": [144, 238, 144, 200],
            "C": [255, 255,   0, 200],
            "D": [255, 165,   0, 200],
            "E": [255,   0,   0, 200],
            "F": [139,   0,   0, 200],
        }
        # cast to str to drop any 'category' dtype
        labels = filtered_df[color_by].astype(str)
        # build one list per row, defaulting to grey if key missing
        return [
            class_colors.get(lbl, [128, 128, 128, 200])
            for lbl in labels
        ]

    # 2) Otherwise, map numeric ranges to colors as before
    else:
        vals = filtered_df[color_by]
        min_val, max_val = vals.min(), vals.max()

        def map_to_color(val):
            if max_val > min_val:
                norm = (val - min_val) / (max_val - min_val)
            else:
                norm = 0.5

            if norm > 0.8:
                return [255,   0,   0, 200]
            elif norm > 0.5:
                return [255, 165,   0, 200]
            elif norm > 0.3:
                return [255, 255,   0, 200]
            else:
                return [0,   255,   0, 200]

        return vals.apply(map_to_color).tolist()

def display_map(filtered_df: pd.DataFrame, city_name: str, color_by: str):
    required_cols = {"latitude", "longitude"}
    if not required_cols.issubset(filtered_df.columns):
        st.error(f"Dataframe missing columns: {required_cols - set(filtered_df.columns)}")
        return

    if filtered_df.empty:
        st.info("No buildings match the current filters.")
        return

    # Couleurs
    filtered_df = filtered_df.copy()
    filtered_df["color"] = get_color_mapping(filtered_df, color_by)

    # Layer pydeck
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=filtered_df,
        get_position=["longitude", "latitude"],
        get_fill_color="color",
        get_radius=30,
        pickable=True,
        id="buildings",
    )

    view_state = pdk.ViewState(
        latitude=filtered_df["latitude"].mean(),
        longitude=filtered_df["longitude"].mean(),
        zoom=14,
    )

    tooltip_txt = (
        "Building ID: {building_id}\n"
        "Class: {class_label}\n"
        "CO₂: {CO2_Usage} kg\n"
        "Water: {Water_Usage} L\n"
        "Energy: {Energy_Consumption} kWh"
    )

    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": tooltip_txt}))
