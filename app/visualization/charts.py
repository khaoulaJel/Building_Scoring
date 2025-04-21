import streamlit as st
import pandas as pd
import plotly.express as px

def display_relationship_plot(filtered_df, color_by):
    """
    Display a scatter plot showing the relationship between two selected metrics
    
    Args:
        filtered_df (pd.DataFrame): Filtered dataframe with buildings
        color_by (str): Column name to color by
    """
    x_metric = st.selectbox(
        "X-Axis",
        ["CO2_Usage", "Water_Usage", "Energy_Consumption", "height"],
        key="x_axis"
    )
    y_metric = st.selectbox(
        "Y-Axis",
        ["Energy_Consumption", "CO2_Usage", "Water_Usage"],
        key="y_axis"
    )
    
    class_colors_map = {
        'A': 'green', 'B': 'lightgreen', 'C': 'yellow',
        'D': 'orange', 'E': 'red', 'F': 'darkred'
    }
    
    scatter = px.scatter(
        filtered_df,
        x=x_metric,
        y=y_metric,
        color="class_label" if color_by == "class_label" else color_by,
        color_discrete_map=class_colors_map if color_by == "class_label" else None,
        color_continuous_scale="Viridis" if color_by != "class_label" else None,
        hover_data=[
            "building_id", "class_label", "CO2_Usage", "Water_Usage",
            "Energy_Consumption"
        ]
    )
    
    scatter.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        coloraxis_colorbar=dict(title=color_by) if color_by != "class_label" else None
    )
    
    st.plotly_chart(scatter, use_container_width=True)

def display_distribution_plot(filtered_df, color_by):
    """
    Display a distribution plot for the selected attribute
    
    Args:
        filtered_df (pd.DataFrame): Filtered dataframe with buildings
        color_by (str): Column name to color by
    """
    if color_by != "class_label":
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
    else:
        st.subheader("Class Distribution")
        class_counts = filtered_df['class_label'].value_counts().reset_index()
        class_counts.columns = ['Class', 'Count']
        class_colors = {
            'A': 'green', 'B': 'lightgreen', 'C': 'yellow',
            'D': 'orange', 'E': 'red', 'F': 'darkred'
        }