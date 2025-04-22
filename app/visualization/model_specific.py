import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np

def display_model_visualization(df: pd.DataFrame, classification_method: str) -> None:
    """
    Display model-specific visualizations based on classification method.
    
    Args:
        df (pd.DataFrame): DataFrame with building data and classifications
        classification_method (str): Selected classification method
    """
    st.markdown(f"#### {classification_method} Visualization")
    
    # Validate required columns
    required_columns = ["Energy_Consumption", "CO2_Usage", "Water_Usage", "class_label"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return
    
    # Validate data types and values
    df = df.copy()
    for col in required_columns[:3]:  # Check numeric columns
        if not pd.api.types.is_numeric_dtype(df[col]):
            st.error(f"Column '{col}' contains non-numeric data.")
            return
        if df[col].isna().any() or np.isinf(df[col]).any():
            st.warning(f"Column '{col}' contains NaN or infinite values. Removing invalid rows.")
            df = df[df[col].notna() & ~np.isinf(df[col])]
    
    # Validate class_label
    valid_classes = ['A', 'B', 'C', 'D', 'E', 'F']
    if not df["class_label"].isin(valid_classes).all():
        st.error(f"Column 'class_label' contains invalid values: {df['class_label'].unique()}")
        return
    
    # If DataFrame is empty after cleaning
    if df.empty:
        st.warning("No valid data available for visualization after cleaning.")
        return
    
    # Define hover data based on classification method
    hover_data = ["building_id", "class_label"]
    if classification_method == "Mahalanobis Distance" and "Mahalanobis_Distance" in df.columns:
        hover_data.append("Mahalanobis_Distance")
    elif classification_method == "Bayesian Classification" and "Bayesian_Certainty" in df.columns:
        hover_data.append("Bayesian_Certainty")
    else:
        hover_data.append("Energy_Consumption")  # Fallback hover data
    
    # Create 3D scatter plot
    try:
        fig = px.scatter_3d(
            df,
            x="Energy_Consumption",
            y="CO2_Usage",
            z="Water_Usage",
            color="class_label",
            color_discrete_map={
                'A': '#28a745', 'B': '#5cb85c', 'C': '#ffc107',
                'D': '#fd7e14', 'E': '#dc3545', 'F': '#6c757d'
            },
            hover_data=hover_data,
            labels={
                "Energy_Consumption": "Energy (kWh)",
                "CO2_Usage": "CO₂ (kg)",
                "Water_Usage": "Water (L)",
                "class_label": "Class"
            }
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title="Energy (kWh)",
                yaxis_title="CO₂ (kg)",
                zaxis_title="Water (L)",
                xaxis=dict(range=[df["Energy_Consumption"].min(), df["Energy_Consumption"].max()]),
                yaxis=dict(range=[df["CO2_Usage"].min(), df["CO2_Usage"].max()]),
                zaxis=dict(range=[df["Water_Usage"].min(), df["Water_Usage"].max()])
            ),
            height=600,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
    
    except Exception as e:
        st.error(f"Error rendering 3D scatter plot: {str(e)}")
        st.write("DataFrame head:")
        st.write(df.head())