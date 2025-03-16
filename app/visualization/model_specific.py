import streamlit as st
import plotly.express as px
def display_model_visualization(df, classification_method):
    if classification_method == "Euclidean Distance":
        fig = px.scatter_3d(
            df,
            x='Energy_Consumption',
            y='CO2_Usage',
            z='Water_Usage',
            color='class_label',
            hover_data=['building_id', 'class_label']
        )
        fig.update_traces(marker=dict(size=4, opacity=0.8))
        st.plotly_chart(fig, use_container_width=True)
    elif classification_method == "Mahalanobis Distance":
        fig = px.scatter_3d(
            df,
            x='CO2_Usage',
            y='Water_Usage',
            z='Energy_Consumption',
            color='class_label',
            labels={
                'CO2_Usage': 'CO₂ Usage',
                'Water_Usage': 'Water Usage',
                'Energy_Consumption': 'Energy Consumption',
                'class_label': 'Class'
            },
            hover_data={'Mahalanobis_Distance': ':.2f', 'class_label': True}
        )
        fig.update_traces(marker=dict(size=6, opacity=0.8))
        st.plotly_chart(fig, use_container_width=True)
    # Add more conditions for other classification methods