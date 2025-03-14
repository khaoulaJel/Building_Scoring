import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import osmnx as ox
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from joblib import Parallel, delayed

st.set_page_config(page_title="Building Analytics Dashboard", layout="wide")
st.title("Ben Guerir Building Analytics Dashboard")

with st.sidebar:
    st.header("Data and Filters")
    data_source = st.radio("Data Source", ["Sample Data", "Load OSM Data (Slower)"])

    def process_data(df):
        """
        Process building data by normalizing metrics and classifying buildings.
        
        Args:
            df (DataFrame): DataFrame containing building data
            
        Returns:
            DataFrame: Processed DataFrame with additional columns
        """
        # Normalize data
        df['energy_norm'] = df['Energy_Consumption'] / df['Energy_Consumption'].max()
        df['carbon_norm'] = df['CO2_Usage'] / df['CO2_Usage'].max()
        df['water_norm'] = df['Water_Usage'] / df['Water_Usage'].max()

        # Vectorized distance calculation
        df['distance'] = np.sqrt(df['energy_norm']**2 + df['carbon_norm']**2 + df['water_norm']**2)

        # Class assignment
        thresholds = np.linspace(df['distance'].min(), df['distance'].max(), 7)
        
        # Use numpy digitize for efficiency instead of Parallel
        df['class'] = np.digitize(df['distance'], thresholds[1:], right=True)

        # Map class numbers to letters
        class_labels = ['A', 'B', 'C', 'D', 'E', 'F']
        df['class_label'] = df['class'].apply(lambda x: class_labels[min(x, len(class_labels)-1)])

        return df

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
        
        # Create building polygons
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
        df = pd.DataFrame(data)
        
        # Process data to add normalized values and class labels
        return process_data(df)

    @st.cache_data
    def load_osm_data():
        city_name = "Ben Guerir, Morocco"
        with st.spinner("Fetching OSM data for Ben Guerir..."):
            gdf = ox.features_from_place(city_name, tags={"building": True})
            gdf = gdf[gdf.geometry.type == "Polygon"]
            
            # Generate sample data for each building
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
            
            df = pd.DataFrame({
                'building_id': range(n),
                'CO2_Usage': gdf["CO2_Usage"],
                'Water_Usage': gdf["Water_Usage"],
                'Energy_Consumption': gdf["Energy_Consumption"],
                'height': gdf["height"],
                'latitude': gdf["latitude"],
                'longitude': gdf["longitude"],
                'polygon': gdf["polygon"]
            })
            
            # Process data to add normalized values and class labels
            return process_data(df)

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
    
    filtered_df = df[
        (df["CO2_Usage"] >= co2_min) & (df["CO2_Usage"] <= co2_max) &
        (df["Water_Usage"] >= water_min) & (df["Water_Usage"] <= water_max) &
        (df["Energy_Consumption"] >= energy_min) & (df["Energy_Consumption"] <= energy_max) &
        (df["height"] >= height_min) & (df["height"] <= height_max) &
        (df["class_label"].isin(selected_classes))
    ]
    st.write(f"Showing {len(filtered_df)} of {len(df)} buildings")

col1, col2 = st.columns([2, 1])

def get_color_mapping(values, metric):
    if metric == "class_label":
        class_colors = {
            'A': [0, 255, 0, 200],    # Green
            'B': [144, 238, 144, 200], # Light Green
            'C': [255, 255, 0, 200],   # Yellow
            'D': [255, 165, 0, 200],   # Orange
            'E': [255, 0, 0, 200],     # Red
            'F': [139, 0, 0, 200]      # Dark Red
        }
        return filtered_df[metric].map(class_colors).tolist()
    else:
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
        latitude=filtered_df["latitude"].mean() if not filtered_df.empty else 32.23,
        longitude=filtered_df["longitude"].mean() if not filtered_df.empty else -7.95,
        zoom=14,
        pitch=45,
    )
    deck = pdk.Deck(
        layers=[building_layer],
        initial_view_state=view_state,
        tooltip={
            "text": "Building ID: {building_id}\nClass: {class_label}\nHeight: {height}m\nCO₂: {CO2_Usage}kg\nWater: {Water_Usage}L\nEnergy: {Energy_Consumption}kWh"
        }
    )
    st.pydeck_chart(deck)

with col2:
    st.subheader("Metrics Overview")
    if not filtered_df.empty:
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Average CO₂ Usage", f"{filtered_df['CO2_Usage'].mean():.1f} kg")
            st.metric("Average Water Usage", f"{filtered_df['Water_Usage'].mean():.0f} L")
        with metric_col2:
            st.metric("Average Energy", f"{filtered_df['Energy_Consumption'].mean():.0f} kWh")
            most_common_class = filtered_df['class_label'].mode().iloc[0] if not filtered_df['class_label'].empty else "N/A"
            st.metric("Class Distribution", f"{most_common_class} (most common)")
    else:
        st.write("No buildings match the current filters.")
    
    if not filtered_df.empty:
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
            fig = px.bar(
                class_counts,
                x='Class',
                y='Count',
                color='Class',
                color_discrete_map=class_colors
            )
            fig.update_layout(
                xaxis_title="Building Class",
                yaxis_title="Number of Buildings",
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 3D Classification plot
        st.subheader("3D Classification Visualization")
        
        def plot_3d_classification(df):
            """3D classification visualization with spheres representing class boundaries."""
            if df is None or df.empty:
                st.write("No data available for visualization")
                return None
                
            class_colors = {
                'A': 'green', 'B': 'lightgreen', 'C': 'yellow',
                'D': 'orange', 'E': 'red', 'F': 'darkred'
            }
            
            fig = px.scatter_3d(
                df, 
                x='Energy_Consumption', 
                y='CO2_Usage',
                z='Water_Usage', 
                color='class_label',
                color_discrete_map=class_colors,
                labels={
                    'Energy_Consumption': 'Energy (kWh)',
                    'CO2_Usage': 'Carbon (kg)',
                    'Water_Usage': 'Water (L)',
                    'class_label': 'Class'
                },
                title="3D Classification of Buildings",
                hover_data={
                    'building_id': True,
                    'Energy_Consumption': ':.2f',
                    'CO2_Usage': ':.2f',
                    'Water_Usage': ':.2f',
                    'class_label': True,
                    'distance': ':.3f'
                }
            )
            
            fig.update_traces(marker=dict(size=4, opacity=0.8))
            
            # Use a sample of buildings for better performance
            if len(df) > 100:
                df_sample = df.sample(100)
            else:
                df_sample = df
                
            # Calculate the origin (0,0,0) point in the normalized space
            origin = [0, 0, 0]
            
            # Get the thresholds used for classification
            d_min = df['distance'].min()
            d_max = df['distance'].max()
            thresholds = np.linspace(d_min, d_max, 7)
            
            # Add spheres for class boundaries
            if len(df) > 0:
                for i, threshold in enumerate(thresholds[1:]):
                    class_label = chr(65 + i)
                    if class_label in selected_classes:  # Only show spheres for selected classes
                        # Create a lower resolution sphere mesh for better performance
                        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:20j]
                        
                        # Scale the spheres from the normalized space to the actual units
                        # This is an approximation - we're using the maximum values as scaling factors
                        energy_scale = df['Energy_Consumption'].max()
                        carbon_scale = df['CO2_Usage'].max()
                        water_scale = df['Water_Usage'].max()
                        
                        # Create the sphere
                        x = origin[0] + threshold * energy_scale * np.cos(u) * np.sin(v)
                        y = origin[1] + threshold * carbon_scale * np.sin(u) * np.sin(v)
                        z = origin[2] + threshold * water_scale * np.cos(v)
                        
                        
            
            fig.update_layout(
                scene=dict(
                    xaxis_title='Energy Consumption (kWh)',
                    yaxis_title='Carbon Footprint (kg)',
                    zaxis_title='Water Usage (L)'
                ),
                legend_title_text='Class',
                margin=dict(l=0, r=0, b=0, t=40),
                height=500
            )
            
            return fig
        
        fig = plot_3d_classification(filtered_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Relationship Between Metrics")
        x_metric = st.selectbox("X-Axis", ["CO2_Usage", "Water_Usage", "Energy_Consumption", "height", "distance"], key="x_axis")
        y_metric = st.selectbox("Y-Axis", ["Energy_Consumption", "CO2_Usage", "Water_Usage", "height", "distance"], key="y_axis")
        
        class_colors = {
            'A': 'green', 'B': 'lightgreen', 'C': 'yellow',
            'D': 'orange', 'E': 'red', 'F': 'darkred'
        }
        
        scatter = px.scatter(
            filtered_df,
            x=x_metric,
            y=y_metric,
            color="class_label" if color_by == "class_label" else color_by,
            color_discrete_map=class_colors if color_by == "class_label" else None,
            color_continuous_scale="Viridis" if color_by != "class_label" else None,
            hover_data=["building_id", "class_label", "CO2_Usage", "Water_Usage", "Energy_Consumption", "height", "distance"]
        )
        scatter.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_colorbar=dict(title=color_by) if color_by != "class_label" else None
        )
        st.plotly_chart(scatter, use_container_width=True)

if not filtered_df.empty:
    st.subheader("Building Data")
    st.dataframe(
        filtered_df[["building_id", "class_label", "CO2_Usage", "Water_Usage", "Energy_Consumption", "height", "distance", "latitude", "longitude"]], 
        use_container_width=True,
        hide_index=True
    )
else:
    st.subheader("Building Data")
    st.write("No buildings match the current filters.")

st.sidebar.markdown("---")
st.sidebar.subheader("How to use this dashboard")
st.sidebar.markdown("""
1. Use the sliders to filter buildings by metrics
2. Select which building classes to display using the Class Filter
3. Choose which metric to use for coloring the buildings
4. Explore the 3D visualization by dragging, zooming, and rotating
5. Analyze the 3D classification plot showing building efficiency classes
6. Examine relationships between metrics using the scatter plot
7. View detailed data in the table below
""")
st.subheader("Export Data")
col1, col2 = st.columns(2)

with col1:
    if st.button("Export to CSV"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="building_data.csv",
            mime="text/csv"
        )

with col2:
    if st.button("Generate Report Summary"):
        report = f"""# Building Analysis Report
Generated on {pd.Timestamp.now().strftime('%Y-%m-%d')}

## Summary
- Total Buildings: {len(filtered_df)}
- Average Energy Consumption: {filtered_df['Energy_Consumption'].mean():.1f} kWh
- Average CO2 Usage: {filtered_df['CO2_Usage'].mean():.1f} kg
- Average Water Usage: {filtered_df['Water_Usage'].mean():.1f} L

## Class Distribution
{filtered_df['class_label'].value_counts().to_string()}
"""
        st.download_button(
            label="Download Report",
            data=report,
            file_name="building_report.md",
            mime="text/markdown"
        )
        
def add_benchmark_comparison():
    st.subheader("Benchmark Comparison")
    
    # Define benchmarks (these would come from industry standards)
    benchmarks = {
        "Energy_Consumption": {
            "Excellent": 1000,
            "Good": 2000,
            "Average": 3000,
            "Poor": 4000
        },
        "CO2_Usage": {
            "Excellent": 100,
            "Good": 200,
            "Average": 300,
            "Poor": 400
        },
        "Water_Usage": {
            "Excellent": 2000,
            "Good": 4000,
            "Average": 6000,
            "Poor": 8000
        }
    }
    
    metric_for_benchmark = st.selectbox(
        "Select metric for benchmark comparison", 
        ["Energy_Consumption", "CO2_Usage", "Water_Usage"]
    )
    
    avg_value = filtered_df[metric_for_benchmark].mean()
    
    fig = go.Figure()
    
    # Add average value
    fig.add_trace(go.Indicator(
        mode = "number+gauge",
        value = avg_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Average {metric_for_benchmark}"},
        gauge = {
            'axis': {'range': [0, benchmarks[metric_for_benchmark]["Poor"]*1.2]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, benchmarks[metric_for_benchmark]["Excellent"]], 'color': 'green'},
                {'range': [benchmarks[metric_for_benchmark]["Excellent"], benchmarks[metric_for_benchmark]["Good"]], 'color': 'lightgreen'},
                {'range': [benchmarks[metric_for_benchmark]["Good"], benchmarks[metric_for_benchmark]["Average"]], 'color': 'yellow'},
                {'range': [benchmarks[metric_for_benchmark]["Average"], benchmarks[metric_for_benchmark]["Poor"]], 'color': 'orange'},
                {'range': [benchmarks[metric_for_benchmark]["Poor"], benchmarks[metric_for_benchmark]["Poor"]*1.2], 'color': 'red'}
            ]
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Add text interpretation
    if avg_value <= benchmarks[metric_for_benchmark]["Excellent"]:
        performance = "Excellent - Buildings are performing at top efficiency levels"
    elif avg_value <= benchmarks[metric_for_benchmark]["Good"]:
        performance = "Good - Buildings are performing well but have room for improvement"
    elif avg_value <= benchmarks[metric_for_benchmark]["Average"]:
        performance = "Average - Consider moderate efficiency improvements"
    elif avg_value <= benchmarks[metric_for_benchmark]["Poor"]:
        performance = "Below Average - Significant improvements recommended"
    else:
        performance = "Poor - Urgent efficiency measures required"
        
    st.write(f"**Performance Assessment:** {performance}")

# Add this call after your other visualizations
add_benchmark_comparison()