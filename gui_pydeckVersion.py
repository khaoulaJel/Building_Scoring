import streamlit as st 
import pandas as pd
import numpy as np
import pydeck as pdk
import osmnx as ox
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from joblib import Parallel, delayed
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix


# --------------------------------------------------------------------------------
# 1) CLASSIFICATION FUNCTIONS
# --------------------------------------------------------------------------------

def classify_euclidean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Original Euclidean distance-based classification (the same process_data logic you had).
    Produces a 'class_label' column with A-F.
    """
    df['energy_norm'] = df['Energy_Consumption'] / df['Energy_Consumption'].max()
    df['carbon_norm'] = df['CO2_Usage'] / df['CO2_Usage'].max()
    df['water_norm'] = df['Water_Usage'] / df['Water_Usage'].max()

    df['distance'] = np.sqrt(df['energy_norm']**2 + df['carbon_norm']**2 + df['water_norm']**2)

    thresholds = np.linspace(df['distance'].min(), df['distance'].max(), 7)
    df['class'] = np.digitize(df['distance'], thresholds[1:], right=True)

    class_labels = ['A', 'B', 'C', 'D', 'E', 'F']
    df['class_label'] = df['class'].apply(lambda x: class_labels[min(x, len(class_labels)-1)])
    return df


def classify_mahalanobis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mahalanobis-distance-based classification. Produces 'class_label' with A-F.
    """
    kpi_cols = ["CO2_Usage", "Water_Usage", "Energy_Consumption"]
    
    optimal_point = np.array([
        df["CO2_Usage"].min(),
        df["Water_Usage"].min(),
        df["Energy_Consumption"].min()
    ])
    
    cov_matrix = df[kpi_cols].cov().values
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    
    def mahalanobis_distance(row_vector, optimal_pt, inv_cov):
        return mahalanobis(row_vector, optimal_pt, inv_cov)
    
    df["Mahalanobis_Distance"] = df.apply(
        lambda row: mahalanobis_distance(row[kpi_cols], optimal_point, inv_cov_matrix),
        axis=1
    )
    
    num_classes = 6
    df["Global_Class"] = pd.cut(
        df["Mahalanobis_Distance"],
        bins=num_classes,
        labels=['A', 'B', 'C', 'D', 'E', 'F']
    ).astype(str)
    
    df['class_label'] = df['Global_Class']
    return df

def classify_bayesian(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bayesian Network-based classification of buildings into A-F.
    Produces df['class_label'] with A-F classes, plus 
    df['Bayesian_Certainty'] for the probability of that class.
    """

    # Rename df to real_data internally, for clarity with your snippet
    real_data = df.copy()

    # 1) Optimal reference point (minimum CO₂, Water, Energy)
    optimal_point = np.array([
        real_data["CO2_Usage"].min(),
        real_data["Water_Usage"].min(),
        real_data["Energy_Consumption"].min()
    ])

    # 2) Discretize with KBinsDiscretizer
    n_bins = 5
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    features = real_data[["CO2_Usage", "Water_Usage", "Energy_Consumption"]].values
    discretized_features = discretizer.fit_transform(features)
    discretized_data = pd.DataFrame(
        discretized_features,
        columns=["CO2_Level", "Water_Level", "Energy_Level"]
    )

    # 3) Create an initial class assignment (Euclidean from optimal)
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_optimal = scaler.transform([optimal_point])[0]
    distances = np.sqrt(np.sum((scaled_features - scaled_optimal)**2, axis=1))

    num_classes = 6  # A-F
    class_labels = ['A', 'B', 'C', 'D', 'E', 'F']
    bins = np.linspace(distances.min(), distances.max(), num_classes + 1)
    initial_classes = np.digitize(distances, bins)
    initial_classes = np.clip(initial_classes, 1, num_classes) - 1  # 0-based
    discretized_data["Class"] = [class_labels[i] for i in initial_classes]

    # 4) Define a Bayesian Network structure
    model = BayesianNetwork([
        ('CO2_Level', 'Class'),
        ('Water_Level', 'Class'),
        ('Energy_Level', 'Class'),
        ('CO2_Level', 'Energy_Level'),
        ('Water_Level', 'Energy_Level')
    ])

    # 5) Fit the BN with MaximumLikelihoodEstimator
    model.fit(discretized_data, estimator=MaximumLikelihoodEstimator)

    # 6) Infer class probabilities for each building
    inference = VariableElimination(model)

    def get_class_probabilities(co2_level, water_level, energy_level):
        evidence = {
            'CO2_Level': co2_level,
            'Water_Level': water_level,
            'Energy_Level': energy_level
        }
        return inference.query(variables=['Class'], evidence=evidence)

    # Create arrays to hold results
    bayesian_classes = []
    certainty_scores = []

    for row in discretized_features:
        co2_level, water_level, energy_level = row
        prob_dist = get_class_probabilities(co2_level, water_level, energy_level)
        
        # Extract class probabilities
        probs = {}
        for j, state in enumerate(prob_dist.state_names['Class']):
            probs[state] = prob_dist.values[j]
        
        # Pick the most likely class
        best_class = max(probs, key=probs.get)
        bayesian_classes.append(best_class)

        # Certainty = probability of that top class
        certainty_scores.append(probs[best_class])

    # Attach results
    real_data['Bayesian_Class'] = bayesian_classes
    real_data['Bayesian_Certainty'] = certainty_scores

    # For uniformity with other classification methods,
    # we store final classification in 'class_label'.
    real_data['class_label'] = real_data['Bayesian_Class']

    return real_data

def classify_pca(df: pd.DataFrame) -> pd.DataFrame:
    """
    PCA-based classification where lower PC1 => better performance.
    Produces 'class_label' with A-F.
    """
    kpi_cols = ["CO2_Usage", "Water_Usage", "Energy_Consumption"]
    scaler = MinMaxScaler()
    norm_cols = [f"{col}_norm" for col in kpi_cols]
    df[norm_cols] = scaler.fit_transform(df[kpi_cols])
    
    pca = PCA(n_components=1)
    df["PC1"] = pca.fit_transform(df[norm_cols])
    
    corr_CO2 = np.corrcoef(df["PC1"], df["CO2_Usage_norm"])[0, 1]
    if corr_CO2 < 0:
        df["PC1"] = -df["PC1"]
    
    percentiles = df["PC1"].quantile([0.1, 0.3, 0.6, 0.8, 0.9])
    
    def classify_pca_score(score):
        if score <= percentiles[0.1]:
            return "A"
        elif score <= percentiles[0.3]:
            return "B"
        elif score <= percentiles[0.6]:
            return "C"
        elif score <= percentiles[0.8]:
            return "D"
        elif score <= percentiles[0.9]:
            return "E"
        else:
            return "F"
    
    df["PCA_Class"] = df["PC1"].apply(classify_pca_score)
    df['class_label'] = df['PCA_Class']
    return df


def classify_weighted(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted classification of KPIs. Lower 'Global_Score' => better (A).
    """
    kpi_cols = ["CO2_Usage", "Water_Usage", "Energy_Consumption"]
    scaler = MinMaxScaler()
    norm_cols = [f"{col}_norm" for col in kpi_cols]
    df[norm_cols] = scaler.fit_transform(df[kpi_cols])
    
    weights = np.array([0.2, 0.3, 0.5])  # Example weighting
    df["Global_Score"] = np.sum(df[norm_cols] * weights, axis=1)
    
    percentiles = df["Global_Score"].quantile([0.1, 0.3, 0.6, 0.8, 0.9])
    def classify_score(score):
        if score <= percentiles[0.1]:
            return "A"
        elif score <= percentiles[0.3]:
            return "B"
        elif score <= percentiles[0.6]:
            return "C"
        elif score <= percentiles[0.8]:
            return "D"
        elif score <= percentiles[0.9]:
            return "E"
        else:
            return "F"
    df["Global_Class"] = df["Global_Score"].apply(classify_score)
    df['class_label'] = df['Global_Class']
    return df


# --------------------------------------------------------------------------------
# 2) STREAMLIT APP SETUP
# --------------------------------------------------------------------------------
st.set_page_config(page_title="Building Analytics Dashboard", layout="wide")
st.title("Building Analytics Dashboard")

with st.sidebar:
    @st.cache_data
    def load_osm_data(city_name):
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
        ["Euclidean Distance", "Mahalanobis Distance", "PCA Classification", "Weighted Classification","Bayesian Classification"]
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


# --------------------------------------------------------------------------------
# 3) MAIN LAYOUT: MAP & CHARTS
# --------------------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

def get_color_mapping(values, metric):
    if metric == "class_label":
        class_colors = {
            'A': [0, 255, 0, 200],
            'B': [144, 238, 144, 200],
            'C': [255, 255, 0, 200],
            'D': [255, 165, 0, 200],
            'E': [255, 0, 0, 200],
            'F': [139, 0, 0, 200]
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
    st.subheader(f"Building Distribution in {city_name}")
    if not filtered_df.empty:
        # We'll keep the PyDeck 3D only for demonstration (often used with Euclidean).
        # You can remove or adapt it if you prefer a single approach for all.
        colors = get_color_mapping(filtered_df[color_by], color_by)
        cdf = filtered_df.copy()
        cdf["color"] = colors

        building_layer = pdk.Layer(
            "PolygonLayer",
            cdf,
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
    else:
        st.write("No buildings match the current filters.")
        
# --------------------------------------------------------
        # 3D Classification Visualization (CHANGES per method)
        # --------------------------------------------------------
    st.subheader("Classification-Specific Visualization")

        # We'll do a big conditional on classification_method:
    if classification_method == "Euclidean Distance":
        # -- Example: Use the Plotly 3D scatter (similar to your existing code) --
        euc_fig = px.scatter_3d(
            filtered_df,
            x='Energy_Consumption',
            y='CO2_Usage',
            z='Water_Usage',
            color='class_label',
            hover_data=['building_id', 'class_label']
        )
        euc_fig.update_traces(marker=dict(size=4, opacity=0.8))
        st.plotly_chart(euc_fig, use_container_width=True)

    elif classification_method == "Mahalanobis Distance":
        # -- Example: 3D classification using Mahalanobis distance --
        # We'll show the snippet you provided
        # (We've already calculated 'Mahalanobis_Distance' & 'class_label'.)
        maha_fig = px.scatter_3d(
            filtered_df,
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
        maha_fig.update_traces(marker=dict(size=6, opacity=0.8))
        st.plotly_chart(maha_fig, use_container_width=True)

    elif classification_method == "PCA Classification":
        # -- PCA-based hist & pairplot (Seaborn) --
        # 1) Distribution of PCA Classes
        pca_class_colors = {
            "A": "green",
            "B": "lightgreen",
            "C": "yellow",
            "D": "orange",
            "E": "red",
            "F": "darkred",
        }
        fig_w1, ax_w1 = plt.subplots(figsize=(8, 6))
        sns.histplot(
            data=filtered_df,
            x="PC1",
            bins=30,
            kde=True,
            hue="PCA_Class",
            palette=pca_class_colors,
            multiple="stack",
            ax=ax_w1
        )
        ax_w1.set_title("Distribution of PC1 Scores with PCA Class Coloring\n(Lower PC1 = Better Performance)")
        ax_w1.set_xlabel("PC1 Score")
        ax_w1.set_ylabel("Frequency")
        st.pyplot(fig_w1)
        
    elif classification_method == "Weighted Classification":
        # -- Weighted classification distribution plots --
        global_class_colors = {
            "A": "green",
            "B": "lightgreen",
            "C": "yellow",
            "D": "orange",
            "E": "red",
            "F": "darkred",
        }
        fig_w2, ax_w2 = plt.subplots(figsize=(10, 6))
        sns.histplot(
            data=filtered_df,
            x="Global_Score",
            bins=30,
            kde=True,
            hue="Global_Class",
            palette=global_class_colors,
            multiple="stack",
            ax=ax_w2
        )
        ax_w2.set_title("Distribution of Global Scores with Global Class Coloring")
        ax_w2.set_xlabel("Global Score")
        ax_w2.set_ylabel("Frequency")
        ax_w2.legend(title="Global Class")
        st.pyplot(fig_w2)
        
    elif classification_method == "Bayesian Classification":
        # Create a Plotly 3D scatter for Bayesian classes
        fig_bayes = px.scatter_3d(
            filtered_df, 
            x='CO2_Usage',
            y='Water_Usage',
            z='Energy_Consumption',
            color='class_label',  # which is 'Bayesian_Class'
            size='Bayesian_Certainty',  # highlight certainty
            size_max=15,
            color_discrete_map={
                'A': 'green', 'B': 'lightgreen', 'C': 'yellow',
                'D': 'orange', 'E': 'red', 'F': 'darkred'
            },
            labels={
                'CO2_Usage': 'CO₂ Usage',
                'Water_Usage': 'Water Usage',
                'Energy_Consumption': 'Energy Consumption',
                'class_label': 'Class',
                'Bayesian_Certainty': 'Certainty'
            },
            title="3D Classification (Bayesian Network)",
            hover_data={
                'Bayesian_Certainty': ':.2f',
                'class_label': True
            }
        )
        # Optionally add the 'optimal_point' if you want
        fig_bayes.update_traces(marker=dict(opacity=0.8))
        st.plotly_chart(fig_bayes, use_container_width=True)


    # --------------------------------------------------------
    # Relationship Plot
    # --------------------------------------------------------
    st.subheader("Relationship Between Metrics")
    x_metric = st.selectbox(
        "X-Axis",
        ["CO2_Usage", "Water_Usage", "Energy_Consumption", "height"],
        key="x_axis"
    )
    y_metric = st.selectbox(
        "Y-Axis",
        ["Energy_Consumption", "CO2_Usage", "Water_Usage", "height"],
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
            "Energy_Consumption", "height"
        ]
    )
    scatter.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        coloraxis_colorbar=dict(title=color_by) if color_by != "class_label" else None
    )
    st.plotly_chart(scatter, use_container_width=True)

# Distribution of "color_by"
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

    

with col2:
    st.subheader("Metrics Overview")
    if not filtered_df.empty:
        mc1, mc2 = st.columns(2)
        with mc1:
            st.metric("Average CO₂ Usage", f"{filtered_df['CO2_Usage'].mean():.1f} kg")
            st.metric("Average Water Usage", f"{filtered_df['Water_Usage'].mean():.0f} L")
        with mc2:
            st.metric("Average Energy", f"{filtered_df['Energy_Consumption'].mean():.0f} kWh")
            most_common_class = filtered_df['class_label'].mode().iloc[0]
            st.metric("Class Distribution", f"{most_common_class} (most common)")
    else:
        st.write("No buildings match the current filters.")
        
        
    
        

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

# --------------------------------------------------------------------------------
# 4) Export + Usage Instructions
# --------------------------------------------------------------------------------
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

st.subheader("Export Data")
col1, col2 = st.columns(2)

with col1:
    if st.button("Export to CSV"):
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
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

# --------------------------------------------------------------------------------
# 5) BENCHMARK COMPARISON
# --------------------------------------------------------------------------------
def add_benchmark_comparison():
    st.subheader("Benchmark Comparison")

    if filtered_df.empty:
        st.write("No buildings match the current filters.")
        return

    benchmarks = {
        "Energy_Consumption": {"Excellent": 1000,"Good": 2000,"Average": 3000,"Poor": 4000},
        "CO2_Usage":          {"Excellent": 100, "Good": 200, "Average": 300, "Poor": 400},
        "Water_Usage":        {"Excellent": 2000,"Good": 4000,"Average": 6000,"Poor": 8000}
    }

    metric_for_benchmark = st.selectbox(
        "Select metric for benchmark comparison",
        ["Energy_Consumption", "CO2_Usage", "Water_Usage"]
    )

    avg_value = filtered_df[metric_for_benchmark].mean()

    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode="number+gauge",
        value=avg_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Average {metric_for_benchmark}"},
        gauge={
            'axis': {'range': [0, benchmarks[metric_for_benchmark]["Poor"] * 1.2]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0,   benchmarks[metric_for_benchmark]["Excellent"]], 'color': 'green'},
                {'range': [benchmarks[metric_for_benchmark]["Excellent"], benchmarks[metric_for_benchmark]["Good"]], 'color': 'lightgreen'},
                {'range': [benchmarks[metric_for_benchmark]["Good"], benchmarks[metric_for_benchmark]["Average"]], 'color': 'yellow'},
                {'range': [benchmarks[metric_for_benchmark]["Average"], benchmarks[metric_for_benchmark]["Poor"]], 'color': 'orange'},
                {'range': [benchmarks[metric_for_benchmark]["Poor"], benchmarks[metric_for_benchmark]["Poor"] * 1.2], 'color': 'red'}
            ]
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Performance interpretation
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

add_benchmark_comparison()
