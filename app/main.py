import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
pd.set_option("styler.render.max_elements", 500_000)  

from models.mahalanobis import classify_mahalanobis
from scripts.validate_data import add_classifications, validate_and_preprocess_dataset
from visualization.map import display_map
from visualization.charts import display_relationship_plot, display_distribution_plot
from visualization.model_specific import display_model_visualization
from utils.metrics import display_metrics_overview
from utils.export import add_export_section, add_benchmark_comparison
from pathlib import Path

# at the top, alongside your other utils imports
from utils.building_selection import (
    setup_building_selection,
    display_clickable_map,
    display_building_classifications
)
 

# Set page config with icon and expanded layout
st.set_page_config(
    page_title="Building Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        /* Modern Enterprise UI Styles */
        .main {
            background-color: #f8f9fa;
        }
        
        .stApp {
            max-width: 1800px;
            margin: 0 auto;
        }
        
        .main-header {
            background: linear-gradient(90deg, #0052a5, #0077cc);
            color: white;
            padding: 1.5rem;
            border-radius: 0px;
            margin-bottom: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .card {
            background-color: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
            border: 1px solid #e0e0e0;
        }
        
        .metric-card {
            text-align: center;
            padding: 1rem;
            border-radius: 6px;
            background: white;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            border: 1px solid #e0e0e0;
            transition: transform 0.2s;
        }
        
        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 3px 6px rgba(0,0,0,0.1);
        }
        .metric-card h2, 
        .metric-card h3, 
        .metric-card p {
            color: #333 !important;
        }
        
        /* Clean up layout */
        .css-1d391kg {
            padding-top: 1rem;
        }
        
        /* Hide default Streamlit header and footer */
        #MainMenu, footer {visibility: hidden;}
        
        /* Custom tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            white-space: nowrap;
            background-color: #f1f3f4;
            border-radius: 4px 4px 0 0;
            padding: 0 16px;
            font-weight: 500;
            font-size: 0.9rem;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #0077cc !important;
            color: black !important;
        }
        
        /* Filter panel styling */
        .filter-panel {
            background-color: white;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            border: 1px solid #e0e0e0;
        }
        
        /* Larger map container */
        .map-container {
            height: 600px;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        
        /* Compact form elements */
        .stSlider, .stSelectbox, .stTextInput {
            margin-bottom: 0.5rem;
        }
        
        /* Status bar */
        .status-bar {
            background-color: #f1f3f4;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            margin-bottom: 1rem;
            font-size: 0.9rem;
        }
            /* Full screen map container */
    .full-map-container {
        position: relative;
        height: 100vh;
        width: 100%;
        overflow: hidden;
        margin: 0;
        padding: 0;

    }

    /* Slide-in panel styling */
    .slide-panel {
        position: absolute;
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 0 10px rgba(0,0,0,0.2);
        padding: 15px;
        transition: transform 0.3s ease-in-out;
        z-index: 1000;
        max-height: 85vh;
        overflow-y: auto;
    }

    .slide-panel-left {
        left: 0;
        top: 0;
        width: 300px;
        height: 100%;
        transform: translateX(-100%);
    }

    .slide-panel-right {
        right: 0;
        top: 0;
        width: 300px;
        height: 100%;
        transform: translateX(100%);
    }

    .slide-panel-visible-left {
        transform: translateX(0);
    }

    .slide-panel-visible-right {
        transform: translateX(0);
    }

    .panel-toggle {
        position: absolute;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 30px;
        height: 60px;
        background-color: white;
        border-radius: 0 8px 8px 0;
        box-shadow: 2px 0 5px rgba(0,0,0,0.1);
        cursor: pointer;
        z-index: 999;
    }

    .panel-toggle-left {
        left: 0;
        top: 50%;
        transform: translateY(-50%);
    }

    .panel-toggle-right {
        right: 0;
        top: 50%;
        transform: translateY(-50%);
        border-radius: 8px 0 0 8px;
    }

    
    </style>
""", unsafe_allow_html=True)

# App header with gradient
st.markdown('<div class="main-header"><h1 style="text-align: center;"> Building Analytics Dashboard</h1></div>', unsafe_allow_html=True)

# Initialize session state for comparison
if 'comparison_buildings' not in st.session_state:
    st.session_state['comparison_buildings'] = []


# Sidebar configuration
with st.sidebar:
    st.title("Dashboard Controls")
    
    # Dataset selection
    st.header("📊 Dataset Selection")
    dataset_option = st.selectbox(
        "Choose Dataset",
        ["Default (Lyon)", "Gordes", "Upload Custom Dataset"],
        key="dataset_option"
    )
    
    uploaded_file = None
    if dataset_option == "Upload Custom Dataset":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    # Load and validate dataset
    if dataset_option == "Default (Lyon)":
        with st.spinner("Loading Lyon data..."):
            df = pd.read_csv("data/reduced_lyon_buildings.csv")
            selected_city = "Lyon"
    elif dataset_option == "Gordes":
        with st.spinner("Loading Gordes data..."):
            try:
                df = pd.read_csv("data/reduced_gordes_buildings.csv")
                selected_city = "Gordes"
            except FileNotFoundError:
                st.error("Gordes dataset file 'data/reduced_gordes_buildings.csv' not found.")
                df = pd.read_csv("data/reduced_lyon_buildings.csv")
                selected_city = "Lyon"
                st.warning("Reverted to default Lyon dataset")
                st.write(f"Fallback dataset: {selected_city}")  # Debug
    else:  # Upload Custom Dataset
        if uploaded_file is not None:
            with st.spinner("Loading uploaded dataset..."):
                try:
                    df = pd.read_csv(uploaded_file)
                    selected_city = "Custom Dataset"
                except Exception as e:
                    st.error(f"Error loading CSV file: {str(e)}")
                    df = pd.read_csv("data/reduced_lyon_buildings.csv")
                    selected_city = "Lyon"
                    st.warning("Reverted to default Lyon dataset")
                    st.write(f"Fallback dataset: {selected_city}")  # Debug
        else:
            st.info("Please upload a CSV file to proceed.")
            df = pd.read_csv("data/reduced_lyon_buildings.csv")
            selected_city = "Lyon"
            st.write(f"Using default dataset: {selected_city}")  # Debug
            
    # Scoring basis
    st.subheader("Scoring Basis")
    scoring_basis = st.radio("Choose scoring basis", ["Total (kWh)", "Per m² (kWh/m²/year)"])
    
    # Validate and preprocess dataset
    df = validate_and_preprocess_dataset(df, scoring_basis)
    if df is None:
        df = pd.read_csv("data/reduced_lyon_buildings.csv")
        selected_city = "Lyon"
        st.warning("Invalid dataset. Reverted to default Lyon dataset")
        df = validate_and_preprocess_dataset(df, scoring_basis)
    
    # Replace metrics for intensity-based scoring
    if scoring_basis == "Per m² (kWh/m²/year)":
        if "Energy_Intensity" in df.columns and "CO2_Intensity" in df.columns:
            df["Energy_Consumption"] = df["Energy_Intensity"]
            df["CO2_Usage"] = df["CO2_Intensity"]
    
    # Feature selection for classification
    st.subheader("Classification Features")
    available_features = [
        "Energy_Consumption", "CO2_Usage", "Water_Usage",
        "Energy_Intensity", "CO2_Intensity"
    ]
    available_features = [f for f in available_features if f in df.columns]
    selected_features = st.multiselect(
        "Select 3 Features",
        options=available_features,
        default=["Energy_Consumption", "CO2_Usage", "Water_Usage"],
        max_selections=3,
        key="classification_features"
    )
    
    if len(selected_features) != 3:
        st.error("Please select exactly 3 features for classification.")
        st.stop()
    
    # Add classifications if missing
    df = add_classifications(df)
    
    # Cache in session state
    st.session_state["df"] = df
    st.session_state["city"] = selected_city
    
    # Classification method selection
    st.header("Analysis Method")
    classification_methods = {
        "Euclidean Distance": "Euclidean Distance",
        "Mahalanobis Distance": "Mahalanobis Distance",
        "PCA Classification": "PCA Classification",
        "Weighted Classification": "Weighted Classification",
        "Bayesian Classification": "Bayesian Classification"
    }
    
    classification_method = st.radio(
        "Select Classification Method",
        list(classification_methods.keys()),
        format_func=lambda x: classification_methods[x]
    )
    
    # Apply selected classification
    with st.spinner(f"Applying {classification_method}..."):
        class_column_mapping = {
            "Euclidean Distance": "class_euclidean",
            "Mahalanobis Distance": "class_mahalanobis",
            "PCA Classification": "class_pca",
            "Weighted Classification": "class_weighted",
            "Bayesian Classification": "class_bayesian"
        }
        selected_class_column = class_column_mapping[classification_method]
        
        # Apply the selected classification method
        if classification_method == "Euclidean Distance":
            from models.euclidean import classify_euclidean
            df = classify_euclidean(df, features=selected_features)
        elif classification_method == "Mahalanobis Distance":
            from models.mahalanobis import classify_mahalanobis
            df = classify_mahalanobis(df, features=selected_features, return_distance=True)
        elif classification_method == "PCA Classification":
            from models.pca import classify_pca
            df = classify_pca(df, features=selected_features)
        elif classification_method == "Weighted Classification":
            from models.weighted import classify_weighted
            df = classify_weighted(df, features=selected_features)
        elif classification_method == "Bayesian Classification":
            from models.bayesian import classify_bayesian
            df = classify_bayesian(df, features=selected_features)
        
        # Assign class_label from the mapped column
        if selected_class_column in df.columns:
            df["class_label"] = df[selected_class_column]
        else:
            st.error(f"Column '{selected_class_column}' not found after classification. Check the classification function.")
            st.stop()
        
        # Cache updated DataFrame
        st.session_state["df"] = df
        
with st.spinner(f"Applying {classification_method}..."):
    class_column_mapping = {
        "Euclidean Distance": "class_euclidean",
        "Mahalanobis Distance": "class_mahalanobis",
        "PCA Classification": "class_pca",
        "Weighted Classification": "class_weighted",
        "Bayesian Classification": "class_bayesian"
    }
    selected_class_column = class_column_mapping[classification_method]
    
    # Apply the selected classification method
    features = ["Energy_Consumption", "CO2_Usage", "Water_Usage"]
    if classification_method == "Euclidean Distance":
        from models.euclidean import classify_euclidean
        df = classify_euclidean(df, features=features)
    elif classification_method == "Mahalanobis Distance":
        from models.mahalanobis import classify_mahalanobis
        df = classify_mahalanobis(df, features=features, return_distance=True)
    elif classification_method == "PCA Classification":
        from models.pca import classify_pca
        df = classify_pca(df, features=features)
    elif classification_method == "Weighted Classification":
        from models.weighted import classify_weighted
        df = classify_weighted(df, features=features)
    elif classification_method == "Bayesian Classification":
        from models.bayesian import classify_bayesian
        df = classify_bayesian(df, features=features)
    
    # Assign class_label from the mapped column
    if selected_class_column in df.columns:
        df["class_label"] = df[selected_class_column]
    else:
        st.error(f"Column '{selected_class_column}' not found after classification. Check the classification function.")
        st.stop()
    
    # Cache updated DataFrame
    st.session_state["df"] = df
        

# Check if empty
if df.empty:
    st.error("No data available. Please check your dataset.")
    st.stop()
# Advanced filters in an expander
with st.expander("🔍 Advanced Filters", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        co2_min, co2_max = st.slider(
            "CO₂ Usage (kg)", 
            float(df["CO2_Usage"].min()), 
            float(df["CO2_Usage"].max()), 
            (float(df["CO2_Usage"].min()), float(df["CO2_Usage"].max()))
        )
        
        water_min, water_max = st.slider(
            "Water Usage (L)", 
            float(df["Water_Usage"].min()), 
            float(df["Water_Usage"].max()), 
            (float(df["Water_Usage"].min()), float(df["Water_Usage"].max()))
        )
    
    with col2:
        energy_min, energy_max = st.slider(
            "Energy (kWh)", 
            float(df["Energy_Consumption"].min()), 
            float(df["Energy_Consumption"].max()), 
            (float(df["Energy_Consumption"].min()), float(df["Energy_Consumption"].max()))
        )
        
    
    # Class filter with colored chips
    st.subheader("Building Class Filter")
    all_classes = ['A', 'B', 'C', 'D', 'E', 'F']
    class_colors = {
        'A': '#28a745', 'B': '#5cb85c', 'C': '#ffc107',
        'D': '#fd7e14', 'E': '#dc3545', 'F': '#6c757d'
    }
    
    class_cols = st.columns(6)
    selected_classes = []
    
    for i, cls in enumerate(all_classes):
        with class_cols[i]:
            if st.checkbox(f"Class {cls}", value=True, key=f"class_{cls}"):
                selected_classes.append(cls)
    
    color_by = st.selectbox(
        "Color Buildings By", 
        ["class_label", "CO2_Usage", "Water_Usage", "Energy_Consumption"],
        format_func=lambda x: {
            "class_label": "Energy Class",
            "CO2_Usage": "CO₂ Emissions",
            "Water_Usage": "Water Consumption",
            "Energy_Consumption": "Energy Usage",
        }.get(x, x)
    )

# Filter the dataframe
filtered_df = df[
    (df["CO2_Usage"] >= co2_min) & (df["CO2_Usage"] <= co2_max) &
    (df["Water_Usage"] >= water_min) & (df["Water_Usage"] <= water_max) &
    (df["Energy_Consumption"] >= energy_min) & (df["Energy_Consumption"] <= energy_max) &
    (df["class_label"].isin(selected_classes))
]

# Info bar
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    st.info(f" Showing {len(filtered_df)} of {len(df)} buildings in {selected_city}")
with col2:
    st.metric(
        "Average Energy Class", 
        f"{filtered_df['class_label'].mode()[0] if not filtered_df.empty else 'N/A'}",
        delta=None
    )
with col3:
    st.metric(
        "Data Last Updated", 
        datetime.now().strftime("%Y-%m-%d"),
        delta=None
    )

# Main tabs for better organization
tab1, tab2, tab3, tab4 = st.tabs([
    " Interactive Map", 
    " Analytics & Insights", 
    " Building Data", 
    " Export & Reports"
])

with tab1:
    # Map section
    st.markdown(f"<h2 style='text-align: center;'>Interactive Map: {selected_city}</h2>", unsafe_allow_html=True)
    
    # Initialize session state for panel visibility if not exists
    if 'left_panel_visible' not in st.session_state:
        st.session_state['left_panel_visible'] = False
    if 'right_panel_visible' not in st.session_state:
        st.session_state['right_panel_visible'] = False
    
    
    # Create columns for the three sections
    main_container = st.container()
    
    

    
    # Right panel content (Building details)
    right_col = st.container()
    with right_col:
        st.subheader("Building Details")
        
        if not filtered_df.empty:
            building_id = st.selectbox(
                "Select Building",
                filtered_df["building_id"].tolist(),
                format_func=lambda x: f"Building {x}"
            )
            
            selected_building = filtered_df[filtered_df["building_id"] == building_id]
            if not selected_building.empty:
                st.markdown(f"**Class:** {selected_building['class_label'].values[0]}")
                st.markdown(f"**CO₂:** {selected_building['CO2_Usage'].values[0]:.1f}kg")
                st.markdown(f"**Energy:** {selected_building['Energy_Consumption'].values[0]:.1f}kWh")
                
                # Add to comparison
                if st.button("➕ Add to Comparison"):
                    if building_id not in [b["building_id"] for b in st.session_state['comparison_buildings']]:
                        st.session_state['comparison_buildings'].append(selected_building.iloc[0].to_dict())
                        st.success("Added to comparison")
                    else:
                        st.warning("Already in comparison")
    
        # Main content - Map + detail sidebar
    with main_container:
        if not filtered_df.empty:
            selected_building_id = display_clickable_map(filtered_df, selected_city, color_by)
            
            with st.container():
                st.markdown("<hr>", unsafe_allow_html=True)
                display_building_classifications(df, selected_building_id)
        else:
            st.warning("No buildings match the current filters")



with tab2:
    # Analytics
    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
        " Class Distribution", 
        " Classification Results", 
        " Relationships"
    ])
    
    with analysis_tab1:
        st.markdown("### Building Class Distribution")
        if not filtered_df.empty:
            display_distribution_plot(filtered_df, color_by)
            
            # Class metrics
            st.markdown("### Class Breakdown")
            class_counts = filtered_df["class_label"].value_counts().reset_index()
            class_counts.columns = ["Class", "Count"]
            
            class_metrics = st.columns(len(class_counts))
            for i, (_, row) in enumerate(class_counts.iterrows()):
                with class_metrics[i]:
                    st.markdown(
                        f"<div class='metric-card' style='border-left: 5px solid {class_colors.get(row['Class'], '#777')};'>"
                        f"<h3 style='color: #333;'>Class {row['Class']}</h3>"
                        f"<h2 style='color: #333;'>{row['Count']}</h2>"
                        f"<p style='color: #333;'>{row['Count']/len(filtered_df)*100:.1f}% of buildings</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
    
    with analysis_tab2:
        st.markdown("### Classification Visualization")
        if not filtered_df.empty:
            display_model_visualization(filtered_df, classification_method)
            st.markdown("### Classification Summary")
            st.write(f"Using **{classification_method}** to classify buildings in {selected_city}")
            
            # Metrics overview
            st.markdown('<div class="card">', unsafe_allow_html=True)
            display_metrics_overview(filtered_df)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with analysis_tab3:
        st.markdown("### Metric Relationships")
        if not filtered_df.empty:
            display_relationship_plot(filtered_df, color_by)
            
            # Correlation heatmap
            st.markdown("### Correlation Analysis")
            numeric_df = filtered_df.select_dtypes(include=[np.number])
            corr = numeric_df.corr()
            
            fig = {
                "data": [{
                    "type": "heatmap",
                    "z": corr.values,
                    "x": corr.columns,
                    "y": corr.columns,
                    "colorscale": "Blues"
                }],
                "layout": {
                    "title": "Feature Correlation Matrix",
                    "height": 500
                }
            }
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Building data
    st.markdown("### Building Data Explorer")
    
    if not filtered_df.empty:
        # Search and sort options
        search_col, sort_col = st.columns(2)
        with search_col:
            search_term = st.text_input("🔍 Search by Building ID")
        
        with sort_col:
            sort_by = st.selectbox(
                "Sort By", 
                ["building_id", "class_label", "CO2_Usage", "Water_Usage", "Energy_Consumption"]
            )
            ascending = st.checkbox("Ascending", value=True)
        
        # Filter by search and sort
        if search_term:
            display_df = filtered_df[filtered_df["building_id"].astype(str).str.contains(search_term)]
        else:
            display_df = filtered_df
        
        display_df = display_df.sort_values(by=sort_by, ascending=ascending)
        
        # Limit rows for better rendering (e.g., 300 rows only)
        preview_df = display_df.head(300)

        st.dataframe(
            preview_df.style.format({
                "CO2_Usage": "{:.2f}",
                "Water_Usage": "{:.2f}",
                "Energy_Consumption": "{:.2f}",
                "latitude": "{:.5f}",
                "longitude": "{:.5f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        
        # Building comparison section
        if st.session_state['comparison_buildings']:
            st.markdown("### Building Comparison")
            
            # Create comparison dataframe
            comparison_df = pd.DataFrame(st.session_state['comparison_buildings'])
            
            # Display comparison
            st.dataframe(
                comparison_df[[
                    "building_id", "class_label", "CO2_Usage", "Water_Usage",
                    "Energy_Consumption"
                ]].style.format({
                    "CO2_Usage": "{:.2f}",
                    "Water_Usage": "{:.2f}",
                    "Energy_Consumption": "{:.2f}"                }),
                use_container_width=True,
                hide_index=True
            )
            
            # Comparison chart
            if len(st.session_state['comparison_buildings']) > 1:
                metrics_to_compare = ["CO2_Usage", "Water_Usage", "Energy_Consumption"]
                
                for metric in metrics_to_compare:
                    fig = {
                        "data": [{
                            "type": "bar",
                            "x": comparison_df["building_id"],
                            "y": comparison_df[metric],
                            "marker": {"color": comparison_df["class_label"].map(class_colors)}
                        }],
                        "layout": {
                            "title": f"{metric} Comparison",
                            "xaxis": {"title": "Building ID"},
                            "yaxis": {"title": metric}
                        }
                    }
                    st.plotly_chart(fig, use_container_width=True)
            
            if st.button("Clear Comparison"):
                st.session_state['comparison_buildings'] = []
                st.rerun()


    else:
        st.warning("No buildings match the current filters")

with tab4:
    # Export and reports
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        st.markdown("### Export Filtered Data")
        add_export_section(filtered_df)
    
    with export_col2:
        st.markdown("### Generate Report")
        report_name = st.text_input("Report Name", f"{selected_city} Building Analysis")
        include_map = st.checkbox("Include Map", value=True)
        include_analytics = st.checkbox("Include Analytics", value=True)
        include_raw_data = st.checkbox("Include Raw Data", value=False)
        
        if st.button("Generate PDF Report"):
            with st.spinner("Generating report..."):
                # This would typically connect to a report generation function
                st.success(f"Report '{report_name}' generated successfully!")
                st.download_button(
                    label="Download Report",
                    data=b"This would be a PDF report",  # Replace with actual PDF data
                    file_name=f"{report_name.replace(' ', '_')}.pdf",
                    mime="application/pdf"
                )
    
    # Benchmarks section
    
    add_benchmark_comparison(filtered_df)



# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 30px; padding: 10px; background-color: #f1f3f4; border-radius: 5px;">
        <p style="margin: 0; color: #555;">Building Analytics Dashboard • Created with ❤️ • Data updated: April 2025</p>
    </div>
""", unsafe_allow_html=True)