import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import logging
# Configure logging to suppress INFO messages from pgmpy
logging.getLogger('pgmpy').setLevel(logging.WARNING)

pd.set_option("styler.render.max_elements", 500_000)  

from models.mahalanobis import classify_mahalanobis
from models.pca import classify_pca
from models.weighted import classify_weighted
from models.bayesian import classify_bayesian
from models.manhattan import classify_manhattan
from scripts.validate_data import add_classifications, validate_and_preprocess_dataset, ensure_classifications
from visualization.map import display_map
from visualization.charts import display_relationship_plot, display_distribution_plot
from visualization.model_specific import display_model_visualization
from utils.metrics import display_metrics_overview
from utils.export import add_export_section, add_benchmark_comparison
from utils.css_loader import load_css
from pathlib import Path

from utils.building_selection import (
    display_building_lookup,
    display_building_classifications,
)
from scripts.compute_feature_ranges import compute_ranges

# Set page config with icon and expanded layout
st.set_page_config(
    page_title="Building Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
CITY_PATHS = {
    "Lyon":   "data/reduced_lyon_buildings_all_years.csv",
    "Gordes": "data/reduced_gordes_buildings_all_years.csv",
    # add more cities here as needed
}
@st.cache_data
def load_city_data(city_name):
    df_city = pd.read_csv(CITY_PATHS[city_name])
    return validate_and_preprocess_dataset(df_city, scoring_basis)

# Load CSS from external file
load_css("styles.css")

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
            df = pd.read_csv("data/reduced_lyon_buildings_all_years.csv")
            df = df[df["year"] == datetime.now().year]  # ✨ Keep only current year
            selected_city = "Lyon"

    elif dataset_option == "Gordes":
        with st.spinner("Loading Gordes data..."):
            try:
                df = pd.read_csv("data/reduced_gordes_buildings_all_years.csv")
                df = df[df["year"] == datetime.now().year]
                selected_city = "Gordes"
            except FileNotFoundError:
                st.error("Gordes dataset file 'data/reduced_gordes_buildings_all_years.csv' not found.")
                df = pd.read_csv("data/reduced_lyon_buildings_all_years.csv")
                selected_city = "Lyon"
                st.warning("Reverted to default Lyon dataset")
                st.write(f"Fallback dataset: {selected_city}")  # Debug
    else:  # Upload Custom Dataset
        if uploaded_file is not None:
            with st.spinner("Loading uploaded dataset..."):
                try:
                    df = pd.read_csv(uploaded_file)
                    df = df[df["year"] == datetime.now().year]
                    selected_city = "Custom Dataset"
                except Exception as e:
                    st.error(f"Error loading CSV file: {str(e)}")
                    df = pd.read_csv("data/reduced_lyon_buildings_all_years.csv")
                    df = df[df["year"] == datetime.now().year]
                    selected_city = "Lyon"
                    st.warning("Reverted to default Lyon dataset")
                    st.write(f"Fallback dataset: {selected_city}")  # Debug
        else:
            st.info("Please upload a CSV file to proceed.")
            df = pd.read_csv("data/reduced_lyon_buildings_all_years.csv")
            df = df[df["year"] == datetime.now().year]
            selected_city = "Lyon"
            st.write(f"Using default dataset: {selected_city}")  # Debug
            
    # Scoring basis
    st.subheader("Scoring Basis")
    scoring_basis = st.radio(
        "Choose scoring basis",
        ["Total (kWh)", "Per m² (kWh/m²/year)"]
    )

    # Validate and preprocess dataset
    df = validate_and_preprocess_dataset(df, scoring_basis)
    if df is None:
        df = pd.read_csv("data/reduced_lyon_buildings_all_years.csv")
        selected_city = "Lyon"
        st.warning("Invalid dataset. Reverted to default Lyon dataset")
        df = validate_and_preprocess_dataset(df, scoring_basis)

    # Replace metrics for intensity‐based scoring
    if scoring_basis == "Per m² (kWh/m²/year)":
        if "Energy_Intensity" in df.columns and "CO2_Intensity" in df.columns:
            df["Energy_Consumption"] = df["Energy_Intensity"]
            df["CO2_Usage"] = df["CO2_Intensity"]

    st.subheader("Classification Features")
    available_features = [
        "Energy_Consumption", "CO2_Usage", "Water_Usage",
        "Energy_Intensity", "CO2_Intensity"
    ]
    available_features = [f for f in available_features if f in df.columns]

    selected_features = st.multiselect(
        "Select Features for Classification",
        options=available_features,
        default=["Energy_Consumption", "CO2_Usage"],
        # no max_selections → user can pick as many as they want
        key="classification_features"
    )

    # enforce at least one feature
    if len(selected_features) < 1:
        st.error("Please select at least one feature for classification.")
        st.stop()


    # Cache in session state
    st.session_state["df"]   = df
    st.session_state["city"] = selected_city

    # Classification method selection
    st.header("Analysis Method")
    classification_methods = {
        "Manhattan Distance"    : "Manhattan Distance",
        "Mahalanobis Distance"  : "Mahalanobis Distance",
        "PCA Classification"    : "PCA Classification",
        "Weighted Classification": "Weighted Classification",
        "Bayesian Classification": "Bayesian Classification",
        "Topsis"             : "Topsis",
    }

    classification_method = st.radio(
        "Select Classification Method",
        options=list(classification_methods.keys()),
        format_func=lambda x: classification_methods[x]
    )
    
    weights = None
    if classification_method == "Weighted Classification":
        st.subheader("Enter weights for each feature")
        weights = []
        # Show a number_input for each selected feature:
        for feat in selected_features:
            w = st.number_input(
                f"Weight for {feat}",
                min_value=0.0, 
                max_value=1.0, 
                value=round(1/len(selected_features), 2),
                step=0.01,
                key=f"weight_{feat}"
            )
            weights.append(w)
        # Pass the user's list straight into add_classifications
    df = ensure_classifications(df, selected_features, weights)

        
    # Apply selected classification
    with st.spinner(f"Applying {classification_method}..."):
        class_column_mapping = {
            "Manhattan Distance": "class_manhattan",
            "Mahalanobis Distance": "class_mahalanobis",
            "PCA Classification": "class_pca",
            "Weighted Classification": "class_weighted",
            "Bayesian Classification": "class_bayesian",
            "Topsis": "class_topsis"
        }
        selected_class_column = class_column_mapping[classification_method]
        
        # Apply the selected classification method
        if classification_method == "Manhattan Distance":
            from models.manhattan import classify_manhattan
            df = classify_manhattan(df, features=selected_features)
        elif classification_method == "Mahalanobis Distance":
            from models.mahalanobis import classify_mahalanobis
            df = classify_mahalanobis(df, features=selected_features, return_distance=True)
        elif classification_method == "PCA Classification":
            from models.pca import classify_pca
            df = classify_pca(df, features=selected_features)
        elif classification_method == "Weighted Classification":
            from models.weighted import classify_weighted
            df = classify_weighted(df, features=selected_features, weights=weights)
        elif classification_method == "Bayesian Classification":
            from models.bayesian import classify_bayesian
            df = classify_bayesian(df, features=selected_features)
        elif classification_method == "Topsis":
            from models.topsis import classify_topsis
            df = classify_topsis(df, features=selected_features, weights=weights)
        
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

tab1, tab2, tab3, tab4, tab5,tab6,tab7 = st.tabs([
    "Interactive Map", 
    "Analytics & Insights", 
    "Building Data", 
    "Export & Reports",
    "City Statistics",
    "Year-over-Year Comparison",
    "Compare Cities"
])


with tab1:
    if not filtered_df.empty:
        # Limit to maximum 2000 buildings for map display
        map_df = filtered_df.head(2000) if len(filtered_df) > 2000 else filtered_df
        if len(filtered_df) > 2000:
            st.warning(f"Map showing 2,000 of {len(filtered_df):,} buildings. Apply filters to refine results.")
        display_map(map_df, selected_city, color_by)
    else:
        st.warning("No buildings match the current filters")
        st.stop() 

    st.markdown("---")

    # Free-text lookup below the map
    st.subheader("🔎 Find a Building by Keyword")
    search_fields = [
        "adresse_ban", "nom_rue_ban", "code_postal_ban",
        "nom_commune_ban", "adresse_brut"
    ]
    building_id = display_building_lookup(
        st.session_state["df"],
        info_fields=search_fields
    )

    # Show details & "Add to Comparison"
    if building_id:
        bd = st.session_state["df"].loc[
            st.session_state["df"]["building_id"] == building_id
        ].iloc[0]

        st.markdown("### 🏢 Building Details")
        col1, col2 = st.columns([3,1])
        with col1:
            st.markdown(f"**ID:** {building_id}")
            st.markdown(f"**Class:** {bd['class_label']}")
            st.markdown(f"**CO₂ Usage:** {bd['CO2_Usage']:.1f} kg")
            st.markdown(f"**Energy Consumption:** {bd['Energy_Consumption']:.1f} kWh")
        with col2:
            if st.button("➕ Add to Comparison", key="add_to_comp"):
                existing = [b["building_id"] for b in st.session_state['comparison_buildings']]
                if building_id in existing:
                    st.warning("Already in comparison")
                else:
                    st.session_state['comparison_buildings'].append(bd.to_dict())
                    st.success("Added to comparison")

        st.markdown("---")
        display_building_classifications(st.session_state["df"], building_id)

    # Comparison table at the bottom
    if st.session_state['comparison_buildings']:
        st.markdown("---")
        st.subheader("📊 Comparison of Selected Buildings")
        comp_df = pd.DataFrame(st.session_state['comparison_buildings'])
        st.dataframe(
            comp_df[[
                "building_id", "class_label",
                "CO2_Usage", "Energy_Consumption", "Water_Usage"
            ]],
            use_container_width=True
        )



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

with tab5:
    import streamlit as st
    import pandas as pd
    import plotly.express as px

    st.header("📊 City Statistics by Feature & Class")

    # 1) Pick method & class
    methods = {
        "PCA"        : "class_pca",
        "Manhattan"  : "class_manhattan",
        "Mahalanobis": "class_mahalanobis",
        "Weighted"   : "class_weighted",
        "Bayesian"   : "class_bayesian",
        "Topsis"     : "class_topsis",
    }
    method_name    = st.selectbox("Classification Method", list(methods))
    class_col      = methods[method_name]
    classes        = sorted(st.session_state["df"][class_col].dropna().unique())
    selected_class = st.selectbox("Energy Class", classes)

    # 2) Features split
    consumption_feats = {
        "Energy_Consumption": "Energy (kWh)",
        "CO2_Usage":          "CO₂ (kg)",
        "Water_Usage":        "Water (L)"
    }
    intensity_feats = {
        "Energy_Intensity": "Energy Intensity (kWh/m²)",
        "CO2_Intensity":    "CO₂ Intensity (kg/m²)"
    }

    dfc = st.session_state["df"]
    dfc = dfc[dfc[class_col] == selected_class]

    def compute_stats(feat_map):
        rows = []
        for feat, label in feat_map.items():
            if feat not in dfc.columns:
                continue
            mn  = dfc[feat].min()
            mx  = dfc[feat].max()
            avg = dfc[feat].mean()
            rows.append({
                "Feature": label,
                "Min":     mn,
                "Mean":    avg,
                "Max":     mx
            })
        return pd.DataFrame(rows)

    # 3) Consumption stats & chart
    cons_df = compute_stats(consumption_feats)
    st.subheader(f"🛢️ Consumption Stats for Class {selected_class} ({method_name})")
    st.table(cons_df.style.format({"Min":"{:.1f}","Mean":"{:.1f}","Max":"{:.1f}"}))

    fig1 = px.bar(
        cons_df.melt(id_vars="Feature", var_name="Stat", value_name="Value"),
        x="Value", y="Feature", color="Stat",
        barmode="group", text="Value",
        color_discrete_map={"Min":"#A6A6A6","Mean":"#1F78B4","Max":"#333333"},
        labels={"Value":"Usage","Feature":""},
        title="Consumption: Min vs Mean vs Max"
    )
    fig1.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig1.update_layout(margin=dict(l=150, r=20, t=50, b=20), height=350)
    st.plotly_chart(fig1, use_container_width=True)

    # 4) Intensity stats & chart (if available)
    int_df = compute_stats(intensity_feats)
    if not int_df.empty:
        st.subheader(f"📐 Intensity Stats for Class {selected_class} ({method_name})")
        st.table(int_df.style.format({"Min":"{:.2f}","Mean":"{:.2f}","Max":"{:.2f}"}))

        fig2 = px.bar(
            int_df.melt(id_vars="Feature", var_name="Stat", value_name="Value"),
            x="Value", y="Feature", color="Stat",
            barmode="group", text="Value",
            color_discrete_map={"Min":"#A6A6A6","Mean":"#33A02C","Max":"#333333"},
            labels={"Value":"Intensity","Feature":""},
            title="Intensity: Min vs Mean vs Max"
        )
        fig2.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig2.update_layout(margin=dict(l=200, r=20, t=50, b=20), height=300)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No intensity columns found; showing consumption only.")

with tab6:
    st.header("📊 Year-over-Year Comparison")

    # — 1) Reload and preprocess the full “all years” dataset —
    df_all = pd.read_csv(CITY_PATHS[selected_city])
    df_all = validate_and_preprocess_dataset(df_all, scoring_basis)
    if df_all is None or df_all.empty:
        st.error("Unable to load full historical data for year-over-year comparison.")
        st.stop()

    # — 2) Re-run classification on the full dataset —
    # (so that we have a `class_label` column)
    df_all = ensure_classifications(df_all, selected_features, weights)


    if classification_method == "Manhattan Distance":
        from models.manhattan import classify_manhattan
        df_all = classify_manhattan(df_all, features=selected_features)
        col = "class_manhattan"

    elif classification_method == "Mahalanobis Distance":
        from models.mahalanobis import classify_mahalanobis
        df_all = classify_mahalanobis(df_all, features=selected_features, return_distance=True)
        col = "class_mahalanobis"

    elif classification_method == "PCA Classification":
        from models.pca import classify_pca
        df_all = classify_pca(df_all, features=selected_features)
        col = "class_pca"

    elif classification_method == "Weighted Classification":
        from models.weighted import classify_weighted
        df_all = classify_weighted(df_all, features=selected_features, weights=weights)
        col = "class_weighted"

    elif classification_method == "Topsis":
        from models.topsis import classify_topsis
        df_all = classify_topsis(df_all, features=selected_features, weights=weights)
        col = "class_topsis"
    else:  # Bayesian Classification
        from models.bayesian import classify_bayesian
        df_all = classify_bayesian(df_all, features=selected_features)
        col = "class_bayesian"

    df_all["class_label"] = df_all[col]


    # — 3) Compute counts by class and year —
    counts = df_all.groupby(["class_label", "year"]).size().unstack(fill_value=0).sort_index(axis=1)
    st.subheader("Building Counts by Class and Year")
    st.dataframe(counts.style.format("{:,}"))

    # — 4) Grouped bar chart comparing values by class across years —
    import plotly.graph_objects as go
    years_present = list(counts.columns)
    
    # Create DataFrames to store min, max, and avg values for each metric, class and year
    metrics = {
        "Energy_Consumption": "Energy (kWh)",
        "CO2_Usage": "CO₂ Emissions (kg)",
        "Water_Usage": "Water Usage (L)"
    }
    
    # Dictionary to hold all data frames
    data_frames = {
        "Energy_Consumption": {
            "min": pd.DataFrame(index=counts.index, columns=years_present),
            "max": pd.DataFrame(index=counts.index, columns=years_present),
            "avg": pd.DataFrame(index=counts.index, columns=years_present)
        },
        "CO2_Usage": {
            "min": pd.DataFrame(index=counts.index, columns=years_present),
            "max": pd.DataFrame(index=counts.index, columns=years_present),
            "avg": pd.DataFrame(index=counts.index, columns=years_present)
        },
        "Water_Usage": {
            "min": pd.DataFrame(index=counts.index, columns=years_present),
            "max": pd.DataFrame(index=counts.index, columns=years_present),
            "avg": pd.DataFrame(index=counts.index, columns=years_present)
        }
    }
    
    # Calculate min, max, and average values for each metric, class and year
    for yr in years_present:
        year_data = df_all[df_all["year"] == yr]
        for cls in counts.index:
            class_data = year_data[year_data["class_label"] == cls]
            if not class_data.empty:
                for metric in metrics.keys():
                    data_frames[metric]["min"].loc[cls, yr] = class_data[metric].min()
                    data_frames[metric]["max"].loc[cls, yr] = class_data[metric].max()
                    data_frames[metric]["avg"].loc[cls, yr] = class_data[metric].mean()
    
    # Let user choose comparison type, metric and scale
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        comparison_type = st.radio(
            "Select comparison type:",
            ["Minimum Values", "Maximum Values", "Average Values"],
            horizontal=True
        )
    with col2:
        selected_metric = st.selectbox(
            "Select metric to compare:",
            options=list(metrics.keys()),
            format_func=lambda x: metrics[x]
        )
    with col3:
        use_log_scale = st.checkbox("Use log scale", value=True, 
                                  help="Logarithmic scale works better for data with large variations")
    
    # Add explanation about multi-feature classification
    if comparison_type != "Building Counts":
        st.info(
            "📊 **Note on Classification vs. Metrics:** Buildings are classified based on multiple features "
            "(energy, CO₂, water usage), not just the selected metric. This means a Class B building might "
            "have higher values in one metric than a Class C building, while performing better on other metrics. "
            "These visualizations show actual metric values within each class, not the classification criteria."
        )

    # Show the appropriate visualization based on user selection
    
    if comparison_type == "Minimum Values":
        # Get relevant dataframe
        values_df = data_frames[selected_metric]["min"]
        unit = metrics[selected_metric].split('(')[1].strip(')')
        
        # Create figure for values
        fig_values = go.Figure()
        for yr in years_present:
            # Convert to numeric and handle NaN values safely
            y_values = pd.to_numeric(values_df[yr], errors='coerce').fillna(0)
            # Format text to handle large numbers nicely
            text_values = [f"{v:,.1f}" for v in y_values]
            
            fig_values.add_trace(go.Bar(
                name=f"{yr}",
                x=values_df.index,
                y=y_values,
                text=text_values,
                textposition="auto"
            ))
        
        fig_values.update_layout(
            barmode="group",
            title=f"Minimum {metrics[selected_metric]} by Class Across Years",
            xaxis_title="Energy Class",
            yaxis_title=f"Min {metrics[selected_metric]}",
            legend_title="Year",
            height=500,
        )
        
        if use_log_scale:
            fig_values.update_layout(yaxis_type="log")
            
        st.plotly_chart(fig_values, use_container_width=True)
        
        # Add table view option for detailed comparison
        if st.checkbox(f"Show detailed data for minimum {selected_metric}"):
            st.dataframe(
                values_df.style.format("{:,.2f}"),
                use_container_width=True
            )
    
    elif comparison_type == "Maximum Values":
        # Get relevant dataframe
        values_df = data_frames[selected_metric]["max"]
        unit = metrics[selected_metric].split('(')[1].strip(')')
        
        # Create figure for values
        fig_values = go.Figure()
        for yr in years_present:
            # Convert to numeric and handle NaN values safely
            y_values = pd.to_numeric(values_df[yr], errors='coerce').fillna(0)
            # Format text to handle large numbers nicely
            text_values = [f"{v:,.1f}" for v in y_values]
            
            fig_values.add_trace(go.Bar(
                name=f"{yr}",
                x=values_df.index,
                y=y_values,
                text=text_values,
                textposition="auto"
            ))
        
        fig_values.update_layout(
            barmode="group",
            title=f"Maximum {metrics[selected_metric]} by Class Across Years",
            xaxis_title="Energy Class",
            yaxis_title=f"Max {metrics[selected_metric]}",
            legend_title="Year",
            height=500,
        )
        
        if use_log_scale:
            fig_values.update_layout(yaxis_type="log")
            
        st.plotly_chart(fig_values, use_container_width=True)
        
        # Add table view option for detailed comparison
        if st.checkbox(f"Show detailed data for maximum {selected_metric}"):
            st.dataframe(
                values_df.style.format("{:,.2f}"),
                use_container_width=True
            )
    
    else:  # Average Values
        # Get relevant dataframe
        values_df = data_frames[selected_metric]["avg"]
        unit = metrics[selected_metric].split('(')[1].strip(')')
        
        # Create figure for values
        fig_values = go.Figure()
        for yr in years_present:
            # Convert to numeric and handle NaN values safely
            y_values = pd.to_numeric(values_df[yr], errors='coerce').fillna(0)
            # Format text to handle large numbers nicely
            text_values = [f"{v:,.1f}" for v in y_values]
            
            fig_values.add_trace(go.Bar(
                name=f"{yr}",
                x=values_df.index,
                y=y_values,
                text=text_values,
                textposition="auto"
            ))
        
        fig_values.update_layout(
            barmode="group",
            title=f"Average {metrics[selected_metric]} by Class Across Years",
            xaxis_title="Energy Class",
            yaxis_title=f"Avg {metrics[selected_metric]}",
            legend_title="Year",
            height=500,
        )
        
        if use_log_scale:
            fig_values.update_layout(yaxis_type="log")
            
        st.plotly_chart(fig_values, use_container_width=True)
        
        # Add table view option for detailed comparison
        if st.checkbox(f"Show detailed data for average {selected_metric}"):
            st.dataframe(
                values_df.style.format("{:,.2f}"),
                use_container_width=True
            )
    
    # Show comparison between highest class (A) and lowest class (F)
    if comparison_type != "Building Counts" and "A" in counts.index and "F" in counts.index:
        st.subheader("Class A vs F Comparison")
        
        # Get the currently selected values dataframe
        if comparison_type == "Minimum Values":
            values_df = data_frames[selected_metric]["min"]
            metric_name = f"Min {selected_metric}"
        elif comparison_type == "Maximum Values":
            values_df = data_frames[selected_metric]["max"]
            metric_name = f"Max {selected_metric}"
        else:
            values_df = data_frames[selected_metric]["avg"]
            metric_name = f"Avg {selected_metric}"
        
        unit = metrics[selected_metric].split('(')[1].strip(')')
        
        # Calculate A-F ratio for each year
        ratio_data = []
        for yr in years_present:
            class_a = pd.to_numeric(values_df.loc["A", yr], errors='coerce')
            class_f = pd.to_numeric(values_df.loc["F", yr], errors='coerce')
            if not pd.isna(class_a) and not pd.isna(class_f) and class_f != 0:
                ratio = class_a / class_f
                ratio_data.append({
                    "Year": yr,
                    f"Class A ({unit})": class_a,
                    f"Class F ({unit})": class_f,
                    "A:F Ratio": ratio
                })
        
        if ratio_data:
            ratio_df = pd.DataFrame(ratio_data)
            cols = st.columns(len(ratio_df))
            for i, (_, row) in enumerate(ratio_df.iterrows()):
                with cols[i]:
                    st.metric(
                        f"Year {row['Year']}",
                        f"{row['A:F Ratio']:.2f}x",
                        help=f"Class A: {row[f'Class A ({unit})']:,.2f} {unit} vs Class F: {row[f'Class F ({unit})']:,.2f} {unit}"
                    )
                    st.caption(f"{metric_name} - A to F ratio")
        else:
            st.info("Not enough data to calculate A-F comparison")

with tab7:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from datetime import datetime

    # — Helper to load & classify (no hashing on method_fn) —
    @st.cache_data(show_spinner=False)
    def load_and_prepare(path, year, features, weights, _method_fn, col_name, sel_class, scoring_basis):
        df = pd.read_csv(path)
        df = validate_and_preprocess_dataset(df, scoring_basis)
        df = df[df["year"] == year]
        df = ensure_classifications(df, features, weights)

        df = _method_fn(df)
        df["class_label"] = df[col_name]
        if sel_class != "All":
            df = df[df["class_label"] == sel_class]
        return df

    # — Controls —
    years = sorted(pd.read_csv(CITY_PATHS[list(CITY_PATHS)[0]])["year"].unique())
    current_year = datetime.now().year
    idx = years.index(current_year) if current_year in years else 0
    sel_year = st.selectbox("Select Year to Compare", years, index=idx)

    sel_cities = st.multiselect(
        "Cities to Compare",
        options=list(CITY_PATHS.keys()),
        default=list(CITY_PATHS.keys())[:2]
    )
    if len(sel_cities) < 2:
        st.info("Please select at least two cities.")
        st.stop()

    sel_class = st.selectbox("Filter by Energy Class (optional)", ["All"] + list("ABCDEF"))

    methods = {
        "Manhattan":   lambda d: classify_manhattan(d, features=selected_features),
        "Mahalanobis": lambda d: classify_mahalanobis(d, features=selected_features, return_distance=True),
        "PCA":         lambda d: classify_pca(d, features=selected_features),
        "Weighted":    lambda d: classify_weighted(d, features=selected_features, weights=weights),
        "Bayesian":    lambda d: classify_bayesian(d, features=selected_features),
        "Topsis":     lambda d: classify_topsis(d, features=selected_features, weights=weights)
    }
    cols_map = {
        "Manhattan":   "class_manhattan",
        "Mahalanobis": "class_mahalanobis",
        "PCA":         "class_pca",
        "Weighted":    "class_weighted",
        "Bayesian":    "class_bayesian",
        "Topsis":     "class_topsis"
    }
    sel_method = st.selectbox("Classification Method", list(methods.keys()))
    method_fn  = methods[sel_method]
    col_name   = cols_map[sel_method]

    # — Load & prepare each city's data —
    city_dfs = {}
    for city in sel_cities:
        dfc = load_and_prepare(
            CITY_PATHS[city],
            sel_year,
            selected_features,
            weights,
            method_fn,
            col_name,
            sel_class,
            scoring_basis
        )
        if not dfc.empty:
            city_dfs[city] = dfc

    if not city_dfs:
        st.error(f"No data for {sel_year} with those filters.")
        st.stop()

    # — Build summary DataFrame —
    records = []
    for city, dfc in city_dfs.items():
        e_col = "Energy_Consumption" if scoring_basis == "Total (kWh)" else "Energy_Intensity"
        records.append({
            "City": city,
            "Total Buildings": len(dfc),
            "Avg Energy (kWh)": dfc[e_col].mean(),
            "Avg CO₂ (kg)":     dfc["CO2_Usage"].mean(),
            "Avg Water (L)":    dfc["Water_Usage"].mean()
        })
    summary_df = pd.DataFrame(records).set_index("City")

    # — Dashboard Title & Highlights —
    st.markdown(f"## City Comparison for {sel_year}")
    best = summary_df["Avg Energy (kWh)"].idxmin()
    worst = summary_df["Avg Energy (kWh)"].idxmax()
    st.markdown(
        f"- 🔥 **Lowest average energy**: {best} ({summary_df.loc[best,'Avg Energy (kWh)']:.1f} kWh)\n"
        f"- ❄️ **Highest average energy**: {worst} ({summary_df.loc[worst,'Avg Energy (kWh)']:.1f} kWh)"
    )
    st.markdown("---")

    # — Metric Cards: Total buildings + averages —
    st.subheader("🏆 Key Metrics by City")
    metric_cols = st.columns(len(summary_df))
    for (city, row), col in zip(summary_df.iterrows(), metric_cols):
        col.markdown(f"**{city}**")
        col.metric("🏘️ Total Buildings", f"{row['Total Buildings']:,}")
        col.metric("⚡ Avg Energy",      f"{row['Avg Energy (kWh)']:.1f}")
        col.metric("🌱 Avg CO₂",         f"{row['Avg CO₂ (kg)']:.1f}")
        col.metric("💧 Avg Water",       f"{row['Avg Water (L)']:.1f}")

    st.markdown("---")

    # — Two-column charts: Grouped bar + Radar —
    left, right = st.columns(2)

    with left:
        st.subheader("📊 Multimetric Bar Chart")
        melt = summary_df.reset_index().melt(
            id_vars="City",
            value_vars=["Avg Energy (kWh)", "Avg CO₂ (kg)", "Avg Water (L)"],
            var_name="Metric",
            value_name="Value"
        )
        fig_bar = px.bar(
            melt,
            x="City",
            y="Value",
            color="Metric",
            barmode="group",
            text_auto=".1f",
            title="Avg Energy, CO₂ & Water"
        )
        fig_bar.update_layout(yaxis_title="Value", height=450)
        st.plotly_chart(fig_bar, use_container_width=True)

    with right:
        st.subheader("📊 Multimetric Radar Chart")
        radar = go.Figure()
        for city, row in summary_df.iterrows():
            radar.add_trace(go.Scatterpolar(
                r=[row["Avg Energy (kWh)"], row["Avg CO₂ (kg)"], row["Avg Water (L)"]],
                theta=["Energy","CO₂","Water"],
                name=city,
                fill="toself"
            ))
        radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, tickformat=".1f")),
            showlegend=True,
            height=450,
            title="Radar: Energy vs CO₂ vs Water"
        )
        st.plotly_chart(radar, use_container_width=True)

    st.markdown("---")

    # — Feature comparison across cities by class —
    st.subheader("📊 Feature Comparison by Class Across Cities")
    
    # Define available metrics with proper display names and units
    metrics = {
        "Energy_Consumption": "Energy (kWh)",
        "CO2_Usage": "CO₂ Emissions (kg)",
        "Water_Usage": "Water Usage (L)"
    }
    
    # UI Controls
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        comparison_type = st.radio(
            "Select comparison type:",
            ["Minimum Values", "Maximum Values", "Average Values"],
            horizontal=True,
            key="city_comparison_type"
        )
    with col2:
        selected_metric = st.selectbox(
            "Select metric to compare:",
            options=list(metrics.keys()),
            format_func=lambda x: metrics[x],
            key="city_comparison_metric"
        )
    with col3:
        use_log_scale = st.checkbox("Log scale", value=True, 
                                   help="Logarithmic scale works better for data with large variations",
                                   key="city_log_scale")
    
    # Add explanation about multi-feature classification
    st.info(
        "📊 **Note on Classification vs. Metrics:** Buildings are classified based on multiple features "
        "(energy, CO₂, water usage), not just the selected metric. This means a Class B building might "
        "have higher values in one metric than a Class C building, while performing better on other metrics. "
        "These visualizations show actual metric values within each class, not the classification criteria."
    )
    
    # Create dataframes to store the values
    all_classes = sorted(set().union(*[set(df["class_label"]) for df in city_dfs.values()]))
    
    # Build the data for the visualization
    comparison_data = []
    
    for city, city_df in city_dfs.items():
        for cls in all_classes:
            class_data = city_df[city_df["class_label"] == cls]
            if not class_data.empty:
                if comparison_type == "Minimum Values":
                    value = class_data[selected_metric].min()
                    type_label = "Min"
                elif comparison_type == "Maximum Values":
                    value = class_data[selected_metric].max()
                    type_label = "Max"
                else:  # Average Values
                    value = class_data[selected_metric].mean()
                    type_label = "Avg"
                
                comparison_data.append({
                    "City": city,
                    "Class": cls,
                    "Value": value,
                    "Metric": metrics[selected_metric]
                })
    
    if comparison_data:
        # Convert to DataFrame
        comp_df = pd.DataFrame(comparison_data)
        
        # Create visualization
        fig_comp = px.bar(
            comp_df,
            x="Class",
            y="Value",
            color="City",
            barmode="group",
            title=f"{type_label} {metrics[selected_metric]} by Class Across Cities",
            labels={"Value": f"{type_label} {metrics[selected_metric]}"}
        )
        
        if use_log_scale:
            fig_comp.update_layout(yaxis_type="log")
        
        fig_comp.update_layout(height=500)
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Optional detailed data table
        if st.checkbox(f"Show detailed data for {type_label.lower()} {selected_metric}"):
            # Pivot the data for better display
            pivot_df = comp_df.pivot(index="Class", columns="City", values="Value")
            st.dataframe(
                pivot_df.style.format("{:,.2f}"),
                use_container_width=True
            )
        
        # Show best and worst cities for each class
        st.subheader("🏆 Best Performing Cities by Class")
        
        # Prepare a summary table
        summary_rows = []
        for cls in all_classes:
            cls_data = comp_df[comp_df["Class"] == cls]
            if not cls_data.empty:
                if comparison_type == "Minimum Values" or comparison_type == "Average Values":
                    # For min and avg, lower is better
                    best_city = cls_data.loc[cls_data["Value"].idxmin()]["City"]
                    best_value = cls_data["Value"].min()
                    worst_city = cls_data.loc[cls_data["Value"].idxmax()]["City"]
                    worst_value = cls_data["Value"].max()
                else:
                    # For max, higher is better
                    best_city = cls_data.loc[cls_data["Value"].idxmax()]["City"]
                    best_value = cls_data["Value"].max()
                    worst_city = cls_data.loc[cls_data["Value"].idxmin()]["City"]
                    worst_value = cls_data["Value"].min()
                
                summary_rows.append({
                    "Class": cls,
                    "Best City": best_city,
                    f"Best Value ({selected_metric})": best_value,
                    "Worst City": worst_city,
                    f"Worst Value ({selected_metric})": worst_value,
                    "Difference (%)": ((worst_value - best_value) / best_value * 100) if best_value != 0 else 0
                })
        
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(
                summary_df.style.format({
                    f"Best Value ({selected_metric})": "{:,.2f}",
                    f"Worst Value ({selected_metric})": "{:,.2f}",
                    "Difference (%)": "{:+.1f}%"
                }),
                use_container_width=True
            )
    else:
        st.warning("Not enough data to compare features across cities by class.")

    st.markdown("---")
    
st.markdown("""
    <div style="text-align: center; margin-top: 30px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
        <p style="margin: 0; color: #1a1a1a;">Building Analytics Dashboard • Created with ❤️ • Data updated: April 2025</p>
    </div>
""", unsafe_allow_html=True)