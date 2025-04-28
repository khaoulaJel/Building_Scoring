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
            selected_city = "Lyon"
    elif dataset_option == "Gordes":
        with st.spinner("Loading Gordes data..."):
            try:
                df = pd.read_csv("data/reduced_gordes_buildings_all_years.csv")
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
                    selected_city = "Custom Dataset"
                except Exception as e:
                    st.error(f"Error loading CSV file: {str(e)}")
                    df = pd.read_csv("data/reduced_lyon_buildings_all_years.csv")
                    selected_city = "Lyon"
                    st.warning("Reverted to default Lyon dataset")
                    st.write(f"Fallback dataset: {selected_city}")  # Debug
        else:
            st.info("Please upload a CSV file to proceed.")
            df = pd.read_csv("data/reduced_lyon_buildings_all_years.csv")
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
        "Euclidean Distance"    : "Euclidean Distance",
        "Mahalanobis Distance"  : "Mahalanobis Distance",
        "PCA Classification"    : "PCA Classification",
        "Weighted Classification": "Weighted Classification",
        "Bayesian Classification": "Bayesian Classification"
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
    df = add_classifications(df, features=selected_features, weights=weights)
        
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
            df = classify_weighted(df, features=selected_features, weights=weights)
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

with tab5:
    import streamlit as st
    import pandas as pd
    import plotly.express as px

    st.header("📊 City Statistics by Feature & Class")

    # 1) Pick method & class
    methods = {
        "PCA"        : "class_pca",
        "Euclidean"  : "class_euclidean",
        "Mahalanobis": "class_mahalanobis",
        "Weighted"   : "class_weighted",
        "Bayesian"   : "class_bayesian"
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
    st.header("📊 Year-over-Year Comparison (2024 vs 2025)")

    # 1) Class counts by year
    counts = df.groupby(["class_label", "year"]).size().unstack(fill_value=0)
    st.subheader("Building Counts by Class and Year")
    st.dataframe(counts.style.format("{:,}"))

    # 2) Bar chart (Plotly)
    import plotly.graph_objects as go
    fig = go.Figure([
        go.Bar(name="2024", x=counts.index, y=counts[2024]),
        go.Bar(name="2025", x=counts.index, y=counts[2025]),
    ])
    fig.update_layout(
        barmode="group",
        xaxis_title="Class",
        yaxis_title="Count of Buildings",
        legend_title="Year",
    )
    st.plotly_chart(fig, use_container_width=True)

    # 3) Delta per class
    delta = counts[2025] - counts[2024]
    pct  = (delta / counts[2024] * 100).fillna(0)
    delta_df = pd.DataFrame({
        "Class":        counts.index,
        "Δ Count":      delta.values,
        "Δ %":          pct.values
    })
    st.subheader("Change in Counts (2025 – 2024)")
    st.dataframe(delta_df.style.format({"Δ Count":"{:+,}","Δ %":"{:+.1f}%"}), use_container_width=True)

    # 4) Summary metrics
    tot2024 = len(df[df.year == 2024])
    tot2025 = len(df[df.year == 2025])
    totΔ     = tot2025 - tot2024
    avg_cons = df.groupby("year")["Energy_Consumption"].mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Buildings 2024", f"{tot2024:,}")
    c2.metric("Total Buildings 2025", f"{tot2025:,}", delta=f"{totΔ:+,}")
    c3.metric("Avg Energy 2024", f"{avg_cons.get(2024,0):.1f} kWh")
    c4.metric("Avg Energy 2025", f"{avg_cons.get(2025,0):.1f} kWh",
             delta=f"{(avg_cons.get(2025,0)-avg_cons.get(2024,0)):+.1f} kWh")
    
    # 5) Class Comparison Over Years - Area Chart Only
    st.subheader("Building Class Distribution Evolution")

    # Get unique years dynamically from the dataframe
    available_years = sorted(df['year'].unique())

    # Prepare data for stacked area chart
    class_years = {}
    for year in available_years:
        for cls in counts.index:
            if year in counts.columns:
                class_years.setdefault(cls, []).append(counts.loc[cls, year] if cls in counts.index else 0)

    # Create stacked area chart showing class breakdown over years
    fig_area = go.Figure()

    # Colors for consistency
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Add traces for each class
    for i, cls in enumerate(counts.index):
        fig_area.add_trace(go.Scatter(
            x=available_years,
            y=class_years.get(cls, [0]*len(available_years)),
            mode='lines',
            name=cls,
            line=dict(width=0.5, color=colors[i % len(colors)]),
            fill='tonexty',  # fills area between traces
            stackgroup='one'  # create stacked area
        ))

    # Customize layout
    fig_area.update_layout(
        title='Building Class Distribution by Year',
        xaxis_title='Year',
        yaxis_title='Number of Buildings',
        template='plotly_white',
        height=500,
        hovermode='x unified'
    )

    st.plotly_chart(fig_area, use_container_width=True)

    # 6) Class Proportions - Pie Charts Comparison
    st.subheader("Building Class Proportion Comparison")

    # Create a row of pie charts, one for each year
    chart_cols = st.columns(len(available_years))

    for i, year in enumerate(available_years):
        with chart_cols[i]:
            year_data = counts[year] if year in counts.columns else pd.Series(0, index=counts.index)
            
            # Create pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=year_data.index,
                values=year_data.values,
                hole=.4,
                marker_colors=colors[:len(year_data)]
            )])
            
            fig_pie.update_layout(
                title=f'{year} Distribution',
                height=400,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                )
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
with tab7:
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from datetime import datetime
    from models.euclidean import classify_euclidean
    from models.mahalanobis import classify_mahalanobis
    from models.pca import classify_pca
    from models.weighted import classify_weighted
    from models.bayesian import classify_bayesian

    # --- Helper to load & classify (avoid hashing the function) ---
    @st.cache_data(show_spinner=False)
    def load_and_prepare(city_path, year, features, weights, _method_fn, col_name, sel_class, scoring_basis):
        df = pd.read_csv(city_path)
        df = validate_and_preprocess_dataset(df, scoring_basis)
        df = df[df.year == year]
        df = add_classifications(df, features=features, weights=weights)
        df = _method_fn(df)
        df['class_label'] = df[col_name]
        if sel_class != "All":
            df = df[df.class_label == sel_class]
        return df

    # --- Controls ---
    years      = sorted(pd.read_csv(CITY_PATHS[next(iter(CITY_PATHS))])['year'].unique())
    sel_years  = st.multiselect("Snapshot Years", options=years, default=[datetime.now().year])
    norm_opt   = st.radio("Metric Basis", ["Absolute (Total)", "Normalized (per m²)"])
    sel_cities = st.multiselect("Cities to compare", list(CITY_PATHS), default=list(CITY_PATHS)[:2])
    if len(sel_cities) < 2:
        st.info("Select at least two cities to compare.")
        st.stop()
    sel_class  = st.selectbox("Filter by Energy Class", ["All"] + list("ABCDEF"))

    methods = {
        "Euclidean":   lambda d: classify_euclidean(d, features=selected_features),
        "Mahalanobis": lambda d: classify_mahalanobis(d, features=selected_features, return_distance=True),
        "PCA":         lambda d: classify_pca(d, features=selected_features),
        "Weighted":    lambda d: classify_weighted(d, features=selected_features, weights=weights),
        "Bayesian":    lambda d: classify_bayesian(d, features=selected_features)
    }
    cols_map = {
        "Euclidean":   "class_euclidean",
        "Mahalanobis": "class_mahalanobis",
        "PCA":         "class_pca",
        "Weighted":    "class_weighted",
        "Bayesian":    "class_bayesian"
    }
    sel_method = st.selectbox("Classification Method", list(methods.keys()))
    method_fn  = methods[sel_method]
    col_name   = cols_map[sel_method]

    # --- Load & summarize each year ---
    yearly_summaries = {}
    yearly_city_dfs  = {}
    for year in sel_years:
        city_dfs = {}
        for city in sel_cities:
            df = load_and_prepare(
                CITY_PATHS[city], year,
                selected_features, weights,
                method_fn, col_name,
                sel_class, scoring_basis
            )
            if not df.empty:
                city_dfs[city] = df
        if len(city_dfs) >= 2:
            yearly_city_dfs[year] = city_dfs
            summary = []
            for city, df in city_dfs.items():
                e_col = "Energy_Consumption" if norm_opt=="Absolute (Total)" else "Energy_Intensity"
                c_col = "CO2_Usage"
                w_col = "Water_Usage"
                summary.append({
                    "City":     city,
                    "AvgEnergy": df[e_col].mean(),
                    "AvgCO2":    df[c_col].mean(),
                    "AvgWater":  df[w_col].mean()
                })
            yearly_summaries[year] = pd.DataFrame(summary).set_index("City")

    if not yearly_summaries:
        st.error("No valid data to compare."); st.stop()

    # --- Dashboard for the latest selected year ---
    latest   = max(yearly_summaries)
    summ_df  = yearly_summaries[latest]
    city_dfs = yearly_city_dfs[latest]

    # 1) Avg metrics
    st.subheader(f"🏆 {latest} Avg Metrics")
    cols = st.columns(len(summ_df))
    for (city, row), col in zip(summ_df.iterrows(), cols):
        col.subheader(city)
        col.metric("⚡ Energy", f"{row.AvgEnergy:.1f} kWh")
        col.metric("🌱 CO₂",    f"{row.AvgCO2:.1f} kg")
        col.metric("💧 Water",  f"{row.AvgWater:.1f} L")

    # 2) Energy class distribution
    st.subheader(f"🏷️ {latest} Class Distribution")
    dist = (
        pd.concat([df.class_label.value_counts(normalize=True)*100 for df in city_dfs.values()], axis=1)
          .fillna(0)
    )
    dist.columns = sel_cities
    dist = dist.T.reset_index().melt(id_vars="index", var_name="Class", value_name="Pct")
    dist.rename(columns={"index":"City"}, inplace=True)
    fig1 = px.bar(dist, x="City", y="Pct", color="Class", barmode="stack", text_auto=".1f")
    fig1.update_layout(yaxis_title="%", height=350)
    st.plotly_chart(fig1, use_container_width=True)

    # 3) Radar chart
    st.subheader("📊 Multivariate Radar Chart")
    fig_radar = go.Figure()
    for city, row in summ_df.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row.AvgEnergy, row.AvgCO2, row.AvgWater],
            theta=["Energy","CO₂","Water"],
            name=city,
            fill="toself"
        ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, height=450)
    st.plotly_chart(fig_radar, use_container_width=True)

    # 4) Indexed Year-on-Year Change (Base = first selected year → 100)
    # 4) Yearly Average Trends (raw values)
    st.subheader("📈 Yearly Average Trends")

    trend_list = []
    for yr, city_dfs in yearly_city_dfs.items():
        for city, df in city_dfs.items():
            e_col = "Energy_Consumption" if norm_opt=="Absolute (Total)" else "Energy_Intensity"
            c_col = "CO2_Usage"         if norm_opt=="Absolute (Total)" else "CO2_Intensity"
            w_col = "Water_Usage"       # always use absolute for now

            trend_list.append({"Year": yr, "City": city, "Metric": "Energy", "Value": df[e_col].mean()})
            trend_list.append({"Year": yr, "City": city, "Metric": "CO₂",    "Value": df[c_col].mean()})
            trend_list.append({"Year": yr, "City": city, "Metric": "Water",  "Value": df[w_col].mean()})

    trend_df = pd.DataFrame(trend_list)

    fig_trend = px.line(
        trend_df,
        x="Year", y="Value",
        color="City",
        facet_col="Metric",
        facet_col_wrap=3,
        markers=True,
        title="Average Consumption by City Over Years"
    )
    # allow each facet to scale independently
    fig_trend.update_yaxes(matches=None)
    st.plotly_chart(fig_trend, use_container_width=True, height=600)


    # 5) Key insights & export
    st.subheader("💡 Key Insights")
    best  = summ_df.AvgEnergy.idxmin()
    worst = summ_df.AvgEnergy.idxmax()
    st.markdown(
        f"- **{best}** has the lowest avg energy in {latest} ({summ_df.loc[best,'AvgEnergy']:.1f} kWh).\n"
        f"- **{worst}** has the highest avg energy ({summ_df.loc[worst,'AvgEnergy']:.1f} kWh)."
    )

    st.subheader("📥 Export Summary")
    csv = summ_df.to_csv().encode("utf-8")
    st.download_button("Download CSV", csv, f"city_compare_{latest}.csv", "text/csv")

    

    
st.markdown("""
    <div style="text-align: center; margin-top: 30px; padding: 10px; background-color: #f1f3f4; border-radius: 5px;">
        <p style="margin: 0; color: #555;">Building Analytics Dashboard • Created with ❤️ • Data updated: April 2025</p>
    </div>
""", unsafe_allow_html=True)