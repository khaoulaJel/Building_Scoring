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

# ──────────────────────────────────────────────────────────────
# Initialize session state for comparison
# ──────────────────────────────────────────────────────────────
if 'comparison_buildings' not in st.session_state:
    st.session_state['comparison_buildings'] = []

# ──────────────────────────────────────────────────────────────
# Sidebar configuration (replaces the old block)
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Dashboard Controls")
    
    # 1) Dataset selection
    st.header("📊 Dataset Selection")
    dataset_option = st.selectbox(
        "Choose Dataset",
        ["Default (Lyon)", "Gordes", "Upload Custom Dataset"],
        key="dataset_option"
    )
    uploaded_file = None
    if dataset_option == "Upload Custom Dataset":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if dataset_option == "Default (Lyon)":
        raw_path = "data/reduced_lyon_buildings_all_years.csv"
        selected_city = "Lyon"
    elif dataset_option == "Gordes":
        raw_path = "data/reduced_gordes_buildings_all_years.csv"
        selected_city = "Gordes"
    else:
        if uploaded_file is None:
            st.info("Please upload a CSV file to proceed.")
            st.stop()
        raw_path = uploaded_file
        selected_city = "Custom Dataset"
    
    with st.spinner(f"Loading {selected_city} data..."):
        df_raw = pd.read_csv(raw_path)
    
    # 2) Analysis year selector
    years = sorted(df_raw["year"].dropna().unique())
    analysis_year = st.selectbox("Analysis Year", years, index=len(years)-1)
    
    # 3) Scoring basis
    st.subheader("Scoring Basis")
    scoring_basis = st.radio(
        "Choose scoring basis",
        ["Total (kWh)", "Per m² (kWh/m²/year)"]
    )
    
    # 4) Validate & preprocess full dataset
    df_all = validate_and_preprocess_dataset(df_raw, scoring_basis)
    if df_all is None:
        st.error("Data validation failed. Please check your dataset.")
        st.stop()
    
    # 5) Slice to the selected year for tabs 1–5
    # 5) Create both full and year-filtered datasets
    df_all_years = df_all.copy()  # Keep all years for Tab6
    df = df_all[df_all["year"] == analysis_year].copy()  # Filtered for other tabs
    if df.empty:
        st.error(f"No data available for the year {analysis_year}.")
        st.stop()

    # Then store both in session state:
    st.session_state["df"] = df  # For tabs 1-5
    st.session_state["df_all_years"] = df_all_years  # For Tab6
        
    # 6) Intensity‐based scoring swap
    if scoring_basis == "Per m² (kWh/m²/year)":
        if "Energy_Intensity" in df.columns and "CO2_Intensity" in df.columns:
            df["Energy_Consumption"] = df["Energy_Intensity"]
            df["CO2_Usage"]          = df["CO2_Intensity"]
    
    # 7) Classification Features
    st.subheader("Classification Features")
    base_feats = [
        "Energy_Consumption", "CO2_Usage", "Water_Usage",
        "Energy_Intensity", "CO2_Intensity"
    ]
    available_features = [f for f in base_feats if f in df.columns]
    selected_features = st.multiselect(
        "Select Features for Classification",
        options=available_features,
        default=["Energy_Consumption", "CO2_Usage"]
    )
    if not selected_features:
        st.error("Please select at least one feature.")
        st.stop()
    
    # 8) Analysis Method
    st.header("Analysis Method")
    classification_method = st.radio(
        "Select Classification Method",
        [
            "Euclidean Distance", "Mahalanobis Distance",
            "PCA Classification", "Weighted Classification",
            "Bayesian Classification"
        ]
    )
    
    weights = None
    if classification_method == "Weighted Classification":
        st.subheader("Enter weights for each feature")
        weights = [
            st.number_input(
                f"Weight for {feat}",
                min_value=0.0, max_value=1.0,
                value=round(1/len(selected_features), 2),
                step=0.01
            )
            for feat in selected_features
        ]
    
    # 9) Run classification on this one-year slice
    df = add_classifications(df, features=selected_features, weights=weights)
    with st.spinner(f"Applying {classification_method}..."):
        if classification_method == "Euclidean Distance":
            from models.euclidean import classify_euclidean
            df = classify_euclidean(df, selected_features)
            class_col = "class_euclidean"
        elif classification_method == "Mahalanobis Distance":
            from models.mahalanobis import classify_mahalanobis
            df = classify_mahalanobis(df, selected_features, return_distance=True)
            class_col = "class_mahalanobis"
        elif classification_method == "PCA Classification":
            from models.pca import classify_pca
            df = classify_pca(df, selected_features)
            class_col = "class_pca"
        elif classification_method == "Weighted Classification":
            from models.weighted import classify_weighted
            df = classify_weighted(df, selected_features, weights=weights)
            class_col = "class_weighted"
        else:
            from models.bayesian import classify_bayesian
            df = classify_bayesian(df, selected_features)
            class_col = "class_bayesian"
    
        # Add class_label to all years
    df_all["class_label"] = df_all[class_col]

    # Then create the year-filtered version
    df = df_all[df_all["year"] == analysis_year].copy()

# Store both in session state
    st.session_state["df"] = df  # For tabs 1-5
    st.session_state["df_all_years"] = df_all  # For tab6
    st.session_state["df"]   = df
    st.session_state["df_all"] = df_all
    st.session_state["city"] = selected_city
    st.session_state["year"] = analysis_year

# ──────────────────────────────────────────────────────────────
# Advanced filters (unchanged – operates on st.session_state["df"])
# ──────────────────────────────────────────────────────────────
df = st.session_state["df"]
with st.expander("🔍 Advanced Filters", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        co2_min, co2_max = st.slider(
            "CO₂ Usage (kg)",
            float(df["CO2_Usage"].min()), float(df["CO2_Usage"].max()),
            (float(df["CO2_Usage"].min()), float(df["CO2_Usage"].max()))
        )
        water_min, water_max = st.slider(
            "Water Usage (L)",
            float(df["Water_Usage"].min()), float(df["Water_Usage"].max()),
            (float(df["Water_Usage"].min()), float(df["Water_Usage"].max()))
        )
    with col2:
        energy_min, energy_max = st.slider(
            "Energy (kWh)",
            float(df["Energy_Consumption"].min()), float(df["Energy_Consumption"].max()),
            (float(df["Energy_Consumption"].min()), float(df["Energy_Consumption"].max()))
        )
    st.subheader("Building Class Filter")
    all_classes = ['A','B','C','D','E','F']
    class_cols = st.columns(6)
    selected_classes = []
    for i, cls in enumerate(all_classes):
        with class_cols[i]:
            if st.checkbox(f"Class {cls}", value=True, key=f"class_{cls}"):
                selected_classes.append(cls)
    color_by = st.selectbox(
        "Color Buildings By",
        ["class_label","CO2_Usage","Water_Usage","Energy_Consumption"],
        format_func=lambda x: {
            "class_label":"Energy Class","CO2_Usage":"CO₂ Emissions",
            "Water_Usage":"Water Consumption","Energy_Consumption":"Energy Usage"
        }[x]
    )

# ──────────────────────────────────────────────────────────────
# Filtered DataFrame for main tabs
# ──────────────────────────────────────────────────────────────
filtered_df = df[
    (df["CO2_Usage"] >= co2_min) & (df["CO2_Usage"] <= co2_max) &
    (df["Water_Usage"] >= water_min) & (df["Water_Usage"] <= water_max) &
    (df["Energy_Consumption"] >= energy_min) & (df["Energy_Consumption"] <= energy_max) &
    (df["class_label"].isin(selected_classes))
]

# Info bar
col1, col2, col3 = st.columns([2,2,1])
with col1:
    st.info(f"Showing {len(filtered_df)} of {len(df)} buildings in {selected_city}")
with col2:
    st.metric("Average Energy Class",
              filtered_df['class_label'].mode()[0] if not filtered_df.empty else "N/A")
with col3:
    st.metric("Data Last Updated",
              datetime.now().strftime("%Y-%m-%d"))

# ──────────────────────────────────────────────────────────────
# Tabs (unchanged – follow immediately after)
# ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Interactive Map",
    "Analytics & Insights",
    "Building Data",
    "Export & Reports",
    "City Statistics",
    "Year-over-Year Comparison",
    "Compare Cities"
])
class_colors = {
    'A': '#28a745',
    'B': '#5cb85c',
    'C': '#ffc107',
    'D': '#fd7e14',
    'E': '#dc3545',
    'F': '#6c757d'
}

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

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

with tab6:
    st.header("📊 Year-over-Year Comparison")
    st.markdown("This section dynamically compares the earliest and latest years in your dataset and shows trends across all years.")

    # Get the full dataset from session state
    df_all_years = st.session_state.get("df_all_years", None)
    if df_all_years is None:
        st.error("Year-over-year data not available. Please check your dataset.")
        st.stop()

    # Ensure class_label exists (apply same classification to all years)
    if "class_label" not in df_all_years.columns:
        # You'll need to apply your classification method to the full dataset
        # This should mirror what you do in the sidebar
        df_all_years = add_classifications(df_all_years, 
                                         features=selected_features, 
                                         weights=weights)
        # Apply the selected classification method to all years
        # (Same logic as in your sidebar)

    # --- Dynamically discover years ---
    years = sorted(df_all_years["year"].unique())
    if len(years) < 2:
        st.error("Need at least 2 years of data for comparison")
        st.stop()
        
    first_year, last_year = years[0], years[-1]

    # --- Filter to those two for direct comparison ---
    df_first = df_all_years[df_all_years.year == first_year]
    df_last = df_all_years[df_all_years.year == last_year]
    
    # --- Core KPIs ---
    total_first = len(df_first)
    total_last = len(df_last)
    Δ_total = total_last - total_first
    pct_total = (Δ_total / total_first * 100) if total_first else 0

    avg_e_first = df_first["Energy_Consumption"].mean()
    avg_e_last = df_last["Energy_Consumption"].mean()
    Δ_energy = avg_e_last - avg_e_first
    pct_energy = (Δ_energy / avg_e_first * 100) if avg_e_first else 0

    # --- Class-by-year counts for trend & delta ---
    # First ensure we have the counts in the right format
    counts = df_all_years.groupby(["class_label", "year"]).size().unstack(fill_value=0)
    
    # Verify the years exist in the counts columns
    if first_year not in counts.columns or last_year not in counts.columns:
        st.error("Missing year data in the counts table")
        st.stop()
        
    delta = counts[last_year] - counts[first_year]
    pct_cls = (delta / counts[first_year] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    
    delta_df = (
        pd.DataFrame({
            "Class": counts.index,
            f"{first_year}": counts[first_year],
            f"{last_year}": counts[last_year],
            "Δ Count": delta,
            "Δ %": pct_cls
        })
        .sort_values("Δ %", ascending=False)
    )

    # --- Extract top/bottom classes ---
    top = delta_df.iloc[0]
    bot = delta_df.iloc[-1]


    # --- KPI Row ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"🏢 Total Buildings ({last_year})", f"{total_last:,}",
              delta=f"{Δ_total:+,} ({pct_total:.1f}%)")
    c2.metric(f"⚡ Avg Energy ({last_year})", f"{avg_e_last:.1f} kWh",
              delta=f"{Δ_energy:+.1f} kWh ({pct_energy:.1f}%)")
    c3.metric("🚀 Fastest Growth Class", top["Class"],
              delta=f"{top['Δ %']:+.1f}%")
    c4.metric("🐌 Slowest Growth Class", bot["Class"],
              delta=f"{bot['Δ %']:+.1f}%")

    st.markdown("---")

    # --- 1) Trend Across All Years (Stacked Area) ---
    st.subheader("Evolution of Building Counts by Class")
    trend = counts.reset_index().melt(
        id_vars="class_label", var_name="Year", value_name="Count"
    )
    fig_area = px.area(
        trend,
        x="Year", y="Count", color="class_label",
        labels={"class_label": "Class"},
        title=None
    )
    fig_area.update_layout(xaxis=dict(dtick=1), height=300, hovermode="x unified")
    st.plotly_chart(fig_area, use_container_width=True)

    st.markdown("---")

    # --- 2) Absolute & % Δ Between First & Last Year ---
    st.subheader(f"Change from {first_year} to {last_year} by Class")
    dfc = delta_df.copy()
    dfc_sorted = dfc.sort_values("Δ Count", ascending=False)

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name="Δ Count",
        x=dfc_sorted["Class"],
        y=dfc_sorted["Δ Count"],
        marker_color=["green" if v>0 else "red" for v in dfc_sorted["Δ Count"]],
        text=dfc_sorted["Δ Count"].map("{:+,}".format),
        textposition="outside"
    ))
    fig_bar.add_trace(go.Scatter(
        name="% Δ",
        x=dfc_sorted["Class"],
        y=dfc_sorted["Δ %"],
        mode="markers+text",
        marker=dict(symbol="diamond", size=10, color="navy"),
        text=dfc_sorted["Δ %"].map("{:+.1f}%".format),
        textposition="top center",
        yaxis="y2"
    ))
    fig_bar.update_layout(
        barmode="group",
        xaxis_title="Building Class",
        yaxis_title="Absolute Change",
        yaxis2=dict(title="% Change", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.02, x=0.5),
        height=350,
        hovermode="x unified"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # --- 3) Average Energy by Class (Grouped Bar) ---
    st.subheader("Average Energy Consumption by Class")
    energy_cls = df.groupby(["class_label", "year"])["Energy_Consumption"] \
                   .mean().unstack(fill_value=0)
    melt_eng = energy_cls.reset_index().melt(
        id_vars="class_label", var_name="Year", value_name="AvgEnergy"
    )
    fig_eng = px.bar(
        melt_eng,
        x="class_label", y="AvgEnergy", color="Year",
        barmode="group", text="AvgEnergy",
        labels={"class_label":"Class", "AvgEnergy":"kWh"}
    )
    fig_eng.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_eng.update_layout(height=300, uniformtext_mode="hide")
    st.plotly_chart(fig_eng, use_container_width=True)

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