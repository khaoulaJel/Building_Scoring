# building_selection.py

import streamlit as st
import pandas as pd
import pydeck as pdk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def setup_building_selection():
    """Initialize session state variables for building selection"""
    if 'selected_building_id' not in st.session_state:
        st.session_state.selected_building_id = None
    if 'map_view' not in st.session_state:
        st.session_state.map_view = "2D"  # Set default to 2D

def get_color_mapping(filtered_df, color_by):
    """Generate colors for map markers based on selected attribute"""
    if color_by == "class_label":
        # Enhanced color palette for building classes
        class_colors = {
            'A': [39, 174, 96, 220],   # Emerald green
            'B': [46, 204, 113, 220],  # Green sea
            'C': [241, 196, 15, 220],  # Yellow
            'D': [230, 126, 34, 220],  # Orange
            'E': [231, 76, 60, 220],   # Red
            'F': [192, 57, 43, 220]    # Dark red
        }
        return filtered_df[color_by].map(class_colors).tolist()
    else:
        # Create a smooth color gradient for numeric values
        min_val, max_val = filtered_df[color_by].min(), filtered_df[color_by].max()
        
        # Define a custom colormap with more gradual transitions
        colors = [(0.0, '#2ecc71'),   # Green
                 (0.3, '#f1c40f'),    # Yellow
                 (0.6, '#e67e22'),    # Orange
                 (1.0, '#e74c3c')]    # Red
        
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
        
        def map_to_color(val):
            norm = (val - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            rgba = cmap(norm)
            return [int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255), 220]

        return filtered_df[color_by].apply(map_to_color).tolist()


def create_clickable_map(filtered_df, color_by):
    """Create an interactive map with building markers"""
    cdf = filtered_df.copy()
    colors = get_color_mapping(cdf, color_by)

    sel = st.session_state.selected_building_id
    if sel:
        # Highlight selected building with a bright blue color
        colors = [
            [41, 128, 185, 255] if bid == sel else col
            for bid, col in zip(cdf["building_id"], colors)
        ]

    cdf["color"] = colors
    
    if st.session_state.map_view == "2D":
        layer = pdk.Layer(
            "ScatterplotLayer", 
            data=cdf,
            get_position=["longitude", "latitude"],
            get_radius=35,
            get_fill_color="color",
            pickable=True,
            stroked=True,
            get_line_color=[255, 255, 255],
            get_line_width=2,
            auto_highlight=True
        )
        pitch = 0

    view_state = pdk.ViewState(
        latitude=cdf["latitude"].mean(),
        longitude=cdf["longitude"].mean(),
        zoom=14,
        pitch=pitch,
        bearing=0
    )

    # Create a more informative and styled tooltip
    tooltip = {
        "html": """
        <div style="background-color:rgba(30,30,30,0.95); color:white; border-radius:8px; 
                    padding:12px; font-family:Arial; font-size:13px; box-shadow:0 0 10px rgba(0,0,0,0.5);">
            <h4 style="margin:0 0 8px 0; font-size:16px; border-bottom:1px solid #555; padding-bottom:5px;">
                Building #<span style="color:#3498db;">{building_id}</span>
            </h4>
            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                <span style="font-weight:bold; min-width:80px;">Class:</span>
                <span style="color:#f1c40f;">{class_label}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                <span style="font-weight:bold; min-width:80px;">CO₂:</span>
                <span>{CO2_Usage} kg</span>
            </div>  
            <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                <span style="font-weight:bold; min-width:80px;">Water:</span>
                <span>{Water_Usage} L</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="font-weight:bold; min-width:80px;">Energy:</span>
                <span>{Energy_Consumption} kWh</span>
            </div>
            <div style="font-size:11px; margin-top:8px; text-align:center; opacity:0.7;">
                Click for detailed information
            </div>
        </div>
        """,
        "style": {"z-index": 1}
    }

    return pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_provider="mapbox",
        map_style="mapbox://styles/mapbox/dark-v10",
        tooltip=tooltip
    )


def create_color_legend(color_by):
    """Create a color legend for the map"""
    fig, ax = plt.subplots(figsize=(10, 1.4))
    
    if color_by == "class_label":
        labels = ["A (Excellent)", "B (Good)", "C (Average)", "D (Below Avg)", "E (Poor)", "F (Very Poor)"]
        colors = ["#27ae60", "#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#c0392b"]
    else:
        labels = ["Low", "Medium-Low", "Medium-High", "High"]
        colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
    
    # Add selected building to legend
    labels.append("Selected")
    colors.append("#2980b9")
    
    # Create a more visually appealing legend
    handles = [plt.Rectangle((0,0), 1, 1, color=colors[i], ec="black", lw=0.5) 
               for i in range(len(labels))]
    
    ax.legend(handles, labels, loc='center', ncol=len(labels), 
              frameon=True, fancybox=True, shadow=True, framealpha=0.7,
              prop={'size': 10, 'weight': 'bold'})
    
    ax.set_axis_off()
    plt.tight_layout()
    return fig


def display_clickable_map(filtered_df, city_name, color_by):
    """Main function to display the interactive map and related UI elements"""
    setup_building_selection()
    
    if filtered_df.empty:
        st.warning("No buildings to display in this area.")
        return None

    # Create header with improved styling
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #1f2630 0%, #2c3e50 100%); 
                padding:12px; border-radius:8px; margin-bottom:15px; 
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h2 style="margin:0; color:#ecf0f1; font-family:'Arial'; text-align:center;">
            <i class="fas fa-building" style="margin-right:10px;"></i>
            {city_name.upper()} BUILDING EXPLORER
        </h2>
    </div>
    """, unsafe_allow_html=True)

    # Add map controls
    col1, col2, col3 = st.columns([1,1,1])
    
    with col1:
        view_options = ["2D"]
        st.session_state.map_view = st.radio("Map View", view_options, 
                                           horizontal=True, index=view_options.index(st.session_state.map_view))
    
    with col2:
        color_options = {"Energy Consumption": "Energy_Consumption", 
                       "CO₂ Usage": "CO2_Usage", 
                       "Water Usage": "Water_Usage",
                       "Building Class": "class_label"}
        
        selected_color = st.selectbox("Color by", options=list(color_options.keys()))
        color_by = color_options[selected_color]
    
    with col3:
        metric_units = {"Energy_Consumption": "kWh", "CO2_Usage": "kg", "Water_Usage": "L"}
        if color_by != "class_label":
            st.metric(
                f"Avg {selected_color}", 
                f"{filtered_df[color_by].mean():.1f} {metric_units[color_by]}",
                f"{(filtered_df[color_by].mean() - filtered_df[color_by].median()):.1f}"
            )
        else:
            st.metric("Most Common Class", filtered_df["class_label"].mode()[0])

    # Create interactive map
    deck = create_clickable_map(filtered_df, color_by)
    st.pydeck_chart(deck, use_container_width=True)

    # Building selection controls with improved UI
    st.markdown("<div style='margin:15px 0;'></div>", unsafe_allow_html=True)
    
    cols = st.columns([3,1])
    with cols[0]:
        opts = ["None"] + sorted(filtered_df["building_id"].astype(str).tolist())
        sel_index = 0 if st.session_state.selected_building_id is None else opts.index(st.session_state.selected_building_id)
        sel = st.selectbox("Select Building ID", opts, index=sel_index,
                         help="Choose a building from the dropdown or click directly on the map")
        st.session_state.selected_building_id = None if sel == "None" else sel
    
    with cols[1]:
        if st.button("Clear Selection", use_container_width=True, 
                   help="Remove current building selection"):
            st.session_state.selected_building_id = None
            st.experimental_rerun()

    # Add legend with improved styling
    st.markdown("""
    <div style="background:rgba(44,62,80,0.7); padding:10px; border-radius:8px; margin:15px 0;">
        <h4 style="margin:0 0 10px 0; color:white; text-align:center; font-family:Arial;">
            Color Legend
        </h4>
    </div>
    """, unsafe_allow_html=True)
    
    fig = create_color_legend(color_by)
    st.pyplot(fig)

    return st.session_state.selected_building_id


def display_building_classifications(df, building_id):
    """Display detailed information for the selected building"""
    if building_id is None:
        st.info("🏢 Click on a building or select from the dropdown to view detailed information.")
        return

    bd = df[df["building_id"] == building_id].iloc[0]
    
    st.markdown(f"""
    **Building #{building_id}**  

    DPE: {bd['true_energy_label']}  
    GES: {bd['true_ges_label']}
    """)

    # Create a comparison table of classification methods
    st.markdown("""
    <h3 style="margin:20px 0 10px 0; color:#3498db; font-family:Arial;">
        Classification Method Comparison
    </h3>
    """, unsafe_allow_html=True)
    
    class_cols = [c for c in df.columns if c.startswith('class_')]
    data = [[c.replace('class_','').title(), bd[c]] for c in class_cols]
    cls_df = pd.DataFrame(data, columns=['Method','Class'])
    
    # Style the dataframe
    def highlight_class(val):
        colors = {
            'A': 'background-color: #27ae60; color: white;',
            'B': 'background-color: #2ecc71; color: white;',
            'C': 'background-color: #f1c40f; color: black;',
            'D': 'background-color: #e67e22; color: white;',
            'E': 'background-color: #e74c3c; color: white;',
            'F': 'background-color: #c0392b; color: white;'
        }
        return colors.get(val, '')
    
    styled_df = cls_df.style.applymap(highlight_class, subset=['Class'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Show recommendations based on the energy classification (DPE)
    # This change makes recommendations based on the actual energy performance
    # rather than the composite class_label
    show_building_recommendations(bd, use_energy_class=True)


def get_class_color(class_label):
    """Return color hex code for a building class"""
    class_colors = {
        'A': '#27ae60',  # Emerald green
        'B': '#2ecc71',  # Green sea
        'C': '#f1c40f',  # Yellow
        'D': '#e67e22',  # Orange
        'E': '#e74c3c',  # Red
        'F': '#c0392b'   # Dark red
    }
    return class_colors.get(class_label, '#95a5a6')


def show_building_recommendations(building_data, use_energy_class=False):
    """Show recommendations based on building performance
    
    Parameters:
    -----------
    building_data : pandas.Series
        Data for the selected building
    use_energy_class : bool
        If True, use true_energy_label for recommendations
        If False, use class_label (default)
    """
    # Determine which class to use for recommendations
    if use_energy_class and 'true_energy_label' in building_data:
        class_label = building_data['true_energy_label']
        title = "Energy Performance (DPE)"
    else:
        class_label = building_data['class_label']
        title = "Overall Class"
    
    recommendations = {
        'A': ["Maintain current excellent performance", 
              "Consider energy storage solutions to optimize usage"],
        'B': ["Minor improvements in water conservation could achieve Class A",
              "Review HVAC scheduling for additional energy savings"],
        'C': ["Implement lighting efficiency upgrades", 
              "Install water-saving fixtures", 
              "Consider energy audit"],
        'D': ["Significant energy efficiency improvements needed",
              "Update insulation and window sealing",
              "Replace outdated HVAC equipment"],
        'E': ["Major renovation recommended", 
              "Consider comprehensive energy retrofit", 
              "Replace inefficient appliances and systems"],
        'F': ["Urgent retrofit required", 
              "Deep energy retrofit needed", 
              "Consider building envelope improvements", 
              "Replace heating/cooling systems"]
    }
    
    if class_label in recommendations:
        st.markdown(f"""
        <h3 style="margin:20px 0 10px 0; color:#3498db; font-family:Arial;">
            Recommendations for {title} Class {class_label}
        </h3>
        """, unsafe_allow_html=True)
        
        for i, rec in enumerate(recommendations[class_label], 1):
            st.markdown(f"- {rec}")