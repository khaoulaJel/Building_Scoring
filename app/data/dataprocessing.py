"""
Module: dataprocessing.py

This module provides data-loading, cleaning, feature-engineering, synthetic-data generation, 
clustering, and analysis utilities for building-performance datasets. It supports both local 
CSV paths and in-memory file-like objects (e.g., Streamlit uploads).
"""

import os
from datetime import datetime
import pandas as pd
import numpy as np
from pyproj import Transformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


def process_city_data(city_name, input_filename=None, year=None):
    """
    Process building data for a specific city with optional year tagging.
    
    Args:
        city_name (str): Name of the city to process
        input_filename (str, optional): Custom input filename
        year (int, optional): Year to tag this dataset with
    """
    print(f"Processing data for {city_name}...")
    
    # Load data with flexible filename
    if input_filename is None:
        input_file = f"{city_name.lower()}.csv"
    else:
        input_file = input_filename
        
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        return
    
    df = pd.read_csv(input_file)
    
    # Convert coordinates
    def convert_coordinates(df):
        transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(df["coordonnee_cartographique_x_ban"].values,
                                         df["coordonnee_cartographique_y_ban"].values)
        df["longitude"] = lon
        df["latitude"] = lat
        return df

    try:
        df = convert_coordinates(df)
    except KeyError as e:
        print(f"Warning: Coordinate conversion failed. Missing column: {e}")
        print("Continuing without coordinate conversion...")
    
    # Rename and engineer features
    df = df.rename(columns={
        "numero_dpe": "building_id",
        "conso_5 usages_ef": "Energy_Consumption",
        "emission_ges_5_usages": "CO2_Usage",
        "etiquette_dpe": "true_energy_label",
        "etiquette_ges": "true_ges_label"
    })
    
    # Check if required columns exist before calculations
    if "Energy_Consumption" in df.columns:
        df["Water_Usage"] = df["Energy_Consumption"] * 0.3
    else:
        print("Warning: Energy_Consumption column not found, skipping Water_Usage calculation")
        
    if "conso_5 usages_par_m2_ef" in df.columns:
        df["Energy_Intensity"] = df["conso_5 usages_par_m2_ef"]
    else:
        print("Warning: conso_5 usages_par_m2_ef column not found")
        
    if "emission_ges_5_usages par_m2" in df.columns:
        df["CO2_Intensity"] = df["emission_ges_5_usages par_m2"]
    else:
        print("Warning: emission_ges_5_usages par_m2 column not found")

    final_cols = [
        #–– Primary key & location ––
        "building_id",
        "latitude", "longitude",

        #–– Core performance metrics ––
        "Energy_Consumption", "CO2_Usage", "Water_Usage",
        "Energy_Intensity", "CO2_Intensity",

        #–– Official DPE labels ––
        "true_energy_label", "true_ges_label",

        #–– BAN address fields (normalized) ––
        "adresse_ban",         # full standardized address
        "numero_voie_ban",     # street number
        "nom_rue_ban",         # street name
        "code_postal_ban",     # postal code
        "nom_commune_ban",     # commune name
        "identifiant_ban",     # BAN address ID

        #–– Raw-fallback address fields ––
        "adresse_brut",
        "nom_commune_brut",
        "code_postal_brut",

        #–– Basic building attributes for filters ––
        "annee_construction",
        "surface_habitable_immeuble",
    ]

    # Filter only columns that exist in the dataframe
    available_cols = [col for col in final_cols if col in df.columns]
    print(f"Available columns: {len(available_cols)}/{len(final_cols)}")
    
    required = [
        "building_id", "Energy_Consumption", "CO2_Usage", "Water_Usage", 
        "Energy_Intensity", "CO2_Intensity"
    ]
    
    # Adjust required columns if location data is missing
    if "latitude" in df.columns and "longitude" in df.columns:
        required.extend(["latitude", "longitude"])
    
    # Check which required columns are available
    available_required = [col for col in required if col in df.columns]
    missing_required = [col for col in required if col not in df.columns]
    
    if missing_required:
        print(f"Warning: Missing required columns: {missing_required}")
    
    # select & only drop rows missing available required columns
    df_clean = df[available_cols].dropna(subset=available_required)
    
    # Add log-transformed features
    numeric_cols = [
        "Energy_Consumption", "CO2_Usage", 
        "Energy_Intensity", "CO2_Intensity"
    ]
    
    # Only transform columns that exist
    existing_numeric = [col for col in numeric_cols if col in df_clean.columns]
    for col in existing_numeric:
        df_clean[f"log1p_{col}"] = np.log1p(df_clean[col])
    
    # Add year tag if provided
    df_clean['city'] = city_name
    if year is not None:
        df_clean['year'] = year
        output_file = f"reduced_{city_name.lower()}_buildings_{year}.csv"
    else:
        output_file = f"reduced_{city_name.lower()}_buildings.csv"
    
    # Save cleaned dataset
    df_clean.to_csv(output_file, index=False)
    print(f"Clean dataset saved to {output_file}.")
    
    return df_clean


def draw_multipliers(n_buildings, improvement_chance=0.6):
    """
    Generate multipliers for synthetic future data that maintains class distribution
    while allowing for some improvements.
    
    Args:
        n_buildings (int): Number of buildings
        improvement_chance (float): Chance of improvement vs degradation (0.0 to 1.0)
    
    Returns:
        numpy.ndarray: Array of multipliers centered around 1.0
    """
    # Start with a base of 1.0 for all buildings
    base = np.ones(n_buildings)
    
    # Add small random variations for natural fluctuations
    noise = np.random.normal(loc=0, scale=0.02, size=n_buildings)
    
    # Determine which buildings will improve vs degrade
    improve_mask = np.random.random(size=n_buildings) < improvement_chance
    degrade_mask = ~improve_mask
    
    # Generate improvements and degradations
    # Improvements are stronger to encourage overall progress
    improvements = np.zeros(n_buildings)
    degradations = np.zeros(n_buildings)
    
    if improve_mask.sum() > 0:
        improvements[improve_mask] = np.random.uniform(0.02, 0.08, size=improve_mask.sum())
    if degrade_mask.sum() > 0:
        degradations[degrade_mask] = np.random.uniform(-0.04, -0.01, size=degrade_mask.sum())
    
    # Combine all effects
    multipliers = base + noise + improvements + degradations
      # Ensure no negative or extreme values
    # Allow slightly more improvement than degradation
    multipliers = np.clip(multipliers, 0.85, 1.15)
    
    # Normalize to ensure the mean change is slightly negative
    # This helps maintain reasonable long-term trends
    mean_mult = np.mean(multipliers)
    if mean_mult > 1.0:
        multipliers = multipliers / mean_mult * 0.98
    
    return multipliers


def generate_future_data(df, city_name, target_year):
    """
    Generate synthetic data for future year based on existing data.
    
    Args:
        df (DataFrame): Source dataframe
        city_name (str): Name of the city
        target_year (int): Target year for synthetic data
    """
    print(f"Generating synthetic data for {city_name}, year {target_year}...")
    
    # Make a copy of the dataframe
    future_df = df.copy()
    future_df['city'] = city_name
    # Update year
    future_df['year'] = target_year
    
    # Get the number of buildings
    n_buildings = len(future_df)
    
    # Generate multipliers for different metrics
    energy_multipliers = draw_multipliers(n_buildings)
    co2_multipliers = draw_multipliers(n_buildings)
    water_multipliers = draw_multipliers(n_buildings)
    
    # Apply multipliers to core metrics
    if "Energy_Consumption" in future_df.columns:
        future_df["Energy_Consumption"] = future_df["Energy_Consumption"] * energy_multipliers
        
    if "Energy_Intensity" in future_df.columns:
        future_df["Energy_Intensity"] = future_df["Energy_Intensity"] * energy_multipliers
        
    if "CO2_Usage" in future_df.columns:
        future_df["CO2_Usage"] = future_df["CO2_Usage"] * co2_multipliers
        
    if "CO2_Intensity" in future_df.columns:
        future_df["CO2_Intensity"] = future_df["CO2_Intensity"] * co2_multipliers
        
    if "Water_Usage" in future_df.columns:
        future_df["Water_Usage"] = future_df["Water_Usage"] * water_multipliers
    
    # Re-calculate log transformations with new values
    numeric_cols = [
        "Energy_Consumption", "CO2_Usage", 
        "Energy_Intensity", "CO2_Intensity"
    ]
    
    # Only transform columns that exist
    existing_numeric = [col for col in numeric_cols if col in future_df.columns]
    for col in existing_numeric:
        log_col = f"log1p_{col}"
        if log_col in future_df.columns:  # Update existing log columns
            future_df[log_col] = np.log1p(future_df[col])
    
    # Generate some random changes in labels if they exist
    if "true_energy_label" in future_df.columns:
        # Randomly improve some labels (5% chance)
        mask = np.random.random(size=n_buildings) < 0.05
        label_map = {"G": "F", "F": "E", "E": "D", "D": "C", "C": "B", "B": "A", "A": "A"}
        future_df.loc[mask, "true_energy_label"] = future_df.loc[mask, "true_energy_label"].map(
            lambda x: label_map.get(x, x) if isinstance(x, str) else x
        )
    
    if "true_ges_label" in future_df.columns:
        # Randomly improve some labels (5% chance)
        mask = np.random.random(size=n_buildings) < 0.05
        label_map = {"G": "F", "F": "E", "E": "D", "D": "C", "C": "B", "B": "A", "A": "A"}
        future_df.loc[mask, "true_ges_label"] = future_df.loc[mask, "true_ges_label"].map(
            lambda x: label_map.get(x, x) if isinstance(x, str) else x
        )
    
    # Save future dataset
    output_file = f"reduced_{city_name.lower()}_buildings_{target_year}.csv"
    future_df.to_csv(output_file, index=False)
    print(f"Synthetic {target_year} data saved to {output_file}.")
    
    # Run K-Means clustering on the future data
    try:
        cluster_future_data(future_df, city_name, target_year)
    except Exception as e:
        print(f"Warning: Could not perform clustering on future data: {e}")
    
    return future_df


def cluster_future_data(df, city_name, year):
    """
    Apply K-means clustering to future data.
    
    Args:
        df (DataFrame): Data to cluster
        city_name (str): City name
        year (int): Year of data
    """
    # Find log columns for clustering
    log_cols = [col for col in df.columns if col.startswith('log1p_')]
    
    if len(log_cols) < 2:
        print("Not enough log-transformed columns for clustering")
        return df
    
    # Prepare data for clustering
    X = df[log_cols].values
    X_scaled = StandardScaler().fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)
    df["PC1"], df["PC2"] = pcs[:,0], pcs[:,1]
    
    # Apply K-Means
    kmeans = KMeans(n_clusters=4, random_state=42)
    df["cluster"] = kmeans.fit_predict(X_scaled)
    
    return df


def analyze_city_data(df, city_name, year=None):
    """
    Analyze the processed city data and generate visualizations.
    
    Args:
        df (DataFrame): Processed dataframe
        city_name (str): City name for titles
        year (int, optional): Year of the data for titles
    """
    title_prefix = f"{city_name}"
    if year is not None:
        title_prefix = f"{city_name} ({year})"
    
    print(f"\nAnalyzing data for {title_prefix}...")
    
    # Quick overview
    print("===== DATA INFO =====")
    df.info()
    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())
    print("\n===== DESCRIPTIVE STATISTICS =====")
    print(df.describe().T)
    
    # Check for numeric columns availability
    base_numeric_cols = [
        "Energy_Consumption", "CO2_Usage", "Water_Usage",
        "Energy_Intensity", "CO2_Intensity"
    ]
    numeric_cols = [col for col in base_numeric_cols if col in df.columns]
    
    if len(numeric_cols) < 2:
        print("Insufficient numeric columns for analysis.")
        return
    
    # Univariate analysis
    for col in numeric_cols:
        plt.figure(figsize=(8,4))
        sns.histplot(df[col], kde=True)
        plt.title(f"{title_prefix} - {col} — Distribution + KDE")
        plt.tight_layout()
        plt.close()

        plt.figure(figsize=(6,2))
        sns.boxplot(x=df[col])
        plt.title(f"{title_prefix} - {col} — Boxplot")
        plt.tight_layout()
        plt.close()

    # Correlation matrix
    corr = df[numeric_cols].corr()
    print("\n===== CORRELATION MATRIX =====")
    print(corr)
    plt.figure(figsize=(6,5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title(f"{title_prefix} - Correlation Matrix")
    plt.tight_layout()
    plt.close()

    # Pairwise scatterplots
    sns.pairplot(df[numeric_cols], diag_kind="kde", corner=True)
    plt.suptitle(f"{title_prefix} - Pairplot of Numeric Features", y=1.02)
    plt.tight_layout()
    plt.close()

    # Geographic scatter if coordinates are available
    if "longitude" in df.columns and "latitude" in df.columns:
        plt.figure(figsize=(6,6))
        sns.scatterplot(x=df["longitude"], y=df["latitude"], 
                        hue=df["Energy_Consumption"] if "Energy_Consumption" in df.columns else None, 
                        palette="viridis", s=20)
        plt.title(f"{title_prefix} - Map Scatter: Energy Consumption")
        plt.tight_layout()
        plt.close()

        # Geo‑bubble plots
        for col in numeric_cols:
            plt.figure(figsize=(6,6))
            sns.scatterplot(x=df["longitude"], y=df["latitude"], 
                            size=df[col], sizes=(10,200), alpha=0.6, legend=False)
            plt.title(f"{title_prefix} - Geo‑Bubble: {col}")
            plt.tight_layout()
            plt.close()

    # Check for existing log columns
    log_cols = [col for col in df.columns if col.startswith('log1p_')]
    
    if log_cols:
        corr2 = df[log_cols].corr()
        plt.figure(figsize=(5,4))
        sns.heatmap(corr2, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title(f"{title_prefix} - Corr of Log‑Transformed Features")
        plt.close()

        sns.pairplot(df[log_cols], diag_kind="kde", corner=True)
        plt.suptitle(f"{title_prefix} - Pairplot of Log‑Transformed Metrics", y=1.02)
        plt.tight_layout()
        plt.close()
    
    # PCA visualization if PC columns exist
    if "PC1" in df.columns and "PC2" in df.columns:
        plt.figure(figsize=(6,5))
        sns.scatterplot(x="PC1", y="PC2", data=df, alpha=0.3)
        plt.title(f"{title_prefix} - PCA of Building Metrics")
        plt.tight_layout()
        plt.close()
        
        # K-Means visualization if cluster column exists
        if "cluster" in df.columns:
            plt.figure(figsize=(6,5))
            sns.scatterplot(x="PC1", y="PC2", hue="cluster", data=df, palette="tab10", alpha=0.5)
            plt.title(f"{title_prefix} - K‑Means Clusters on PCA space")
            plt.tight_layout()
            plt.close()
    
    return df


def process_city_with_years(city_name, input_filename=None, base_year=2024, future_years=[2025]):
    """
    Process city data and generate data for multiple years.
    
    Args:
        city_name (str): Name of the city
        input_filename (str, optional): Input filename
        base_year (int): Base year for initial data
        future_years (list): List of future years to generate
    
    Returns:
        DataFrame: Combined DataFrame with all years' data
    """
    print(f"Processing {city_name} data with future projections...")
    
    # Process base year data
    base_df = process_city_data(city_name, input_filename, base_year)
    if base_df is None:
        print(f"❌ Error: Could not process base data for {city_name}")
        return None
        
    # Generate data for future years
    all_dfs = [base_df]  # Store all frames for concatenation
    
    current_df = base_df
    for year in future_years:
        try:
            future_df = generate_future_data(current_df, city_name, year)
            if future_df is not None:
                all_dfs.append(future_df)
                current_df = future_df  # Use latest year as base for next year
                print(f"✅ Generated projections for {year}")
        except Exception as e:
            print(f"❌ Error generating data for {year}: {e}")
            break  # Stop if we hit an error, but keep what we have
    
    # Create a combined dataset with all years
    try:
        print("Combining all years...")
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Save combined dataset
        combined_output = f"reduced_{city_name.lower()}_buildings_all_years.csv"
        combined_df.to_csv(combined_output, index=False)
        print(f"✅ Combined multi-year dataset saved as {combined_output}")
        
        # Apply clustering to the combined dataset
        try:
            print("Applying clustering to combined dataset...")
            combined_df = cluster_future_data(combined_df, city_name, base_year)
            print("✅ Applied clustering to combined dataset")
        except Exception as e:
            print(f"⚠️ Warning: Could not perform clustering: {e}")
        
        return combined_df
        
    except Exception as e:
        print(f"❌ Error combining data: {e}")
        return None


def process_uploaded_data(city_name, uploaded_file, year=None):
    """
    Process uploaded building data file through the same pipeline as local files.
    
    Args:
        city_name (str): Name of the city/dataset
        uploaded_file: File object (e.g., Streamlit uploaded file)
        year (int, optional): Year to tag this dataset with
    
    Returns:
        pd.DataFrame: Processed dataframe ready for analysis
    """
    print(f"Processing uploaded data for {city_name}...")
    
    try:
        # Read the uploaded file directly into pandas
        df = pd.read_csv(uploaded_file)
        print(f"📊 Read {len(df):,} rows from uploaded file")
        
        if len(df) == 0:
            print("❌ Error: Uploaded file is empty!")
            return None
            
        print(f"ℹ️ Found columns: {', '.join(df.columns)}")
            
    except Exception as e:
        print(f"❌ Error reading uploaded file: {e}")
        return None

    # Convert coordinates if they exist
    def convert_coordinates_safe(df):
        coord_cols = ["coordonnee_cartographique_x_ban", "coordonnee_cartographique_y_ban"]
        if all(col in df.columns for col in coord_cols):
            try:
                transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
                lon, lat = transformer.transform(df[coord_cols[0]], df[coord_cols[1]])
                df["longitude"] = lon
                df["latitude"] = lat
                print("✅ Coordinates converted successfully")
            except Exception as e:
                print(f"⚠️ Warning: Coordinate conversion failed: {e}")
        return df

    df = convert_coordinates_safe(df)

    # Handle column mapping with more variations
    column_mapping = {
        # Standard DPE columns
        "numero_dpe": "building_id",
        "conso_5 usages_ef": "Energy_Consumption", 
        "emission_ges_5_usages": "CO2_Usage",
        "etiquette_dpe": "true_energy_label",
        "etiquette_ges": "true_ges_label",
        "conso_5 usages_par_m2_ef": "Energy_Intensity",
        "emission_ges_5_usages par_m2": "CO2_Intensity",
        
        # Alternative spellings/formats
        "conso_5_usages_ef": "Energy_Consumption",
        "conso_5_usages_par_m2_ef": "Energy_Intensity",
        "emission_ges_5_usages_par_m2": "CO2_Intensity",
        
        # Generic variations
        "id": "building_id",
        "building": "building_id",
        "energy": "Energy_Consumption",
        "co2": "CO2_Usage",
        "ghg": "CO2_Usage",
        "energy_label": "true_energy_label",
        "ges_label": "true_ges_label"
    }
    
    # Apply mappings only for existing columns
    existing_mappings = {old: new for old, new in column_mapping.items() if old in df.columns}
    if existing_mappings:
        print(f"🔄 Mapping columns: {existing_mappings}")
        df = df.rename(columns=existing_mappings)

    # Handle building IDs
    if "building_id" not in df.columns:
        print("ℹ️ No building_id found, generating sequential IDs")
        df["building_id"] = [f"{city_name.upper()}_BLDG_{i+1}" for i in range(len(df))]

    # Required features
    required_features = ["building_id", "Energy_Consumption", "CO2_Usage"]
    
    # Handle Energy Consumption
    if "Energy_Consumption" not in df.columns:
        energy_cols = [col for col in df.columns if any(term in col.lower() 
            for term in ["conso", "energy", "kwh", "consommation"])]
        if energy_cols:
            print(f"ℹ️ Using {energy_cols[0]} for Energy_Consumption")
            df["Energy_Consumption"] = df[energy_cols[0]]
        else:
            print("❌ No energy consumption data found")
            return None

    # Handle CO2 emissions
    if "CO2_Usage" not in df.columns:
        co2_cols = [col for col in df.columns if any(term in col.lower() 
            for term in ["ges", "co2", "emission", "ghg"])]
        if co2_cols:
            print(f"ℹ️ Using {co2_cols[0]} for CO2_Usage")
            df["CO2_Usage"] = df[co2_cols[0]]
        else:
            print("❌ No CO2 emission data found")
            return None

    # Derive Water Usage (if not present)
    if "Water_Usage" not in df.columns:
        df["Water_Usage"] = df["Energy_Consumption"] * 0.3
        print("ℹ️ Estimated Water_Usage from Energy_Consumption")

    # Handle intensity metrics
    for base, intensity in [
        ("Energy_Consumption", "Energy_Intensity"),
        ("CO2_Usage", "CO2_Intensity")
    ]:
        if intensity not in df.columns and base in df.columns:
            if "surface_habitable_immeuble" in df.columns:
                df[intensity] = df[base] / df["surface_habitable_immeuble"]
                print(f"✅ Calculated {intensity} using building surface area")
            else:
                avg_surface = 100  # reasonable default
                df[intensity] = df[base] / avg_surface
                print(f"ℹ️ Used estimated surface area for {intensity}")

    # Clean numeric data
    numeric_cols = ["Energy_Consumption", "CO2_Usage", "Water_Usage",
                   "Energy_Intensity", "CO2_Intensity"]
    
    for col in numeric_cols:
        if col in df.columns:
            # Convert to numeric and handle errors
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Ensure positive values
            df[col] = df[col].clip(lower=0.01)
            # Add log transform
            df[f"log1p_{col}"] = np.log1p(df[col])

    # Remove rows with missing required data
    original_len = len(df)
    df = df.dropna(subset=[col for col in required_features if col in df.columns])
    if len(df) < original_len:
        print(f"⚠️ Removed {original_len - len(df):,} rows with missing required data")

    if len(df) == 0:
        print("❌ No valid data remains after cleaning")
        return None    # Add metadata
    df['city'] = city_name
    
    # Handle year data
    has_years = 'year' in df.columns and not df['year'].isna().all()
    if not has_years:
        # If no year data, just use current year
        df['year'] = year if year is not None else datetime.now().year
        print("ℹ️ No year data found, using current year")
    else:
        print(f"✅ Using existing year data: {sorted(df['year'].unique())}")

    print(f"✅ Successfully processed {len(df):,} buildings")

    # Save the processed data
    output_file = f"reduced_{city_name.lower()}_buildings.csv"
    df.to_csv(output_file, index=False)
    print(f"💾 Saved processed data to {output_file}")

    return df


def validate_city_data(df: pd.DataFrame) -> bool:
    """
    Validate that the processed city data has the required columns and formats.
    Enhanced version that checks for the essential columns needed by the dashboard.
    """
    if df is None or len(df) == 0:
        return False
    
    # Essential columns that must exist
    essential_columns = ['building_id', 'Energy_Consumption', 'CO2_Usage']
    
    # Check if essential columns exist
    missing_essential = [col for col in essential_columns if col not in df.columns]
    if missing_essential:
        print(f"Missing essential columns: {missing_essential}")
        return False
    
    # Check if data types are reasonable
    try:
        # building_id should be string-like
        if not df['building_id'].dtype in ['object', 'string']:
            df['building_id'] = df['building_id'].astype(str)
        
        # Energy and CO2 should be numeric
        for col in ['Energy_Consumption', 'CO2_Usage']:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Remove rows where conversion failed
                df = df.dropna(subset=[col])
        
        # Check if we still have data after cleaning
        if len(df) == 0:
            print("No valid data remains after type conversion")
            return False
            
    except Exception as e:
        print(f"Data validation error: {e}")
        return False
    
    print("Data validation successful")
    return True
# Additional utility functions for compatibility with your existing code

def validate_city_data(df: pd.DataFrame) -> bool:
    """Validate that the uploaded city data has the required columns and formats."""
    required_columns = ['building_id', 'Energy_Consumption', 'CO2_Usage']
    return all(col in df.columns for col in required_columns)


def add_uploaded_city_dataset(city_name: str, uploaded_file) -> pd.DataFrame:
    """Process and add a new city dataset from an uploaded file.
    
    Args:
        city_name: Name of the city (will be used as identifier)
        uploaded_file: File object containing the city data
    
    Returns:
        Processed DataFrame with the city data
    
    Raises:
        ValueError: If the data format is invalid or missing required columns
    """
    if not city_name or not city_name.strip():
        raise ValueError("City name cannot be empty")
    
    data_dir = os.path.dirname(__file__)
    csv_path = os.path.join(data_dir, f"reduced_{city_name.lower()}_buildings.csv")
    
    # Save uploaded file
    try:
        with open(csv_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Try reading the file to validate format
        df = pd.read_csv(csv_path)
        
        if not validate_city_data(df):
            os.remove(csv_path)  # Clean up invalid file
            raise ValueError("Uploaded file missing required columns. Required: building_id, Energy_Consumption, CO2_Usage")
        
        # Process the city data
        processed_df = process_city_data(city_name, input_filename=csv_path)
        
        # Generate future year predictions
        future_df = process_city_with_years(city_name, csv_path, 2024, [2025])
        
        return future_df
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(csv_path):
            os.remove(csv_path)
        raise ValueError(f"Error processing uploaded file: {str(e)}")