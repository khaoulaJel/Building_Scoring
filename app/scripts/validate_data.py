from datetime import time
from pathlib import Path
import pandas as pd
from pyproj import Transformer
import requests
import streamlit as st
import pandas as pd
from models.euclidean import classify_euclidean
from models.mahalanobis import classify_mahalanobis
from models.pca import classify_pca
from models.weighted import classify_weighted
from models.bayesian import classify_bayesian


REQUIRED_COLUMNS = [
    "building_id", "latitude", "longitude",
    "CO2_Usage", "Water_Usage", "Energy_Consumption"
]
INTENSITY_COLUMNS = ["Energy_Intensity", "CO2_Intensity"]


# Function to validate and preprocess dataset
def validate_and_preprocess_dataset(df, scoring_basis):
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        if scoring_basis == "Per m² (kWh/m²/year)":
            if "Energy_Intensity" in df.columns and "CO2_Intensity" in df.columns:
                df["Energy_Consumption"] = df["Energy_Intensity"]
                df["CO2_Usage"] = df["CO2_Intensity"]
            else:
                st.error(f"Missing required columns for intensity-based scoring: {missing_cols} or Energy_Intensity/CO2_Intensity")
                return None
        else:
            st.error(f"Missing required columns: {missing_cols}")
            return None
    
    numeric_cols = ["latitude", "longitude", "CO2_Usage", "Water_Usage", "Energy_Consumption"]
    for col in numeric_cols:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            st.error(f"Column {col} must contain numeric values")
            return None
    
    df = df.dropna(subset=REQUIRED_COLUMNS)
    
    if df.empty:
        st.error("Dataset is empty after preprocessing")
        return None
    
    return df

@st.cache_data
# Modified function to handle DataFrame output from classification methods

def add_classifications(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add classification columns to the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        pd.DataFrame: DataFrame with classification columns
    """
    df = df.copy()
    
    # Features for classification
    features = ["Energy_Consumption", "CO2_Usage", "Water_Usage"]
    
    # Euclidean Classification
    try:
        df_euclidean = classify_euclidean(df, features=features)
        df["class_euclidean"] = df_euclidean["class_euclidean"]
    except Exception as e:
        st.error(f"Euclidean Classification failed: {str(e)}")
        df["class_euclidean"] = "C"  # Fallback
    
    # Mahalanobis Classification
    try:
        df_mahalanobis = classify_mahalanobis(df, features=features, return_distance=True)
        df["class_mahalanobis"] = df_mahalanobis["class_label"]
        if "Mahalanobis_Distance" in df_mahalanobis.columns:
            df["Mahalanobis_Distance"] = df_mahalanobis["Mahalanobis_Distance"]
    except Exception as e:
        st.error(f"Mahalanobis Classification failed: {str(e)}")
        df["class_mahalanobis"] = "C"  # Fallback
    
    # PCA Classification
    try:
        df_pca = classify_pca(df, features=features)
        df["class_pca"] = df_pca["class_pca"]
    except Exception as e:
        st.error(f"PCA Classification failed: {str(e)}")
        df["class_pca"] = "C"  # Fallback
    
    # Weighted Classification
    try:
        df_weighted = classify_weighted(df, features=features)
        df["class_weighted"] = df_weighted["class_weighted"]
    except Exception as e:
        st.error(f"Weighted Classification failed: {str(e)}")
        df["class_weighted"] = "C"  # Fallback
    
    # Bayesian Classification
    try:
        df_bayesian = classify_bayesian(df, features=features)
        df["class_bayesian"] = df_bayesian["class_bayesian"]
        if "Bayesian_Certainty" in df_bayesian.columns:
            df["Bayesian_Certainty"] = df_bayesian["Bayesian_Certainty"]
    except Exception as e:
        st.error(f"Bayesian Classification failed: {str(e)}")
        df["class_bayesian"] = "C"  # Fallback
    
    return df

# Now, let's fix the classification method selection code
def apply_classification_method(df, classification_method, features=None):
    """
    Apply the selected classification method to the dataframe
    
    Args:
        df (pd.DataFrame): DataFrame containing building data
        classification_method (str): Name of the classification method to apply
        features (list, optional): List of feature columns to use for classification
        
    Returns:
        pd.DataFrame: DataFrame with the selected classification applied
    """
    # Default features if none are provided
    if features is None:
        features = ["Energy_Consumption", "CO2_Usage", "Water_Usage"]
    
    # Map method names to column names
    class_column_mapping = {
        "Euclidean Distance": "class_euclidean",
        "Mahalanobis Distance": "class_mahalanobis",
        "PCA Classification": "class_pca",
        "Weighted Classification": "class_weighted",
        "Bayesian Classification": "class_bayesian"
    }
    
    # Get the corresponding column name
    selected_class_column = class_column_mapping[classification_method]
    
    # Apply the selected classification method
    with st.spinner(f"Applying {classification_method}..."):
        try:
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
            
            # Set the class_label to the selected classification
            df["class_label"] = df[selected_class_column]
            
        except Exception as e:
            st.error(f"Error applying {classification_method}: {str(e)}")
            st.warning("Using default classification instead")
            # Keep existing classification or set to default
            if selected_class_column in df.columns:
                df["class_label"] = df[selected_class_column]
            else:
                df["class_label"] = "C"
    
    return df

@st.cache_data
def fetch_ademe_data(city, limit=1000):
    """
    Fetches DPE data for a city from data.gouv.fr API or CSV download.
    Returns a DataFrame with required columns for building analytics dashboard.
    """
    cache_dir = Path("data")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{city.lower()}_buildings.csv"
    
    # Check if cached data exists
    if cache_file.exists():
        try:
            df = pd.read_csv(cache_file)
            required_cols = ['building_id', 'latitude', 'longitude', 'CO2_Usage', 'Water_Usage', 'Energy_Consumption']
            if all(col in df.columns for col in required_cols):
                return df[required_cols]
        except Exception as e:
            st.warning(f"Error reading cached data for {city}: {str(e)}")
    
    # Step 1: Find dataset ID via catalog search
    dataset_id = None
    try:
        catalog_url = "https://www.data.gouv.fr/api/1/datasets/?q=dpe&organization=534fff75a3a7292c64a77e8e"
        response = requests.get(catalog_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            for dataset in data.get('data', []):
                if 'dpe-france' in dataset.get('slug', '').lower() or 'diagnostic-de-performance-energetique' in dataset.get('slug', '').lower():
                    dataset_id = dataset['id']
                    break
            if not dataset_id:
                st.warning(f"No matching DPE dataset found in catalog for {city}")
        else:
            st.warning(f"Catalog API returned {response.status_code}: {response.text}")
    except Exception as e:
        st.warning(f"Catalog API error: {str(e)}")
    
    # Step 2: Try data.gouv.fr API with dataset_id
    if dataset_id:
        try:
            url = f"https://www.data.gouv.fr/api/1/datasets/{dataset_id}/records"
            postal_codes = {
                "Lyon": "69",
                "Paris": "75",
                "Marseille": "13",
                "Toulouse": "31"
            }
            params = {
                "where": f"nom_commune_ban='{city}' and code_postal_ban like '{postal_codes.get(city, '')}%'",
                "limit": limit,
                "select": "n_dpe,coordonnee_cartographique_x_ban,coordonnee_cartographique_y_ban,conso_energetique,emissions_ges,nom_commune_ban,code_postal_ban"
            }
            for attempt in range(3):
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    df = pd.DataFrame([record['fields'] for record in data.get('records', [])])
                    if df.empty:
                        st.warning(f"No data returned for {city} from data.gouv.fr API")
                        return None
                    
                    # Process data
                    df = process_data(df)
                    if df is not None:
                        df.to_csv(cache_file, index=False)
                        return df
                    return None
                elif response.status_code == 404:
                    st.warning(f"API returned 404 for dataset {dataset_id}: {response.text}. Falling back to CSV download.")
                    break
                else:
                    st.warning(f"Attempt {attempt + 1} failed with status {response.status_code}: {response.text}")
                    time.sleep(2)
        except Exception as e:
            st.warning(f"data.gouv.fr API error for {city}: {str(e)}. Falling back to CSV download.")
    
    # Step 3: Fallback to CSV download
    try:
        csv_url = "https://files.data.gouv.fr/dpe-france/dpe-france.csv"
        df = pd.read_csv(csv_url, low_memory=False)
        df = df[df['nom_commune_ban'].str.lower() == city.lower()]
        if df.empty:
            st.warning(f"No data found for {city} in CSV download")
            return None
        
        # Process data
        df = process_data(df)
        if df is not None:
            df.to_csv(cache_file, index=False)
            return df
        return None
    except Exception as e:
        st.error(f"Failed to fetch data for {city} via CSV: {str(e)}")
        return None

def process_data(df):
    """
    Processes raw DPE data into dashboard-compatible format.
    """
    # Rename and map columns
    df = df.rename(columns={
        'n_dpe': 'building_id',
        'coordonnee_cartographique_x_ban': 'longitude',
        'coordonnee_cartographique_y_ban': 'latitude',
        'conso_energetique': 'Energy_Consumption',  # kWh/m²/year
        'emissions_ges': 'CO2_Usage',  # kgCO₂eq/m²/year
        'nom_commune_ban': 'Nom_commune',
        'code_postal_ban': 'Code_postal'
    })
    
    # Add default Water_Usage
    df['Water_Usage'] = 5000  # Default value (L)
    
    # Convert coordinates (Lambert 93 to WGS84)
    transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326")
    df['latitude'], df['longitude'] = zip(*df.apply(
        lambda row: transformer.transform(row['longitude'], row['latitude'])
        if pd.notnull(row['latitude']) and pd.notnull(row['longitude']) else (None, None), axis=1))
    
    # Convert to numeric
    for col in ['latitude', 'longitude', 'Energy_Consumption', 'CO2_Usage']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing required columns
    required_cols = ['building_id', 'latitude', 'longitude', 'CO2_Usage', 'Water_Usage', 'Energy_Consumption']
    df = df.dropna(subset=required_cols)
    
    if df.empty:
        st.warning("No valid data after processing")
        return None
    
    return df[required_cols]
    
    
