import pandas as pd
import numpy as np

def classify_euclidean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Euclidean distance-based classification
    Produces a 'class_label' column with A-F
    
    Args:
        df (pd.DataFrame): Input dataframe with Energy_Consumption, CO2_Usage, Water_Usage columns
    
    Returns:
        pd.DataFrame: Modified dataframe with classification added
    """
    df = df.copy()
    df['energy_norm'] = df['Energy_Consumption'] / df['Energy_Consumption'].max()
    df['carbon_norm'] = df['CO2_Usage'] / df['CO2_Usage'].max()
    df['water_norm'] = df['Water_Usage'] / df['Water_Usage'].max()

    df['distance'] = np.sqrt(df['energy_norm']**2 + df['carbon_norm']**2 + df['water_norm']**2)

    thresholds = np.linspace(df['distance'].min(), df['distance'].max(), 7)
    df['class'] = np.digitize(df['distance'], thresholds[1:], right=True)

    class_labels = ['A', 'B', 'C', 'D', 'E', 'F']
    df['class_label'] = df['class'].apply(lambda x: class_labels[min(x, len(class_labels)-1)])
    return df