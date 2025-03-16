import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis

def classify_mahalanobis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mahalanobis-distance-based classification
    Produces 'class_label' with A-F
    
    Args:
        df (pd.DataFrame): Input dataframe with Energy_Consumption, CO2_Usage, Water_Usage columns
    
    Returns:
        pd.DataFrame: Modified dataframe with classification added
    """
    df = df.copy()
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