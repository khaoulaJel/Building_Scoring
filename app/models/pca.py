import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def classify_pca(df: pd.DataFrame) -> pd.DataFrame:
    """
    PCA-based classification where lower PC1 => better performance
    Produces 'class_label' with A-F
    
    Args:
        df (pd.DataFrame): Input dataframe with Energy_Consumption, CO2_Usage, Water_Usage columns
    
    Returns:
        pd.DataFrame: Modified dataframe with classification added
    """
    df = df.copy()
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