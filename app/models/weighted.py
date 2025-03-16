import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def classify_weighted(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted classification of KPIs. Lower 'Global_Score' => better (A)
    
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
    
    weights = np.array([0.2, 0.3, 0.5])  # Example weighting
    df["Global_Score"] = np.sum(df[norm_cols] * weights, axis=1)
    
    percentiles = df["Global_Score"].quantile([0.1, 0.3, 0.6, 0.8, 0.9])
    
    def classify_score(score):
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
            
    df["Global_Class"] = df["Global_Score"].apply(classify_score)
    df['class_label'] = df['Global_Class']
    return df