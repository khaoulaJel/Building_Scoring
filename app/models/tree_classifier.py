import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score

def classify_robust_tree(
    df: pd.DataFrame,
    numeric_features: list,
    categorical_features: list = None,
    target_column: str = 'class_label',
    test_size: float = 0.2,
    random_state: int = 42,
    return_metrics: bool = True
) -> pd.DataFrame:
    """
    Trains a robust tree-based classifier for building classification (A–F).
    
    Args:
        df: Input dataframe
        numeric_features: List of numeric feature column names
        categorical_features: List of categorical feature column names (optional)
        target_column: Name of target column
        test_size: Proportion for test split
        random_state: Random seed
        return_metrics: Whether to print evaluation metrics
    
    Returns:
        DataFrame with added predictions and probabilities
    """
    
    data = df.copy()
    
    # Prepare feature lists
    if categorical_features is None:
        categorical_features = []
    
    all_features = numeric_features + categorical_features
    
    # Check if all features exist
    missing_features = [f for f in all_features if f not in data.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataframe: {missing_features}")
    
    X = data[all_features]
    
    # Create preprocessing pipeline
    preprocessors = []
    
    if numeric_features:
        # Apply QuantileTransformer to numeric features
        numeric_transformer = QuantileTransformer(output_distribution='normal')
        preprocessors.append(('numeric', numeric_transformer, numeric_features))
    
    if categorical_features:
        # For categorical features, we could add processing here
        pass
        
    # Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=preprocessors,
        remainder='passthrough'
    )
    
    # Transform features
    X_transformed = preprocessor.fit_transform(X)
    
    # Initialize base model with balanced class weights
    base_model = LGBMClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=random_state
    )

    # If we don't have a target column in input data, create synthetic classes
    # This allows us to still use the classifier for new data
    if target_column not in data.columns:
        print("No target column found, generating initial classes from feature distributions...")
        # Use feature distributions to create initial class assignments
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_transformed)
        distances = np.linalg.norm(X_scaled, axis=1)
        data[target_column] = pd.qcut(
            distances, 
            q=6,  # A-F classes
            labels=['A', 'B', 'C', 'D', 'E', 'F']
        ).astype(str)

    y = data[target_column]
    
    # Train calibrated model
    calibrated_model = CalibratedClassifierCV(
        base_model, 
        method='isotonic', 
        cv=3
    )
    
    try:
        calibrated_model.fit(X_transformed, y)
    except Exception as e:
        print(f"Warning: Error fitting model: {e}")
        print("Falling back to simpler classification...")
        # Fallback to simple distance-based classification
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_transformed)
        distances = np.linalg.norm(X_scaled, axis=1)
        data['class_tree'] = pd.qcut(
            distances, 
            q=6, 
            labels=['A', 'B', 'C', 'D', 'E', 'F']
        ).astype(str)
        data['class_label'] = data['class_tree']
        return data

    # Get predictions for all data
    data['class_tree'] = calibrated_model.predict(X_transformed)
    data['class_label'] = data['class_tree']  # For consistency
      # Only show metrics if we had a target column in input data
    if return_metrics and target_column in df.columns:
        print("=== Model Performance ===")
        print(classification_report(y, data['class_tree']))
    
    # Ensure we have both class_tree and class_label for consistency
    data['class_tree'] = data['class_tree'].astype(str)
    data['class_label'] = data['class_tree']
    
    return data

# Example usage:
"""
# For numeric features only
result = classify_robust_tree(
    df=your_dataframe,
    numeric_features=['co2_emissions', 'energy_consumption', 'water_usage'],
    target_column='environmental_class'
)

# For mixed features
result = classify_robust_tree(
    df=your_dataframe,
    numeric_features=['co2_emissions', 'energy_consumption'],
    categorical_features=['zone_climatique', 'building_type'],
    target_column='environmental_class'
)
"""