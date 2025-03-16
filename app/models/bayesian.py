import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

def classify_bayesian(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bayesian Network-based classification of buildings into A-F.
    Produces df['class_label'] with A-F classes, plus 
    df['Bayesian_Certainty'] for the probability of that class.
    
    Args:
        df (pd.DataFrame): Input dataframe with Energy_Consumption, CO2_Usage, Water_Usage columns
    
    Returns:
        pd.DataFrame: Modified dataframe with classification added
    """
    # Rename df to real_data internally, for clarity
    real_data = df.copy()

    # 1) Optimal reference point (minimum CO₂, Water, Energy)
    optimal_point = np.array([
        real_data["CO2_Usage"].min(),
        real_data["Water_Usage"].min(),
        real_data["Energy_Consumption"].min()
    ])

    # 2) Discretize with KBinsDiscretizer
    n_bins = 5
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    features = real_data[["CO2_Usage", "Water_Usage", "Energy_Consumption"]].values
    discretized_features = discretizer.fit_transform(features)
    discretized_data = pd.DataFrame(
        discretized_features,
        columns=["CO2_Level", "Water_Level", "Energy_Level"]
    )

    # 3) Create an initial class assignment (Euclidean from optimal)
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_optimal = scaler.transform([optimal_point])[0]
    distances = np.sqrt(np.sum((scaled_features - scaled_optimal)**2, axis=1))

    num_classes = 6  # A-F
    class_labels = ['A', 'B', 'C', 'D', 'E', 'F']
    bins = np.linspace(distances.min(), distances.max(), num_classes + 1)
    initial_classes = np.digitize(distances, bins)
    initial_classes = np.clip(initial_classes, 1, num_classes) - 1  # 0-based
    discretized_data["Class"] = [class_labels[i] for i in initial_classes]

    # 4) Define a Bayesian Network structure
    model = BayesianNetwork([
        ('CO2_Level', 'Class'),
        ('Water_Level', 'Class'),
        ('Energy_Level', 'Class'),
        ('CO2_Level', 'Energy_Level'),
        ('Water_Level', 'Energy_Level')
    ])

    # 5) Fit the BN with MaximumLikelihoodEstimator
    model.fit(discretized_data, estimator=MaximumLikelihoodEstimator)

    # 6) Infer class probabilities for each building
    inference = VariableElimination(model)

    def get_class_probabilities(co2_level, water_level, energy_level):
        evidence = {
            'CO2_Level': co2_level,
            'Water_Level': water_level,
            'Energy_Level': energy_level
        }
        return inference.query(variables=['Class'], evidence=evidence)

    # Create arrays to hold results
    bayesian_classes = []
    certainty_scores = []

    for row in discretized_features:
        co2_level, water_level, energy_level = row
        prob_dist = get_class_probabilities(co2_level, water_level, energy_level)
        
        # Extract class probabilities
        probs = {}
        for j, state in enumerate(prob_dist.state_names['Class']):
            probs[state] = prob_dist.values[j]
        
        # Pick the most likely class
        best_class = max(probs, key=probs.get)
        bayesian_classes.append(best_class)

        # Certainty = probability of that top class
        certainty_scores.append(probs[best_class])

    # Attach results
    real_data['Bayesian_Class'] = bayesian_classes
    real_data['Bayesian_Certainty'] = certainty_scores

    # For uniformity with other classification methods,
    # we store final classification in 'class_label'
    real_data['class_label'] = real_data['Bayesian_Class']

    return real_data