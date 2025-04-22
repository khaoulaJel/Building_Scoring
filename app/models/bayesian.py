import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

def classify_bayesian(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Bayesian Network-based classification of buildings into A-F.
    Produces df['class_label'] with A-F classes, plus 
    df['Bayesian_Certainty'] for the probability of that class.
    
    Args:
        df (pd.DataFrame): Input dataframe
        features (list): List of 3 columns (e.g. log1p_CO2_Usage, Water_Usage, log1p_Energy_Consumption)
    
    Returns:
        pd.DataFrame: Modified dataframe with classification added
    """
    assert len(features) == 3, "You must provide exactly 3 features"

    real_data = df.copy()
    
    # Optimal reference point (min values per feature)
    optimal_point = real_data[features].min().values

    # 1 ▸ Discretize features
    n_bins = 5
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    raw_values = real_data[features].values
    discretized_features = discretizer.fit_transform(raw_values)
    
    # Build column names dynamically
    level_names = [f"{feat}_Level" for feat in ["Feature1", "Feature2", "Feature3"]]
    discretized_data = pd.DataFrame(discretized_features, columns=level_names)

    # 2 ▸ Initial class via Euclidean distance to optimal
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(raw_values)
    scaled_optimal = scaler.transform([optimal_point])[0]
    distances = np.linalg.norm(scaled_values - scaled_optimal, axis=1)

    num_classes = 6
    class_labels = ['A', 'B', 'C', 'D', 'E', 'F']
    bins = np.linspace(distances.min(), distances.max(), num_classes + 1)
    digitized = np.digitize(distances, bins)
    digitized = np.clip(digitized, 1, num_classes) - 1  # convert to 0-based
    discretized_data["Class"] = [class_labels[i] for i in digitized]

    # 3 ▸ Define Bayesian Network structure
    model = BayesianNetwork([
        (level_names[0], "Class"),
        (level_names[1], "Class"),
        (level_names[2], "Class"),
        (level_names[0], level_names[2]),
        (level_names[1], level_names[2])
    ])

    model.fit(discretized_data, estimator=MaximumLikelihoodEstimator)
    inference = VariableElimination(model)

    def get_class_probabilities(l1, l2, l3):
        evidence = {
            level_names[0]: l1,
            level_names[1]: l2,
            level_names[2]: l3
        }
        return inference.query(variables=['Class'], evidence=evidence)

    bayesian_classes = []
    certainty_scores = []

    for row in discretized_features:
        prob_dist = get_class_probabilities(*row)

        probs = {
            state: prob_dist.values[j]
            for j, state in enumerate(prob_dist.state_names['Class'])
        }

        best_class = max(probs, key=probs.get)
        bayesian_classes.append(best_class)
        certainty_scores.append(probs[best_class])

    real_data["Bayesian_Class"] = bayesian_classes
    real_data["Bayesian_Certainty"] = certainty_scores
    real_data["class_label"] = real_data["Bayesian_Class"]

    return real_data
