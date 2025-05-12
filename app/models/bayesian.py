import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from pgmpy.models import DiscreteBayesianNetwork  # Use DiscreteBayesianNetwork instead of BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination


def classify_bayesian(
    df: pd.DataFrame,
    features: list,
    n_bins: int = 5,
    class_labels: list = ['A', 'B', 'C', 'D', 'E', 'F']
) -> pd.DataFrame:
    """
    Bayesian Network–based classification of buildings into classes.
    Works with any number of features.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        features (list of str): Column names to use as features.
        n_bins (int): Number of quantile bins for discretization.
        class_labels (list): Ordered labels for classes.
    
    Returns:
        pd.DataFrame: df with added columns:
          - 'class_label' : MAP class from the Bayesian network
          - 'Bayesian_Certainty' : P(class_label | evidence)
    """
    real_data = df.copy()
    X = real_data[features].values
    n_features = len(features)
    n_classes = len(class_labels)
    
    # 0 ▸ Compute the "optimal" reference point
    optimal_point = X.min(axis=0)
    
    # 1 ▸ Discretize each feature into n_bins quantiles
    discretizer = KBinsDiscretizer(
        n_bins=n_bins,
        encode='ordinal',
        strategy='quantile'
    )
    X_disc = discretizer.fit_transform(X)
    
    # level_names: e.g. ['CO2_Usage_Level', 'Water_Usage_Level', ...]
    level_names = [f"{feat}_Level" for feat in features]
    
    # Create DataFrame with discretized values
    # CRITICAL: Convert discretized values directly to strings with the same format
    # that pgmpy will use internally - this avoids the state mismatch error
    disc_df = pd.DataFrame(
        {level_names[i]: X_disc[:, i].astype(float).astype(str) for i in range(len(level_names))},
        index=real_data.index
    )
    
    # 2 ▸ Initial class assignment by Euclidean distance to optimal
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    opt_scaled = scaler.transform([optimal_point])[0]
    distances = np.linalg.norm(X_scaled - opt_scaled, axis=1)
    
    # Use quantile-based binning instead of equal-width bins
    # This better handles skewed distance distributions
    try:
        disc_df['Class'] = pd.qcut(
            distances, 
            q=n_classes, 
            labels=class_labels,
            duplicates='drop'
        )
        
        # Handle case where qcut fails due to too many duplicates
        if disc_df['Class'].isna().any():
            raise ValueError("Too many duplicate values for qcut")
        
    except ValueError:
        # Fallback to rank-based approach which guarantees equal-sized bins
        ranks = pd.Series(distances).rank(method='first')
        bin_edges = np.linspace(0, len(ranks), n_classes + 1).astype(int)
        bins = pd.cut(ranks, bins=bin_edges, labels=False, include_lowest=True)
        disc_df['Class'] = [class_labels[i] for i in bins]
    
    # Ensure Class column is string type
    disc_df['Class'] = disc_df['Class'].astype(str)
    
    # 3 ▸ Build a naïve‐Bayes structure: every feature‐level → Class
    edges = [(lvl, 'Class') for lvl in level_names]
    model = DiscreteBayesianNetwork(edges)
    
    # 4 ▸ Fit parameters
    model.fit(disc_df, estimator=MaximumLikelihoodEstimator)
    
    # Store the valid states for each feature from the fitted model
    # This will be used to check if evidence values are valid during inference
    valid_states = {}
    for node in model.nodes():
        cpd = model.get_cpds(node)
        if cpd is not None and node in level_names:
            valid_states[node] = list(cpd.state_names[node])
    
    # 5 ▸ Prepare inference engine
    infer = VariableElimination(model)
    
    def query_class(evidence: dict):
        """Return a dict: {label: probability}"""
        q = infer.query(variables=['Class'], evidence=evidence)
        return {
            state: q.values[i]
            for i, state in enumerate(q.state_names['Class'])
        }
    
    # 6 ▸ For each row, compute MAP class + certainty
    bayes_labels = []
    certainties = []
    
    for i, levels in enumerate(X_disc):
        # Convert discretized values to the same string format used during training
        evidence = {}
        for j, lvl in enumerate(level_names):
            value = str(float(levels[j]))
            
            # Check if this value exists in model's valid states
            if lvl in valid_states and value in valid_states[lvl]:
                evidence[lvl] = value
            else:
                # Skip this feature if its value isn't in the training data
                # This prevents the "unknown state" error
                continue
        
        try:
            # If we have valid evidence for at least one feature, make prediction
            if evidence:
                probs = query_class(evidence)
                best = max(probs, key=probs.get)
                bayes_labels.append(best)
                certainties.append(probs[best])
            else:
                # If we have no valid evidence, fall back to initial classification
                raise ValueError("No valid evidence variables")
                
        except Exception as e:
            # Handle any inference failures gracefully
            print(f"Warning: Inference failed for row {i}: {e}")
            # Fallback to the initial classification
            bayes_labels.append(disc_df['Class'].iloc[i])  
            certainties.append(0.0)  # Zero certainty indicates inference failure
    
    real_data['class_label'] = bayes_labels
    real_data['Bayesian_Certainty'] = certainties
    return real_data


