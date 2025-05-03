import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

def classify_bayesian(
    df: pd.DataFrame,
    features: list,
    n_bins: int = 5,
    class_labels: list = ['A','B','C','D','E','F']
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

    # 0 ▸ Compute the “optimal” reference point
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
    disc_df = pd.DataFrame(X_disc, columns=level_names, index=real_data.index)

    # 2 ▸ Initial class assignment by Euclidean distance to optimal
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    opt_scaled = scaler.transform([optimal_point])[0]
    distances = np.linalg.norm(X_scaled - opt_scaled, axis=1)

    # Digitize distances into equal-width bins for class labels
    bins = np.linspace(distances.min(), distances.max(), n_classes + 1)
    idx = np.digitize(distances, bins, right=False) - 1
    idx = np.clip(idx, 0, n_classes-1)
    disc_df['Class'] = [class_labels[i] for i in idx]

    # 3 ▸ Build a naïve‐Bayes structure: every feature‐level → Class
    edges = [(lvl, 'Class') for lvl in level_names]
    model = BayesianNetwork(edges)

    # 4 ▸ Fit parameters
    model.fit(disc_df, estimator=MaximumLikelihoodEstimator)

    # 5 ▸ Prepare inference
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
    for levels in X_disc:
        ev = {lvl: int(levels[j]) for j, lvl in enumerate(level_names)}
        probs = query_class(ev)
        best = max(probs, key=probs.get)
        bayes_labels.append(best)
        certainties.append(probs[best])

    real_data['class_label'] = bayes_labels
    real_data['Bayesian_Certainty'] = certainties
    return real_data
