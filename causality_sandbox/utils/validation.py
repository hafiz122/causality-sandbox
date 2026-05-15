"""
Cross-validation and sensitivity analysis utilities.
"""

import numpy as np
from sklearn.model_selection import KFold


def cross_validate_att(estimator_class, X, treatment, outcome, n_splits=5, **kwargs):
    """
    Cross-validate a causal estimator by partitioning data and checking
    stability of the ATT estimate across folds.
    
    Useful for assessing robustness of causal estimates.
    
    Parameters
    ----------
    estimator_class : class
        Estimator class (e.g., PropensityScoreMatching).
    X, treatment, outcome : array-like
        Data.
    n_splits : int, default 5
        Number of CV folds.
    **kwargs : dict
        Additional arguments for estimator.
    
    Returns
    -------
    dict with ATT estimates across folds and summary statistics.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    X = np.asarray(X)
    treatment = np.asarray(treatment).flatten()
    outcome = np.asarray(outcome).flatten()
    
    estimates = []
    
    for fold, (train_idx, _) in enumerate(kf.split(X)):
        try:
            est = estimator_class(**kwargs)
            est.fit(X[train_idx], treatment[train_idx], outcome[train_idx])
            estimates.append(est.att_)
        except:
            continue
    
    estimates = np.array(estimates)
    
    return {
        'estimates': estimates,
        'mean': np.mean(estimates),
        'std': np.std(estimates),
        'cv_ratio': np.std(estimates) / np.abs(np.mean(estimates)) if np.mean(estimates) != 0 else np.inf,
        'stable': np.std(estimates) / np.abs(np.mean(estimates)) < 0.5 if np.mean(estimates) != 0 else False
    }
