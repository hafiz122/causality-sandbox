"""
Covariate balance diagnostics for observational studies.
"""

import numpy as np
import pandas as pd


def check_balance(X, treatment, weights=None):
    """
    Compute standardized mean differences for covariate balance assessment.
    
    Values near 0 indicate good balance between treated and control groups.
    Values above 0.1 or 0.25 suggest imbalance (Rubin, 2001).
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Covariates.
    treatment : array-like of shape (n_samples,)
        Treatment indicator.
    weights : array-like or None
        Optional sample weights (e.g., from propensity score weighting).
    
    Returns
    -------
    pd.DataFrame with balance statistics for each covariate.
    """
    X = np.asarray(X)
    treatment = np.asarray(treatment).flatten()
    
    treated_mask = treatment == 1
    control_mask = treatment == 0
    
    results = []
    
    for i in range(X.shape[1]):
        if weights is not None:
            w = np.asarray(weights)
            treated_mean = np.average(X[treated_mask, i], weights=w[treated_mask])
            control_mean = np.average(X[control_mask, i], weights=w[control_mask])
        else:
            treated_mean = np.mean(X[treated_mask, i])
            control_mean = np.mean(X[control_mask, i])
        
        pooled_std = np.sqrt(
            (np.var(X[treated_mask, i], ddof=1) + 
             np.var(X[control_mask, i], ddof=1)) / 2
        )
        
        std_diff = (treated_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        
        results.append({
            'covariate': f'X{i}',
            'treated_mean': treated_mean,
            'control_mean': control_mean,
            'pooled_std': pooled_std,
            'std_diff': std_diff,
            'balanced': abs(std_diff) < 0.1
        })
    
    return pd.DataFrame(results)
