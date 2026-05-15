"""
Common support / overlap diagnostics for propensity score methods.
"""

import numpy as np
import warnings


def check_overlap(propensity_scores, treatment, trim=False):
    """
    Assess overlap between treatment and control propensity score distributions.
    
    Poor overlap indicates potential positivity violations.
    
    Parameters
    ----------
    propensity_scores : array-like
        Estimated propensity scores.
    treatment : array-like
        Treatment indicator.
    trim : bool, default False
        If True, return indices of units within common support.
    
    Returns
    -------
    dict with overlap diagnostics, or indices if trim=True.
    """
    ps = np.asarray(propensity_scores).flatten()
    t = np.asarray(treatment).flatten()
    
    ps_t = ps[t == 1]
    ps_c = ps[t == 0]
    
    # Common support boundaries
    lower = max(ps_t.min(), ps_c.min())
    upper = min(ps_t.max(), ps_c.max())
    
    # Check for overlap issues
    n_t_outside = np.sum((ps_t < lower) | (ps_t > upper))
    n_c_outside = np.sum((ps_c < lower) | (ps_c > upper))
    
    # Trim if requested
    if trim:
        mask = (ps >= lower) & (ps <= upper)
        return mask
    
    return {
        'common_support_lower': lower,
        'common_support_upper': upper,
        'treated_outside_support': n_t_outside,
        'control_outside_support': n_c_outside,
        'overlap_ratio': (upper - lower) / (ps.max() - ps.min()) if ps.max() > ps.min() else 0,
        'good_overlap': (n_t_outside + n_c_outside) < 0.05 * len(ps)
    }
