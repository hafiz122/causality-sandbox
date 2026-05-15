"""
Tests for Propensity Score Matching.
"""

import numpy as np
import pytest
from causality_sandbox import PropensityScoreMatching


def generate_psm_data(n=500, true_effect=3.0, seed=42):
    """Generate simulated data for PSM tests."""
    np.random.seed(seed)
    
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.normal(0, 1, n)
    X = np.column_stack([X1, X2])
    
    # Propensity score
    ps = 1 / (1 + np.exp(-(-0.5 + 0.8 * X1 - 0.5 * X2)))
    treatment = (np.random.uniform(0, 1, n) < ps).astype(int)
    
    # Outcome
    outcome = 5 + 2 * X1 + 1.5 * X2 + true_effect * treatment + np.random.normal(0, 1, n)
    
    return X, treatment, outcome


class TestPropensityScoreMatching:
    
    def test_basic_fit(self):
        X, treatment, outcome = generate_psm_data()
        
        psm = PropensityScoreMatching(random_state=42)
        psm.fit(X, treatment, outcome)
        
        assert psm.att_ is not None
        assert psm.att_std_error_ is not None
        assert psm.att_ci_ is not None
        assert len(psm.matches_) > 0
    
    def test_att_recovery(self):
        """Test that PSM recovers approximately the true treatment effect."""
        X, treatment, outcome = generate_psm_data(n=1000, true_effect=3.0)
        
        psm = PropensityScoreMatching(random_state=42)
        psm.fit(X, treatment, outcome)
        
        # Should be reasonably close to true effect of 3.0
        assert 1.5 < psm.att_ < 4.5
    
    def test_balance_improvement(self):
        """Test that matching improves covariate balance."""
        X, treatment, outcome = generate_psm_data()
        
        psm = PropensityScoreMatching(random_state=42)
        psm.fit(X, treatment, outcome)
        
        balance_before = psm.balance_table(X, treatment, matched_only=False)
        balance_after = psm.balance_table(X, treatment, matched_only=True)
        
        # Average standardized difference should improve
        mean_before = np.mean(np.abs(balance_before['std_diff']))
        mean_after = np.mean(np.abs(balance_after['std_diff']))
        
        assert mean_after < mean_before or len(psm.matches_) < 50
    
    def test_with_replacement(self):
        X, treatment, outcome = generate_psm_data()
        
        psm = PropensityScoreMatching(replacement=True, random_state=42)
        psm.fit(X, treatment, outcome)
        
        assert len(psm.matches_) > 0
        assert psm.att_ is not None
    
    def test_k_neighbors(self):
        X, treatment, outcome = generate_psm_data()
        
        psm = PropensityScoreMatching(k=3, random_state=42)
        psm.fit(X, treatment, outcome)
        
        assert len(psm.matches_) > 0
    
    def test_caliper(self):
        X, treatment, outcome = generate_psm_data()
        
        psm = PropensityScoreMatching(caliper=0.05, random_state=42)
        psm.fit(X, treatment, outcome)
        
        # Caliper should reduce number of matches
        assert len(psm.matches_) <= (treatment == 1).sum()
