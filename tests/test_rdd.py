"""
Tests for Regression Discontinuity Design.
"""

import numpy as np
import pytest
from causality_sandbox import RegressionDiscontinuity


def generate_rdd_data(n=1000, cutoff=70, true_effect=0.5, seed=42):
    """Generate data for RDD tests."""
    np.random.seed(seed)
    
    running = np.random.normal(65, 12, n)
    treatment = (running >= cutoff).astype(float)
    outcome = 2.5 + 0.02 * running + true_effect * treatment + np.random.normal(0, 0.3, n)
    
    return running, outcome, treatment


class TestRegressionDiscontinuity:
    
    def test_basic_fit(self):
        running, outcome, _ = generate_rdd_data()
        
        rdd = RegressionDiscontinuity(cutoff=70)
        rdd.fit(running, outcome)
        
        assert rdd.late_ is not None
        assert rdd.std_error_ is not None
        assert rdd.ci_ is not None
        assert rdd.bandwidth_ > 0
    
    def test_effect_recovery(self):
        """Test that RDD recovers approximately the true treatment effect."""
        running, outcome, _ = generate_rdd_data(n=2000, true_effect=0.5)
        
        rdd = RegressionDiscontinuity(cutoff=70, bandwidth=15)
        rdd.fit(running, outcome)
        
        # Should be reasonably close to true effect of 0.5
        assert 0.2 < rdd.late_ < 0.8
    
    def test_mccrary_test(self):
        """Test McCrary density test on non-manipulated data."""
        running, outcome, _ = generate_rdd_data()
        
        rdd = RegressionDiscontinuity(cutoff=70)
        rdd.fit(running, outcome)
        
        mccrary = rdd.mccrary_test(running)
        
        # Should not detect manipulation in clean data
        assert not mccrary['manipulation_detected']
    
    def test_different_kernels(self):
        running, outcome, _ = generate_rdd_data()
        
        for kernel in ['uniform', 'triangular', 'epanechnikov']:
            rdd = RegressionDiscontinuity(cutoff=70, kernel=kernel)
            rdd.fit(running, outcome)
            assert rdd.late_ is not None
