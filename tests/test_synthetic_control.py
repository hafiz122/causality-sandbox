"""
Tests for Synthetic Control Method.
"""

import numpy as np
import pytest
from causality_sandbox import SyntheticControl


def generate_sc_data(n_pre=20, n_post=10, n_controls=5, true_effect=10.0, seed=42):
    """Generate synthetic control panel data."""
    np.random.seed(seed)
    
    time = np.arange(n_pre + n_post)
    trend = 50 + 2 * time
    
    # Treatment effect kicks in after n_pre periods
    te = np.concatenate([np.zeros(n_pre), np.full(n_post, true_effect)])
    treated = trend + te + np.random.normal(0, 3, n_pre + n_post)
    
    # Controls: similar trends
    pre_controls = np.zeros((n_pre, n_controls))
    post_controls = np.zeros((n_post, n_controls))
    
    for i in range(n_controls):
        control_trend = 50 + (2 + np.random.normal(0, 0.2)) * time
        noise = np.random.normal(0, 4, n_pre + n_post)
        full = control_trend + noise
        pre_controls[:, i] = full[:n_pre]
        post_controls[:, i] = full[n_pre:]
    
    return treated[:n_pre], pre_controls, treated[n_pre:], post_controls


class TestSyntheticControl:
    
    def test_basic_fit(self):
        pre_t, pre_c, post_t, post_c = generate_sc_data()
        
        sc = SyntheticControl()
        sc.fit(pre_t, pre_c, post_t, post_c)
        
        assert sc.weights_ is not None
        assert np.abs(np.sum(sc.weights_) - 1.0) < 1e-6
        assert np.all(sc.weights_ >= -1e-10)  # Non-negative
        assert sc.treatment_effect_ is not None
        assert sc.rmspe_ > 0
    
    def test_treatment_effect_recovery(self):
        """Test that SC recovers approximately the true treatment effect."""
        pre_t, pre_c, post_t, post_c = generate_sc_data(
            n_pre=30, n_post=10, n_controls=8, true_effect=10.0
        )
        
        sc = SyntheticControl()
        sc.fit(pre_t, pre_c, post_t, post_c)
        
        avg_te = np.mean(sc.treatment_effect_)
        # Should be reasonably close to true effect of 10
        assert 5 < avg_te < 15
    
    def test_pre_treatment_fit(self):
        """Test that synthetic closely matches pre-treatment outcomes."""
        pre_t, pre_c, post_t, post_c = generate_sc_data(n_pre=30, n_controls=8)
        
        sc = SyntheticControl()
        sc.fit(pre_t, pre_c, post_t, post_c)
        
        # RMSPE should be relatively small compared to outcome scale
        assert sc.rmspe_ < 10
    
    def test_simplex_method(self):
        pre_t, pre_c, post_t, post_c = generate_sc_data()
        
        sc = SyntheticControl(optimization_method='simplex')
        sc.fit(pre_t, pre_c, post_t, post_c)
        
        assert sc.weights_ is not None
        assert np.abs(np.sum(sc.weights_) - 1.0) < 1e-4
    
    def test_placebo_test(self):
        pre_t, pre_c, post_t, post_c = generate_sc_data(n_controls=8)
        
        sc = SyntheticControl()
        sc.fit(pre_t, pre_c, post_t, post_c)
        
        all_pre = np.column_stack([pre_t.reshape(-1, 1), pre_c])
        all_post = np.column_stack([post_t.reshape(-1, 1), post_c])
        
        placebo = sc.placebo_test(all_pre, all_post)
        
        assert placebo['p_value'] is not None
        assert 0 <= placebo['p_value'] <= 1
