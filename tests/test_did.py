"""
Tests for Difference-in-Differences.
"""

import numpy as np
import pandas as pd
import pytest
from causality_sandbox import DifferenceInDifferences


def generate_did_data(n_units=200, n_periods=3, true_effect=4.0, seed=42):
    """Generate panel data for DiD tests."""
    np.random.seed(seed)
    
    units = []
    for i in range(n_units):
        treated = 1 if i < n_units // 2 else 0
        unit_fe = np.random.normal(0, 3)
        
        for t in range(n_periods):
            post = 1 if t == n_periods - 1 else 0
            outcome = 10 + unit_fe + 2 * t + 1.5 * treated + true_effect * treated * post
            outcome += np.random.normal(0, 2)
            
            units.append({
                'unit_id': i,
                'period': t,
                'treated': treated,
                'post': post,
                'outcome': outcome
            })
    
    return pd.DataFrame(units)


class TestDifferenceInDifferences:
    
    def test_basic_fit(self):
        df = generate_did_data()
        
        did = DifferenceInDifferences()
        did.fit(df, outcome='outcome', treatment_col='treated', time_col='post')
        
        assert did.att_ is not None
        assert did.std_error_ is not None
        assert did.ci_ is not None
    
    def test_att_recovery(self):
        """Test that DiD recovers approximately the true treatment effect."""
        df = generate_did_data(n_units=500, true_effect=4.0)
        
        did = DifferenceInDifferences()
        did.fit(df, outcome='outcome', treatment_col='treated', time_col='post')
        
        # Should be close to true effect of 4.0
        assert 2.5 < did.att_ < 5.5
    
    def test_placebo_test(self):
        """Test placebo test returns expected structure."""
        df = generate_did_data(n_units=500, n_periods=4, true_effect=4.0, seed=123)
        
        did = DifferenceInDifferences()
        did.fit(df, outcome='outcome', treatment_col='treated', time_col='post')
        
        placebo = did.placebo_test(
            df, outcome='outcome', treatment_col='treated',
            time_col='post', placebo_period=1
        )
        
        # Should return valid values
        assert 'placebo_att' in placebo
        assert 'p_value' in placebo
    
    def test_with_unit_fe(self):
        df = generate_did_data()
        
        did = DifferenceInDifferences()
        did.fit(
            df, outcome='outcome', treatment_col='treated',
            time_col='post', unit_col='unit_id'
        )
        
        assert did.att_ is not None
