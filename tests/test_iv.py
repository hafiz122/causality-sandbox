"""
Tests for Instrumental Variables (2SLS).
"""

import numpy as np
import pytest
from causality_sandbox import InstrumentalVariable


def generate_iv_data(n=1000, true_effect=5.0, seed=42):
    """Generate data with endogenous treatment and valid instrument."""
    np.random.seed(seed)
    
    # Confounder
    u = np.random.normal(0, 1, n)
    
    # Instrument
    z = np.random.normal(0, 1, n)
    
    # Endogenous treatment
    treatment = 10 + 2.0 * z + 1.5 * u + np.random.normal(0, 1, n)
    
    # Outcome with endogeneity
    outcome = 20 + true_effect * treatment + 3.0 * u + np.random.normal(0, 3, n)
    
    return treatment, z, outcome


class TestInstrumentalVariable:
    
    def test_basic_fit(self):
        treatment, z, outcome = generate_iv_data()
        
        iv = InstrumentalVariable()
        iv.fit(endogenous=treatment, instrument=z, outcome=outcome)
        
        assert iv.late_ is not None
        assert iv.std_error_ is not None
        assert iv.ci_ is not None
        assert iv.first_stage_f_ > 0
    
    def test_late_recovery(self):
        """Test that 2SLS recovers approximately the true causal effect."""
        treatment, z, outcome = generate_iv_data(n=2000, true_effect=5.0)
        
        iv = InstrumentalVariable()
        iv.fit(endogenous=treatment, instrument=z, outcome=outcome)
        
        # Should be close to true effect of 5.0
        assert 3.0 < iv.late_ < 7.0
    
    def test_first_stage_strength(self):
        """Test that first stage F-statistic indicates instrument strength."""
        treatment, z, outcome = generate_iv_data()
        
        iv = InstrumentalVariable()
        iv.fit(endogenous=treatment, instrument=z, outcome=outcome)
        
        # F-statistic should exceed rule-of-thumb threshold of 10
        assert iv.first_stage_f_ > 10
    
    def test_hausman_detects_endogeneity(self):
        """Test Hausman test detects endogenous treatment."""
        treatment, z, outcome = generate_iv_data()
        
        iv = InstrumentalVariable()
        hausman = iv.durbin_wu_hausman(treatment, z, outcome)
        
        # Should detect endogeneity in this setup
        assert hausman['endogenous']
    
    def test_with_controls(self):
        """Test IV with control variables."""
        treatment, z, outcome = generate_iv_data()
        controls = np.random.normal(0, 1, len(treatment))
        
        iv = InstrumentalVariable()
        iv.fit(endogenous=treatment, instrument=z, outcome=outcome, controls=controls)
        
        assert iv.late_ is not None
