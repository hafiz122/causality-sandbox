"""
Causality Sandbox: A lightweight causal inference toolkit for observational studies.

Provides methods for estimating causal effects from observational data:
    - Propensity Score Matching (PSM)
    - Difference-in-Differences (DiD)
    - Regression Discontinuity Design (RDD)
    - Instrumental Variables (IV)
    - Synthetic Control Method
"""

from causality_sandbox.methods.propensity_score import PropensityScoreMatching
from causality_sandbox.methods.difference_in_differences import DifferenceInDifferences
from causality_sandbox.methods.regression_discontinuity import RegressionDiscontinuity
from causality_sandbox.methods.instrumental_variable import InstrumentalVariable
from causality_sandbox.methods.synthetic_control import SyntheticControl

__version__ = "0.1.0"
__author__ = "Muhammad Hafiz bin Ismail"

__all__ = [
    "PropensityScoreMatching",
    "DifferenceInDifferences",
    "RegressionDiscontinuity",
    "InstrumentalVariable",
    "SyntheticControl",
]
