"""
Example: Instrumental Variables (2SLS) on simulated data.

Demonstrates IV estimation when treatment is endogenous.
"""

import numpy as np
from causality_sandbox import InstrumentalVariable

np.random.seed(42)

# Simulate: Effect of education on earnings
# Education is endogenous (ability affects both education and earnings)
# Instrument: distance to nearest college (affects education but not directly earnings)

n = 1000

# Unobserved ability (confounder)
ability = np.random.normal(0, 1, n)

# Instrument: distance to college (standardized, negative = closer)
distance = np.random.normal(0, 1, n)

# Endogenous treatment: years of education
# Closer to college = more education
education = (
    12
    - 0.8 * distance    # Instrument effect
    + 1.5 * ability     # Confounding
    + np.random.normal(0, 1, n)
)

# Outcome: earnings (in thousands)
# True return to education: $5k per year
true_effect = 5
earnings = (
    20
    + true_effect * education  # Causal effect
    + 8 * ability              # Confounding
    + np.random.normal(0, 5, n)
)

print("=" * 60)
print("Instrumental Variables Example")
print("Returns to Education")
print("=" * 60)
print(f"\nSample size: {n}")
print(f"True causal effect: ${true_effect}k per year of education")
print(f"Instrument: Distance to nearest college")

# Fit IV
print("\n--- 2SLS Estimation ---")
iv = InstrumentalVariable()
iv.fit(endogenous=education, instrument=distance, outcome=earnings)

# Results
print("\n")
iv.summary()

# Compare with naive OLS
print("\n--- Comparison with OLS ---")
import statsmodels.api as sm
X_ols = sm.add_constant(education)
ols = sm.OLS(earnings, X_ols).fit()
print(f"Naive OLS estimate: ${ols.params[1]:.2f}k (biased upward due to confounding)")
print(f"2SLS estimate:      ${iv.late_:.2f}k (closer to true ${true_effect}k)")

# Hausman test
print("\n--- Hausman Test ---")
hausman = iv.durbin_wu_hausman(education, distance, earnings)
print(f"Hausman statistic: {hausman['hausman_statistic']:.4f}")
print(f"p-value: {hausman['p_value']:.4f}")
print(f"Endogeneity detected: {hausman['endogenous']}")
