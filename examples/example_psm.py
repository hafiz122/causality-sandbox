"""
Example: Propensity Score Matching on simulated data.

This example demonstrates how to use PSM to estimate the causal effect
of a job training program on earnings.
"""

import numpy as np
from causality_sandbox import PropensityScoreMatching
from causality_sandbox.utils import check_overlap

# Set seed for reproducibility
np.random.seed(42)

# Simulate data
n = 1000

# Confounders: age, education, prior earnings
age = np.random.normal(35, 10, n)
education = np.random.normal(12, 3, n)
prior_earnings = np.random.normal(30000, 10000, n)

# Treatment assignment depends on confounders (selection bias)
logit_treat = -2 + 0.05 * (age - 35) + 0.3 * (education - 12) + 0.00002 * (prior_earnings - 30000)
prob_treat = 1 / (1 + np.exp(-logit_treat))
treatment = (np.random.uniform(0, 1, n) < prob_treat).astype(int)

# True treatment effect = $2500
# Outcome: earnings after training
noise = np.random.normal(0, 5000, n)
outcome = (
    25000 
    + 500 * (age - 35)
    + 1500 * (education - 12)
    + 0.8 * (prior_earnings - 30000)
    + 2500 * treatment  # True causal effect
    + noise
)

# Prepare covariates
X = np.column_stack([
    (age - age.mean()) / age.std(),
    (education - education.mean()) / education.std(),
    (prior_earnings - prior_earnings.mean()) / prior_earnings.std()
])

print("=" * 60)
print("Propensity Score Matching Example")
print("Simulated Job Training Program")
print("=" * 60)
print(f"\nSample size: {n}")
print(f"Treated: {treatment.sum()}, Control: {(1 - treatment).sum()}")
print(f"True treatment effect: $2500")

# Check overlap
print("\n--- Overlap Diagnostics ---")
from sklearn.linear_model import LogisticRegression
ps_model = LogisticRegression(max_iter=1000).fit(X, treatment)
ps = ps_model.predict_proba(X)[:, 1]
overlap = check_overlap(ps, treatment)
print(f"Common support: [{overlap['common_support_lower']:.3f}, {overlap['common_support_upper']:.3f}]")
print(f"Good overlap: {overlap['good_overlap']}")

# Fit PSM
print("\n--- Matching ---")
psm = PropensityScoreMatching(
    model='logistic',
    caliper=0.1,
    replacement=False,
    k=1,
    random_state=42
)
psm.fit(X, treatment, outcome)

# Results
print("\n")
psm.summary()

# Balance check
print("\n--- Covariate Balance ---")
balance = psm.balance_table(X, treatment, matched_only=True)
print(balance[['treated_mean', 'control_mean', 'std_diff']])
