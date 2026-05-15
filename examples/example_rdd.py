"""
Example: Regression Discontinuity Design on simulated data.

Demonstrates sharp RDD where treatment is assigned based on a test score cutoff.
"""

import numpy as np
from causality_sandbox import RegressionDiscontinuity

np.random.seed(42)

# Simulate: Scholarship given to students with test score >= 70
n = 1000
test_score = np.random.normal(65, 15, n)

# Treatment: scholarship for scores >= 70
scholarship = (test_score >= 70).astype(int)

# Outcome: future GPA (treatment effect = +0.3 GPA points)
gpa = (
    2.5
    + 0.02 * test_score
    + 0.3 * scholarship
    + np.random.normal(0, 0.4, n)
)

print("=" * 60)
print("Regression Discontinuity Design Example")
print("Scholarship Eligibility Based on Test Score")
print("=" * 60)
print(f"\nSample size: {n}")
print(f"Cutoff: 70")
print(f"Scholarship recipients: {scholarship.sum()}")
print(f"True treatment effect: +0.30 GPA points")

# Fit sharp RDD
print("\n--- Estimation ---")
rdd = RegressionDiscontinuity(
    cutoff=70,
    bandwidth='optimal',
    kernel='triangular',
    polynomial=1
)
rdd.fit(test_score, gpa)

# Results
print("\n")
rdd.summary()

# McCrary test for manipulation
print("\n--- Manipulation Check (McCrary) ---")
mccrary = rdd.mccrary_test(test_score)
print(f"Log-density discontinuity: {mccrary['theta']:.4f}")
print(f"p-value: {mccrary['p_value']:.4f}")
print(f"Manipulation detected: {mccrary['manipulation_detected']}")
