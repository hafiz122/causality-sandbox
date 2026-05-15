"""
Example: Difference-in-Differences on simulated panel data.

Demonstrates the classic DiD setup with pre/post and treatment/control groups.
"""

import numpy as np
import pandas as pd
from causality_sandbox import DifferenceInDifferences

np.random.seed(42)

# Simulate panel data
n_units = 200
n_periods = 3  # 2 pre, 1 post

units = []
for i in range(n_units):
    # Treatment group: first half
    treated = 1 if i < n_units // 2 else 0
    
    # Unit fixed effect
    unit_fe = np.random.normal(0, 5)
    
    for t in range(n_periods):
        post = 1 if t == n_periods - 1 else 0
        
        # Outcome with parallel trends + treatment effect
        outcome = (
            10 + unit_fe + 2 * t +  # Time trend
            3 * treated +           # Treatment group baseline diff
            4 * treated * post +    # True treatment effect = 4
            np.random.normal(0, 2)  # Noise
        )
        
        units.append({
            'unit_id': i,
            'period': t,
            'treated': treated,
            'post': post,
            'outcome': outcome
        })

df = pd.DataFrame(units)

print("=" * 60)
print("Difference-in-Differences Example")
print("Simulated Policy Evaluation")
print("=" * 60)
print(f"\nUnits: {n_units}")
print(f"Periods: {n_periods} (2 pre-treatment, 1 post)")
print(f"True treatment effect: 4.0")

# Fit DiD
print("\n--- Estimation ---")
did = DifferenceInDifferences()
did.fit(df, outcome='outcome', treatment_col='treated', time_col='post', unit_col='unit_id')

# Results
print("\n")
did.summary()

# Placebo test using period 1 as fake treatment
print("\n--- Placebo Test ---")
placebo = did.placebo_test(df, outcome='outcome', treatment_col='treated', time_col='post', placebo_period=1)
print(f"Placebo ATT: {placebo['placebo_att']:.4f}")
print(f"Placebo p-value: {placebo['p_value']:.4f}")
print(f"Placebo significant (bad): {placebo['significant']}")
