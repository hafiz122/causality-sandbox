"""
Example: Synthetic Control Method on simulated panel data.

This example demonstrates the classic setup: one treated unit
and multiple control units observed over time.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from causality_sandbox import SyntheticControl

# Set seed
np.random.seed(42)

# Setup: 20 pre-treatment periods, 10 post-treatment periods
# 1 treated unit, 10 control units
n_pre = 20
n_post = 10
n_controls = 10

# Create time trends
time = np.arange(n_pre + n_post)

# Treated unit: follows a trend then gets treatment effect
trend = 50 + 2 * time + 0.1 * time**2
treatment_effect = np.concatenate([
    np.zeros(n_pre),
    15 + 0.5 * np.arange(n_post)  # Growing treatment effect
])
treated_outcome = trend + treatment_effect + np.random.normal(0, 3, n_pre + n_post)

# Control units: similar trends with different weights
control_weights = np.random.dirichlet(np.ones(n_controls))
control_outcomes = np.zeros((n_pre + n_post, n_controls))
for i in range(n_controls):
    noise = np.random.normal(0, 4, n_pre + n_post)
    # Each control has slightly different trend
    control_trend = 50 + (2 + np.random.normal(0, 0.3)) * time + (0.1 + np.random.normal(0, 0.01)) * time**2
    control_outcomes[:, i] = control_trend + noise

# Split into pre/post
pre_treated = treated_outcome[:n_pre]
pre_controls = control_outcomes[:n_pre, :]
post_treated = treated_outcome[n_pre:]
post_controls = control_outcomes[n_pre:, :]

control_names = [f'State_{i+1}' for i in range(n_controls)]

print("=" * 60)
print("Synthetic Control Method Example")
print("Simulated Policy Intervention")
print("=" * 60)
print(f"\nPre-treatment periods:  {n_pre}")
print(f"Post-treatment periods: {n_post}")
print(f"Control units:          {n_controls}")
print(f"True avg treatment effect: {np.mean(treatment_effect[n_pre:]):.2f}")

# Fit synthetic control
sc = SyntheticControl(optimization_method='wls')
sc.fit(pre_treated, pre_controls, post_treated, post_controls, control_names=control_names)

# Summary
sc.summary()

# Placebo test
print("\n--- Placebo Test ---")
all_pre = np.column_stack([pre_treated.reshape(-1, 1), pre_controls])
all_post = np.column_stack([post_treated.reshape(-1, 1), post_controls])
placebo = sc.placebo_test(all_pre, all_post)
if placebo['p_value'] is not None:
    print(f"Placebo p-value: {placebo['p_value']:.4f}")
    print(f"Treated unit ratio: {placebo['treated_ratio']:.4f}")
else:
    print("Placebo test could not be computed.")

# Plot
print("\n--- Generating Plot ---")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Outcomes over time
ax1.plot(time[:n_pre], pre_treated, 'b-', linewidth=2, label='Treated', zorder=5)
ax1.plot(time[n_pre-1:], treated_outcome[n_pre-1:], 'b--', linewidth=2, zorder=5)

synthetic_pre = pre_controls @ sc.weights_
synthetic_post = post_controls @ sc.weights_
synthetic_full = np.concatenate([synthetic_pre, synthetic_post])
ax1.plot(time[:n_pre], synthetic_pre, 'r-', linewidth=2, label='Synthetic', zorder=4)
ax1.plot(time[n_pre-1:], synthetic_full[n_pre-1:], 'r--', linewidth=2, zorder=4)

for i in range(n_controls):
    ax1.plot(time, control_outcomes[:, i], 'gray', alpha=0.3, linewidth=0.5)

ax1.axvline(x=n_pre-1, color='black', linestyle=':', alpha=0.7, label='Intervention')
ax1.set_xlabel('Time Period')
ax1.set_ylabel('Outcome')
ax1.set_title('A. Treated vs. Synthetic Control')
ax1.legend(loc='upper left')

# Panel B: Treatment effect
ax2.bar(range(1, n_post + 1), sc.treatment_effect_, color='steelblue', alpha=0.8)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax2.set_xlabel('Post-Treatment Period')
ax2.set_ylabel('Treatment Effect')
ax2.set_title(f'B. Estimated Treatment Effect (RMSPE: {sc.rmspe_:.2f})')

plt.tight_layout()
plt.savefig('synthetic_control_example.png', dpi=150, bbox_inches='tight')
print("Plot saved: synthetic_control_example.png")
