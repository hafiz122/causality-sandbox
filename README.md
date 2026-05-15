# Causality Sandbox

A lightweight Python toolkit for causal inference from observational data. Built for researchers, data scientists, and anyone who wants to move beyond correlation.

Most people doing "data science" are still running regressions and calling it causal. This library gives you the real tools, propensity score matching, difference-in-differences, regression discontinuity, instrumental variables, and synthetic controls. All with clean APIs, proper statistical inference, and built-in diagnostics.

## Installation

```bash
pip install causality-sandbox
```

Or from source:

```bash
git clone https://github.com/hafiz122/causality-sandbox.git
cd causality-sandbox
pip install -e .
```

## Quick Start

```python
import numpy as np
from causality_sandbox import PropensityScoreMatching

# Your observational data
X = covariates          # shape: (n_samples, n_features)
D = treatment           # shape: (n_samples,)
Y = outcome             # shape: (n_samples,)

# Estimate the causal effect
psm = PropensityScoreMatching(random_state=42)
psm.fit(X, D, Y)

# See results
psm.summary()
# ATT Estimate: 2.8471
# 95% CI:       [1.9234, 3.7708]
# p-value:      0.0000
```

## What Problem This Solves

You ran an experiment. Great, use a t-test. But most of the time in the real world, you cannot randomize. You need to estimate what would have happened if the treatment had not occurred. That is the fundamental problem of causal inference.

Causality Sandbox gives you five proven strategies for answering this question, each with the right assumptions and diagnostics so you know when to trust the result.

## The Five Methods

### 1. Propensity Score Matching (PSM)

**When to use:** You have confounders that affect both treatment assignment and outcomes, but no unmeasured confounders (strong ignorability).

**The idea:** Match treated units with control units that have similar propensity scores. The propensity score is the probability of receiving treatment given observed covariates. Under strong ignorability, conditioning on the propensity score is sufficient for unbiased treatment effect estimation.

**Math:** The propensity score is estimated via logistic regression:

```
e(X) = P(D = 1 | X) = 1 / (1 + exp(-(beta_0 + beta'X)))
```

For nearest-neighbor matching, each treated unit i is matched to the control unit j that minimizes |e(X_i) - e(X_j)|. The ATT is then:

```
ATT = (1 / N_t) * sum_{i in treated} (Y_i - Y_{j(i)})
```

where j(i) is the matched control for treated unit i.

**Assumptions to check:** Common support (overlap in propensity scores), covariate balance (standardized mean differences < 0.1 after matching).

```python
from causality_sandbox import PropensityScoreMatching
from causality_sandbox.utils import check_overlap

psm = PropensityScoreMatching(caliper=0.1, random_state=42)
psm.fit(X, treatment, outcome)
psm.summary()

# Check balance
balance = psm.balance_table(X, treatment, matched_only=True)
```

### 2. Difference-in-Differences (DiD)

**When to use:** You have panel data (repeated observations on the same units over time), with a treatment group and a control group. The key assumption is parallel trends: in the absence of treatment, the treatment and control groups would have followed the same trend.

**The idea:** Compare the change in outcomes over time between the treated group and the control group. The treatment effect is the difference in these changes.

**Math:** The canonical DiD estimator:

```
tau = (Y_bar_{treat,post} - Y_bar_{treat,pre}) - (Y_bar_{control,post} - Y_bar_{control,pre})
```

Or equivalently, via regression:

```
Y_it = alpha + beta * Treat_i + gamma * Post_t + tau * (Treat_i * Post_t) + epsilon_it
```

The coefficient tau on the interaction term is the DiD estimate of the ATT.

**Assumptions to check:** Parallel trends (pre-treatment trends should be similar), no anticipation effects.

```python
from causality_sandbox import DifferenceInDifferences

did = DifferenceInDifferences()
did.fit(data, outcome='Y', treatment_col='treat', time_col='post', unit_col='unit_id')
did.summary()

# Placebo test
did.placebo_test(data, outcome='Y', treatment_col='treat', time_col='post', placebo_period=1)
```

### 3. Regression Discontinuity Design (RDD)

**When to use:** Treatment is assigned based on a sharp cutoff in a continuous running variable. Units just above and below the cutoff are assumed to be comparable.

**The idea:** Compare outcomes for units just above and just below the cutoff. The discontinuity in the outcome at the threshold estimates the local average treatment effect (LATE).

**Math:** For sharp RDD with a local linear regression:

```
Y = alpha + tau * D + beta * (R - c) + gamma * (R - c) * D + epsilon
```

where D = 1(R >= c) is the treatment indicator, R is the running variable, and c is the cutoff. The coefficient tau is the LATE at the cutoff.

The bandwidth h is selected to minimize the mean squared error:

```
h_opt = C * n^(-1/5)
```

**Assumptions to check:** No manipulation at the cutoff (McCrary density test), continuity of covariates at the cutoff.

```python
from causality_sandbox import RegressionDiscontinuity

rdd = RegressionDiscontinuity(cutoff=70, kernel='triangular', polynomial=1)
rdd.fit(running_variable, outcome)
rdd.summary()

# Check for manipulation
rdd.mccrary_test(running_variable)
```

### 4. Instrumental Variables (IV / 2SLS)

**When to use:** Treatment is endogenous (correlated with the error term), but you have an instrument that affects treatment but affects the outcome only through its effect on treatment.

**The idea:** Use the instrument to isolate the exogenous variation in treatment, then estimate the effect of this exogenous variation on the outcome.

**Math:** Two-stage least squares:

Stage 1 (first stage):
```
D = pi_0 + pi_1 * Z + controls + v
```

Stage 2:
```
Y = beta_0 + beta_1 * D_hat + controls + epsilon
```

The LATE is beta_1. The first-stage F-statistic tests instrument strength (F > 10 is the rule of thumb).

**Assumptions to check:** Relevance (F > 10), exclusion restriction (instrument affects Y only through D), monotonicity.

```python
from causality_sandbox import InstrumentalVariable

iv = InstrumentalVariable()
iv.fit(endogenous=education, instrument=distance, outcome=earnings)
iv.summary()

# Hausman test for endogeneity
iv.durbin_wu_hausman(education, distance, earnings)
```

### 5. Synthetic Control Method

**When to use:** You have one treated unit and multiple control units observed over time. The treated unit is exposed to an intervention at a known date. You want to construct a counterfactual for what would have happened without the intervention.

**The idea:** Create a weighted combination of control units that closely matches the treated unit's pre-intervention outcome trajectory. The post-intervention divergence between the treated unit and its synthetic counterpart estimates the treatment effect.

**Math:** Find weights w that minimize pre-treatment mean squared error:

```
w* = argmin_w ||Y_{1,pre} - Y_{0,pre} * w||^2

subject to: sum(w_j) = 1, w_j >= 0
```

The treatment effect at time t is:
```
tau_t = Y_{1t} - sum_j w*_j * Y_{0jt}
```

Statistical inference is done via placebo tests: apply the same procedure to each control unit and compare the treated unit's effect to this distribution.

```python
from causality_sandbox import SyntheticControl

sc = SyntheticControl()
sc.fit(pre_treated, pre_controls, post_treated, post_controls,
       control_names=['State_A', 'State_B', ...])
sc.summary()

# Placebo test for inference
sc.placebo_test(all_pre, all_post)
```

## Example Gallery

| Example | Method | Description |
|---------|--------|-------------|
| `example_psm.py` | PSM | Job training program effect on earnings |
| `example_did.py` | DiD | Policy evaluation with panel data |
| `example_rdd.py` | RDD | Scholarship effect on GPA |
| `example_iv.py` | 2SLS | Returns to education |
| `example_synthetic_control.py` | SC | Simulated policy intervention |

Run any example:
```bash
cd examples
python example_psm.py
```

## Project Structure

```
causality_sandbox/
├── causality_sandbox/
│   ├── __init__.py
│   ├── methods/
│   │   ├── __init__.py
│   │   ├── propensity_score.py         # PSM with matching + balance checks
│   │   ├── difference_in_differences.py # DiD with placebo tests
│   │   ├── regression_discontinuity.py  # Sharp + fuzzy RDD, McCrary test
│   │   ├── instrumental_variable.py     # 2SLS with Hausman test
│   │   └── synthetic_control.py         # SC with placebo inference
│   └── utils/
│       ├── __init__.py
│       ├── balance.py                   # Covariate balance diagnostics
│       ├── overlap.py                   # Common support checks
│       └── validation.py                # Cross-validation utilities
├── tests/                               # pytest test suite
├── examples/                            # Runnable examples
├── README.md
├── pyproject.toml
└── LICENSE
```

## Running Tests

```bash
pytest tests/ -v
```

## Dependencies

- numpy >= 1.20.0
- scipy >= 1.7.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- statsmodels >= 0.13.0
- matplotlib >= 3.4.0 (for examples)

## Why This Exists

I built this because I studied statistics and got tired of seeing people run regressions and call it causal inference. The field deserves tools that are rigorous but accessible. This library is opinionated in the right ways: it forces you to check assumptions, provides proper standard errors, and makes diagnostics front and center rather than an afterthought.

If you are doing observational research and want to make causal claims, this toolkit gives you the foundation to do it right.

## Citation

If you use this library in your research:

```bibtex
@software{causality_sandbox,
  author = {Ismail, Muhammad Hafiz bin},
  title = {Causality Sandbox: A lightweight causal inference toolkit},
  url = {https://github.com/hafiz122/causality-sandbox},
  year = {2025}
}
```

## Key References

- Rosenbaum, P.R. & Rubin, D.B. (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika*, 70(1), 41-55.
- Card, D. & Krueger, A.B. (1994). Minimum wages and employment. *American Economic Review*, 84(4), 772-793.
- Thistlethwaite, D.L. & Campbell, D.T. (1960). Regression-discontinuity analysis. *Journal of Educational Psychology*, 51(6), 309-317.
- Angrist, J.D. & Pischke, J.S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.
- Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control methods for comparative case studies. *JASA*, 105(490), 493-505.

## License

MIT License. See LICENSE for details.

Built with care from Malaysia.
