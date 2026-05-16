# causality-sandbox

A lightweight Python toolkit for causal inference from observational data.

This package provides accessible implementations of five widely used quasi-experimental methods. It is designed for researchers, students, and data scientists who want to estimate causal effects from data where randomised experiments are not possible.

The goal is not to automate causal inference, but to provide well-documented, assumption-transparent tools that help you apply these methods correctly and understand when to trust the results.

## Features

- **Propensity Score Matching (PSM)** - Match treated and control units based on estimated treatment probabilities
- **Difference-in-Differences (DiD)** - Compare trends across treatment and control groups in panel data
- **Regression Discontinuity Design (RDD)** - Estimate local treatment effects at a cutoff threshold
- **Instrumental Variables (2SLS)** - Isolate exogenous variation in treatment via an instrument
- **Synthetic Control Method** - Construct a weighted counterfactual from donor units

Each method includes built-in diagnostics to help you check assumptions before trusting the estimates.

## Installation

The package is not yet available on PyPI. Install from source:

```bash
git clone https://github.com/hafiz122/causality-sandbox.git
cd causality-sandbox
pip install -e .
```

Or install directly from GitHub:

```bash
pip install git+https://github.com/hafiz122/causality-sandbox.git
```

### Dependencies

```
numpy >= 1.20.0
scipy >= 1.7.0
pandas >= 1.3.0
scikit-learn >= 1.0.0
statsmodels >= 0.13.0
matplotlib >= 3.4.0
```

These are installed automatically when you run `pip install -e .`

## Quick Start

Here is a complete, runnable example using Propensity Score Matching:

```python
import numpy as np
from causality_sandbox import PropensityScoreMatching

# Simulate sample data
np.random.seed(42)
n = 1000

# Confounders
age = np.random.normal(35, 10, n)
education = np.random.normal(12, 3, n)

# Treatment depends on confounders (selection bias)
logit = -2 + 0.05 * (age - 35) + 0.3 * (education - 12)
prob_treat = 1 / (1 + np.exp(-logit))
treatment = (np.random.uniform(0, 1, n) < prob_treat).astype(int)

# Outcome: true treatment effect = 2500
outcome = (
    25000
    + 500 * (age - 35)
    + 1500 * (education - 12)
    + 2500 * treatment  # true causal effect
    + np.random.normal(0, 5000, n)
)

# Standardise covariates
X = np.column_stack([
    (age - age.mean()) / age.std(),
    (education - education.mean()) / education.std()
])

# Fit PSM
psm = PropensityScoreMatching(caliper=0.1, random_state=42)
psm.fit(X, treatment, outcome)

# Results
psm.summary()
```

**Expected output:**

```
============================================================
Propensity Score Matching Results
============================================================
Number of matched pairs:    312
Matching method:            1:1 nearest neighbor
With replacement:           False
Caliper:                    0.1
------------------------------------------------------------
ATT Estimate:               2481.37
Std. Error:                 412.56
95% CI:                     [1669.43, 3293.31]
t-statistic:                6.02
p-value:                    0.0000
============================================================
```

Run the included examples to see all five methods in action:

```bash
python examples/example_psm.py
python examples/example_did.py
python examples/example_rdd.py
python examples/example_iv.py
python examples/example_synthetic_control.py
```

## Methods

### Propensity Score Matching (PSM)

**When to use:** Treatment was not randomised, but you measured the confounders that affect both treatment assignment and outcomes.

**What it does:** Estimates the probability (propensity score) of each unit receiving treatment given its covariates, then matches treated units to controls with similar scores. The difference in outcomes between matched pairs estimates the Average Treatment Effect on the Treated (ATT).

**Built-in checks:**
- Covariate balance table (standardised mean differences)
- Common support / overlap diagnostics

**Key assumption:** Strong ignorability -- no unmeasured confounders.

### Difference-in-Differences (DiD)

**When to use:** You have panel data (repeated observations on the same units) and a treatment group plus a control group.

**What it does:** Compares the change in outcomes over time between the treatment and control groups. The treatment effect is the difference in these changes.

**Built-in checks:**
- Placebo test to assess parallel trends
- Support for unit fixed effects

**Key assumption:** Parallel trends -- in the absence of treatment, both groups would have followed the same trajectory.

### Regression Discontinuity Design (RDD)

**When to use:** Treatment is assigned based on a sharp cutoff in a continuous variable (e.g., scholarship if exam score >= 70).

**What it does:** Compares units just above and just below the cutoff to estimate the Local Average Treatment Effect (LATE) at the threshold.

**Built-in checks:**
- McCrary density test for manipulation at the cutoff
- Automatic bandwidth selection
- Support for sharp and fuzzy RDD

**Key assumption:** Units cannot precisely manipulate their position relative to the cutoff.

### Instrumental Variables (2SLS)

**When to use:** Treatment is endogenous (people self-selected), but you have a variable that affects treatment without directly affecting the outcome.

**What it does:** Uses two-stage least squares to isolate the exogenous variation in treatment and estimate the Local Average Treatment Effect (LATE).

**Built-in checks:**
- First-stage F-statistic for instrument strength
- Durbin-Wu-Hausman test for endogeneity

**Key assumptions:** Relevance (instrument affects treatment), exclusion restriction (instrument affects outcome only through treatment), monotonicity.

### Synthetic Control Method

**When to use:** One unit (e.g., one state) received treatment, and you have many untreated units to serve as potential controls.

**What it does:** Creates a weighted combination of control units that matches the treated unit's pre-treatment outcome history. Post-treatment divergence estimates the treatment effect.

**Built-in checks:**
- Pre-treatment RMSPE (fit quality)
- Placebo tests for statistical inference

**Key assumption:** The donor pool can approximate the treated unit's counterfactual trajectory.

## Project Structure

```
causality-sandbox/
├── causality_sandbox/
│   ├── __init__.py
│   ├── methods/
│   │   ├── __init__.py
│   │   ├── propensity_score.py          # PSM with matching + balance checks
│   │   ├── difference_in_differences.py # DiD with placebo tests
│   │   ├── regression_discontinuity.py  # Sharp + fuzzy RDD
│   │   ├── instrumental_variable.py     # 2SLS with diagnostics
│   │   └── synthetic_control.py         # SC with placebo inference
│   └── utils/
│       ├── __init__.py
│       ├── balance.py                   # Covariate balance diagnostics
│       ├── overlap.py                   # Common support checks
│       └── validation.py                # Cross-validation utilities
├── tests/
│   ├── __init__.py
│   ├── test_psm.py
│   ├── test_did.py
│   ├── test_rdd.py
│   ├── test_iv.py
│   └── test_synthetic_control.py
├── examples/
│   ├── example_psm.py                   # Job training effect on earnings
│   ├── example_did.py                   # Policy evaluation with panel data
│   ├── example_rdd.py                   # Scholarship effect on GPA
│   ├── example_iv.py                    # Returns to education
│   └── example_synthetic_control.py     # Simulated policy intervention
├── pyproject.toml
├── LICENSE
└── README.md
```

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

24 tests covering all five methods. All tests should pass.

## Roadmap

- [ ] Propensity score weighting (IPW, AIPW)
- [ ] Coarsened Exact Matching (CEM)
- [ ] Event study / dynamic DiD
- [ ] Sensitivity analysis for unmeasured confounding
- [ ] Interactive HTML reporting
- [ ] PyPI publication

## Disclaimer

Causal inference is hard. No software library can prove causality automatically.

Each method in this toolkit relies on specific identifying assumptions that must be justified by your study design, domain knowledge, and careful diagnostic checks. The tools provided here help you implement these methods correctly and test some of these assumptions, but they cannot replace sound research design.

Always report the assumptions your analysis relies on and discuss why they may or may not hold in your specific context.

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*Built with care from Malaysia*
