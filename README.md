# Causality Sandbox

A lightweight Python toolkit for causal inference from observational data. Built for researchers, data scientists, and anyone who wants to move beyond correlation.

Most people doing "data science" are still running regressions and calling it causal. This library gives you the real tools: Propensity Score Matching, Difference-in-Differences, Regression Discontinuity, Instrumental Variables, and Synthetic Control. All with clean APIs, proper statistical inference, and built-in diagnostics.

Every formula in this README is rendered with proper LaTeX math, because when we are doing statistics, notation matters.

---

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

---

## Quick Start

```python
from causality_sandbox import PropensityScoreMatching

# Your observational data
X = covariates          # shape: (n_people, n_features)
D = treatment           # 1 = got the treatment, 0 = did not
Y = outcome             # the result you care about

# Match and estimate
psm = PropensityScoreMatching(caliper=0.1, random_state=42)
psm.fit(X, D, Y)

# See results
psm.summary()
# ATT Estimate: 2.8471
# 95% CI:       [1.9234, 3.7708]
# p-value:      0.0000
```

---

## The Five Methods

### 1. Propensity Score Matching (PSM)

**Use when:** You have confounders that affect both treatment assignment and outcomes, but no unmeasured confounders (strong ignorability).

**The idea:** Match treated units with control units that have similar propensity scores. Under strong ignorability, conditioning on the propensity score is sufficient for unbiased treatment effect estimation.

**The propensity score** is estimated via logistic regression:

$$
e(X_i) = \mathbb{P}(D_i = 1 \mid X_i) = \frac{1}{1 + \exp\{- (\beta_0 + \beta^{\top} X_i)\}}
$$

**Nearest-neighbor matching:** Each treated unit $i$ is matched to the control unit $j$ that minimizes the absolute propensity score distance:

$$
j(i) = \arg\min_{j \in \mathcal{C}} \, |e(X_i) - e(X_j)|
$$

**The ATT estimator** is the mean difference between matched pairs:

$$
\widehat{\text{ATT}} = \frac{1}{N_T} \sum_{i \in \mathcal{T}} \bigl( Y_i - Y_{j(i)} \bigr)
$$

where $\mathcal{T}$ is the set of treated units, $\mathcal{C}$ is the set of control units, and $N_T = |\mathcal{T}|$.

**Standard error** (Abadie & Imbers, 2006):

$$
\widehat{\text{SE}} = \sqrt{\frac{1}{N_T^2} \sum_{i \in \mathcal{T}} \bigl(Y_i - Y_{j(i)} - \widehat{\text{ATT}}\bigr)^2}
$$

**Assumptions to check:** Common support (overlap in propensity score distributions), covariate balance (standardized mean differences $< 0.1$ after matching).

```python
from causality_sandbox import PropensityScoreMatching
from causality_sandbox.utils import check_overlap

psm = PropensityScoreMatching(caliper=0.1, random_state=42)
psm.fit(X, treatment, outcome)
psm.summary()

# Check balance
balance = psm.balance_table(X, treatment, matched_only=True)
```

---

### 2. Difference-in-Differences (DiD)

**Use when:** You have panel data (repeated observations on the same units over time), with a treatment group and a control group. The key assumption is parallel trends: in the absence of treatment, both groups would have followed the same trajectory.

**The canonical DiD estimator** compares the change in outcomes over time between the two groups:

$$
\widehat{\tau} = \bigl(\bar{Y}_{\text{treat, post}} - \bar{Y}_{\text{treat, pre}}\bigr) - \bigl(\bar{Y}_{\text{control, post}} - \bar{Y}_{\text{control, pre}}\bigr)
$$

**Equivalently**, via regression with the interaction term:

$$
Y_{it} = \alpha + \beta \cdot \text{Treat}_i + \gamma \cdot \text{Post}_t + \tau \cdot (\text{Treat}_i \times \text{Post}_t) + \varepsilon_{it}
$$

The coefficient $\tau$ on the interaction term is the DiD estimate of the ATT. The parallel trends assumption requires:

$$
\mathbb{E}\bigl[Y_{it}(0) - Y_{i,t-1}(0) \mid \text{Treat}_i = 1\bigr] = \mathbb{E}\bigl[Y_{it}(0) - Y_{i,t-1}(0) \mid \text{Treat}_i = 0\bigr]
$$

for all pre-treatment periods $t$.

**Assumptions to check:** Parallel trends (pre-treatment trends should be parallel), no anticipation effects (units do not change behavior before treatment).

```python
from causality_sandbox import DifferenceInDifferences

did = DifferenceInDifferences()
did.fit(data, outcome='Y', treatment_col='treat', time_col='post', unit_col='unit_id')
did.summary()

# Check parallel trends with placebo test
placebo = did.placebo_test(data, outcome='Y', treatment_col='treat',
                            time_col='post', placebo_period=1)
```

---

### 3. Regression Discontinuity Design (RDD)

**Use when:** Treatment is assigned based on a sharp cutoff in a continuous running variable. Units just above and below the cutoff are assumed to be comparable.

**The local linear regression specification** for sharp RDD:

$$
Y_i = \alpha + \tau \cdot D_i + \beta \cdot (R_i - c) + \gamma \cdot (R_i - c) \cdot D_i + \varepsilon_i
$$

where $D_i = \mathbf{1}(R_i \geq c)$ is the treatment indicator, $R_i$ is the running variable, and $c$ is the cutoff. The coefficient $\tau$ is the **Local Average Treatment Effect (LATE)** at the cutoff.

**Optimal bandwidth** (Imbens-Kalyanaraman):

$$
h_{\text{opt}} = C \cdot N^{-1/5}
$$

**Kernel weighting** for triangular kernel:

$$
K(u) = (1 - |u|) \cdot \mathbf{1}(|u| \leq 1)
$$

where observations outside the bandwidth $h$ receive zero weight.

**Assumptions to check:** No manipulation at the cutoff (McCrary density test), continuity of covariates at the cutoff.

```python
from causality_sandbox import RegressionDiscontinuity

rdd = RegressionDiscontinuity(cutoff=70, kernel='triangular', polynomial=1)
rdd.fit(running_variable=test_score, outcome=gpa)
rdd.summary()

# Check for manipulation
rdd.mccrary_test(running_variable)
```

---

### 4. Instrumental Variables (IV / 2SLS)

**Use when:** Treatment is endogenous (people self-selected into treatment), but you have an instrument that affects treatment assignment without directly affecting the outcome.

**Two-Stage Least Squares:**

**Stage 1** (first stage, regress treatment on instrument):

$$
D_i = \pi_0 + \pi_1 Z_i + X_i^{\top} \delta + v_i
$$

**Stage 2** (regress outcome on fitted treatment):

$$
Y_i = \beta_0 + \beta_1 \widehat{D}_i + X_i^{\top} \gamma + \varepsilon_i
$$

The coefficient $\beta_1$ is the **Local Average Treatment Effect (LATE)**. For a valid instrument, the first-stage F-statistic should satisfy:

$$
F = \frac{R^2 / k}{(1 - R^2) / (N - k - 1)} > 10
$$

**The key assumptions:**
- **Relevance:** $\pi_1 \neq 0$ (the instrument affects treatment)
- **Exclusion restriction:** $\text{Cov}(Z, \varepsilon) = 0$ (the instrument affects $Y$ only through $D$)
- **Monotonicity:** The instrument moves all units in the same direction

```python
from causality_sandbox import InstrumentalVariable

iv = InstrumentalVariable()
iv.fit(endogenous=education, instrument=distance, outcome=earnings)
iv.summary()

# Hausman test for endogeneity
hausman = iv.durbin_wu_hausman(education, distance, earnings)
```

---

### 5. Synthetic Control Method

**Use when:** One unit (e.g., one country, one state) received treatment, and you have many untreated units to serve as potential controls. You want to construct a counterfactual for what the treated unit would have experienced without treatment.

**The synthetic control** is a weighted combination of control units:

$$
\widehat{Y}_{1t}^{\text{synth}} = \sum_{j=2}^{J+1} w_j^* \, Y_{jt}
$$

where the optimal weights minimize the pre-treatment prediction error:

$$
w^* = \arg\min_{w \in \mathcal{W}} \; \sum_{t=1}^{T_0} \bigl( Y_{1t} - \sum_{j=2}^{J+1} w_j \, Y_{jt} \bigr)^2
$$

subject to the constraint set:

$$
\mathcal{W} = \Bigl\{ w \in \mathbb{R}^{J} : \; w_j \geq 0 \;\; \forall j, \;\; \sum_{j=2}^{J+1} w_j = 1 \Bigr\}
$$

**The treatment effect** at each post-treatment period:

$$
\widehat{\tau}_t = Y_{1t} - \widehat{Y}_{1t}^{\text{synth}} = Y_{1t} - \sum_{j=2}^{J+1} w_j^* \, Y_{jt} \qquad \text{for } t = T_0 + 1, \ldots, T
$$

**Statistical inference** uses placebo tests: apply the same procedure to each control unit and compare the treated unit's effect magnitude to the distribution of placebo effects.

```python
from causality_sandbox import SyntheticControl

sc = SyntheticControl()
sc.fit(pre_treated, pre_controls, post_treated, post_controls,
       control_names=['State_A', 'State_B', 'State_C'])
sc.summary()

# Placebo test for statistical significance
all_pre = np.column_stack([pre_treated, pre_controls])
all_post = np.column_stack([post_treated, post_controls])
placebo = sc.placebo_test(all_pre, all_post)
```

---

## Assumptions at a Glance

| Method | Key Assumption | What to Check |
|--------|---------------|---------------|
| PSM | Strong ignorability: $(Y(0), Y(1)) \perp D \mid X$ | Common support, covariate balance (SMD $< 0.1$) |
| DiD | Parallel trends | Pre-treatment trend comparison, placebo test |
| RDD | Continuity at cutoff | McCrary density test, covariate continuity |
| IV | Exclusion restriction, relevance | First-stage $F > 10$, overidentification tests |
| SC | Donor pool approximates treated unit | Pre-treatment RMSPE, placebo distribution |

---

## Example Gallery

| Example | Method | Description |
|---------|--------|-------------|
| `example_psm.py` | PSM | Job training program effect on earnings |
| `example_did.py` | DiD | Policy evaluation with panel data |
| `example_rdd.py` | RDD | Scholarship effect on GPA (cutoff at 70) |
| `example_iv.py` | 2SLS | Returns to education |
| `example_synthetic_control.py` | SC | Simulated policy intervention with plot |

Run any example:
```bash
cd examples
python example_psm.py
```

---

## Running Tests

```bash
pytest tests/ -v
```

All 24 tests should pass.

---

## Project Structure

```
causality_sandbox/
├── causality_sandbox/
│   ├── methods/
│   │   ├── propensity_score.py
│   │   ├── difference_in_differences.py
│   │   ├── regression_discontinuity.py
│   │   ├── instrumental_variable.py
│   │   └── synthetic_control.py
│   └── utils/
│       ├── balance.py
│       ├── overlap.py
│       └── validation.py
├── tests/         # 24 pytest tests
├── examples/      # 5 runnable examples
├── README.md
├── pyproject.toml
└── LICENSE
```

---

## Dependencies

```
numpy >= 1.20.0
scipy >= 1.7.0
pandas >= 1.3.0
scikit-learn >= 1.0.0
statsmodels >= 0.13.0
matplotlib >= 3.4.0
```

---

## Why This Exists

I built this because I studied statistics and got tired of seeing people run regressions and call it causal inference. The field deserves tools that are rigorous but accessible. This library is opinionated in the right ways: it forces you to check assumptions, provides proper standard errors, and makes diagnostics front and center rather than an afterthought.

If you are doing observational research and want to make causal claims, this toolkit gives you the foundation to do it right.

---

## Citation

```bibtex
@software{causality_sandbox,
  author = {Ismail, Muhammad Hafiz bin},
  title = {Causality Sandbox: A lightweight causal inference toolkit},
  url = {https://github.com/hafiz122/causality-sandbox},
  year = {2025}
}
```

---

## Key References

- Rosenbaum, P.R. & Rubin, D.B. (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika*, 70(1), 41-55.

- Card, D. & Krueger, A.B. (1994). Minimum wages and employment. *American Economic Review*, 84(4), 772-793.

- Thistlethwaite, D.L. & Campbell, D.T. (1960). Regression-discontinuity analysis. *Journal of Educational Psychology*, 51(6), 309-317.

- Angrist, J.D. & Pischke, J.S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.

- Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control methods for comparative case studies. *JASA*, 105(490), 493-505.

---

## License

MIT License. See LICENSE for details.

Built with care from Malaysia.
