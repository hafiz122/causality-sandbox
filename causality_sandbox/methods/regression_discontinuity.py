"""
Regression Discontinuity Design (RDD) for causal inference.

Thistlethwaite & Campbell (1960): Units just above and below a cutoff threshold
are comparable, allowing causal identification of treatment effects at the cutoff.
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


class RegressionDiscontinuity:
    """
    Sharp and Fuzzy Regression Discontinuity Design.
    
    Estimates the local average treatment effect (LATE) at a cutoff point
    where treatment assignment changes discontinuously.
    
    For sharp RDD: treatment is a deterministic function of the running variable.
    For fuzzy RDD: treatment probability jumps at the cutoff but is not deterministic.
    
    Parameters
    ----------
    cutoff : float
        The threshold value where treatment assignment changes.
    bandwidth : float or str, default 'optimal'
        Window around cutoff to include. 'optimal' uses Imbens-Kalyanaraman.
    kernel : str, default 'triangular'
        Weighting kernel. 'triangular', 'uniform', or 'epanechnikov'.
    polynomial : int, default 1
        Degree of polynomial for local regression.
    
    Attributes
    ----------
    late_ : float
        Estimated Local Average Treatment Effect at the cutoff.
    std_error_ : float
        Robust standard error.
    ci_ : tuple
        95% confidence interval.
    bandwidth_ : float
        Bandwidth used for estimation.
    
    References
    ----------
    Thistlethwaite, D.L. & Campbell, D.T. (1960). Regression-discontinuity
    analysis: An alternative to the ex post facto experiment. Journal of
    Educational Psychology, 51(6), 309-317.
    """
    
    def __init__(self, cutoff, bandwidth='optimal', kernel='triangular', polynomial=1):
        self.cutoff = cutoff
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.polynomial = polynomial
        
        self.late_ = None
        self.std_error_ = None
        self.ci_ = None
        self.bandwidth_ = None
        self._results = None
    
    def fit(self, running_variable, outcome, treatment=None):
        """
        Fit the RDD model.
        
        Parameters
        ----------
        running_variable : array-like
            The forcing variable that determines treatment assignment.
        outcome : array-like
            Outcome variable.
        treatment : array-like or None
            Treatment indicator (None for sharp RDD).
        
        Returns
        -------
        self : RegressionDiscontinuity
        """
        r = np.asarray(running_variable).flatten()
        y = np.asarray(outcome).flatten()
        
        # Center running variable at cutoff
        r_centered = r - self.cutoff
        
        # Determine bandwidth
        if self.bandwidth == 'optimal':
            self.bandwidth_ = self._optimal_bandwidth(r_centered, y)
        else:
            self.bandwidth_ = float(self.bandwidth)
        
        # Select observations within bandwidth
        mask = np.abs(r_centered) <= self.bandwidth_
        r_bw = r_centered[mask]
        y_bw = y[mask]
        
        # Create kernel weights
        weights = self._kernel_weights(r_bw / self.bandwidth_)
        
        if treatment is None:
            # Sharp RDD
            self._fit_sharp(r_bw, y_bw, weights)
        else:
            # Fuzzy RDD
            d = np.asarray(treatment).flatten()[mask]
            self._fit_fuzzy(r_bw, y_bw, d, weights)
        
        return self
    
    def _optimal_bandwidth(self, r, y):
        """
        Imbens-Kalyanaraman bandwidth selector.
        Simple rule-of-thumb implementation.
        """
        n = len(r)
        
        # Separate data
        r_pos = r[r > 0]
        r_neg = r[r <= 0]
        y_pos = y[r > 0]
        y_neg = y[r <= 0]
        
        if len(r_pos) < 10 or len(r_neg) < 10:
            # Fallback: use Silverman's rule
            h = 1.06 * np.std(r) * n ** (-1/5)
            return max(h, np.std(r) * 0.5)
        
        # Simple ROT bandwidth
        h_pos = np.std(r_pos)
        h_neg = np.std(r_neg)
        h = min(h_pos, h_neg) * (n ** (-1/5))
        
        return max(h, 0.01)  # Minimum bandwidth
    
    def _kernel_weights(self, u):
        """Compute kernel weights."""
        if self.kernel == 'uniform':
            return np.where(np.abs(u) <= 1, 0.5, 0)
        elif self.kernel == 'triangular':
            return np.where(np.abs(u) <= 1, 1 - np.abs(u), 0)
        elif self.kernel == 'epanechnikov':
            return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _fit_sharp(self, r, y, weights):
        """Fit sharp RDD model."""
        above = (r > 0).astype(float)
        
        # Build polynomial terms
        X = pd.DataFrame({'const': 1, 'above': above})
        for p in range(1, self.polynomial + 1):
            X[f'r{p}'] = r ** p
            X[f'r{p}_above'] = (r ** p) * above
        
        # Weighted least squares
        model = sm.WLS(y, X, weights=weights)
        self._results = model.fit()
        
        self.late_ = self._results.params['above']
        self.std_error_ = self._results.bse['above']
        
        margin = stats.norm.ppf(0.975) * self.std_error_
        self.ci_ = (self.late_ - margin, self.late_ + margin)
    
    def _fit_fuzzy(self, r, y, d, weights):
        """Fit fuzzy RDD using 2SLS."""
        above = (r > 0).astype(float)
        
        # Build polynomial terms
        X_endog = pd.DataFrame({'const': 1, 'treatment': d})
        X_instrument = pd.DataFrame({'const': 1, 'above': above})
        
        for p in range(1, self.polynomial + 1):
            X_endog[f'r{p}'] = r ** p
            X_endog[f'r{p}_treat'] = (r ** p) * d
            X_instrument[f'r{p}'] = r ** p
            X_instrument[f'r{p}_above'] = (r ** p) * above
        
        # 2SLS
        model = sm.IV2SLS(y, X_endog, X_instrument)
        self._results = model.fit()
        
        self.late_ = self._results.params['treatment']
        self.std_error_ = self._results.bse['treatment']
        
        margin = stats.norm.ppf(0.975) * self.std_error_
        self.ci_ = (self.late_ - margin, self.late_ + margin)
    
    def mccrary_test(self, running_variable, bin_width=None):
        """
        McCrary (2008) density test for manipulation at cutoff.
        
        Tests whether the density of the running variable is continuous
        at the cutoff. A significant result suggests manipulation.
        
        Parameters
        ----------
        running_variable : array-like
            The running variable.
        bin_width : float or None
            Bin width for histogram. None uses automatic selection.
        
        Returns
        -------
        dict with test statistic and p-value.
        """
        r = np.asarray(running_variable).flatten()
        
        if bin_width is None:
            # Scott's rule
            bin_width = 3.5 * np.std(r) * len(r) ** (-1/3)
        
        # Create histogram
        bins = np.arange(self.cutoff - 5*bin_width, self.cutoff + 5*bin_width, bin_width)
        hist, edges = np.histogram(r, bins=bins)
        
        # Bin centers
        centers = (edges[:-1] + edges[1:]) / 2
        
        # Separate left and right of cutoff
        left_mask = centers < self.cutoff
        right_mask = centers >= self.cutoff
        
        # Simple log-density regression
        log_dens = np.log(hist + 0.5)
        
        # Local linear regression on each side
        r_centered = centers - self.cutoff
        
        X = pd.DataFrame({
            'const': 1,
            'r': r_centered,
            'above': (r_centered >= 0).astype(float)
        })
        
        try:
            model = sm.OLS(log_dens, X).fit()
            theta = model.params['above']
            se = model.bse['above']
            t_stat = theta / se if se > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(np.abs(t_stat)))
        except:
            theta, se, t_stat, p_value = np.nan, np.nan, np.nan, np.nan
        
        return {
            'theta': theta,
            'std_error': se,
            't_statistic': t_stat,
            'p_value': p_value,
            'manipulation_detected': p_value < 0.05
        }
    
    def summary(self):
        """Print summary of RDD results."""
        if self.late_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        rdd_type = "Sharp" if self._results is not None and 'treatment' not in self._results.params.index else "Fuzzy"
        
        print("=" * 60)
        print("Regression Discontinuity Design Results")
        print("=" * 60)
        print(f"RDD Type:                   {rdd_type}")
        print(f"Cutoff:                     {self.cutoff:.4f}")
        print(f"Bandwidth:                  {self.bandwidth_:.4f}")
        print(f"Kernel:                     {self.kernel}")
        print(f"Polynomial:                 {self.polynomial}")
        print("-" * 60)
        print(f"LATE Estimate:              {self.late_:.4f}")
        print(f"Std. Error:                 {self.std_error_:.4f}")
        print(f"95% CI:                     [{self.ci_[0]:.4f}, {self.ci_[1]:.4f}]")
        
        if self._results is not None:
            t_stat = self.late_ / self.std_error_
            p_value = 2 * (1 - stats.norm.cdf(np.abs(t_stat)))
            print(f"t-statistic:                {t_stat:.4f}")
            print(f"p-value:                    {p_value:.4f}")
        
        print("=" * 60)
