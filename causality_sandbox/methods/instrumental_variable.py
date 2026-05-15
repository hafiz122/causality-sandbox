"""
Instrumental Variables (IV) for causal inference.

Angrist & Pischke (2009): When treatment is endogenous, a valid instrument
that affects treatment but not the outcome directly can identify causal effects.
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


class InstrumentalVariable:
    """
    Two-Stage Least Squares (2SLS) instrumental variable estimation.
    
    Estimates the local average treatment effect (LATE) using an instrumental
    variable that is correlated with treatment but affects the outcome only
    through its effect on treatment (exclusion restriction).
    
    The 2SLS estimator:
        Stage 1: D = pi_0 + pi_1 * Z + controls + error
        Stage 2: Y = beta_0 + beta_1 * D_hat + controls + error
    
    Parameters
    ----------
    None
    
    Attributes
    ----------
    late_ : float
        Estimated Local Average Treatment Effect (beta_1 from 2SLS).
    std_error_ : float
        Robust standard error.
    ci_ : tuple
        95% confidence interval.
    first_stage_f_ : float
        F-statistic from first stage (rule of thumb: F > 10 for strong instrument).
    
    References
    ----------
    Angrist, J.D. & Pischke, J.S. (2009). Mostly Harmless Econometrics:
    An Empiricist's Companion. Princeton University Press.
    """
    
    def __init__(self):
        self.late_ = None
        self.std_error_ = None
        self.ci_ = None
        self.first_stage_f_ = None
        self._first_stage = None
        self._second_stage = None
    
    def fit(self, endogenous, instrument, outcome, controls=None):
        """
        Fit 2SLS instrumental variable model.
        
        Parameters
        ----------
        endogenous : array-like
            Endogenous treatment variable D.
        instrument : array-like
            Instrumental variable Z.
        outcome : array-like
            Outcome variable Y.
        controls : array-like or None
            Additional exogenous control variables.
        
        Returns
        -------
        self : InstrumentalVariable
        """
        D = np.asarray(endogenous).flatten()
        Z = np.asarray(instrument).flatten()
        Y = np.asarray(outcome).flatten()
        
        n = len(D)
        
        # Stage 1: Regress D on Z and controls
        X1 = pd.DataFrame({'const': 1, 'Z': Z})
        if controls is not None:
            controls = np.asarray(controls)
            if controls.ndim == 1:
                controls = controls.reshape(-1, 1)
            for i in range(controls.shape[1]):
                X1[f'control_{i}'] = controls[:, i]
        
        self._first_stage = sm.OLS(D, X1).fit()
        D_hat = self._first_stage.fittedvalues
        
        # First-stage F-statistic for instrument strength
        r_squared = self._first_stage.rsquared
        k = 1  # number of instruments
        df_resid = self._first_stage.df_resid
        self.first_stage_f_ = (r_squared / k) / ((1 - r_squared) / df_resid)
        
        # Stage 2: Regress Y on D_hat and controls
        X2 = pd.DataFrame({'const': 1, 'D_hat': D_hat})
        if controls is not None:
            for i in range(controls.shape[1]):
                X2[f'control_{i}'] = controls[:, i]
        
        self._second_stage = sm.OLS(Y, X2).fit(cov_type='HC1')
        
        self.late_ = self._second_stage.params['D_hat']
        self.std_error_ = self._second_stage.bse['D_hat']
        
        margin = stats.norm.ppf(0.975) * self.std_error_
        self.ci_ = (self.late_ - margin, self.late_ + margin)
        
        return self
    
    def summary(self):
        """Print summary of IV results."""
        if self.late_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        print("=" * 60)
        print("Instrumental Variable (2SLS) Results")
        print("=" * 60)
        print("--- First Stage ---")
        print(f"Instrument F-statistic:     {self.first_stage_f_:.2f}")
        print(f"  (F > 10 suggests strong instrument)")
        if self._first_stage is not None:
            print(f"First stage R-squared:      {self._first_stage.rsquared:.4f}")
        print("-" * 60)
        print("--- Second Stage ---")
        print(f"LATE Estimate:              {self.late_:.4f}")
        print(f"Std. Error:                 {self.std_error_:.4f}")
        print(f"95% CI:                     [{self.ci_[0]:.4f}, {self.ci_[1]:.4f}]")
        
        t_stat = self.late_ / self.std_error_
        p_value = 2 * (1 - stats.norm.cdf(np.abs(t_stat)))
        print(f"t-statistic:                {t_stat:.4f}")
        print(f"p-value:                    {p_value:.4f}")
        print("=" * 60)
    
    def durbin_wu_hausman(self, endogenous, instrument, outcome, controls=None):
        """
        Durbin-Wu-Hausman test for endogeneity.
        
        Tests whether OLS and 2SLS estimates differ significantly,
        indicating endogeneity in the treatment variable.
        
        Returns
        -------
        dict with test statistic and p-value.
        """
        D = np.asarray(endogenous).flatten()
        Y = np.asarray(outcome).flatten()
        
        if controls is not None:
            controls = np.asarray(controls)
            if controls.ndim == 1:
                controls = controls.reshape(-1, 1)
            X_ols = pd.DataFrame({'const': 1, 'D': D})
            for i in range(controls.shape[1]):
                X_ols[f'control_{i}'] = controls[:, i]
        else:
            X_ols = pd.DataFrame({'const': 1, 'D': D})
        
        # OLS estimate
        ols_model = sm.OLS(Y, X_ols).fit()
        beta_ols = ols_model.params['D']
        
        # 2SLS estimate
        self.fit(D, instrument, Y, controls)
        beta_2sls = self.late_
        
        # Simple Hausman statistic
        diff = beta_2sls - beta_ols
        var_diff = self.std_error_**2 + ols_model.bse['D']**2
        
        h_stat = diff**2 / var_diff
        p_value = 1 - stats.chi2.cdf(h_stat, df=1)
        
        return {
            'hausman_statistic': h_stat,
            'p_value': p_value,
            'endogenous': p_value < 0.05,
            'ols_estimate': beta_ols,
            '2sls_estimate': beta_2sls
        }
