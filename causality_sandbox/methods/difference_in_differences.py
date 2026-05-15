"""
Difference-in-Differences (DiD) for causal inference.

Card & Krueger (1994) classic: Compare changes in outcomes over time between
a treatment group and a control group, assuming parallel trends.
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import warnings


class DifferenceInDifferences:
    """
    Difference-in-Differences estimator for treatment effects in panel data.
    
    Compares the change in outcomes over time between a group that received
treatment and a group that did not, under the parallel trends assumption.
    
    The basic estimator is:
        tau = (Y_treat_post - Y_treat_pre) - (Y_control_post - Y_control_pre)
    
    Parameters
    ----------
    cluster : str or None, default None
        Column name for clustering standard errors.
    
    Attributes
    ----------
    att_ : float
        Estimated Average Treatment Effect on the Treated (DiD estimate).
    std_error_ : float
        Standard error of the estimate.
    ci_ : tuple
        95% confidence interval.
    p_value_ : float
        Two-sided p-value for the null hypothesis of no effect.
    
    References
    ----------
    Card, D. & Krueger, A.B. (1994). Minimum wages and employment: A case study
    of the fast-food industry in New Jersey and Pennsylvania. American Economic
    Review, 84(4), 772-793.
    """
    
    def __init__(self, cluster=None):
        self.cluster = cluster
        
        self.att_ = None
        self.std_error_ = None
        self.ci_ = None
        self.p_value_ = None
        self._model = None
        self._results = None
    
    def fit(self, data, outcome, treatment_col, time_col, unit_col=None):
        """
        Fit the DiD model.
        
        Parameters
        ----------
        data : pd.DataFrame
            Panel data containing outcome, treatment indicator, and time indicator.
        outcome : str
            Column name for outcome variable.
        treatment_col : str
            Column name for treatment group indicator (1=treatment group, 0=control).
        time_col : str
            Column name for time period indicator (1=post, 0=pre).
        unit_col : str or None
            Column name for unit identifier (for fixed effects).
        
        Returns
        -------
        self : DifferenceInDifferences
        """
        df = data.copy()
        
        # Create DiD interaction term: treated * post
        df['did_interaction'] = df[treatment_col] * df[time_col]
        
        # Build regression formula
        X = pd.DataFrame({
            'const': 1,
            'treated': df[treatment_col],
            'post': df[time_col],
            'did': df['did_interaction']
        })
        
        y = df[outcome]
        
        # Add unit fixed effects if specified
        if unit_col is not None:
            unit_dummies = pd.get_dummies(df[unit_col], prefix='unit', drop_first=True).astype(float)
            X = pd.concat([X, unit_dummies], axis=1)
        
        # Fit OLS
        self._model = sm.OLS(y, X)
        self._results = self._model.fit(
            cov_type='cluster' if self.cluster else 'HC3',
            cov_kwds={'groups': df[self.cluster]} if self.cluster else {}
        )
        
        # Extract DiD coefficient
        self.att_ = self._results.params['did']
        self.std_error_ = self._results.bse['did']
        self.p_value_ = self._results.pvalues['did']
        
        # Confidence interval
        margin = stats.norm.ppf(0.975) * self.std_error_
        self.ci_ = (self.att_ - margin, self.att_ + margin)
        
        # Check parallel trends assumption if possible
        self._check_parallel_trends(df, outcome, treatment_col, time_col)
        
        return self
    
    def _check_parallel_trends(self, df, outcome, treatment, time):
        """Warn if parallel trends assumption may be violated."""
        # Check if we have more than 2 time periods
        n_periods = df[time].nunique()
        if n_periods <= 2:
            return  # Cannot check with only pre/post
        
        # Group means by treatment and time
        grouped = df.groupby([treatment, time])[outcome].mean().unstack(level=0)
        
        if grouped.shape[1] == 2:
            gap = grouped[1] - grouped[0]
            # Check if pre-treatment gaps are roughly constant
            pre_periods = gap.index[gap.index < gap.index.max()]
            if len(pre_periods) > 1:
                pre_gaps = gap[pre_periods]
                variation = np.std(pre_gaps)
                if variation > 0.1 * np.abs(np.mean(pre_gaps)):
                    warnings.warn(
                        "Pre-treatment trends show variation. "
                        "Parallel trends assumption may be violated.",
                        UserWarning
                    )
    
    def summary(self):
        """Print summary of DiD results."""
        if self.att_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        print("=" * 60)
        print("Difference-in-Differences Results")
        print("=" * 60)
        print(f"DiD Estimate (ATT):         {self.att_:.4f}")
        print(f"Std. Error:                 {self.std_error_:.4f}")
        print(f"95% CI:                     [{self.ci_[0]:.4f}, {self.ci_[1]:.4f}]")
        print(f"p-value:                    {self.p_value_:.4f}")
        print(f"Significant at 5%:          {'Yes' if self.p_value_ < 0.05 else 'No'}")
        
        if self._results is not None:
            print("\n--- Full Regression Output ---")
            print(self._results.summary().tables[1])
        
        print("=" * 60)
    
    def get_results(self):
        """Return results as a dictionary."""
        return {
            'att': self.att_,
            'std_error': self.std_error_,
            'ci_95': self.ci_,
            'p_value': self.p_value_,
        }
    
    def placebo_test(self, data, outcome, treatment_col, time_col, placebo_period):
        """
        Run placebo test using a fake treatment period before actual treatment.
        
        A significant placebo effect suggests violation of parallel trends.
        
        Parameters
        ----------
        placebo_period : int or float
            The time period to use as fake post-treatment.
        
        Returns
        -------
        dict with placebo estimate and p-value.
        """
        df = data.copy()
        
        # Create fake post indicator
        df['placebo_post'] = (df[time_col] == placebo_period).astype(int)
        df['placebo_interaction'] = df[treatment_col] * df['placebo_post']
        
        # Only use pre-treatment data
        pre_data = df[df[time_col] <= placebo_period].copy()
        
        X = pd.DataFrame({
            'const': 1,
            'treated': pre_data[treatment_col],
            'post': pre_data['placebo_post'],
            'did': pre_data['placebo_interaction']
        })
        y = pre_data[outcome]
        
        model = sm.OLS(y, X).fit(cov_type='HC3')
        
        return {
            'placebo_att': model.params['did'],
            'p_value': model.pvalues['did'],
            'significant': model.pvalues['did'] < 0.05
        }
