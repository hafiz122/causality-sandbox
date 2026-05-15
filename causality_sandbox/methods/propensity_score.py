"""
Propensity Score Matching (PSM) for causal inference.

Rosenbaum & Rubin (1983): Conditioning on the propensity score is sufficient
for unbiased treatment effect estimation under strong ignorability.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import warnings


class PropensityScoreMatching:
    """
    Propensity Score Matching for estimating Average Treatment Effect on the Treated (ATT).
    
    Estimates causal effects by matching treated units to control units with similar
    propensity scores, reducing selection bias from confounding covariates.
    
    Parameters
    ----------
    model : str, default 'logistic'
        Model for propensity score estimation. 'logistic' or 'probit'.
    caliper : float or None, default None
        Maximum allowable distance for matching. None means no caliper.
    replacement : bool, default False
        Whether to match with replacement.
    k : int, default 1
        Number of nearest neighbors to match.
    random_state : int or None, default None
        Random seed for reproducibility.
    
    Attributes
    ----------
    propensity_scores_ : ndarray
        Estimated propensity scores.
    matches_ : list of tuples
        Matched pairs (treated_idx, control_idx).
    att_ : float
        Estimated Average Treatment Effect on the Treated.
    att_std_error_ : float
        Standard error of ATT estimate.
    att_ci_ : tuple
        95% confidence interval for ATT.
    
    References
    ----------
    Rosenbaum, P.R. & Rubin, D.B. (1983). The central role of the propensity score
    in observational studies for causal effects. Biometrika, 70(1), 41-55.
    """
    
    def __init__(self, model='logistic', caliper=None, replacement=False, k=1, random_state=None):
        self.model = model
        self.caliper = caliper
        self.replacement = replacement
        self.k = k
        self.random_state = random_state
        
        self.propensity_scores_ = None
        self.matches_ = None
        self.att_ = None
        self.att_std_error_ = None
        self.att_ci_ = None
        self._propensity_model = None
    
    def fit(self, X, treatment, outcome):
        """
        Estimate propensity scores and compute ATT.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariates/confounders.
        treatment : array-like of shape (n_samples,)
            Binary treatment indicator (1=treated, 0=control).
        outcome : array-like of shape (n_samples,)
            Outcome variable.
        
        Returns
        -------
        self : PropensityScoreMatching
        """
        X = np.asarray(X)
        treatment = np.asarray(treatment).flatten()
        outcome = np.asarray(outcome).flatten()
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Step 1: Estimate propensity scores
        self._estimate_propensity_scores(X, treatment)
        
        # Step 2: Match treated to control units
        self._match(treatment)
        
        # Step 3: Estimate ATT from matched pairs
        self._estimate_att(treatment, outcome)
        
        return self
    
    def _estimate_propensity_scores(self, X, treatment):
        """Estimate propensity scores using logistic regression."""
        if self.model == 'logistic':
            self._propensity_model = LogisticRegression(
                max_iter=1000, 
                solver='lbfgs',
                random_state=self.random_state
            )
        elif self.model == 'probit':
            # Use LogisticRegression as approximation; statsmodels for exact probit
            self._propensity_model = LogisticRegression(
                max_iter=1000,
                solver='lbfgs',
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown model: {self.model}")
        
        self._propensity_model.fit(X, treatment)
        self.propensity_scores_ = self._propensity_model.predict_proba(X)[:, 1]
        
        # Check overlap/common support
        treated_ps = self.propensity_scores_[treatment == 1]
        control_ps = self.propensity_scores_[treatment == 0]
        
        if np.min(treated_ps) < np.min(control_ps) or np.max(treated_ps) > np.max(control_ps):
            warnings.warn(
                "Limited overlap detected between treatment and control propensity scores. "
                "Consider trimming units outside common support.",
                UserWarning
            )
    
    def _match(self, treatment):
        """Match treated units to nearest control units on propensity score."""
        treated_idx = np.where(treatment == 1)[0]
        control_idx = np.where(treatment == 0)[0]
        
        treated_ps = self.propensity_scores_[treated_idx].reshape(-1, 1)
        control_ps = self.propensity_scores_[control_idx].reshape(-1, 1)
        
        if self.replacement:
            # Matching with replacement
            nbrs = NearestNeighbors(n_neighbors=self.k, metric='euclidean')
            nbrs.fit(control_ps)
            distances, indices = nbrs.kneighbors(treated_ps)
            
            self.matches_ = []
            for i, t_idx in enumerate(treated_idx):
                for j in range(self.k):
                    matched_control_idx = control_idx[indices[i, j]]
                    if self.caliper is None or distances[i, j] <= self.caliper:
                        self.matches_.append((t_idx, matched_control_idx))
        else:
            # Matching without replacement using greedy algorithm
            available_controls = set(range(len(control_idx)))
            self.matches_ = []
            
            # Randomize order of treated units to reduce ordering bias
            order = np.random.permutation(len(treated_idx))
            
            for i in order:
                t_idx = treated_idx[i]
                t_ps = treated_ps[i]
                
                best_dist = float('inf')
                best_j = None
                
                for j in available_controls:
                    dist = np.abs(t_ps - control_ps[j])[0]
                    if dist < best_dist:
                        best_dist = dist
                        best_j = j
                
                if best_j is not None and (self.caliper is None or best_dist <= self.caliper):
                    self.matches_.append((t_idx, control_idx[best_j]))
                    available_controls.remove(best_j)
    
    def _estimate_att(self, treatment, outcome):
        """Estimate ATT from matched pairs."""
        if not self.matches_:
            raise RuntimeError("No matches found. Check caliper or data.")
        
        treated_outcomes = []
        control_outcomes = []
        
        for t_idx, c_idx in self.matches_:
            treated_outcomes.append(outcome[t_idx])
            control_outcomes.append(outcome[c_idx])
        
        treated_outcomes = np.array(treated_outcomes)
        control_outcomes = np.array(control_outcomes)
        
        # ATT estimate
        self.att_ = np.mean(treated_outcomes - control_outcomes)
        
        # Standard error (Abadie & Imbens, 2006)
        n = len(self.matches_)
        var = np.var(treated_outcomes - control_outcomes, ddof=1)
        self.att_std_error_ = np.sqrt(var / n)
        
        # 95% confidence interval
        margin = stats.t.ppf(0.975, df=n-1) * self.att_std_error_
        self.att_ci_ = (self.att_ - margin, self.att_ + margin)
    
    def summary(self):
        """Print a summary of the matching results."""
        if self.att_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        
        n_matches = len(self.matches_)
        
        print("=" * 60)
        print("Propensity Score Matching Results")
        print("=" * 60)
        print(f"Number of matched pairs:    {n_matches}")
        print(f"Matching method:            {self.k}:1 nearest neighbor")
        print(f"With replacement:           {self.replacement}")
        print(f"Caliper:                    {self.caliper if self.caliper else 'None'}")
        print("-" * 60)
        print(f"ATT Estimate:               {self.att_:.4f}")
        print(f"Std. Error:                 {self.att_std_error_:.4f}")
        print(f"95% CI:                     [{self.att_ci_[0]:.4f}, {self.att_ci_[1]:.4f}]")
        
        # Significance test
        t_stat = self.att_ / self.att_std_error_
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n_matches-1))
        print(f"t-statistic:                {t_stat:.4f}")
        print(f"p-value:                    {p_value:.4f}")
        print("=" * 60)
    
    def balance_table(self, X, treatment, matched_only=True):
        """
        Compute covariate balance before and after matching.
        
        Returns standardized mean differences for each covariate.
        Values near 0 indicate good balance.
        """
        X = np.asarray(X)
        treatment = np.asarray(treatment).flatten()
        
        treated_mask = treatment == 1
        control_mask = treatment == 0
        
        if matched_only and self.matches_ is not None:
            matched_treated = [t_idx for t_idx, _ in self.matches_]
            matched_control = [c_idx for _, c_idx in self.matches_]
            treated_mask_matched = np.zeros(len(treatment), dtype=bool)
            control_mask_matched = np.zeros(len(treatment), dtype=bool)
            treated_mask_matched[matched_treated] = True
            control_mask_matched[matched_control] = True
            treated_mask = treated_mask_matched
            control_mask = control_mask_matched
        
        balance = {}
        for i in range(X.shape[1]):
            treated_mean = np.mean(X[treated_mask, i])
            control_mean = np.mean(X[control_mask, i])
            pooled_std = np.sqrt(
                (np.var(X[treatment==1, i], ddof=1) + np.var(X[treatment==0, i], ddof=1)) / 2
            )
            
            if pooled_std > 0:
                std_diff = (treated_mean - control_mean) / pooled_std
            else:
                std_diff = 0
            
            balance[f'X{i}'] = {
                'treated_mean': treated_mean,
                'control_mean': control_mean,
                'std_diff': std_diff
            }
        
        return pd.DataFrame(balance).T
