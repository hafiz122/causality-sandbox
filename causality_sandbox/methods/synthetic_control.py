"""
Synthetic Control Method for causal inference.

Abadie, Diamond & Hainmueller (2010): Create a weighted combination of
control units that best reproduces the pre-treatment outcome trajectory
of the treated unit, then compare post-treatment outcomes.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import warnings


class SyntheticControl:
    """
    Synthetic Control Method for comparative case studies.
    
    Constructs a weighted combination of control units (the "synthetic control")
    that closely matches the treated unit's pre-treatment outcomes. The
    treatment effect is the post-treatment divergence between the treated
    unit and its synthetic counterpart.
    
    Parameters
       ----------
    optimization_method : str, default 'wls'
        Method for estimating weights. 'wls' for weighted least squares,
        'simplex' for Nelder-Mead optimization.
    
    Attributes
    ----------
    weights_ : ndarray
        Optimal weights for each control unit.
    treatment_effect_ : ndarray
        Estimated treatment effect at each post-treatment period.
    rmspe_ : float
        Root mean squared prediction error in pre-treatment period.
    
    References
    ----------
    Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control
    methods for comparative case studies: Estimating the effect of California's
    tobacco control program. Journal of the American Statistical Association,
    105(490), 493-505.
    """
    
    def __init__(self, optimization_method='wls'):
        self.optimization_method = optimization_method
        
        self.weights_ = None
        self.treatment_effect_ = None
        self.rmspe_ = None
        self._pre_treated = None
        self._pre_controls = None
        self._post_treated = None
        self._post_controls = None
        self._control_names = None
    
    def fit(self, pre_treated, pre_controls, post_treated, post_controls,
            control_names=None, feature_weights=None):
        """
        Fit the synthetic control.
        
        Parameters
        ----------
        pre_treated : ndarray of shape (n_pre_periods,)
            Pre-treatment outcomes for the treated unit.
        pre_controls : ndarray of shape (n_pre_periods, n_controls)
            Pre-treatment outcomes for control units.
        post_treated : ndarray of shape (n_post_periods,)
            Post-treatment outcomes for the treated unit.
        post_controls : ndarray of shape (n_post_periods, n_controls)
            Post-treatment outcomes for control units.
        control_names : list or None
            Names for control units.
        feature_weights : ndarray or None
            Importance weights for pre-treatment periods.
        
        Returns
        -------
        self : SyntheticControl
        """
        self._pre_treated = np.asarray(pre_treated).flatten()
        self._pre_controls = np.asarray(pre_controls)
        self._post_treated = np.asarray(post_treated).flatten()
        self._post_controls = np.asarray(post_controls)
        self._control_names = control_names
        
        if self._pre_controls.ndim == 1:
            self._pre_controls = self._pre_controls.reshape(-1, 1)
        if self._post_controls.ndim == 1:
            self._post_controls = self._post_controls.reshape(-1, 1)
        
        # Estimate optimal weights
        if self.optimization_method == 'wls':
            self._estimate_weights_wls(feature_weights)
        elif self.optimization_method == 'simplex':
            self._estimate_weights_simplex()
        else:
            raise ValueError(f"Unknown method: {self.optimization_method}")
        
        # Compute treatment effects
        self._compute_treatment_effect()
        
        # Compute pre-treatment fit quality
        synthetic_pre = self._pre_controls @ self.weights_
        self.rmspe_ = np.sqrt(np.mean((self._pre_treated - synthetic_pre) ** 2))
        
        return self
    
    def _estimate_weights_wls(self, feature_weights=None):
        """Estimate weights using weighted least squares."""
        X = self._pre_controls
        y = self._pre_treated
        
        if feature_weights is not None:
            W = np.diag(feature_weights)
            Xw = W @ X
            yw = W @ y
        else:
            Xw = X
            yw = y
        
        # Non-negative least squares with sum-to-one constraint
        # Use quadratic programming approach
        n_controls = X.shape[1]
        
        def objective(w):
            residual = y - X @ w
            return np.sum(residual ** 2)
        
        # Constraints: weights >= 0 and sum to 1
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = [(0, 1) for _ in range(n_controls)]
        
        # Initialize with equal weights
        w0 = np.ones(n_controls) / n_controls
        
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-10, 'maxiter': 1000}
        )
        
        if not result.success:
            warnings.warn("Optimization did not converge. Results may be unreliable.")
        
        self.weights_ = result.x
    
    def _estimate_weights_simplex(self):
        """Estimate weights using Nelder-Mead simplex."""
        n_controls = self._pre_controls.shape[1]
        
        def objective(w):
            residual = self._pre_treated - self._pre_controls @ w
            return np.sum(residual ** 2)
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        w0 = np.ones(n_controls) / n_controls
        
        result = minimize(
            objective,
            w0,
            method='Nelder-Mead',
            constraints=constraints,
            options={'maxiter': 5000}
        )
        
        # Project onto simplex
        w = result.x
        w = np.maximum(w, 0)
        w = w / np.sum(w)
        
        self.weights_ = w
    
    def _compute_treatment_effect(self):
        """Compute treatment effect as difference between treated and synthetic."""
        synthetic_post = self._post_controls @ self.weights_
        self.treatment_effect_ = self._post_treated - synthetic_post
    
    def placebo_test(self, pre_controls, post_controls, n_permutations=None):
        """
        Run placebo tests by applying synthetic control to each control unit.
        
        Compares the treated unit's treatment effect to the distribution of
        placebo effects to assess statistical significance.
        
        Parameters
        ----------
        pre_controls : ndarray
            Pre-treatment outcomes for all units including treated.
        post_controls : ndarray
            Post-treatment outcomes for all units including treated.
        n_permutations : int or None
            Number of placebo runs. None uses all controls.
        
        Returns
        -------
        dict with placebo distribution and p-value.
        """
        pre_controls = np.asarray(pre_controls)
        post_controls = np.asarray(post_controls)
        
        if pre_controls.ndim == 1:
            pre_controls = pre_controls.reshape(-1, 1)
        if post_controls.ndim == 1:
            post_controls = post_controls.reshape(-1, 1)
        
        n_units = pre_controls.shape[1]
        
        if n_permutations is None or n_permutations > n_units:
            n_permutations = n_units
        
        placebo_effects = []
        rmspes = []
        
        for i in range(n_permutations):
            # Treat unit i as the "treated" unit
            treated_pre = pre_controls[:, i]
            treated_post = post_controls[:, i]
            
            # Remaining units as controls
            other_idx = [j for j in range(n_units) if j != i]
            controls_pre = pre_controls[:, other_idx]
            controls_post = post_controls[:, other_idx]
            
            # Check if enough variation
            if np.std(treated_pre) < 1e-10:
                continue
            
            # Run synthetic control
            try:
                sc = SyntheticControl(optimization_method=self.optimization_method)
                sc.fit(treated_pre, controls_pre, treated_post, controls_post)
                
                placebo_effects.append(sc.treatment_effect_)
                rmspes.append(sc.rmspe_)
            except:
                continue
        
        if not placebo_effects:
            return {'p_value': None, 'placebo_effects': None}
        
        # Compute ratio: post-treatment effect / pre-treatment RMSPE
        ratios = []
        for i, eff in enumerate(placebo_effects):
            if rmspes[i] > 0:
                ratios.append(np.mean(np.abs(eff)) / rmspes[i])
        
        # Our treated effect ratio
        our_ratio = np.mean(np.abs(self.treatment_effect_)) / self.rmspe_
        
        # P-value: proportion of placebo ratios >= our ratio
        p_value = np.mean([r >= our_ratio for r in ratios]) if ratios else None
        
        return {
            'p_value': p_value,
            'placebo_effects': placebo_effects,
            'ratios': ratios,
            'treated_ratio': our_ratio
        }
    
    def summary(self):
        """Print summary of synthetic control results."""
        if self.weights_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        print("=" * 60)
        print("Synthetic Control Method Results")
        print("=" * 60)
        print("--- Control Unit Weights ---")
        
        if self._control_names is not None:
            for name, w in zip(self._control_names, self.weights_):
                if w > 0.001:
                    print(f"  {name:20s}: {w:.4f}")
        else:
            for i, w in enumerate(self.weights_):
                if w > 0.001:
                    print(f"  Control {i:3d}:     {w:.4f}")
        
        print("-" * 60)
        print(f"Pre-treatment RMSPE:        {self.rmspe_:.4f}")
        print(f"Treatment Effect (avg):     {np.mean(self.treatment_effect_):.4f}")
        
        if len(self.treatment_effect_) > 1:
            print(f"  Period-by-period:")
            for t, te in enumerate(self.treatment_effect_):
                print(f"    Period {t+1}: {te:.4f}")
        
        print("=" * 60)
