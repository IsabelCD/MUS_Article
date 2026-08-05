"""
precision_estimation.py
-----------------------
Summarise the *precision* of the error estimator across simulation runs.

These functions operate on the aggregated ``results`` DataFrame produced
by the simulation loop, not on individual samples.

Public API
----------
error_precision_metrics(results, method, EE_true, z_score)
    -> tuple[float, float, float]   (Bias_EE, SE_true, accuracy_true)

precision_of_precision_metrics(results, method, SE_true, z_score)
    -> tuple[float, float, float]   (Bias_SE, SE_of_SE, accuracy_of_SE)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, beta as beta_dist, gamma as gamma_dist

def precision_poisson_stringer(
    sample_s: pd.DataFrame,
    EE: float,
    SI: float,
    cl: float,
):

    errors = (
        sample_s.loc[
            (sample_s["E"] != 0),
            ["ER"],
        ]
        .copy()
        .sort_values("ER", ascending=False)
        .reset_index(drop=True)
    )

    taints = errors["ER"].to_numpy(dtype=float)

    basic_rf = gamma_dist.ppf(q=cl, a=1, scale=1)

    BP = SI * basic_rf

    ranks = np.arange(1, len(taints) + 1)

    incremental_factors = (
        gamma_dist.ppf(q=cl, a=ranks + 1, scale=1) - gamma_dist.ppf(q=cl, a=ranks, scale=1) - 1
    )

    IA = SI * np.dot(incremental_factors, taints)

    SE = BP + IA
    ULE = EE + SE
    VAR = 0

    return SE, VAR, ULE

def precision_binomial_stringer(
    sample_s: pd.DataFrame,
    EE: float,
    BV: float,
    cl: float,
    ns: int,
):
    errors = (
            sample_s.loc[
                (sample_s["E"] != 0),
                ["ER"],
            ]
            .copy()
            .sort_values("ER", ascending=False)
            .reset_index(drop=True)
        )
    
    taints = errors["ER"].to_numpy(dtype=float)

    BP = (1 - (1 - cl) ** (1 / ns))

    ranks = np.arange(1, len(taints) + 1)

    incremental_factors = (
        beta_dist.ppf(q=cl, a=ranks + 1, b=ns - ranks) - beta_dist.ppf(q=cl, a=ranks, b=ns - (ranks-1))
    )

    IA = np.dot(incremental_factors, taints)

    SE = (BP + IA) * BV
    ULE = EE + SE
    VAR = 0

    return SE, VAR, ULE


def precision_HH(sample_s: pd.DataFrame, EE: float, BVs: float, ns: int, z_score: float, cl: float, SI: float = None):
    # of population 
    sample_s['E/BV'] = sample_s['E'] / sample_s['BV']
    sr = np.std(sample_s['E/BV'], ddof=1) 
    SE = z_score * sr * BVs / np.sqrt(ns)

    if sample_s['E'].sum() == 0:
        basic_rf = gamma_dist.ppf(q=cl, a=1, scale=1)
        SE = SI*basic_rf # This is equal to BP

    # Save Variance of estimator
    VAR = ((sr * BVs)**2) / ns
    ULE = EE + SE  

    return SE, VAR, ULE 

def precision_modified_HH(sample_s: pd.DataFrame, EE: float, BVs: float, ns: int, z_score: float, cl: float, SI: float = None):
    sr = np.std(sample_s['ER'], ddof=1) 
    SE = z_score * sr * BVs * np.sqrt((BVs-sample_s['BV'].sum())/BVs) / np.sqrt(ns)

    if sample_s['E'].sum() == 0:
        basic_rf = gamma_dist.ppf(q=cl, a=1, scale=1)
        SE = SI*basic_rf # This is equal to BP
    
    # Save Variance of estimator
    VAR = ((sr * BVs)**2) / ns
    ULE = EE + SE

    return SE, VAR, ULE


def precision_moment_bound(sample_s: pd.DataFrame, EE: float, BVs: float, ns: int, z_score: float):
    """
    Moment bound (Dworin & Grimlund, 1984, 1986), as implemented (as an
    experimental bound) by ``MUS.moment.bound`` in the R package ``MUS``.

    Instead of relying on the central limit theorem (which under-covers when
    the tainting distribution is highly skewed, as is typical in audit
    populations with mostly-zero errors), the moment bound approximates the
    *sampling distribution of the mean tainting* by a 3-parameter (shifted)
    gamma distribution. Its three parameters are fit, by the method of
    moments, to the first three sample moments of the observed taintings
    (``ER`` = E / BV) taken over the *whole* sample (i.e. including the
    zero-tainting, correctly-stated items) - mirroring the ``MUS`` package,
    which fits the moment bound on the full tainting vector of the
    evaluated sample.

    Steps
    -----
    1. M1, M2, M3 = 1st raw / 2nd & 3rd central sample moments of ER
       (M2, M3 use the biased, divide-by-n convention, as in the original
       Dworin & Grimlund method-of-moments fit).
    2. The sampling distribution of the mean tainting T-bar has (to this
       order of approximation): mean = M1, variance = M2 / ns,
       skewness = (M3 / M2**1.5) / sqrt(ns).
    3. Fit a shifted Gamma(shape=k, scale=theta, shift=tau) to
       (mean, variance, skewness) of T-bar via the standard gamma
       method-of-moments relations:
           k     = 4 / skewness**2
           theta = sd * skewness / 2
           tau   = mean - 2 * sd / skewness
    4. The upper bound on the mean tainting at the confidence level implied
       by ``z_score`` (confidence_level = Phi(z_score)) is
           tau + theta * Gamma^{-1}(confidence_level; k)
       and the upper bound on the total error is BVs times that quantity.

    If the sample shows no (or negative) skewness -- e.g. a small,
    zero-error sample -- the method falls back to the usual normal
    (z_score) approximation, since the gamma method of moments is
    undefined/unstable in that regime.

    Parameters
    ----------
    sample_s : DataFrame with an 'ER' column (E / BV tainting) for every
        sampled item (zeros included).
    EE : point estimate of the total error.
    BVs : total population book value.
    ns : sample size.
    z_score : normal quantile associated with the desired one-sided
        confidence level (used both as the CLT fallback, and to fix the
        confidence level for the gamma quantile via confidence_level =
        Phi(z_score)).

    Returns
    -------
    (SE, VAR, ULE)
    """
    taints = sample_s['ER'].to_numpy(dtype=float)
    n = len(taints)

    M1 = taints.mean()
    M2 = np.mean((taints - M1) ** 2)
    M3 = np.mean((taints - M1) ** 3)

    mean_bar = M1
    var_bar = M2 / n
    sd_bar = np.sqrt(var_bar) if var_bar > 0 else 0.0

    confidence_level = norm.cdf(z_score)

    if sd_bar == 0:
        # Degenerate sample (e.g. a zero-error sample): no dispersion to
        # extrapolate a bound from.
        SE = 0.0
        VAR = 0.0
        ULE = EE
        return SE, VAR, ULE

    skew_pop = M3 / (M2 ** 1.5)          # skewness of the individual taintings
    skew_bar = skew_pop / np.sqrt(n)     # skewness of the sample mean (CLT scaling)

    if skew_bar <= 0:
        # No usable positive skew: fall back to the normal approximation.
        upper_mean = mean_bar + z_score * sd_bar
    else:
        k = 4.0 / (skew_bar ** 2)
        theta = sd_bar * skew_bar / 2.0
        tau = mean_bar - 2.0 * sd_bar / skew_bar
        upper_mean = tau + theta * gamma_dist.ppf(confidence_level, k)

    SE = BVs * (upper_mean - mean_bar)
    VAR = var_bar * (BVs ** 2)

    ULE = EE + SE

    return SE, VAR, ULE


def precision_estimator(bound_estimator, **kwargs):
    estimators = {
        "HH": precision_HH,
        "Mod_HH": precision_modified_HH,
        "Poisson_Stringer": precision_poisson_stringer,
        "Binomial_Stringer": precision_binomial_stringer,
        "Moment": precision_moment_bound,
    }

    try:
        estimator = estimators[bound_estimator]
    except KeyError:
        raise ValueError(f"Unknown bound estimator: {bound_estimator}")

    return estimator(**kwargs)