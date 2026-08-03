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
from config import RF_TABLE


def precision_poisson_stringer(sample_s: pd.DataFrame, EE: float, SI: float):
    

    # Reliability factor of order 0
    RF = 2.31

    sample_s = sample_s[sample_s['E']!=0].sort_values(by=['ER'], ascending=False)

    # Precision for Con
    RF_table_aux = pd.concat([RF_TABLE.reset_index(drop=True), sample_s.reset_index(drop=True)], axis=1)
    RF_table_aux.dropna(subset='BV', inplace=True)
    RF_table_aux['IA'] =  RF_table_aux['RF_factor']*SI*RF_table_aux['ER']
    
    IA = sum(RF_table_aux['IA'])
    BP = SI*RF # RF = 2.31
    SE = BP+IA

    VAR = 0

    LLE = EE - SE
    ULE = EE + SE

    return SE, VAR, LLE, ULE 


def precision_binomial_stringer(sample_s: pd.DataFrame, EE: float, SI: float, ns: int,
                                 confidence_level: float = 0.95):
    """
    Binomial Stringer bound.
 
    ``precision_poisson_stringer`` (above) implements the classic Stringer bound
    using the *Poisson* approximation to the binomial upper confidence limits
    (via the pre-computed ``RF_TABLE`` reliability/precision-gap-widening
    factors). This function implements the exact *binomial* analogue.
 
    Following Stringer (1963) (see e.g. Berger, Chiodini & Zenga, "Bounds for
    monetary-unit sampling in auditing", eq. (11)-(11'), and the many
    equivalent descriptions in the audit-sampling literature), let
 
        u_i := unique solution in (0, 1) of
               sum_{k=0}^{i} C(ns, k) * u_i^k * (1 - u_i)^(ns - k) = 1 - confidence_level
 
    for i = 0, 1, ..., ns - 1, with u_ns := 1. Because the left-hand side is
    the CDF of a Binomial(ns, u_i) evaluated at i, u_i is exactly the
    Clopper-Pearson upper confidence limit for a binomial proportion with i
    "successes" (errors) out of ns trials, which has the closed form
 
        u_i = Beta^{-1}(confidence_level; i + 1, ns - i)     (Beta quantile / ppf)
 
    The bound itself is then, with ER_(1) >= ER_(2) >= ... >= ER_(m) the
    taintings of the m items with a non-zero error, ranked in decreasing
    order:
 
        UEL = SI * ns * u_0  +  SI * ns * sum_{i=1}^{m} (u_i - u_{i-1}) * ER_(i)
 
    ``SI`` is the sampling interval (population book value / ns), so
    ``SI * ns`` is the total population book value; this mirrors exactly the
    scaling used in ``precision_poisson_stringer`` (there, ``SI * RF`` with
    ``RF`` the Poisson zero-error reliability factor plays the role of
    ``SI * ns * u_0`` here, since for rare errors ``ns * u_0`` converges to
    the Poisson factor).
 
    Parameters
    ----------
    sample_s : DataFrame with columns 'E' (dollar error) and 'ER' (tainting,
        i.e. E / BV) for every sampled item.
    EE : point estimate of the total error.
    SI : sampling interval (population book value / ns).
    ns : total number of items in the sample (not just the erroneous ones).
    confidence_level : one-sided confidence level (default 0.95).
 
    Returns
    -------
    (SE, VAR, LLE, ULE)
    """
    alpha = 1 - confidence_level
 
    sample_s = sample_s[sample_s['E'] != 0].sort_values(by=['ER'], ascending=False)
    m = len(sample_s)
 
    if m >= ns:
        raise ValueError("Number of tainted items cannot exceed the sample size ns.")
 
    # u_0, u_1, ..., u_m  (u_i = Beta^{-1}(confidence_level; i+1, ns-i))
    idx = np.arange(m + 1)
    u = beta_dist.ppf(confidence_level, idx + 1, ns - idx)
 
    rf_factor = np.diff(u)  # rf_factor[i-1] = u_i - u_{i-1}, for i = 1..m
 
    BP = SI * ns * u[0]
    IA = SI * ns * np.sum(rf_factor * sample_s['ER'].to_numpy())
    SE = BP + IA
 
    VAR = 0
 
    LLE = EE - SE
    ULE = EE + SE
 
    return SE, VAR, LLE, ULE

def precision_HH(sample_s: pd.DataFrame, EE: float, BVs: float, ns: int, z_score: float):
    # of population 
    sample_s['E/BV'] = sample_s['E'] / sample_s['BV']
    sr = np.std(sample_s['E/BV'], ddof=1) 
    SE = z_score * sr * BVs / np.sqrt(ns)

    # Save Variance of estimator
    VAR = ((sr * BVs)**2) / ns

    LLE = EE - SE
    ULE = EE + SE

    return SE, VAR, LLE, ULE 

def precision_modified_HH(sample_s: pd.DataFrame, EE: float, BVs: float, ns: int, z_score: float):
    sr = np.std(sample_s['ER'], ddof=1) 
    SE = z_score * sr * BVs * np.sqrt((BVs-sample_s['BV'].sum())/BVs) / np.sqrt(ns)
    
    # Save Variance of estimator
    VAR = ((sr * BVs)**2) / ns

    LLE = EE - SE
    ULE = EE + SE

    return SE, VAR, LLE, ULE


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
    (SE, VAR, LLE, ULE)
    """
    er = sample_s['ER'].to_numpy(dtype=float)
    n = len(er)

    M1 = er.mean()
    M2 = np.mean((er - M1) ** 2)
    M3 = np.mean((er - M1) ** 3)

    mean_bar = M1
    var_bar = M2 / n
    sd_bar = np.sqrt(var_bar) if var_bar > 0 else 0.0

    confidence_level = norm.cdf(z_score)

    if sd_bar == 0:
        # Degenerate sample (e.g. a zero-error sample): no dispersion to
        # extrapolate a bound from.
        SE = 0.0
        VAR = 0.0
        LLE = EE
        ULE = EE
        return SE, VAR, LLE, ULE

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

    LLE = EE - SE
    ULE = EE + SE

    return SE, VAR, LLE, ULE


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