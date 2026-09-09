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

    taints = sample_s["ER"].to_numpy(dtype=float)
    taints = np.sort(taints[taints > 0])[::-1]

    basic_rf = gamma_dist.ppf(q=cl, a=1, scale=1)

    BP = SI * basic_rf

    ranks = np.arange(1, len(taints) + 1)

    incremental_factors = (
        gamma_dist.ppf(q=cl, a=ranks + 1, scale=1) - gamma_dist.ppf(q=cl, a=ranks, scale=1) - 1
    )

    IA = SI * np.dot(incremental_factors, taints)

    SE = BP + IA
    ULE = EE + SE

    if SE < 0:
            raise ValueError(
                f"Negative Poisson Stringer precision ({SE}). "
                "Check EE, SI, cl, and the definition of ER."
            )

    return SE, "", ULE

def precision_binomial_stringer(
    sample_s: pd.DataFrame,
    EE: float,
    SI: float,
    cl: float,
    sample_size: int
):

    taints = sample_s["ER"].to_numpy(dtype=float)
    taints = np.sort(taints[taints > 0])[::-1]

    basic_rf = sample_size * beta_dist.ppf(q=cl, a=1, b=sample_size,)

    BP = SI * basic_rf

    ranks = np.arange(1, len(taints) + 1)

    # p_k^U = Beta^-1(cl; k+1, n-k) is undefined at k=n (every sampled item
    # tainted): shape2 = n-k = 0. The analytical limit is 1 exactly, since an
    # error rate cannot exceed 1 -- in this n-scaled convention that means
    # the first term becomes n*1 = n there, instead of NaN. (Only the first
    # term can hit this: the second term's shape2 = n-(k-1) >= 1 always.)
    # NB the previous version of this fix used incremental_factors==np.nan,
    # which is always False (NaN never equals anything, including itself)
    # and so never actually replaced anything -- np.where on the boolean
    # b_first==0 mask (computed before the NaN appears) avoids that trap.
    first_term = np.where(
        ranks < sample_size,
        sample_size * beta_dist.ppf(q=cl, a=ranks + 1, b=sample_size - ranks,),
        sample_size,
    )
    second_term = sample_size * beta_dist.ppf(q=cl, a=ranks, b=sample_size - ranks + 1)
    incremental_factors = first_term - second_term - 1

    IA = SI * np.dot(incremental_factors, taints)

    SE = BP + IA
    ULE = EE + SE

    if SE < 0:
            raise ValueError(
                f"Negative Binomial Stringer precision ({SE}). "
                "Check EE, SI, cl, and the definition of ER."
            )

    return SE, "", ULE

def precision_HH(sample_s: pd.DataFrame, 
                 EE: float, 
                 EEe: float,
                 BVs: float, 
                 ns: int, 
                 z_score: float, 
                 cl: float, 
                 SI: float = None
                 ):
    # of population 
    sample_s['E/BV'] = sample_s['E'] / sample_s['BV']
    sr = np.std(sample_s['E/BV'], ddof=1) 
    SE_main = z_score * sr * BVs / np.sqrt(ns)

    # alternative bound
    basic_rf = sample_s.shape[0] * beta_dist.ppf(q=cl, a=1, b=sample_s.shape[0],)
    SE_spec = SI * basic_rf # This is equal to BP

    # define the Upper limit of error by the two rules
    ULE_main = EE + SE_main
    ULE_spec = EEe + SE_spec

    # Save the one that is highest, and update the SE accordingly. 
    # This is the one that will be used for coverage and acceptance rate calculations.
    ULE = max(ULE_main, ULE_spec)
    ULE_name = "main" if ULE_main >= ULE_spec else "spec"
    SE = SE_main if ULE_name == "main" else SE_spec

    return SE, ULE_main, ULE 


def precision_modified_HH(sample_s: pd.DataFrame, 
                 EE: float, 
                 EEe: float,
                 BVs: float, 
                 ns: int, 
                 z_score: float, 
                 cl: float, 
                 SI: float = None
                 ):
    sr = np.std(sample_s['ER'], ddof=1) 
    SE_main = z_score * sr * BVs * np.sqrt((BVs-sample_s['BV'].sum())/BVs) / np.sqrt(ns)
    
    # alternative bound
    basic_rf = sample_s.shape[0] * beta_dist.ppf(q=cl, a=1, b=sample_s.shape[0],)
    SE_spec = SI * basic_rf # This is equal to BP

    # define the Upper limit of error by the two rules
    ULE_main = EE + SE_main
    ULE_spec = EEe + SE_spec

    # Save the one that is highest, and update the SE accordingly. 
    # This is the one that will be used for coverage and acceptance rate calculations.
    ULE = max(ULE_main, ULE_spec)
    ULE_name = "main" if ULE_main >= ULE_spec else "spec"
    SE = SE_main if ULE_name == "main" else SE_spec

    return SE, ULE_main, ULE


def precision_moment_bound(sample_s: pd.DataFrame, 
                           EE: float, 
                           BVs: float, 
                           cl: float, 
                           EEe: float
                           ):

    taints = np.asarray(sample_s['ER'], dtype=float)

    # Total number of observations
    N = len(taints)

    # Non-zero taintings
    tall = taints[taints != 0]
    n = len(tall)

    # Hypothetical tainting
    mean_tall = np.mean(tall) if n > 0 else 0.0
    tstar = (
        0.81
        * (1 - 0.667 * np.tanh(10 * mean_tall))
        * (1 + 0.667 * np.tanh(n / 10))
    )

    # TN
    ncm1_z = (tstar + np.sum(tall)) / (n + 1)
    ncm2_z = (tstar**2 + np.sum(tall**2)) / (n + 1)
    ncm3_z = (tstar**3 + np.sum(tall**3)) / (n + 1)

    # RN
    ncm1_e = (n + 1) / (N + 2)
    ncm2_e = ncm1_e * (n + 2) / (N + 3)
    ncm3_e = ncm2_e * (n + 3) / (N + 4)

    # UN
    ncm1_t = ncm1_e * ncm1_z

    ncm2_t = (
        ncm1_e * ncm2_z
        + (N - 1) * ncm2_e * ncm1_z**2
    ) / N

    ncm3_t = (
        ncm1_e * ncm3_z
        + 3 * (N - 1) * ncm2_e * ncm1_z * ncm2_z
        + (N - 1) * (N - 2) * ncm3_e * ncm1_z**3
    ) / N**2

    # UC
    cm2_t = ncm2_t - ncm1_t**2
    cm3_t = (
        ncm3_t
        - 3 * ncm1_t * ncm2_t
        + 2 * ncm1_t**3
    )

    # A, B, G
    A = 4 * cm2_t**3 / cm3_t**2
    B = 0.5 * cm3_t / cm2_t
    G = ncm1_t - 2 * cm2_t**2 / cm3_t

    # One-sided upper confidence bound
    Z = norm.ppf(cl)

    ULE = G + A * B * (
        1
        + Z / np.sqrt(9 * A)
        - 1 / (9 * A)
    )**3

    # transform into monetary value
    ULE = EEe + ULE * BVs

    SE = ULE - EE

    return SE, "", ULE



def precision_moment_bound_jfa_inventory(sample_s: pd.DataFrame,
                           EE: float,
                           BVs: float,
                           cl: float,
                           EEe: float
                           ):
    """
    Moment bound, m.type="inventory" variant from jfa's .moment() (R/methods.R,
    koenderks/jfa). Identical Cornish-Fisher machinery to precision_moment_bound
    (jfa's m.type="accounts"); the only difference is the hypothetical-tainting
    term tstar, which drops the (1 + 0.667*tanh(n/10)) factor and uses
    abs(mean(tall)) instead of mean(tall) -- unlike "accounts", this keeps
    tstar bounded at 0.81 regardless of taint count or sign, rather than
    letting it grow past 1 (a physically impossible taint) as errors
    accumulate. Matches jfa in not special-casing zero errors: tstar falls
    back to 0.81*(1-0.667*tanh(0)) = 0.81 and the same formula runs through
    unchanged, rather than substituting a different bound entirely.
    """

    taints = np.asarray(sample_s['ER'], dtype=float)

    # Total number of observations
    N = len(taints)

    # Non-zero taintings
    tall = taints[taints != 0]
    n = len(tall)

    # Hypothetical tainting (jfa m.type="inventory")
    mean_tall = np.mean(tall) if n > 0 else 0.0
    tstar = 0.81 * (1 - 0.667 * np.tanh(10 * np.abs(mean_tall)))

    # TN
    ncm1_z = (tstar + np.sum(tall)) / (n + 1)
    ncm2_z = (tstar**2 + np.sum(tall**2)) / (n + 1)
    ncm3_z = (tstar**3 + np.sum(tall**3)) / (n + 1)

    # RN
    ncm1_e = (n + 1) / (N + 2)
    ncm2_e = ncm1_e * (n + 2) / (N + 3)
    ncm3_e = ncm2_e * (n + 3) / (N + 4)

    # UN
    ncm1_t = ncm1_e * ncm1_z

    ncm2_t = (
        ncm1_e * ncm2_z
        + (N - 1) * ncm2_e * ncm1_z**2
    ) / N

    ncm3_t = (
        ncm1_e * ncm3_z
        + 3 * (N - 1) * ncm2_e * ncm1_z * ncm2_z
        + (N - 1) * (N - 2) * ncm3_e * ncm1_z**3
    ) / N**2

    # UC
    cm2_t = ncm2_t - ncm1_t**2
    cm3_t = (
        ncm3_t
        - 3 * ncm1_t * ncm2_t
        + 2 * ncm1_t**3
    )

    # A, B, G
    A = 4 * cm2_t**3 / cm3_t**2
    B = 0.5 * cm3_t / cm2_t
    G = ncm1_t - 2 * cm2_t**2 / cm3_t

    # One-sided upper confidence bound
    Z = norm.ppf(cl)

    ULE = G + A * B * (
        1
        + Z / np.sqrt(9 * A)
        - 1 / (9 * A)
    )**3

    # transform into monetary value
    ULE = EEe + ULE * BVs

    SE = ULE - EE

    return SE, "", ULE


def precision_moment_bound_jfa_accounts(sample_s: pd.DataFrame,
                           EE: float,
                           BVs: float,
                           cl: float,
                           EEe: float,
                           ):
    """
    Moment bound, m.type="accounts" variant from jfa's .moment() (R/methods.R,
    koenderks/jfa). Identical Cornish-Fisher machinery to
    precision_moment_bound_jfa_inventory (jfa's m.type="inventory"); the only
    difference is the hypothetical-tainting term tstar, which includes the
    (1 + 0.667*tanh(n/10)) factor and uses mean(tall) (not abs()) -- unlike
    "inventory", this lets tstar grow past 1 (a physically impossible taint)
    as the number of errors grows. Matches jfa in not special-casing zero
    errors: tstar falls back to 0.81*(1-0.667*tanh(0))*(1+0.667*tanh(0)) =
    0.81 and the same formula runs through unchanged, rather than
    substituting a different bound entirely.
    """

    taints = np.asarray(sample_s['ER'], dtype=float)

    # Total number of observations
    N = len(taints)

    # Non-zero taintings
    tall = taints[taints != 0]
    n = len(tall)

    # Hypothetical tainting (jfa m.type="accounts")
    mean_tall = np.mean(tall) if n > 0 else 0.0
    tstar = 0.81 * (1 - 0.667 * np.tanh(10 * mean_tall)) * (1 + 0.667 * np.tanh(n / 10))

    # TN
    ncm1_z = (tstar + np.sum(tall)) / (n + 1)
    ncm2_z = (tstar**2 + np.sum(tall**2)) / (n + 1)
    ncm3_z = (tstar**3 + np.sum(tall**3)) / (n + 1)

    # RN
    ncm1_e = (n + 1) / (N + 2)
    ncm2_e = ncm1_e * (n + 2) / (N + 3)
    ncm3_e = ncm2_e * (n + 3) / (N + 4)

    # UN
    ncm1_t = ncm1_e * ncm1_z

    ncm2_t = (
        ncm1_e * ncm2_z
        + (N - 1) * ncm2_e * ncm1_z**2
    ) / N

    ncm3_t = (
        ncm1_e * ncm3_z
        + 3 * (N - 1) * ncm2_e * ncm1_z * ncm2_z
        + (N - 1) * (N - 2) * ncm3_e * ncm1_z**3
    ) / N**2

    # UC
    cm2_t = ncm2_t - ncm1_t**2
    cm3_t = (
        ncm3_t
        - 3 * ncm1_t * ncm2_t
        + 2 * ncm1_t**3
    )

    # A, B, G
    A = 4 * cm2_t**3 / cm3_t**2
    B = 0.5 * cm3_t / cm2_t
    G = ncm1_t - 2 * cm2_t**2 / cm3_t

    # One-sided upper confidence bound
    Z = norm.ppf(cl)

    ULE = G + A * B * (
        1
        + Z / np.sqrt(9 * A)
        - 1 / (9 * A)
    )**3

    # transform into monetary value
    ULE = EEe + ULE * BVs

    SE = ULE - EE

    return SE, "", ULE

def precision_estimator(bound_estimator, **kwargs):
    estimators = {
        "HH": precision_HH,
        "Mod_HH": precision_modified_HH,
        "Poisson_Stringer": precision_poisson_stringer,
        "Binomial_Stringer": precision_binomial_stringer,
        "Moment": precision_moment_bound,
        "Moment_jfa_inventory": precision_moment_bound_jfa_inventory,
        "Moment_jfa_accounts": precision_moment_bound_jfa_accounts,
    }

    try:
        estimator = estimators[bound_estimator]
    except KeyError:
        raise ValueError(f"Unknown bound estimator: {bound_estimator}")

    return estimator(**kwargs)