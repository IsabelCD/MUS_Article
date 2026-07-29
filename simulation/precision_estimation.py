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
from scipy.stats import norm
from config import RF_TABLE


def precision_conservative(sample_s: pd.DataFrame, EE: float, SI: float):
    

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

def precision_HH(sample_s: pd.DataFrame, EE: float, Qs: float, ns: int, z_score: float):
    # of population 
    sample_s['E/Q'] = sample_s['E'] / sample_s['Q']
    sr = np.std(sample_s['E/Q'], ddof=1) 
    SE = z_score * sr * Qs / np.sqrt(ns)

    # Save Variance of estimator
    VAR = ((sr * Qs)**2) / ns

    LLE = EE - SE
    ULE = EE + SE

    return SE, VAR, LLE, ULE 

def precision_modified_HH(sample_s: pd.DataFrame, EE: float, Qs: float, ns: int, z_score: float):
    sr = np.std(sample_s['ER'], ddof=1) 
    SE = z_score * sr * Qs * np.sqrt((Qs-sample_s['BV'].sum())/Qs) / np.sqrt(ns)
    
    # Save Variance of estimator
    VAR = ((sr * Qs)**2) / ns

    LLE = EE - SE
    ULE = EE + SE

    return SE, VAR, LLE, ULE

def precision_ratio(sample_s: pd.DataFrame, EE: float, Qs: float, ns: int, z_score: float):
    [...]

    return SE, VAR, LLE, ULE


def precision_estimator(bound_estimator, **kwargs):
    estimators = {
        "HH": precision_HH,
        "Mod_HH": precision_modified_HH,
        "Con": precision_conservative,
        "Ratio": precision_ratio,
    }

    try:
        estimator = estimators[bound_estimator]
    except KeyError:
        raise ValueError(f"Unknown bound estimator: {bound_estimator}")

    return estimator(**kwargs)