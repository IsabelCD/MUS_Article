"""
metrics.py
----------
Compute diagnostic metrics from simulation results.

Public API
----------
other_metrics(results, method, iterations, AE_true, SE_true,
              BV, pop, z_score, n)
    -> tuple[float, float, float, float, float]

build_metrics_row(param_identifier, method,
                  AE_true, Bias_AE, SE_true, accuracy_true,
                  Bias_SE, SE_of_SE, accuracy_of_SE,
                  coverage, inconclusive, needed_n, formula_n, skew)
    -> dict
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Module-level constant – kept here so callers can override if needed.
# ---------------------------------------------------------------------------
RF: float = 2.31    # reliability factor (order 0)
EF: float = 1.5     # expansion factor


def sample_size_conservative(BV: float, AE: float, TE: float):
    if TE <= AE * EF:
        return np.nan
    else:
        formula_n = (BV*RF)/(TE-(AE*EF))
        return formula_n 

def sample_size_HH(BV: float, z_score: float, TE: float, std: float, AE: float):
    formula_n = ((z_score*BV*np.mean(std))/(TE-AE))**2 
    return formula_n 



def calculate_n_from_formula(bound_estimator, **kwargs):
    estimators = {
        "HH": sample_size_HH,
        "Poisson_Stringer": sample_size_conservative,
    }

    try:
        estimator = estimators[bound_estimator]
    except KeyError:
        raise ValueError(f"Unknown bound estimator: {bound_estimator}")

    return estimator(**kwargs)