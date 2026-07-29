"""
metrics.py
----------
Compute diagnostic metrics from simulation results.

Public API
----------
other_metrics(results, method, iterations, EE_true, SE_true,
              BV, pop, z_score, n)
    -> tuple[float, float, float, float, float]

build_metrics_row(param_identifier, method,
                  EE_true, Bias_EE, SE_true, accuracy_true,
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


def sample_size_conservative(Q: float, EE: float, TE: float):
    if TE <= EE * EF:
        return np.nan
    else:
        formula_n = (Q*RF)/(TE-(EE*EF))
        return formula_n 

def sample_size_HH(Q: float, z_score: float, TE: float, std: float, EE: float):
    formula_n = ((z_score*Q*np.mean(std))/(TE-EE))**2 
    return formula_n 

# def sample_size_modified_HH(sample_s: pd.DataFrame, EE: float, Qs: float, ns: int, z_score: float):
#     [...]

#     return formula_n

def sample_size_ratio(sample_s: pd.DataFrame, EE: float, Qs: float, ns: int, z_score: float):
    [...]

    return formula_n


def calculate_n_from_formula(bound_estimator, **kwargs):
    estimators = {
        "HH": sample_size_HH,
        "Mod_HH": sample_size_HH, #TODO: needs discussion w supervisor
        "Con": sample_size_conservative,
        "Ratio": sample_size_ratio,
    }

    try:
        estimator = estimators[bound_estimator]
    except KeyError:
        raise ValueError(f"Unknown bound estimator: {bound_estimator}")

    return estimator(**kwargs)