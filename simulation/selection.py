"""
selection.py
------------
Probability-of-inclusion assignment and PPS systematic sample selection.

Public API
----------
prob_inclusion_sample_Stan(population, BV, n) -> pd.DataFrame
    Adds / updates the 'Stan_HV' column on *population* in place and returns it.

selection_type(population, method, SI) -> pd.DataFrame
    Draws a PPS systematic sample from *population* and returns the sampled rows.
"""

import random
import pandas as pd
import numpy as np
    

def systematic_samping(population: pd.DataFrame, SI: float, rng: float) -> pd.DataFrame:
    """
    Draw a PPS systematic sample from *population*.

    Units already flagged as HV (column *method* == 1) are included with
    certainty. The remaining units are sampled using a random start with a
    fixed sampling interval *SI* applied to cumulative book values: the
    marks are ``random_start, random_start + SI, random_start + 2*SI, ...``
    and each mark selects the first remaining item (in *population*'s
    current row order) whose cumulative book value exceeds it. Row order
    therefore affects which items get selected, same as any systematic
    sample -- this is why the caller shuffles the population beforehand.

    Selection is vectorized (cumulative sum + ``np.searchsorted``) instead
    of filtering/concatenating one row at a time; it selects the same items
    as the equivalent per-mark loop for the same random draw.

    Parameters
    ----------
    population : pd.DataFrame
        Population frame; must contain columns 'BV' and *method*.
    SI : float
        Sampling interval.

    Returns
    -------
    pd.DataFrame
        Sampled rows (HV certainty units + PPS-selected units), deduplicated.
    """
    sample = population[population["HV"] == 1].copy()
    remainder = population[population["HV"] != 1]

    cum_sum_BV = remainder["BV"].to_numpy().cumsum()
    total = cum_sum_BV[-1]

    random_dollar = rng.uniform(0.0, SI) #TODO:check

    # Every mark random_dollar + k*SI below total, computed with a generous
    # upper bound on k and then filtered by the same "< total" condition the
    # original while-loop used, so the exact stopping point matches.
    max_marks = int(np.ceil((total - random_dollar) / SI)) + 2
    marks = random_dollar + SI * np.arange(max(max_marks, 0))
    marks = marks[marks < total]

    # First position whose cumulative BV exceeds each mark -- equivalent to
    # `remainder[remainder["cum_sum_BV"] > mark].head(1)` for every mark at
    # once. Marks are increasing, so any item spanning multiple marks
    # produces adjacent repeated positions, which np.unique collapses.
    positions = np.searchsorted(cum_sum_BV, marks, side="right")
    positions = np.unique(positions)

    sampled_remainder = remainder.iloc[positions]
    sample = pd.concat([sample, sampled_remainder], axis=0)
    sample = sample.loc[~sample.index.duplicated()]
    return sample

def python_selection(population: pd.DataFrame, n: int, rng: float) -> pd.DataFrame:
    """
    Perform weighted sampling without replacement.
    Always include rows where HV == 1, then sample the remaining rows
    using BV as weights until the total sample size is ns.
    """

    hv_sample = population[population["HV"] == 1].copy()
    remainder = population[population["HV"] != 1].copy()

    n_to_choose = n - len(hv_sample)

    if n_to_choose > len(remainder):
        raise ValueError("ns is larger than the available population size.")

    BV_sum = remainder["BV"].sum()
    if BV_sum <= 0:
        raise ValueError("BV weights must sum to a positive value.")

    weights = remainder["BV"] * n / BV_sum
    
    # Normalize weights to probabilities (p needs to sum up to 1)
    probabilities = weights / np.sum(weights)

    chosen_indexes = rng.choice(
        remainder.index,
        size=n_to_choose,
        replace=False,
        p=probabilities
    )

    sampled_remainder = remainder.loc[chosen_indexes]

    sample = pd.concat([hv_sample, sampled_remainder])
    sample.drop_duplicates(inplace=True)

    return sample


def select_sample(selection_type, **kwargs):
    """
    Draw a PPS systematic sample from *population*.

    Units already flagged as HV (column *method* == 1) are included with
    certainty. The remaining units are sampled using a random start with a
    fixed sampling interval *SI* applied to cumulative book values.

    Parameters
    ----------
    population : pd.DataFrame
        Population frame; must contain columns 'BV' and *method*.
    SI : float
        Sampling interval.

    Returns
    -------
    pd.DataFrame
        Sampled rows (HV certainty units + PPS-selected units), deduplicated.
    """

    methods = {
        "systematic_sampling": systematic_samping,
        "python": python_selection,
    }

    try:
        method = methods[selection_type]
    except KeyError:
        raise ValueError(f"Unknown bound estimator: {selection_type}")

    return method(**kwargs)