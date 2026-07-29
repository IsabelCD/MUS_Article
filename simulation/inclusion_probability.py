import numpy as np
import pandas as pd


def iterative_hv_selection(population: pd.DataFrame, Q: float, n: int) -> pd.DataFrame:
    """
    Compute inclusion probabilities and flag high-value (HV) units.

    Units whose inclusion probability exceeds 1 are set to 1 (certainty
    stratum). The remaining probability mass is redistributed iteratively
    until the HV set stabilises.

    Parameters
    ----------
    population : pd.DataFrame
        Population frame with at least a 'BV' column.
    BV : float
        Total book value of the population.
    n : int
        Target sample size.

    Returns
    -------
    pd.DataFrame
        *population* with the 'HV' column added / overwritten.
    """
    population = population.copy()
    population["HV"] = np.where((population["Q"] / Q) * n > 1, 1, (population["Q"] / Q) * n)

    n_hvs = (population["HV"] == 1).sum()
    old_n_hvs = 0

    if n_hvs > 0:
        while (n_hvs - old_n_hvs) != 0:
            Nr = n - n_hvs
            Q_nr = population[population["HV"] != 1]["Q"].sum()

            population["HV"] = np.where(
                population["HV"] == 1,
                1,
                (population["Q"] / Q_nr) * Nr,
            )
            population["HV"] = np.where(population["HV"] > 1, 1, population["HV"])

            old_n_hvs = n_hvs
            n_hvs = (population["HV"] == 1).sum()

    return population


def separate_no_hvs(population: pd.DataFrame) -> pd.DataFrame:

    population["HV"] = 0

    return population


def assign_hv_by_method(hv_selection, **kwargs):
    methods = {
        "nothing": separate_no_hvs,
        "iterative": iterative_hv_selection,
    }

    try:
        method = methods[hv_selection]
    except KeyError:
        raise ValueError(f"Unknown bound estimator: {hv_selection}")

    return method(**kwargs)