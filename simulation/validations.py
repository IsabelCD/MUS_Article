"""Reusable validation helpers for population and simulation data.

The functions in this module deliberately have no side effects.  They raise a
``TypeError`` or ``ValueError`` with a focused message when an invariant is
violated and otherwise return ``None``.  Production callers can adopt them
incrementally; defining a validation here does not automatically activate it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral, Real

import numpy as np
import pandas as pd


POPULATION_BASE_COLUMNS = ("BV", "E", "ER")
SIMULATION_RESULT_COLUMNS = (
    "EE_pred",
    "SE_pred",
    "VAR_pred",
    "LLE_pred",
    "ULE_pred",
    "real_n"
)

ALLOWED_METHODS = frozenset({"MUS", "MRS"})
ALLOWED_HV_SELECTIONS = frozenset({"nothing", "iterative"})
ALLOWED_SELECTION_TYPES = frozenset({"systematic_sampling", "python"})
ALLOWED_BOUND_ESTIMATORS = frozenset({"Con", "HH", "Mod_HH", "Ratio"})


def _require_dataframe(df: pd.DataFrame, frame_name: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{frame_name} must be a pandas DataFrame.")


def _require_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite.")
    return numeric_value


def validation_non_empty(df: pd.DataFrame, frame_name: str = "DataFrame") -> None:
    """Require a DataFrame with at least one row."""
    _require_dataframe(df, frame_name)
    if df.empty:
        raise ValueError(f"{frame_name} must contain at least one row.")


def validation_required_columns(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    frame_name: str = "DataFrame",
) -> None:
    """Require all named columns to be present."""
    _require_dataframe(df, frame_name)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}.")


def validation_NAs(df: pd.DataFrame, frame_name: str = "DataFrame") -> None:
    """Reject missing values and report only columns that contain them."""
    _require_dataframe(df, frame_name)
    bad_counts = df.isna().sum()
    bad_counts = bad_counts[bad_counts > 0]
    if not bad_counts.empty:
        raise ValueError(
            f"Missing values detected in {frame_name}:\n{bad_counts.to_string()}"
        )


def validation_numeric_finite(
    df: pd.DataFrame,
    columns: Sequence[str],
    frame_name: str = "DataFrame",
) -> None:
    """Require selected columns to be numeric and contain only finite values."""
    validation_required_columns(df, columns, frame_name)

    non_numeric = [
        column
        for column in columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric:
        raise TypeError(f"{frame_name} columns must be numeric: {non_numeric}.")

    values = df.loc[:, list(columns)].to_numpy(dtype=float, copy=False)
    if not np.isfinite(values).all():
        invalid_counts = {
            column: int((~np.isfinite(df[column].to_numpy(dtype=float))).sum())
            for column in columns
        }
        invalid_counts = {
            column: count for column, count in invalid_counts.items() if count
        }
        raise ValueError(
            f"Non-finite values detected in {frame_name}: {invalid_counts}."
        )


def validation_population(
    population: pd.DataFrame,
    method: str | None = None,
    *,
    check_error_consistency: bool = False,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> None:
    """Validate the population schema and core monetary/risk invariants.

    Parameters
    ----------
    population:
        Population containing ``BV``, ``E`` and ``ER``.  MRS additionally
        requires ``P``.
    method:
        Optional ``"MUS"`` or ``"MRS"``.  When omitted, ``P`` is validated if
        it is present but is not required.
    check_error_consistency:
        If true, require ``E`` to be numerically equal to ``BV * ER``.
    """
    validation_non_empty(population, "population")

    if method is not None and method not in ALLOWED_METHODS:
        raise ValueError(
            f"Unknown sampling method {method!r}; expected one of "
            f"{sorted(ALLOWED_METHODS)}."
        )

    required_columns = list(POPULATION_BASE_COLUMNS)
    if method == "MRS":
        required_columns.append("P")

    validation_required_columns(population, required_columns, "population")

    numeric_columns = list(required_columns)
    if "P" in population.columns and "P" not in numeric_columns:
        numeric_columns.append("P")
    validation_numeric_finite(population, numeric_columns, "population")
    validation_NAs(population.loc[:, numeric_columns], "population")

    if (population["BV"] <= 0).any():
        raise ValueError("population BV must be strictly positive.")
    if (population["E"] < 0).any():
        raise ValueError("population E must not be negative.")
    if ((population["ER"] < 0) | (population["ER"] > 1)).any():
        raise ValueError("population ER must lie in the interval [0, 1].")

    if "P" in numeric_columns:
        lower_bound = 0 if method != "MRS" else np.nextafter(0.0, 1.0)
        if ((population["P"] < lower_bound) | (population["P"] > 1)).any():
            if method == "MRS":
                raise ValueError("MRS population P must lie in the interval (0, 1].")
            raise ValueError("population P must lie in the interval [0, 1].")

    if check_error_consistency:
        expected_error = population["BV"] * population["ER"]
        if not np.allclose(
            population["E"].to_numpy(dtype=float),
            expected_error.to_numpy(dtype=float),
            rtol=rtol,
            atol=atol,
        ):
            max_difference = float((population["E"] - expected_error).abs().max())
            raise ValueError(
                "population E is inconsistent with BV * ER; "
                f"maximum absolute difference is {max_difference}."
            )


def validation_Q_distribution(population: pd.DataFrame) -> None:
    """Require a finite, strictly positive measure of size for every unit."""
    validation_non_empty(population, "population")
    validation_numeric_finite(population, ["Q"], "population")
    validation_NAs(population[["Q"]], "population Q")

    if (population["Q"] <= 0).any():
        raise ValueError("Q must not contain negative or zero values.")
    if not np.isfinite(float(population["Q"].sum())) or population["Q"].sum() <= 0:
        raise ValueError("Total Q must be finite and positive.")


def validation_simulation_config(
    simulation_config: Mapping[str, object],
    *,
    population_size: int | None = None,
    implemented_estimators: set[str] | frozenset[str] | None = None,
) -> None:
    """Validate simulation settings and configuration compatibility.

    ``implemented_estimators`` is optional so callers can distinguish accepted
    names from estimators that are ready in a particular version of the code.
    """
    if not isinstance(simulation_config, Mapping):
        raise TypeError("simulation_config must be a mapping.")

    required_settings = {
        "sample_sizes",
        "CL",
        "iterations",
        "seed",
        "TE_perc",
        "configurations",
    }
    missing_settings = sorted(required_settings.difference(simulation_config))
    if missing_settings:
        raise ValueError(
            f"simulation_config is missing required settings: {missing_settings}."
        )

    cl = _require_real(simulation_config["CL"], "CL")
    if not 0 < cl < 1:
        raise ValueError("CL must lie strictly between 0 and 1.")

    te_percentage = _require_real(simulation_config["TE_perc"], "TE_perc")
    if te_percentage <= 0:
        raise ValueError("TE_perc must be strictly positive.")

    iterations = simulation_config["iterations"]
    if isinstance(iterations, bool) or not isinstance(iterations, Integral):
        raise TypeError("iterations must be an integer.")
    if iterations < 2:
        raise ValueError("iterations must be at least 2 for variance metrics.")

    seed = simulation_config["seed"]
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer.")

    if population_size is not None:
        if isinstance(population_size, bool) or not isinstance(population_size, Integral):
            raise TypeError("population_size must be an integer.")
        if population_size <= 0:
            raise ValueError("population_size must be strictly positive.")

    sample_sizes = simulation_config["sample_sizes"]
    if (
        not isinstance(sample_sizes, Sequence)
        or isinstance(sample_sizes, (str, bytes))
        or not sample_sizes
    ):
        raise ValueError("sample_sizes must be a non-empty sequence.")

    normalized_sizes: list[int] = []
    for sample_size in sample_sizes:
        if isinstance(sample_size, bool) or not isinstance(sample_size, Integral):
            raise TypeError("Every sample size must be an integer.")
        if sample_size <= 0:
            raise ValueError("Every sample size must be strictly positive.")
        if population_size is not None and sample_size > population_size:
            raise ValueError(
                f"Sample size {sample_size} exceeds population size {population_size}."
            )
        normalized_sizes.append(int(sample_size))

    if len(set(normalized_sizes)) != len(normalized_sizes):
        raise ValueError("sample_sizes must not contain duplicates.")

    configurations = simulation_config["configurations"]
    if (
        not isinstance(configurations, Sequence)
        or isinstance(configurations, (str, bytes))
        or not configurations
    ):
        raise ValueError("configurations must be a non-empty sequence.")

    required_configuration_keys = {
        "method",
        "hv_selection",
        "selection_type",
        "bound_estimator",
    }
    seen_configurations: set[tuple[object, ...]] = set()

    for index, configuration in enumerate(configurations):
        if not isinstance(configuration, Mapping):
            raise TypeError(f"Configuration {index} must be a mapping.")

        missing_keys = sorted(required_configuration_keys.difference(configuration))
        if missing_keys:
            raise ValueError(
                f"Configuration {index} is missing required keys: {missing_keys}."
            )

        method = configuration["method"]
        hv_selection = configuration["hv_selection"]
        selection_type = configuration["selection_type"]
        bound_estimator = configuration["bound_estimator"]

        allowed_values = (
            ("method", method, ALLOWED_METHODS),
            ("hv_selection", hv_selection, ALLOWED_HV_SELECTIONS),
            ("selection_type", selection_type, ALLOWED_SELECTION_TYPES),
            ("bound_estimator", bound_estimator, ALLOWED_BOUND_ESTIMATORS),
        )
        for field, value, allowed in allowed_values:
            if value not in allowed:
                raise ValueError(
                    f"Configuration {index} has invalid {field}={value!r}; "
                    f"expected one of {sorted(allowed)}."
                )

        if bound_estimator == "Con" and method != "MUS":
            raise ValueError(
                f"Configuration {index} is invalid: Con is only compatible with MUS."
            )

        if implemented_estimators is not None and bound_estimator not in implemented_estimators:
            raise ValueError(
                f"Configuration {index} selects estimator {bound_estimator!r}, "
                "which is not marked as implemented."
            )

        identity = (method, hv_selection, selection_type, bound_estimator)
        if identity in seen_configurations:
            raise ValueError(f"Configuration {index} duplicates an earlier configuration.")
        seen_configurations.add(identity)


def validation_sample_design(
    population: pd.DataFrame,
    sample_size: int,
    *,
    hv_selection: str,
    SI: float | None = None,
    tolerance: float = 1e-10,
) -> None:
    """Validate HV allocation and the derived systematic sampling interval."""
    validation_non_empty(population, "population")
    validation_required_columns(population, ["Q", "HV"], "population")
    validation_numeric_finite(population, ["Q", "HV"], "population")

    if isinstance(sample_size, bool) or not isinstance(sample_size, Integral):
        raise TypeError("sample_size must be an integer.")
    if sample_size <= 0 or sample_size > len(population):
        raise ValueError("sample_size must lie between 1 and population size.")
    if hv_selection not in ALLOWED_HV_SELECTIONS:
        raise ValueError(f"Unknown HV selection method: {hv_selection!r}.")
    if ((population["HV"] < 0) | (population["HV"] > 1)).any():
        raise ValueError("HV values must lie in the interval [0, 1].")

    certainty_count = int((population["HV"] == 1).sum())
    if certainty_count > sample_size:
        raise ValueError("The number of certainty units exceeds sample_size.")

    sampling_size = sample_size - certainty_count
    remainder = population[population["HV"] != 1]
    remainder_q = float(remainder["Q"].sum())

    if sampling_size == 0:
        if not remainder.empty:
            raise ValueError("No sample slots remain for non-certainty units.")
        return
    if remainder.empty or remainder_q <= 0:
        raise ValueError("The non-certainty stratum must have positive total Q.")

    expected_si = remainder_q / sampling_size
    if SI is not None:
        supplied_si = _require_real(SI, "SI")
        if supplied_si <= 0:
            raise ValueError("SI must be strictly positive.")
        if not np.isclose(supplied_si, expected_si, rtol=tolerance, atol=tolerance):
            raise ValueError(
                f"SI={supplied_si} is inconsistent with Qs/ns={expected_si}."
            )

    if hv_selection == "iterative" and (remainder["Q"] > expected_si + tolerance).any():
        raise ValueError(
            "Iterative HV allocation left a non-certainty unit with Q greater than SI."
        )


def validation_selected_sample(
    sample: pd.DataFrame,
    population: pd.DataFrame,
    *,
    expected_size: int | None = None,
    allow_duplicate_indices: bool = False,
) -> None:
    """Validate that a selected sample is non-empty and belongs to its frame."""
    validation_non_empty(sample, "sample")
    validation_non_empty(population, "population")
    validation_required_columns(sample, ["Q", "HV"], "sample")
    validation_NAs(sample, "sample")

    if expected_size is not None:
        if isinstance(expected_size, bool) or not isinstance(expected_size, Integral):
            raise TypeError("expected_size must be an integer.")
        if len(sample) != expected_size:
            raise ValueError(
                f"Selected sample contains {len(sample)} rows; expected {expected_size}."
            )

    if not allow_duplicate_indices and sample.index.duplicated().any():
        raise ValueError("Selected sample contains duplicate population indices.")
    if not sample.index.isin(population.index).all():
        raise ValueError("Selected sample contains indices outside the population.")


def validation_precision_inputs(
    sample_s: pd.DataFrame,
    *,
    Qs: float,
    ns: int,
    z_score: float,
) -> None:
    """Validate inputs needed by HH-family variance estimators."""
    validation_non_empty(sample_s, "non-certainty sample")
    validation_required_columns(sample_s, ["E", "Q"], "non-certainty sample")
    validation_numeric_finite(sample_s, ["E", "Q"], "non-certainty sample")

    if isinstance(ns, bool) or not isinstance(ns, Integral):
        raise TypeError("ns must be an integer.")
    if ns < 2:
        raise ValueError("ns must be at least 2 for a ddof=1 variance estimate.")
    if len(sample_s) != ns:
        raise ValueError(
            f"ns={ns} does not match non-certainty sample length {len(sample_s)}."
        )

    q_total = _require_real(Qs, "Qs")
    if q_total <= 0:
        raise ValueError("Qs must be strictly positive.")
    if (sample_s["Q"] <= 0).any():
        raise ValueError("Sample Q values must be strictly positive.")

    critical_value = _require_real(z_score, "z_score")
    if critical_value <= 0:
        raise ValueError("z_score must be strictly positive.")


def validation_simulation_results(
    results: pd.DataFrame,
    *,
    expected_iterations: int | None = None,
    check_bound_identity: bool = True,
    tolerance: float = 1e-8,
) -> None:
    """Validate iteration-level estimates before aggregation or export."""
    validation_non_empty(results, "simulation results")
    validation_required_columns(results, SIMULATION_RESULT_COLUMNS, "simulation results")
    validation_numeric_finite(results, SIMULATION_RESULT_COLUMNS, "simulation results")
    validation_NAs(results.loc[:, list(SIMULATION_RESULT_COLUMNS)], "simulation results")

    if expected_iterations is not None:
        if (
            isinstance(expected_iterations, bool)
            or not isinstance(expected_iterations, Integral)
            or expected_iterations <= 0
        ):
            raise ValueError("expected_iterations must be a positive integer.")
        if len(results) != expected_iterations:
            raise ValueError(
                f"Simulation produced {len(results)} rows; expected {expected_iterations}."
            )

    if (results["SE_pred"] < 0).any():
        raise ValueError("SE_pred must not be negative.")
    if (results["VAR_pred"] < 0).any():
        raise ValueError("VAR_pred must not be negative.")
    if (results["LLE_pred"] > results["ULE_pred"]).any():
        raise ValueError("Every lower limit must be less than or equal to its upper limit.")

    if check_bound_identity:
        expected_lower = results["EE_pred"] - results["SE_pred"]
        expected_upper = results["EE_pred"] + results["SE_pred"]
        if not np.allclose(results["LLE_pred"], expected_lower, atol=tolerance, rtol=tolerance):
            raise ValueError("LLE_pred is inconsistent with EE_pred - SE_pred.")
        if not np.allclose(results["ULE_pred"], expected_upper, atol=tolerance, rtol=tolerance):
            raise ValueError("ULE_pred is inconsistent with EE_pred + SE_pred.")


def validation_upper_coverage_inputs(results: pd.DataFrame, true_error: float) -> None:
    """Validate the inputs for upper-limit-only coverage calculation."""
    validation_non_empty(results, "simulation results")
    validation_numeric_finite(results, ["ULE_pred"], "simulation results")
    _require_real(true_error, "true_error")


def validation_tolerable_error(
    tolerable_error: float,
    expected_error: float,
    *,
    expansion_factor: float | None = None,
) -> None:
    """Validate denominators used by analytical sample-size formulas."""
    te = _require_real(tolerable_error, "tolerable_error")
    ee = _require_real(expected_error, "expected_error")
    if te <= 0 or ee < 0:
        raise ValueError("tolerable_error must be positive and expected_error non-negative.")

    if expansion_factor is None:
        if np.isclose(te, ee):
            raise ValueError(
                "tolerable_error and expected_error must differ for the HH formula."
            )
        return

    factor = _require_real(expansion_factor, "expansion_factor")
    if factor <= 0:
        raise ValueError("expansion_factor must be strictly positive.")
    if te <= ee * factor:
        raise ValueError(
            "No finite conservative sample size exists when tolerable_error "
            "is not greater than expected_error * expansion_factor."
        )
