"""
sample_planning_analysis.py
-------------
Orchestrates the Monte Carlo simulation by coordinating the four
domain modules:

    selection           → assign HVs, draw samples
    error_estimation    → project EE, SE, VAR for each sample
    precision_estimation → summarise bias/precision across iterations
    metrics             → coverage, inconclusiveness, skew, sample-size rules

Entry point
-----------
    run_simulation(n, CL, pop_nr)
        Runs `iterations` Monte Carlo draws for one (n, CL, population)
        combination and writes an Excel workbook to ./results/.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from pathlib import Path

from sklearn.utils import shuffle
from scipy.stats import norm

from config import RESULTS_DIR
from simulation.sample import Sample
from simulation.inclusion_probability import iterative_hv_selection
from simulation.sample_size_calculation import calculate_n_from_formula, EF
from simulation.validations import validation_NAs


# ---------------------------------------------------------------------------
# Simulation class
# ---------------------------------------------------------------------------

class Simulation:
    """
    One Monte Carlo simulation for a single (n, CL, pop_nr) combination.

    Attributes set after __init__
    -----------------------------
    label : str           e.g. "n100_z0.8_pop18"
    z_score : float       normal quantile for CL
    population : DataFrame  population frame (BV, E, ER, Con_HV, Stan_HV)
    BV : float            total book value
    EE_true : float       true population error
    Con_SI, Stan_SI : float  sampling intervals
    Stan_BVs : float      non-certainty stratum book value (Standard)
    Stan_ns : int         non-certainty stratum sample size (Standard)

    Attributes set after run()
    --------------------------
    results : DataFrame   one row per iteration
    metrics_df : DataFrame  summary metrics for Con and Stan
    """

    def __init__(self, 
                 population_ID: str,
                 population: pd.DataFrame, 
                 simulation_config: dict,
                 ) -> None:
        
        #Simulation parameters
        self.population_ID=population_ID
        self.population=population
        self.confidence_levels= simulation_config["CL"]

        self.simulation_configs = simulation_config["configurations"]
        self.anticipated_errors = simulation_config["anticipated_errors"]
        self.anticipated_stds = simulation_config["anticipated_stds"]
        self.sample_size_combinations = simulation_config["sample_size_combinations"]

        #Population characteristics
        self.N = population.shape[0]
        self.EE = population['E'].sum()
        self.BV = population['BV'].sum()
        self.TE = simulation_config["TE_perc"] * self.BV

        self.seed = simulation_config["seed"]
        self.iterations = simulation_config["iterations"]

        # Populated by run()
        self.results:    pd.DataFrame | None = None
        self.metrics_df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Execute all Monte Carlo iterations and compute summary metrics.
        Populates self.results and self.metrics_df.
        """
        self.results, self.metrics_df = self._run_iterations()
        print(f"Finished simulation {self.population_ID}")


    def _run_iterations(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        predictions = []
        all_metrics = []
        ratio_EQ_std = (self.population["E"] / self.population["BV"]).std(ddof=1)

        validation_NAs(self.population)
        nr_configs = len(self.simulation_configs)
        combination_idx = 0

        for base_config in self.simulation_configs:
            print(f"Starting configuration {combination_idx+1}/{nr_configs}")
            config = base_config.copy()

            for cl in self.confidence_levels:
                z_score = norm.ppf(cl)
                config["confidence_level"] = cl
                config["z_score"] = z_score

                for anticipated_error in self.anticipated_errors:
                    config["anticipated_error_perc"] = anticipated_error
                    config["anticipated_std"] = 1
                    if config["bound_estimator"] == "Poisson_Stringer":
                        sample_size = calculate_n_from_formula(bound_estimator=config["bound_estimator"], 
                                                                BV=self.BV, 
                                                                TE=self.TE, 
                                                                AE=anticipated_error*self.EE)
                    elif config["bound_estimator"] == "HH":
                        sample_size = calculate_n_from_formula(bound_estimator=config["bound_estimator"],
                                                               BV=self.BV,
                                                               z_score=z_score,
                                                               TE=self.TE,
                                                               std=ratio_EQ_std,
                                                               AE=anticipated_error*self.EE)

                    reason = self._infeasible_reason(config["bound_estimator"], config["hv_selection"],
                                                      sample_size, self.TE, anticipated_error * self.EE)
                    if reason:
                        print(f"Skipping configuration {combination_idx} "
                              f"({config['bound_estimator']}, anticipated_error_perc={anticipated_error}, "
                              f"CL={cl}): {reason}")
                        all_metrics.append(config | self._infeasible_metrics_row(reason))
                        combination_idx += 1
                        continue

                    iteration_predictions, metrics = self._run_single_combination(config, sample_size, combination_idx)
                    predictions.extend(iteration_predictions)
                    all_metrics.append(config | metrics)
                    combination_idx += 1

                if config["bound_estimator"] == "HH":
                    for anticipated_std in self.anticipated_stds:
                        config["anticipated_error_perc"] = 1
                        config["anticipated_std"] = anticipated_std
                        sample_size = calculate_n_from_formula(bound_estimator=config["bound_estimator"],
                                                               BV=self.BV,
                                                               z_score=z_score,
                                                               TE=self.TE,
                                                               std=anticipated_std*ratio_EQ_std,
                                                               AE=self.EE)

                        reason = self._infeasible_reason(config["bound_estimator"], config["hv_selection"],
                                                          sample_size, self.TE, self.EE)
                        if reason:
                            print(f"Skipping configuration {combination_idx} "
                                  f"({config['bound_estimator']}, anticipated_std={anticipated_std}, "
                                  f"CL={cl}): {reason}")
                            all_metrics.append(config | self._infeasible_metrics_row(reason))
                            combination_idx += 1
                            continue

                        iteration_predictions, metrics = self._run_single_combination(config, sample_size, combination_idx)
                        predictions.extend(iteration_predictions)
                        all_metrics.append(config | metrics)
                        combination_idx += 1

                    for ss_config in self.sample_size_combinations:
                        config["anticipated_error_perc"] = ss_config["anticipated_error_perc"]
                        config["anticipated_std"] = ss_config["anticipated_std"]
                        sample_size = calculate_n_from_formula(bound_estimator=config["bound_estimator"],
                                                               BV=self.BV,
                                                               z_score=z_score,
                                                               TE=self.TE,
                                                               std=ss_config["anticipated_std"]*ratio_EQ_std,
                                                               AE=ss_config["anticipated_error_perc"]*self.EE)

                        reason = self._infeasible_reason(config["bound_estimator"], config["hv_selection"],
                                                          sample_size, self.TE,
                                                          ss_config["anticipated_error_perc"] * self.EE)
                        if reason:
                            print(f"Skipping configuration {combination_idx} "
                                  f"({config['bound_estimator']}, {ss_config}, CL={cl}): {reason}")
                            all_metrics.append(config | self._infeasible_metrics_row(reason))
                            combination_idx += 1
                            continue

                        iteration_predictions, metrics = self._run_single_combination(config, sample_size, combination_idx)
                        predictions.extend(iteration_predictions)
                        all_metrics.append(config | metrics)
                        combination_idx += 1

        return (
            pd.DataFrame.from_records(predictions),
            pd.DataFrame.from_records(all_metrics),
        )


    def _run_single_combination(self, config: dict, sample_size: int, config_idx: int) -> tuple[list[dict], dict]:
        combo_predictions = []

        hv_selection = config["hv_selection"]
        selection = config["selection_type"]
        bound = config["bound_estimator"]
        cl = config["confidence_level"]
        z_score = config["z_score"]

        # Certainty-unit (HV) assignment depends only on BV/sample_size, not
        # on the per-iteration shuffle order, so compute it once per combo
        # instead of recomputing it from scratch on every iteration.
        hv_lookup = None
        if hv_selection == "iterative":
            hv_lookup = iterative_hv_selection(self.population, self.BV, sample_size)["HV"]

        for i in tqdm(range(self.iterations)):
            random_state = (self.seed + config_idx * self.iterations + i)

            shuffled_population = shuffle(self.population, random_state=random_state)

            sample = Sample(
                population=shuffled_population,
                N=self.N,
                EE=self.EE,
                BV=self.BV,
                sample_size=sample_size,
                z_score=z_score,
                cl=cl,
                hv_selection=hv_selection,
                selection_type=selection,
                bound_estimator=bound,
                random_state=random_state,
                hv_lookup=hv_lookup,
            ).run()


            prediction = {
                "iteration": i,
                "population_ID": self.population_ID,
                "sample_size": sample_size,
                "confidence_level": cl,
                "hv_selection": hv_selection,
                "selection_type": selection,
                "bound_estimator": bound,
                "anticipated_error_perc": config["anticipated_error_perc"],
                "anticipated_std": config["anticipated_std"],
                **sample.get_results(),
            }

            combo_predictions.append(prediction)

        combo_predictions_df = pd.DataFrame.from_records(combo_predictions)

        validation_NAs(combo_predictions_df)

        metrics = self._metrics_for_method(combo_predictions_df)

        return combo_predictions, metrics
    
    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _metrics_for_method(self, it_results: pd.DataFrame) -> dict:
        """Compute and assemble all metrics for one combination"""
        #True population parameters
        EE_pred = it_results["EE_pred"]

        # Coverage
        coverage = sum(it_results['ULE_pred']>=self.EE)/self.iterations

        # Inconclusive
        inconclusive = sum((it_results['ULE_pred'] > self.TE) & (EE_pred < self.TE)) / self.iterations

        rate_of_acceptance = sum(it_results["ULE_pred"] <= self.TE) / self.iterations
        rate_of_rejection = sum(EE_pred > self.TE) / self.iterations

        #Number of samples without errors 
        samples_without_errors = sum(it_results["number_errors"] == 0) / self.iterations

        BV_true = self.BV
        ER_true = self.EE / BV_true


        return {"Population ID": self.population_ID,
            "Population Book Value": BV_true,
            "Population Error Amount": self.EE,
            "Population Error Rate": ER_true,
            "Coverage": coverage,
            "Inconclusive": inconclusive,
            "Samples without Errors": samples_without_errors,
            "Rate of Acceptance": rate_of_acceptance,
            "Rate of Rejection": rate_of_rejection,
            "Average Error Estimation": it_results["EE_pred"].mean(),
            "Average Precision Estimation": it_results["SE_pred"].mean(),
            "obs": None,
            }

    def _infeasible_reason(
        self, bound_estimator: str, hv_selection: str, sample_size: float, TE: float, AE: float
    ) -> str | None:
        """
        Return a human-readable reason if `sample_size` (from an analytical
        sample-size formula) cannot be used to run this configuration, or
        None if it is usable.

        Two known failure modes are guarded here:
        - Poisson_Stringer's conservative formula is undefined (returns NaN)
          whenever TE <= AE * EF; this branch also covers any other formula
          that returns a non-finite or non-positive value.
        - With hv_selection="iterative", a sample_size at or above the
          population size drives every unit into the certainty stratum,
          leaving none to draw a systematic sample from; the non-certainty
          stratum sample then has no capacity for the units the certainty
          allocation didn't already claim, and selection.systematic_samping
          fails outright on the empty remainder.
        """
        if not (np.isfinite(sample_size) and sample_size > 0):
            if bound_estimator == "Poisson_Stringer":
                return (
                    f"Poisson_Stringer's conservative sample-size formula is undefined here: "
                    f"it requires TE > AE * EF (EF={EF}), but TE={TE:,.2f} and AE*EF={AE * EF:,.2f}."
                )
            return (
                f"{bound_estimator}'s analytical sample-size formula returned a non-finite or "
                f"non-positive value ({sample_size})."
            )

        if hv_selection == "iterative" and sample_size >= self.N:
            return (
                f"{bound_estimator}'s analytical sample size ({sample_size:,.0f}) is not smaller "
                f"than the population size (N={self.N}); iterative certainty-unit allocation "
                f"would leave no non-certainty units to draw a systematic sample from."
            )

        return None

    def _infeasible_metrics_row(self, reason: str) -> dict:
        """Metrics row for a configuration that could not be run (see _infeasible_reason)."""
        BV_true = self.BV
        ER_true = self.EE / BV_true
        return {
            "Population ID": self.population_ID,
            "Population Book Value": BV_true,
            "Population Error Amount": self.EE,
            "Population Error Rate": ER_true,
            "Coverage": np.nan,
            "Inconclusive": np.nan,
            "Samples without Errors": np.nan,
            "Rate of Acceptance": np.nan,
            "Rate of Rejection": np.nan,
            "Average Error Estimation": np.nan,
            "Average Precision Estimation": np.nan,
            "obs": reason,
        }

