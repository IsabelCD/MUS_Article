"""
simulation.py
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

import pandas as pd
import numpy as np
from tqdm import tqdm

from pathlib import Path
from itertools import product

from sklearn.utils import shuffle
from scipy.stats import norm

from config import RESULTS_DIR
from simulation.sample import Sample
from simulation.sample_size_calculation import calculate_n_from_formula
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

        self.sample_sizes = simulation_config["sample_sizes"]
        self.simulation_configs = simulation_config["configurations"]

        #Population characteristics
        self.N = population.shape[0]
        self.EE = population['E'].sum()
        self.BV = population['BV'].sum()
        self.TE = simulation_config["TE_perc"] * self.BV
        self.ratio_EQ_std = (self.population["E"] / self.population["BV"]).std(ddof=1)

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

        validation_NAs(self.population)
        nr_configs = len(self.simulation_configs)
        combination_idx = 0

        for base_config in self.simulation_configs:
            print(f"Starting configuration {combination_idx+1}/{nr_configs}")
            config = base_config.copy()

            for sample_size, cl in product(self.sample_sizes, self.confidence_levels):
                config["sample_size"] = sample_size

                config["confidence_level"] = cl
                z_score = norm.ppf(cl)
                config["z_score"] = z_score

                iteration_predictions, metrics = self._run_single_combination(config, combination_idx)
                predictions.extend(iteration_predictions)
                all_metrics.append(config | metrics)
                combination_idx += 1

        return (
            pd.DataFrame.from_records(predictions),
            pd.DataFrame.from_records(all_metrics),
        )


    def _run_single_combination(self, config: dict,  config_idx: int) -> tuple[list[dict], dict]:
        combo_predictions = []

        hv_selection = config["hv_selection"]
        selection = config["selection_type"]
        bound = config["bound_estimator"]
        cl = config["confidence_level"]
        z_score = config["z_score"]
        sample_size = config["sample_size"]

        for i in tqdm(range(self.iterations)):
            random_state = (self.seed + config_idx * self.iterations + i)

            shuffled_population = shuffle(self.population, random_state=random_state)

            sample = Sample(
                population=shuffled_population,
                N=self.N,
                EE=self.EE,
                BV=self.BV,
                sample_size=sample_size,
                cl=cl,
                z_score=z_score,
                hv_selection=hv_selection,
                selection_type=selection,
                bound_estimator=bound,
                random_state=random_state,
            ).run()

            prediction = {
                "iteration": i,
                "sample_size": sample_size,
                "confidence_level": cl,
                "hv_selection": hv_selection,
                "selection_type": selection,
                "bound_estimator": bound,
                **sample.get_results(),
            }

            combo_predictions.append(prediction)

        combo_predictions_df = pd.DataFrame.from_records(combo_predictions)

        validation_NAs(combo_predictions_df)

        metrics = self._metrics_for_method(
            combo_predictions_df,
            config,
        )
        
        return combo_predictions, metrics

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _metrics_for_method(self, it_results: pd.DataFrame, config_info: dict) -> dict:
        """Compute and assemble all metrics for one method ('Con' or 'Stan')."""
        Bias_EE, SE_true, accuracy_true = self._error_precision_metrics(it_results, config_info["z_score"])
        Bias_SE, SE_of_SE, accuracy_of_SE = self._precision_of_precision_metrics(it_results, config_info["bound_estimator"], SE_true, config_info["z_score"])
        coverage, inconclusive, rate_of_acceptance, rate_of_rejection, samples_without_errors, needed_n, formula_n, skew = self._other_metrics(it_results, SE_true, config_info)
        BV_true = self.BV
        ER_true = self.EE / BV_true


        return {"Population ID": self.population_ID, 
            "Population Book Value": BV_true,
            "Population Error Amount": self.EE,
            "Population Error Rate": ER_true,
            "Average Error Estimation": it_results["EE_pred"].mean(), 
            "Bias of Error Estimation": Bias_EE,
            "Precision of Error Estimation": SE_true,
            "Accuracy of Error Estimation": accuracy_true,
            "Average Precision Estimation": it_results["SE_pred"].mean(),
            "Bias of Precision Estimation": Bias_SE,
            "Precision of Precision Estimation": SE_of_SE,
            "Accuracy of Precision Estimation": accuracy_of_SE,
            "Coverage": coverage,
            "Inconclusive": inconclusive,
            "Rate of Acceptance": rate_of_acceptance,
            "Rate of Rejection": rate_of_rejection,
            "Samples without Errors": samples_without_errors,
            "Needed n": needed_n,
            "Formula n": formula_n,
            "Skew": skew
            }
            

    
    def _error_precision_metrics(self, 
                                it_results: pd.DataFrame,
                                z_score: float = None
                                ) -> tuple[float, float, float]:
        """
        Evaluate how well the EE estimator tracks the true population error.

        Parameters
        ----------
        results : pd.DataFrame
            Simulation output; must contain a ``EE_pred`` column.

        Returns
        -------
        Bias_EE : float
            Mean EE minus true EE.
        SE_true : float
            z_score × empirical standard deviation of EE (true precision).
        accuracy_true : float
            z_score × √(empirical variance + Bias_EE²)  — root-mean-squared
            accuracy combining bias and precision.
        """
        #True population parameters
        EE_pred = it_results["EE_pred"]

        #MUS article method
        Bias_EE = EE_pred.mean() - self.EE
        SE_true = z_score * np.sqrt(EE_pred.var(ddof=1))

        MSE_error = (SE_true / z_score) ** 2 + Bias_EE ** 2
        accuracy_true = z_score * np.sqrt(MSE_error)

        return Bias_EE, SE_true, accuracy_true


    def _precision_of_precision_metrics(self, 
        it_results: pd.DataFrame,
        bound_estimator: str,
        SE_true: float,
        z_score: float = None
    ) -> tuple[float, float, float]:
        """
        Evaluate how well the SE estimator tracks the true precision.

        For the **Standard** method, the final precision estimate is derived
        from the average variance across iterations
        (``z_score × √mean(Stan_VAR)``), consistent with the variance-based
        estimator used in each sample.

        Parameters
        ----------
        results : pd.DataFrame
            Simulation output; must contain ``{method}_SE`` (and ``Stan_VAR``
            for method='Stan').
        method : str
            'Con' or 'Stan'.
        SE_true : float
            Empirical (true) precision obtained from
            :func:`error_precision_metrics`.

        Returns
        -------
        Bias_SE : float
            Mean estimated SE minus SE_true.
        SE_of_SE : float
            z_score × empirical standard deviation of the SE estimates
            (precision of the precision estimator).
        accuracy_of_SE : float
            z_score × √(variance of SE estimates + Bias_SE²).
        """
        SE_pred = it_results["SE_pred"]

        if bound_estimator == "HH":
            # Standard: use variance-based final estimate
            final_estimated_precision = z_score * np.sqrt(it_results["VAR_pred"].mean())
            Bias_SE = final_estimated_precision - SE_true
            
        else:
            Bias_SE = SE_pred.mean() - SE_true

        # Precision of Precision
        SE_of_SE = z_score * SE_pred.std(ddof=1)

        # Calculate Mean Squared Error (MSE) to then obtain the accuracy of precision estimation
        MSE_precision = SE_pred.var(ddof=1) + Bias_SE ** 2
        accuracy_of_SE = z_score * np.sqrt(MSE_precision)

        return Bias_SE, SE_of_SE, accuracy_of_SE
    

    def _other_metrics(self, 
                       it_results: pd.DataFrame, 
                       SE_true: float, 
                       config_info: dict
                       ) -> tuple[float, float, float, float, float]:
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

        # Needed sample size
        needed_n = (SE_true*np.sqrt(config_info["sample_size"])/(self.TE-self.EE))**2

        if config_info["bound_estimator"] == "Poisson_Stringer":
            formula_n = calculate_n_from_formula(bound_estimator=config_info["bound_estimator"], 
                                                BV=self.BV, 
                                                TE=self.TE, 
                                                AE=self.EE)
        elif config_info["bound_estimator"] == "HH":
            formula_n = calculate_n_from_formula(bound_estimator=config_info["bound_estimator"], 
                                                BV=self.BV, 
                                                z_score=config_info["z_score"], 
                                                TE=self.TE, 
                                                std=self.ratio_EQ_std, 
                                                AE=self.EE)
        else:
            formula_n = np.nan

        # Simple Skew metric
        average_estimated_EE = EE_pred.mean()
        median_estimated_EE = EE_pred.median()
        std_EE = EE_pred.std(ddof=1)

        if np.isclose(std_EE, 0):
            skew = 0.0
        else:
            skew = (3*(average_estimated_EE-median_estimated_EE)) / std_EE

        return coverage, inconclusive, rate_of_acceptance, rate_of_rejection, samples_without_errors, needed_n, formula_n, skew
    

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self) -> Path:
        """
        Write results and metrics to an Excel workbook in config.results_dir.

        Returns
        -------
        Path
            Path to the written file.
        """
        if self.results is None or self.metrics_df is None:
            raise RuntimeError("Call run() before export().")
        
        path = RESULTS_DIR / f"results_{self.population_ID}.xlsx"
        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
            self.results.to_excel(writer, sheet_name="results (€)", index=False)

            group_cols = ["sample_size", "hv_selection", "selection_type", "bound_estimator", "confidence_level"]
            excluded_cols = group_cols + ["iteration"]
            value_cols = self.results.columns.difference(excluded_cols)
            summary = (
                    self.results
                    .groupby(group_cols)[value_cols]
                    .describe()
                    .T
                )
            summary.to_excel(writer, sheet_name="descriptive statistics (€)", index=True)

            self.metrics_df.to_excel(writer, sheet_name="metrics", index=False)

        print(f"Exported → {path}")
        return path
