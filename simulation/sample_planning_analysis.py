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

import pandas as pd
from tqdm import tqdm

from pathlib import Path

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
        self.confidence_levels= simulation_config["confidence_levels"]

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
        ratio_EQ_std = self.population['E'].std() / self.population['E'].mean()

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

        BV_true = self.BV
        ER_true = self.EE / BV_true


        return {"Population ID": self.population_ID, 
            "Population Book Value": BV_true,
            "Population Error Amount": self.EE,
            "Population Error Rate": ER_true,
            "Coverage": coverage,
            "Inconclusive": inconclusive,
            "Rate of Acceptance": rate_of_acceptance,
            "Rate of Rejection": rate_of_rejection}
    

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
        
        path = RESULTS_DIR / f"resultados_{self.population_ID}.xlsx"
        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
            self.results.to_excel(writer, sheet_name="resultados (€)", index=False)

            group_cols = ["sample_size", "anticipated_error_perc", "anticipated_std", 
                          "hv_selection", "selection_type", "bound_estimator", "confidence_level"]
            excluded_cols = group_cols + ["iteration", "z_score"]
            value_cols = self.results.columns.difference(excluded_cols)
            summary = (
                    self.results
                    .groupby(group_cols)[value_cols]
                    .describe()
                    .T
                )
            summary.to_excel(writer, sheet_name="estatisticas descritivas (€)", index=True)

            self.metrics_df.to_excel(writer, sheet_name="métricas", index=False)

        print(f"Exported → {path}")
        return path


