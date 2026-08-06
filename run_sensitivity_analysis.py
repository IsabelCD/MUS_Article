import pandas as pd

from config import *
from simulation.sample_planning_analysis import Simulation
from simulation.aggregate_metrics import aggregate_metrics
from create_population.import_population import import_population


def main():
    predictions = pd.DataFrame()
    metrics = pd.DataFrame()
    for bv_nr in POPULATION_CONFIGS["BV_pop"]:
        for f_target in POPULATION_CONFIGS["f_target"]:
            for corr_target in POPULATION_CONFIGS["corr_target"]:
                for r_target in POPULATION_CONFIGS["r_target"]:

                    population_config = {
                        "BV_pop": bv_nr,
                        "f_target": f_target,
                        "corr_target": corr_target,
                        "r_target": r_target
                    }
                    print(f"Running simulation for population config: {population_config}")

                    artificial_pop = import_population(**population_config)

                    population_ID = f"BV{bv_nr}_F{f_target}_C{corr_target}_R{r_target}"
                    simulation = Simulation(
                        population_ID=population_ID,
                        population=artificial_pop,
                        simulation_config=SAMPLE_PLANNING_SIMULATION_SETTINGS,
                    )

                    simulation.run()

                    iteration_results = simulation.results
                    iteration_metrics = simulation.metrics_df
                    iteration_metrics["BV_pop"] = bv_nr
                    iteration_metrics["f_target"] = f_target
                    iteration_metrics["corr_target"] = corr_target
                    iteration_metrics["r_target"] = r_target

                    predictions = pd.concat([predictions, iteration_results], ignore_index=True)
                    metrics = pd.concat([metrics, iteration_metrics], ignore_index=True)

    path = RESULTS_DIR / "sensitivity_analysis_results.xlsx"
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        # predictions.to_excel(writer, sheet_name="results (€)", index=False)

        metrics.to_excel(writer, sheet_name="metrics", index=False)

        aggregate_metrics(metrics, "sensitivity").to_excel(writer, sheet_name="aggregate_metrics", index=False)

    print(f"Exported → {path}")


if __name__ == "__main__":
    main()