from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from config import *
from simulation.sample_planning_analysis import Simulation
from simulation.aggregate_metrics import aggregate_metrics
from create_population.import_population import import_population


def _population_id(population_config: dict) -> str:
    return (
        f"BV{population_config['BV_pop']}_F{population_config['f_target']}"
        f"_C{population_config['corr_target']}_R{population_config['r_target']}"
    )


def _run_one_population(population_config: dict, population: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full SAMPLE_PLANNING_SIMULATION_SETTINGS grid for one
    already-loaded population and return its metrics (tagged with the
    population config columns). Runs in its own process. Population loading
    happens in the parent (see main()) so worker processes never touch the
    source Excel file. `simulation.results` (iteration-level predictions) is
    not returned: it is never written to the combined output workbook (see
    the commented-out `predictions.to_excel` line this replaced), so
    shipping it back across the process boundary would only add pickling
    cost for no observable benefit.
    """
    simulation = Simulation(
        population_ID=_population_id(population_config),
        population=population,
        simulation_config=SAMPLE_PLANNING_SIMULATION_SETTINGS,
    )
    simulation.run()

    iteration_metrics = simulation.metrics_df
    iteration_metrics["BV_pop"] = population_config["BV_pop"]
    iteration_metrics["f_target"] = population_config["f_target"]
    iteration_metrics["corr_target"] = population_config["corr_target"]
    iteration_metrics["r_target"] = population_config["r_target"]
    return iteration_metrics


def main(max_workers: int | None = None):
    """
    Run every population config in POPULATION_CONFIGS in parallel, one
    process per population, then combine all of their metrics into a single
    workbook exactly as the sequential version did. `max_workers` defaults
    to os.cpu_count() (all logical cores).

    Populations are loaded here, in the parent process, before any worker is
    dispatched -- see main.py's main() for why.
    """
    population_configs = [
        {
            "BV_pop": bv_nr,
            "f_target": f_target,
            "corr_target": corr_target,
            "r_target": r_target,
        }
        for bv_nr in POPULATION_CONFIGS["BV_pop"]
        for f_target in POPULATION_CONFIGS["f_target"]
        for corr_target in POPULATION_CONFIGS["corr_target"]
        for r_target in POPULATION_CONFIGS["r_target"]
    ]

    print(f"Loading {len(population_configs)} populations...")
    populations = [import_population(**cfg) for cfg in population_configs]

    metrics_frames = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_one_population, cfg, pop): cfg
            for cfg, pop in zip(population_configs, populations)
        }
        for future in as_completed(futures):
            cfg = futures[future]
            try:
                metrics_frames.append(future.result())
                print(f"Finished {_population_id(cfg)}")
            except Exception:
                print(f"Population config {cfg} FAILED:")
                raise

    metrics = pd.concat(metrics_frames, ignore_index=True)

    path = RESULTS_DIR / "sensitivity_analysis_results.xlsx"
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        metrics.to_excel(writer, sheet_name="metrics", index=False)

        aggregate_metrics(metrics, "sensitivity").to_excel(writer, sheet_name="aggregate_metrics", index=True)

    print(f"Exported -> {path}")


if __name__ == "__main__":
    main()