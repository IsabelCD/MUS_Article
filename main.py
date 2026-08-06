import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

import xlsxwriter
from config import *
from simulation.run_sim import Simulation
from create_population.import_population import import_population


def _population_id(population_config: dict) -> str:
    return (
        f"BV{population_config['BV_pop']}_F{population_config['f_target']}"
        f"_C{population_config['corr_target']}_R{population_config['r_target']}"
    )


def _run_one_population(population_config: dict, population: pd.DataFrame) -> str:
    """
    Run and export the full SIMULATION_SETTINGS grid for one already-loaded
    population. Runs in its own process; writes its own results file, so
    workers never share state or output files with each other. Population
    loading happens in the parent (see main()) so worker processes never
    touch the source Excel file.
    """
    simulation = Simulation(
        population_ID=_population_id(population_config),
        population=population,
        simulation_config=SIMULATION_SETTINGS,
    )

    simulation.run()
    return simulation.export()


def main(max_workers: int | None = None):
    """
    Run every population config in POPULATION_CONFIGS in parallel, one
    process per population. Each population's SIMULATION_SETTINGS grid
    (sample sizes x confidence levels x configurations) still runs
    sequentially within its own process -- only the outer population loop
    is parallelized. `max_workers` defaults to os.cpu_count() (all logical
    cores); pass a smaller number to leave headroom for other work.

    Populations are loaded here, in the parent process, before any worker
    is dispatched: every population config sharing a BV_pop variant reads
    the same source sheet, and import_population() caches that read, so
    loading all of them here costs one real read per BV_pop variant (3
    total) instead of one per worker process that happens to touch it.
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

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_one_population, cfg, pop): cfg
            for cfg, pop in zip(population_configs, populations)
        }
        for future in as_completed(futures):
            cfg = futures[future]
            try:
                path = future.result()
                print(f"Finished {_population_id(cfg)} -> {path}")
            except Exception:
                print(f"Population config {cfg} FAILED:")
                raise


if __name__ == "__main__":
    main()