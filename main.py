from pathlib import Path
import pandas as pd

import xlsxwriter
from config import *
from simulation.run_sim import Simulation
from create_population.import_population import import_population


def main():
    for bv_nr in POPULATION_CONFIGS["BV_pop"]:
        for f_target in POPULATION_CONFIGS["f_target"]:
            for corr_target in POPULATION_CONFIGS["corr_target"]:
                for r_target in POPULATION_CONFIGS["r_target"]:

                    print(f"Running simulation for population config: {population_config}")
                    population_config = {
                        "BV_pop": bv_nr,
                        "f_target": f_target,
                        "corr_target": corr_target,
                        "r_target": r_target
                    }
                    
                    artificial_pop = import_population(**population_config)

                    simulation = Simulation(
                        population_ID=population_config["id"],
                        population=artificial_pop,
                        simulation_config=SIMULATION_SETTINGS,
                    )

                    simulation.run()
                    simulation.export()


if __name__ == "__main__":
    main()