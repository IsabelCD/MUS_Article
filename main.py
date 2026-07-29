from pathlib import Path
import pandas as pd

import xlsxwriter
from config import *
from simulation.run_sim import Simulation
from create_population.pop_from_eu_funds import import_population

funds = ['FSE', "FEDER_FC"]
def main():
    for fund in funds: 
        print(f'starting simulation of fund {fund}')
        population_config  = {
            "pop_file": "EU_funds",
            "id": fund
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