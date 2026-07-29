from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DATA_DIR = PROJECT_ROOT / "clean_data"
DOCS_DIR = PROJECT_ROOT / "docs"

# Configuration of population
POPULATION_CONFIG  = {
    "pop_file": "EU_funds",
    "id": "FSE"
}

# Configuration for each sampling method
SIMULATION_SETTINGS = {
    "sample_sizes": [30, 65, 100],
    "CL": 0.80,
    "iterations": 5000,
    "seed": 120,
    "TE_perc": 0.02,
    "configurations": [
        {
            "method": "MUS",
            "hv_selection": "nothing",
            "selection_type": "systematic_sampling",
            "bound_estimator": "Con",
        },
        {
            "method": "MUS",
            "hv_selection": "iterative",
            "selection_type": "systematic_sampling",
            "bound_estimator": "HH",
        },
        # {
        #     "method": "MUS",
        #     "hv_selection": "iterative",
        #     "selection_type": "systematic_sampling",
        #     "bound_estimator": "Mod_HH",
        # },
        # {
        #     "method": "MRS",
        #     "hv_selection": "iterative",
        #     "selection_type": "systematic_sampling",
        #     "bound_estimator": "HH",
        # },
    ],
    }

rf_path = DATA_DIR / 'reliability factor.xlsx'
RF_TABLE = pd.read_excel(rf_path)
RF_TABLE.drop(index=0, inplace=True)