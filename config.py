from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DATA_DIR = PROJECT_ROOT / "clean_data"
DOCS_DIR = PROJECT_ROOT / "docs"

# Configuration of population
POPULATION_CONFIGS  = {
    "BV_pop": ["BV_5pct_above_SI", "BV_15pct_above_SI", "BV_30pct_above_SI"],
    "f_target": [0.05, 0.20, 0.50],
    "corr_target": [0.10, 0.25, 0.50],
    "r_target": [0.002, 0.01, 0.015, 0.025, 0.03, 0.05]
}

# Configuration for each sampling method
SIMULATION_SETTINGS = {
    "sample_sizes": [30, 65, 100, 150, 200],
    "CL": [0.80, 0.90, 0.95],
    "iterations": 10_000,
    "seed": 120,
    "TE_perc": 0.02,
    "configurations": [
        {
            "method": "MUS",
            "hv_selection": "nothing",
            "selection_type": "systematic_sampling",
            "bound_estimator": "Poisson_Stringer",
        },
        {
            "method": "MUS",
            "hv_selection": "nothing",
            "selection_type": "systematic_sampling",
            "bound_estimator": "Binomial_Stringer",
        },
        {
            "method": "MUS",
            "hv_selection": "nothing",
            "selection_type": "systematic_sampling",
            "bound_estimator": "Moment",
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
    ],
    }


SAMPLE_PLANNING_SIMULATION_SETTINGS = {
    "anticipated_errors": [0.7, 0.85, 1, 1.15, 1.3], 
    "anticipated_stds": [0.7, 0.85, 1.15, 1.3],
    "sample_size_combinations": [
        {
            "anticipated_error_perc": 0.7,
            "anticipated_std": 0.7,
        },
        {
            "anticipated_error_perc": 0.85,
            "anticipated_std": 0.85,
        },
        {
            "anticipated_error_perc": 1.15,
            "anticipated_std": 1.15,
        },
        {
            "anticipated_error_perc": 1.3,
            "anticipated_std": 1.3,
        },
        {
            "anticipated_error_perc": 1.3,
            "anticipated_std": 0.7,
        },
        {
            "anticipated_error_perc": 0.7,
            "anticipated_std": 1.3,
        },

    ],
    "CL": [0.80, 0.90, 0.95],
    "iterations": 10_000,
    "seed": 120,
    "TE_perc": 0.02,
    "configurations": [
        {
            "method": "MUS",
            "hv_selection": "nothing",
            "selection_type": "systematic_sampling",
            "bound_estimator": "Poisson_Stringer",
        },
        {
            "method": "MUS",
            "hv_selection": "iterative",
            "selection_type": "systematic_sampling",
            "bound_estimator": "HH",
        },
    ],
    }

rf_path = DATA_DIR / 'reliability factor.xlsx'
RF_TABLE = pd.read_excel(rf_path)
RF_TABLE.drop(index=0, inplace=True)