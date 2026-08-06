
import functools

import pandas as pd
from config import DATA_DIR


@functools.lru_cache(maxsize=None)
def _load_population_sheet(BV_pop: str) -> pd.DataFrame:
    """
    Read one BV_pop sheet from simulated_error_populations.xlsx.

    Cached because each sheet holds every (f_target, corr_target, r_target)
    combination for that BV_pop -- import_population() is called once per
    combination (54 times per BV_pop in the current POPULATION_CONFIGS grid),
    and re-reading/re-parsing the same ~270k-row sheet from disk on every one
    of those calls (~10s each) would dominate runtime for no reason.
    """
    data_path = DATA_DIR / 'simulated_error_populations.xlsx'
    return pd.read_excel(data_path, sheet_name=f"Detail_{BV_pop}")


def import_population(BV_pop: str, f_target: float, corr_target: float, r_target: float) -> pd.DataFrame:
    sheet = _load_population_sheet(BV_pop)

    mask = (
        (sheet["f_target"] == f_target)
        & (sheet["corr_target"] == corr_target)
        & (sheet["r_target"] == r_target)
    )
    artificial_univ = sheet.loc[mask].rename(columns={"error": "E", "book_value": "BV"})

    artificial_univ["ER"] = artificial_univ["E"] / artificial_univ["BV"]
    artificial_univ = artificial_univ[["BV", "ER", "E"]]

    return artificial_univ
