
import pandas as pd
from config import DATA_DIR


def import_population(BV_pop: str, f_target: float, corr_target: float, r_target: float) -> pd.DataFrame:
    # load data
    data_path = DATA_DIR / 'simulated_error_populations.xlsx'
    artificial_univ = pd.read_excel(data_path, sheet_name=f"Detail_{BV_pop}")

    artificial_univ = artificial_univ[artificial_univ['f_target'] == f_target]
    artificial_univ = artificial_univ[artificial_univ['corr_target'] == corr_target]
    artificial_univ = artificial_univ[artificial_univ['r_target'] == r_target]

    artificial_univ.rename(columns={"error": "E","book_value": "BV"}, inplace=True)

    artificial_univ['ER'] = artificial_univ["E"] / artificial_univ['BV']
    artificial_univ = artificial_univ[['BV', 'ER', 'E']]

    return artificial_univ