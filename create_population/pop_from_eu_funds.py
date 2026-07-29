
import pandas as pd
import numpy as np
from config import DATA_DIR


def import_population(bv_nr: str, pop_nr: str = None) -> pd.DataFrame:
    # load data
    data_path = DATA_DIR / 'populations_1_28_new_corr.xlsx'
    artificial_univ = pd.read_excel(data_path, sheet_name=bv_nr)

    artificial_univ.rename(columns={f"E{pop_nr}": "E"}, inplace=True)

    artificial_univ['ER'] = artificial_univ["E"] / artificial_univ['BV']
    artificial_univ['BV'] = artificial_univ['BV'].astype(float)
    artificial_univ = artificial_univ[['BV', 'ER', 'E']]

    return artificial_univ