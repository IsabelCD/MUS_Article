"""
Create variations of a cost/value column where items above SI (SUM/100)
contribute a target % of the total, with minimal distortion to the
existing distribution.

Method: split values into an "above SI" group and an "at/below SI" group.
Scale each group by a single multiplier so the above-group sums to the
target % of the (fixed) total. Uniform per-group scaling preserves each
group's internal relative proportions -- the smallest possible change to
the distribution's shape. Membership is re-checked after scaling in case
an item crosses the SI threshold, and re-solved until stable.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR

INPUT_PATH = DATA_DIR / "first_book_value_population.xlsx"
OUTPUT_PATH = DATA_DIR / "book_value_populations.xlsx"
VALUE_COL = "BV"  # name of the cost/value column

# target share of the total that "above SI" items should hold, per scenario
SCENARIOS = {
    "BV_5pct_above_SI": 0.05,
    "BV_15pct_above_SI": 0.15,
    "BV_30pct_above_SI": 0.30,
}


def rescale_above_si(values, target_pct, si, max_iter=20):
    """
    Rescale `values` (1D array-like) so items above `si` sum to
    target_pct * sum(values), total sum stays fixed, and both groups
    are each scaled by one uniform multiplier.

    Returns (new_values, k_above, k_below).
    """
    bv = np.asarray(values, dtype=float)
    T = bv.sum()
    mask = bv > si  # start from the original classification

    for _ in range(max_iter):
        sum_above = bv[mask].sum()
        sum_below = T - sum_above

        k_above = (target_pct * T) / sum_above if sum_above else 0.0
        k_below = ((1 - target_pct) * T) / sum_below if sum_below else 0.0

        new_values = np.where(mask, bv * k_above, bv * k_below)
        new_mask = new_values > si  # did anyone cross the SI line?

        if np.array_equal(new_mask, mask):
            return new_values, k_above, k_below

        mask = new_mask  # re-solve with the corrected grouping

    raise RuntimeError("Rescaling did not converge -- check input data")


def main():
    df = pd.read_excel(INPUT_PATH)
    bv = df[VALUE_COL]

    T = bv.sum()
    SI = T / 100

    print(f"Total (T)     = {T:,.2f}")
    print(f"SI = T / 100  = {SI:,.2f}")
    print()

    for col_name, target_pct in SCENARIOS.items():
        new_values, k_above, k_below = rescale_above_si(bv, target_pct, SI)
        df[col_name] = new_values
        print(f"{col_name}: target={target_pct:.0%}  "
              f"k_above={k_above:.6f}  k_below={k_below:.6f}")

    # --- verification ---
    print("\nVerification:")
    for col_name in SCENARIOS:
        total = df[col_name].sum()
        above = df.loc[df[col_name] > SI, col_name].sum()
        print(f"  {col_name}: total={total:,.2f}  "
              f"pct_above_SI={above / total:.4%}")

    df = df.drop(columns=[VALUE_COL])  # drop the original column as it doesn't belong to the simulation

    df.to_excel(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()