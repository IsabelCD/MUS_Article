"""
Simulate error populations for audit sampling testing.

Reads book value populations from an Excel file and, for every combination of:
    f    (error frequency, i.e. % of items in the population that contain an error)
    corr (correlation between the size of the error and the book value of the item)
    r    (error rate, i.e. total simulated error as a % of the population's total book value)

...generates a simulated "audited" population with errors injected.

--------------------------------------------------------------------------------------
ASSUMPTIONS ABOUT THE INPUT FILE (adjust the CONFIG section below if different):
- Single sheet, one column per population, book values listed down the rows.
- Column headers are the population names (e.g. "Population_1", "Population_2", ...).
- Populations may have different lengths; shorter columns can contain blank cells.
--------------------------------------------------------------------------------------

METHOD NOTES:
- For each combination, f% of items (rounded, at least 1) are randomly selected to
  contain an error.
- The total euro amount of error injected into the population equals r * (total book
  value of the WHOLE population) - this matches your definition (r=1% and BV=1000 ->
  total injected error = 10).
- That total is distributed across the selected "erroneous" items so that the Pearson
  correlation between (book value of erroneous items) and (error amount of erroneous
  items) equals corr. This is done by building a latent variable
      latent = corr * z(book_value) + sqrt(1-corr^2) * noise
  and turning it into non-negative weights (via a pure shift + positive rescale, both
  of which are affine transforms and therefore do NOT change the Pearson correlation).
  The weights are then multiplied by the target total error amount.
  => "achieved_corr" reported in the Summary sheet should closely match the target
     corr (exact in expectation; sample noise for small n_errors).
- Correlation is computed only among items that actually contain an error (comparing
  it against the full population, including the zero-error items, would mechanically
  distort/deflate it and isn't a meaningful reading of "correlation between error and
  book value").
"""

import numpy as np
import pandas as pd

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR

# =========================== CONFIG =====================================
INPUT_FILE = DATA_DIR / "book_value_populations.xlsx"   # path to your uploaded Excel file
INPUT_SHEET = "Sheet1"                                  # sheet name or index containing the 4 populations
OUTPUT_FILE = DATA_DIR / "simulated_error_populations.xlsx"

F_LEVELS = [0.05, 0.20, 0.50]                # error frequency levels
CORR_LEVELS = [0.10, 0.25, 0.50]             # correlation levels
R_LEVELS = [0.002, 0.01, 0.015, 0.025, 0.03, 0.05]  # error rate levels (% of total BV)

RANDOM_SEED = 42      # set to None for non-reproducible runs
# ==========================================================================


def load_bv_populations(path, sheet=0):
    """Load book value populations from Excel. Returns dict {population_name: np.array of book values}."""
    df = pd.read_excel(path, sheet_name=sheet)
    populations = {}
    for col in df.columns:
        values = pd.to_numeric(df[col], errors="coerce").dropna().values
        if len(values) == 0:
            continue
        populations[str(col)] = values.astype(float)
    return populations


def simulate_errors(book_values, f, corr, r, rng=None):
    """
    Inject simulated errors into one book value population for one (f, corr, r) combo.

    Returns:
        errors_full   : np.array, same length as book_values, error amount per item (0 if no error)
        is_error      : boolean mask of which items contain an error
        achieved_corr : Pearson correlation actually achieved between book value and
                        error amount, computed over the erroneous items only (NaN if
                        fewer than 2 erroneous items or no variance)
        target_total  : the intended total error amount (r * sum(book_values))
    """
    if rng is None:
        rng = np.random.default_rng()

    bv = np.asarray(book_values, dtype=float)
    n = len(bv)
    n_err = max(1, int(round(f * n)))
    n_err = min(n_err, n)

    idx = rng.choice(n, size=n_err, replace=False)
    sel_bv = bv[idx]

    target_total = r * bv.sum()

    if sel_bv.std() > 0:
        z_bv = (sel_bv - sel_bv.mean()) / sel_bv.std()
    else:
        z_bv = np.zeros_like(sel_bv)

    noise = rng.normal(0.0, 1.0, n_err)
    latent = corr * z_bv + np.sqrt(max(0.0, 1 - corr ** 2)) * noise

    # Affine-only transforms below (shift then positive rescale) -> correlation preserved
    shifted = latent - latent.min() + 1e-9
    weights = shifted / shifted.sum()
    error_amounts = weights * target_total

    errors_full = np.zeros(n)
    errors_full[idx] = error_amounts

    is_error = np.zeros(n, dtype=bool)
    is_error[idx] = True

    achieved_corr = np.nan
    if n_err > 1 and sel_bv.std() > 0 and np.std(error_amounts) > 0:
        achieved_corr = np.corrcoef(sel_bv, error_amounts)[0, 1]

    return errors_full, is_error, achieved_corr, target_total


def build_detail_frame(pop_name, book_values, f, corr, r, rng):
    errors_full, is_error, achieved_corr, target_total = simulate_errors(
        book_values, f, corr, r, rng=rng
    )
    detail = pd.DataFrame({
        "population": pop_name,
        "f_target": f,
        "corr_target": corr,
        "r_target": r,
        "item_id": np.arange(1, len(book_values) + 1),
        "book_value": book_values,
        "error": errors_full,
        "audited_value": book_values + errors_full,
        "is_error": is_error,
    })
    summary_row = {
        "population": pop_name,
        "f_target": f,
        "corr_target": corr,
        "r_target": r,
        "n_items": len(book_values),
        "n_errors": int(is_error.sum()),
        "total_book_value": float(book_values.sum()),
        "target_error_total": target_total,
        "achieved_error_total": float(errors_full.sum()),
        "achieved_corr": achieved_corr,
    }
    return detail, summary_row


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    populations = load_bv_populations(INPUT_FILE, sheet=INPUT_SHEET)
    if not populations:
        raise ValueError("No populations found in the input file - check INPUT_FILE / INPUT_SHEET.")

    print(f"Loaded {len(populations)} population(s): {list(populations.keys())}")

    summary_rows = []
    details_by_population = {name: [] for name in populations}

    combo_count = 0
    for pop_name, book_values in populations.items():
        for f in F_LEVELS:
            for corr in CORR_LEVELS:
                for r in R_LEVELS:
                    detail, summary_row = build_detail_frame(
                        pop_name, book_values, f, corr, r, rng
                    )
                    details_by_population[pop_name].append(detail)
                    summary_rows.append(summary_row)
                    combo_count += 1

    print(f"Generated {combo_count} simulated error populations "
          f"({len(F_LEVELS)} x {len(CORR_LEVELS)} x {len(R_LEVELS)} combos x "
          f"{len(populations)} population(s)).")

    summary_df = pd.DataFrame(summary_rows)

    with pd.ExcelWriter(OUTPUT_FILE, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        for pop_name, frames in details_by_population.items():
            detail_df = pd.concat(frames, ignore_index=True)
            # Excel sheet names: max 31 chars, no invalid chars
            safe_name = f"Detail_{pop_name}"[:31]
            if len(detail_df) > 1_000_000:
                csv_path = f"Detail_{pop_name}.csv"
                detail_df.to_csv(csv_path, index=False)
                print(f"Detail for '{pop_name}' too large for Excel ({len(detail_df)} rows) "
                      f"-> written to {csv_path} instead.")
            else:
                detail_df.to_excel(writer, sheet_name=safe_name, index=False)

    print(f"Done. Output written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()