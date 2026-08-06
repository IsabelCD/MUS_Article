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
- For each combination, f% of items (rounded, at least 1) contain an error; the rest
  have error = 0.
- The total euro amount of error injected into the population equals r * (total book
  value of the WHOLE population) - this matches your definition (r=1% and BV=1000 ->
  total injected error = 10).
- corr is defined and achieved as the Pearson correlation you'd get from
  =CORREL(book_value, error) computed over the WHOLE population for that combination
  (i.e. including the (1-f)% of items whose error is 0).
- HARD CONSTRAINT: no item's error amount can exceed its own book value
  (|error_i| <= book_value_i), so audited_value never goes negative or more than
  doubles. This is enforced item-by-item.

FEASIBILITY CHECK (this is the part relevant to your question):
- With k = round(f * n_items) items allowed to contain an error, and each error
  capped at that item's own book value, the absolute maximum total error achievable
  -no matter which corr you want- is the sum of the k LARGEST book values in the
  population (put the error at 100% of book value on each of those k items).
- If target_total = r * total_book_value exceeds that maximum, the (population, f, r)
  combination is mathematically IMPOSSIBLE regardless of corr (corr only changes how
  the total is distributed across items, not the size of the total). The script
  checks this BEFORE generating anything, skips those combinations, and lists them in
  an "Infeasible_Combinations" sheet plus prints them to the console.
- For feasible combinations, the error total is distributed using a latent variable
  correlated with book value (to hit your target corr, solved by bisection), then
  passed through a water-filling allocation that respects the per-item cap while
  getting as close as possible to both the target total and the target corr. If the
  initially-chosen set of erroneous items doesn't have enough combined book value to
  reach the target total (even though the population as a whole does), the script
  swaps in higher-book-value items to guarantee the target total is met exactly.
  A `capacity_constrained` flag is added per combination in the Summary sheet: True
  means at least one item was capped at 100% of its book value, which can pull the
  achieved corr away from the target - check achieved_corr in that case.
- SECOND, SUBTLER FEASIBILITY ISSUE: even when the TOTAL is achievable, a specific
  target corr may not be with a given random draw. This mainly bites LOW f combined
  with a LOW target corr and a non-trivial r: with only k=f*n items allowed to carry
  the error, spreading a large total fairly evenly (needed for low corr) can require
  some items to hold more error than their own book value allows, forcing the
  allocation onto larger items instead - pushing the achieved correlation ABOVE the
  target even though the total is fine. WHICH items get selected as the k erroneous
  ones depends on a random draw, so this is retried: each combination gets up to
  MAX_CORR_RETRIES (default 25) independent random draws, stopping as soon as one
  lands within CORR_TOLERANCE (default 0.02) of the target, and keeping the best
  (lowest-deviation) draw otherwise. Only combinations that still miss tolerance after
  every retry are flagged as an alert (printed to console, `corr_within_tolerance` /
  `n_attempts` columns in the Summary sheet) - these represent corr targets that
  appear genuinely unreachable for that (population, f, r) given the error cap, not
  just unlucky randomness.
- By default all injected errors are OVERSTATEMENTS (positive numbers added to book
  value). Set DIRECTION = -1 below to simulate understatements instead (the cap then
  keeps audited_value >= 0). Note this flips the sign of achieved correlation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from config import RANDOM_SEED
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR

# =========================== CONFIG =====================================
INPUT_FILE = DATA_DIR / "book_value_populations.xlsx"
INPUT_SHEET = 0                              # sheet name or index containing the 4 populations
OUTPUT_FILE = DATA_DIR / "simulated_error_populations.xlsx"

F_LEVELS = [0.05, 0.20, 0.50]                # error frequency levels
CORR_LEVELS = [0.10, 0.25, 0.50]             # correlation levels
R_LEVELS = [0.002, 0.01, 0.015, 0.025, 0.03, 0.05]  # error rate levels (% of total BV)

CORR_TOLERANCE = 0.05 # flag a combo if achieved corr misses target by more than this
                      # (this can happen even when the TOTAL is feasible - see note below)
MAX_CORR_RETRIES = 500 # if a random draw misses CORR_TOLERANCE, retry with a fresh draw
                      # up to this many times before giving up and alerting
# ==========================================================================


def load_populations(path, sheet=0):
    """Load book value populations from Excel. Returns dict {population_name: np.array of book values}."""
    df = pd.read_excel(path, sheet_name=sheet)
    populations = {}
    for col in df.columns:
        values = pd.to_numeric(df[col], errors="coerce").dropna().values
        if len(values) == 0:
            continue
        if (values <= 0).any():
            raise ValueError(
                f"Population '{col}' contains zero or negative book values, which "
                f"breaks the error-capping logic (a cap of 0 means that item can "
                f"never contain an error). Please clean the data first."
            )
        populations[str(col)] = values.astype(float)
    return populations


def max_capacity(book_values, k):
    """Maximum total error achievable with k erroneous items, each capped at its own book value."""
    bv_sorted = np.sort(book_values)[::-1]
    return bv_sorted[:k].sum()


def check_feasibility(book_values, f, r):
    """
    Return (feasible: bool, k: int, target_total: float, max_total: float) for a
    given (population, f, r) combination - independent of corr.
    """
    n = len(book_values)
    k = max(1, min(n, int(round(f * n))))
    target_total = r * book_values.sum()
    max_total = max_capacity(book_values, k)
    return target_total <= max_total + 1e-9, k, target_total, max_total


def _water_fill(weights, caps, total):
    """
    Allocate `total` across items proportional to `weights`, never exceeding each
    item's `caps` value. Assumes sum(caps) >= total (caller must guarantee this).
    """
    n = len(weights)
    amounts = np.zeros(n)
    remaining = float(total)
    active = np.ones(n, dtype=bool)
    w = weights.astype(float).copy()

    for _ in range(n + 1):
        if remaining <= 1e-9 or not active.any():
            break
        w_sum = w[active].sum()
        if w_sum <= 0:
            desired = np.zeros(n)
            desired[active] = remaining / active.sum()
        else:
            desired = np.zeros(n)
            desired[active] = remaining * w[active] / w_sum

        over = active & (desired > caps + 1e-9)
        if not over.any():
            amounts[active] = desired[active]
            remaining = 0.0
            break
        amounts[over] = caps[over]
        remaining -= caps[over].sum()
        active[over] = False
        w[over] = 0.0

    return amounts


def _build_support(bv, latent, k, target_total):
    """
    Choose the k erroneous items (top-k by latent value), then swap in higher
    book-value items if the chosen support can't cover target_total under the
    per-item cap (guaranteed possible since global feasibility was pre-checked).
    """
    n = len(bv)
    idx = list(np.argsort(-latent)[:k])
    support = set(idx)
    outside = set(range(n)) - support

    while bv[list(support)].sum() < target_total and outside:
        min_in = min(support, key=lambda i: bv[i])
        max_out = max(outside, key=lambda i: bv[i])
        if bv[max_out] <= bv[min_in]:
            break
        support.remove(min_in)
        support.add(max_out)
        outside.remove(max_out)
        outside.add(min_in)

    return np.array(sorted(support))


def _errors_for_rho(bv, z_bv, rho, noise_all, k, target_total):
    """
    Build a full-length (unsigned) error vector for a trial rho, respecting the
    per-item cap (error_i <= book_value_i), and return the resulting whole-
    population Pearson correlation with book value.
    """
    latent = rho * z_bv + np.sqrt(max(0.0, 1 - rho ** 2)) * noise_all
    idx = _build_support(bv, latent, k, target_total)

    l_sel = latent[idx]
    shifted = l_sel - l_sel.min() + 1e-9
    weights = shifted / shifted.sum()
    caps = bv[idx]

    amounts = _water_fill(weights, caps, target_total)

    errors_full = np.zeros(len(bv))
    errors_full[idx] = amounts

    if amounts.std() > 0:
        achieved = np.corrcoef(bv, errors_full)[0, 1]
    else:
        achieved = 0.0
    return errors_full, idx, achieved


def _solve_rho(bv, z_bv, noise_all, k, target_total, target_corr,
               tol=5e-4, max_iter=60, grid_points=81):
    """
    Numerically find the rho that makes the whole-population Pearson correlation
    between book value and error match target_corr (bisection, with a coarse grid
    search first to bracket the root robustly).
    """
    grid = np.linspace(-1.0, 1.0, grid_points)
    achieved_vals = np.array([
        _errors_for_rho(bv, z_bv, g, noise_all, k, target_total)[2] for g in grid
    ])
    diffs = achieved_vals - target_corr

    sign_changes = np.where(np.diff(np.sign(diffs)) != 0)[0]
    if len(sign_changes) == 0:
        best = grid[np.argmin(np.abs(diffs))]
        lo, hi = max(-1.0, best - 0.05), min(1.0, best + 0.05)
    else:
        i = sign_changes[0]
        lo, hi = grid[i], grid[i + 1]

    _, _, c_lo = _errors_for_rho(bv, z_bv, lo, noise_all, k, target_total)
    mid = lo
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        _, _, c_mid = _errors_for_rho(bv, z_bv, mid, noise_all, k, target_total)
        if abs(c_mid - target_corr) < tol:
            break
        if (c_lo - target_corr) * (c_mid - target_corr) <= 0:
            hi = mid
        else:
            lo, c_lo = mid, c_mid
    return mid


def simulate_errors(book_values, f, corr, r, rng=None,
                     tolerance=0.02, max_retries=25):
    """
    Inject simulated errors into one book value population for one (f, corr, r) combo.
    Caller must have already confirmed feasibility via check_feasibility().

    Retries with fresh random noise draws (up to max_retries times) whenever the
    achieved correlation misses corr by more than `tolerance`, keeping the best
    (lowest-deviation) attempt seen. Stops early as soon as an attempt lands within
    tolerance. This helps because WHICH items end up carrying the error (and how hard
    their caps bind) depends on the random draw, not just on f/corr/r - some draws
    give the water-filling algorithm more room to hit the target corr than others.

    Returns:
        errors_full          : np.array, error amount per item (0 if no error)
        is_error             : boolean mask of which items contain an error
        achieved_corr        : Pearson correlation actually achieved (whole population, signed)
        target_total         : the intended total error amount (r * sum(book_values))
        capacity_constrained : True if at least one item was capped at 100% of its book value
                                (in the best attempt kept)
        n_attempts           : how many random draws were tried
        within_tolerance     : whether the best attempt landed within `tolerance` of corr
    """
    if rng is None:
        rng = np.random.default_rng()

    bv = np.asarray(book_values, dtype=float)
    n = len(bv)
    k = max(1, min(n, int(round(f * n))))

    target_total = r * bv.sum()
    z_bv = (bv - bv.mean()) / bv.std() if bv.std() > 0 else np.zeros_like(bv)

    best = None  # (deviation, errors_full, idx, achieved_corr)
    n_attempts = 0
    within_tolerance = False

    for attempt in range(1, max_retries + 1):
        n_attempts = attempt
        noise_all = rng.normal(0.0, 1.0, n)

        rho = _solve_rho(bv, z_bv, noise_all, k, target_total, corr)
        errors_full, idx, _ = _errors_for_rho(bv, z_bv, rho, noise_all, k, target_total)

        achieved_corr = np.corrcoef(bv, errors_full)[0, 1] if errors_full.std() > 0 else np.nan
        deviation = abs(achieved_corr - corr) if not np.isnan(achieved_corr) else np.inf

        if best is None or deviation < best[0]:
            best = (deviation, errors_full, idx, achieved_corr)

        if deviation <= tolerance:
            within_tolerance = True
            break

    deviation, errors_full, idx, achieved_corr = best

    capacity_constrained = bool(np.any(np.isclose(np.abs(errors_full[idx]), bv[idx], rtol=1e-6)))
    is_error = np.zeros(n, dtype=bool)
    is_error[idx] = True

    return (errors_full, is_error, achieved_corr, target_total,
            capacity_constrained, n_attempts, within_tolerance)


def build_detail_frame(pop_name, book_values, f, corr, r,  rng):
    (errors_full, is_error, achieved_corr, target_total, capacity_constrained,
     n_attempts, within_tolerance) = simulate_errors(
        book_values, f, corr, r, rng=rng,
        tolerance=CORR_TOLERANCE, max_retries=MAX_CORR_RETRIES
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
        "achieved_error_total": float(errors_full.sum()),  # magnitude, sign-agnostic
        "achieved_corr": achieved_corr,
        "capacity_constrained": capacity_constrained,
        "n_attempts": n_attempts,
        "corr_within_tolerance": within_tolerance,
    }
    return detail, summary_row


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    populations = load_populations(INPUT_FILE, sheet=INPUT_SHEET)
    if not populations:
        raise ValueError("No populations found in the input file - check INPUT_FILE / INPUT_SHEET.")

    print(f"Loaded {len(populations)} population(s): {list(populations.keys())}")

    # --- Step 1: feasibility check for every (population, f, r) combo -----------
    infeasible_rows = []
    feasible_fr = {}  # population -> set of (f, r) that are feasible
    for pop_name, book_values in populations.items():
        feasible_fr[pop_name] = set()
        for f in F_LEVELS:
            for r in R_LEVELS:
                ok, k, target_total, max_total = check_feasibility(book_values, f, r)
                if ok:
                    feasible_fr[pop_name].add((f, r))
                else:
                    infeasible_rows.append({
                        "population": pop_name,
                        "f": f,
                        "r": r,
                        "n_items": len(book_values),
                        "n_error_items_k": k,
                        "total_book_value": float(book_values.sum()),
                        "target_error_total": target_total,
                        "max_achievable_total_given_cap": max_total,
                        "shortfall": target_total - max_total,
                    })

    if infeasible_rows:
        print(f"\n*** ALERT: {len(infeasible_rows)} (population, f, r) combination(s) are "
              f"mathematically IMPOSSIBLE under the 'error <= book value' constraint "
              f"(applies to ALL corr levels for that combo) ***")
        for row in infeasible_rows:
            print(f"  - {row['population']}: f={row['f']:.0%}, r={row['r']:.1%} -> "
                  f"needs total error {row['target_error_total']:.2f}, but max possible "
                  f"with {row['n_error_items_k']} erroneous item(s) is "
                  f"{row['max_achievable_total_given_cap']:.2f} "
                  f"(shortfall {row['shortfall']:.2f}). SKIPPED.")
    else:
        print("\nAll (population, f, r) combinations are feasible under the error-cap constraint.")

    # --- Step 2: generate simulated errors for all feasible combinations ---------
    summary_rows = []
    details_by_population = {name: [] for name in populations}

    combo_count = 0
    skipped_count = 0
    for pop_name, book_values in populations.items():
        for f in F_LEVELS:
            for corr in CORR_LEVELS:
                for r in R_LEVELS:
                    if (f, r) not in feasible_fr[pop_name]:
                        skipped_count += 1
                        continue
                    detail, summary_row = build_detail_frame(
                        pop_name, book_values, f, corr, r, rng
                    )
                    details_by_population[pop_name].append(detail)
                    summary_rows.append(summary_row)
                    combo_count += 1

    total_possible = len(F_LEVELS) * len(CORR_LEVELS) * len(R_LEVELS) * len(populations)
    print(f"\nGenerated {combo_count} simulated error populations out of {total_possible} "
          f"possible combinations ({skipped_count} skipped as infeasible).")

    summary_df = pd.DataFrame(summary_rows)
    infeasible_df = pd.DataFrame(infeasible_rows)

    # --- Step 3: flag combos where the target corr couldn't be hit even after retries ---
    if not summary_df.empty:
        summary_df["corr_deviation"] = (summary_df["corr_target"] - summary_df["achieved_corr"]).abs()

        corr_issues = summary_df[~summary_df["corr_within_tolerance"]]
        if not corr_issues.empty:
            print(f"\n*** ALERT: {len(corr_issues)} generated combination(s) could NOT hit their "
                  f"target correlation within {CORR_TOLERANCE} even after up to {MAX_CORR_RETRIES} "
                  f"retries with fresh random draws - the error-cap constraint appears to make this "
                  f"corr genuinely unreachable for this (population, f, r). Generated using the best "
                  f"attempt found (total error and frequency are still exact) - review 'achieved_corr' "
                  f"vs 'corr_target' before using these: ***")
            for _, row in corr_issues.sort_values("corr_deviation", ascending=False).iterrows():
                print(f"  - {row['population']}: f={row['f_target']:.0%}, "
                      f"corr_target={row['corr_target']:.2f}, r={row['r_target']:.1%} -> "
                      f"best achieved_corr={row['achieved_corr']:.3f} "
                      f"(off by {row['corr_deviation']:.3f}, after {row['n_attempts']} attempts)")
        else:
            retried = summary_df[summary_df["n_attempts"] > 1]
            if not retried.empty:
                print(f"\nAll generated combinations hit their target correlation within tolerance "
                      f"({len(retried)} needed more than one random draw to get there, "
                      f"max attempts used: {retried['n_attempts'].max()}).")
            else:
                print("\nAll generated combinations hit their target correlation within tolerance "
                      "on the first draw.")

    with pd.ExcelWriter(OUTPUT_FILE, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        if not infeasible_df.empty:
            infeasible_df.to_excel(writer, sheet_name="Infeasible_Combinations", index=False)
        for pop_name, frames in details_by_population.items():
            if not frames:
                continue
            detail_df = pd.concat(frames, ignore_index=True)
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
