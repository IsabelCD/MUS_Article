"""
Create variations of a cost/value column where items classified as
high-value (HV) contribute a target % of the total, without altering any
item that is already high-value in the original data.

High-value status uses the same definition as the simulation itself:
simulation.inclusion_probability.iterative_hv_selection(population, BV, n=100)
-- the certainty-unit rule for a systematic PPS sample of size 100, which is
what SI = T / 100 originally approximated.

Method, per scenario:
- Items already HV in the original data keep their exact original value --
  never touched, in either direction.
- To RAISE the HV share above its natural level: non-HV items are converted
  to HV one at a time, in random order, each given a random value in
  (SI, STABLE_CAP]. Conversion stops once the HV share reaches the target;
  the last item converted is then adjusted to hit the target exactly
  (unless that would drop it back below SI, in which case the slight
  overshoot from the random draw is kept instead).
- To LOWER the HV share below its natural level: the single largest
  original HV item is reduced to the exact value that makes the HV share
  hit the target (it may drop out of HV status entirely).
- Total book value is kept fixed in both directions: any value added to
  (or removed from) converted/reduced items is offset by a proportional
  counter-adjustment across the untouched non-HV pool, so T and
  SI = T / 100 never move.

STABLE_CAP (the "raise" branch's upper bound, `min(max original BV,
T / STABLE_HV_SAMPLE_SIZE * STABLE_CAP_MARGIN)`) keeps every *engineered*
HV item below the certainty threshold of the simulation's second-smallest
sample size (STABLE_HV_SAMPLE_SIZE, currently 65). Certainty status at that
sample size is then determined only by whichever items are naturally that
large in the source data, instead of inheriting most of the 100-sample-size
target's engineered items by chance of the uniform draw -- which previously
made the certainty stratum jump from empty (n=30, no item in the source
data is large enough to be certain there at all) to a large chunk of the
n=100 target's size at n=65, and disproportionately depleted the
non-certainty stratum's error signal at that sample size as a result. The
target share at n=100 (and above) is unaffected: those items still need to
be, and are, above SI(100).
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RANDOM_SEED
from config import DATA_DIR
from config import SIMULATION_SETTINGS
from simulation.inclusion_probability import iterative_hv_selection

INPUT_PATH = DATA_DIR / "first_book_value_population.xlsx"
OUTPUT_PATH = DATA_DIR / "book_value_populations.xlsx"
VALUE_COL = "BV"  # name of the cost/value column
HV_SAMPLE_SIZE = 100  # matches SI = T / 100; see module docstring

# Engineered ("raised") HV items are capped below the certainty threshold of
# this sample size, so that sample size's certainty stratum stays limited to
# whatever is naturally that large in the source data. Must be strictly
# smaller than HV_SAMPLE_SIZE (a larger SI) and is otherwise the smallest
# sample size actually run below HV_SAMPLE_SIZE in SIMULATION_SETTINGS
# (config.py) for which a naturally-sized certainty stratum is achievable
# -- see the module docstring.
STABLE_HV_SAMPLE_SIZE = 65
STABLE_CAP_MARGIN = 0.995  # keeps capped draws strictly below SI(STABLE_HV_SAMPLE_SIZE)

# target share of the total that HV items should hold, per scenario
SCENARIOS = {
    "BV_5pct_above_SI": 0.05,
    "BV_15pct_above_SI": 0.15,
    "BV_30pct_above_SI": 0.30,
}

def hv_mask(values: np.ndarray, T: float, n: int = HV_SAMPLE_SIZE) -> np.ndarray:
    """Boolean mask of high-value (certainty) units, per iterative_hv_selection."""
    result = iterative_hv_selection(pd.DataFrame({"BV": values}), BV=T, n=n)
    return (result["HV"] == 1).to_numpy()


def _offset_total(values: np.ndarray, indices: np.ndarray, delta: float) -> None:
    """
    Distribute `delta` proportionally (by current value) across
    values[indices], in place, so the array's total changes by exactly
    `delta`. Used to keep the population's total book value fixed after
    adding value to (delta<0 here, since it offsets an addition) or
    removing value from (delta>0) other items.
    """
    if len(indices) == 0:
        if abs(delta) > 1e-6:
            raise RuntimeError("No untouched items available to offset total book value.")
        return
    weights = values[indices]
    total_w = weights.sum()
    if total_w <= 0:
        raise RuntimeError("Cannot distribute total-value offset: pool has zero value.")
    values[indices] = weights + delta * (weights / total_w)


def build_scenario(
    values: np.ndarray,
    target_pct: float,
    T: float,
    si: float,
    orig_hv: np.ndarray,
    rng: np.random.Generator,
    stable_cap: float | None = None,
) -> np.ndarray:
    """
    Return a new values array whose HV share of T equals target_pct.
    `orig_hv` (the original HV mask) is never modified in the "raise" branch
    and is the only eligible source item in the "lower" branch.

    `stable_cap`, when given, caps the "raise" branch's random draw below
    this value (see STABLE_CAP in the module docstring) so engineered items
    stay below a tighter sample size's certainty threshold; `None` keeps the
    original behaviour of drawing up to the largest original BV.
    """
    values = values.copy()
    max_bv = values[orig_hv].max()
    draw_cap = min(max_bv, stable_cap) if stable_cap is not None else max_bv

    current_hv_sum = values[orig_hv].sum()
    target_hv_sum = target_pct * T

    if np.isclose(current_hv_sum, target_hv_sum, rtol=1e-9):
        return values

    if target_hv_sum > current_hv_sum:
        pool = np.where(~orig_hv)[0]
        rng.shuffle(pool)

        needed = target_hv_sum - current_hv_sum
        added = 0.0
        converted = []

        for idx in pool:
            if added >= needed:
                break
            old_value = values[idx]
            new_value = rng.uniform(si, draw_cap)
            values[idx] = new_value
            added += new_value - old_value
            converted.append((idx, old_value))
        else:
            raise RuntimeError(
                f"Ran out of convertible items before reaching target_pct={target_pct:.0%} "
                f"for the {len(pool)}-item non-HV pool."
            )

        # Hit the target exactly by adjusting whichever converted item can
        # absorb the overshoot while staying above SI (tried most-recent
        # first); if none can, keep the small overshoot from the random draws.
        overshoot = added - needed
        if overshoot > 0:
            for idx, _ in reversed(converted):
                adjusted_value = values[idx] - overshoot
                if adjusted_value > si:
                    values[idx] = adjusted_value
                    added -= overshoot
                    break

        # Keep total fixed: shrink the untouched (non-HV, non-converted) pool.
        converted_idx = {idx for idx, _ in converted}
        untouched_pool = np.array([i for i in pool if i not in converted_idx])
        _offset_total(values, untouched_pool, -added)

    else:
        # Changing one HV item is a discrete choice: once its new value drops
        # low enough, it stops contributing to the HV sum entirely -- so the
        # achievable outcomes per candidate item are not a continuous range.
        # Search all original HV items for whichever candidate gets closest
        # to the target (exact hit preferred).
        #
        # A value strictly above si is *unconditionally* HV: pass 1 of
        # iterative_hv_selection compares against BV/n = T/100 = si directly,
        # and once flagged certain an item is never demoted in later passes.
        # But the reverse isn't true: dropping an item toward si is NOT
        # unconditionally safe, because removing other items into certainty
        # shrinks the pool the iteration redistributes over, which *lowers*
        # the effective threshold for whatever is left -- a value just under
        # si can still get swept back into certainty by a later pass (this
        # was verified against actual data: pass 1 leaves it at ~0.9999... ,
        # not 1, but pass 2's shrunk-denominator threshold ends up below it).
        # So "drop" candidates are verified against the real hv_mask(),
        # not derived from a formula, and shrunk further on the rare retry.
        hv_indices = np.where(orig_hv)[0]

        best_idx, best_new_value, best_miss = None, None, None
        for idx in hv_indices:
            old_value = values[idx]
            other_sum = current_hv_sum - old_value
            exact_value = target_hv_sum - other_sum

            if si < exact_value < old_value:
                candidate_new_value, miss = exact_value, 0.0
            else:
                # Drop this item out of HV status entirely. Its exact
                # sub-threshold value doesn't affect the HV sum, but must
                # stay strictly positive (simulate_errors.py's
                # load_populations() rejects non-positive book values).
                candidate_new_value = si * 0.5
                for _ in range(20):
                    trial = values.copy()
                    trial[idx] = candidate_new_value
                    if not hv_mask(trial, T)[idx]:
                        break
                    candidate_new_value /= 2
                else:
                    raise RuntimeError(
                        f"Could not find a value for item {idx} that drops it out of "
                        f"HV status after 20 halvings."
                    )
                achieved = other_sum  # this item's (now-excluded) value contributes nothing
                miss = abs(achieved - target_hv_sum)

            if best_miss is None or miss < best_miss - 1e-9:
                best_idx, best_new_value, best_miss = idx, candidate_new_value, miss

        reduction = values[best_idx] - best_new_value
        values[best_idx] = best_new_value

        if best_miss > 1e-6:
            print(f"    note: target_pct={target_pct:.2%} cannot be hit exactly by changing a "
                  f"single original HV item; closest achievable misses by {best_miss:,.2f} "
                  f"({best_miss / T:.4%} of T).")

        # Keep total fixed: grow the untouched non-HV pool back up.
        pool = np.where(~orig_hv)[0]
        _offset_total(values, pool, reduction)

    return values


def main():
    df = pd.read_excel(INPUT_PATH)
    bv = df[VALUE_COL].to_numpy(dtype=float)

    T = bv.sum()
    SI = T / 100
    stable_cap = (T / STABLE_HV_SAMPLE_SIZE) * STABLE_CAP_MARGIN
    orig_hv = hv_mask(bv, T)

    print(f"Total (T)          = {T:,.2f}")
    print(f"SI = T / 100       = {SI:,.2f}")
    print(f"Stable cap (T / {STABLE_HV_SAMPLE_SIZE} * {STABLE_CAP_MARGIN}) = {stable_cap:,.2f}")
    print(f"Original HV items  = {orig_hv.sum()} (sum={bv[orig_hv].sum():,.2f}, "
          f"{bv[orig_hv].sum() / T:.2%} of T)")
    print()

    rng = np.random.default_rng(RANDOM_SEED)

    for col_name, target_pct in SCENARIOS.items():
        new_values = build_scenario(bv, target_pct, T, SI, orig_hv, rng, stable_cap=stable_cap)
        df[col_name] = new_values
        print(f"{col_name}: target={target_pct:.0%}")

    # --- verification ---
    print("\nVerification:")
    for col_name in SCENARIOS:
        values = df[col_name].to_numpy(dtype=float)
        total = values.sum()
        mask = hv_mask(values, T)
        above = values[mask].sum()
        untouched_original_hv = np.array_equal(values[orig_hv], bv[orig_hv])
        print(f"  {col_name}: total={total:,.2f}  pct_HV={above / total:.4%}  "
              f"n_HV={mask.sum()}  original_HV_untouched={untouched_original_hv}")

    print("\nCertainty count across the actual sample-size grid (SIMULATION_SETTINGS):")
    for col_name in SCENARIOS:
        values = df[col_name].to_numpy(dtype=float)
        total = values.sum()
        row = []
        for n in SIMULATION_SETTINGS["sample_sizes"]:
            mask_n = (iterative_hv_selection(pd.DataFrame({"BV": values}), BV=total, n=n)["HV"] == 1).to_numpy()
            row.append(f"n={n}:cert={int(mask_n.sum())},ns={n - int(mask_n.sum())}")
        print(f"  {col_name}: " + "  ".join(row))

    df = df.drop(columns=[VALUE_COL])  # drop the original column as it doesn't belong to the simulation

    df.to_excel(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
