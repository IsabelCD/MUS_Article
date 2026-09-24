# MUS Sampling Simulation

This project studies monetary unit sampling (MUS) audit precision through
Monte Carlo simulation, using synthetic audit populations with injected
errors of known frequency, correlation, and total rate. It compares seven
precision/bound estimators (Hansen-Hurwitz and its modified variant, the
Poisson and Binomial Stringer bounds, and three Cornish-Fisher moment-bound
variants) under one shared sampling and error-projection pipeline.

## Project structure

```text
.
|-- main.py                          Full Monte Carlo simulation entry point
|-- run_sensitivity_analysis.py      Sample-size planning sensitivity entry point
|-- config.py                        Paths and simulation-parameter presets
|-- requirements.txt                 Pinned Python dependencies
|-- clean_data/                      Source and generated population workbooks
|-- create_population/
|   |-- simulate_book_values.py      Rescales a base book-value column into
|   |                                 "% of value above SI" population variants
|   |-- simulate_errors.py           Injects errors of a target frequency,
|   |                                 correlation, and rate into each variant
|   `-- import_population.py         Loads one (population, f, corr, r)
|                                     combination as a simulation-ready frame
|-- simulation/
|   |-- inclusion_probability.py     Certainty-unit (HV) assignment
|   |-- selection.py                 PPS sample selection (systematic / python)
|   |-- sample.py                    Operations for one sample draw
|   |-- precision_estimation.py      Precision/bound estimators
|   |-- sample_size_calculation.py   Analytical sample-size formulas
|   |-- validations.py               Validation helpers (available for ad-hoc
|   |                                 use; see [Validation](#validation))
|   |-- run_sim.py                   Monte Carlo orchestration for main.py
|   |-- sample_planning_analysis.py  Monte Carlo orchestration for
|   |                                 run_sensitivity_analysis.py
|   `-- aggregate_metrics.py         Aggregates metrics across population
|                                     configurations
|-- testing/                         Diagnostic and reporting notebooks (manual)
`-- results/                         Generated simulation workbooks
```

There is no `docs/` folder and no automated test suite in the current
checkout; see [Testing](#testing) below.

## Requirements

- Python 3.12 is recommended.
- The population workbooks under `clean_data/` must be available locally
  (see [Population pipeline](#population-pipeline)).

Install the pinned dependencies from `requirements.txt`.

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run all commands below from the repository root. File locations are resolved
relative to `config.py`, so execution does not depend on the shell's current
working directory once the project has been started correctly.

## Population pipeline

Populations are synthetic, built from a real book-value column in three steps:

1. **`create_population/simulate_book_values.py`** reads
   `clean_data/first_book_value_population.xlsx` and rescales the book-value
   column into three variants, each with a different share of total value
   concentrated in "above-SI" items (`SI = total / 100`):
   `BV_5pct_above_SI`, `BV_15pct_above_SI`, `BV_30pct_above_SI`. Output:
   `clean_data/book_value_populations.xlsx`.
2. **`create_population/simulate_errors.py`** reads that workbook and, for
   every combination of error frequency (`f`), book-value/error correlation
   (`corr`), and total error rate (`r`), injects errors into each variant
   under the constraint that no item's error can exceed its own book value.
   Feasibility is checked up front; infeasible `(population, f, r)`
   combinations are skipped and logged. Output:
   `clean_data/simulated_error_populations.xlsx`, with one `Detail_<variant>`
   sheet per book-value variant (e.g. `Detail_BV_5pct_above_SI`) plus a
   `Summary` sheet.
3. **`create_population/import_population.py`** (`import_population`) loads
   one `(BV_pop, f_target, corr_target, r_target)` combination from that
   workbook and returns a simulation-ready `pd.DataFrame` with columns:

| Column | Meaning |
|---|---|
| `BV` | Book value |
| `E` | Injected monetary error |
| `ER` | Error rate / tainting (`E / BV`) |


## Configuration

`config.py` defines:

- `POPULATION_CONFIGS`: the grid of `BV_pop` / `f_target` / `corr_target` /
  `r_target` values. `main.py` iterates over the full grid;
  `run_sensitivity_analysis.py` iterates over the same `BV_pop` /
  `f_target` / `corr_target` values but narrows `r_target` to
  `[0.002, 0.01]` (the values below the 2% materiality threshold used for
  sample-size planning).
- `SIMULATION_SETTINGS`: passed to `simulation.run_sim.Simulation` (used by
  `main.py`). Shape:

```python
SIMULATION_SETTINGS = {
    "sample_sizes": [30, 65, 100, 150, 200],
    "CL": [0.80, 0.90, 0.95],
    "iterations": 10_000,
    "seed": 120,
    "TE_perc": 0.02,
    "configurations": [
        {"method": "MUS", "hv_selection": "nothing",   "selection_type": "systematic_sampling", "bound_estimator": "Poisson_Stringer"},
        {"method": "MUS", "hv_selection": "nothing",   "selection_type": "systematic_sampling", "bound_estimator": "Binomial_Stringer"},
        {"method": "MUS", "hv_selection": "nothing",   "selection_type": "systematic_sampling", "bound_estimator": "Moment_jfa_inventory"},
        {"method": "MUS", "hv_selection": "iterative", "selection_type": "systematic_sampling", "bound_estimator": "HH"},
        # Mod_HH is implemented (see Estimator status) but left out of the
        # active grid here; uncomment to include it in a run.
    ],
}
```

- `SAMPLE_PLANNING_SIMULATION_SETTINGS`: passed to
  `simulation.sample_planning_analysis.Simulation` (used by
  `run_sensitivity_analysis.py`). Instead of a fixed `sample_sizes` list, the
  sample size for each run is computed from `sample_size_calculation.py`
  using an anticipated-error and anticipated-std percentage of the true
  values (`anticipated_errors`, `anticipated_stds`,
  `sample_size_combinations`), so this settings dict evaluates how sample
  planning behaves as those anticipations deviate from the truth. Only
  bound estimators with an analytical sample-size formula (`HH`,
  `Poisson_Stringer`, `Binomial_Stringer`) can appear in its
  `configurations` list, since the formula is what picks each run's `n`.
  When a formula-derived sample size is not usable — infeasible (e.g.
  Poisson_Stringer's `TE <= AE * EF`) or at/above the population size —
  `Simulation` records the reason in that row's `obs` column instead of
  running it (see `_infeasible_reason` / `_infeasible_metrics_row` in
  `sample_planning_analysis.py`).

### Configuration fields

| Field | Accepted values | Description |
|---|---|---|
| `method` | `MUS` | Defines the measure of size `Q` (only MUS is exercised currently) |
| `hv_selection` | `nothing`, `iterative` | Controls advance separation of certainty/high-value units |
| `selection_type` | `systematic_sampling`, `python` | Selects the PPS drawing implementation |
| `bound_estimator` | `HH`, `Mod_HH`, `Poisson_Stringer`, `Binomial_Stringer`, `Moment`, `Moment_jfa_inventory`, `Moment_jfa_accounts` | Selects the precision estimator (see [Estimator status](#estimator-status)) |

`CL` is a *list* of confidence levels; every sample size (or anticipated-error
combination) is run once per entry in `CL`, and the normal critical value is
`scipy.stats.norm.ppf(CL)` for each.

## Running the simulations

### Full Monte Carlo simulation

```powershell
python main.py
```

`main.py` loads every population in `POPULATION_CONFIGS` up front in the
parent process (so worker processes never touch the source workbook, and
`import_population`'s cache is reused across populations that share a
`BV_pop` sheet), then runs one population per worker via
`ProcessPoolExecutor` (defaults to all logical cores; pass
`main(max_workers=N)` to limit it). For each population, its worker:

1. Runs `simulation.run_sim.Simulation` over every `(sample_size, CL,
   configuration)` combination in `SIMULATION_SETTINGS`, for `iterations`
   Monte Carlo draws each.
2. Exports that population's own workbook to
   `results/results_<population_ID>.xlsx` via `Simulation.export()`.

`testing/export_main.ipynb` then concatenates every `results/results_*.xlsx`
file's `metrics` sheet, derives `BV_pop` / `f_target` / `corr_target` /
`r_target` back out of the `Population ID` string, adds the relative/percent
metrics also produced by `simulation/aggregate_metrics.py`, and writes the
combined report (`metrics` plus one `agg_<column>` sheet per grouping column)
used for cross-population analysis.

### Sample-size planning sensitivity analysis

```powershell
python run_sensitivity_analysis.py
```

Structured the same way as `main.py` (populations loaded once in the parent
process, one `ProcessPoolExecutor` worker per population), but each worker
runs `simulation.sample_planning_analysis.Simulation` over
`SAMPLE_PLANNING_SIMULATION_SETTINGS` for its one population and returns that
population's metrics rather than writing its own file. The parent process
concatenates every population's metrics and writes a single combined
workbook to `results/sensitivity_analysis_results.xlsx`, with a `metrics`
sheet and one `agg_<column>` sheet per grouping column (`BV_pop`,
`f_target`, `corr_target`, `r_target`, `confidence_level`, `sample_size`) —
this aggregation step is built into the script itself, unlike `main.py`'s
combined report, which is produced separately by `export_main.ipynb`.

For a smaller verification run, reduce `iterations` and the size of the
relevant lists in `config.py` before starting a full experiment.

## Output

`Simulation.export()` (used by `main.py`) writes
`results/results_<population_ID>.xlsx` with sheets:

| Sheet | Contents |
|---|---|
| `results (€)` | One row per Monte Carlo iteration and configuration |
| `descriptive statistics (€)` | Descriptive statistics grouped by configuration |
| `metrics` | Bias, precision, coverage, inconclusiveness, sample-size, and skew metrics |

Iteration-level output (`Sample.get_results()`, one row per Monte Carlo draw)
includes:

- `EE_pred`: estimated population error;
- `SE_pred`: estimated precision (the value actually used for `ULE_pred`);
- `ULE_pred`: upper error limit (`EE_pred + SE_pred`, except where a
  zero-error rescue rule substitutes a different `EE`/`SE` pair — see
  [Precision estimators](#precision-estimators));
- `ULE_HH`: the upper limit from the estimator's main (non-rescued) formula
  alone, for every bound estimator — reported regardless of whether the
  rescue rule fired, so `ULE_pred == ULE_HH` identifies iterations where it
  did not;
- `real_n`: realised sample size (systematic PPS selection can deduplicate
  to slightly below the intended `sample_size`);
- `number_errors`: number of erroneous items drawn into the non-certainty
  stratum;
- `sample_std_dev`, `sample_mean`, `sample_min`, `sample_max`: descriptive
  statistics of `E` within the non-certainty sample.

The `metrics` sheet (one row per `(sample_size or anticipated-error, CL,
configuration)` combination) aggregates these across all iterations of that
combination: bias, precision, coverage, inconclusiveness, sample-size
formula comparisons, and skew. For `HH`, it additionally splits `Coverage`
and `Rate of Acceptance` by whether the zero-error rescue rule fired
(`Coverage (rule applied)` / `Coverage (rule NOT applied)`, and the `Rate of
Acceptance` equivalents), alongside `Rate rule applied` (the fraction of
iterations where it did).

### Coverage definition

Coverage is evaluated only against the upper limit:

```text
coverage = proportion of iterations where ULE_pred >= true population error
```

## Estimator status

| Estimator | Status |
|---|---|
| `HH` | Implemented (precision + analytical sample size) |
| `Mod_HH` | Implemented (precision only); not included in the active `SIMULATION_SETTINGS` grid in `config.py`, but usable by uncommenting its entry there |
| `Poisson_Stringer` | Implemented (precision + analytical sample size) |
| `Binomial_Stringer` | Implemented (precision + analytical sample size) |
| `Moment` | Implemented (Cornish-Fisher moment bound, precision only; no analytical sample-size formula) |
| `Moment_jfa_inventory` | Implemented (moment bound, jfa `m.type="inventory"` tail-tainting variant; precision only) |
| `Moment_jfa_accounts` | Implemented (moment bound, jfa `m.type="accounts"` tail-tainting variant; precision only) |

Selecting an estimator without an analytical sample-size implementation is
fine for `main.py` (the formula-based `Needed n` / `Formula n` metrics are
simply `NaN`), but `run_sensitivity_analysis.py` requires one, since it uses
the formula to *pick* each run's sample size — its `configurations` list is
therefore limited to `HH`, `Poisson_Stringer`, and `Binomial_Stringer`.

## Precision estimators

All bound estimators in `simulation/precision_estimation.py` share the same
error projection (`Sample.estimate_error`): a certainty-stratum sum `EEe`
plus a PPS ratio-estimator projection `SI * Σ(E/BV)` over the non-certainty
sample. They differ in how they turn that projection into a precision
figure (`SE`) and upper limit (`ULE`):

- **`HH`** computes a classical variance-based term, `SE_main = z * sr *
  BVs / sqrt(ns)` (`sr` = sample standard deviation of `E/BV` in the
  non-certainty stratum), alongside a fixed Binomial-Stringer-style "basic
  precision" floor, `SE_spec = SI * ns * Beta⁻¹(cl; 1, ns)`. Whenever the
  non-certainty sample contains at least one error, `SE_main`/`ULE_main` is
  used; when it contains none, `SE_spec` is used instead (`EE + SE_main`
  would otherwise collapse to the point estimate itself, since `sr = 0`).
  `ULE_HH` in the output always reports `EE + SE_main` regardless of which
  branch was actually used, so it can be compared against `ULE_pred` to
  identify rescued iterations (see [Output](#output)).
- **`Mod_HH`** uses the same two terms, but applies a finite-population-style
  correction to the variance term (`SE_main` scaled by `sqrt((BVs -
  sample_s["BV"].sum()) / BVs)`) and always compares both branches
  (`ULE = max(ULE_main, ULE_spec)`), rather than switching purely on whether
  any error was observed.
- **`Poisson_Stringer`** and **`Binomial_Stringer`** build `SE` from a basic
  precision term (`SI` times a Poisson- or Beta-distribution reliability
  factor) plus an incremental-adjustment sum over the sample's observed
  taints, ranked and weighted by the respective distribution's quantiles —
  the classical Stringer-bound construction, independent of any assumed
  standard deviation.
- **`Moment`**, **`Moment_jfa_inventory`**, and **`Moment_jfa_accounts`**
  compute a Cornish-Fisher-expansion upper bound from the sample's taint
  moments, following the `jfa` R package's `.moment()` method; the three
  differ only in the hypothetical-tainting term `tstar` used to stabilise
  the bound when no (or few) errors are observed — see the docstrings in
  `precision_estimation.py` for the exact `tstar` definitions.

## Validation

`simulation/validations.py` is a standalone library of validation helpers
(population schema, configuration compatibility, HV/sample-design checks,
simulation-result invariants, etc.) with no side effects — each function
raises on the invariant it checks and otherwise returns `None`. Only
`validation_NAs` is called automatically, from `run_sim.py` and
`sample_planning_analysis.py`, right after each combination's iteration
results are assembled; the rest of the module is available for ad-hoc use
(interactive checks, notebooks) rather than wired into the simulation loop
itself.

## Testing

`testing/` holds manual, diagnostic, and reporting notebooks rather than an
automated suite — there is currently no automated test suite in this
checkout. `export_main.ipynb` combines `main.py`'s per-population output
into the report described in
[Running the simulations](#running-the-simulations); the
`test_hh_special_case*`, `test_moment_vs_stringer.ipynb`, and
`test_precision.ipynb` notebooks are ad-hoc investigations into individual
estimators' behaviour, built directly against the current
`create_population` / `simulation` modules. `test_data.ipynb` predates the
current module layout (it calls `synthetic_population_eu_funds`, which no
longer exists) and is not runnable as-is.

## Research-use note

This repository implements an experimental simulation framework for
research. Estimator formulas, certainty-unit treatment, and configuration
compatibility should be independently validated against the applicable audit
methodology before results are used operationally.
