# MUS Sampling Simulation

This project studies monetary unit sampling (MUS) audit precision through
Monte Carlo simulation, using synthetic audit populations with injected
errors of known frequency, correlation, and total rate.

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
|   |-- validations.py               Optional validation helpers (not wired
|   |                                 into the simulation loop; see below)
|   |-- run_sim.py                   Monte Carlo orchestration for main.py
|   |-- sample_planning_analysis.py  Monte Carlo orchestration for
|   |                                 run_sensitivity_analysis.py
|   `-- aggregate_metrics.py         Aggregates metrics across population
|                                     configurations
|-- testing/                         Exploratory notebooks (manual, not automated)
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
  `r_target` values that both entry points iterate over.
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
        {"method": "MUS", "hv_selection": "nothing",   "selection_type": "systematic_sampling", "bound_estimator": "Moment"},
        {"method": "MUS", "hv_selection": "iterative", "selection_type": "systematic_sampling", "bound_estimator": "HH"},
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
  planning behaves as those anticipations deviate from the truth.

### Configuration fields

| Field | Accepted values | Description |
|---|---|---|
| `method` | `MUS` | Defines the measure of size `Q` (only MUS is exercised currently) |
| `hv_selection` | `nothing`, `iterative` | Controls advance separation of certainty/high-value units |
| `selection_type` | `systematic_sampling`, `python` | Selects the PPS drawing implementation |
| `bound_estimator` | `HH`, `Mod_HH`, `Poisson_Stringer`, `Binomial_Stringer`, `Moment` | Selects the precision estimator (see [Estimator status](#estimator-status)) |

`CL` is a *list* of confidence levels; every sample size (or anticipated-error
combination) is run once per entry in `CL`, and the normal critical value is
`scipy.stats.norm.ppf(CL)` for each.

## Running the simulations

### Full Monte Carlo simulation

```powershell
python main.py
```

For every combination in `POPULATION_CONFIGS`, `main.py`:

1. Loads the corresponding population via `import_population`.
2. Runs `simulation.run_sim.Simulation` over every `(sample_size, CL,
   configuration)` combination in `SIMULATION_SETTINGS`, for `iterations`
   Monte Carlo draws each.
3. Exports one workbook per population to
   `results/results_<population_ID>.xlsx` via `Simulation.export()`.

### Sample-size planning sensitivity analysis

```powershell
python run_sensitivity_analysis.py
```

For every combination in `POPULATION_CONFIGS`, this script runs
`simulation.sample_planning_analysis.Simulation` over
`SAMPLE_PLANNING_SIMULATION_SETTINGS`, collects metrics across all
populations, and writes a single combined workbook to
`results/sensitivity_analysis_results.xlsx` with `metrics` and
`aggregate_metrics` sheets (via `simulation/aggregate_metrics.py`).

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

Iteration-level output includes:

- `EE_pred`: estimated population error;
- `SE_pred`: estimated precision or bound;
- `VAR_pred`: estimated variance;
- `ULE_pred`: upper error limit;
- `real_n`: real sample size;
- `number_errors`: number of erroneous items drawn.

### Coverage definition

Coverage is evaluated only against the upper limit:

```text
coverage = proportion of iterations where ULE_pred >= true population error
```

## Estimator status

| Estimator | Status |
|---|---|
| `HH` | Implemented (precision + analytical sample size) |
| `Mod_HH` | Precision implemented; commented out of `SIMULATION_SETTINGS` pending review of the book-value correction factor described below |
| `Poisson_Stringer` | Implemented (precision + analytical sample size) |
| `Binomial_Stringer` | Implemented (precision only; no analytical sample-size formula yet) |
| `Moment` | Implemented (moment bound, precision only; no analytical sample-size formula yet) |

Selecting an estimator without an analytical sample-size implementation is
fine for `main.py` (the formula-based `Needed n` / `Formula n` metrics are
simply `NaN`), but `run_sensitivity_analysis.py` requires one, since it uses
the formula to *pick* each run's sample size.

## Modified-HH correction factor

`precision_modified_HH` (in `simulation/precision_estimation.py`) reduces
estimated precision according to the proportion of non-certainty book value
represented by the sample. Define:

- `sr` as the sample standard deviation of `ER`;
- `BVs` as total `BV` in the non-certainty population;
- `ns` as the non-certainty sample size;
- `Bs` as `sample_s["BV"].sum()`;
- `z` as the normal critical value.

The uncorrected variance is:

```text
V0 = (BVs * sr)^2 / ns
```

The correction factor applied by the current `SE` formula is:

```text
c = sqrt((BVs - Bs) / BVs)
```

but `VAR` is still returned as the *uncorrected* `V0`, i.e. `SE` and `VAR` are
not derived from the same quantity (`SE != z * sqrt(VAR)`). This is why
`Mod_HH` is commented out of `SIMULATION_SETTINGS` in `config.py` — the two
outputs need to be reconciled (either apply `c**2` to `VAR` too, or drop the
correction from `SE`) before this estimator is used for real analysis.

## Known limitations

- **`validations.py` is not wired into the simulation loop.** Only
  `validation_NAs` is actually called (from `run_sim.py` and
  `sample_planning_analysis.py`); the rest of the module (population schema,
  configuration compatibility, HV/sample-design checks, etc.) is available
  for ad-hoc use but does not run automatically.
- **`config.py`'s `RF_TABLE`** (loaded from `clean_data/reliability
  factor.xlsx` at import time) is not referenced anywhere in the codebase —
  the implemented Stringer bounds compute their factors analytically via
  `scipy.stats.gamma`/`beta` quantiles rather than a lookup table. Every
  import of `config` still depends on that workbook being present.
- **Formula-derived sample sizes are not capped against the population
  size.** `run_sensitivity_analysis.py` picks each run's sample size from
  `sample_size_calculation.py`. For anticipated-error/std inputs far from a
  population's actual values, that formula can return an `n` far larger than
  the population itself; `iterative_hv_selection` then classifies every unit
  as a certainty unit, leaving no non-certainty units to sample and causing
  `simulation.selection.systematic_samping` to fail. Sanity-check
  `anticipated_errors` / `anticipated_stds` against a population's actual
  error rate and `ratio_EQ_std` before adding it to
  `SAMPLE_PLANNING_SIMULATION_SETTINGS`.

## Testing

`testing/test_data.ipynb` and `testing/test_simulation.ipynb` are exploratory
notebooks for manual inspection. They predate the current
`create_population` / `simulation` module layout (they call functions such as
`synthetic_population_eu_funds` that no longer exist) and are not runnable
as-is. There is currently no automated test suite in this checkout.

## Research-use note

This repository implements an experimental simulation framework for
research. Estimator formulas, certainty-unit treatment, and configuration
compatibility should be independently validated against the applicable audit
methodology before results are used operationally.
