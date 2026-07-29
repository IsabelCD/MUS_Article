# MUS and MRS Population Sampling Simulations

This project compares monetary unit sampling (MUS) with monetary risk sampling
(MRS) through Monte Carlo simulation. It builds a synthetic audit population
from historical EU-funds data, draws probability-proportional-to-size samples,
estimates population error and precision, and exports iteration-level results
and summary metrics to Excel.

For MUS, the measure of size is book value. For MRS, book value is adjusted by
an estimated probability of error:

```text
MUS: Q = BV
MRS: Q = BV * P
```

where `BV` is book value and `P` is the modelled probability of error.

## Project structure

```text
.
|-- main.py                         Default simulation entry point
|-- config.py                       Project-relative data and result paths
|-- requirements.txt                Pinned Python dependencies
|-- clean_data/                     Source population workbooks
|-- create_population/
|   |-- error_predictions.py        Logistic error-risk model
|   `-- pop_from_eu_funds.py        EU-funds synthetic population builder
|-- simulation/
|   |-- inclusion_probability.py    Certainty-unit/HV assignment
|   |-- selection.py                PPS sample selection
|   |-- sample.py                   Operations for one sample draw
|   |-- precision_estimation.py     Precision and bound estimators
|   |-- sample_size_calculation.py  Analytical sample-size formulas
|   |-- validations.py              Population and result validation
|   `-- run_sim.py                  Monte Carlo orchestration and export
|-- testing/                         Exploratory verification notebooks
|-- results/                         Generated simulation workbooks
`-- docs/                            Supporting project documentation
```

Generated files in `results/` are outputs and should not be treated as source
data.

## Requirements

- Python 3.12 is recommended.
- The source population workbook must be available locally.
- The conservative estimator additionally requires a reliability-factor table;
  see [Conservative estimator data](#conservative-estimator-data).

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

## Input data

The default entry point reads:

```text
clean_data/all_dfs_w_2526_population.xlsx
```

The EU-funds population-building path expects the following common columns:

- `fund`, `year`, `moment`, `stratum`, and `code_operation`;
- `in_sample` and `high_value`;
- `total_certified_cost`;
- `error_binary`, `error_rate`, and `error_monetary_value`.

It also expects the model features selected for the requested fund.

For `FEDER_FC`:

- `typology_of_action_agg`;
- `legal_nature_beneficiary_agg`;
- `no_programs_of_operations_beneficiary_managed_cumulative_agg`;
- `operational_program`.

For `FSE`:

- `typology_of_action_agg`;
- `operational_program`;
- `op_debt_status_lvl3_agg`;
- `type_of_support`;
- `legal_nature_beneficiary_lvl2_agg`;
- `no_operations_managed_by_beneficiary_cumulative_agg`.

After population construction, the simulation expects one row per artificial
population unit with these columns:

| Column | Meaning |
|---|---|
| `BV` | Book value |
| `E` | Monetary error |
| `ER` | Error rate or tainting |
| `P` | Predicted probability of error |

The simulation derives `Q` from these fields according to MUS or MRS.

## Configuration

`Simulation` receives a `simulation_config` dictionary with a shared list of
sample sizes and an explicit list of method configurations:

```python
simulation_config = {
    "sample_sizes": [30, 65, 100],
    "CL": 0.80,
    "iterations": 5000,
    "seed": 120,
    "TE_perc": 0.02,
    "configurations": [
        {
            "method": "MUS",
            "hv_selection": "nothing",
            "selection_type": "systematic_sampling",
            "bound_estimator": "Con",
        },
        {
            "method": "MUS",
            "hv_selection": "iterative",
            "selection_type": "systematic_sampling",
            "bound_estimator": "HH",
        },
        {
            "method": "MRS",
            "hv_selection": "iterative",
            "selection_type": "systematic_sampling",
            "bound_estimator": "HH",
        },
    ],
}
```

Each sample size is run for every entry in `configurations`.

### Configuration fields

| Field | Accepted values | Description |
|---|---|---|
| `method` | `MUS`, `MRS` | Defines the measure of size `Q` |
| `hv_selection` | `nothing`, `iterative` | Controls advance separation of certainty/high-value units |
| `selection_type` | `systematic_sampling`, `python` | Selects the PPS drawing implementation |
| `bound_estimator` | `Con`, `HH`, `Mod_HH`, `Ratio` | Selects the precision and analytical sample-size method |

The intended supported combinations, once the blocking integration issues in
[Implementation status](#implementation-status) are resolved, are:

- MUS + iterative HV selection + systematic sampling + HH;
- MRS + iterative HV selection + systematic sampling + HH;
- MUS + systematic sampling + conservative bounds, once the required
  reliability-factor data and certainty-unit policy have been supplied.

See [Implementation status](#implementation-status) before enabling other
combinations.

### Simulation parameters

```python
simulation = Simulation(
    population_ID="FEDER_FC",
    population=population,
    simulation_config=simulation_config,
)
```

| Parameter | Description |
|---|---|
| `population_ID` | Identifier used in messages and the output filename |
| `population` | DataFrame containing `BV`, `E`, `ER`, and `P` |
| `simulation_config` | Sample sizes, global settings, and method configurations |
| `simulation_config["CL"]` | Central confidence level used to obtain the normal critical value |
| `simulation_config["seed"]` | Base seed for shuffling and sample selection |
| `simulation_config["iterations"]` | Monte Carlo draws per sample-size/configuration combination |
| `simulation_config["TE_perc"]` | Tolerable error as a proportion of total book value |

With `CL=0.80`, the code uses the 90th percentile of the standard normal
distribution for the upper endpoint of a central 80% interval. Consequently,
the nominal one-sided coverage of that upper endpoint is 90%. If `CL` is meant
to represent one-sided coverage directly, the critical-value definition must
be changed in the implementation.

## Running the simulation

> **Development status:** the current checkout is under active refactoring and
> still has unresolved argument wiring between `Simulation`, `Sample`, and the
> selection dispatcher. As a result, `main.py` does not yet complete an
> end-to-end run. The command below describes the intended entry point after
> those blocking issues are corrected.

Review the file paths, fund, sample sizes, and method configurations in
`main.py`, then run:

```powershell
python main.py
```

The default workflow:

1. Reads the source workbook from `clean_data/`.
2. Trains a logistic model on historical sampled records.
3. Predicts error probabilities for the selected population year.
4. Constructs the artificial population.
5. Computes the MUS or MRS measure of size.
6. Draws and evaluates the configured samples over all iterations.
7. Calculates configuration-level performance metrics.
8. Exports the results to Excel.

For a smaller verification run, reduce `iterations` and use one small sample
size before starting the full experiment.

## Output

`Simulation.export()` writes:

```text
results/resultados_<population_ID>.xlsx
```

The workbook contains:

| Sheet | Contents |
|---|---|
| `resultados (EUR)` | One row per Monte Carlo iteration and configuration |
| `estatisticas descritivas (EUR)` | Descriptive statistics grouped by configuration |
| `metricas` | Bias, precision, coverage, inconclusiveness, sample-size, and skew metrics |

Depending on the Excel writer and source encoding, the displayed sheet names
may use the euro symbol and Portuguese accents.

Iteration-level output includes:

- `EE_pred`: estimated population error;
- `SE_pred`: estimated precision or bound;
- `VAR_pred`: estimated variance;
- `LLE_pred` and `ULE_pred`: lower and upper error limits;
- `real_n`: real sample size;
- configuration and iteration identifiers.

### Coverage definition

Coverage is intentionally evaluated only against the upper limit because the
upper error limit is the relevant decision quantity in this audit context:

```text
coverage = proportion of iterations where ULE_pred >= EE_true
```

The lower limit is not part of the coverage metric.

## Modified-HH correction factor

The Modified-HH estimator currently reduces estimated precision according to
the proportion of non-certainty book value represented by the sample. Define:

- `sr` as the sample standard deviation of `ER`;
- `Qs` as total `Q` in the non-certainty population;
- `ns` as the non-certainty sample size;
- `Bs` as `sample_s["BV"].sum()`;
- `z` as the normal critical value.

The uncorrected variance is:

```text
V0 = (Qs * sr)^2 / ns
```

The correction factor applied by the current `SE` formula is:

```text
c = sqrt((Qs - Bs) / Qs)
  = sqrt(1 - Bs / Qs)
```

This factor approaches one when the sample covers little of the non-certainty
book value and approaches zero as coverage approaches the entire stratum. In
this project, `SE_pred` includes the critical value and therefore represents a
precision or margin-of-error amount rather than the raw standard deviation of
the estimator.

### Keep the correction and correct `VAR`

If this book-value correction is part of the intended Modified-HH method, it
must also be included in variance. Because variance uses the square of the
correction factor:

```text
VAR_corrected = V0 * c^2
              = ((Qs * sr)^2 / ns) * ((Qs - Bs) / Qs)

SE = z * sqrt(VAR_corrected)
```

An internally consistent implementation is:

```python
remaining_fraction = (Qs - sample_s["BV"].sum()) / Qs

if not 0 <= remaining_fraction <= 1:
    raise ValueError(
        "Modified-HH correction must lie between 0 and 1."
    )

VAR = ((Qs * sr) ** 2 / ns) * remaining_fraction
SE = z_score * np.sqrt(VAR)
```

### Remove the correction and retain the current `VAR`

If the existing uncorrected variance is the intended formula, remove the
factor from `SE` instead:

```text
VAR = (Qs * sr)^2 / ns
SE  = z * sqrt(VAR)
    = z * Qs * sr / sqrt(ns)
```

For MUS, `Q = BV` and `E / Q = ER`. Removing the correction therefore makes
Modified-HH effectively identical to the current HH estimator. If
Modified-HH is intended to differ through population-coverage adjustment, the
first option—correcting `VAR`—is the internally consistent choice.

The present correction is dimensionally meaningful for MUS because `Qs` and
`Bs` are both book-value amounts. For MRS, `Qs = sum(BV * P)`, so subtracting
sampled `BV` from `Qs` mixes different quantities and can make the expression
negative. The formula should therefore be limited to MUS unless a separate
MRS-compatible correction is established. This book-value correction should
also not be confused with the conventional simple-random-sampling finite
population correction `sqrt((N - n) / (N - 1))`.

## Conservative estimator data

The conservative estimator expects:

```text
clean_data/reliability factor.xlsx
```

The workbook must contain an `RF_factor` column in the order required by the
chosen Stringer/reliability-factor methodology. The implementation also uses
`ER` from the selected sample to calculate incremental allowance. The factor
table must contain enough rows for the maximum possible number of nonzero-error
sample items.

The reliability-factor workbook is currently present in `clean_data/`. Its
schema and factor ordering should still be validated before production runs.

## Implementation status

| Component | Status |
|---|---|
| Simulation-to-sample integration | Blocked by inconsistent selector argument names |
| Systematic PPS selection | Random start is implemented with the seeded generator; dispatcher naming and certainty-unit handling still require correction |
| Python weighted selection | RNG invocation is implemented; estimator compatibility still requires validation |
| HH precision | Implemented |
| HH analytical sample size | Implemented |
| Conservative precision | Implemented; reliability-factor loading and methodology require validation |
| Conservative analytical sample size | Implemented |
| Modified HH precision | Implemented, but `SE` and `VAR` are currently inconsistent; see the correction-factor section |
| Modified HH analytical sample size | Provisionally mapped to HH and awaiting methodological confirmation |
| Ratio precision | Not implemented |
| Ratio analytical sample size | Not implemented |

Do not include an unfinished estimator in `simulation_config`. The dispatcher
currently exposes these names, but selecting an unfinished implementation will
raise an exception during or after the simulation.

## Validation and reproducibility

The simulation validates that:

- input and output DataFrames contain no missing values;
- every `Q` value is positive;
- total `Q` is positive.

The base seed is combined with configuration and iteration indices. This makes
the intended simulation design reproducible once every selection path uses the
provided NumPy random generator.

Before a production run, also verify that:

- `0 < CL < 1`;
- `iterations >= 2`;
- every sample size is smaller than the population size;
- at least two non-certainty units are selected for variance estimation;
- tolerable error exceeds the anticipated-error term required by the selected
  analytical formula;
- the method, selection design, estimator, and certainty-unit treatment are
  statistically compatible.

## Testing

The automated `unittest` suite is located in:

- `testing/test_population.py`;
- `testing/test_simulation.py`.

It uses small in-memory populations and does not read the large EU-funds source
workbook. Run it from the repository root with:

```powershell
python -m unittest discover -s testing -p "test_*.py" -v
```

The tests cover population schemas, inverse-probability expansion,
configuration compatibility, HV allocation, deterministic systematic
selection, precision inputs, result bounds, tolerable-error conditions, an HH
sample smoke test, and upper-limit-only coverage.

The exploratory notebooks `testing/test_data.ipynb` and
`testing/test_simulation.ipynb` remain available for manual inspection but are
not part of the automated suite.

At the time of this update, 24 of 27 automated tests pass. The remaining tests
identify current integration or environment problems:

- an unknown population source returns `None` instead of raising `ValueError`;
- `sample.py` imports `select_sample`, while the dispatcher currently has a
  different name in `selection.py`;
- the local virtual environment does not provide `tqdm`, preventing
  `run_sim.py` from importing during the coverage test.

Further recommended tests include:

- deterministic sampling for a fixed seed;
- exact achieved sample size;
- certainty-unit identification and removal;
- hand-calculated HH and conservative examples;
- behavior when `TE` equals or falls below expected error;
- invalid configuration rejection before the Monte Carlo loop;
- upper-limit coverage on a controlled population;
- end-to-end export with a small population and iteration count.

## Troubleshooting

### The virtual environment no longer starts

Virtual environments contain absolute interpreter paths. Delete and recreate
`.venv` if Python was upgraded, removed, or installed in a different location.

### The conservative estimator cannot find its workbook

Place `reliability factor.xlsx` in `clean_data/` and confirm it contains the
required `RF_factor` column.

### A run fails only after many iterations

Check that the selected bound estimator has both precision and analytical
sample-size implementations. `Mod_HH` currently uses a provisional HH
sample-size mapping, and `Ratio` is not complete.

### The simulation reports missing or nonpositive `Q`

For MUS, inspect `BV`. For MRS, inspect both `BV` and `P`; their product must be
strictly positive for every population unit.

## Research-use note

This repository implements an experimental simulation framework for research.
Estimator formulas, certainty-unit treatment, reliability factors, and
configuration compatibility should be independently validated against the
applicable audit methodology before results are used operationally.
