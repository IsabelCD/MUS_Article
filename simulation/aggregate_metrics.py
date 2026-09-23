import pandas as pd
import numpy as np

def aggregate_metrics(metrics_df: pd.DataFrame, analysis: str = "main", TE_perc: float = 0.02) -> dict[str, pd.DataFrame]:
    """
    Aggregates the metrics DataFrame by calculating descriptive statistics
    for each metric, one table per grouping column (e.g. one table grouped
    only by `f_target`, one grouped only by `r_target`, etc.) rather than a
    single table grouped by their combination.

    Parameters:
    - metrics_df (pd.DataFrame): The DataFrame containing the metrics to be aggregated.

    Returns:
    - dict[str, pd.DataFrame]: one aggregated table per grouping column, keyed by column name.
    """

    # Define the columns to group by
    group_cols = ["BV_pop", "f_target", "corr_target", "r_target", "confidence_level","sample_size"]

    if analysis =="main":
        # Get rates of Correct/Incorrect acceptance and rejection based on materiality
        metrics_df["Correct Acceptance"] = np.where(metrics_df["Population Error Rate"] <= TE_perc, metrics_df["Rate of Acceptance"], np.nan)
        metrics_df["Incorrect Rejection"] = np.where(metrics_df["Population Error Rate"] <= TE_perc, metrics_df["Rate of Rejection"], np.nan)
        metrics_df["Incorrect Acceptance"] = np.where(metrics_df["Population Error Rate"] > TE_perc, metrics_df["Rate of Acceptance"], np.nan)
        metrics_df["Correct Rejection"] = np.where(metrics_df["Population Error Rate"] > TE_perc, metrics_df["Rate of Rejection"], np.nan)
        # Calculate the relative/% version of the metrics
        metrics_df["Relative Bias of Error Estimation"] = metrics_df["Bias of Error Estimation"] / metrics_df["Average Error Estimation"]
        metrics_df["Relative Precision of Error Estimation"] = metrics_df["Precision of Error Estimation"] / metrics_df["Average Error Estimation"]
        metrics_df["Precision of Error Estimation in %"] = metrics_df["Precision of Error Estimation"] / metrics_df["Population Book Value"]
        metrics_df["Relative Bias of Precision Estimation"] = metrics_df["Bias of Precision Estimation"] / metrics_df["Average Precision Estimation"]
        metrics_df["Relative Precision of Precision Estimation"] = metrics_df["Precision of Precision Estimation"] / metrics_df["Average Precision Estimation"]

        value_cols = ["Average Error Estimation", "Bias of Error Estimation", "Precision of Error Estimation", "Accuracy of Error Estimation",
                    "Average Precision Estimation", "Bias of Precision Estimation", "Precision of Precision Estimation", "Accuracy of Precision Estimation",
                    "Coverage", "Inconclusive", "Samples without Errors", "Needed n", "Formula n", "Skew",
                    "Correct Acceptance","Incorrect Rejection","Incorrect Acceptance","Correct Rejection",
                    "Relative Bias of Error Estimation","Relative Precision of Error Estimation","Precision of Error Estimation in %",
                    "Relative Bias of Precision Estimation","Relative Precision of Precision Estimation"]

    elif analysis =="sensitivity":
        value_cols = ["Coverage","Inconclusive", "Samples without Errors","Correct Acceptance","Incorrect Rejection","Incorrect Acceptance","Correct Rejection"]

    # One table per grouping column, not one table grouped by their combination
    aggregated_tables = {
        group_col: metrics_df.groupby([group_col, "bound_estimator"])[value_cols].describe().T
        for group_col in group_cols
    }

    if analysis == "sensitivity":
        # anticipated_error_perc (std held at 1), anticipated_std (error held
        # at 1), and sample_size_combinations (joint error/std pairs) are
        # three separate sweeps -- see sample_planning_analysis.py's "sweep"
        # column -- that share column names but are not directly comparable.
        # Grouping the combined pool by anticipated_error_perc alone would,
        # for example, mix rows from the anticipated_error sweep (std=1)
        # with sample_size_combinations rows (std != 1) that happen to share
        # the same anticipated_error_perc value. Each sweep therefore gets
        # its own table, built only from its own rows, in the reporting
        # order: anticipated_error_perc, then anticipated_std, then the
        # sample_size_combinations pairs.
        sweep_specs = [
            ("anticipated_error_perc", "anticipated_error", ["anticipated_error_perc", "bound_estimator"]),
            ("anticipated_std", "anticipated_std", ["anticipated_std", "bound_estimator"]),
            ("sample_size_combinations", "sample_size_combination", ["anticipated_error_perc", "anticipated_std", "bound_estimator"]),
        ]
        for table_name, sweep, sweep_group_cols in sweep_specs:
            rows = metrics_df[metrics_df["sweep"] == sweep]
            aggregated_tables[table_name] = rows.groupby(sweep_group_cols)[value_cols].describe().T

    return aggregated_tables
