import pandas as pd
import numpy as np

def aggregate_metrics(metrics_df: pd.DataFrame, analysis: str = "main", TE_perc: float = 0.2) -> pd.DataFrame:
    """
    Aggregates the metrics DataFrame by calculating the mean of each metric for each unique combination of parameters.

    Parameters:
    - metrics_df (pd.DataFrame): The DataFrame containing the metrics to be aggregated.

    Returns:
    - pd.DataFrame: A new DataFrame with the aggregated metrics.
    """

    # Get rates of Correct/Incorrect acceptance and rejection based on materiality
    metrics_df["Correct Acceptance"] = np.where(metrics_df["Population Error Rate"] <= TE_perc, metrics_df["Rate of Acceptance"], np.nan)
    metrics_df["Incorrect Rejection"] = np.where(metrics_df["Population Error Rate"] <= TE_perc, metrics_df["Rate of Acceptance"], np.nan)
    metrics_df["Incorrect Acceptance"] = np.where(metrics_df["Population Error Rate"] > TE_perc, metrics_df["Rate of Acceptance"], np.nan)
    metrics_df["Correct Rejection"] = np.where(metrics_df["Population Error Rate"] > TE_perc, metrics_df["Rate of Acceptance"], np.nan)

    # Define the columns to group by
    group_cols = ["BV_pop", "f_target", "corr_target", "r_target"]

    if analysis =="main":
        # Calculate the relative/% version of the metrics
        metrics_df["Relative Bias of Error Estimation"] = metrics_df["Bias of Error Estimation"] / metrics_df["Average Error Estimation"]
        metrics_df["Relative Precision of Error Estimation"] = metrics_df["Precision of Error Estimation"] / metrics_df["Population Error Amount"]
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

    # Group by the specified columns and calculate the mean for each metric
    aggregated_metrics = (
                metrics_df
                .groupby(group_cols)[value_cols]
                .describe()
                .T
            )

    return aggregated_metrics
