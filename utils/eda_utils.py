"""
This module provides utility functions for performing exploratory data analysis (EDA)
on pandas DataFrames. The functions help retrieve group-specific values, identify and
fix inconsistencies, check for missing values, an so on.

These utilities streamline the EDA process by automating common data quality checks
and facilitating data preparation tasks.
"""

import pandas as pd
import numpy as np
from typing import Union


def check_missing_values(df: pd.DataFrame, missing_value: int = None) -> None:
    """
    Checks for missing values in the given DataFrame and prints the results.
    Optionally includes a specific value as a missing indicator.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be checked for missing values.
    missing_value : int, optional
        A value to consider as missing, in addition to NaN. Default is None.
    """
    missing_na = df.isna().sum()
    if missing_value is not None:
        missing_custom = (df == missing_value).sum()
        missing_na = missing_custom

    if missing_na.any():
        print("Missing value counts by column:")
        print(missing_na)
    else:
        print("The dataset does not contain any missing values.")


def check_duplicates(df: pd.DataFrame, column_names: list = None) -> pd.DataFrame:
    """
    Checks for duplicate rows in the DataFrame based on the specified column names.
    If no column names are provided, checks for duplicates across the entire DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be checked for duplicates.
    column_names : list, optional
        List of column names to check for duplicates. Defaults to None, meaning full
        rows will be checked.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the duplicate rows. If no duplicates are found,
        an empty DataFrame is returned.
    """
    if column_names is None:
        column_names = df.columns.tolist()

    duplicate_rows = df[df.duplicated(subset=column_names, keep=False)]

    if not duplicate_rows.empty:
        print(
            f"There are {len(duplicate_rows)} duplicate rows based on the columns: "
            f"{column_names}"
        )
        return duplicate_rows
    else:
        print(f"No duplicate rows found based on the columns: {column_names}")


def find_outlier_rows_by_iqr(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Identify and return rows in a DataFrame that contain outliers based on the
    Interquartile Range (IQR) method.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to analyze for outliers.
    columns : list, optional
        A list of column names to check for outliers. If None (default), the function
        will analyze all numerical columns (int64 and float64).

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing only the rows from the original DataFrame that are
        identified as having outliers in any of the specified columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    outlier_mask = pd.Series(False, index=df.index)

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        col_outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_mask |= col_outlier_mask

    return df[outlier_mask]


def filter_correlation_indices(
    corr_matrix: pd.DataFrame, corr_range: list[float, float]
) -> Union[pd.DataFrame, str]:
    """
    Filters correlations in the lower triangle of the correlation matrix based
    on the given range.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        A DataFrame representing the correlation matrix.
    corr_range : list[float, float]
        A list specifying the range for filtering correlations.
        The first value is the lower bound (exclusive), and the second value is
        the upper bound (inclusive).

    Returns
    -------
    Union[pd.DataFrame, str]
        A DataFrame containing pairs of features with their correlation indices
        within the given range. If no correlations are found within the range,
        a string message is returned.
    """
    lower_triangle = corr_matrix.where(
        np.tril(np.ones(corr_matrix.shape), k=-1).astype(bool)
    )
    filtered_corr_series = (
        lower_triangle[
            (lower_triangle > corr_range[0]) & (lower_triangle <= corr_range[1])
        ]
        .stack()
        .sort_values(ascending=False)
    )

    filtered_corr_df = filtered_corr_series.reset_index().rename(
        columns={"level_0": "feature_1", "level_1": "feature_2", 0: "correlation_index"}
    )

    return (
        filtered_corr_df
        if len(filtered_corr_df)
        else f"There are no values within {corr_range} range."
    )


def find_typical_range(
    df: pd.DataFrame,
    numerical_columns: list[str],
    bins: list[Union[int, float]],
    labels: list[str],
) -> pd.DataFrame:
    """
    Identifies the typical range (bin) for each numerical column in the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing numerical data.
    numerical_columns : List[str]
        List of column names in the DataFrame to analyze.
    bins : List[Union[int, float]]
        A list of bin edges to categorize the maximum values in each column.
    labels : List[str]
        Labels corresponding to the bins for categorization.

    Returns
    -------
    pd.DataFrame
        A DataFrame with column names as the index and their respective typical
        ranges (bins) as values.
    """
    typical_ranges = {}
    for col in numerical_columns:
        max_val = df[col].max()
        max_bin = pd.cut([max_val], bins=bins, labels=labels, right=False)[0]
        typical_ranges[col] = max_bin

    typical_range_df = pd.DataFrame(
        {"typical_range": list(typical_ranges.values())}, index=typical_ranges.keys()
    )

    return typical_range_df
