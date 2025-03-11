"""
A collection of statistical utility functions for hypothesis testing, confidence
interval estimation,bootstrap resampling, etc.
"""

import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency, norm, normaltest, ttest_ind, mannwhitneyu

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.utils import resample
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer

from typing import Any


def compute_confidence_intervals(
    successes: int, total: int, confidence: float = 0.95
) -> tuple[float, float]:
    """
    Computes the confidence interval for a proportion using the normal approximation
    method.

    This function calculates the confidence interval for a proportion (success rate)
    given the number of successes and total observations. It uses the normal
    approximation to the binomial distribution (Wald method).

    Parameters
    ----------
    successes : int
        The number of successful outcomes.
    total : int
        The total number of trials or observations.
    confidence : float, optional
        The confidence level for the interval, default is 0.95 (95% confidence).

    Returns
    -------
    Tuple[float, float]
        A tuple containing the lower and upper bounds of the confidence interval.
    """
    if total == 0:
        return 0, 0

    z = norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / total
    ci_range = z * np.sqrt((p_hat * (1 - p_hat)) / total)

    lower_bound = max(0, p_hat - ci_range)
    upper_bound = min(1, p_hat + ci_range)
    return lower_bound, upper_bound


def compute_vif(
    df: pd.DataFrame, target_col: str, exclude_binary: bool = True
) -> pd.DataFrame:
    """
    Computes the Variance Inflation Factor (VIF) for a given dataset to detect
    multicollinearity.

    The function calculates the VIF for each independent feature, excluding
    the target column. If `exclude_binary` is set to `True`, binary variables
    (features with only two unique values) are excluded from the analysis.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing independent variables and a target column.
    target_col : str
        The name of the target column to be excluded from the VIF computation.
        Default None if dataframe does not contain target columns
    exclude_binary : bool, optional, default=True
        Whether to exclude binary variables (features with only two unique values)
        from the computation.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the following columns:
        - `variable`: Feature names in the dataset.
        - `VIF`: The computed Variance Inflation Factor for each feature.
        - `multicollinearity_interpretation`: Interpretation of the VIF values.
    """
    if target_col not in df.columns.to_list():
        X = df
    else:
        X = df.drop(columns=[target_col])
    if exclude_binary:
        X = X.loc[:, X.nunique() > 2]
    X = X.replace([np.inf, -np.inf], np.nan).dropna()

    X = sm.add_constant(X)

    vif_data = pd.DataFrame()
    vif_data["variable"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
    ]

    vif_data["multicollinearity_interpretation"] = vif_data["VIF"].apply(
        lambda vif: (
            "No Multicollinearity"
            if vif < 1
            else (
                "Low Multicollinearity"
                if 1 <= vif < 5
                else (
                    "Moderate Multicollinearity"
                    if 5 <= vif < 10
                    else "High Multicollinearity (Consider Removing Feature)"
                )
            )
        )
    )

    return vif_data


def check_categorical_features_multicollinearity(
    df: pd.DataFrame, categorical_cols: list[str], significance_level: float
) -> pd.DataFrame:
    """
    This function performs pairwise Chi-Square tests (H₀: The two variables are
    independent.) of independence between categorical variables to determine
    if they are significantly correlated. It returns a DataFrame containing
    the test results, including the Chi-Square statistic, p-value, and significance
    determination.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing categorical variables.
    categorical_cols : list[str]
        A list of column names representing  categorical features in the dataset.
    significance_level : float
        The threshold for statistical significance (typically 0.05). If the p-value is
        below this threshold, the relationship between the two features is considered
        significant.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the results of the Chi-Square test for each pair of
        variables.
    """
    results = []

    for i, col1 in enumerate(categorical_cols):
        for j in range(i + 1, len(categorical_cols)):
            col2 = categorical_cols[j]
            contingency_table = pd.crosstab(df[col1], df[col2])
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            results.append(
                {
                    "feature1": col1,
                    "feature2": col2,
                    "chi-Square": chi2,
                    "p-value": p,
                    "dependent": "Yes" if p < significance_level else "No",
                }
            )
    results_df = pd.DataFrame(results).sort_values(by="p-value")

    return results_df


def bootstrap_bias_variance(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bootstraps: int = 100,
) -> tuple[float, float]:
    """
    Estimate bias and variance using bootstrap resampling for a given model or pipeline.

    Parameters
    ----------
    model : object
        A scikit-learn compatible model or pipeline.
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training target vector.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        Test target vector.
    n_bootstraps : int, optional
        Number of bootstrap samples, by default 100.

    Returns
    -------
    tuple[float, float]
        Bias squared and variance of the model.
    """
    bootstrap_predictions = []

    for _ in range(n_bootstraps):
        X_bootstrap, y_bootstrap = resample(X_train, y_train, random_state=_)
        model.fit(X_bootstrap, y_bootstrap)
        y_pred_bootstrap = model.predict(X_test)
        bootstrap_predictions.append(y_pred_bootstrap)

    bootstrap_predictions = np.array(bootstrap_predictions)
    mean_prediction = np.mean(bootstrap_predictions, axis=0)

    bias_squared = np.mean((y_test - mean_prediction) ** 2)
    variance = np.mean(np.var(bootstrap_predictions, axis=0))

    return round(bias_squared, 4), round(variance, 4)


def linear_model_binary_accuracy_scorer(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """
    Computes accuracy for a linear regression model used for binary classification
    using a threshold of 0.5.

    Parameters
    ----------
    y_true : np.ndarray
        The true binary labels (0 or 1).
    y_pred : np.ndarray
        The predicted continuous values from the linear regression model.

    Returns
    -------
    float
        The accuracy score (between 0 and 1).
    """
    y_pred_binary = (y_pred > 0.5).astype(int)
    return accuracy_score(y_true, y_pred_binary)


def compute_permutation_importance(
    model: BaseEstimator,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    name: str,
) -> pd.DataFrame:
    """
    Computes permutation importance for models that don't have built-in feature
    importances. Permutation importance is a technique to compute the importance
    of features in a model by evaluating how much the model's performance decreases
    when the values of a feature are randomly shuffled. It provides a model-agnostic
    measure of feature importance and can be used for any machine learning model.

    Parameters
    ----------
    model : sklearn estimator
        The machine learning model (e.g., KNN, Polynomial SVM).
    preprocessor : ColumnTransformer
        The preprocessor to encode the data.
    X_train : pd.DataFrame
        Training feature matrix.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        Test target vector.

    Returns
    -------
    importance_df : pd.DataFrame
        DataFrame containing permutation importances for each feature.
    """
    preprocessor.fit(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    encoded_feature_names = preprocessor.get_feature_names_out(
        input_features=X_train.columns
    )
    cleaned_feature_names = [
        name.replace("remainder__", "").replace("categorical__", "")
        for name in encoded_feature_names
    ]
    perm_importance = permutation_importance(
        model,
        X_test_transformed,
        y_test,
        scoring="roc_auc",
        n_repeats=30,
        random_state=42,
    )

    importance_df = pd.DataFrame(
        {
            "Feature": cleaned_feature_names,
            name: perm_importance.importances_mean,
        }
    )
    importance_df.sort_values(by=name)
    importance_df.set_index("Feature", inplace=True)
    return importance_df


def test_categorical_variables(
    df: pd.DataFrame, categorical_cols: list[str], target_col: str, alpha: float
) -> pd.DataFrame:
    """
    Performs a Chi-Square test of independence for each categorical variable
    against a binary target column, only testing categories where all
    contingency table cells have ≥5 entries.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing categorical variables and the target column.
    categorical_cols : List[str]
        A list of column names representing categorical variables to be tested.
    target_col : str
        The binary target column (0 or 1).
    alpha : float
        The significance level to determine statistical significance.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing:
        - feature: The categorical feature tested.
        - test: "Chi-Square".
        - statistic: Chi-Square test statistic.
        - p-value: The p-value.
        - conclusion: Whether the result is statistically significant at the given
        alpha level.
        - warning: Categories that were excluded due to low counts.
    """
    results = []

    for col in categorical_cols:
        contingency_table = pd.crosstab(df[col], df[target_col])

        valid_categories = contingency_table[(contingency_table >= 5).all(axis=1)].index
        df_filtered = df[df[col].isin(valid_categories)]

        if len(valid_categories) < 2:
            results.append(
                {
                    "feature": col,
                    "test": "Chi-Square",
                    "statistic": None,
                    "p-value": None,
                    "conclusion": "Not Tested",
                    "warning: too little entries": (
                        f"All categories in '{col}' " f"had cells with <5 counts"
                    ),
                }
            )
            continue

        contingency_table_filtered = pd.crosstab(
            df_filtered[col], df_filtered[target_col]
        )
        chi2, p_value, _, _ = chi2_contingency(contingency_table_filtered)

        conclusion = "Significant" if p_value < alpha else "Not Significant"
        warning = (
            ""
            if len(valid_categories) == df[col].nunique()
            else f"Excluded categories: {set(df[col].unique()) - set(valid_categories)}"
        )

        results.append(
            {
                "feature": col,
                "test": "Chi-Square",
                "statistic": chi2,
                "p-value": p_value,
                "conclusion": conclusion,
                "warning: too little entries": warning,
            }
        )

    return pd.DataFrame(results)


def test_numerical_variables(
    df: pd.DataFrame, numerical_cols: list[str], target_col: str, alpha: float
) -> pd.DataFrame:
    """
    Performs a T-Test (if normal) or Mann-Whitney U Test (if non-normal)
    to compare means or distributions of two groups.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.
    numerical_cols : List[str]
        A list of column names representing numerical variables.
    target_col : str
        The binary target column (0 or 1).
    alpha : float
        The significance level to determine statistical significance.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the test results with the following columns:
        - feature: The numerical feature tested.
        - test: "T-Test" or "Mann-Whitney U".
        - statistic: The test statistic.
        - p-value: The p-value.
        - conclusion: Whether the result is statistically significant at the given
        alpha level.
    """
    results = []

    for col in numerical_cols:
        group1 = df[df[target_col] == 0][col].dropna()
        group2 = df[df[target_col] == 1][col].dropna()

        _, p_normality_1 = normaltest(group1) if len(group1) >= 20 else (1, 1)
        _, p_normality_2 = normaltest(group2) if len(group2) >= 20 else (1, 1)

        if p_normality_1 > alpha and p_normality_2 > alpha:
            test_stat, p_value = ttest_ind(group1, group2, equal_var=False)
            test_name = "T-Test"
        else:
            test_stat, p_value = mannwhitneyu(group1, group2, alternative="two-sided")
            test_name = "Mann-Whitney U"

        results.append(
            {
                "feature": col,
                "test": test_name,
                "statistic": test_stat,
                "p-value": p_value,
                "conclusion": "Significant" if p_value < alpha else "Not Significant",
                "warning: too little entries": "",
            }
        )

    return pd.DataFrame(results)


def encode_nominal_categorical_features(
    df: pd.DataFrame, mapping_dict: dict[str, dict[Any, int]]
) -> pd.DataFrame:
    """
    Encodes nominal categorical features in a DataFrame using a predefined
    mapping dictionary.

    This function replaces categorical values with their corresponding numerical
    mappings. If a category is not found in the mapping dictionary,
    it is replaced with -1.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing categorical features.
    mapping_dict : Dict[str, Dict[Any, int]]
        A dictionary where keys are column names, and values are dictionaries mapping
        category labels to numerical values.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with nominal categorical features replaced by their
        numeric encodings.
    """
    df = df.copy()
    for col, mapping in mapping_dict.items():
        df[col] = df[col].map(mapping).fillna(-1).astype(int)
    return df
