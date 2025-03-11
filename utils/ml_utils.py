"""
A collection of utility functions for machine learning tasks, including
feature evaluation, data preprocessing, model evaluation, etc.
"""

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy.stats import entropy

from sklearn.feature_selection import (
    mutual_info_classif,
    SelectFromModel,
    RFE,
    SelectKBest,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import joblib
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance

from typing import Optional, Any


def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the dataset by handling missing values and encoding categorical
    features.

    Parameters:
    ----------
    X : pd.DataFrame
        Feature matrix.

    Returns:
    -------
    pd.DataFrame
        Preprocessed feature matrix.
    """
    for colname in X.select_dtypes(["number"]):
        X[colname] = X[colname].fillna(-999)

    for colname in X.select_dtypes(["category"]):
        X[colname] = X[colname].cat.codes

    for colname in X.select_dtypes(["object"]):
        X[colname] = X[colname].fillna("missing")
        X[colname] = X[colname].astype("category").cat.codes

    return X


def compute_normalized_mutual_information(
    X: pd.DataFrame,
    y: pd.Series,
    discrete_feature_list: Optional[list[str]] = None,
    sort_results: bool = True,
) -> pd.DataFrame:
    """
    Computes normalized mutual information (NMI) scores for each feature in X
    against a binary target variable y, using entropy-based normalization.

    Parameters:
    ----------
    X : pd.DataFrame
        Feature matrix (independent variables).
    y : pd.Series
        Binary target variable (dependent variable).
    discrete_feature_list : list, optional
        List of features that should be treated as discrete.
    sort_results : bool, optional
        Whether to return the results sorted by normalized mutual information (default
        is True).

    Returns:
    -------
    pd.DataFrame
        DataFrame containing feature names and their respective normalized mutual
        information scores.
    """

    X = X.copy()
    X = preprocess_features(X)

    if discrete_feature_list is None:
        discrete_features = [
            col in X.select_dtypes(["category", "int64"]).columns for col in X.columns
        ]
    else:
        discrete_features = [col in discrete_feature_list for col in X.columns]

    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features)
    entropy_X = np.array(
        [entropy(np.unique(X[col], return_counts=True)[1]) for col in X.columns]
    )
    entropy_y = entropy(np.unique(y, return_counts=True)[1])

    with np.errstate(divide="ignore", invalid="ignore"):
        nmi_scores = np.where(
            entropy_X > 0, mi_scores / np.sqrt(entropy_X * entropy_y), 0
        )

    mi_df = pd.DataFrame(
        {"feature": X.columns, "normalized_mutual_information": nmi_scores}
    )

    if sort_results:
        mi_df = mi_df.sort_values(by="normalized_mutual_information", ascending=False)

    return mi_df


def compute_mutual_information(
    X: pd.DataFrame,
    y: pd.Series,
    discrete_feature_list: Optional[list[str]] = None,
    sort_results: bool = True,
) -> pd.DataFrame:
    """
    Computes mutual information scores for each feature in X against a binary
    target variable y.

    Parameters:
    ----------
    X : pd.DataFrame
        Feature matrix (independent variables).
    y : pd.Series
        Binary target variable (dependent variable).
    discrete_feature_list : list, optional
        List of features that should be treated as discrete, in cases where dataset
        contains integer features that should be treated as continuous.
    sort_results : bool, optional
        Whether to return the results sorted by mutual information (default is True).

    Returns:
    -------
    pd.DataFrame
        DataFrame containing feature names and their respective mutual information
        scores.
    """

    X = X.copy()
    X = preprocess_features(X)

    if discrete_feature_list is None:
        discrete_features = [
            col in X.select_dtypes(["category", "int64"]).columns for col in X.columns
        ]
    else:
        discrete_features = [col in discrete_feature_list for col in X.columns]

    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features)
    mi_df = pd.DataFrame({"feature": X.columns, "mutual_information": mi_scores})

    if sort_results:
        mi_df = mi_df.sort_values(by="mutual_information", ascending=False)

    return mi_df


def score_dataset_classification_task(
    X: pd.DataFrame,
    y: pd.Series,
    model: ClassifierMixin = DummyClassifier(strategy="stratified"),
) -> float:
    """
    Scores a dataset using cross-validation using Precision-Recall AUC as the metric.

    Parameters:
    ----------
    X : pd.DataFrame
        Feature matrix (independent variables).
    y : pd.Series
        Target variable.
    model : object
        Model object to be used for scoring (default is DummyClassifier).
    Returns:
    -------
    float
        Mean cross-validated score.
    """
    X = X.copy()
    X = preprocess_features(X)
    if type(model).__name__ == "LogisticRegression":
        scaler = StandardScaler()
        num_features = X.select_dtypes(["number"]).columns
        X[num_features] = scaler.fit_transform(X[num_features])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring="average_precision")

    return np.mean(scores)


def create_ordinal_bins(X: pd.DataFrame, binning_info: dict) -> pd.DataFrame:
    """
    Creates ordinal categorical bins for numerical columns in dataframe X.

    Parameters:
    -----------
    X : pd.DataFrame
        The dataset containing numerical features.
    binning_info : dict
        Dictionary containing binning information for each feature.

    Returns:
    --------
    pd.DataFrame
        Updated DataFrame with new ordinal categorical columns.
    """
    X = X.copy()
    for feature, info in binning_info.items():
        group_name = f"{feature}_group"
        X[group_name] = pd.cut(
            X[feature], bins=info["bins"], labels=info["labels"], include_lowest=True
        )
        X[group_name] = X[group_name].cat.add_categories("unknown").fillna("unknown")
        category_order = ["unknown"] + info["labels"]
        X[group_name] = X[group_name].astype(
            CategoricalDtype(categories=category_order, ordered=True)
        )
    return X


def select_features_lasso_kbest(
    X: pd.DataFrame, y: pd.Series, n_features: int = 11, alpha: float = 0.01
) -> pd.DataFrame:
    """
    Selects features using LASSO (L1 Regularization) to remove collinear features,
    then applies SelectKBest for final feature selection.

    Parameters:
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    n_features : int, optional
        Number of top features to select (default is 11).
    alpha : float, optional
        Regularization strength for LASSO (default is 0.01).

    Returns:
    -------
    pd.DataFrame
        The dataset with only the selected features.
    """
    X = X.copy()
    X = preprocess_features(X)

    model = LogisticRegression(
        penalty="l1", solver="liblinear", C=1 / alpha, max_iter=10000
    )
    sfm = SelectFromModel(model)
    sfm.fit_transform(X, y)

    selected_features_lasso = X.columns[sfm.get_support()]
    X_lasso = X[selected_features_lasso]

    selector = SelectKBest(
        mutual_info_classif, k=min(n_features, len(selected_features_lasso))
    )
    selector.fit_transform(X_lasso, y)

    final_selected_features = X_lasso.columns[selector.get_support()]

    print(
        f"Selected features (LASSO + SelectKBest) (univariate FS): "
        f"{final_selected_features.tolist()}"
    )

    return final_selected_features


def select_features_rfe_lasso(
    X: pd.DataFrame, y: pd.Series, n_features: int = 11, alpha: float = 0.01
) -> pd.DataFrame:
    """
    Selects top features using a combination of Recursive Feature Elimination (RFE)
    and LASSO (L1 Regularization). If there are any, LASSO removes collinear features.

    Parameters:
    ----------
    X : pd.DataFrame
        Preprocessed feature matrix.
    y : pd.Series
        Target variable.
    n_features : int, optional
        Number of features to select using RFE (default is 11).
    alpha : float, optional
        Regularization strength for LASSO (default is 0.01).

    Returns:
    -------
    pd.DataFrame
        The dataset with only the selected top features.
    """
    X = X.copy()
    X = preprocess_features(X)

    model = LogisticRegression(
        penalty="l1", solver="liblinear", C=1 / alpha, max_iter=10000
    )
    sfm = SelectFromModel(model)
    sfm.fit_transform(X, y)

    selected_features_lasso = X.columns[sfm.get_support()]
    X_lasso = X[selected_features_lasso]

    rfe_model = RandomForestClassifier(random_state=42)
    rfe = RFE(rfe_model, n_features_to_select=n_features)
    rfe.fit_transform(X_lasso, y)

    selected_features_final = X_lasso.columns[rfe.support_]
    print(
        f"Selected features (LASSO + RFE) (multivariate FS): "
        f"{selected_features_final.tolist()}"
    )

    return selected_features_final


def create_dynamic_preprocessor(
    X: pd.DataFrame, binning_info: dict[str, Any]
) -> ColumnTransformer:
    """
    Creates a preprocessing pipeline dynamically based on the selected features.

    The function detects:
    - **Ordered categorical** columns and applies **Ordinal Encoding**..
    - **Unordered categorical** columns and applies **One-Hot Encoding**.
    - **Binary numerical** columns (0/1 values) and passes them as-is.
    - **Continuous numerical** columns and applies **Imputation
        (fill missing with -999).

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix.
    binning_info : dict
        Dictionary containing binning details. Expected keys are column names
        (without `_group`) and their label encoding.

    Returns
    -------
    ColumnTransformer
        A dynamically constructed preprocessing pipeline for categorical, numerical,
        and binary features.
    """
    transformers = []

    selected_ord_cols = X.select_dtypes(["category"]).columns.to_list()
    selected_binary_columns = [
        col for col in X.select_dtypes(["number"]) if X[col].dropna().isin([0, 1]).all()
    ]
    selected_num_cols = [
        col for col in X.select_dtypes(["number"]) if col not in selected_binary_columns
    ]
    categories = [
        binning_info[col.replace("_group", "")]["labels"] for col in selected_ord_cols
    ]
    if selected_ord_cols:
        transformers.append(
            (
                "ordinal",
                Pipeline(
                    [
                        (
                            "encoder",
                            OrdinalEncoder(
                                categories=categories,
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                selected_ord_cols,
            )
        )

    if selected_num_cols:
        transformers.append(
            (
                "num_preprocessing",
                SimpleImputer(strategy="median"),
                selected_num_cols,
            )
        )

    return ColumnTransformer(transformers, remainder="passthrough")


def compute_classification_metrics(
    confusion_matrices: dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Computes classification metrics (FPR, FNR, Recall, Precision, Specificity, Accuracy)
    from a dictionary of confusion matrices.

    Parameters
    ----------
    confusion_matrices : dict
        A dictionary where keys are model names and values are confusion matrices
        (2x2 numpy arrays).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing classification metrics for each model, sorted by
        False Positive Rate (FPR).
    """
    metrics_df = pd.DataFrame(
        columns=["fpr", "fnr", "recall_tpr", "precision", "specificity_tnr", "accuracy"]
    )
    for model_name, cm in confusion_matrices.items():
        cm = np.array(cm)
        TN, FP, FN, TP = cm.ravel()
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) != 0 else 0
        fnr = FN / (FN + TP) if (FN + TP) != 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
        metrics_df.loc[model_name] = [
            fpr,
            fnr,
            recall,
            precision,
            specificity,
            accuracy,
        ]

    return metrics_df.sort_values(by="fnr")


def train_and_evaluate_models_no_smote(
    models: dict[str, object],
    hyperparameters: dict[str, dict[str, list]],
    feature_sets: dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]],
    binning_info: dict[str, dict[str, list]],
    n_splits: int = 10,
    random_search_iter: int = 20,
    scoring_metric: str = "average_precision",
    probability_threshold: float = 0.3,
) -> tuple[pd.DataFrame, dict[str, list], dict[str, dict], dict[str, np.ndarray]]:
    """
    Trains and evaluates multiple models using cross-validation and hyperparameter
    tuning.

    Parameters
    ----------
    models : dict
        Dictionary of model names and model instances.
    hyperparameters : dict
        Dictionary of model names and their hyperparameter grids for tuning.
    feature_sets : dict
        Dictionary of feature set names mapping to (X_train, y_train, X_test, y_test).
    binning_info : dict
        Dictionary containing binning information for categorical features.
    n_splits : int, optional
        Number of folds for Stratified K-Fold cross-validation, by default 10.
    random_search_iter : int, optional
        Number of iterations for RandomizedSearchCV, by default 20.
    scoring_metric : str, optional
        Scoring metric for hyperparameter tuning, by default "average_precision".
    probability_threshold : float, optional
        Probability threshold to finalize prediction 1 if above the threshold,
        and 0 if below. By default 0.3.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, list], Dict[str, dict], Dict[str, np.ndarray]]
        Returns:
        - `cv_results_df`: DataFrame with cross-validation scores.
        - `cv_fold_accuracies`: Dictionary with per-fold accuracy scores.
        - `best_hyperparameters`: Dictionary of best hyperparameters for each model.
        - `confusion_matrices`: Dictionary with confusion matrices.
    """
    cv_means = []
    cv_stds = []
    model_names = []
    cv_fold_accuracies = {}
    best_hyperparameters = {}
    confusion_matrices = {}
    f1_scores = []
    balanced_accuracy_scores = []

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for feat_name, (
        X_train_fs,
        y_train_fs,
        X_test_fs,
        y_test_fs,
    ) in feature_sets.items():
        feature_list = X_train_fs.columns.to_list()
        preprocessor = create_dynamic_preprocessor(
            X=X_train_fs, binning_info=binning_info
        )
        preprocessor.fit(X_train_fs)
        X_train_fs = pd.DataFrame(X_train_fs, columns=feature_list)

        for name, model in models.items():

            if isinstance(model, lgb.LGBMClassifier):
                model.set_params(verbose=-1)

            pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

            param_grid = hyperparameters.get(name, {})
            param_grid = {
                key.replace("model__", "model__"): value
                for key, value in param_grid.items()
            }
            param_space_size = (
                np.prod([len(vals) for vals in param_grid.values()])
                if param_grid
                else 1
            )
            optimal_n_iter = min(param_space_size, random_search_iter)

            random_search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=optimal_n_iter,
                cv=kfold,
                scoring=scoring_metric,
                random_state=42,
                n_jobs=-1,
            )

            random_search.fit(X_train_fs, y_train_fs)
            best_model = random_search.best_estimator_
            best_hyperparameters[f"{name} ({feat_name})"] = random_search.best_params_

            cv_result = cross_val_score(
                best_model,
                X_train_fs,
                y_train_fs,
                cv=kfold,
                scoring=scoring_metric,
            )
            cv_fold_accuracies[f"{name} ({feat_name})"] = cv_result.tolist()
            model_names.append(f"{name} ({feat_name})")
            cv_means.append(cv_result.mean())
            cv_stds.append(cv_result.std())

            y_probs = best_model.predict_proba(X_test_fs)[:, 1]
            y_pred = (y_probs > probability_threshold).astype(int)
            confusion_matrices[f"{name} ({feat_name})"] = confusion_matrix(
                y_test_fs, y_pred
            )
            f1_scores.append(f1_score(y_test_fs, y_pred))
            balanced_accuracy_scores.append(balanced_accuracy_score(y_test_fs, y_pred))
            print(f"recall {cv_result.mean()}, f1 {f1_score(y_test_fs, y_pred)}\n\n")

    cv_results_df = pd.DataFrame(
        {
            "model": model_names,
            "mean_cv_score": cv_means,
            "std_cv_score": cv_stds,
            "f1_score_test": f1_scores,
            "balanced_accuracy_test": balanced_accuracy_scores,
        }
    )

    return cv_results_df, cv_fold_accuracies, best_hyperparameters, confusion_matrices


def train_and_evaluate_models_smote(
    models: dict[str, object],
    hyperparameters: dict[str, dict[str, list]],
    feature_sets: dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]],
    binning_info: dict[str, dict[str, list]],
    n_splits: int = 10,
    random_search_iter: int = 20,
    scoring_metric: str = "recall",
) -> tuple[pd.DataFrame, dict[str, list], dict[str, dict], dict[str, np.ndarray]]:
    """
    Trains and evaluates multiple models using SMOTE for resampling imbalanced data,
    cross-validation and hyperparameter tuning.

    Parameters
    ----------
    models : dict
        Dictionary of model names and model instances.
    hyperparameters : dict
        Dictionary of model names and their hyperparameter grids for tuning.
    feature_sets : dict
        Dictionary of feature set names mapping to (X_train, y_train, X_test, y_test).
    binning_info : dict
        Dictionary containing binning information for categorical features.
    n_splits : int, optional
        Number of folds for Stratified K-Fold cross-validation, by default 10.
    random_search_iter : int, optional
        Number of iterations for RandomizedSearchCV, by default 20.
    scoring_metric : str, optional
        Scoring metric for hyperparameter tuning, by default "average_precision".

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, list], Dict[str, dict], Dict[str, np.ndarray]]
        Returns:
        - `cv_results_df`: DataFrame with cross-validation scores.
        - `cv_fold_accuracies`: Dictionary with per-fold accuracy scores.
        - `best_hyperparameters`: Dictionary of best hyperparameters for each model.
        - `confusion_matrices`: Dictionary with confusion matrices.
    """
    cv_means = []
    cv_stds = []
    model_names = []
    cv_fold_accuracies = {}
    feature_set_names = []
    best_hyperparameters = {}
    confusion_matrices = {}
    f1_scores = []
    balanced_accuracy_scores = []

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for feat_name, (
        X_train_fs,
        y_train_fs,
        X_test_fs,
        y_test_fs,
    ) in feature_sets.items():

        preprocessor = create_dynamic_preprocessor(
            X=X_train_fs, binning_info=binning_info
        )
        preprocessor.fit(X_train_fs)

        X_train_fs_transformed = preprocessor.transform(X_train_fs)
        X_test_fs_transformed = preprocessor.transform(X_test_fs)

        smote = SMOTE(sampling_strategy="minority", random_state=42)
        X_train_fs_resampled, y_train_fs_resampled = smote.fit_resample(
            X_train_fs_transformed, y_train_fs
        )

        for name, model in models.items():
            print(f"{name} ({feat_name})")
            if isinstance(model, lgb.LGBMClassifier):
                model.set_params(verbose=-1)

            param_grid = hyperparameters.get(name, {})
            param_grid = {
                key.replace("model__", ""): value for key, value in param_grid.items()
            }
            param_space_size = (
                np.prod([len(vals) for vals in param_grid.values()])
                if param_grid
                else 1
            )
            optimal_n_iter = min(param_space_size, random_search_iter)

            random_search = RandomizedSearchCV(
                model,
                param_distributions=param_grid,
                n_iter=optimal_n_iter,
                cv=kfold,
                scoring=scoring_metric,
                random_state=42,
                n_jobs=-1,
            )

            random_search.fit(X_train_fs_resampled, y_train_fs_resampled)
            best_model = random_search.best_estimator_
            best_hyperparameters[f"{name} ({feat_name})"] = random_search.best_params_

            cv_result = cross_val_score(
                best_model,
                X_train_fs_resampled,
                y_train_fs_resampled,
                cv=kfold,
                scoring=scoring_metric,
            )
            cv_fold_accuracies[f"{name} ({feat_name})"] = cv_result.tolist()
            model_names.append(f"{name} ({feat_name})")
            cv_means.append(cv_result.mean())
            cv_stds.append(cv_result.std())
            feature_set_names.append(feat_name)

            y_probs = best_model.predict_proba(X_test_fs_transformed)[:, 1]
            y_pred = (y_probs > 0.3).astype(int)
            confusion_matrices[f"{name} ({feat_name})"] = confusion_matrix(
                y_test_fs, y_pred
            )
            f1_scores.append(f1_score(y_test_fs, y_pred))
            balanced_accuracy_scores.append(balanced_accuracy_score(y_test_fs, y_pred))
            print(f"recall {cv_result.mean()}, f1 {f1_score(y_test_fs, y_pred)}\n\n")

    cv_results_df = pd.DataFrame(
        {
            "model": model_names,
            "mean_cv_score": cv_means,
            "std_cv_score": cv_stds,
            "f1_score": f1_scores,
            "balanced_accuracy": balanced_accuracy_scores,
        }
    )

    return cv_results_df, cv_fold_accuracies, best_hyperparameters, confusion_matrices


def compute_permutation_importance_pretrained(
    model_path: str,
    preprocessor: ColumnTransformer,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    name: str,
) -> pd.DataFrame:
    """
    Computes permutation importance for a **pre-trained model** using
    a provided test set.

    Parameters
    ----------
    model_path : str
        Path to the saved model file (e.g., ".pkl" or ".joblib").
    preprocessor : ColumnTransformer
        The preprocessor used to transform the features.
    X_test : pd.DataFrame
        Test feature matrix.
    y_test : pd.Series
        Test target vector.
    name : str
        Name for labeling the feature importance results.

    Returns
    -------
    importance_df : pd.DataFrame
        DataFrame containing permutation importances for each feature.
    """
    model: BaseEstimator = joblib.load(model_path)
    X_test_transformed = preprocessor.transform(X_test)

    encoded_feature_names = preprocessor.get_feature_names_out(
        input_features=X_test.columns
    )
    cleaned_feature_names = [
        name.replace("remainder__", "")
        .replace("categorical__", "")
        .replace("num__", "")
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

    importance_df.sort_values(by=name, ascending=False, inplace=True)
    importance_df.set_index("Feature", inplace=True)

    return importance_df
