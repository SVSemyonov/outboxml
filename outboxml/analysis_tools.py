from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime as dt

import numpy as np
import pandas as pd
import phik
from loguru import logger

from catboost import (
    EFeaturesSelectionAlgorithm,
    EShapCalcType,
    Pool,
    CatBoostClassifier,
    CatBoostRegressor
)
from sklearn.metrics import get_scorer_names
from sklearn.model_selection import cross_val_score

from outboxml.core.pydantic_models import FeatureSelectionConfig, ModelConfig
from outboxml.data_subsets import ModelDataSubset


def train_data(data_subset: ModelDataSubset) -> tuple:
    """Prepare train and test data for model fitting.

    Applies exposure normalization if exposure columns are present.

    :param data_subset: Prepared dataset
    :type data_subset: ModelDataSubset
    :return: Tuple containing train/test data and categorical features
    :rtype: tuple
    """

    X_train = data_subset.X_train
    X_test = data_subset.X_test
    cat_features = data_subset.features_categorical

    y_train = (
        data_subset.y_train / data_subset.exposure_train
        if data_subset.exposure_train is not None
        else data_subset.y_train
    )
    y_test = (
        data_subset.y_test / data_subset.exposure_test
        if data_subset.exposure_test is not None
        else data_subset.y_test
    )

    return X_train, X_test, y_train, y_test, cat_features


def catboost_model(objective: str, params: dict, cat_features: list = None):
    """Create CatBoost model instance.

    Automatically selects classifier or regressor
    based on the objective function.

    :param objective: CatBoost objective name
    :type objective: str
    :param params: CatBoost model parameters
    :type params: dict
    :param cat_features: Indices or names of categorical features
    :type cat_features: list, optional
    :return: Initialized CatBoost model
    """

    if objective == "Logloss":
        logger.info('Classification')
        model = CatBoostClassifier(
            objective=objective,
            cat_features=cat_features,
            verbose=False,
            **params
        )
    else:
        logger.info('Regression')
        model = CatBoostRegressor(
            objective=objective,
            cat_features=cat_features,
            verbose=False,
            **params
        )

    return model


class Analysis(ABC):
    """Abstract base class for analytical components."""

    @abstractmethod
    def result(self, *params):
        """Execute analysis and return result."""
        pass


class CorrelationMatrix(Analysis):
    """Correlation-based feature filtering using PhiK.

    Identifies highly correlated features and removes
    less important ones based on feature importance ranking.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_importance_list: list,
        features_numerical: list,
        threshold: float = 0.9,
    ):
        """Initialize correlation matrix analysis.

        :param data: Feature dataframe
        :type data: pd.DataFrame
        :param feature_importance_list: Ranked feature list
        :type feature_importance_list: list
        :param features_numerical: List of numerical features
        :type features_numerical: list
        :param threshold: Correlation threshold
        :type threshold: float
        """

        self.X = data
        self.last = feature_importance_list
        self.threshold = threshold
        self.features_numerical = features_numerical

    def result(self) -> list:
        """Calculate correlated features to drop.

        :return: List of feature names to drop
        :rtype: list
        """

        self.X = self.X[reversed(self.last)]
        logger.debug('Feature selection || Calculating correlations')

        phik_matrix = self.X.phik_matrix(
            interval_cols=self.features_numerical
        )

        upper = phik_matrix.where(
            np.triu(
                np.ones(phik_matrix.shape),
                k=1
            ).astype(bool)
        )

        to_drop = [
            column
            for column in upper.columns
            if any(upper[column] > self.threshold)
        ]

        if to_drop:
            logger.info(f'Dropping correlated features || {to_drop}')

        return to_drop


class CVStability(Analysis):
    """Cross-validation stability analysis.

    Evaluates feature stability by measuring the variation
    of cross-validation scores when excluding individual features.
    """

    def __init__(
        self,
        list_to_exclude: list,
        data_subset: ModelDataSubset,
        config: FeatureSelectionConfig,
        features: list,
        objective: str,
        catboost_params: dict = None
    ):
        """Initialize CV stability analyzer.

        :param list_to_exclude: Initial list of features to exclude
        :type list_to_exclude: list
        :param data_subset: Prepared dataset
        :type data_subset: ModelDataSubset
        :param config: Feature selection configuration
        :type config: FeatureSelectionConfig
        :param features: Candidate feature list
        :type features: list
        :param objective: CatBoost objective
        :type objective: str
        :param catboost_params: CatBoost parameters
        :type catboost_params: dict, optional
        """

        self.catboost_params = catboost_params
        self.data_subset = data_subset
        self.to_drop = list_to_exclude
        self.config = config
        self.objective = objective

        self.params = (
            self.config.params
            if catboost_params is None
            else catboost_params
        )

        self.features = features

    def result(self) -> list:
        """Run CV-based stability check.

        :return: Updated list of unstable features
        :rtype: list
        """

        if not self.features:
            return self.features

        features_for_calc = [
            f for f in self.features
            if f not in self.to_drop
        ]

        cat_features_for_calc = [
            f for f in self.data_subset.features_categorical
            if f not in self.to_drop
        ]

        X_train, _, y_train, _, _ = train_data(self.data_subset)

        for feature in features_for_calc:
            logger.debug(f'CV stability check for feature || {feature}')

            if len(features_for_calc) > 1:
                features_for_cv = features_for_calc.copy()
                features_for_cv.remove(feature)
                X = X_train.drop(columns=features_for_cv)
            else:
                X = X_train

            catboost_features = [
                f for f in cat_features_for_calc
                if f in X.columns
            ]

            model = catboost_model(
                objective=self.objective,
                params=self.catboost_params,
                cat_features=catboost_features
            )

            try:
                scoring = self.__choose_scoring_fun(
                    self.data_subset.model_name
                )

                scores = cross_val_score(
                    model, X, y_train, cv=3, scoring=scoring
                )

                diff = np.max(scores) / np.min(scores) - 1
                logger.info(f'CV diff || {diff}')

                if diff > self.config.cv_diff_value:
                    logger.info(
                        'Dropping non-stable feature || ' + feature
                    )
                    self.to_drop.append(feature)

            except Exception as exc:
                logger.error(exc)
                logger.info('Skipping CV for feature')

        return self.to_drop

    def __choose_scoring_fun(self, model_name: str) -> str:
        """Select scoring function for cross-validation.

        :param model_name: Model identifier
        :type model_name: str
        :return: Scoring function name
        :rtype: str
        """

        metric = self.config.metric_eval.get(model_name)

        if metric in get_scorer_names():
            return metric

        logger.error(
            'Unknown metric for CV || using neg_mean_absolute_error'
        )
        return 'neg_mean_absolute_error'


class CatboostShapAnalysis(Analysis):
    """Recursive SHAP-based feature selection using CatBoost."""

    def __init__(
        self,
        data_subset: ModelDataSubset,
        config: FeatureSelectionConfig,
        objective: str = 'RMSE',
        params: dict = None
    ):
        """Initialize SHAP-based feature selector.

        :param data_subset: Prepared dataset
        :type data_subset: ModelDataSubset
        :param config: Feature selection configuration
        :type config: FeatureSelectionConfig
        :param objective: CatBoost objective
        :type objective: str
        :param params: CatBoost parameters
        :type params: dict, optional
        """

        self.objective = objective
        self.config = config
        self.data_subset = data_subset
        self.params = params or self.config.params

    def result(self) -> dict:
        """Run recursive SHAP-based feature selection.

        :return: Feature selection summary
        :rtype: dict
        """

        logger.debug('Feature selection || Fitting CatBoost')

        X_train, X_test, y_train, y_test, cat_features = train_data(
            self.data_subset
        )

        train_pool = Pool(
            X_train, y_train,
            feature_names=list(X_train.columns),
            cat_features=cat_features
        )

        test_pool = Pool(
            X_test, y_test,
            feature_names=list(X_train.columns),
            cat_features=cat_features
        )

        steps = X_train.shape[1]

        model = catboost_model(
            objective=self.objective,
            params=self.params
        )

        summary = model.select_features(
            train_pool,
            eval_set=test_pool,
            features_for_select=f'0-{steps - 1}',
            num_features_to_select=1,
            steps=steps - 1,
            algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
            shap_calc_type=EShapCalcType.Regular,
            train_final_model=True,
            logging_level='Silent',
            plot=False
        )

        return summary