from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import pandas as pd
from loguru import logger
import phik
from datetime import datetime as dt
import phik
from catboost import EFeaturesSelectionAlgorithm, EShapCalcType, Pool, CatBoostClassifier, CatBoostRegressor
from loguru import logger
from sklearn.metrics import get_scorer_names
from sklearn.model_selection import cross_val_score

from outboxml.core.pydantic_models import FeatureSelectionConfig, ModelConfig
from outboxml.data_subsets import ModelDataSubset


def train_data(data_subset: ModelDataSubset) -> tuple:
    X_train = data_subset.X_train
    X_test = data_subset.X_test
    cat_features = data_subset.features_categorical
    y_train = data_subset.y_train / data_subset.exposure_train if data_subset.exposure_train is not None else data_subset.y_train
    y_test = data_subset.y_test / data_subset.exposure_test if data_subset.exposure_test is not None else data_subset.y_test
    return X_train, X_test, y_train, y_test, cat_features

def catboost_model(objective, params, cat_features=None):
    if objective == "Logloss":
        logger.info('Classification')
        model = CatBoostClassifier(objective=objective, cat_features=cat_features, verbose=False, **params)

    else:
        logger.info('Regression')
        model = CatBoostRegressor(objective=objective, cat_features=cat_features, verbose=False, **params)

    return model


class Analysis(ABC):
    @abstractmethod
    def result(self, *params):
        pass


class CorrelationMatrix(Analysis):
    def __init__(self,
                 data: pd.DataFrame,
                 feature_importance_list: list,
                 features_numerical: list,
                 threshold: float = 0.9,
                 ):
        self.X = data
        self.last = feature_importance_list
        self.threshold = threshold
        self.features_numerical = features_numerical

    def result(self):
        self.X = self.X[reversed(self.last)]  # упорядочен по значимости
        logger.debug('Feature selection||Calculating correlations')
        phik_matrix = self.X.phik_matrix(interval_cols=self.features_numerical)
        upper = phik_matrix.where(np.triu(np.ones(phik_matrix.shape), k=1).astype(
            bool))  # берем из набора скоррелированных только самую значимую

        # Найти признаки с корреляцией выше порогового значения
        to_drop = [column for column in upper.columns if any(upper[column] > self.threshold)]
        if len(to_drop) > 0:
            logger.info('Dropping ' + str(to_drop))
        return to_drop



class CVStability(Analysis):
    def __init__(self,
                 list_to_exclude: list,
                 data_subset: ModelDataSubset,
                 config: FeatureSelectionConfig,
                 features: list,
                 objective: str,
                 catboost_params: dict = None
                 ):

        self.catboost_params = catboost_params
        self.data_subset = data_subset
        self.to_drop = list_to_exclude
        self.config = config
        self.objective = objective
        if catboost_params is None:
            self.params  = self.config.params
        else:
            self.params = catboost_params
        self.features = features


    def result(self,):
        """Calculation of stability using phik matrix"""
        # TODO разобраться со списками
        features = self.features
        if features == []:
            return features
        else:
            features_for_calc = features.copy()
            cat_features = self.data_subset.features_categorical.copy()
            cat_features_for_calc = cat_features.copy()
            for feature in features:
                if feature in self.to_drop:
                    features_for_calc.remove(feature)
            for cat_feature in cat_features:
                if cat_feature in self.to_drop:
                    cat_features_for_calc.remove(cat_feature)
            X_train, X_test, y_train, y_test, _ = train_data(self.data_subset)
            for feature in features_for_calc:
                catboost_features = cat_features_for_calc.copy()
                if len(features_for_calc) > 1:
                    features_for_cv = features_for_calc.copy()
                    features_for_cv.remove(feature)
                    X = X_train[X_train.columns[~X_train.columns.isin(features_for_cv)]]
                else:
                    X = X_train
                logger.debug('CV for feature ' + str(feature))
                for cat_feature in cat_features_for_calc:
                    if cat_feature not in X.columns:
                        catboost_features.remove(cat_feature)

                model = catboost_model(objective=self.objective,
                                       params=self.catboost_params,
                                        cat_features=catboost_features)

                try:
                    scoring = self.__choose_scoring_fun(model_name=self.data_subset.model_name)
                    scores = cross_val_score(model, X, y_train, cv=3, scoring=scoring)
                    logger.info('CV dif for feature||' + str(np.max(scores) / np.min(scores)))
                    if (np.max(scores) / np.min(scores) - 1) > self.config.cv_diff_value:
                        logger.info('Dropping non-stable feature')
                        self.to_drop.append(feature)

                except Exception as exc:
                    logger.error(exc)
                    logger.info('No CV for feature')
            return self.to_drop

    def __choose_scoring_fun(self, model_name: str):
        if self.config.metric_eval[model_name] in get_scorer_names():
            return self.config.metric_eval[model_name]
        else:
            logger.error('Unknown metric for cv||Returning neg_mean_absolute_error')
            return 'neg_mean_absolute_error'


class CatboostShapAnalysis(Analysis):
    def __init__(self,
                 data_subset: ModelDataSubset,
                 config: FeatureSelectionConfig,
                 objective: str='RMSE',
                 params: dict=None):

        self.objective = objective
        self.config = config
        self.data_subset = data_subset
        if params is None:
            self.params  = self.config.params
        else:
            self.params = params

    def result(self,):
        logger.debug('Feature selection||Fitting catboost')
        X_train, X_test, y_train, y_test, cat_features = train_data(self.data_subset)
        train_pool = Pool(X_train, y_train, feature_names=list(X_train.columns),
                          cat_features=cat_features)
        test_pool = Pool(X_test, y_test, feature_names=list(X_train.columns),
                         cat_features=cat_features)
        steps = X_train.shape[1]
        model = catboost_model(objective=self.objective, params=self.params)
        summary = model.select_features(
            train_pool,
            eval_set=test_pool,
            features_for_select=f'0-{steps - 1}',
            num_features_to_select=1,
            #     steps=train_X.shape[1] - 1,
            steps=steps - 1,
            algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
            shap_calc_type=EShapCalcType.Regular,
            train_final_model=True,
            logging_level='Silent',
            plot=False
        )
        return summary




class CutNumberAnalysis(Analysis):
    def __init__(self, data: pd.Series,):
        self.data = data
    def result(self):

