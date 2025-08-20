from abc import ABC, abstractmethod

import numpy as np
import patsy
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
import statsmodels.formula.api as sf
from typing import List, Dict, Optional, Union, Any
from typing_extensions import Literal
import pandas as pd
from itertools import chain
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from loguru import logger

from outboxml.core.pydantic_models import ModelConfig
from outboxml.core.enums import ModelsParams
from outboxml.data_subsets import ModelDataSubset


class BaseWrapperModel(ABC):
    """Абстрактный класс для библиотечных моделей внутри фреймворка"""

    @abstractmethod
    def fit(self, **params):
        pass


class DefaultModels:
    """
    Класс для вызова библиотечных моделей.
    Вызывается из datasetsmanager при отсутствии пользовательских моделей на входе.
    """

    def __init__(self,
                 group_name: str,
                 dataset: pd.DataFrame,
                 data_subsets: Dict[str, ModelDataSubset],
                 models_configs: List[ModelConfig],
                 baseline_model: int = 0
                 ):
        self.group_name = group_name
        self.dataset = dataset
        self.data_subsets = data_subsets
        self.models_configs = models_configs
        self.baseline_model = baseline_model
        self.models_names = []

    def load_default(self) -> dict:
        for model in self.models_configs:  # self.data_config.data.targetcolumns:
            self.models_names.append(model.name)
        if self.baseline_model > 0:
            models_dict = self.__load_baseline_models(model_number=self.baseline_model)
        else:
            models_dict = self.__load_default_models()
        return models_dict

    def __load_baseline_models(self, model_number: int = 0):
        models_dict = {}
        for data_subset in self.data_subsets.values():
            model_name = data_subset.model_name
            models_dict[model_name] = BaselineModels(dataset=data_subset,
                                                     model_name=model_name,
                                                     model_number=model_number).choose_model()

        return models_dict

    def __load_default_models(self):

        wrapper_list = []
        for model in self.models_configs:
            wrapper_list.append(model.wrapper)
        if not all((wrapper == '' or wrapper == None) for wrapper in wrapper_list):

            models_dict = ModelsWrapper(data_subsets=self.data_subsets,
                                        models_configs=self.models_configs).models_dict()
            try:
                logger.info('Models in accordance with config||Wrapper models are setting')
            except:
                logger.error('Wrong parameters for wrapper model in config||Check target type and models')
                logger.info('No default model for group and project||Baseline models are setting')
                models_dict = self.__load_baseline_models()
        else:
            logger.info('No default model for group and project||Baseline models are setting')
            models_dict = self.__load_baseline_models()
        return models_dict


class BaselineModels:
    def __init__(self,
                 dataset,
                 model_name: str,
                 model_number: int):
        self.__dataset = dataset
        self.model_name = model_name
        self.__model_number = model_number

    def choose_model(self) -> BaseWrapperModel:
        if self.__model_number == 1:
            model = RandomForestClassifierModel(dataset=self.__dataset,
                                                model_name=self.model_name,
                                                ).fit()
            logger.info('RandomForest Baseline Model')
        elif self.__model_number == 2:
            model = BaseLineModel(dataset=self.__dataset,
                                  model_name=self.model_name,
                                  strategy='median'
                                  ).fit()
            logger.info('DummyRegressor Median Baseline model')
        elif self.__model_number == 3:
            model = BaseLineModel(dataset=self.__dataset,
                                  model_name=self.model_name,
                                  strategy='mean'
                                  ).fit()
            logger.info('DummyRegressor Mean Baseline model')
        else:
            model = BaseLineModel(dataset=self.__dataset,
                                  model_name=self.model_name,
                                  strategy='mean'
                                  ).fit()
            logger.error('No baseline model for value ' + str(self.__model_number) + '||DummyRegressor Mean Selected')
        return model


class RandomForestClassifierModel(BaseWrapperModel):
    """Random forest regressor """

    def __init__(self,
                 dataset, model_name: str = 'general'):
        self.__dataset = dataset
        self._model_name = model_name
        self._models_dict = {self._model_name: None
                             }

    def fit(self):
        train_data = self.__dataset.X_train.copy()
        y_train = self.__dataset.y_train.copy()
        le = LabelEncoder()
        for column_name in train_data.columns:
            train_data[column_name] = le.fit_transform(train_data[column_name])

        model = RandomForestRegressor(max_depth=4, n_estimators=85).fit(train_data, y_train)
        return model


class BaseLineModel(BaseWrapperModel):
    """Dummy regressor baseline"""

    def __init__(self,
                 dataset, model_name: str = 'general',
                 strategy: str = 'median'):
        self.__dataset = dataset
        self._model_name = model_name
        self.__strategy = strategy

    def fit(self):
        model = DummyRegressor(strategy=self.__strategy).fit(self.__dataset.X_train, self.__dataset.y_train)
        return model


class GLMCatboostCombineModel(BaseWrapperModel):
    def __init__(
            self,
            model_name: str,
            wrapper: Literal[
                ModelsParams.glm, ModelsParams.glm_without_scaler, ModelsParams.catboost],
            min_max_scaler,
            model,
            features_numerical: Optional[List[str]] = None,
            features_categorical: Optional[List[str]] = None,
    ):
        self.model_name = model_name
        self._wrapper = wrapper
        self.min_max_scaler = min_max_scaler
        self.model = model
        self.features_numerical = features_numerical
        self.features_categorical = features_categorical

    def predict(self, X: pd.DataFrame) -> pd.Series:
        features_numerical = self.features_numerical if self.features_numerical else []
        features_categorical = self.features_categorical if self.features_categorical else []

        if self._wrapper in (ModelsParams.glm, ModelsParams.glm_without_scaler):
            dataset_sm = X.copy()
            if features_numerical and self.min_max_scaler is not None:
                dataset_sm[features_numerical] = self.min_max_scaler.transform(dataset_sm[features_numerical])
            prediction = self.model.predict(dataset_sm[chain(features_numerical, features_categorical)])

        elif self._wrapper == ModelsParams.catboost:
            if isinstance(self.model, CatBoostRegressor) and self.model.get_param("objective") == "RMSEWithUncertainty":
                prediction = self.model.virtual_ensembles_predict(
                    X[chain(features_numerical, features_categorical)],
                    prediction_type="TotalUncertainty",
                    virtual_ensembles_count=10
                )[:, 0]
            elif isinstance(self.model, CatBoostRegressor):
                prediction = self.model.predict(X[chain(features_numerical, features_categorical)])
            elif isinstance(self.model, CatBoostClassifier):
                prediction = self.model.predict_proba(X[chain(features_numerical, features_categorical)])[:, 1]

        if isinstance(prediction, np.ndarray):
            prediction = pd.Series(prediction, index=X.index)

        return prediction

    def fit(self, **params):
        pass


class CatboostOverGLMModel(BaseWrapperModel, RegressorMixin, BaseEstimator):
    def __init__(self,

                 model_config: ModelConfig,
                 sm_model: GLMCatboostCombineModel,
                 data_subset: Any,
                 ):
        self.model_config = model_config
        self.sm_model = sm_model
        self.data_subset = data_subset
        self._model_name: str = data_subset.model_name
        self._objective: Literal[
            ModelsParams.poisson, ModelsParams.gamma, ModelsParams.binary] = self.model_config.objective
        self._wrapper: Literal[ModelsParams.catboost] = self.model_config.wrapper
        self._X_train: pd.DataFrame = data_subset.X_train
        self._y_train: pd.Series = data_subset.y_train
        self._features_numerical: Optional[List[str]] = data_subset.features_numerical
        self._features_categorical: Optional[List[str]] = data_subset.features_categorical
        self._exposure_train: Optional[pd.Series] = data_subset.exposure_train
        self._params_catboost: Optional[Dict[str, Optional[Union[int, float, str, bool]]]] = model_config.params_catboost
        self._model_sm = self.sm_model.model
        self._model_ctb = None
        self._min_max_scaler = self.sm_model.min_max_scaler

    def fit(self, X=None, y=None, **params):
        if X is not None and y is not None:
            self._X_train = X
            self._y_train = y
            self._params_catboost = params
        y_train_pred = self.__predict_glm(X=self._X_train)
        logger.info('Fitting catboost||Catboost over glm model')
        self._model_ctb = self.__fit_catboost(
            X_train=self._X_train,
            y_train=self._y_train,
            features_numerical=self._features_numerical,
            features_categorical=self._features_categorical,
            params_catboost=self._params_catboost,
            exposure_train=self._exposure_train,
            y_train_pred_glm=y_train_pred

        )
        self._X_train = None
        self._exposure_train = None
        self._y_train = None
        self.data_subset = None
        logger.debug('Wrapper model||Catboost over glm is fitted')

    def predict(self, X):
        glm_prediction = self.__predict_glm(X)

        prediction = glm_prediction * self._model_ctb.predict(
            X[chain(self._features_numerical, self._features_categorical)])

        if isinstance(prediction, np.ndarray):
            logger.error('numpy type converting to pandas')
            prediction = pd.Series(prediction, index=X.index)
        if self._objective == ModelsParams.poisson or self._objective == ModelsParams.gamma:
            prediction.loc[prediction < 0] = 0.000001
        return prediction

    def __predict_glm(self, X):
        dataset_sm = X.copy()
        if self._features_numerical and self._min_max_scaler is not None:
            dataset_sm[self._features_numerical] = self._min_max_scaler.transform(dataset_sm[self._features_numerical])
        prediction = self._model_sm.predict(dataset_sm[chain(self._features_numerical, self._features_categorical)])

        return prediction

    def __fit_catboost(self,
                       X_train: pd.DataFrame,
                       y_train: pd.Series,
                       features_numerical: Optional[List[str]] = None,
                       features_categorical: Optional[List[str]] = None,
                       params_catboost: Optional[Dict[str, Optional[Union[int, float]]]] = None,
                       exposure_train: Optional[pd.Series] = None,
                       y_train_pred_glm: Optional[pd.Series] = None,
                       ):
        features_numerical = features_numerical if features_numerical else []
        features_categorical = features_categorical if features_categorical else []

        catboost_wrapper = CatBoostRegressor
        catboost_objective = "RMSE"
        assert y_train_pred_glm is not None
        assert len(y_train) == len(y_train_pred_glm)
        if exposure_train is not None:
            y_train_ctb = (y_train / exposure_train) / y_train_pred_glm
        else:
            y_train_ctb = y_train / y_train_pred_glm
        model_ctb = catboost_wrapper(
            objective=catboost_objective,
            **params_catboost if params_catboost else {},
        )

        ctb_train_pool = Pool(
            data=X_train[chain(features_numerical, features_categorical)],
            label=y_train_ctb,
            cat_features=features_categorical,
            has_header=True,
            weight=exposure_train if exposure_train is not None else None
        )
        model_ctb = model_ctb.fit(ctb_train_pool, silent=True)
        return model_ctb


class ModelsWrapper(BaseWrapperModel):

    def __init__(self, data_subsets:Dict[str,ModelDataSubset], models_configs: List[ModelConfig]):
        self._data_subsets = data_subsets
        self._models_configs = models_configs
        self._models_dict = {}

    def fit(self):
        for model in self._models_configs:
            logger.info('Fitting model ' + model.name + '||' + str(model.wrapper))
            self._models_dict[model.name] = self.__fit_one_model(data_subset=self._data_subsets[model.name],
                                                                 wrapper=model.wrapper,
                                                                 model_config=model)

    def models_dict(self, *params) -> Dict:
        self.fit()
        return self._models_dict

    @staticmethod
    def __fit_one_model(data_subset, wrapper: str, model_config: ModelConfig):

        # GLM Модель
        if wrapper in (ModelsParams.glm, ModelsParams.glm_without_scaler, ModelsParams.catboost_over_glm):
            model = StatsmodelsModel(data_subset=data_subset, model_config=model_config).fit()
            result_model = model
        # CatBoost
        elif wrapper == ModelsParams.catboost:
            model = CatboostModel(data_subset=data_subset, model_config=model_config).fit()
            result_model = model
        # CatboostOverGlm
        if wrapper == ModelsParams.catboost_over_glm:
            result_model = CatboostOverGLMModel(data_subset=data_subset, model_config=model_config, sm_model=model)
            result_model.fit()

        return result_model


class StatsmodelsModel(BaseWrapperModel):

    def __init__(self, data_subset, model_config: ModelConfig):
        self.model_name: str = data_subset.model_name

        self.objective: Literal[ModelsParams.poisson, ModelsParams.gamma] = model_config.objective
        self.wrapper: Literal[
            ModelsParams.glm, ModelsParams.glm_without_scaler, ModelsParams.catboost_over_glm] = model_config.wrapper
        logger.info('GLM Objective||' + str(self.objective))
        self.X_train: pd.DataFrame = data_subset.X_train
        self.y_train: pd.Series = data_subset.y_train
        self.features_numerical: Optional[List[str]] = data_subset.features_numerical
        self.features_categorical: Optional[List[str]] = data_subset.features_categorical
        self.exposure_train: Optional[pd.Series] = data_subset.exposure_train
        self.treatment_dict: Optional[Dict[str, str]] = model_config.treatment_dict
        self.glm_params = model_config.params_glm

        self.__stats_models_params = {'method': 'lbfgs', 'maxiter': 750}
        if self.glm_params is not None:
            logger.info('User-defined glm params||GLM Wrapper')
            self.__stats_models_params = self.glm_params

    def fit(self):
        features_numerical = self.features_numerical if self.features_numerical else []
        features_categorical = self.features_categorical if self.features_categorical else []

        X_train_sm = self.X_train.copy()
        X_train_sm = sm.add_constant(X_train_sm)
        min_max_scaler = None
        y_train = self.y_train.copy()
        exposure = self.exposure_train
        freq_weights = None

        if features_numerical and (self.wrapper == ModelsParams.glm or self.wrapper == ModelsParams.catboost_over_glm):
            min_max_scaler = MinMaxScaler()
            min_max_scaler.fit(X_train_sm[features_numerical])
            X_train_sm[features_numerical] = min_max_scaler.transform(X_train_sm[features_numerical])

        elif self.wrapper == ModelsParams.glm_without_scaler:
            exposure = None
            freq_weights = self.exposure_train
            y_train = self.y_train / self.exposure_train if self.exposure_train is not None else self.y_train
            y_train.name = self.y_train.name

        if self.objective == ModelsParams.poisson:
            sm_family = sm.families.Poisson(sm.families.links.Log())
        elif self.objective == ModelsParams.gamma:
            sm_family = sm.families.Gamma(sm.families.links.Log())
        else:
            sm_family = sm.families.Gaussian()
        if self.treatment_dict:
            logger.info('Wrapper model||Reading user base levels for GLM')
            features_manual_base = self.treatment_dict.keys()
            features_auto = [item for item in features_categorical if item not in features_manual_base]
            formula = f"{self.y_train.name} ~ {' + '.join(features_numerical)} {' + ' if features_numerical and features_auto else ' '} {' + '.join(['C(' + feature + ')' for feature in features_auto])} {' + ' if features_manual_base else ' '} {' + '.join(['C(' + feature + ', Treatment(reference=' + self.treatment_dict[feature] + '))' for feature in features_manual_base])} "

        else:
            formula = f"{self.y_train.name} ~ {' + '.join(features_numerical)} {' + ' if features_numerical and features_categorical else ' '} {' + '.join(['C(' + feature + ')' for feature in features_categorical])}"

        try:
            model_sm = (
                sf
                .glm(
                    formula,
                    pd.concat([X_train_sm, y_train], axis=1, join="outer", ignore_index=False),
                    family=sm_family,
                    exposure=exposure,
                    freq_weights=freq_weights
                )
                .fit(**self.__stats_models_params)
            )
        except patsy.PatsyError as e:
            logger.error('Wring Base Level name for GLM||Reset to default' + str(e))
            formula = f"{self.y_train.name} ~ {' + '.join(features_numerical)} {' + ' if features_numerical and features_categorical else ' '} {' + '.join(['C(' + feature + ')' for feature in features_categorical])}"
            model_sm = (
                sf
                .glm(
                    formula,
                    pd.concat([X_train_sm, y_train], axis=1, join="outer", ignore_index=False),
                    family=sm_family,
                    exposure=exposure,
                    freq_weights=freq_weights
                )
                .fit(**self.__stats_models_params)
            )

        for feature in features_categorical:
            unique_values = X_train_sm[feature].unique()
            model_sm._results.model.data.frame[feature].iloc[:len(unique_values)] = unique_values

        # Уменьшаем вес модели, т.к. model_sm.remove_data() не трогает data.frame, data.orig_exog и data.orig_endog
        model_sm._results.model.data.frame = model_sm._results.model.data.frame.iloc[:500]
        model_sm._results.model.data.orig_exog = model_sm._results.model.data.orig_exog.iloc[:500]
        model_sm._results.model.data.orig_endog = model_sm._results.model.data.orig_endog.iloc[:500]

        model_sm.remove_data()
        return GLMCatboostCombineModel(model_name=self.model_name,
                                       wrapper=self.wrapper,
                                       min_max_scaler=min_max_scaler,
                                       model=model_sm,
                                       features_numerical=self.features_numerical,
                                       features_categorical=features_categorical,
                                       )


class CatboostModel(BaseWrapperModel):

    def __init__(self, data_subset, model_config: ModelConfig):
        self.model_name: str = data_subset.model_name
        self.objective: Literal[ModelsParams.poisson, ModelsParams.gamma, ModelsParams.binary] = model_config.objective
        logger.info('Model objective||' + str(self.objective))
        self.wrapper: Literal[ModelsParams.catboost] = model_config.wrapper
        self.X_train: pd.DataFrame = data_subset.X_train
        self.y_train: pd.Series = data_subset.y_train
        self.features_numerical: Optional[List[str]] = data_subset.features_numerical
        self.features_categorical: Optional[List[str]] = data_subset.features_categorical
        self.exposure_train: Optional[pd.Series] = data_subset.exposure_train
        self.params_catboost: Optional[Dict[str, Optional[Union[int, float, str, bool]]]] = model_config.params_catboost

    def fit(self):
        features_numerical = self.features_numerical if self.features_numerical else []
        features_categorical = self.features_categorical if self.features_categorical else []

        if self.wrapper == ModelsParams.catboost and self.objective == ModelsParams.poisson:
            catboost_wrapper = CatBoostRegressor
            catboost_objective = "Poisson"
            try:
                y_train_ctb = self.y_train / self.exposure_train
            except:
                logger.warning('Error with y_train/exposure||Exposure = 1 is set')
                y_train_ctb = self.y_train

        elif self.wrapper == ModelsParams.catboost and self.objective == ModelsParams.gamma:
            catboost_wrapper = CatBoostRegressor
            catboost_objective = "Tweedie:variance_power=1.9999999"
            try:
                y_train_ctb = self.y_train / self.exposure_train
            except:
                logger.warning('Error with y_train/exposure||Exposure = 1 is set')
                y_train_ctb = self.y_train

        elif self.wrapper == ModelsParams.catboost and self.objective == ModelsParams.binary:
            catboost_wrapper = CatBoostClassifier
            catboost_objective = "Logloss"
            y_train_ctb = self.y_train

        elif self.wrapper == ModelsParams.catboost and self.objective == ModelsParams.rmsewithuncertainty:
            catboost_wrapper = CatBoostRegressor
            catboost_objective = "RMSEWithUncertainty"
            try:
                y_train_ctb = self.y_train / self.exposure_train
            except:
                logger.warning('Error with y_train/exposure||Exposure = 1 is set')
                y_train_ctb = self.y_train
        else:
            catboost_wrapper = CatBoostRegressor
            catboost_objective = "RMSE"
            logger.info('Catboost Regressor||RMSE by default')
            try:
                y_train_ctb = self.y_train / self.exposure_train
            except:
                logger.warning('Error with y_train/exposure||Exposure = 1 is set')
                y_train_ctb = self.y_train
        model_ctb = catboost_wrapper(
            objective=catboost_objective,
            **self.params_catboost if self.params_catboost else {},
        )
        X_train = self.X_train.copy()
        ctb_train_pool = Pool(
            data=X_train[chain(features_numerical, features_categorical)],
            label=y_train_ctb,
            cat_features=features_categorical,
            has_header=True,
            weight=self.exposure_train if self.exposure_train is not None else None
        )
        model_ctb = model_ctb.fit(ctb_train_pool, silent=True)
        return GLMCatboostCombineModel(model_name=self.model_name,
                                       wrapper=self.wrapper,
                                       min_max_scaler=None,
                                       model=model_ctb,
                                       features_numerical=features_numerical,
                                       features_categorical=features_categorical,
                                       )


class StatsModelsEstimator(RegressorMixin, BaseEstimator):
    def __init__(self,
                 sm_model,
                 model_config: ModelConfig,
                 datasubset: Any, ):
        self.sm_model = sm_model
        self.model_config = model_config
        self.datasubset = datasubset
        self._min_max_scaler = None
        self._model = None

    def predict(self, X):
        features_numerical = self.datasubset.features_numerical if self.datasubset.features_numerical else []
        features_categorical = self.datasubset.features_categorical if self.datasubset.features_categorical else []

        dataset_sm = X.copy()
        dataset_sm[features_numerical] = self._min_max_scaler.transform(dataset_sm[features_numerical])
        prediction = self._model.predict(dataset_sm[chain(features_numerical, features_categorical)])
        return prediction

    def fit(self, X, y, **params):
        X_train = X.copy()
        X = X_train
        if self.model_config.objective is not None:
            objective = self.model_config.objective
            if objective == ModelsParams.poisson:
                sm_family = sm.families.Poisson(sm.families.links.Log())
            elif objective == ModelsParams.gamma:
                sm_family = sm.families.Gamma(sm.families.links.Log())
            else:
                sm_family = sm.families.Gaussian()
        else:
            logger.info('Gaussian family setting')
            sm_family = sm.families.Gaussian()
        if self.model_config.treatment_dict:
            logger.info('Wrapper model||Reading user base levels for GLM')
            features_manual_base = self.model_config.treatment_dict.keys()
            features_auto = [item for item in self.datasubset.features_categorical if item not in features_manual_base]
            formula = f"{y.name} ~ {' + '.join(self.datasubset.features_numerical)} {' + ' if self.datasubset.features_numerical and features_auto else ' '} {' + '.join(['C(' + feature + ')' for feature in features_auto])} {' + ' if features_manual_base else ' '} {' + '.join(['C(' + feature + ', Treatment(reference=' + self.model_config.treatment_dict[feature] + '))' for feature in features_manual_base])} "

        else:
            formula = f"{y.name} ~ {' + '.join(self.datasubset.features_numerical)} {' + ' if self.datasubset.features_numerical and self.datasubset.features_categorical else ' '} {' + '.join(['C(' + feature + ')' for feature in self.datasubset.features_categorical])}"

        self._min_max_scaler = MinMaxScaler()
        self._min_max_scaler.fit(X[self.datasubset.features_numerical])
        X[self.datasubset.features_numerical] = self._min_max_scaler.transform(X[self.datasubset.features_numerical])

        exposure_train = self.datasubset.exposure_train.loc[self.datasubset.exposure_train.index.isin(X.index)] if self.datasubset.exposure_train is not None else None
        self._model = (self.sm_model(
            formula,
            pd.concat([X, y], axis=1, join="outer", ignore_index=False),
            family=sm_family,
            freq_weights=exposure_train,
        ).fit(**params))
        return self._model

    @property
    def min_max_scaler(self):
        return self._min_max_scaler

    @property
    def model(self):
        return self._model