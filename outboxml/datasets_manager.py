import json
import os
import pickle
from copy import deepcopy
from itertools import chain
from pathlib import Path

from pydantic import ValidationError
from sklearn.base import is_classifier
import pandas as pd
import numpy as np
from loguru import logger
from typing import List, Dict, Any, Optional, Union
from sklearn.preprocessing import LabelEncoder

from outboxml.core.enums import ModelsParams
from outboxml.data_subsets import DataPreprocessor, ModelDataSubset
from outboxml.dataset_retro import RetroDataset
from outboxml.datadrift import DataDrift
from outboxml.core.data_prepare import prepare_dataset
from outboxml.core.pydantic_models import AllModelsConfig, DataModelConfig, ModelConfig
from outboxml.extractors import Extractor, BaseExtractor
from outboxml.metrics.base_metrics import BaseMetric, BaseMetrics
from outboxml.core.prepared_datasets import PrepareDataset, TrainTestIndexes
from outboxml.metrics.processor import ModelMetrics
from outboxml.models import DefaultModels
from outboxml import config


class DSManagerResult:
    """Класс контейнер результатов

    Parameters
    ___________
    model_name: имя модели
    config: конфиг файл с исходными данными
    model: обученная модель
    datasubset: Объект ModelDataSubset, содержащий вектора X_train/test, Y_train/test, имена числовых и категориальных фичей, экспозицию
    model config: конфиг модели
    __________
    Methods
    dict_for_prod_export - Возвращает словаь для формирования pickle файла для сервиса
    from_pickle_model_result - Конвертер словаря из пикл сервиса в объект (class method)
    ______
    Properties:
    ______
    X - вектор X
    y_pred - Predictions
    y - y_true
    exposure - вектор экспозиции
    """

    def __init__(self,
                 model_name: str,
                 model: Any,
                 data_subset: ModelDataSubset,
                 model_config: ModelConfig = None,
                 config: AllModelsConfig = None,
                 predictions: dict = None,
                 metrics: dict = None,
                 ):
        if predictions is None:
            self.predictions = {'train': None, 'test': None}
        else:
            self.predictions = predictions
        self.model_name = model_name
        self.config = config
        if metrics is None:
            self.metrics = {'train': None, 'test': None}
        else:
            self.metrics = metrics
        self.model = model
        self.data_subset = data_subset
        self.model_config = model_config

    def load_metrics(self, metrics: dict, ds_type: str = None):
        if ds_type is not None:
            self.metrics[ds_type] = metrics
        else:
            self.metrics = metrics

    def load_predictions(self, df: Union[pd.DataFrame, pd.Series], ds_type: str):
        self.predictions[ds_type] = df

    def dict_for_prod_export(self, ):
        """Конверте DSManagerResult в пикл"""
        model = self.model
        model_results = {
            "model_config": self.model_config.model_dump(),
            "model": model,
            "min_max_scaler": None,
            "features_numerical": self.data_subset.features_numerical,
            "features_categorical": self.data_subset.features_categorical,
        }

        return model_results

    @classmethod
    def from_pickle_model_result(cls, model_result: dict, all_model_config: AllModelsConfig, ):
        """Конвертер пикла в DSManagerResult. Используется библиотечный вид модели (wrapper)"""

        model_config = model_result['model_config']
        model_name = model_result['model_config']['name']
        model_config = ModelConfig.model_validate(model_config)
        model = model_result['model']
        features_numerical = model_result['features_numerical']
        features_categorical = model_result['features_categorical']
        return cls(model_name=model_name,
                   config=all_model_config,
                   data_subset=ModelDataSubset(model_name=model_name,
                                               features_numerical=features_numerical,
                                               features_categorical=features_categorical,
                                               ),
                   model=model,
                   model_config=model_config)

    @property
    def X(self):
        return pd.concat([self.data_subset.X_train, self.data_subset.X_test])

    @property
    def y(self):
        return pd.concat([self.data_subset.y_train, self.data_subset.y_test])

    @property
    def y_pred(self):
        y_pred = pd.concat([self.predictions['train'], self.predictions['test']])
        return y_pred

    @property
    def exposure(self):
        exposure = pd.concat([self.data_subset.exposure_train, self.data_subset.exposure_test])
        return exposure


class DataSetsManager:
    """Основной класс для работы с моделями.

    Для работы с фреймоврком необходим заполенный по правилам config файл.
    Из коробки работа производится по параметрам конфига DataSetsManager(config_name = config).
    
    Для пользовательской настройки фреймоворка необходимо импортировать модули:
    from outboxml.extractors.extractor import Extractor, RTDMExtractor, ActuarExtractor
    from outboxml.metrics.metrics import BaseMetric
    from outboxml.models import Model
    from outboxml.dataset_retro import RetroDataset
    from outboxml.export_results import ResultsExport

    Работа с фрейморком начинается с создания объекта с параметрами.
    Обязательный параметр на входе: путь к конфиг файлу или сам валидированный конифиг-файл AllModesConfig
    Остальные параметры могут быть установлены автоматически "из коробки",
    Информация о работе выводится в виде лог-файла.
    Результаты моделирования выводятся в контейнере результатов DSManagerResult


    Parameters:
    ----------
    config_name: path to config or validated config file

    extractor: User-defined extractor object inheritanced by Extractor interface.
               main method - extract_dataset() should return Pandas Dataframe
               extractor should contain check_object() method with data validation and verification
               Use library RTDMExtractor or ActuarExtractor for working with databases

    modified_data: dict {name: PreparedDataset} with models preparation objects. Wrapper of prepare_dataset function
                        PreparedDataset by default uses model_config features and has no prep and post prep functions


    models_dict: dict {name: Model} with models for train and prediction inheritanced by Model class.
                class should have fit(), predict() and models_dict() methods.
                By default model is chosen by group and project name. You can import models from library

    business_metric: user-defined business metric.
                    The inheritace of BaseMetric. Main method is calculate_metric()

    use_baseline_model: bool , Выбор Baseline. 1 - RandomForestRegressor, 2 - DummyRegressor median; 3 - mean

    retroChanges: RetroDataset object for retro analysis.

    Methods:
    __________
    load_dataset() - loading due to source and path from config file or user-defined extractor class
    get_trainDfs(model_name: str) - return prepared datasubset to train user model
    get_testDfs(model_name: str) - return prepared datasubset to test user model
    fit_models({model_name: model, ...}, need_fit=True) - fit< predict and get metrics for all models in model_dict
    get_result() - return container of results DSManagerResult
    check_datadrift(model_name: str) - returns dataframe with datadrift analysis result
    ----------
    Examples:
    ----------
    To see more examples go to outboxml/examples repository


    Examples:
    _______
    1. Titanic

        #post prep function
        def data_post_prep_func(data: pd.DataFrame):
            data["SEX"] = pd.to_numeric(data["SEX"])
            return data

        titanic_ds_manager =  DataSetsManager(config_name=config_name,
                                         extractor=TitanicExampleExtractor(path_to_file=path_to_data),
                                         prepared_datasets={
                                                        'first': PrepareDataset(group_name='survived1',
                                                                                    data_post_prep_func=data_post_prep_func,
                                                                                    check_prepared=True,
                                                                                    calc_corr=True),
                                                        'second': PrepareDataset(group_name='survived2',)
                                                            },
                                         business_metric=TitanicExampleMetric()

                                         )
        titanic_TrainDs = titanic_ds_manager.get_TrainDfs(model_name='first')
        titanic_results = titanic_ds_manager.fit_models()

    """

    def __init__(
            self,
            config_name: Union[str, Dict],
            extractor: Optional[Extractor] = None,
            prepared_datasets: Optional[Dict[str, PrepareDataset]] = None,
            models_dict: Optional[Dict] = None,
            business_metric: Optional[BaseMetric] = None,
            use_baseline_model: int = 0,
            retro_changes: Optional[RetroDataset] = None,
            external_config=None,
            use_temp_files: bool=False,
    ):
        if external_config is None:
            self._external_config = config
        else:
            self._external_config = external_config
        self._use_temp_files = use_temp_files
        self._exposure = {}
        self._all_models_config_name: Union[str, Dict] = config_name
        self._results: Dict[str, DSManagerResult] = {}
        self._extractor: Optional[Extractor] = extractor
        self._prepare_datasets: Optional[Dict[str, PrepareDataset]] = prepared_datasets
        self._models_dict: Optional[Dict] = models_dict
        self._use_baseline_model = use_baseline_model
        self._business_metric: Optional[BaseMetric] = business_metric
        self.dataset: Optional[pd.DataFrame] = None
        self._data_preprocessor: Optional[DataPreprocessor] = None
        self.X: Optional[pd.DataFrame] = None
        self.Y: Optional[pd.DataFrame] = None
        self.index_train: Optional[pd.Index] = None
        self.index_test: Optional[pd.Index] = None
        self.targets_columns_names = []
        self.extra_columns: Optional[pd.DataFrame] = None
        self.all_models_config: Optional[AllModelsConfig] = None
        self.group_name = 'general'
        self.data_config: Optional[DataModelConfig] = None
        self._models_configs: List[ModelConfig] = []
        self._retro = False
        self.business_metric_value = {}
        self._is_initialized: bool = False
        self.__test_train: bool = False
        self.__train_test_indexes: bool = False

        self._retro_changes = retro_changes
        self._retro_dataset = None

        self._default_name = None

    @property
    def config(self):
        config_to_return = deepcopy(self._all_models_config)
        if self._results != {}:
            updated_models_configs = []
            for result in self._results.values():
                updated_models_configs.append(result.model_config)
            config_to_return.models_configs = updated_models_configs

        return config_to_return

    def get_result(self) -> Dict[str, DSManagerResult]:
        return self._results

    def load_dataset(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """Load data from source due to config or user-defined extractor object. Use .env file or external config for extracor
        Also you can load dataset by parameter data"""

        if not self._is_initialized:
            self._init_dsmanager()

        logger.debug("Dataset loading")
        if data is not None:
            self.dataset = data

        elif self._extractor is not None:
            logger.info("Load data from user extractor")
            if self._extractor.load_config_from_env:
                logger.info("Reading config from env")
                self._extractor.load_config(connection_config=config)
            self.dataset = self._extractor.extract_dataset()

        else:
            self.dataset = BaseExtractor(data_config=self.data_config).extract_dataset()
        logger.debug('DataSet is extracted')

        self._is_initialized = True

        return self.dataset

    def get_TrainDfs(self, model_name: str = None, save_prepared_subsets: bool = False,
                     load_subsets_from_pickle: bool = False):
        """Returns dataset for training user-defined models.
        Use save_prepared_subsets parameter for saving in enviroment
        For exposure or extra_columns data use ds_manager.data_subsets[model_name]"""

        if not self.__test_train:
            self._make_test_train()
        if model_name is None:
            model_name = self._default_name
        data_subset = self.get_subset(model_name)

        logger.debug('Model ' + model_name + ' || Train subset export')
        return data_subset.X_train, data_subset.y_train.copy()

    def get_TestDfs(self, model_name: str = None, save_prepared_subsets: bool = False,
                    load_subsets_from_pickle: bool = False):
        """Returns dataset for training user-defined models
        Use save_prepared_subsets parameter for saving in enviroment
        For exposure or extra_columns data use ds_manager.data_subsets[model_name]"""
        if not self.__test_train:
            self._make_test_train()
        if model_name is None: model_name = self._default_name
        logger.debug('Model ' + model_name + ' || Test subset export')
        data_subset = self.get_subset(model_name)
        return data_subset.X_test, data_subset.y_test.copy()

    def get_subset(self, model_name):
        if not self.__test_train:
            self._make_test_train()
        if model_name is None: model_name = self._default_name
        logger.debug('Model ' + model_name + ' || Subset export')
        return self._data_preprocessor.get_subset(model_name)

    @property
    def data_subsets(self, ):

        return self._data_preprocessor.data_subsets()

    def fit_models(self, models_dict: dict = None, need_fit: bool = False, model_name: str = None,
                  ) -> dict:
        """Fitting and calculating metrics for models. If 'need_fit' option then fit methods are calling for models
        Uf load_subsets_from_pickle option then loading previously saved datasubsets in enviroment"""
        if not self.__test_train: self._make_test_train()
        fitted = False
        logger.debug('Fitting model started')
        if models_dict is not None:
            models = models_dict
            if need_fit:
                fitted = False
            logger.info('User-defined models')
        else:
            if self._models_dict is None:
                logger.info('Setting default models')
                self.__load_models()
                fitted = False
            models = self._models_dict

        if model_name is not None:
            try:
                chosen_model = models[model_name]
                models = {chosen_model: models[model_name]}
            except KeyError:
                logger.error('Wrong model name in input')

        fitted_models = self.__get_fitted_models(models=models, fitted=fitted)
        metrics = {}
        for model_name, model in fitted_models.items():
            data_subset = self.get_subset(model_name)
            predictions_train = self._predict(model, data_subset.X_train)
            predictions_test = self._predict(model, data_subset.X_test)
            metrics[model_name] = ModelMetrics(data_config=self.data_config,
                                               model_config=self._prepare_datasets[model_name].get_model_config(),
                                               data_subset=data_subset,
                                               ).result_dict(predictions={'train': predictions_train,
                                                                          'test': predictions_test})


            self._results[model_name] = DSManagerResult(model_name=model_name,
                                                        model=model,
                                                        config=self.config,
                                                        data_subset=data_subset,
                                                        model_config=self._prepare_datasets[
                                                            model_name].get_model_config(),
                                                        predictions={'train': predictions_train,
                                                                     'test': predictions_test},
                                                        metrics=metrics[model_name])
        try:
            metrics.update(self._calculate_business_metric())
        except Exception as exc:
            logger.error('Error while calculation business metric||'+ str(exc))
        return metrics


    def check_datadrift(self, model_name: str) -> pd.DataFrame:
        """Method for checking datadrift between train and test. Using DataDrift library"""
        X_train, y_train = self.get_TrainDfs(model_name)
        X_test, y_test = self.get_TestDfs(model_name=model_name)
        report = DataDrift().report(train_data=X_train, test_data=X_test, )

        return report

    def model_predict(self,
                      data: pd.DataFrame,
                      model_name: str,
                      model_result=None,
                      train_ind: pd.Index = pd.Index([]),
                      test_ind: pd.Index = None,
                      full_output: bool = True,
                      ) -> DSManagerResult:
        """Method for constructing DSManagerResult for external model or data.
        Use model_result as dict from service or DSManagerResult
        Use full_output option to get only prediction vector of full output"""
        logger.debug('Prediction for external data||' + model_name)
        if test_ind is None:
            test_ind = data.index
        if model_result is None:
            logger.info('No external model||Using inner results')
            result = self.get_result()
            model_result = result[model_name]
        else:
            if isinstance(model_result, dict):
                try:
                    model_result = model_result[model_name]
                except KeyError as e:
                    # logger.error(e)
                    logger.debug('Converting pickle to DSManagerResult')
                    model_result = DSManagerResult.from_pickle_model_result(model_result=model_result,
                                                                            all_model_config=self._all_models_config)

        model_config = model_result.model_config
        model = model_result.model
        features_numerical = model_result.data_subset.features_numerical
        features_categorical = model_result.data_subset.features_categorical
        data_to_predict = data.copy()
        Y = data[model_config.column_target] if model_config.column_target else pd.DataFrame()
        data_subset = DataPreprocessor(prepare_dataset_interface_dict={model_name:
                                                                           PrepareDataset(model_config=model_config,
                                                   check_prepared=False,
                                                   )},
                                       dataset=data_to_predict,
                                       train_ind=train_ind,
                                       test_ind=test_ind).get_subset(model_name, from_pickle=False)

        output_model = model
        prediction = model.predict(data_subset.X[chain(features_numerical, features_categorical)])
        if isinstance(prediction, np.ndarray):
            prediction = pd.Series(prediction, index=data_subset.X.index)
        metrics = ModelMetrics(model_config=model_config,
                               data_subset=data_subset,
                               data_config=None).result_dict(
            predictions={'train': prediction[prediction.index.isin(train_ind)],
                         'test': prediction[prediction.index.isin(test_ind)]},
        )

        res = DSManagerResult(model_name=model_name,
                              config=model_result.config,
                              model=output_model,
                              data_subset=data_subset,
                              model_config=model_config,
                              predictions={'train': prediction.loc[prediction.index.isin(train_ind)],
                                           'test': prediction.loc[prediction.index.isin(test_ind)]},
                              metrics=metrics,
                              )

        logger.debug('Prediction for external data finished')
        return res

    def _separateTestTrain(self, retro_changes: RetroDataset = None):

        if self.dataset is None:
            self.load_dataset()
        logger.debug("Separation started")
        if not self.targets_columns_names:
            logger.info('No target for clustering||Adding temp field')
            self.targets_columns_names = ['TEMP_TARGET']
            self.dataset['TEMP_TARGET'] = 1
        try:
            self.Y = self.dataset[self.targets_columns_names]

        except KeyError as e:
            logger.error("No target columns in dataset")
            raise KeyError(f"No target columns in dataset: {e}")

        self.X = self.dataset.drop(columns=self.targets_columns_names)

        logger.info(f"X shape: {str(self.X.shape)}, Y shape: {str(self.Y.shape)}")
        logger.info(f"Y: {self.Y.columns.values}")

        if self.data_config.extra_columns:
            self.extra_columns = self.dataset[self.data_config.extra_columns]
            logger.info(f"Extra columns: {str(self.extra_columns.columns.values)}")

        if retro_changes:
            logger.debug("Retro changes")
            X_extra = self._retro_dataset
            Y_extra = self._retro_dataset[self.targets_columns_names]

            if not (all(X_extra.index == self.X.index) and all(Y_extra.index == self.Y.index)):
                raise Exception(f"Wrong indexes for extra and retro data")
            self.X = X_extra
            self.Y = Y_extra

        self.index_train, self.index_test = TrainTestIndexes(
            X=self.X,
            separation_config=self.data_config.separation,
        ).train_test_indexes()
        self._data_preprocessor = DataPreprocessor(prepare_dataset_interface_dict=self._prepare_datasets,
                                                   dataset=self.dataset,
                                                   train_ind=self.index_train,
                                                   test_ind=self.index_test,
                                                   external_config=self._external_config,
                                                   version=self.version,
                                                   use_saved_files=self._use_temp_files,
                                                   extra_columns=self.data_config.extra_columns,
                                                   retro=self._retro
                                                   )
        self.__train_test_indexes = True
        logger.debug('Separation finished')

    def _make_test_train(self):

        if not self.__train_test_indexes:
            self._separateTestTrain(retro_changes=self._retro_changes)
        self.__test_train = True

    def __get_fitted_models(self, models: dict, fitted: bool = False) -> dict:
        if not fitted:
            logger.info('Fitting')
            for model_name, model in models.items():
                try:
                    model.fit()
                except:
                    logger.debug('User-defined model needs X, Y for train. Using datasubsets')
                    data_subset = self.get_subset(model_name)
                    model.fit(data_subset.X_train, data_subset.y_train)
                    logger.debug('Model ' + str(model_name) + ' is fitted')
        return models

    def _predict(self, model, X):
        if X is None: return None
        if type(model).__name__ == 'Prophet':
            logger.info('Prophet in work..')
            data = model.predict(X)
            prediction_series = pd.Series(data=np.expm1(data['yhat']), index=data.index)
            logger.info('Prophet finished')
        elif is_classifier(model):
            prediction_series = pd.Series(data=model.predict_proba(X)[:, 1], index=X.index)
        else:
            try:
                data = model.predict(X)
                prediction_series = pd.Series(data=data, index=X.index)
            except:
                logger.info('Using label encoder for prediction')
                le = LabelEncoder()
                for column_name in X.columns:
                    X[column_name] = le.fit_transform(X[column_name])
                prediction_series = pd.Series(data=model.predict(X), index=X.index)
        return prediction_series

    def _calculate_business_metric(self,) -> dict:
        metric = {}
        try:
            if self._business_metric is not None:
                metric = self._business_metric.calculate_metric(self.get_result())
                self.business_metric_value = metric
                logger.info('Business metric value||'+ str(metric))
        except ModuleNotFoundError as e:
            logger.info('No business metrics')

        logger.debug('Calculating metrics finished')
        return metric

    def __load_all_models_config(self):

        if isinstance(self._all_models_config_name, dict):
            logger.info("All models config from dict")
            all_models_config = json.dumps(self._all_models_config_name)

        else:
            logger.info("All models config from path")
            try:
                with open(self._all_models_config_name, "r", encoding='utf-8') as f:
                    all_models_config = f.read()
            except FileNotFoundError:
                logger.error("Invalid all models config name")
                raise FileNotFoundError("Invalid config name")

        try:
            self._all_models_config = AllModelsConfig.model_validate_json(all_models_config)
        except ValidationError as e:
            logger.error("Config validation error")
            raise ValidationError(e)
        self.data_config = self._all_models_config.data_config
        self._models_configs = self._all_models_config.models_configs
        self.version = self._all_models_config.version
        for model in self._models_configs:

            file_path = os.path.join(self._external_config.results_path, model.name + '_v' + self.version + '_subset.pickle')
            if os.path.exists(file_path):
                if not self._retro:
                    logger.warning(f'{model.name}||File {file_path} already exists. Change version in config file to for new data prepare')
                else:
                    logger.warning(
                        f'{model.name}||File {file_path} already exists. Changing version in config file for A/B test')
                    self.version = self.version + '_new'
        self.group_name = f"{self._all_models_config.project}_{self._all_models_config.version}"

        logger.info("Config is loaded")

    def __load_targets_names(self):

        self.random_state = self.data_config.separation.random_state

        if self.random_state:
            np.random.seed(self.random_state)

        self.targets_columns_names = list(set(
            [model.column_target for model in self._models_configs if model.column_target]
            + [model.column_exposure for model in self._models_configs if model.column_exposure]
        ))

    def __load_prepare_datasets(self):

        i = 0
        if self._prepare_datasets is None:
            self._prepare_datasets = {}
            logger.info("Load models prepare datasets")
            for model_config in self._models_configs:
                self._prepare_datasets[model_config.name] = PrepareDataset(model_config=model_config,
                                                                           check_prepared=True,
                                                                           group_name=self.group_name)

        else:
            logger.info("User models prepare datasets")
            for value in self._prepare_datasets.values():
                if value.get_model_config() is None:
                    logger.info(f"Load config  from DS Manager")
                    value.load_model_config(model_config=self._models_configs[i])
                    i += 1
        self._default_name = list(self._prepare_datasets.keys())[0]

    def __load_models(self):
        if self._models_dict is None:
            self._models_dict = DefaultModels(dataset=self.dataset,
                                              data_subsets=self._data_preprocessor.data_subsets(),
                                              models_configs=self._models_configs,
                                              group_name=self.group_name,
                                              baseline_model=self._use_baseline_model).load_default()

    def __init_retro(self):
        logger.debug('Initializing retro')
        self._retro_dataset = self._retro_changes.get_retro_dataset()
        self._models_configs = self._retro_changes.models_config_for_retro(models_config=self._models_configs,
                                                                           target_columns_names=self.targets_columns_names)

    def _init_dsmanager(self):
        logger.debug('Initializing DSManager')
        self.__load_all_models_config()
        self.__load_targets_names()
        if self._retro_changes is not None:
            self.__init_retro()
        self.__load_prepare_datasets()
        logger.debug('Initializing completed')
