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
from outboxml.dataset_retro import RetroDataset
from outboxml.datadrift import DataDrift
from outboxml.core.data_prepare import prepare_dataset
from outboxml.core.pydantic_models import AllModelsConfig, DataModelConfig, ModelConfig
from outboxml.extractors import Extractor, BaseExtractor
from outboxml.metrics.base_metrics import BaseMetric, BaseMetrics
from outboxml.core.prepared_datasets import PrepareDataset, TrainTestIndexes
from outboxml.models import DefaultModels
from outboxml import config


class ModelDataSubset:
    """Container of prepared datasets and features"""

    def __init__(
            self,
            model_name: str,
            X_train: pd.DataFrame = pd.DataFrame(),
            y_train: pd.Series = pd.Series(),
            X_test: Optional[pd.DataFrame] = None,
            y_test: Optional[pd.Series] = None,
            features_numerical: Optional[List[str]] = None,
            features_categorical: Optional[List[str]] = None,
            X: Optional[pd.DataFrame] = None,
            exposure_train: Optional[pd.Series] = None,
            exposure_test: Optional[pd.Series] = None,
            extra_columns: Optional[pd.DataFrame] = None
    ):
        self.model_name: str = model_name
        #  self.wrapper: str = wrapper
        self.X_train: pd.DataFrame = X_train
        self.y_train: pd.Series = y_train
        self.X_test: Optional[pd.DataFrame] = X_test
        self.y_test: Optional[pd.Series] = y_test
        self.features_numerical: Optional[List[str]] = features_numerical
        self.features_categorical: Optional[List[str]] = features_categorical
        self.X: Optional[pd.DataFrame] = X
        self.exposure_train: Optional[pd.Series] = exposure_train
        self.exposure_test: Optional[pd.Series] = exposure_test
        self.extra_columns = extra_columns

    @classmethod
    def load_subset(
            cls,
            model_name: str,
            X: pd.DataFrame,
            Y: pd.DataFrame,
            index_train: pd.Index,
            index_test: pd.Index,
            features_numerical: Optional[List[str]] = None,
            features_categorical: Optional[List[str]] = None,
            column_exposure: Optional[str] = None,
            column_target: Optional[str] = None,
            extra_columns: Optional[pd.DataFrame] = None,
    ):
        X_train = X[X.index.isin(X.index.intersection(index_train))]
        Y_train = Y[Y.index.isin(Y.index.intersection(index_train))]

        exposure_train = Y[Y.index.isin(Y.index.intersection(index_train))][
            column_exposure] if column_exposure else None

        X_test = X[X.index.isin(X.index.intersection(index_test))]
        Y_test = Y[Y.index.isin(Y.index.intersection(index_test))]

        if column_target is not None:
            Y_train = Y_train[column_target]
            Y_test = Y_test[column_target]
        exposure_test = Y[Y.index.isin(Y.index.intersection(index_test))][column_exposure] if column_exposure else None

        return cls(
            model_name,
            X_train,
            Y_train,
            X_test,
            Y_test,
            features_numerical,
            features_categorical,
            X,
            exposure_train,
            exposure_test,
            extra_columns

        )


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
                 datasubset: ModelDataSubset,
                 model_config: ModelConfig=None,
                 config: AllModelsConfig = None
                 ):
        self.predictions = {'train': None, 'test': None}
        self.model_name = model_name
        self.config = config
        self.metrics = {'train': None, 'test': None}
        self.model = model
        self.datasubset = datasubset
        self.model_config = model_config

    def load_metrics(self, metrics: dict, ds_type: str):
        self.metrics[ds_type] = metrics

    def load_predictions(self, df: pd.DataFrame, ds_type: str):
        self.predictions[ds_type] = df

    def dict_for_prod_export(self, ):
        """Конверте DSManagerResult в пикл"""
        model = self.model
        model_results = {
            "model_config": self.model_config.model_dump(),
            "model": model,
            "min_max_scaler":  None,
            "features_numerical": self.datasubset.features_numerical,
            "features_categorical": self.datasubset.features_categorical,
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
                   datasubset=ModelDataSubset(model_name=model_name,
                                              features_numerical=features_numerical,
                                              features_categorical=features_categorical,
                                              ),
                   model=model,
                   model_config=model_config)

    @property
    def X(self):
        return pd.concat([self.datasubset.X_train, self.datasubset.X_test])

    @property
    def y(self):
        return pd.concat([self.datasubset.y_train, self.datasubset.y_test])

    @property
    def y_pred(self):
        y_pred = pd.concat([self.predictions['train'], self.predictions['test']])
        return y_pred

    @property
    def exposure(self):
        exposure = pd.concat([self.datasubset.exposure_train, self.datasubset.exposure_test])
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
    ):
        self._exposure = {}
        self._all_models_config_name: Union[str, Dict] = config_name
        self._results: [Dict[str, DSManagerResult]] = {}
        self._extractor: Optional[Extractor] = extractor
        self._prepare_datasets: Optional[Dict[str, PrepareDataset]] = prepared_datasets
        self._models_dict: Optional[Dict] = models_dict
        self._use_baseline_model = use_baseline_model
        self._business_metric: Optional[BaseMetric] = business_metric
        self.dataset: Optional[pd.DataFrame] = None
        self.X: Optional[pd.DataFrame] = None
        self.Y: Optional[pd.DataFrame] = None
        self.index_train: Optional[pd.Index] = None
        self.index_test: Optional[pd.Index] = None
        self.data_subsets = {}
        self.targets_columns_names = []
        self.extra_columns: Optional[pd.DataFrame] = None
        self.all_models_config: Optional[AllModelsConfig] = None
        self.group_name = 'general'
        self.data_config: Optional[DataModelConfig] = None
        self._models_configs: List[ModelConfig] = []

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

    def _init_dsmanager(self):
        logger.debug('Initializing DSManager')
        self.__load_all_models_config()
        self.__load_targets_names()
        if self._retro_changes is not None:
            self.__init_retro()
        self.__load_prepare_datasets()
        logger.debug('Initializing completed')

    def load_dataset(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """Load data from source due to config or user-defined extractor object. Use .env file or external config for extracor
        Also you can load dataset by parameter data"""

        if not self._is_initialized:
            self._init_dsmanager()

        logger.debug("Dataset loading")
        if data is not None:
            logger.debug("Data from user import")
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

        logger.debug("Dataset is extracted")
        self._is_initialized = True

        return self.dataset

    def get_TrainDfs(self, model_name: str = None, save_prepared_subsets: bool = False, load_subsets_from_pickle: bool=False):
        """Returns dataset for training user-defined models.
        Use save_prepared_subsets parameter for saving in enviroment
        For exposure or extra_columns data use ds_manager.data_subsets[model_name]"""

        if load_subsets_from_pickle:
           self._load_subsets_from_pickle(model_name)
        else:
            if not self.__test_train: self._make_test_train()
            if save_prepared_subsets:
                with open(os.path.join(Path(__file__).parent.parent, 'prepared_subsets.pickle'), "wb") as f:
                    pickle.dump(self.data_subsets, f)
        if model_name is None:
            model_name = self._default_name
        logger.debug('Model ' + model_name + ' || Train subset export')
        return self.data_subsets[model_name].X_train.copy(), self.data_subsets[model_name].y_train.copy()

    def get_TestDfs(self, model_name: str = None, save_prepared_subsets: bool = False, load_subsets_from_pickle: bool=False):
        """Returns dataset for training user-defined models
        Use save_prepared_subsets parameter for saving in enviroment
        For exposure or extra_columns data use ds_manager.data_subsets[model_name]"""

        if load_subsets_from_pickle:
           self._load_subsets_from_pickle(model_name)
        else:
            if not self.__test_train: self._make_test_train()
            if save_prepared_subsets:
                with open(os.path.join(Path(__file__).parent.parent, 'prepared_subsets.pickle'), "wb") as f:
                    pickle.dump(self.data_subsets, f)
        if model_name is None: model_name = self._default_name
        logger.debug('Model ' + model_name + ' || Test subset export')
        return self.data_subsets[model_name].X_test.copy(), self.data_subsets[model_name].y_test.copy()

    def fit_models(self, models_dict: dict = None, need_fit: bool = False, model_name: str = None,
                   load_subsets_from_pickle: bool = False, classification :bool=False):
        """Fitting and calculating metrics for models. If 'need_fit' option then fit methods are calling for models
        Uf load_subsets_from_pickle option then loading previously saved datasubsets in enviroment"""
        if load_subsets_from_pickle:
            self._load_subsets_from_pickle()
        else:
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
        metrics_train = self.__getTrainResults(models=models, fitted=fitted)
        try:
            metrics_test = self.__getTestResults(models=models)
        except:
            logger.error('No data for test metrics')
        return metrics_train
    def _load_subsets_from_pickle(self, model_name: str=None):
        try:
            logger.debug('Loading saved subsets')
            with open(os.path.join(Path(__file__).parent.parent, 'prepared_subsets.pickle'), "rb") as f:
                self.data_subsets = pickle.load(f)
            if model_name is not None:
                for key in list(self.data_subsets.keys()):
                    if key != model_name:
                        self.data_subsets.pop(key, None)
        except Exception as exc:
            logger.error('Loading subsets error||Using internal subsets' + str(exc))
            if not self.__test_train: self._make_test_train()

    def __getTestResults(self, models: dict):
        return self.__prepare_metrics(models, 'test')

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
        features_numerical = model_result.datasubset.features_numerical
        features_categorical = model_result.datasubset.features_categorical
        data_to_predict = data.copy()
        dataset = prepare_dataset(
            model_name,
            data_to_predict,
            train_ind=train_ind,
            test_ind=test_ind,
            model_config=model_config,
            check_prepared=False,
            calc_corr=False,
            save_data=False
        ).data
        if full_output:
            datasubset = ModelDataSubset.load_subset(model_name=model_name,
                                                     X=dataset,
                                                     Y=data,
                                                     features_numerical=features_numerical,
                                                     features_categorical=features_categorical,
                                                     index_train=train_ind,
                                                     index_test=test_ind,
                                                     column_exposure=model_config.column_exposure,
                                                     column_target=model_config.column_target
                                                     )
        else:
            datasubset = None
        output_model = model
        prediction = model.predict(dataset[chain(features_numerical, features_categorical)])
        if isinstance(prediction, np.ndarray):
            prediction = pd.Series(prediction, index=dataset.index)
        res = self._prepare_ds_result(model_name=model_name,
                                      config=model_result.config,
                                      model=output_model,
                                      datasubset=datasubset,
                                      model_config=model_config,
                                      prediction=prediction,
                                      full_output=full_output)

        logger.debug('Prediction for external data finished')
        return res

    def _separateTestTrain(self, retro_changes: RetroDataset = None):

        if self.dataset is None:
            self.load_dataset()
        logger.debug("Separation started")
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

        self.__train_test_indexes = True
        logger.debug('Separation finished')

    def _make_test_train(self):

        if not self.__train_test_indexes:
            self._separateTestTrain(retro_changes=self._retro_changes)

        for model_name in self._prepare_datasets.keys():
            self._default_name = model_name
            logger.debug('Model ' + model_name + ' || Data preparation started')
            X, y, target = self._filter_data_by_exposure(model_name)
            model_config = self._prepare_datasets[model_name]._model_config
            prepare_dataset_result = self._prepare_datasets[model_name].prepare_dataset(
                data=X,
                train_ind=self.index_train,
                test_ind=self.index_test,
                target=target
            )
            X = prepare_dataset_result.data
            self._prepare_datasets[model_name]._model_config = deepcopy(prepare_dataset_result.model_config)

            self.data_subsets[model_name] = ModelDataSubset.load_subset(
                model_name=model_name,
                X=X,
                Y=y,
                index_train=self.index_train,
                index_test=self.index_test,
                features_numerical=prepare_dataset_result.features_numerical if model_config is not None else [],
                features_categorical=prepare_dataset_result.features_categorical if model_config is not None else [],
                column_exposure=model_config.column_exposure if model_config.column_exposure else None,
                column_target=model_config.column_target,
                extra_columns=self.extra_columns
            )

            logger.debug('Model ' + model_name + ' || Data preparation finished')
        self.__test_train = True

    def __getTrainResults(self, models: dict, fitted: bool = False):


        if not fitted:
            logger.info('Fitting')
            fitted_models = {}
            for model_name, model in models.items():
                try:
                    model.fit()
                except:
                    logger.debug('User-defined model needs X, Y for train. Using datasubsets')
                    model.fit(self.data_subsets[model_name].X_train, self.data_subsets[model_name].y_train, )
                    logger.debug('Model '+ str(model_name) + ' is fitted')

        metrics = self.__prepare_metrics(models, 'train')
        logger.info('Result metrics:' + str(metrics))
        return metrics

    def _predict(self, model, X):
        if (type(model).__name__ == 'CatBoostRegressor') and (
                model.get_param('loss_function') == 'RMSEWithUncertainty'):
            logger.info('CatBoostRegressor in work..')
            predictionSeries = pd.Series(data=model.predict(X.astype(str))[:, 0], index=X.index)
            logger.info('CatBoostRegressor finished')
        elif type(model).__name__ == 'CatBoostRegressor':
            logger.info('CatBoostRegressor in work..')
            predictionSeries = pd.Series(data=model.predict(X.astype(str)), index=X.index)
            logger.info('CatBoostRegressor finished')
        elif type(model).__name__ == 'Prophet':
            logger.info('Prophet in work..')
            data = model.predict(X)
            predictionSeries = pd.Series(data=np.expm1(data['yhat']), index=data.index)
            logger.info('Prophet finished')
        elif is_classifier(model):
            predictionSeries = pd.Series(data=model.predict_proba(X)[:, 1], index=X.index)
        else:
            try:
                data = model.predict(X)
                predictionSeries = pd.Series(data=data, index=X.index)
            except:
                logger.info('Using label encoder for prediction')
                le = LabelEncoder()
                for column_name in X.columns:
                    X[column_name] = le.fit_transform(X[column_name])
                predictionSeries = pd.Series(data=model.predict(X), index=X.index)
        return predictionSeries

    def __prepare_y_df(self, dsType, data_subsets):
        y_trueDf = pd.DataFrame()
        slicedDf = pd.DataFrame()
        choose_name_func = lambda nameColumn: nameColumn + '_model_1'
        if dsType == 'test':
            for d in data_subsets:
                y_trueSeries = pd.Series(data=d.y_test, index=d.y_test.index)
                y_trueDf = pd.concat([y_trueDf, y_trueSeries.rename(choose_name_func(d.model_name))], axis=1)
                slicedDf = d.X_test.copy()

        elif dsType == 'train':
            for d in data_subsets:
                y_trueSeries = pd.Series(data=d.y_train, index=d.y_train.index)
                y_trueDf = pd.concat([y_trueDf, y_trueSeries.rename(choose_name_func(d.model_name))], axis=1)
                slicedDf = d.X_train.copy()
        else:
            logger.error('Unknown dstype')
        return y_trueDf, slicedDf

    def __prepare_metrics(self, models: dict, dsType: str):
        dsNames = list(models.keys())
        dataSubsets = [x for x in self.data_subsets.values() if x.model_name in dsNames]
        y_trueDf, slicedDf = self.__prepare_y_df(dsType=dsType, data_subsets=dataSubsets)
        modelResults = pd.DataFrame()

        for model_name in self._prepare_datasets.keys():
            targetColumn = self._prepare_datasets[model_name].get_model_config().column_target
            objective = self._prepare_datasets[model_name].get_model_config().objective
            classification = True if objective == ModelsParams.binary else False
            logger.info('Prediction ' + str(model_name))
            if model_name in dsNames:
                dataSubset = next((x for x in dataSubsets if x.model_name == model_name), None)
                if dsType == 'test':
                    X = dataSubset.X_test.copy()
                elif dsType == 'train':
                    X = dataSubset.X_train.copy()
                else:
                    raise 'Unknown dsType!'
                if not isinstance(models[model_name], list):
                    models[model_name] = [models[model_name]]
                for idx, model in enumerate(models[model_name]):
                    prediction_series = self._predict(model=model, X=X)
                    targetColumnEnsembly = model_name + '_model_' + str(idx + 1)
                    modelResults = pd.concat([modelResults, prediction_series.rename(targetColumnEnsembly)], axis=1)
                    if idx > 0:
                        y_trueDf[targetColumnEnsembly] = y_trueDf[model_name + '_model_1']
                        y_trueDf[targetColumnEnsembly].index = y_trueDf[model_name + '_model_1'].index
                        modelResults[targetColumnEnsembly] = modelResults[targetColumnEnsembly] + modelResults[
                            model_name + '_model_' + str(idx)]
                    try:
                        self._results[model_name]

                    except KeyError:
                        self._results[model_name] = DSManagerResult(model_name=model_name,
                                                                    config=self._all_models_config,
                                                                    model=model,
                                                                    datasubset=dataSubset,
                                                                    model_config=self._prepare_datasets[
                                                                        model_name].get_model_config()
                                                                    )
                    self._results[model_name].load_predictions(df=prediction_series, ds_type=dsType)

        metrics = self._metric_loop(modelResults, y_trueDf, slicedDf, classification)
        for key in self._results.keys():
            self._results[key].load_metrics(metrics, ds_type=dsType)
        return metrics

    def _metric_loop(self, modelTestResults, y_testDf, slicedDf, classification) -> dict:
        results = {}
        # modelTestResults.to_parquet('Model_test_results.gzip')
        logger.info('calculate dataset metrics')
        results['full'] = self._calculateMetrics(y_testDf, modelTestResults, classification)

        try:
            if len(self.data_config.data.targetslices) > 0:

                for dataSlice in self.data_config.data.targetslices:
                    logger.info('Metrics for slice ' + dataSlice['column'])
                    if dataSlice['type'] == 'numerical':
                        slicedDf['slice'] = dataSlice['column'] + '_' + pd.cut(slicedDf[dataSlice['column']],
                                                                               dataSlice['slices']).astype(str)
                    elif dataSlice['type'] == 'categorical':
                        slicedDf['slice'] = dataSlice['column'] + '_' + slicedDf[dataSlice['column']].astype(str)
                    else:
                        raise Exception("Unknown slice type")
                    for name in slicedDf['slice'].unique():
                        logger.info('Slice ' + name)
                        sliceIndex = slicedDf.loc[slicedDf['slice'] == name].index
                        y_trueDfIndexed = y_testDf[y_testDf.index.isin(sliceIndex)]
                        modelTestResultsIndexed = modelTestResults[modelTestResults.index.isin(sliceIndex)]
                        if (y_trueDfIndexed.empty):
                            results[name] = None
                        else:
                            results[name] = self._calculateMetrics(y_trueDfIndexed, modelTestResultsIndexed, classification)
        except AttributeError:
            logger.error('Metrics calculation||No Target slices in config')
        return results

    def _calculateMetrics(self, y_trueDf: pd.DataFrame, modelTestResults: pd.DataFrame, classification: bool=False):
        ## add
        # return y_trueDf
        logger.debug('Calculating metrics')
        returnMetrics = {}
        exposure_names = list(self._prepare_datasets.keys())
        i = 0
        for column in modelTestResults:
            returnMetrics[column] = {}
            logger.debug('Metrics for ' + column)
            exposure = None
            y_true = y_trueDf[column].dropna()
            y_pred = modelTestResults[column].dropna()
            if self._exposure[exposure_names[i]] is not None:
                exposure = self._exposure[exposure_names[i]].loc[
                    self._exposure[exposure_names[i]].index.isin(y_true.index)]

            metrics_dict = BaseMetrics(y_true, y_pred, exposure,
                                       ).calculate_metric(classification)
            returnMetrics[column] = metrics_dict
            i += 1
            logger.info(metrics_dict)
        try:
            returnMetrics['BusinessValue'] = self._calculate_business_metric(y_trueDf, modelTestResults,
                                                                             self.extra_columns)

        except:
            logger.error('Business metric error')
        return returnMetrics

    def _calculate_business_metric(self, y_true, modelTestResults, extra_columns) -> dict:
        metric = 0
        try:
            if self._business_metric is not None:
                metric = self._business_metric.calculate_metric(modelTestResults, y_true, extra_columns)
                logger.info('calculating user business metric')
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

        self.group_name = f"{self._all_models_config.project}_{self._all_models_config.version}"
        self.data_config = self._all_models_config.data_config
        self._models_configs = self._all_models_config.models_configs
        logger.info("Config is loaded")

    def __load_targets_names(self):

        self.random_state = self.data_config.separation.random_state

        if self.random_state:
            np.random.seed(self.random_state)

        self.targets_columns_names = list(set(
            [model.column_target for model in self._models_configs]
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

    def __load_models(self):
        if self._models_dict is None:
            self._models_dict = DefaultModels(dataset=self.dataset,
                                              data_subsets=self.data_subsets,
                                              models_configs=self._models_configs,
                                              group_name=self.group_name,
                                              baseline_model=self._use_baseline_model).load_default()

    def get_result(self):
        return self._results

    def __init_retro(self):
        logger.debug('Initializing retro')
        self._retro_dataset = self._retro_changes.get_retro_dataset()
        self._models_configs = self._retro_changes.models_config_for_retro(models_config=self._models_configs,
                                                                           target_columns_names=self.targets_columns_names)

    def _filter_data_by_exposure(self, model_name):
        X = self.X.copy()
        y = self.Y.copy()
        self._exposure[model_name] = None
        model_config = None
        if self._prepare_datasets[model_name]._model_config is not None:
            model_config = self._prepare_datasets[model_name]._model_config
        target = y[model_config.column_target]
        if model_config.column_exposure:
            self._exposure[model_name] = y[model_config.column_exposure]
            X = X.loc[self._exposure[model_name] > 0]
            y = y.loc[y.index.isin(X.index)]
            target = y[model_config.column_target] / y[model_config.column_exposure]
        return X, y, target

    def _prepare_ds_result(self, model_name: str, config, model, datasubset, model_config, prediction,
                           full_output):
        res = DSManagerResult(model_name=model_name,
                              config=config,
                              model=model,
                              datasubset=datasubset,
                              model_config=model_config,
                              )
        res.load_predictions(prediction.loc[prediction.index.isin(datasubset.y_train.index)], ds_type='train')
        res.load_predictions(prediction.loc[prediction.index.isin(datasubset.y_test.index)], ds_type='test')
        if full_output:
            metrics_train = {}
            metrics_train['full'] = BaseMetrics(y_true=datasubset.y_train,
                                                y_pred=prediction.loc[prediction.index.isin(datasubset.y_train.index)],
                                                exposure=datasubset.exposure_train).calculate_metric()

            metrics_test = {}
            metrics_test['full'] = BaseMetrics(y_true=datasubset.y_test,
                                               y_pred=prediction.loc[prediction.index.isin(datasubset.y_test.index)],
                                               exposure=datasubset.exposure_test).calculate_metric()
            res.load_metrics(metrics_train, ds_type='train')
            res.load_metrics(metrics_test, ds_type='test')
            logger.info('Train' + str(metrics_train))
            logger.info('Test' + str(metrics_test))
        return res
