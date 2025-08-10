import os
import pickle
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Union, Callable
import multiprocessing as mp

import pandas as pd
from loguru import logger

from outboxml import config
from outboxml.core.data_prepare import prepare_dataset
from outboxml.core.prepared_datasets import PrepareDataset
from outboxml.extractors import Extractor


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


class DataPreprocessor:
    def __init__(self,
                 prepare_dataset_interface_dict: Dict[str, PrepareDataset],
                 dataset: Union[pd.DataFrame, Extractor],
                 train_ind: pd.Index,
                 version: str = '1',
                 extra_columns: list = None,
                 test_ind: pd.Index = None,
                 external_config=None,
                 use_saved_files: bool=False,
                 retro: bool=False):
        self._version = version
        self._prepare_datasets = prepare_dataset_interface_dict
        self._dataset = dataset.copy()
        self._use_saved_files = use_saved_files
        self.index_train = train_ind
        self.index_test = test_ind
        self.config = external_config
        self._extra_columns = extra_columns
        self._extra_columns_data: Optional[pd.DataFrame] = None
        if external_config is None:
            self.config = config
        self._prepared_subsets = {}
        self.model_names = list(self._prepare_datasets.keys())
        self._pickle_subset = PickleModelSubset(config=self.config, version=self._version)
        self._parquet_dataset = ParquetDataset(config=self.config, parquet_name='temp_dataset_v' + self._version
                                                                                      )
        self.temp_subset: Optional[ModelDataSubset] = None
        self._data_columns = []
        self._retro = retro

    @property
    def dataset(self):
        if isinstance(self._dataset, pd.DataFrame):
            logger.info('Saving data to parquet')
            if not self._retro:
                self._collect_features_list()
                data_to_save = self._dataset[self._data_columns]
            else:
                data_to_save = self._dataset
            self._parquet_dataset.save_parquet(data_to_save)
            self._extra_columns_data = self._dataset[self._extra_columns].copy() if self._extra_columns is not None else None
       #     self._dataset = None

        elif isinstance(self._dataset, Extractor):
            data = self._dataset.extract_dataset()
            if not self._retro:
                self._collect_features_list()
                data_to_save = self._dataset[self._data_columns]
            else:
                data_to_save = self._dataset
            self._parquet_dataset.save_parquet(data_to_save)
            logger.info('Saving data to parquet')
            self._extra_columns_data = data[self._extra_columns].copy() if self._extra_columns is not None else None
         #   self._dataset = None
        logger.info('Reading data from parquet')
        return self._parquet_dataset.read_parquet()

    def save_subset_to_pickle(self, model_name: str, data_subset: ModelDataSubset, rewrite: bool=False):
        self._pickle_subset.save_subset_to_pickle(model_name, data_subset, rewrite)

    def get_subset(self, model_name: str = None, from_pickle: bool = True, prepare_func: Callable = None,
                   args: dict = None) -> ModelDataSubset:
        if model_name is None: model_name = self.model_names[0]
        if from_pickle:
            if not self._check_prepared_subset(model_name):
                self._prepare_subset(model_name, True, prepare_func, args)

            return self._pickle_subset.load_subsets_from_pickle(model_name)
        else:
            self._prepare_subset(model_name, to_pickle=False)
            return self.temp_subset

    def data_subsets(self, ) -> Dict[str, ModelDataSubset]:
        data_subsets = {}
        for model_name in self._prepare_datasets.keys():
            if not self._check_prepared_subset(model_name):
                self._prepare_subset(model_name=model_name)

        for model_name in self._prepare_datasets.keys():
            data_subsets[model_name] = self._pickle_subset.load_subsets_from_pickle(model_name)

        return data_subsets

    def _prepare_subset(self, model_name, to_pickle: bool = True, prepare_func: Callable = None,
                        args_dict: dict = None):
        logger.debug('Model ' + model_name + ' || Data preparation started')
        if not to_pickle:
            data = self._dataset
        else:
            data = self.dataset
        X, y, target = self._filter_data_by_exposure(model_name=model_name, dataset=data)
        model_config = self._prepare_datasets[model_name].get_model_config()
        if prepare_func is not None:
            prepare_dataset_result = prepare_func(X, self.index_train, self.index_test, target,
                                                  **args_dict)
        else:
            prepare_dataset_result = self._prepare_datasets[model_name].prepare_dataset(
                data=X,
                train_ind=self.index_train,
                test_ind=self.index_test,
                target=target
            )
        X = prepare_dataset_result.data
        self._prepare_datasets[model_name]._model_config = deepcopy(prepare_dataset_result.model_config)
        data_subset = ModelDataSubset.load_subset(
            model_name=model_name,
            X=X,
            Y=y,
            index_train=self.index_train,
            index_test=self.index_test,
            features_numerical=prepare_dataset_result.features_numerical if model_config is not None else [],
            features_categorical=prepare_dataset_result.features_categorical if model_config is not None else [],
            column_exposure=model_config.column_exposure if model_config.column_exposure else None,
            column_target=model_config.column_target if model_config.column_target else None,
            extra_columns=self._extra_columns_data if self._extra_columns_data is not None else None)
        logger.debug('Model ' + model_name + ' || Data preparation finished')
        if to_pickle:
            self._pickle_subset.save_subset_to_pickle(model_name, data_subset, True)
            self._prepared_subsets[model_name] = True
        else:
            self.temp_subset = data_subset

    def _filter_data_by_exposure(self, model_name: str, dataset: pd.DataFrame):
        exposure = {model_name: None}
        model_config = self._prepare_datasets[model_name].get_model_config()
        target = pd.Series()
        if model_config.column_target:
            y = dataset[model_config.column_target]
            target = y
        else:
            target = pd.Series()
            y = pd.Series()
        if model_config.column_exposure:
            exposure[model_name] = dataset[model_config.column_exposure]
            X = dataset.loc[exposure[model_name] > 0]
            y = y.loc[y.index.isin(X.index)]
            target = y / exposure[model_name]
            y = pd.concat([pd.Series(y, name=model_config.column_target),
                           pd.Series(exposure[model_name].loc[exposure[model_name].index.isin(X.index)], name=model_config.column_exposure)],axis=1)

        else:
            X = dataset
            y = pd.DataFrame(y)
        return X, y, target

    def _check_prepared_subset(self, model_name):
        file_path = os.path.join(self.config.results_path, model_name + '_v' + self._version + '_subset.pickle')
        if os.path.exists(file_path):
            if self._use_saved_files:
                logger.info(f'File {file_path} already exists.')
                self._prepared_subsets[model_name] = True
            else:
                self._prepared_subsets[model_name] = False
        if model_name in self._prepared_subsets.keys():
            return self._prepared_subsets[model_name]
        else:
            return False

    def _collect_features_list(self):
        if self._data_columns == []:
            using_features = []
            model_features = {}
            for model_config in self._prepare_datasets.values():
                model = model_config.get_model_config()
                features = model.features.copy()
                model_features[model.name] = []
                for feature in features:
                    model_features[model.name].append(feature.name)
                model_features[model.name].append(model.column_target)

                if model.column_exposure is not None:
                    model_features[model.name].append(model.column_exposure)

                if model.column_target is not None:
                    model_features[model.name].append(model.column_target)
                relative_features = model.relative_features.copy()
                for relative_feature in relative_features:
                    if relative_feature.numerator not in model_features[model.name]:
                        model_features[model.name].append(relative_feature.numerator)
                    if relative_feature.denominator not in model_features[model.name]:
                        model_features[model.name].append(relative_feature.denominator)
                using_features = using_features + model_features[model.name]
            if self._extra_columns is not None:
                self._data_columns = list(set(using_features + self._extra_columns))
            else:
                self._data_columns = list(set(using_features))


class PickleModelSubset:
    def __init__(self, config, version):
        self.results_path = config.results_path
        self.version = version

    def load_subsets_from_pickle(self, model_name: str, version: str = '1') -> ModelDataSubset:
        logger.info(model_name + '||Loading subset from pickle')
        file_path = os.path.join(self.results_path, model_name + '_v' + self.version + '_subset.pickle')
        with open(file_path, "rb") as f:
            subset = pickle.load(f)

        # avoiding cannot set WRITEABLE flag to True of this array error
        subset.X_train = subset.X_train.copy() if subset.X_train is not None else None
        subset.X_test = subset.X_test.copy() if subset.X_test is not None else None
        subset.y_train = subset.y_train.copy() if subset.y_train is not None else None
        subset.y_test = subset.y_test.copy() if subset.y_test is not None else None
        subset.exposure_train = subset.exposure_train.copy() if subset.exposure_train is not None else None
        subset.exposure_test = subset.exposure_test.copy() if subset.exposure_test is not None else None
        return subset

    def save_subset_to_pickle(self, model_name, subset: ModelDataSubset,  rewrite: bool=False):
        file_path = os.path.join(self.results_path, model_name + '_v'+ self.version + '_subset.pickle')

        if os.path.exists(file_path) and not rewrite:
            logger.warning(f'{model_name}||File {file_path} already exists.')
        else:
            logger.info(model_name + '||Saving subset to pickle')
            with open(file_path, "wb") as f:
                pickle.dump(subset, f)


class ParquetDataset:
    def __init__(self, config, parquet_name: str):
        self._parquet_name = parquet_name
        self.results_path = config.results_path

    def save_parquet(self, data: pd.DataFrame):
        file_path = os.path.join(self.results_path, self._parquet_name + '.parquet')

        logger.info('||Saving dataset to parquet')
        data.to_parquet(file_path)

    def read_parquet(self)-> pd.DataFrame:
        file_path = os.path.join(self.results_path, self._parquet_name + '.parquet')
        return pd.read_parquet(file_path)