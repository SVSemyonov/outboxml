import os
import pickle
from abc import abstractmethod, ABC
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Union, Callable, Literal
import multiprocessing as mp

import pandas as pd
import polars as pl
from loguru import logger

from outboxml import config
from outboxml.core.data_prepare import prepare_dataset
from outboxml.core.prepared_datasets import PrepareDataset, TrainTestIndexes, TrainTestIndexesPl, PrepareDatasetPl
from outboxml.core.pydantic_models import DataConfig, DataModelConfig, SeparationModelConfig
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

    @classmethod
    def load_subset_pl(
            cls,
            model_name: str,
            data: pl.DataFrame,
            features_numerical: Optional[List[str]] = None,
            features_categorical: Optional[List[str]] = None,
            column_exposure: Optional[str] = None,
            column_target: Optional[str] = None,
            extra_columns_list: Optional[List[str]] = None,
    ):
        X = data.to_pandas()

        X_train = X.loc[X["is_train_obml"] == 1].drop(columns=["is_train_obml"])
        y_train = X.loc[X["is_train_obml"] == 1][column_target] if column_target else pd.Series()
        exposure_train = X.loc[X["is_train_obml"] == 1][column_exposure] if column_exposure  else None

        X_test = X.loc[X["is_train_obml"] == 0].drop(columns=["is_train_obml"])
        y_test = X.loc[X["is_train_obml"] == 0][column_target] if column_target else pd.Series()
        exposure_test = X.loc[X["is_train_obml"] == 0][column_exposure] if column_exposure else None

        extra_columns_data = X[extra_columns_list] if extra_columns_list else None

        return cls(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            features_numerical=features_numerical,
            features_categorical=features_categorical,
            X=None,
            exposure_train=exposure_train,
            exposure_test=exposure_test,
            extra_columns=extra_columns_data,
        )


class DataPreprocessor:
    def __init__(self,
                 prepare_dataset_interface_dict: Dict[str, PrepareDataset | PrepareDatasetPl],
                 dataset: Union[pd.DataFrame, Extractor],
                 data_config: DataModelConfig,
                 version: str = '1',
                 prepare_engine: Literal['pandas', 'polars'] = 'pandas',
                 external_config=None,
                 use_saved_files: bool = False,
                 retro: bool = False):

        self._prepare_engine = prepare_engine
        self._version = version
        self._prepare_datasets = prepare_dataset_interface_dict
        self._data_config = data_config
        self._dataset = dataset
        self._use_saved_files = use_saved_files
        self.config = external_config
        self._extra_columns = self._data_config.extra_columns
        if external_config is None:
            self.config = config
        self._prepared_subsets = {}
        self.model_names = list(self._prepare_datasets.keys())
        self._pickle_subset = PickleModelSubset(config=self.config,
                                                version=self._version,
                                                prepare_datasets=self._prepare_datasets)
        self._parquet_dataset = ParquetDataset(config=self.config,
                                               parquet_name='temp_dataset_v' + self._version,
                                               prepare_engine=prepare_engine,
                                               )
        self.temp_subset: Optional[ModelDataSubset] = None
        self._data_columns = []
        self._retro = retro
        self.index_train = pd.Index([])
        self.index_test = pd.Index([])

    @property
    def dataset(self)->pd.DataFrame:
        if isinstance(self._dataset, pd.DataFrame):
            if not self._retro:
                self._collect_features_list()
                data_to_save = self._dataset[self._data_columns]
            else:
                data_to_save = self._dataset
            self._parquet_dataset.save_parquet(data_to_save)
            self._dataset = None

        elif isinstance(self._dataset, Extractor):
            data = self._dataset.extract_dataset()
            if not self._retro:
                self._collect_features_list()
                data_to_save = data[self._data_columns]
            else:
                data_to_save = data
            self._parquet_dataset.save_parquet(data_to_save)
            self._dataset = None
        logger.info('Reading data from parquet')
        return self._parquet_dataset.read_parquet()

    def save_subset_to_pickle(self, model_name: str, data_subset: ModelDataSubset, rewrite: bool = False):
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
        if not to_pickle:
            data = self._dataset
        else:
            data = self.dataset
        logger.debug('Model ' + model_name + ' || Data preparation started')
        if self._prepare_engine == 'pandas':
            prepare_engine = PandasInterface(data=data,
                                          prepare_interface=self._prepare_datasets[model_name],
                                          separation_config=self._data_config.separation,
                                          extra_columns=self._extra_columns,
                                          )
            data_subset = prepare_engine.prepared_subset(prepare_func, args_dict)
            self.index_train, self.index_test = prepare_engine.get_train_test_indexes()

        elif self._prepare_engine == 'polars':
            prepare_engine = PolarsInterface(data=self.dataset,
                                             prepare_interface=self._prepare_datasets[model_name],
                                             separation_config=self._data_config.separation,
                                             extra_columns=self._extra_columns)
            data_subset = prepare_engine.prepared_subset(prepare_func, args_dict)
            self.index_train, self.index_test = data_subset.X_train.index, data_subset.X_test.index

        else:
            raise f'Unknow engine for data preparation'

        if to_pickle:
            self._pickle_subset.save_subset_to_pickle(model_name, data_subset, True)
            self._prepared_subsets[model_name] = True
        else:
            self.temp_subset = data_subset


    def _check_prepared_subset(self, model_name):
        file_path = os.path.join(self.config.results_path, model_name + '_v' + self._version + '_subset.pickle')
        if os.path.exists(file_path):
            if self._use_saved_files:
                logger.info(f'File {file_path} already exists.')
                self._prepared_subsets[model_name] = True
                return True
            elif model_name in self._prepared_subsets.keys():
                return self._prepared_subsets[model_name]
            else:
                self._prepared_subsets[model_name] = False

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
    def __init__(self, config, version, prepare_datasets):
        self.prepare_datasets = prepare_datasets
        self.results_path = config.results_path
        self.version = version

    def load_subsets_from_pickle(self, model_name: str, version: str = '1') -> ModelDataSubset:
        logger.info(model_name + '||Loading subset from pickle')
        file_path = os.path.join(self.results_path, model_name + '_v' + self.version + '_subset.pickle')
        file_path_prepare_dataset_model_config = os.path.join(self.results_path,
                                                              model_name + '_v' + self.version + '_prepare_model_config.pickle')
        with open(file_path, "rb") as f:
            subset = pickle.load(f)
        with open(file_path_prepare_dataset_model_config, "rb") as f:
            self.prepare_datasets[model_name]._model_config = pd.read_pickle(f)
        # avoiding cannot set WRITEABLE flag to True of this array error
        subset.X_train = subset.X_train.copy() if subset.X_train is not None else None
        subset.X_test = subset.X_test.copy() if subset.X_test is not None else None
        subset.y_train = subset.y_train.copy() if subset.y_train is not None else None
        subset.y_test = subset.y_test.copy() if subset.y_test is not None else None
        subset.exposure_train = subset.exposure_train.copy() if subset.exposure_train is not None else None
        subset.exposure_test = subset.exposure_test.copy() if subset.exposure_test is not None else None
        return subset

    def save_subset_to_pickle(self, model_name, subset: ModelDataSubset, rewrite: bool = False):
        file_path = os.path.join(self.results_path, model_name + '_v' + self.version + '_subset.pickle')
        file_path_prepare_dataset_model_config = os.path.join(self.results_path,
                                                 model_name + '_v' + self.version + '_prepare_model_config.pickle')
        if os.path.exists(file_path) and not rewrite:
            logger.warning(f'{model_name}||File {file_path} already exists.')
        else:
            logger.info(model_name + '||Saving subset to pickle')
            with open(file_path, "wb") as f:
                pickle.dump(subset, f)

            with open(file_path_prepare_dataset_model_config, "wb") as f:
                pickle.dump(self.prepare_datasets[model_name].get_model_config(), f)



class ParquetDataset:
    def __init__(self, config, parquet_name: str, prepare_engine:str='pandas'):
        self._parquet_name = parquet_name
        self.results_path = config.results_path
        self._prepare_engine=prepare_engine

    def save_parquet(self, data: pd.DataFrame | pl.DataFrame, rewrite: bool = True):
        file_path = os.path.join(self.results_path, self._parquet_name + '.parquet')
        if os.path.exists(file_path) and not rewrite:
            logger.warning(f'||File {file_path} already exists.')

        else:
            logger.info('||Saving dataset to parquet')
            if isinstance(data, pd.DataFrame):
                data.to_parquet(file_path)
            elif isinstance(data, pl.DataFrame):
                data.write_parquet(file_path)
            else:
                logger.error(f'||{type(data)} not supported')


    def read_parquet(self) -> pd.DataFrame|pl.DataFrame:
        file_path = os.path.join(self.results_path, self._parquet_name + '.parquet')
        if self._prepare_engine == 'pandas':
            return pd.read_parquet(file_path)
        elif self._prepare_engine == 'polars':
            return pl.read_parquet(file_path)


class PrepareEngine(ABC):
    def __init__(
            self, dataset, separation_config: SeparationModelConfig
    ):
        if not isinstance(separation_config, SeparationModelConfig):
            logger.error(f"PrepareEngine||separation_config must be SeparationModelConfig, get {type(separation_config)}")
            raise ValueError(f"PrepareEngine||separation_config must be SeparationModelConfig, get {type(separation_config)}")
        self.separation_config = separation_config
        self.dataset = dataset

    @abstractmethod
    def prepared_subset(self, *params):
        pass


class PandasInterface(PrepareEngine):

    def __init__(self, data: pd.DataFrame,
                 prepare_interface: PrepareDataset,
                 separation_config: SeparationModelConfig,
                 extra_columns: list = None):
        super().__init__(data, separation_config)
        self._prepare_interface = prepare_interface
        self.separation_config = separation_config
        self._extra_columns = extra_columns
        self._extra_columns_data = None

    def get_train_test_indexes(self):
        index_train, index_test = TrainTestIndexes(X=self.dataset,
                                                   separation_config=self.separation_config).train_test_indexes()
        return index_train, index_test

    def prepared_subset(self,  prepare_func: Callable = None,
                        args_dict: dict = None):

        index_train, index_test = self.get_train_test_indexes()
        model_config = self._prepare_interface.get_model_config()
        model_name = model_config.name
        X, y, target = self._filter_data_by_exposure(model_name=model_name, dataset=self.dataset)

        if prepare_func is not None:
            prepare_dataset_result = prepare_func(X, index_train, index_test, target,
                                                  **args_dict)
        else:
            prepare_dataset_result = self._prepare_interface.prepare_dataset(
                data=X,
                train_ind=index_train,
                test_ind=index_test,
                target=target
            )
        X = prepare_dataset_result.data
        self._prepare_interface._model_config = deepcopy(prepare_dataset_result.model_config)
        self._extra_columns_data = self.dataset[self._extra_columns] if self._extra_columns is not None else None
        data_subset = ModelDataSubset.load_subset(
            model_name=model_name,
            X=X,
            Y=y,
            index_train=index_train,
            index_test=index_test,
            features_numerical=prepare_dataset_result.features_numerical if model_config is not None else [],
            features_categorical=prepare_dataset_result.features_categorical if model_config is not None else [],
            column_exposure=model_config.column_exposure if model_config.column_exposure else None,
            column_target=model_config.column_target if model_config.column_target else None,
            extra_columns=self._extra_columns_data if self._extra_columns_data is not None else None)
        logger.debug('Model ' + model_name + ' || Data preparation finished')
        return data_subset


    def _filter_data_by_exposure(self, model_name: str, dataset: pd.DataFrame):
        exposure = {model_name: None}
        model_config = self._prepare_interface.get_model_config()
        target = pd.Series()
        if model_config.column_target:
            y = dataset[model_config.column_target]
            target = y
        else:
            target = pd.Series()
            y = pd.Series()
        if model_config.column_exposure:
            logger.info('Pandas Engine||Weighting target on exposure')
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


class PolarsInterface(PrepareEngine):
    def __init__(
            self,
            data: pl.DataFrame,
            prepare_interface: PrepareDatasetPl,
            separation_config: SeparationModelConfig,
            extra_columns: List[str] | None = None
    ):
        if not isinstance(data, pl.DataFrame):
            logger.error(f"PolarsEngine||data must be polars DataFrame, get {type(data)}")
            raise ValueError(f"PolarsEngine||data must be polars DataFrame, get {type(data)}")
        super().__init__(data, separation_config)
        if not isinstance(prepare_interface, PrepareDatasetPl):
            logger.error(f"PolarsEngine||prepare_interface must be PrepareDatasetPl, get {type(prepare_interface)}")
            raise ValueError(f"PolarsEngine||prepare_interface must be PrepareDatasetPl, get {type(prepare_interface)}")
        self._prepare_interface = prepare_interface
        if not isinstance(extra_columns, (list, type(None))):
            logger.error(f"PolarsEngine||extra_columns must be List[str] or None, get {type(extra_columns)}")
            raise ValueError(f"PolarsEngine||extra_columns must be List[str] or None, get {type(extra_columns)}")
        self._extra_columns_list = extra_columns

    def get_train_test_split(self) -> pl.DataFrame:

        return TrainTestIndexesPl(
            dataset=self.dataset, separation_config=self.separation_config
        ).train_test_split()

    def _filter_data_by_exposure(self, dataset: pl.DataFrame) -> (pl.DataFrame, pl.DataFrame | None):

        model_config = self._prepare_interface.get_model_config()

        if model_config.column_exposure:
            logger.info("Polars Engine||Weighting target on exposure")
            X = dataset.filter(pl.col(model_config.column_exposure) > 0)
            target = (
                X.select(pl.col(model_config.column_target) / pl.col(model_config.column_exposure), "is_train_obml")
                if model_config.column_target else None
            )

        else:
            logger.info("Polars Engine||Target without exposure")
            X = dataset
            target = (
                X.select(model_config.column_target, "is_train_obml")
                if model_config.column_target else None
            )

        return X, target

    def prepared_subset(
            self,  prepare_func: Callable = None, args_dict: dict = None
    ) -> ModelDataSubset:

        self.dataset = self.get_train_test_split()

        model_config = self._prepare_interface.get_model_config()
        model_name = model_config.name

        X, target = self._filter_data_by_exposure(self.dataset)

        if prepare_func is not None:
            prepare_dataset_result = prepare_func(X, target, **args_dict)
        else:
            prepare_dataset_result = self._prepare_interface.prepare_dataset(X, target)

        self._prepare_interface._model_config = deepcopy(prepare_dataset_result.model_config)

        data_subset = ModelDataSubset.load_subset_pl(
            model_name=model_name,
            data=prepare_dataset_result.data,
            features_numerical=prepare_dataset_result.features_numerical,
            features_categorical=prepare_dataset_result.features_categorical,
            column_exposure=model_config.column_exposure if model_config.column_exposure else None,
            column_target=model_config.column_target if model_config.column_target else None,
            extra_columns_list=self._extra_columns_list,
        )

        logger.debug('Model ' + model_name + ' || Data preparation finished')
        return data_subset
