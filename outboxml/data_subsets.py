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
from outboxml.core.pydantic_models import DataConfig, DataModelConfig, SeparationModelConfig, ModelConfig
from outboxml.extractors import Extractor


class ModelDataSubset:
    """Container for prepared datasets, targets, exposures, and feature metadata.
    
    Stores train/test splits, feature lists, optional full X matrix, exposure
    series, and any extra columns. Provides helpers to construct subsets from
    pandas or polars data, combine two subsets, and drop columns.
    
    :param model_name: Name of the model the subset belongs to.
    :type model_name: str
    :param X_train: Training feature matrix.
    :type X_train: pandas.DataFrame
    :param y_train: Training target series.
    :type y_train: pandas.Series
    :param X_test: Test feature matrix, if available.
    :type X_test: pandas.DataFrame, optional
    :param y_test: Test target series, if available.
    :type y_test: pandas.Series, optional
    :param features_numerical: List of numerical feature names.
    :type features_numerical: list[str], optional
    :param features_categorical: List of categorical feature names.
    :type features_categorical: list[str], optional
    :param X: Full feature matrix (unsplit), if stored.
    :type X: pandas.DataFrame, optional
    :param exposure_train: Training exposure (weights), if any.
    :type exposure_train: pandas.Series, optional
    :param exposure_test: Test exposure (weights), if any.
    :type exposure_test: pandas.Series, optional
    :param extra_columns: Optional DataFrame of extra columns to carry along.
    :type extra_columns: pandas.DataFrame, optional
    
    .. rubric:: Examples
    
    .. code-block:: python
    
        import pandas as pd
        from outboxml.core.prepared_datasets import PrepareDataset
        
        # Direct construction
        subset = ModelDataSubset(
            model_name="my_model",
            X_train=pd.DataFrame({"f1":[1,2]}),
            y_train=pd.Series([0.1, 0.2], name="target"),
            X_test=pd.DataFrame({"f1":[3]}),
            y_test=pd.Series([0.3], name="target"),
            features_numerical=["f1"],
            features_categorical=[]
        )
        
        # Combine two subsets (features concatenated, targets taken from left)
        combined = subset + subset
        
        # Drop columns in-place and keep feature lists in sync
        ModelDataSubset.drop_columns(combined, ["f1"])
    """

    def __init__(
            self,
            model_name: str,
            X_train: pd.DataFrame = pd.DataFrame(),
            y_train: pd.Series = pd.Series(),
            X_test: Optional[pd.DataFrame] = None,
            y_test: Optional[pd.Series] = None,
            features_numerical: Optional[List[str]] = [],
            features_categorical: Optional[List[str]] = [],
            X: Optional[pd.DataFrame] = None,
            exposure_train: Optional[pd.Series] = None,
            exposure_test: Optional[pd.Series] = None,
            sample_weight_train: Optional[pd.Series] = None,
            sample_weight_test: Optional[pd.Series] = None,
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
        self.sample_weight_train: Optional[pd.Series] = sample_weight_train
        self.sample_weight_test: Optional[pd.Series] = sample_weight_test
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
            column_weight: Optional[str] = None,
            column_target: Optional[str] = None,
            extra_columns: Optional[pd.DataFrame] = None,
    ):
        """Build a ModelDataSubset from pandas DataFrames and index splits.
        
        Uses explicit index_train/index_test to slice X and Y, optionally
        extracting target and exposure columns.
        
        :param model_name: Model name for the subset.
        :type model_name: str
        :param X: Full feature matrix.
        :type X: pandas.DataFrame
        :param Y: DataFrame containing target and optionally exposure columns.
        :type Y: pandas.DataFrame
        :param index_train: Index labels of training rows.
        :type index_train: pandas.Index
        :param index_test: Index labels of test rows.
        :type index_test: pandas.Index
        :param features_numerical: Numerical feature names.
        :type features_numerical: list[str], optional
        :param features_categorical: Categorical feature names.
        :type features_categorical: list[str], optional
        :param column_exposure: Exposure column name in Y, if any.
        :type column_exposure: str, optional
        :param column_target: Target column name in Y, if any.
        :type column_target: str, optional
        :param extra_columns: Optional DataFrame with extra columns to store.
        :type extra_columns: pandas.DataFrame, optional
        :return: Constructed ModelDataSubset.
        :rtype: ModelDataSubset
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            import pandas as pd
            
            X = pd.DataFrame({"f1":[1,2,3], "f2":[10,20,30]})
            Y = pd.DataFrame({"target":[0.1, 0.2, 0.3], "exposure":[1.0, 2.0, 1.0]})
            index_train = pd.Index([0,1])
            index_test = pd.Index([2])
            
            subset = ModelDataSubset.load_subset(
                model_name="my_model",
                X=X,
                Y=Y,
                index_train=index_train,
                index_test=index_test,
                features_numerical=["f1","f2"],
                features_categorical=[],
                column_exposure="exposure",
                column_target="target",
            )
        """
        X_train = X[X.index.isin(X.index.intersection(index_train))]
        Y_train = Y[Y.index.isin(Y.index.intersection(index_train))]

        exposure_train = Y[Y.index.isin(Y.index.intersection(index_train))][
            column_exposure] if column_exposure else None
        sample_weight_train = Y[Y.index.isin(Y.index.intersection(index_train))][
            column_weight] if column_weight else None

        X_test = X[X.index.isin(X.index.intersection(index_test))]
        Y_test = Y[Y.index.isin(Y.index.intersection(index_test))]

        if column_target is not None:
            Y_train = Y_train[column_target]
            Y_test = Y_test[column_target]
        exposure_test = Y[Y.index.isin(Y.index.intersection(index_test))][column_exposure] if column_exposure else None
        sample_weight_test = Y[Y.index.isin(Y.index.intersection(index_test))][column_weight] if column_weight else None

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
            sample_weight_train,
            sample_weight_test,
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
            column_weight: Optional[str] = None,
            column_target: Optional[str] = None,
            extra_columns_list: Optional[List[str]] = None,
    ):
        """Build a ModelDataSubset from a polars DataFrame with split flags.
        
        Expects a boolean/int column 'is_train_obml' to indicate split. Extracts
        target/exposure columns if provided and returns a pandas-based subset.
        
        :param model_name: Model name for the subset.
        :type model_name: str
        :param data: Polars DataFrame with an 'is_train_obml' column.
        :type data: polars.DataFrame
        :param features_numerical: Numerical feature names.
        :type features_numerical: list[str], optional
        :param features_categorical: Categorical feature names.
        :type features_categorical: list[str], optional
        :param column_exposure: Exposure column name, if any.
        :type column_exposure: str, optional
        :param column_target: Target column name, if any.
        :type column_target: str, optional
        :param extra_columns_list: Extra column names to include in extra_columns.
        :type extra_columns_list: list[str], optional
        :return: Constructed ModelDataSubset.
        :rtype: ModelDataSubset
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            import polars as pl
            
            data = pl.DataFrame({
                "is_train_obml":[1,0,1],
                "f1":[1,2,3],
                "f2":[10,20,30],
                "target":[0.1, 0.2, 0.3],
                "exposure":[1.0, 2.0, 1.0],
                "keep_me":[5,6,7],
            })
            
            subset = ModelDataSubset.load_subset_pl(
                model_name="my_model",
                data=data,
                features_numerical=["f1","f2"],
                features_categorical=[],
                column_exposure="exposure",
                column_target="target",
                extra_columns_list=["keep_me"],
            )
        """
        X = data.to_pandas()

        X_train = X.loc[X["is_train_obml"] == 1].drop(columns=["is_train_obml"])
        y_train = X.loc[X["is_train_obml"] == 1][column_target] if column_target else pd.Series()
        exposure_train = X.loc[X["is_train_obml"] == 1][column_exposure] if column_exposure  else None
        sample_weight_train = X.loc[X["is_train_obml"] == 1][column_weight] if column_weight else None

        X_test = X.loc[X["is_train_obml"] == 0].drop(columns=["is_train_obml"])
        y_test = X.loc[X["is_train_obml"] == 0][column_target] if column_target else pd.Series()
        exposure_test = X.loc[X["is_train_obml"] == 0][column_exposure] if column_exposure else None
        sample_weight_test = X.loc[X["is_train_obml"] == 0][column_weight] if column_weight else None

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
            sample_weight_train=sample_weight_train,
            sample_weight_test=sample_weight_test,
            extra_columns=extra_columns_data,
        )
    def __add__(self, other):
        """Combine two ModelDataSubset objects by concatenating aligned columns.
        
        - Concatenates X_train and X_test column-wise
        - Merges feature lists (unique union)
        - Concatenates X if both present
        - Concatenates extra_columns if both present
        
        :param other: Another ModelDataSubset to combine with this one.
        :type other: ModelDataSubset
        :return: New combined ModelDataSubset.
        :rtype: ModelDataSubset
        
        .. note::
            Target and exposure series are taken from ``self``.
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            left = ModelDataSubset(model_name="m", X_train=pd.DataFrame({"a":[1]}), y_train=pd.Series([0.1]))
            right = ModelDataSubset(model_name="m", X_train=pd.DataFrame({"b":[2]}), y_train=pd.Series([0.2]))
            both = left + right
            assert list(both.X_train.columns) == ["a","b"]
        """
        if not isinstance(other, ModelDataSubset):
            raise TypeError(f"Unsupported operand type(s) for +: 'ModelDataSubset' and '{type(other).__name__}'")

        # Объединяем датафреймы и серии
        new_X_train = pd.concat([self.X_train, other.X_train], axis=1)

        # Объединяем X_test (если оба существуют)
        if self.X_test is not None and other.X_test is not None and isinstance(other.X_test , pd.DataFrame):
            if not other.X_test.empty:
                new_X_test = pd.concat([self.X_test, other.X_test],  axis=1)
            else:
                new_X_test = self.X_test
        else:
            new_X_test = self.X_test

        # Объединяем списки features
        new_features_numerical = list(set(self.features_numerical + other.features_numerical))
        new_features_categorical = list(set(self.features_categorical + other.features_categorical))

        # Объединяем X (если оба существуют)
        if self.X is not None:
            new_X = pd.concat([self.X, other.X], axis=1)
        else:
            new_X = None

        # Объединяем extra_columns
        if self.extra_columns is not None and other.extra_columns is not None:
            new_extra_columns = pd.concat([self.extra_columns, other.extra_columns], ignore_index=True)
        elif self.extra_columns is not None:
            new_extra_columns = self.extra_columns.copy()
        elif other.extra_columns is not None:
            new_extra_columns = other.extra_columns.copy()
        else:
            new_extra_columns = None

        return ModelDataSubset(
            model_name=self.model_name,
            X_train=new_X_train,
            y_train=self.y_train,
            X_test=new_X_test,
            y_test=self.y_test,
            features_numerical=new_features_numerical,
            features_categorical=new_features_categorical,
            X=new_X,
            exposure_train=self.exposure_train,
            exposure_test=self.exposure_test,
            sample_weight_train=self.sample_weight_train,
            sample_weight_test=self.sample_weight_test,
            extra_columns=new_extra_columns
        )

    @staticmethod
    def drop_columns(data_subset, columns_to_drop: list):
        """Drop columns in-place from a ModelDataSubset and update feature lists.
        
        :param data_subset: Subset to modify.
        :type data_subset: ModelDataSubset
        :param columns_to_drop: Column names to remove.
        :type columns_to_drop: list[str]
        :return: The modified ModelDataSubset.
        :rtype: ModelDataSubset
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            ds = ModelDataSubset(model_name="m",
                                 X_train=pd.DataFrame({"a":[1], "b":[2]}),
                                 y_train=pd.Series([1.0]),
                                 features_numerical=["a","b"])
            ModelDataSubset.drop_columns(ds, ["b"])
			assert "b" not in ds.X_train.columns and "b" not in ds.features_numerical
        """
        data_subset.X_train = data_subset.X_train.drop(columns=columns_to_drop).copy()
        data_subset.X_test = data_subset.X_test.drop(
            columns=columns_to_drop).copy() if data_subset.X_test is not None else None
        data_subset.X = data_subset.X.drop(columns=columns_to_drop).copy()
        for feature in columns_to_drop:
            if feature in data_subset.features_numerical:
                data_subset.features_numerical.remove(feature)
            elif feature in data_subset.features_categorical:
                data_subset.features_categorical.remove(feature)
        return data_subset


class DataPreprocessor:
    """High-level data preprocessor that prepares and persists model subsets.
    
    Handles:
    - Train/test splitting via TrainTestIndexes or TrainTestIndexesPl
    - Delegating feature preparation to PrepareDataset / PrepareDatasetPl
    - Optional exposure weighting of the target
    - Persisting intermediate datasets to Parquet and subsets to pickle
    - Loading subsets for multiple models
    
    :param prepare_dataset_interface_dict: Mapping of model_name to preparation
        interface (pandas or polars).
    :type prepare_dataset_interface_dict: dict[str, PrepareDataset | PrepareDatasetPl]
    :param dataset: Raw dataset or an Extractor implementation.
    :type dataset: pandas.DataFrame or Extractor
    :param data_config: Data configuration with separation settings and extras.
    :type data_config: DataModelConfig
    :param version: Version string used in filenames. Defaults to '1'.
    :type version: str, optional
    :param prepare_engine: Engine to use ('pandas' or 'polars'). Defaults to 'pandas'.
    :type prepare_engine: Literal['pandas', 'polars'], optional
    :param external_config: External config with results_path. Defaults to None.
    :type external_config: object, optional
    :param use_saved_files: If True, reuse existing pickled subsets. Defaults to False.
    :type use_saved_files: bool, optional
    :param retro: If True, keep full columns when saving dataset for retro analysis.
        Defaults to False.
    :type retro: bool, optional
    
    :var model_names: List of model names to process.
    :var index_train: Cached training index after preparation.
    :var index_test: Cached test index after preparation.
    
    .. rubric:: Examples
    
    .. code-block:: python
    
        # Assume you have:
        # - data_config: DataModelConfig
        # - df: pandas.DataFrame with all required columns
        # - pd_interface: an instance implementing PrepareDataset for pandas
        #
        # Basic pandas flow
        p = DataPreprocessor(
            prepare_dataset_interface_dict={"my_model": pd_interface},
            dataset=df,
            data_config=data_config,
            version="1",
            prepare_engine="pandas",
            use_saved_files=False,
        )
        subset = p.get_subset("my_model")  # prepares, saves, and loads from pickle
        
        # Access all subsets for all configured models
        all_subsets = p.data_subsets()
        
        # Polars flow (if you have a polars PrepareDatasetPl)
        # polars_interface: PrepareDatasetPl
        p_pl = DataPreprocessor(
            prepare_dataset_interface_dict={"my_model_pl": polars_interface},
            dataset=pl.DataFrame(df),  # or a real polars DataFrame
            data_config=data_config,
            version="1",
            prepare_engine="polars",
        )
        subset_pl = p_pl.get_subset("my_model_pl")
    """

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
                                                version=self._version)
        self._parquet_dataset = ParquetDataset(config=self.config,
                                               parquet_name='temp_dataset_v' + self._version,
                                               prepare_engine=prepare_engine,
                                               )
        self._model_config_pickle = ModelConfigPickle(config=self.config,
                                                      version=self._version,
                                                      )

        self.temp_subset: Optional[ModelDataSubset] = None
        self._data_columns = []
        self._retro = retro
        self.index_train = pd.Index([])
        self.index_test = pd.Index([])

    @property
    def dataset(self)->pd.DataFrame:
        """Return the dataset, persisting to Parquet and reloading on first access.
        
        - If a pandas DataFrame or Extractor is supplied, it is saved to Parquet
          (optionally reduced to only used columns if retro=False), then replaced
          with the Parquet-backed version to reduce memory.
        - Subsequent calls read from the Parquet file.
        
        :return: Dataset from Parquet in the selected engine format.
        :rtype: pandas.DataFrame
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            dp = DataPreprocessor(prepare_dataset_interface_dict={"m": pd_interface},
                                  dataset=df,
                                  data_config=data_config)
            # First access triggers save to parquet and then read
            data_for_preparation = dp.dataset
        """
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


    def model_config(self, model_name):
        """Get the ModelConfig for a given model name.
        
        :param model_name: Model name.
        :type model_name: str
        :return: ModelConfig object.
        :rtype: ModelConfig
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            mc = dp.model_config("my_model")
            print(mc.name, mc.column_target)
        """
        return self._prepare_datasets[model_name].get_model_config()

    def save_subset_to_pickle(self, model_name: str, data_subset: ModelDataSubset, rewrite: bool = False):
        """Save a prepared subset and its model config to pickle files.
        
        :param model_name: Model name.
        :type model_name: str
        :param data_subset: Prepared data subset to persist.
        :type data_subset: ModelDataSubset
        :param rewrite: Overwrite existing files if True. Defaults to False.
        :type rewrite: bool, optional
        :return: None
        :rtype: None
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            subset = dp.get_subset("my_model", from_pickle=False)  # prepare in-memory
            dp.save_subset_to_pickle("my_model", subset, rewrite=True)
        """
        self._model_config_pickle.save_config_to_pickle(model_name,self.model_config(model_name), rewrite)
        self._pickle_subset.save_subset_to_pickle(model_name, data_subset, rewrite)

    def get_subset(self, model_name: str = None, from_pickle: bool = True, prepare_func: Callable = None,
                   args: dict = None) -> ModelDataSubset:
        """Get a prepared subset for a model, loading from pickle or preparing anew.
        
        :param model_name: Model name. Defaults to the first in model_names.
        :type model_name: str, optional
        :param from_pickle: If True, load from pickle or prepare-then-load.
        :type from_pickle: bool
        :param prepare_func: Optional custom preparation function to override default.
        :type prepare_func: Callable, optional
        :param args: Keyword arguments to pass to prepare_func.
        :type args: dict, optional
        :return: Prepared ModelDataSubset.
        :rtype: ModelDataSubset
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            # Default path (uses interface.prepare_dataset and pickle cache)
            subset = dp.get_subset("my_model", from_pickle=True)
            
            # Custom preparation function
            def my_prepare(X, index_train, index_test, target, scale=1.0):
                # Return an object compatible with PrepareDatasetResult
                # with attributes: data, model_config, features_numerical, features_categorical
                ...
            subset2 = dp.get_subset("my_model", from_pickle=False, prepare_func=my_prepare, args={"scale": 2.0})
        """
        if model_name is None: model_name = self.model_names[0]
        if from_pickle:
            if not self._check_prepared_subset(model_name):
                self._prepare_subset(model_name, True, prepare_func, args)
            self._prepare_datasets[model_name].load_model_config(self._model_config_pickle.load_config_from_pickle(model_name))
            return self._pickle_subset.load_subsets_from_pickle(model_name)
        else:
            self._prepare_subset(model_name, to_pickle=False)
            return self.temp_subset

    def data_subsets(self, ) -> Dict[str, ModelDataSubset]:
        """Return prepared subsets for all configured models.
        
        Ensures subsets exist (preparing if necessary) and returns a mapping
        of model_name to ModelDataSubset loaded from pickle.
        
        :return: Dictionary of model subsets.
        :rtype: dict[str, ModelDataSubset]
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            for name, subset in dp.data_subsets().items():
                print(name, subset.X_train.shape)
        """
        data_subsets = {}
        for model_name in self._prepare_datasets.keys():
            if not self._check_prepared_subset(model_name):
                self._prepare_subset(model_name=model_name)

        for model_name in self._prepare_datasets.keys():
            data_subsets[model_name] = self._pickle_subset.load_subsets_from_pickle(model_name)

        return data_subsets

    def _prepare_subset(self, model_name, to_pickle: bool = True, prepare_func: Callable = None,
                        args_dict: dict = None):
        """Prepare a subset for a single model using the selected engine.
        
        - Performs train/test split
        - Applies exposure-based filtering/weighting if configured
        - Delegates feature preparation to the model-specific interface
        - Saves to pickle if requested
        
        :param model_name: Model name to prepare.
        :type model_name: str
        :param to_pickle: Save subset and config to pickle if True.
        :type to_pickle: bool
        :param prepare_func: Optional custom preparation function.
        :type prepare_func: Callable, optional
        :param args_dict: Arguments for prepare_func or interface method.
        :type args_dict: dict, optional
        :return: None
        :rtype: None
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            # Usually called internally. To force (re-)prepare one model:
            dp._prepare_subset("my_model", to_pickle=True)
        """
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
            prepare_engine = PolarsInterface(data=data,
                                             prepare_interface=self._prepare_datasets[model_name],
                                             separation_config=self._data_config.separation,
                                             extra_columns=self._extra_columns)
            data_subset = prepare_engine.prepared_subset(prepare_func, args_dict)
            self.index_train, self.index_test = data_subset.X_train.index, data_subset.X_test.index

        else:
            raise f'Unknow engine for data preparation'

        if to_pickle:
            self._pickle_subset.save_subset_to_pickle(model_name, data_subset, True)
            self._model_config_pickle.save_config_to_pickle(model_name, self.model_config(model_name), True)
            self._prepared_subsets[model_name] = True
        else:
            self.temp_subset = data_subset


    def _check_prepared_subset(self, model_name):
        """Check whether a prepared subset pickle exists and can be reused.
        
        Honors the use_saved_files flag. If True and file exists, marks the model
        as prepared; otherwise uses internal cache to determine status.
        
        :param model_name: Model name.
        :type model_name: str
        :return: True if the subset is already prepared and should be used.
        :rtype: bool
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            exists_and_use = dp._check_prepared_subset("my_model")
            if not exists_and_use:
                dp._prepare_subset("my_model")
        """
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
        """Collect the union of all required columns across model configs.
        
        Aggregates feature names, target, exposure, and relative feature inputs
        across all model configurations, plus any extra_columns specified in
        the data configuration. Used to trim the dataset before Parquet save
        when retro=False.
        
        :return: None
        :rtype: None
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            # Usually called internally when saving to parquet; can be invoked manually:
            dp._collect_features_list()
            print(dp._data_columns[:5])
        """
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
                if model.column_weight is not None:
                    model_features[model.name].append(model.column_weight)

                if model.column_target is not None:
                    model_features[model.name].append(model.column_target)
                relative_features = model.relative_features.copy()
                if model.relative_features is not None:
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
    """Utility for persisting and loading ModelDataSubset objects to/from pickle.
    
    :param config: External configuration (must have results_path).
    :type config: object
    :param version: Version string used in filenames.
    :type version: str
    
    .. rubric:: Examples
    
    .. code-block:: python
    
        p = PickleModelSubset(config=config, version="1")
        p.save_subset_to_pickle("my_model", subset, rewrite=True)
        subset2 = p.load_subsets_from_pickle("my_model")
    """

    def __init__(self, config, version):
        self.results_path = config.results_path
        self.version = version

    def load_subsets_from_pickle(self, model_name: str, version: str = '1') -> ModelDataSubset:
        """Load a ModelDataSubset from a pickle file.
        
        :param model_name: Model name.
        :type model_name: str
        :param version: Deprecated/unused parameter (kept for compatibility).
        :type version: str, optional
        :return: Loaded ModelDataSubset with arrays copied to avoid WRITEABLE flag issues.
        :rtype: ModelDataSubset
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            p = PickleModelSubset(config=config, version="1")
            subset = p.load_subsets_from_pickle("my_model")
        """
        file_path = os.path.join(self.results_path, model_name + '_v' + self.version + '_subset.pickle')
        logger.info(model_name + '_v' + self.version + '||Loading subset from pickle')
        with open(file_path, "rb") as f:
            subset = pickle.load(f)

        # avoiding cannot set WRITEABLE flag to True of this array error
        subset.X_train = subset.X_train.copy() if subset.X_train is not None else None
        subset.X_test = subset.X_test.copy() if subset.X_test is not None else None
        subset.y_train = subset.y_train.copy() if subset.y_train is not None else None
        subset.y_test = subset.y_test.copy() if subset.y_test is not None else None
        subset.exposure_train = subset.exposure_train.copy() if subset.exposure_train is not None else None
        subset.exposure_test = subset.exposure_test.copy() if subset.exposure_test is not None else None
        subset.sample_weight_train = subset.sample_weight_train.copy() if subset.sample_weight_train is not None else None
        subset.sample_weight_test = subset.sample_weight_test.copy() if subset.sample_weight_test is not None else None
        return subset

    def save_subset_to_pickle(self, model_name, subset: ModelDataSubset, rewrite: bool = False):
        """Save a ModelDataSubset to a pickle file.
        
        :param model_name: Model name.
        :type model_name: str
        :param subset: Subset to save.
        :type subset: ModelDataSubset
        :param rewrite: Overwrite existing file if True. Defaults to False.
        :type rewrite: bool, optional
        :return: None
        :rtype: None
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            p = PickleModelSubset(config=config, version="1")
            p.save_subset_to_pickle("my_model", subset, rewrite=True)
        """
        file_path = os.path.join(self.results_path, model_name + '_v' + self.version + '_subset.pickle')

        if os.path.exists(file_path) and not rewrite:
            logger.warning(f'{model_name}||File {file_path} already exists.')
        else:
            logger.info(model_name + '_v' + self.version  + '||Saving subset to pickle')
            with open(file_path, "wb") as f:
                pickle.dump(subset, f)


class ParquetDataset:
    """Helper to persist datasets to a Parquet file and read them back.
    
    :param config: External config with results_path.
    :type config: object
    :param parquet_name: Base file name for Parquet dataset.
    :type parquet_name: str
    :param prepare_engine: Engine hint ('pandas' or 'polars') for reading.
    :type prepare_engine: str
    
    .. rubric:: Examples
    
    .. code-block:: python
    
        pq = ParquetDataset(config=config, parquet_name="temp_dataset_v1", prepare_engine="pandas")
        pq.save_parquet(pd.DataFrame({"a":[1,2]}), rewrite=True)
        df = pq.read_parquet()
    """
    def __init__(self, config, parquet_name: str, prepare_engine:str='pandas'):
        self._parquet_name = parquet_name
        self.results_path = config.results_path
        self._prepare_engine=prepare_engine

    def save_parquet(self, data: pd.DataFrame | pl.DataFrame, rewrite: bool = True):
        """Save a DataFrame to Parquet under results_path.
        
        :param data: Data to save (pandas or polars DataFrame).
        :type data: pandas.DataFrame or polars.DataFrame
        :param rewrite: Overwrite existing file if True. Defaults to True.
        :type rewrite: bool, optional
        :return: None
        :rtype: None
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            pq = ParquetDataset(config=config, parquet_name="tmp_v1", prepare_engine="polars")
            pq.save_parquet(pl.DataFrame({"a":[1,2]}), rewrite=True)
        """
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


    def read_parquet(self, to_polars=False) -> pd.DataFrame|pl.DataFrame:
        """Read the persisted Parquet dataset using the configured engine.
        
        :param to_polars: Deprecated parameter; reading engine is determined by init.
        :type to_polars: bool
        :return: Loaded dataset.
        :rtype: pandas.DataFrame or polars.DataFrame
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            pq = ParquetDataset(config=config, parquet_name="tmp_v1", prepare_engine="pandas")
            df = pq.read_parquet()
        """
        file_path = os.path.join(self.results_path, self._parquet_name + '.parquet')
        if self._prepare_engine == 'pandas':
            return pd.read_parquet(file_path)
        elif self._prepare_engine == 'polars':
            return pl.read_parquet(file_path)


class ModelConfigPickle:
    """Persist and load ModelConfig objects to/from pickle files.
    
    :param config: External config with results_path.
    :type config: object
    :param version: Version string used in filenames.
    :type version: str
    
    .. rubric:: Examples
    
    .. code-block:: python
    
        p = ModelConfigPickle(config=config, version="1")
        p.save_config_to_pickle("my_model", model_config, rewrite=True)
        cfg = p.load_config_from_pickle("my_model")
    """
    def __init__(self, config, version):
        self.results_path = config.results_path
        self.version = version


    def load_config_from_pickle(self, model_name: str) -> ModelConfig:
        """Load a ModelConfig from a pickle file.
        
        :param model_name: Model name.
        :type model_name: str
        :return: Loaded ModelConfig.
        :rtype: ModelConfig
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            p = ModelConfigPickle(config=config, version="1")
            cfg = p.load_config_from_pickle("my_model")
        """
        file_path = os.path.join(self.results_path, model_name + '_v' + self.version + '_model_config.pickle')
        logger.info(model_name + '_v' + self.version + '_subset.pickle' + '||Loading model config from pickle')
        with open(file_path, "rb") as f:
            config = pickle.load(f)
        return config


    def save_config_to_pickle(self, model_name, model_config: ModelConfig, rewrite: bool = False):
        """Save a ModelConfig to a pickle file.
        
        :param model_name: Model name.
        :type model_name: str
        :param model_config: Model configuration to persist.
        :type model_config: ModelConfig
        :param rewrite: Overwrite existing file if True. Defaults to False.
        :type rewrite: bool, optional
        :return: None
        :rtype: None
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            p = ModelConfigPickle(config=config, version="1")
            p.save_config_to_pickle("my_model", model_config, rewrite=True)
        """
        file_path = os.path.join(self.results_path, model_name + '_v' + self.version + '_model_config.pickle')

        if os.path.exists(file_path) and not rewrite:
            logger.warning(f'model config {model_name}||File {file_path} already exists.')
        else:
            logger.info(model_name + '_v' + self.version + '_prepare_model_config.pickle' + '||Saving pickle')
            with open(file_path, "wb") as f:
                pickle.dump(model_config, f)


class PrepareEngine(ABC):
    """Abstract base class for dataset preparation engines (pandas/polars).
    
    Validates separation_config type and stores the dataset reference.
    
    :param dataset: Input dataset (pandas or polars).
:type

dataset: Any
    :param separation_config: Train/test split configuration.
    :type separation_config: SeparationModelConfig
    
    .. rubric:: Examples
    
    .. code-block:: python
    
        # Subclassing pattern (simplified)
        class MyEngine(PrepareEngine):
            def prepared_subset(self, *params):
                # return a ModelDataSubset
                ...
    """
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
        """Prepare and return a ModelDataSubset for a single model.
        
        Specific signature depends on concrete engine implementation.
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            # Implemented by PandasInterface / PolarsInterface
            # engine.prepared_subset(...)
            ...
        """
        pass


class PandasInterface(PrepareEngine):
    """Pandas-based preparation engine wrapping a PrepareDataset interface.
    
    Performs:
    - Train/test split computation
    - Optional exposure-based filtering/weighting
    - Model-specific feature preparation via PrepareDataset
    - Packaging into a ModelDataSubset
    
    :param data: Input dataset (pandas DataFrame).
    :type data: pandas.DataFrame
    :param prepare_interface: Model-specific prepare interface.
    :type prepare_interface: PrepareDataset
    :param separation_config: Train/test split configuration.
    :type separation_config: SeparationModelConfig
    :param extra_columns: Extra column names to keep alongside the subset.
    :type extra_columns: list[str], optional
    
    .. rubric:: Examples
    
    .. code-block:: python
    
        # Assume pd_interface implements PrepareDataset and provides get_model_config/prepare_dataset
        engine = PandasInterface(
            data=df,
            prepare_interface=pd_interface,
            separation_config=data_config.separation,
            extra_columns=["id"]
        )
        index_train, index_test = engine.get_train_test_indexes()
        subset = engine.prepared_subset()
    """

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
        """Compute train/test indexes using TrainTestIndexes.
        
        :return: Tuple of (index_train, index_test).
        :rtype: tuple[pandas.Index, pandas.Index]
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            index_train, index_test = engine.get_train_test_indexes()
        """
        index_train, index_test = TrainTestIndexes(X=self.dataset,
                                                   separation_config=self.separation_config).train_test_indexes()
        return index_train, index_test

    def prepared_subset(self,  prepare_func: Callable = None,
                        args_dict: dict = None):
        """Prepare a subset using pandas path and return ModelDataSubset.
        
        :param prepare_func: Optional custom function to prepare the dataset.
		Signature should be (X, index_train, index_test, target, **kwargs).
        :type prepare_func: Callable, optional
        :param args_dict: Keyword arguments for prepare_func or interface method.
        :type args_dict: dict, optional
        :return: Prepared ModelDataSubset.
        :rtype: ModelDataSubset
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            # Default path using interface
            subset = engine.prepared_subset()
            
            # With a custom function
            def my_prepare(X, index_train, index_test, target, foo=1):
                ...
            subset2 = engine.prepared_subset(prepare_func=my_prepare, args_dict={"foo": 2})
        """
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
            column_weight=model_config.column_weight if model_config.column_weight else None,
            column_target=model_config.column_target if model_config.column_target else None,
            extra_columns=self._extra_columns_data if self._extra_columns_data is not None else None)
        logger.debug('Model ' + model_name + ' || Data preparation finished')
        return data_subset


    def _filter_data_by_exposure(self, model_name: str, dataset: pd.DataFrame):
        """Filter rows by positive exposure and compute exposure-weighted target when needed.
        
        - If column_exposure is set, keeps rows with exposure > 0 and creates a DataFrame
          y with target and exposure columns; target series is target/exposure.
        - Otherwise, returns the original dataset and an empty y DataFrame.
        
        :param model_name: Model name for logging.
        :type model_name: str
        :param dataset: Input dataset.
        :type dataset: pandas.DataFrame
        :return: Tuple of (X, y, target_series_or_empty).
        :rtype: tuple[pandas.DataFrame, pandas.DataFrame, pandas.Series]
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            X, y, target = engine._filter_data_by_exposure("my_model", df)
        """
        exposure = {model_name: None}
        model_config = self._prepare_interface.get_model_config()
        target = pd.Series()
        if model_config.column_target:
            y = dataset[model_config.column_target]
            target = y
        else:
            target = pd.Series()
            y = pd.Series(index=dataset.index)
        sample_weight = dataset[model_config.column_weight] if model_config.column_weight else None
        if model_config.column_exposure:
            logger.info('Pandas Engine||Weighting target on exposure')
            exposure[model_name] = dataset[model_config.column_exposure]
            X = dataset.loc[exposure[model_name] > 0]
            y = y.loc[y.index.isin(X.index)]
            target = y / exposure[model_name]
            y_parts = [
                pd.Series(y, name=model_config.column_target),
                pd.Series(
                    exposure[model_name].loc[exposure[model_name].index.isin(X.index)],
                    name=model_config.column_exposure,
                ),
            ]
            if sample_weight is not None:
                y_parts.append(
                    pd.Series(
                        sample_weight.loc[sample_weight.index.isin(X.index)],
                        name=model_config.column_weight,
                    )
                )
            y = pd.concat(y_parts, axis=1)

        else:
            X = dataset
            y = pd.DataFrame(y)
            if sample_weight is not None:
                y[model_config.column_weight] = sample_weight
        return X, y, target


class PolarsInterface(PrepareEngine):
    """Polars-based preparation engine wrapping a PrepareDatasetPl interface.
    
    Performs:
    - Train/test split generation with a split flag
    - Optional exposure-based filtering/weighting
    - Model-specific preparation via PrepareDatasetPl
    - Packaging into a ModelDataSubset
    
    :param data: Input dataset (polars DataFrame).
    :type data: polars.DataFrame
    :param prepare_interface: Model-specific prepare interface (polars).
    :type prepare_interface: PrepareDatasetPl
    :param separation_config: Train/test split configuration.
    :type separation_config: SeparationModelConfig
    :param extra_columns: Extra columns to include in the final subset.
    :type extra_columns: list[str] or None
    
    .. rubric:: Examples
    
    .. code-block:: python
    
        engine = PolarsInterface(
            data=pl.DataFrame(df),  # or a real polars DataFrame
            prepare_interface=polars_interface,
            separation_config=data_config.separation,
            extra_columns=["id"]
        )
        split_df = engine.get_train_test_split()
        subset = engine.prepared_subset()
    """
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
        """Create and return a DataFrame with a train/test split flag.
        
        :return: Polars DataFrame augmented with 'is_train_obml' column.
        :rtype: polars.DataFrame
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            split_df = engine.get_train_test_split()
            assert "is_train_obml" in split_df.columns
        """
        return TrainTestIndexesPl(
            dataset=self.dataset, separation_config=self.separation_config
        ).train_test_split()

    def _filter_data_by_exposure(self, dataset: pl.DataFrame) -> (pl.DataFrame, pl.DataFrame | None):
        """Filter/weight by exposure if configured and return X and target columns.
        
        :param dataset: Input polars DataFrame (with split flag).
        :type dataset: polars.DataFrame
        :return: Tuple of (X, target_df_or_none) with 'is_train_obml' retained.
        :rtype: tuple[polars.DataFrame, polars.DataFrame or None]
        
        .. rubric:: Examples
        
		.. code-block:: python
        
            X, target = engine._filter_data_by_exposure(split_df)
        """
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
        """Prepare a subset using polars path and return ModelDataSubset.
        
        :param prepare_func: Optional custom function to prepare the dataset.
            Signature should be (X_pl_df, target_pl_df_or_none, **kwargs).
        :type prepare_func: Callable, optional
        :param args_dict: Keyword arguments passed to prepare_func or interface method.
        :type args_dict: dict, optional
        :return: Prepared ModelDataSubset.
        :rtype: ModelDataSubset
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            # Default path using interface
            subset = engine.prepared_subset()
            
            # With a custom function
            def my_prepare(X_pl, target_pl, factor=1.0):
                ...
            subset2 = engine.prepared_subset(prepare_func=my_prepare, args_dict={"factor": 2.0})
        """
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
            column_weight=model_config.column_weight if model_config.column_weight else None,
            column_target=model_config.column_target if model_config.column_target else None,
            extra_columns_list=self._extra_columns_list,
        )

        logger.debug('Model ' + model_name + ' || Data preparation finished')
        return data_subset