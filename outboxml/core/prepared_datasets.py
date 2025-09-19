from abc import ABC, abstractmethod
from copy import deepcopy

from loguru import logger
from sklearn.model_selection import train_test_split

from outboxml.core.config_builders import FeatureBuilder
from outboxml.core.data_prepare import prepare_dataset
import pandas as pd
from typing import Optional, Callable

from outboxml.core.enums import SeparationParams, FeatureTypesForSelection
from outboxml.core.pydantic_models import ModelConfig, SeparationModelConfig, FeatureModelConfig, DataConfig
from outboxml.core.data_prepare import PrepareDatasetResult
from outboxml.extractors import Extractor


class BasePrepareDataset(ABC):
    @abstractmethod
    def prepare_dataset(self, **params) -> PrepareDatasetResult:
        pass




class PrepareDataset(BasePrepareDataset):
    def __init__(
            self,
            check_prepared: bool = True,
            calc_corr: bool = False,
            save_data: bool = False,
            corr_threshold: Optional[float] = None,
            model_config: Optional[ModelConfig] = None,
            data_pred_prep_func: Optional[Callable] = None,
            data_post_prep_func: Optional[Callable] = None,
            group_name: str = 'general',
    ):
        self._check_prepared: bool = check_prepared
        self._calc_corr: bool = calc_corr
        self._save_data: bool = save_data
        self._corr_threshold: Optional[float] = corr_threshold
        self._model_config: Optional[ModelConfig] = model_config
        self._data_pred_prep_func: Optional[Callable] = data_pred_prep_func
        self._data_post_prep_func: Optional[Callable] = data_post_prep_func
        self.group_name = group_name

    def prepare_dataset(
            self, data: pd.DataFrame, train_ind: pd.Index, test_ind: pd.Index, target: pd.Series = None
    ) -> PrepareDatasetResult:
        if self._model_config.data_filter_condition is not None:
            data = data.query(self._model_config.data_filter_condition)

        if self._data_pred_prep_func is not None:
            logger.info('User pred prep function')
            try:
                data = self._data_pred_prep_func(data)
                logger.info('User pred prep completed')
            except:
                print('data pred_prep Error')

        prepare_dataset_result = self._prepare_dataset(data=data, train_ind=train_ind, test_ind=test_ind, target=target)

        if self._data_post_prep_func is not None:
            logger.info('User post prep function')
            try:
                prepare_dataset_result.data = self._data_post_prep_func(prepare_dataset_result.data)
                logger.info('User post prep completed')
            except:
                print('data post_prep Error')

        return prepare_dataset_result

    def load_model_config(self, model_config):
        self._model_config = model_config

    def get_model_config(self):
        try:
            return self._model_config
        except TypeError:
            return None

    def _prepare_dataset(
            self, data: pd.DataFrame, train_ind: pd.Index, test_ind: pd.Index, target: pd.Series = None
    ) -> PrepareDatasetResult:
        prepare_dataset_result = prepare_dataset(
            group_name=self.group_name,
            data=data,
            train_ind=train_ind,
            test_ind=test_ind,
            model_config=self._model_config,
            check_prepared=self._check_prepared,
            calc_corr=self._calc_corr,
            save_data=self._save_data,
            corr_threshold=self._corr_threshold,
            target=target
        )

        return prepare_dataset_result


class FeatureSelectionPrepareDataset(BasePrepareDataset):
    def __init__(self,
                 model_config: ModelConfig,
                 ):
        self.model_config = model_config
        self._new_model_config = None
    def prepare_dataset(self, data: pd.DataFrame,  index_train: pd.Index, index_test:pd.Index,
                        target: pd.Series =None, new_features: dict = None, features_params: dict=None) -> PrepareDatasetResult:
        self._new_model_config = deepcopy(self.model_config)
        if new_features is not None:
            for feature_num in new_features[FeatureTypesForSelection.numeric]:
                self._new_model_config.features.append(self._numerical_feature(feature_num, data[feature_num], features_params[feature_num])
                                    )
            for feature_cat in new_features[FeatureTypesForSelection.categorical]:
                self._new_model_config.features.append(self._categorical_feature(feature_cat, data[feature_cat],features_params[feature_cat]))

        return prepare_dataset(
            group_name='feature_selection',
            data=data,
            train_ind=index_train,
            test_ind=index_test,
            model_config=self._new_model_config,
            check_prepared=False,
            calc_corr=False,
            save_data=False,
            corr_threshold=False,
            target=target
        )

    def _numerical_feature(self, name, feature_values, feature_params) -> FeatureModelConfig:

        return FeatureBuilder(name=name, feature_values=feature_values,
                              type='numerical', **feature_params).build()

    def _categorical_feature(self, name, feature_values, feature_params) -> FeatureModelConfig:

        return FeatureBuilder(name=name, feature_values=feature_values,
                              type='categorical', **feature_params).build()


class TrainTestIndexes:

    def __init__(self, X: pd.DataFrame, separation_config: SeparationModelConfig):
        self.X: pd.DataFrame = X
        self._separation_config = separation_config

    def train_test_indexes(self):

        if self._separation_config.kind == SeparationParams.rand:
            logger.info("Random separation")
            train_ind, test_ind = RandomSeparation(self._separation_config).train_test_indexes(self.X)

        elif self._separation_config.kind == SeparationParams.period:
            logger.info("Date separation")
            train_ind, test_ind = DateSeparation(self._separation_config).train_test_indexes(self.X)

        elif self._separation_config.kind == SeparationParams.null:
            logger.info("Null separation")
            train_ind = self.X.index
            test_ind = self.X[~self.X.index.isin(train_ind)].index

        else:
            logger.info("No separation")
            train_ind = self.X.index
            test_ind = self.X[~self.X.index.isin(train_ind)].index

        logger.info(f"Train: {str(len(train_ind))}, test: {str(len(test_ind))}")

        return train_ind, test_ind


class BaseSeparation(ABC):
    @abstractmethod
    def train_test_indexes(self, *params):
        pass


class RandomSeparation(BaseSeparation):

    def __init__(self, separation_config: SeparationModelConfig):
        self._separation_config = separation_config

    def train_test_indexes(self, X: pd.DataFrame):
        train, test = train_test_split(
            X,
            test_size=self._separation_config.test_train_proportion,
            random_state=self._separation_config.random_state
        )
        train_ind = train.index
        test_ind = test.index

        return train_ind, test_ind


class DateSeparation(BaseSeparation):

    def __init__(self, separation_config: SeparationModelConfig):
        self._separation_config = separation_config

    def train_test_indexes(self, X: pd.DataFrame):
        column_period = self._separation_config.period_column[0]
        train_ind = X.loc[X[column_period].between(*self._separation_config.train_period)].index
        test_ind = X.loc[X[column_period].between(*self._separation_config.test_period)].index

        return train_ind, test_ind


class UserSeparation(BaseSeparation):
    def train_test_indexes(self):
        pass



# dataset
# targets_columns_names
# self.X
# self,Y
# self.extra_columns
# self.index_train_