from dataclasses import dataclass

import pandas as pd
from loguru import logger

from outboxml.core.data_prepare import prepare_dataset
from outboxml.core.pydantic_models import ModelConfig, MonitoringConfig
from outboxml.data_subsets import DataPreprocessor



class MonitoringResult:
    def __init__(self, group_name):
        self.group_name = group_name
        self.model_version = 'default'
        self.dataset_name = 'default'
        self.reviews = {}
        self.metric = None
        self.extrapolation_results = {}
        self.reports = {}
        self.grafana_dashboard = None

@dataclass
class DataContext:
    X_train: pd.DataFrame = None
    X_test: pd.DataFrame = None

    base: pd.DataFrame = None
    actual: pd.DataFrame  = None

@dataclass
class MonitoringContext:
    data_preprocessor: DataPreprocessor

    monitoring_result: MonitoringResult
    monitoring_config: MonitoringConfig
    models_config: ModelConfig

    actual: pd.DataFrame

    def get_prepared_data(self) -> DataContext:
        try:
            if not self.models_config:
                raise ValueError("Model config is required for prepared data")

            subset = self.data_preprocessor.get_subset(model_name=self.models_config.name)

            prepared = prepare_dataset(
                group_name=self.monitoring_result.group_name,
                data=self.actual.copy(),
                train_ind=self.actual.index,
                test_ind=pd.Index([]),
                model_config=self.models_config,
            )

            return DataContext(
                base=self.data_preprocessor.dataset,
                actual=self.actual.copy(),
                X_train=subset.X_train,
                X_test=prepared.data
            )

        except Exception as e:
            logger.exception("Failed to prepare data in MonitoringContext")
            raise e

    def get_raw_data(self):
        return DataContext(
            base=self.data_preprocessor.dataset,
            actual=self.actual.copy()
        )
