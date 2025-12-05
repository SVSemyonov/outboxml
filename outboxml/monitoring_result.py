from dataclasses import dataclass

import pandas as pd
from loguru import logger

from outboxml.core.data_prepare import prepare_dataset
from outboxml.core.pydantic_models import ModelConfig, MonitoringConfig
from outboxml.data_subsets import DataPreprocessor
from outboxml.extractors import Extractor
from typing import List


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

    def prepare_data(self, data_preprocessor, models_config):
        try:
            subset = data_preprocessor.get_subset(model_name=models_config.name)

            prepared = prepare_dataset(
                group_name='default',
                data=self.actual.copy(),
                train_ind=self.actual.index,
                test_ind=pd.Index([]),
                model_config=models_config,
            )

            self.X_train = subset.X_train
            self.X_test = prepared.data

        except Exception as e:
            logger.exception("Failed to prepare data in MonitoringContext")
            raise e

@dataclass
class MonitoringContext:
    data_preprocessor: DataPreprocessor

    monitoring_result: MonitoringResult
    monitoring_config: MonitoringConfig
    models_config: List[ModelConfig]

    logs_extractor: Extractor
    #data_context: DataContext
