from loguru import logger
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from outboxml.core.data_prepare import prepare_dataset
from typing import Dict, Any, Optional
from outboxml.core.pydantic_models import MonitoringFactoryConfig

class DataReviewerComponent(ABC):
    def __init__(self, group_model: bool = True):
        self.group_model: bool = group_model

    @abstractmethod
    def review(self, context) -> pd.DataFrame:
        pass


class ReportComponent(ABC):
    def __init__(self, monitoring_result, monitoring_config):
        self.monitoring_result = monitoring_result
        self.monitoring_config = monitoring_config

    @abstractmethod
    def make_report(self, data):
        pass

@dataclass
class DataReviewerContext:
    X_train: Optional[pd.DataFrame] = None
    X_test: Optional[pd.DataFrame] = None

    base: Optional[pd.DataFrame] = None
    actual: Optional[pd.DataFrame] = None

class DataReviewerRegistry:
    _monitorings = {}

    @classmethod
    def register(cls, name: str):
        def decorator(monitoring_class):
            cls._monitorings[name] = monitoring_class
            return monitoring_class
        return decorator

    @classmethod
    def get(cls, name: str):
        if name not in cls._monitorings:
            raise KeyError(f"Monitoring {name} is not registered")
        return cls._monitorings[name]

    @classmethod
    def list(cls):
        return list(cls._monitorings.keys())

class ReportRegistry():
    _reports = {}
    @classmethod
    def register(cls, name: str):
        def decorator(report_class):
            cls._reports[name] = report_class
            return report_class
        return decorator

    @classmethod
    def get(cls, name: str):
        if name not in cls._reports:
            raise KeyError(f"Report {name} is not registered")
        return cls._reports[name]

    @classmethod
    def list(cls):
        return list(cls._reports.keys())


@dataclass
class MonitoringItem:
    data_reviewer: DataReviewerComponent
    reviewer_report: ReportComponent
    name: str


class MonitoringService:
    def __init__(self, ds_manager):
        self.monitoring_items = []
        self._ds_manager = ds_manager

    def add_item(self, item: MonitoringItem):
        self.monitoring_items.append(item)

    def review_all(self, context: DataReviewerContext) -> tuple[dict[Any, Any], dict[Any, Any]]:
        data_reviewer_results = {}
        reviewer_report_results = {}

        for item in self.monitoring_items:
            try:
                if not item.data_reviewer.group_model:
                    models_reviewer_result = {}
                    for model in self._ds_manager._models_configs:
                        context.X_train = self._ds_manager.get_subset(model_name=model.name).X_train
                        context.X_test = prepare_dataset(group_name=self._ds_manager.group_name,
                                                 data=context.actual.copy(),
                                                 train_ind=context.actual.index,
                                                 test_ind=pd.Index([]),
                                                 model_config=model,
                                                 ).data
                        reviewer_result = item.data_reviewer.review(context)
                        models_reviewer_result[model.name] = reviewer_result

                    final_report = item.reviewer_report.make_report(models_reviewer_result)
                    data_reviewer_results[item.name] = models_reviewer_result
                    reviewer_report_results[item.name] = final_report
                else:
                    reviewer_result = item.data_reviewer.review(context)
                    data_reviewer_results[item.name] = reviewer_result
                    reviewer_report_results[item.name] = item.reviewer_report.make_report(reviewer_result)
            except:
                logger.error(f'Cannot review {item.name}')

        return data_reviewer_results, reviewer_report_results

class MonitoringFactory:
    @staticmethod
    def create_from_config(
            monitoring_config,
            ds_manager,
            monitoring_result):
        service = MonitoringService(ds_manager)
        monitoring_factory = monitoring_config.monitoring_factory
        for item in monitoring_factory:
            data_reviewer_type = item.type
            reviewer_report_type = item.report
            params = item.parameters

            try:
                data_reviewer_class = DataReviewerRegistry.get(data_reviewer_type)
                reviewer_report_class = ReportRegistry.get(reviewer_report_type)
            except ValueError as e:
                logger.error(e)
                continue

            data_reviewer_instance = data_reviewer_class(**params)
            reviewer_report_instance = reviewer_report_class(monitoring_result, monitoring_config)

            m_item = MonitoringItem(
                data_reviewer=data_reviewer_instance,
                reviewer_report=reviewer_report_instance,
                name=data_reviewer_type)

            service.add_item(m_item)

        return service