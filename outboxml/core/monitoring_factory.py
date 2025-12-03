from loguru import logger
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from outboxml.core.data_prepare import prepare_dataset
from typing import Dict, Any, Optional
from outboxml.datasets_manager import DataPreprocessor

@dataclass
class DataContext:
    base: pd.DataFrame
    actual: pd.DataFrame

    X_train: pd.DataFrame = None
    X_test: pd.DataFrame = None

@dataclass
class MonitoringContext:
    data_preprocessor: DataPreprocessor

    monitoring_result: Any
    monitoring_config: Any
    models_config: Any

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


class DataReviewerComponent(ABC):

    @abstractmethod
    def review(self, context: MonitoringContext) -> pd.DataFrame:
        pass


class ReportComponent(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def make_report(self, *params) -> pd.DataFrame:
        pass


@dataclass
class MonitoringItem:
    data_reviewer: DataReviewerComponent
    reviewer_report: ReportComponent
    group_models: bool
    name: str
    table_name: Optional[str] = None


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


class MonitoringService:
    def __init__(self):
        self.monitoring_items = []

    def add_item(self, item: MonitoringItem):
        self.monitoring_items.append(item)

    def review_all(self, context: MonitoringContext) -> tuple[dict[Any, Any], dict[Any, Any]]:

        data_reviewer_results = {}
        reviewer_report_results = {}

        for item in self.monitoring_items:
            try:
                if not item.group_models:
                    models_reviewer_result = {}
                    for model in context.models_config:
                        model_ctx = replace(context, models_config=model)
                        reviewer_result = item.data_reviewer.review(model_ctx)
                        models_reviewer_result[model.name] = reviewer_result

                    final_report = item.reviewer_report.make_report(models_reviewer_result, context)
                    data_reviewer_results[item.name] = models_reviewer_result
                    reviewer_report_results[item.name] = {
                        'df': final_report,
                        'db_table': item.table_name,
                    }
                else:
                    reviewer_result = item.data_reviewer.review(context)
                    data_reviewer_results[item.name] = reviewer_result
                    reviewer_report_results[item.name] = {
                        'df': item.reviewer_report.make_report(reviewer_result, context),
                        'db_table': item.table_name,
                    }
            except Exception as e:
                logger.exception(f"Error executing monitoring item '{item.name}'")
                continue

        return data_reviewer_results, reviewer_report_results

class MonitoringFactory:
    @staticmethod
    def create_from_config(
            monitoring_config
    ):
        service = MonitoringService()
        monitoring_factory = monitoring_config.monitoring_factory
        for item in monitoring_factory:
            data_reviewer_type = item.type
            reviewer_report_type = item.report
            group_models = item.group_models
            db_table_name = item.db_table_name
            params = item.parameters

            try:
                data_reviewer_class = DataReviewerRegistry.get(data_reviewer_type)
                reviewer_report_class = ReportRegistry.get(reviewer_report_type)
            except ValueError as e:
                logger.error(e)
                continue

            data_reviewer_instance = data_reviewer_class(**params)
            reviewer_report_instance = reviewer_report_class()

            m_item = MonitoringItem(
                data_reviewer=data_reviewer_instance,
                reviewer_report=reviewer_report_instance,
                name=data_reviewer_type,
                group_models=group_models,
                table_name=db_table_name
            )

            service.add_item(m_item)

        return service