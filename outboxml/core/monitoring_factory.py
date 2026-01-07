from loguru import logger
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

from outboxml.monitoring_result import MonitoringContext, DataContext
from typing import Dict, Any, Optional, Union


class DataReviewerComponent(ABC):
    """
    Abstract base class for data reviewer components.

    Data reviewers are responsible for analyzing datasets
    and producing monitoring metrics or diagnostics.
    """

    @abstractmethod
    def review(self, context: MonitoringContext) -> pd.DataFrame:
        """
        Performs data review using provided monitoring context.

        :param context: Monitoring execution context.
        :type context: MonitoringContext

        :return: DataFrame with review results.
        :rtype: pandas.DataFrame
        """
        pass


class ReportComponent(ABC):
    """
    Abstract base class for monitoring report components.
    """
    def __init__(self):
        """
        Initializes report component.
        """
        pass

    @abstractmethod
    def make_report(self, *params) -> pd.DataFrame:
        """
        Builds a monitoring report.

        :param params: Input results from DataReviewerComponent.
        :type params: tuple

        :return: Final report as DataFrame.
        :rtype: pandas.DataFrame
        """
        pass


@dataclass
class MonitoringItem:
    """
    Configuration container for a single monitoring item.

    Combines a data reviewer, report generator, and execution settings.

    Attributes
    ----------
    data_reviewer : DataReviewerComponent
        Component responsible for data review.
    reviewer_report : ReportComponent
        Report generator component.
    group_models : bool
        Whether models are processed as a group.
    name : str
        Monitoring item name.
    table_name : str or None
        Target database table name.
    """
    data_reviewer: DataReviewerComponent
    reviewer_report: ReportComponent
    group_models: bool
    name: str
    table_name: Optional[str] = None


class DataReviewerRegistry:
    """
    Registry for data reviewer components.

    Allows dynamic registration and lookup of reviewer classes.
    """
    _monitorings = {}

    @classmethod
    def register(cls, name: str):
        """
        Registers a data reviewer class.

        :param name: Reviewer identifier.
        :type name: str

        :return: Class decorator.
        :rtype: callable

        .. rubric:: Examples

        >>> @DataReviewerRegistry.register("datadrift")
        >>> class DataDriftReviewer(DataReviewerComponent):
        >>>     ...
        """
        def decorator(monitoring_class):
            cls._monitorings[name] = monitoring_class
            return monitoring_class
        return decorator

    @classmethod
    def get(cls, name: str):
        """
        Retrieves a registered data reviewer class.

        :param name: Reviewer identifier.
        :type name: str

        :return: Reviewer class.
        :rtype: type

        :raises KeyError: If reviewer is not registered.
        """
        if name not in cls._monitorings:
            raise KeyError(f"Monitoring {name} is not registered")
        return cls._monitorings[name]

    @classmethod
    def list(cls):
        """
        Lists all registered data reviewers.

        :return: List of reviewer names.
        :rtype: list[str]
        """
        return list(cls._monitorings.keys())

class ReportRegistry():
    """
    Registry for monitoring report components.
    """
    _reports = {}
    @classmethod
    def register(cls, name: str):
        """
        Registers a report class.

        :param name: Report identifier.
        :type name: str

        :return: Class decorator.
        :rtype: callable
        """
        def decorator(report_class):
            cls._reports[name] = report_class
            return report_class
        return decorator

    @classmethod
    def get(cls, name: str):
        """
        Retrieves a registered report class.

        :param name: Report identifier.
        :type name: str

        :return: Report class.
        :rtype: type

        :raises KeyError: If report is not registered.
        """
        if name not in cls._reports:
            raise KeyError(f"Report {name} is not registered")
        return cls._reports[name]

    @classmethod
    def list(cls):
        """
        Lists all registered reports.

        :return: List of report names.
        :rtype: list[str]
        """
        return list(cls._reports.keys())


class MonitoringService:
    """
    Service responsible for executing monitoring items.

    Attributes
    ----------
    monitoring_items : list[MonitoringItem]
        List of configured monitoring items.
    """
    def __init__(self):
        """
        Initializes monitoring service.
        """
        self.monitoring_items = []

    def add_item(self, item: MonitoringItem):
        """
        Adds a monitoring item to the service.

        :param item: Monitoring item configuration.
        :type item: MonitoringItem
        """
        self.monitoring_items.append(item)

    def review_all(self, context: MonitoringContext) -> tuple[Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]], Dict[str, Dict[str, Union[pd.DataFrame, str]]]]:
        """
        Executes all monitoring items.

        :param context: Monitoring execution context.
        :type context: MonitoringContext

        :return: Tuple of raw review results and formatted reports.
        :rtype: tuple

        .. rubric:: Examples

        >>> reviews, reports = service.review_all(context)
        """
        data_context = DataContext(
            base=context.data_preprocessor.dataset,
            actual=context.logs_extractor.extract_dataset()
        )
        data_reviewer_results = {}
        reviewer_report_results = {}

        for item in self.monitoring_items:
            try:
                if not item.group_models:
                    models_reviewer_result = {}
                    for model in context.models_config:
                        temp_data_context = replace(data_context)
                        temp_data_context.prepare_data(context.data_preprocessor, model, )
                        reviewer_result = item.data_reviewer.review(temp_data_context)
                        models_reviewer_result[model.name] = reviewer_result

                    final_report = item.reviewer_report.make_report(models_reviewer_result, context)
                    data_reviewer_results[item.name] = models_reviewer_result
                    reviewer_report_results[item.name] = {
                        'df': final_report,
                        'db_table': item.table_name,
                    }
                else:
                    reviewer_result = item.data_reviewer.review(data_context)
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
    """
    Factory for creating monitoring service from configuration.
    """
    @staticmethod
    def create_from_config(
            monitoring_config
    ):
        """
        Creates MonitoringService based on configuration.

        :param monitoring_config: Monitoring configuration object.
        :type monitoring_config: MonitoringConfig

        :return: Initialized monitoring service.
        :rtype: MonitoringService

        .. rubric:: Examples

        >>> service = MonitoringFactory.create_from_config(config)
        """
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


@ReportRegistry.register("base_datadrift_report")
class MonitoringReport(ReportComponent):
    """
    Default monitoring report implementation.
    """
    def __init__(self):
        """
        Initializes default monitoring report.
        """
        super().__init__()

    def make_report(self, data_dict: pd.DataFrame, context: MonitoringContext) -> pd.DataFrame:
        """
        Builds a consolidated monitoring report.

        :param data_dict: Mapping of model names to review results.
        :type data_dict: Dict[str, pandas.DataFrame]

        :param context: Monitoring execution context.
        :type context: MonitoringContext

        :return: Final monitoring report.
        :rtype: pandas.DataFrame
        """
        report = pd.DataFrame()
        for key in data_dict.keys():
            df_result = data_dict[key].copy()
            df_result['model_name'] = key
            report = pd.concat([report, df_result])
        for column in report.columns:
            try:
                report[column] = report[column].astype('float')
            except:
                report[column] = report[column].astype(str)
        report['model_version'] = context.monitoring_result.model_version
        return report
