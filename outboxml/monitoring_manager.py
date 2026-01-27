import json
import os
import pickle

import pandas as pd
from loguru import logger
from pydantic import ValidationError

from outboxml import config
from outboxml.core.email import EMailMonitoring
from outboxml.core.pydantic_models import MonitoringConfig, ModelConfig, AllModelsConfig
from outboxml.datasets_manager import DataSetsManager
from outboxml.ensemble import Ensemble, EnsembleResult
from outboxml.export_results import ResultExport, GrafanaExport
from outboxml.extractors import Extractor
from outboxml.metrics.base_metrics import BaseMetric
from outboxml.core.monitoring_factory import MonitoringFactory
from outboxml.monitoring_result import MonitoringResult, DataContext, MonitoringContext



class MonitoringManager:
    """Orchestrator class for executing model monitoring pipeline.

    This class manages the full monitoring lifecycle: loading configurations,
    extracting data and logs, running monitoring checks, exporting results,
    and sending notifications.

    :var monitoring_service: Service responsible for executing monitoring checks.
    :vartype monitoring_service: MonitoringService
    :var result: Object storing monitoring results.
    :vartype result: MonitoringResult
    :var logs: Extracted production logs.
    :vartype logs: pandas.DataFrame or None

    .. rubric:: Examples

    .. code-block:: python

        manager = MonitoringManager(
            monitoring_config='configs/monitoring_test_config.json',
            models_config='configs/config_example_titanic.json',
            data_extractor=TitanicExampleExtractor(),
            logs_extractor=LogsExtractor()
        )
        result = manager.review(send_mail=True, to_grafana=True)
    """

    def __init__(self,
                 monitoring_config,
                 models_config,
                 external_config=None,
                 logs_extractor: Extractor = None,
                 data_extractor: Extractor = None,
                 target_extractor: Extractor = None,
                 grafana_connection=None,
                 business_metric: BaseMetric = None,
                 email: EMailMonitoring = None,
                 ):
        """
        Initializes monitoring manager.

        :param monitoring_config: Monitoring configuration or path to config file.
        :type monitoring_config: dict or str

        :param models_config: Model training configuration or path to config file.
        :type models_config: dict or str

        :param external_config: External configuration (email, connections, etc.).
        :type external_config: module or None

        :param logs_extractor: Extractor for production logs.
        :type logs_extractor: Extractor or None

        :param data_extractor: Extractor for training data.
        :type data_extractor: Extractor or None

        :param target_extractor: Extractor for target extrapolation.
        :type target_extractor: Extractor or None

        :param grafana_connection: Database connection for Grafana export.
        :type grafana_connection: Any

        :param business_metric: Business metric for model quality evaluation.
        :type business_metric: BaseMetric or None

        :param email: Email interface for notifications.
        :type email: EMailMonitoring or None
        """
        self._monitoring_config = monitoring_config
        self._models_config = models_config
        self._target_extractor = target_extractor
        if external_config is not None:
            self._external_config = external_config
        else:
            self._external_config = config

        if email is None:
            self.email = EMailMonitoring(config=self._external_config)
        else:
            self.email = email

        self.__load_default_models = False
        self.__grafana_connection = grafana_connection

        self._business_metric = business_metric
        self._ds_manager = DataSetsManager(config_name=self._models_config, extractor=data_extractor, external_config=external_config)
        self._result_export = ResultExport(ds_manager=self._ds_manager, config=self._external_config)
        self._logs_extractor = logs_extractor
        self.__init_monitoring()
        self._prod_models_configs = None
        self.__init_prod_models_configs()
        self.result = MonitoringResult(group_name=self._monitoring_config.group_name)
        self.result.dataset_name = self._define_dataset_name()

        self.monitoring_service = MonitoringFactory.create_from_config(
            self._monitoring_config,
        )

    def review(self,
               send_mail: bool = True,
               to_grafana: bool = True) -> MonitoringResult:
        """
        Executes monitoring process.

        This method performs the full monitoring cycle:
        data extraction, model preparation, monitoring checks,
        report generation, export, and notifications.

        :param send_mail: Whether to send email notification.
        :type send_mail: bool

        :param to_grafana: Whether to export results to Grafana.
        :type to_grafana: bool

        :return: Monitoring result object.
        :rtype: MonitoringResult

        .. rubric:: Examples

        Example usage::
        manager.review(send_mail=True, to_grafana=True)
        """
        self._ds_manager._retro = True
        self._ds_manager._init_dsmanager()
        context = MonitoringContext(
            data_preprocessor=self._ds_manager._data_preprocessor,
            logs_extractor=self._logs_extractor,
            monitoring_result=self.result,
            monitoring_config=self._monitoring_config,
            models_config=self._prod_models_configs,
            all_models_config=self.__init_all_models_config(self._models_config)
        )
        service_reviews, service_reports = self.monitoring_service.review_all(context=context)
        self.result.reviews = service_reviews
        self.result.reports = service_reports
        try:
            if to_grafana:
                for k in self.result.reports.keys():
                    table_name = self.result.reports[k]['db_table']
                    if table_name:
                        self._grafana_report(self.result.reports[k]['df'], table_name)
            if send_mail:
                self.email.success_mail(self.result)
        except Exception as exc:
            logger.error(exc)
            self.email.error_mail(group_name=self.result.group_name, error=exc)
        finally:
            return self.result


    def __init_monitoring(self):
        if isinstance(self._monitoring_config, dict):
            logger.info("Monitoring config from dict")
            auto_ml_config = json.dumps(self._monitoring_config)

        else:
            logger.info("All models config from path")
            try:
                with open(self._monitoring_config, "r", encoding='utf-8') as f:
                    auto_ml_config = f.read()
            except FileNotFoundError:
                logger.error("Invalid monitoring config name")
                raise FileNotFoundError("Invalid config name")

        try:
            self._monitoring_config = MonitoringConfig.model_validate_json(auto_ml_config)
        except ValidationError as e:
            logger.error("Config validation error")
            raise ValidationError(e)

    def __init_prod_models_configs(self):
        with open(f'{self._monitoring_config.prod_models_path}/{self._monitoring_config.pickle_name}.pickle',
                  'rb') as f:
            prod_model = pickle.load(f)
        prod_models_config = []
        for model in prod_model:
            if isinstance(prod_model, EnsembleResult):
                for val in model.models:
                    prod_models_config.append(ModelConfig.model_validate(val[2]['model_config']))
            elif isinstance(prod_model, list):
                prod_models_config.append(ModelConfig.model_validate(model['model_config']))
            else:
                logger.error("Invalid prod_models config type")

        self._prod_models_configs = prod_models_config

    def __init_all_models_config(self, models_config):
        if isinstance(models_config, dict):
            logger.info("AllModelsConfig config from dict")
            config = json.dumps(models_config)
        else:
            logger.info("AllModelsConfig from path")
            try:
                with open(models_config, "r", encoding='utf-8') as f:
                    config = f.read()
            except FileNotFoundError:
                logger.error("Invalid AllModelsConfig config name")
                raise FileNotFoundError("Invalid config name")

        try:
            all_models_config = AllModelsConfig.model_validate_json(config)
        except ValidationError as e:
            logger.error("Config validation error")
            raise ValidationError(e)
        return all_models_config

    def _grafana_report(self, report: pd.DataFrame, table_name: str) -> None:
        """
        Exports monitoring report to Grafana database.

        :param report: Monitoring report data.
        :type report: pandas.DataFrame

        :param table_name: Target database table name.
        :type table_name: str
        """
        try:
            GrafanaExport(df=report, connection=self.__grafana_connection,
                          table_name=table_name).load_data_to_db()
        except Exception as exc:
            logger.error(exc)
            logger.info('No results in grafana')


    def _define_dataset_name(self):
        """
        Determines dataset name based on data source.

        :return: Dataset name.
        :rtype: str
        """
        if self._monitoring_config.data_source in ['csv', 'parquet']:
            dataset_name = os.path.basename(os.path.splitext(self._ds_manager.data_config.local_name_source)[0])
        else:
            dataset_name = self._monitoring_config.data_config.table_name_source

        return dataset_name
