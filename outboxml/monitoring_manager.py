import json
import os
import pickle

import pandas as pd
from loguru import logger
from pydantic import ValidationError

from outboxml import config
from outboxml.core.email import EMailMonitoring
from outboxml.core.pydantic_models import MonitoringConfig
from outboxml.datasets_manager import DataSetsManager
from outboxml.export_results import ResultExport, GrafanaExport
from outboxml.extractors import Extractor
from outboxml.metrics.base_metrics import BaseMetric
from outboxml.core.monitoring_factory import (
    MonitoringFactory,
    ReportRegistry,
    ReportComponent,
    MonitoringContext,
)
from outboxml.monitoring_result import MonitoringResult



class MonitoringManager:
    """класс для проведения процесса мониторинга для выбранной модели.

    Для работы необходимы два конфиг-файла:
    конфиг мониторинга и конфиг модели. Дополнительно прописываются экстракторы для получения логов и данных с обучения.
    Также экстрактор для экстраполяции таргета. Расчёт датадрифта производится по стандратному интерфейсу.
    Возможна передача пользовательского интерфейса DataDrift

    Также необходима перегрузка метода review для пользовательской формы отчета

    Parameters:
         monitoring_config: конфиг для мониторинга
         models_config: конфиг модели для обучения
         external_config: конфиг для почты и др. подключений
         logs_extractor: Extractor - экстрактор логов
         data_extractor: Extractor - экстрактор данных обучения модели
         target_extractor: Extractor - экстрактор для экстраполяции таргета
         monitoring_report: MonitoringReport - форма отчета для мониторина. По умолчанию отчет по датадрифту
         target_extrapolation_models: dict - слоаврь моделей вида {model_name: TargetModel}
         grafana_connection: подключения для загрузки данных в БД, передается в pd.to_sql()
         business_metric: BaseMetric - Метрика для расчёта качества модели
         email: EMailMonitoring - интерфейс для отправки письма

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

        self.result = MonitoringResult(group_name=self._monitoring_config.group_name)
        self.result.dataset_name = self._define_dataset_name()

        self.monitoring_service = MonitoringFactory.create_from_config(
            self._monitoring_config,
        )
        self.logs = None

    def review(self,
               send_mail: bool = True,
               to_grafana: bool = True) -> MonitoringResult:
        if self.logs is None:
            self.logs = self._logs_extractor.extract_dataset()
            logger.debug('Logs are loaded')
        context = MonitoringContext(
            data_preprocessor=self._ds_manager._data_preprocessor,
            actual=self.logs,
            monitoring_result=self.result,
            monitoring_config=self._monitoring_config,
            models_config=self._ds_manager._models_configs,
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

    def _grafana_report(self, report: pd.DataFrame, table_name: str) -> None:
        try:
            GrafanaExport(df=report, connection=self.__grafana_connection,
                          table_name=table_name).load_data_to_db()
        except Exception as exc:
            logger.error(exc)
            logger.info('No results in grafana')


    def _define_dataset_name(self):
        if self._monitoring_config.data_source in ['csv', 'parquet']:
            dataset_name = os.path.basename(os.path.splitext(self._ds_manager.data_config.local_name_source)[0])
        else:
            dataset_name = self._monitoring_config.data_config.table_name_source

        return dataset_name

    def _load_prod_model(self):
        with open(
                os.path.join(self._monitoring_config.prod_models_path, f"{self._monitoring_config.pickle_name}.pickle"),
                "rb") as f:
            group = pickle.load(f)
        self.result.model_version = self._monitoring_config.pickle_name
        logger.info(self._monitoring_config.pickle_name + ' is loaded from prod path')
        return group
