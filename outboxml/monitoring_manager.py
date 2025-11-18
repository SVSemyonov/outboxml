import json
import os
import pickle

import pandas as pd
from loguru import logger
from pydantic import ValidationError
from sqlalchemy import create_engine, text

from outboxml import config
from outboxml.core.data_prepare import prepare_dataset
from outboxml.core.email import EMailMonitoring
from outboxml.core.pydantic_models import MonitoringConfig
from outboxml.datadrift import DataDrift
from outboxml.dataset_monitoring_interface import DatasetMonitor
from outboxml.datasets_manager import DataSetsManager
from outboxml.export_results import ResultExport, DashboardExport
from outboxml.extractors import Extractor
from outboxml.metrics.base_metrics import BaseMetric


class MonitoringResult:
    def __init__(self, group_name):
        self.group_name = group_name
        self.model_version = 'default'
        self.datadrift = {}
        self.dataset_monitor = {}
        self.metric = None
        self.extrapolation_results = {}
        self.report = {} #pd.DataFrame()
        self.grafana_dashboard = None



class MonitoringReport:
    def __init__(self, ):
        self._monitoring_result = None

    def make_datadrift_report(self) -> pd.DataFrame:
        report = pd.DataFrame()
        for key in self._monitoring_result.datadrift.keys():
            df_result = self._monitoring_result.datadrift[key].copy()
            df_result['model_name'] = key
            report = pd.concat([report, df_result])
        for column in report.columns:
            try:
                report[column] = report[column].astype('float')
            except:
                report[column] = report[column].astype(str)
        report['model_version'] = self._monitoring_result.model_version
        return report

    def make_dataset_report(self) -> pd.DataFrame:
        report = pd.DataFrame()
        for key in self._monitoring_result.dataset_monitor.keys():
            df_result = self._monitoring_result.dataset_monitor[key].copy()
        return df_result

    def return_report(self, monitoring_result, report_type):
        self._monitoring_result = monitoring_result
        if report_type == 'datadrift':
            return self.make_datadrift_report()
        elif report_type == 'dataset':
            return self.make_dataset_report()

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
         datadrift_interface: DataDrift - интерфейс для расчёта датадрифта
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
                 monitoring_report: MonitoringReport = MonitoringReport(),
                 datadrift_interface: DataDrift = None,
                 dataset_monitor_interface: DatasetMonitor = None,
                 grafana_connection=None,
                 superset_connection=None,
                 business_metric: BaseMetric = None,
                 email: EMailMonitoring = None,
                 ):
        self._monitoring_report = monitoring_report
        self._monitoring_config = monitoring_config
        self._models_config = models_config
        self._target_extractor = target_extractor
        self.__init_monitoring()

        if external_config is not None:
            self._external_config = external_config
        else:
            self._external_config = config

        if datadrift_interface is None:
            self.datadrift = DataDrift(full_calc=True)
        else:
            self.datadrift = datadrift_interface

        if dataset_monitor_interface is None:
            self.dataset_monitor_interface = DatasetMonitor()
        else:
            self.dataset_monitor_interface = dataset_monitor_interface

        if email is None:
            self.email = EMailMonitoring(config=self._external_config)
        else:
            self.email = email

        self.__load_default_models = False
        self.__grafana_connection = grafana_connection
        self.__superset_connection = superset_connection

        self._business_metric = business_metric
        self._ds_manager = DataSetsManager(config_name=self._models_config, extractor=data_extractor, external_config=external_config)

        self._result_export = ResultExport(ds_manager=self._ds_manager, config=self._external_config)
        self._logs_extractor = logs_extractor
        self.__init_monitoring()
        self.result = MonitoringResult(group_name=self._monitoring_config.group_name)
        self.logs = None

    def review(self,
               check_datadrift: bool = True,
               check_dataset: bool = True,
               send_mail: bool = True,
               to_grafana: bool = True,
               to_superset: bool = True) -> MonitoringResult:
        try:
            if check_datadrift:
                self.datadrift_review()
                datadrift_report = self.prepare_report('datadrift')
                self.result.report['datadrift'] = datadrift_report
            if check_dataset:
                self.dataset_monitor_review()
                dataset_monitor_report = self.prepare_report('dataset')
                self.result.report['dataset'] = dataset_monitor_report
            if to_grafana:
                self._dashboard_report(datadrift_report, dashboard_name='grafana')
            if to_superset:
                self._dashboard_report(dataset_monitor_report, dashboard_name='superset')
            if send_mail:
                self.email.success_mail(self.result)
        except Exception as exc:
              logger.error(exc)
              self.email.error_mail(group_name=self.result.group_name, error=exc)

        finally:
            return self.result

    def prepare_report(self, report_type: str) -> pd.DataFrame:
        logger.info(f'Preparing {report_type} report...')
        return self._monitoring_report.return_report(self.result, report_type=report_type)


    def datadrift_review(self):
        if self.logs is None:
            self.logs = self._logs_extractor.extract_dataset()
            logger.debug('Logs are loaded')

        for model in self._ds_manager._models_configs:
            try:
                logger.debug('Calculating datadrift|| ' + model.name)
                data_subset = self._ds_manager.get_subset(model_name=model.name)
                X_test = prepare_dataset(group_name=self._ds_manager.group_name,
                                         data=self.logs.copy(),
                                         train_ind=self.logs.index,
                                         test_ind=pd.Index([]),
                                         model_config=model,
                                         ).data

                self.result.datadrift[model.name] = self._datadrift_report(data_subset.X_train, X_test)
                logger.debug('Finished datadrift|| ' + model.name)
            except Exception as exc:
                logger.error(exc)
                logger.info('No datadrift results for model')
        return self.result.datadrift

    def dataset_monitor_review(self):
        self.result.dataset_monitor[self._monitoring_config.pickle_name] = self._dataset_monitor_report()


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

    def _datadrift_report(self, X_train, X_test):
        logger.info('Calculating datadrift...')
        return self.datadrift.report(X_train, X_test, self._ds_manager.dataset, self.logs)

    def _dataset_monitor_report(self):
        logger.info('Calculating dataset_report...')
        if self._monitoring_config.data_source in ['csv', 'parquet']:
            dataset_name = os.path.basename(os.path.splitext(self._ds_manager.data_config.local_name_source)[0])
        else:
            dataset_name = self._monitoring_config.data_config.table_name_source
        return self.dataset_monitor_interface.report(
            monitoring_config=self._monitoring_config,
            dataset_name=dataset_name,
            data=self._ds_manager.load_dataset()
        )
    # early _grafana_report
    def _dashboard_report(self, report: pd.DataFrame, dashboard_name: str = 'grafana'):
        if dashboard_name == 'grafana':
            conn = self.__grafana_connection
            table_name = self._monitoring_config.grafana_table_name
        elif dashboard_name == 'superset':
            conn = self.__superset_connection
            table_name = self._monitoring_config.superset_table_name
            self._delete_existing_report(conn, report)
        try:
            DashboardExport(df=report, connection=conn,
                          table_name=table_name).load_data_to_db()
        except Exception as exc:
            logger.error(exc)
            logger.info(f'No results in {dashboard_name}')

    def _delete_existing_report(self, connection, df):
        col_dataset_name = self._monitoring_config.report_uniq_cols['dataset']
        col_model_name = self._monitoring_config.report_uniq_cols['model']

        dataset_name = df[col_dataset_name].unique()[0]
        model_name = df[col_model_name].unique()[0]
        table_name = self._monitoring_config.superset_table_name

        engine = create_engine(connection)
        with engine.begin() as conn:
            result = conn.execute(
            text(
                f"""
                DELETE FROM "{table_name}"
                WHERE "{col_dataset_name}" = '{dataset_name}'
                  AND "{col_model_name}" = '{model_name}'
                """
            )
        )
        if result.rowcount > 0:
            logger.warning(f'Deleted {result.rowcount} rows from {table_name}, because dataset "{dataset_name}" and model "{model_name}" already exists')

    def _load_prod_model(self):

        with open(
                os.path.join(self._monitoring_config.prod_models_path, f"{self._monitoring_config.model_name}.pickle"),
                "rb") as f:
            group = pickle.load(f)
        self.result.model_version = self._monitoring_config.model_name
        logger.info(self._monitoring_config.model_name + ' is loaded from prod path')
        return group
