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
from outboxml.datadrift import DataDrift, RollingWindowDataDrift
from outboxml.dataset_monitoring_interface import DatasetMonitor
from outboxml.datasets_manager import DataSetsManager
from outboxml.export_results import ResultExport, DashboardExport
from outboxml.extractors import Extractor
from outboxml.metrics.base_metrics import BaseMetric
from outboxml.core.prepared_datasets import TrainTestIndexes


class MonitoringResult:
    def __init__(self, group_name):
        self.group_name = group_name
        self.model_version = 'default'
        self.datadrift = {}
        self.rolling_datadrift = {}
        self.dataset_monitor = {}
        self.metric = None
        self.extrapolation_results = {}
        self.report = {}  #pd.DataFrame()
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

    def return_report(self, monitoring_result, report_type):
        self._monitoring_result = monitoring_result
        if report_type == 'datadrift':
            return self.make_datadrift_report()
        elif report_type == 'dataset':
            for key in self._monitoring_result.dataset_monitor.keys():
                return self._monitoring_result.dataset_monitor[key].copy()
        elif report_type == 'rolling_datadrift':
            for key in self._monitoring_result.rolling_datadrift.keys():
                return self._monitoring_result.rolling_datadrift[key].copy()


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
         monitoring_report: MonitoringReport - форма отчета для мониторина. По умолчанию отчет по датадрифту
         datadrift_interface: DataDrift - интерфейс для расчёта датадрифта
         rolling_datadrift_interface: RollingWindowDataDrift - интерфейс для расчета датадрифта по скользящему окну даты
         dataset_monitor_interface: DatasetMonitor - интерфейс для расчета пропусков и других метрик в датасете
         dashboard_connection: подключения для загрузки данных в БД, передается в pd.to_sql()
         business_metric: BaseMetric - Метрика для расчёта качества модели
         email: EMailMonitoring - интерфейс для отправки письма

    """

    def __init__(self,
                 monitoring_config,
                 models_config,
                 external_config=None,
                 logs_extractor: Extractor = None,
                 data_extractor: Extractor = None,
                 monitoring_report: MonitoringReport = MonitoringReport(),
                 datadrift_interface: DataDrift = None,
                 rolling_datadrift_interface: RollingWindowDataDrift = None,
                 dataset_monitor_interface: DatasetMonitor = None,
                 dashboard_connection=None,
                 business_metric: BaseMetric = None,
                 email: EMailMonitoring = None
                 ):
        self._monitoring_report = monitoring_report
        self._monitoring_config = monitoring_config
        self._models_config = models_config
        self.__init_monitoring()

        if external_config is not None:
            self._external_config = external_config
        else:
            self._external_config = config

        if datadrift_interface is None:
            self.datadrift = DataDrift()
        else:
            self.datadrift = datadrift_interface

        if dataset_monitor_interface is None:
            self.dataset_monitor_interface = DatasetMonitor()
        else:
            self.dataset_monitor_interface = dataset_monitor_interface

        if rolling_datadrift_interface is None:
            self.rolling_datadrift_interface = RollingWindowDataDrift(date_col=self._monitoring_config.period_column)
        else:
            self.rolling_datadrift_interface = rolling_datadrift_interface

        if email is None:
            self.email = EMailMonitoring(config=self._external_config)
        else:
            self.email = email

        self.__load_default_models = False
        if dashboard_connection is None:
            self.__dashboard_connection = config.connection_params
        else:
            self.__dashboard_connection = dashboard_connection

        self._business_metric = business_metric
        self._ds_manager = DataSetsManager(config_name=self._models_config, extractor=data_extractor,
                                           external_config=external_config)
        self._dataset_name = self._define_dataset_source()
        self._result_export = ResultExport(ds_manager=self._ds_manager, config=self._external_config)
        self._logs_extractor = logs_extractor
        if self._ds_manager.config.data_config.separation.kind == 'date':
            self._train_indexes, _ = TrainTestIndexes(self._ds_manager.dataset, self._ds_manager.config.data_config.separation).train_test_indexes()
        else:
            self._train_indexes = None
        self.result = MonitoringResult(group_name=self._monitoring_config.group_name)
        self.logs = None
        self._prod_group_name = self._load_prod_model()

    def review(self,
               check_datadrift: bool = True,
               check_dataset: bool = True,
               check_rolling_datadrift: bool = False,
               send_mail: bool = True,
               to_dashboard: bool = True,
               prepare_datadrift_data: bool = True) -> MonitoringResult:
        try:
            if check_datadrift:
                self.datadrift_review(prepare_datadrift_data)
                datadrift_report = self.prepare_report('datadrift')
                report_result_datadrift = (datadrift_report, f"{self._monitoring_config.db_base_name}_Datadrift")
                self.result.report['datadrift'] = report_result_datadrift
            if check_dataset:
                self.dataset_monitor_review()
                dataset_monitor_report = self.prepare_report('dataset')
                report_result_dataset = (dataset_monitor_report, f"{self._monitoring_config.db_base_name}_Dataset")
                self.result.report['dataset'] = report_result_dataset
            if check_rolling_datadrift:
                self.rolling_datadrift_review()
                rolling_datadrift_report = self.prepare_report('rolling_datadrift')
                report_result_rolling_datadrift = (rolling_datadrift_report,
                                                   f"{self._monitoring_config.db_base_name}_RollingDatadrift")
                self.result.report['rolling_datadrift'] = report_result_rolling_datadrift
            if to_dashboard:
                for dashboard_type in self.result.report.keys():
                    report_data = self.result.report[dashboard_type]
                    self._dashboard_report(report_data)
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

    def datadrift_review(self, prepare_datadrift_data: bool):
        if self.logs is None:
            self.logs = self._logs_extractor.extract_dataset()
            logger.debug('Logs are loaded')
        for model in self._ds_manager._models_configs:
            try:
                logger.debug('Calculating datadrift|| ' + model.name)
                if prepare_datadrift_data:
                    data_subset = self._ds_manager.get_subset(model_name=model.name)
                    base = data_subset.X_train
                    control = prepare_dataset(group_name=self._ds_manager.group_name,
                                              data=self.logs.copy(),
                                              train_ind=self.logs.index,
                                              test_ind=pd.Index([]),
                                              model_config=model,
                                              ).data
                else:
                    if self._train_indexes is None:
                        base = self._ds_manager.dataset
                    else:
                        base = self._ds_manager.dataset.loc[self._train_indexes[0]:self._train_indexes[1]]
                    control = self.logs
                report = self._datadrift_report(base, control)
                self.result.datadrift[model.name] = report
                logger.debug('Finished datadrift|| ' + model.name)
            except Exception as exc:
                logger.error(exc)
                logger.info('No datadrift results for model')
        return self.result.datadrift

    def dataset_monitor_review(self):
        self.result.dataset_monitor[self._monitoring_config.pickle_name] = self._dataset_monitor_report()

    def rolling_datadrift_review(self):
        self.result.rolling_datadrift[self._monitoring_config.pickle_name] = self._rolling_datadrift_report()

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

    def _datadrift_report(self, base, control):
        logger.info('Calculating datadrift...')
        return self.datadrift.report(base, control, self._dataset_name)

    def _rolling_datadrift_report(self, ):
        logger.info('Calculating rolling datadrift...')
        return self.rolling_datadrift_interface.report(
            dataset_name=self._dataset_name,
            data=self._ds_manager.dataset
        )

    def _dataset_monitor_report(self):
        logger.info('Calculating dataset_report...')
        return self.dataset_monitor_interface.report(
            monitoring_config=self._monitoring_config,
            dataset_name=dataset_name,
            data=self._ds_manager.load_dataset()
            dataset_name=self._dataset_name,
            prod_models_configs=self._prod_group_name,
            data=self._ds_manager.dataset
        )

    def _dashboard_report(self, report: tuple):
        report_df, report_table_name = report
        try:
            self._delete_existing_report(self.__dashboard_connection, report)
        except:
            logger.warning(f"table {report_table_name} does not exist yet")
        try:
            DashboardExport(df=report_df, connection=self.__dashboard_connection,
                            table_name=report_table_name).load_data_to_db()
        except Exception as exc:
            logger.error(exc)
            logger.info(f'No results in dashboard')

    def _delete_existing_report(self, connection, report):
        report_df, report_table_name = report

        def create_condition(df: pd.DataFrame):
            cols = []
            for col, metadata in df.attrs['meta'].items():
                if metadata.get('uniq') == True:
                    cols.append(col)

            condition = ''
            for col in cols:
                uniq_vals = df[col].unique()
                condition += f'''AND "{col}" IN ({", ".join(f"'{val}'" for val in uniq_vals)}) '''

            return condition

        condition = create_condition(report_df)

        engine = create_engine(connection)
        with engine.begin() as conn:
            result = conn.execute(
                text(
                    f"""
                DELETE FROM "{report_table_name}"
                WHERE 1 = 1
                {condition}
                """
                )
            )
        if result.rowcount > 0:
            logger.warning(f'Deleted {result.rowcount} rows from {report_table_name} with condition || {condition}')

    def _define_dataset_source(self):
        if self._monitoring_config.data_source in ['csv', 'parquet']:
            dataset_name = os.path.basename(os.path.splitext(self._ds_manager.data_config.local_name_source)[0])
        else:
            dataset_name = self._monitoring_config.data_config.table_name_source

        return dataset_name

    def _load_prod_model(self):
        with open(
                os.path.join(self._monitoring_config.prod_models_path, f"{self._monitoring_config.pickle_name}."),
                "rb") as f:
            group = pickle.load(f)
        self.result.model_version = os.path.splitext(self._monitoring_config.pickle_name)[0]
        logger.info(self._monitoring_config.pickle_name + ' is loaded from prod path')
        return {self.result.model_version: group}
