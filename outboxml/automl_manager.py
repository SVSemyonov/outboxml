import json
import os
import pickle
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd
import mlflow
from loguru import logger
from pydantic import ValidationError

from outboxml import config
from outboxml.automl_utils import load_last_pickle_models_result
from outboxml.core.email import EMail, AutoMLReviewEMail
from outboxml.core.enums import ModelsParams
from outboxml.core.prepared_datasets import FeatureSelectionPrepareDataset
from outboxml.core.pydantic_models import AutoMLConfig
from outboxml.core.utils import ResultPickle
from outboxml.dataset_retro import RetroDataset
from outboxml.datasets_manager import DataSetsManager, ModelDataSubset
from outboxml.export_results import ResultExport, GrafanaExport
from outboxml.extractors import Extractor
from outboxml.feature_selection import BaseFS, FeatureSelectionInterface
from outboxml.hyperparameter_tuning import HPTuning
from outboxml.main_release import Release
from outboxml.metrics.business_metrics import BaseCompareBusinessMetric
from outboxml.metrics.base_metrics import BaseMetric
from outboxml.plots import DataframeForPlots, CompareModelsPlot


class AutoMLResult:
    """Container of AutoMLResults"""

    def __init__(self, group_name: str):
        self.group_name = group_name
        self.start_run_time = datetime.now()
        self.run_time = {'start': self.start_run_time,
                         'retro': self.start_run_time,
                         'hp_tuning': self.start_run_time,
                         'fitting': self.start_run_time,
                         'comparing models': self.start_run_time,
                         'export results': self.start_run_time, }

        self.features_for_research = []
        self.new_features = {}
        self.new_hp = {}
        self.ds_manager_result = {}
        self.metrics = {'train': {}, 'test': {}}
        self.figures = []
        self.model_result_for_service = []
        self.result_pickle_name = 'No pickle'
        self.compare_metrics_df = pd.DataFrame()
        self.compare_business_metric = pd.DataFrame()
        self.end_time_run = datetime.now()
        self.deployment = False
        self.all_models_config = None


class MLFlowWrapper:
    """Wrapper of MLFlow to perform loading artefacts"""

    def __init__(self, experiment_name: str = 'FrameworkTest',
                 group_name: str = 'example',
                 results_path='results',
                 mlflow_tracking_uri="http://localhost:5000"):
        self.experiment_name = experiment_name
        self.group_name = group_name
        self.mlflow_tracking_uri = mlflow_tracking_uri
        os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self.results_path = results_path

    def start_run(self, ):
        mlflow.start_run(run_name=self.group_name)

    def end_run(self):
        mlflow.end_run()

    def log_results(self, automl_results: AutoMLResult):
        logger.debug('Exporting results to MLFlow')
        with mlflow.start_run(run_name=self.group_name + str(automl_results.run_time['start']), ):
            log = os.path.join(self.results_path, "log.log")
            mlflow.log_artifact(log)
            mlflow.log_artifact(os.path.join(self.results_path, automl_results.result_pickle_name))
            mlflow.set_tag(key='Deployment decision', value=automl_results.deployment)
            try:
                mlflow.log_artifact(os.path.join(self.results_path, automl_results.all_models_config))
            except:
                pass
            if automl_results.compare_business_metric is not None:
                business_metric = {'business_metric': automl_results.compare_business_metric['difference']}
                threshold = {'threshold': automl_results.compare_business_metric['first_model']['threshold']}
                mlflow.log_metrics(business_metric)
                mlflow.log_metrics(threshold)

            for model_name in automl_results.ds_manager_result.keys():

                with mlflow.start_run(run_name=model_name, nested=True):

                    model_config = os.path.join(self.results_path, self.group_name, model_name,
                                                f"{model_name}_model_config.json")
                    features_num = os.path.join(self.results_path, self.group_name, model_name,
                                                f"{model_name}_features_numerical.json")
                    features_cat = os.path.join(self.results_path, self.group_name, model_name,
                                                f"{model_name}_features_categorical.json")
                    model = os.path.join(self.results_path, self.group_name, model_name, f"{model_name}_model.pickle")


                    metrics_train = self.prepare_metrics(
                        automl_results.ds_manager_result[model_name].metrics['train']['full'], model_name)
                    metrics_test = self.prepare_metrics(
                        automl_results.ds_manager_result[model_name].metrics['test']['full'], model_name)
                    mlflow.log_metrics({f"train_{k}": v for k, v in metrics_train.items()})
                    mlflow.log_metrics({f"test_{k}": v for k, v in metrics_test.items()})

                    mlflow.log_artifact(features_num)  # модель
                    mlflow.log_artifact(features_cat)
                    mlflow.log_artifact(model)
                    mlflow.log_artifact(model_config)
                    try:
                        mlflow.log_params(dict(automl_results.new_hp[model_name]))
                    except:
                        logger.info('No parameters to log')
                #    mlflow.log_params(automl_results.new_hp[model_name])

    #  mlflow.end_run()

    def prepare_metrics(self, metrics_dict, model_name):

        metrics1 = {}
        for key in metrics_dict.keys():

            val = key.split(sep='_model')
            if val[0] == model_name:
                new_key = key
            else:
                continue

        try:
            metrics1 = metrics_dict[new_key]
        except:
            logger.error('No metrics in dict')
        return metrics1


class RetroFS(RetroDataset):
    def __init__(self, retro_columns: list):
        super().__init__()
        self.retro_columns = retro_columns

    def load_retro_data(self, *params):
        self.retro_data = pd.DataFrame(columns=self.retro_columns)


class AutoMLManager(DataSetsManager):
    """Класс для проведения AutoML. Использует функционал DataSetsManager

    Основные импорты:
        from outboxml.extractors.extractor import Extractor
        from outboxml.automl_manager import RetroFS, AutoMLManager
        from outboxml.core.prepared_datasets import FeatureSelectionPrepareDataset
        from outboxml.core.pydantic_models import FeatureSelectionConfig
        from outboxml.datasets_manager import DataSetsManager
        from outboxml.export_results import ResultExport
        from outboxml.feature_selection import BaseFS, FeatureSelectionInterface
        from outboxml.hyperparameter_tuning import HPTuning
        from outboxml.metrics.business_metrics import  BaseCompareBusinessMetric
        from outboxml.metrics.metrics import BaseMetric

    Parameters:
        auto_ml_config: путь или AutoMLConfig
        models_config: AllModelsConfig или путь к нему для обучения моделей
        extractor: интерфейс для получения данных
        business_metric: интерфейс расчёта бизнес метрики с результатом {'metric_name' : metricvalue}
        compare_business_metric: интерфейс для сранвения бизнес метрик и выставления порогов
        retro: проведение ретро
        hp_tune: побор гиперпараметров
        grafana_connection: объект подключения к БД. Передается в pd.to_sql()

    Methods:
        update_models()

        """

    def __init__(self,
                 auto_ml_config,
                 models_config,
                 extractor: Optional[Extractor] = None,
                 business_metric: BaseMetric = None,
                 compare_business_metric: BaseCompareBusinessMetric = None,
                 external_config=None,
                 retro: bool = True,
                 hp_tune: bool = True,
                 async_mode: bool = False,
                 save_temp: bool = False,
                 model_timeout_seconds: int = None,
                 grafana_connection=None,
                 models_dict: dict=None
                 ):
        super().__init__(config_name=models_config, extractor=extractor)
        self.models_dict = models_dict
        if external_config is not None:
            self._external_config = external_config
        else:
            self._external_config = config
        self.timeout = model_timeout_seconds
        self._business_metric = business_metric
        self.__default_models_config = models_config
        self._compare_business_metric = compare_business_metric
        self.__grafana_connection = grafana_connection
        self._async_mode = async_mode
        self._save_temp = save_temp
        self.features_list_to_exclude = []
        self._auto_ml_config = auto_ml_config
        self._feature_selection_config = None
        self._hp_tuning_config = None
        self.features_list = None
        self._retro = retro
        self._hp_tune = hp_tune
        self.__init_auto_ml()
        self.automl_results = AutoMLResult(group_name=self.group_name)
        self.mlflow = MLFlowWrapper(experiment_name=self._auto_ml_config.mlflow_experiment,
                                    group_name=self.group_name,
                                    results_path=self._external_config.results_path)
        self.status = {'Loading dataset': False,
                       'Feature selection': False,
                       'HP tuning': False,
                       'Fitting': False,
                       'Compare with previous': False,
                       'Deployment decision': False,
                       'Loading results to MLFLow': False,
                       'EMail Review': False}

        self.errors = {
            'Feature selection': False,
            'HP tuning': False,
            'Fitting': False,
            'Compare with previous': False,
            'Deployment decision': False,
            'Loading results to MLFLow': False,}

    def update_models(self, send_mail: bool = True, parameters_for_optuna: dict = None):

        logger.add(Path(str(self._external_config.results_path) + '/log.log'))
        email = AutoMLReviewEMail(config=self._external_config)
        try:
            self.load_dataset()
            self.status['Loading dataset'] = True
            self.feature_selection()

            if self._hp_tune:
                new_hp = self.hp_tuning(parameters_for_optuna)
                self.automl_results.new_hp = new_hp
                self.__update_hyperparameters(new_hp)
                self.status['HP tuning'] = True
                self.automl_results.run_time['hp_tuning'] = datetime.utcnow()

            results = self.fit_models(models_dict=self.models_dict)
            self.status['Fitting'] = True
            self.automl_results.run_time['fitting'] = datetime.utcnow()

            self.save_results(self._results)
            self.compare_with_previous()
            self.deployment()
            self.review(email, send_mail, )
            self.mlflow.log_results(self.automl_results)
            self.automl_results.run_time['export results'] = datetime.utcnow()
            self.status['Loading results to MLFLow'] = True
        except Exception as exc:
            logger.error(str(exc))
            try:

                self.mlflow.log_results(self.automl_results)
                self.automl_results.run_time['export results'] = datetime.utcnow()
                self.status['Loading results to MLFLow'] = True
            except Exception as exc2:
                logger.error(str(exc2))
            finally:
                email.error_mail(group_name=self.group_name,
                                 error=exc, status=self.status,
                                 )
        finally:

            return self.status

    def feature_selection(self):
        if self._retro:
            logger.debug('Feature selection||Started')
            self._separateTestTrain()
            extra_columns = None
            if self.extra_columns is not None:
                if isinstance(self.extra_columns, pd.Series):
                    extra_columns = pd.DataFrame(self.extra_columns)
                else:
                    extra_columns = self.extra_columns
            features_for_research = RetroFS(retro_columns=self.dataset.columns
                                            ).features_for_reserch(data_column_names=self.X.columns,
                                                                   target_columns_names=self.Y.columns,
                                                                   models_config=self._models_configs,
                                                                   extra_columns=extra_columns,
                                                                   features_list_to_exclude=self.features_list_to_exclude,
                                                                   )
            self.automl_results.features_for_research = features_for_research

            for model in self._models_configs:
                X, y, target = self._filter_data_by_exposure(model_name=model.name)
                feature_selector = BaseFS(data=X,
                                          train_ind=self.index_train,
                                          test_ind=self.index_test,
                                          target=target,
                                          parameters=self._feature_selection_config,
                                          new_features_list=features_for_research,
                                          data_prepare_interface=FeatureSelectionPrepareDataset(model_config=model),
                                          feature_selection_interface=FeatureSelectionInterface(
                                              feature_selection_config=self._feature_selection_config,
                                              objective=model.objective)
                                          )


                new_prepared_data = feature_selector.select_features(self._feature_selection_config.params)
                self.automl_results.new_features[model.name] = feature_selector.result_features
                self._prepare_datasets[model.name]._model_config = deepcopy(new_prepared_data.model_config)

                self.data_subsets[model.name] = ModelDataSubset.load_subset(
                    model_name=model.name,
                    X=new_prepared_data.data,
                    Y=y,
                    index_train=self.index_train,
                    index_test=self.index_test,
                    features_numerical=new_prepared_data.features_numerical,
                    features_categorical=new_prepared_data.features_categorical,
                    column_exposure=model.column_exposure if model.column_exposure else None,
                    column_target=model.column_target,
                    extra_columns=self.extra_columns
                )

            self.dataset = self.dataset.head()
            if self._save_temp:
                logger.info('Saving temp subsets')
                with open('prepared_subsets.pickle', 'wb') as f:
                    pickle.dump(self.data_subsets, f)
                self.data_subsets = None

            logger.debug('Feature selection||Finished')
            self.status['Feature selection'] = True
            self.automl_results.run_time['retro'] = datetime.utcnow()

    def hp_tuning(self, parameters_for_optuna: dict = None):
        new_hp = {}
        ds_manager_for_tune = DataSetsManager(config_name=self.__default_models_config)
        for model in self._models_configs:
            new_hp[model.name] = {}
            try:
                ds_manager_for_tune.load_dataset()
                ds_manager_for_tune.get_TrainDfs(load_subsets_from_pickle=self._save_temp)
                ds_manager_for_tune.dataset = ds_manager_for_tune.dataset.head()
                parameters_for_optuna_func = None
                if parameters_for_optuna is not None:
                    try:
                        parameters_for_optuna_func = parameters_for_optuna[model.name]
                    except KeyError:
                        pass
                new_hp[model.name] = HPTuning(ds_manager=ds_manager_for_tune,
                                              sampler=self._hp_tuning_config.sampling,
                                              scoring_fun=self._hp_tuning_config.metric_score[model.name],
                                              folds_num_for_cv=self._hp_tuning_config.cv_folds_num,
                                              objective=model.objective,
                                              random_state=self.config.data_config.separation.random_state
                                              ).best_params(model_name=model.name,
                                                            parameters_for_optuna_func=parameters_for_optuna_func,
                                                            timeout=self.timeout)
                logger.info(new_hp[model.name])

            except Exception as exc:
                self.errors['HP Tune'] = exc
                logger.info('Returning {}')
        return new_hp

    def save_results(self, results: dict):
        models_results = []
        self.automl_results.ds_manager_result = results

        for key in results:
            models_results.append(results[key].dict_for_prod_export())

        # артефакты и метрики
        logger.debug('Saving pickle for service')
        saving_start_time = datetime.utcnow()
        group_name = self.group_name
        ds_manager_for_save = DataSetsManager(config_name=self.__default_models_config)
        ds_manager_for_save.load_dataset(data=self.dataset)
        ds_manager_for_save._results = self._results
        ResultExport(ds_manager=ds_manager_for_save, config=self._external_config).save(to_mlflow=False,
                                                                                        path_to_save=self._external_config.results_path,
                                                                                        to_pickle=True)
        # Сохранение пикла для сервиса локально
        result_pickle_name = ResultPickle().generate_name(group_name, saving_start_time)
        self.automl_results.result_pickle_name = result_pickle_name
        self.automl_results.all_models_config = f"{group_name}_{saving_start_time.strftime('%Y_%m_%d_%H_%M_%S')}.json"

        with open(os.path.join(self._external_config.results_path, result_pickle_name), "wb") as f:
            pickle.dump(models_results, f)
        with open(os.path.join(self._external_config.results_path, self.automl_results.all_models_config), "wb") as f:
            pickle.dump(self.config.json(), f)

        try:
            logger.debug('Loading metrcis to grafana')
            df = ResultExport(ds_manager=ds_manager_for_save).grafana_export(project_name=group_name,
                                                                             date_time=datetime.utcnow())

            GrafanaExport(df=df, table_name=self._auto_ml_config.grafana_table_name,
                          connection=self.__grafana_connection).load_data_to_db()
        except Exception as exc:
            logger.error('grafana export error||' + str(exc))

    def compare_with_previous(self):
        result_to_compare = self._load_last_result()
        metrics_df = self._compare_base_metrics_df(result_to_compare=result_to_compare)

        self.automl_results.compare_metrics_df = metrics_df
        plots = self._compare_plots(result_to_compare=result_to_compare)
        self.automl_results.figures = plots
        if self._compare_business_metric is not None:
            self.automl_results.compare_business_metric = self._compare_business_metric.calculate_metric(
                result1=self._results,
                result2=result_to_compare,
                threshold=self._auto_ml_config.inference_criteria.threshold,
                )
            logger.info(self.automl_results.compare_business_metric)
        logger.debug('Comparing models is completed')
        self.status['Compare with previous'] = True
        self.automl_results.run_time['comparing models'] = datetime.utcnow()

    def deployment(self):
        self.automl_results.deployment = False
        metrics = self._auto_ml_config.inference_criteria.metric_growth_value

        for key in metrics.keys():
            self.automl_results.deployment = False
            try:
                if key == 'CompareBusinessMetric':
                    if self.automl_results.compare_business_metric is not None:
                        logger.info('Current compare business metric value: ' + str(
                            self.automl_results.compare_business_metric['difference']))
                    if self.automl_results.compare_business_metric['difference'] > metrics[key]:
                        self.automl_results.deployment = True
                else:
                    logger.info(
                        'Current business metric value: ' + str(self.automl_results.compare_metrics_df[key].min()))
                    if metrics[key] < self.automl_results.compare_metrics_df[key].min():
                        self.automl_results.deployment = True
                    else:
                        self.automl_results.deployment = False

            except KeyError:
                logger.error('Business metric error to complete decision for deployment')
        logger.info('Deployment decision: ' + str(self.automl_results.deployment))
        self.status['Deployment decision'] = True

    def review(self, email: EMail, send_mail: bool, error: Exception = None):
        if send_mail:
            if error is not None:
                email.common_error_mail(group_name=self.group_name, error=str(error))
            else:
                email.success_mail(self.automl_results)
                if self.status['Loading pickle to git']:
                    EMail(self._external_config).success_release_mail(self.automl_results.result_pickle_name,
                                                                      new_features=self.automl_results.new_features)
            self.status['EMail Review'] = True

    def __init_auto_ml(self, ):

        if isinstance(self._auto_ml_config, dict):
            logger.info("All models config from dict")
            auto_ml_config = json.dumps(self._auto_ml_config)

        else:
            logger.info("All models config from path")
            try:
                with open(self._auto_ml_config, "r", encoding='utf-8') as f:
                    auto_ml_config = f.read()
            except FileNotFoundError:
                logger.error("Invalid all models config name")
                raise FileNotFoundError("Invalid config name")

        try:
            self._auto_ml_config = AutoMLConfig.model_validate_json(auto_ml_config)
        except ValidationError as e:
            logger.error("Config validation error" + str(e))
            raise ValidationError(e)
        self.group_name = f"{self._auto_ml_config.group_name}"
        self._feature_selection_config = self._auto_ml_config.feature_selection
        self._hp_tuning_config = self._auto_ml_config.hp_tune  # use best model = False
        if self._retro:
            self.features_list_to_exclude = self.__load_features_list()

        logger.debug('AutoML init completed')

    def __load_features_list(self):
        return self._feature_selection_config.features_to_ignore

    def _load_last_result(self) -> dict:
        model_to_compare = load_last_pickle_models_result(self._external_config)
        data = self.load_dataset()
        result_to_compare = {}
        try:
            for key in model_to_compare.keys():

                models = model_to_compare[key]

                for model_result in models:
                    model_name = model_result['model_config']['name']
                    result_to_compare[model_name] = self.model_predict(data=data,
                                                                       model_name=model_name,
                                                                       model_result=model_result,
                                                                       train_ind=self.index_train,
                                                                       test_ind=self.index_test)
        except Exception as exc:
            logger.error(exc)
            logger.info('Cannot get results for last model||' + str(exc))
        self.dataset = self.dataset.head()
        return result_to_compare

    def _compare_base_metrics_df(self, result_to_compare: dict):

        df = pd.DataFrame()
        for key in self._results.keys():
            model_config = self._results[key].config
            ds = DataSetsManager(config_name=self.__default_models_config)
            ds._all_models_config = model_config
            res_export = ResultExport(ds_manager=ds)
            res_export.result = self._results
            try:
                metrics_df = res_export.compare_metrics(model_name=key,
                                                        ds_manager_result=result_to_compare,
                                                        business_metric=self._business_metric,
                                                        show=False, only_main=True)
                metrics_df['Имя модели'] = key
                df = pd.concat([df, metrics_df])
            except Exception as exc:
                logger.error(exc)
        return df

    def _compare_plots(self, result_to_compare: dict):
        figures = {}
        for key in self._results.keys():
            y_graph, features_categorical, features_numerical = DataframeForPlots().df_for_plots(
                result=self._results[key],
                use_exposure=True)
            y_graph2, features_categorical2, features_numerical2 = DataframeForPlots().df_for_plots(
                result=result_to_compare[key],
                use_exposure=True)

            figures[key] = CompareModelsPlot(model_name=key,
                                             df1=y_graph,
                                             df2=y_graph2,
                                             features_categorical=features_categorical,
                                             features_numerical=features_numerical,
                                             show=False).make(plot_type=2,
                                                              cut_min_value=0.1,
                                                              cut_max_value=0.9,
                                                              samples=100,
                                                              cohort_base='model1',
                                                              )
            figures[key].show()

        return figures

    def __update_hyperparameters(self, new_hp: dict):
        logger.info('New parameters are setting')
        for key in new_hp.keys():
            for model in self._models_configs:
                if model.name == key:
                    if model.wrapper == ModelsParams.catboost:
                        model.params_catboost = new_hp[key]
                    elif model.wrapper == ModelsParams.glm or model.wrapper == ModelsParams.glm_without_scaler:
                        model.params_glm = new_hp[key]
                    else:
                        pass
