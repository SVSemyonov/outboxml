import json
import os
import pickle
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import optuna
import pandas as pd
import mlflow
from dotenv.main import rewrite
from loguru import logger
from pydantic import ValidationError

from outboxml import config
from outboxml.automl_utils import load_last_pickle_models_result
from outboxml.core.email import EMail, AutoMLReviewEMail, HTMLReport
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
from outboxml.plots import DataframeForPlots, CompareModelsPlot, MLPlot


class AutoMLResult:
    """Container class for AutoML execution results.
    
    Stores all results, metrics, and metadata from an AutoML training run,
    including feature selection results, hyperparameters, model metrics,
    deployment decisions, and execution times.
    
    :param group_name: Name of the model group for this AutoML run.
    :type group_name: str
    :var start_run_time: Timestamp when the AutoML run started.
    :var run_time: Dictionary mapping stage names to execution timestamps.
    :var features_for_research: List of features selected for research.
    :var new_features: Dictionary mapping model names to lists of new features.
    :var new_hp: Dictionary mapping model names to optimized hyperparameters.
    :var ds_manager_result: Dictionary mapping model names to DSManagerResult objects.
    :var metrics: Dictionary with 'train' and 'test' keys containing metrics.
    :var figures: List of Plotly figures for visualization.
    :var model_result_for_service: List of model results formatted for service deployment.
    :var result_pickle_name: Name of the pickle file containing results.
    :var compare_metrics_df: DataFrame comparing metrics between current and previous models.
    :var compare_business_metric: DataFrame with business metric comparisons.
    :var end_time_run: Timestamp when the AutoML run finished.
    :var deployment: Boolean indicating whether models were deployed to production.
    :var all_models_config: Path or name of the all models configuration file.
    :var business_metric: Dictionary of business metrics.
    
    .. rubric:: Examples
    
    .. code-block:: python
    
        result = AutoMLResult(group_name="Titanic_Model_v1")
        print(result.group_name)
        # Output: Titanic_Model_v1
    """

    def __init__(self, group_name: str):
        """Initialize AutoMLResult instance.
        
        :param group_name: Name of the model group for this AutoML run.
        :type group_name: str
        """
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
        self.business_metric = {}


class MLFlowWrapper:
    """Wrapper for MLFlow to perform artifact logging and experiment tracking.
    
    Provides a convenient interface for logging AutoML results to MLFlow,
    including models, metrics, artifacts, and hyperparameters. Supports
    nested runs for organizing multiple models within a single experiment.
    
    :param experiment_name: Name of the MLFlow experiment. Defaults to 'FrameworkTest'.
    :type experiment_name: str
    :param group_name: Name of the model group/run. Defaults to 'example'.
    :type group_name: str
    :param project: Project name (currently not used). Defaults to 'test_project'.
    :type project: str
    :param tags: Optional dictionary of tags to attach to runs. Defaults to None.
    :type tags: dict, optional
    :param results_path: Path to the results directory containing artifacts.
        Defaults to 'results'.
    :type results_path: str
    :param mlflow_tracking_uri: MLFlow tracking server URI. Defaults to 'http://localhost:5000'.
    :type mlflow_tracking_uri: str
    
    .. rubric:: Examples
    
    .. code-block:: python
    
        wrapper = MLFlowWrapper(
            experiment_name="Titanic_Experiment",
            group_name="Titanic_Model_v1",
            results_path="/path/to/results"
        )
        wrapper.start_run()
        # ... perform training ...
        wrapper.log_results(automl_result)
    """

    def __init__(self, experiment_name: str = 'FrameworkTest',
                 group_name: str = 'example',
                 project: str = 'test_project',
                 tags: dict = None,
                 results_path='results',
                 mlflow_tracking_uri="http://localhost:5000"):
        """Initialize MLFlowWrapper instance.
        
        :param experiment_name: Name of the MLFlow experiment. Defaults to 'FrameworkTest'.
        :type experiment_name: str
        :param group_name: Name of the model group/run. Defaults to 'example'.
        :type group_name: str
        :param project: Project name (currently not used). Defaults to 'test_project'.
        :type project: str
        :param tags: Optional dictionary of tags to attach to runs. Defaults to None.
        :type tags: dict, optional
        :param results_path: Path to the results directory containing artifacts.
            Defaults to 'results'.
        :type results_path: str
        :param mlflow_tracking_uri: MLFlow tracking server URI. Defaults to 'http://localhost:5000'.
        :type mlflow_tracking_uri: str
        """
        self.experiment_name = experiment_name
        self.group_name = group_name
        self.mlflow_tracking_uri = mlflow_tracking_uri
        os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self.results_path = results_path
        if tags is not None:
            self.tags = tags

    def start_run(self, ):
        """Start a new MLFlow run.
        
        Creates and starts a new MLFlow run with the configured group_name.
        
        :return: None
        :rtype: None
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            wrapper = MLFlowWrapper(group_name="MyModel")
            wrapper.start_run()
        """
        mlflow.start_run(run_name=self.group_name)

    def end_run(self):
        """End the current MLFlow run.
        
        Closes the active MLFlow run.
        
        :return: None
        :rtype: None
        
        .. rubric:: Examples
        
        .. code-block:: python
        
            wrapper.end_run()
        """
        mlflow.end_run()

    def log_results(self, automl_results: AutoMLResult, *tags):
        """Log AutoML results to MLFlow.
        
        Logs all artifacts, metrics, and parameters from an AutoML run to MLFlow.
        Creates a main run for the group and nested runs for each model. Logs:
        - Log files
        - Model pickle files
        - Model configurations
        - Feature lists (numerical and categorical)
        - Training and test metrics
        - Hyperparameters
        - Business metrics
        - Deployment decision tag
        
        :param automl_results: AutoMLResult object containing all results to log.
        :type automl_results: AutoMLResult
        :param *tags: Variable number of additional tag arguments (currently not used).
        :return: None
        :rtype: None
        
        .. note::
            - Creates nested runs for each model in the results
            - Errors during artifact logging are caught and logged but don't stop the process
            - Business metrics are logged if available
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            result = AutoMLResult(group_name="Titanic_Model_v1")
            # ... populate result with data ...
            wrapper = MLFlowWrapper(experiment_name="Titanic", group_name="Titanic_Model_v1")
            wrapper.log_results(result)
            # Results logged to MLFlow with nested runs for each model
        """
        logger.debug('Exporting results to MLFlow')
        with mlflow.start_run(run_name=self.group_name + str(automl_results.run_time['start']), ):
            log = os.path.join(self.results_path, "log.log")
            report = os.path.join(self.results_path, "automl_report.html")
            pickle_model = os.path.join(self.results_path, automl_results.result_pickle_name)

            mlflow.log_artifact(log)
            mlflow.log_artifact(report)
            mlflow.log_artifact(pickle_model)
            mlflow.set_tag(key='Deployment_decision', value=automl_results.deployment)
            #mlflow.set_tags()
            try:
                mlflow.log_artifact(os.path.join(self.results_path, automl_results.all_models_config))
            except:
                logger.error('MLflow export||No model config')
            if automl_results.compare_business_metric is not None:
                if automl_results.compare_business_metric['difference'] is not None:
                    business_metric = {'business_metric': automl_results.compare_business_metric['difference']}
                    mlflow.log_metrics(business_metric)
                threshold = {'threshold': automl_results.compare_business_metric['first_model']['threshold']}
                mlflow.log_metrics(threshold)

            for model_name in automl_results.ds_manager_result.keys():

                with mlflow.start_run(run_name=model_name, nested=True):

                    model_config = os.path.join(self.results_path, self.group_name, model_name,
                                                f"{model_name}_model_config.json")
                    features_num = os.path.join(self.results_path, self.group_name, model_name,
                                                f"{model_name}_features_numerical.json")
                    features_cat = os.path.join(self.results_path, self.group_name, model_name,
                                                f"{model_name}_features_categorical.json")
                    model_plot = os.path.join(self.results_path,
                                                f"{model_name}.html")
                    model = os.path.join(self.results_path, self.group_name, model_name, f"{model_name}_model.pickle")


                    metrics_train = automl_results.ds_manager_result[model_name].metrics['train']['full']
                    metrics_test = automl_results.ds_manager_result[model_name].metrics['test']['full']
                    mlflow.log_metrics({f"train_{k}": v for k, v in metrics_train.items()})
                    mlflow.log_metrics({f"test_{k}": v for k, v in metrics_test.items()})

                    mlflow.log_artifact(features_num)  # модель
                    mlflow.log_artifact(features_cat)
                    mlflow.log_artifact(model)
                    mlflow.log_artifact(model_plot)
                    mlflow.log_artifact(model_config)
                    try:
                        mlflow.log_params(dict(automl_results.new_hp[model_name]))
                    except:
                        logger.info('No parameters to log')



class RetroFS(RetroDataset):
    """Retrospective feature selection class.
    
    Extends RetroDataset to provide feature selection functionality for
    retrospective analysis. Used to determine which features should be
    researched and included in models.
    
    :param retro_columns: List of column names to use for retrospective analysis.
    :type retro_columns: list
    
    :var retro_columns: List of column names for retrospective analysis.
    :var retro_data: DataFrame for storing retrospective data.
    
    .. rubric:: Examples
    
    .. code-block:: python
    
        retro_fs = RetroFS(retro_columns=['feature1', 'feature2', 'feature3'])
        retro_fs.load_retro_data()
    """
    def __init__(self, retro_columns: list):
        """Initialize RetroFS instance.
        
        :param retro_columns: List of column names to use for retrospective analysis.
        :type retro_columns: list
        """
        super().__init__()
        self.retro_columns = retro_columns

    def load_retro_data(self, *params):
        """Load retrospective data.
        
        Initializes the retro_data DataFrame with the specified columns.
        Currently creates an empty DataFrame with the column structure.
        
        :param *params: Variable number of additional parameters (currently not used).
        :return: None
        :rtype: None
        
        .. note::
            This is a placeholder method that creates an empty DataFrame.
            Override in subclasses for custom data loading logic.
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            retro_fs = RetroFS(retro_columns=['feature1', 'feature2'])
            retro_fs.load_retro_data()
            # retro_data is now an empty DataFrame with columns ['feature1', 'feature2']
        """
        self.retro_data = pd.DataFrame(columns=self.retro_columns)


class AutoMLManager(DataSetsManager):
    """Main class for conducting AutoML pipeline.
    
    Orchestrates the complete AutoML workflow including feature selection,
    hyperparameter tuning, model training, comparison with previous models,
    deployment decisions, and result logging. Extends DataSetsManager to
    leverage dataset management functionality.
    
    :param auto_ml_config: Path to AutoML configuration file or AutoMLConfig object.
    :type auto_ml_config: str or AutoMLConfig
    :param models_config: Path to models configuration file or AllModelsConfig object
        defining all models to be trained.
    :type models_config: str or AllModelsConfig
    :param extractor: Optional interface for fetching and transforming input data.
        Defaults to None.
    :type extractor: Extractor, optional
    :param business_metric: Optional interface for computing business-specific metrics.
        Should return a dictionary like {'metric_name': metric_value}. Defaults to None.
    :type business_metric: BaseMetric, optional
    :param compare_business_metric: Optional interface for comparing business metrics
        and applying thresholds. Defaults to None (uses BaseCompareBusinessMetric).
    :type compare_business_metric: BaseCompareBusinessMetric, optional
    :param external_config: Optional external configuration object. Defaults to None.
    :type external_config: object, optional
    :param retro: Whether to perform retrospective feature selection. Defaults to True.
    :type retro: bool
    :param hp_tune: Whether to perform hyperparameter tuning. Defaults to True.
    :type hp_tune: bool
    :param async_mode: Whether to run model training asynchronously. Defaults to False.
    :type async_mode: bool
    :param use_temp_files: Whether to use temporary files for data processing.
        Defaults to False.
    :type use_temp_files: bool
    :param model_timeout_seconds: Maximum time in seconds for training a single model.
        Defaults to None (no timeout).
    :type model_timeout_seconds: int, optional
    :param grafana_connection: Database connection object for Grafana export.
        Passed to pd.to_sql(). Defaults to None.
    :type grafana_connection: object, optional
    :param models_dict: Optional dictionary of pre-initialized models. Defaults to None.
    :type models_dict: dict, optional
    
    :var models_dict: Dictionary of pre-initialized models.
    :var timeout: Maximum time for model training.
    :var _business_metric: Business metric calculator.
    :var _compare_business_metric: Business metric comparator.
    :var __grafana_connection: Grafana database connection.
    :var _async_mode: Whether async mode is enabled.
    :var features_list_to_exclude: List of features to exclude from selection.
    :var _auto_ml_config: AutoML configuration object.
    :var _feature_selection_config: Feature selection configuration.
    :var _hp_tuning_config: Hyperparameter tuning configuration.
    :var features_list: List of selected features.
    :var _retro: Whether retrospective analysis is enabled.
    :var _hp_tune: Whether hyperparameter tuning is enabled.
    :var automl_results: AutoMLResult object storing all results.
    :var mlflow: MLFlowWrapper instance for logging.
    :var status: Dictionary tracking completion status of each stage.
    :var errors: Dictionary tracking errors for each stage.
    
    .. rubric:: Examples
    
    .. code-block:: python
    
        from outboxml.automl_manager import AutoMLManager
        automl = AutoMLManager(
            auto_ml_config="configs/automl-config.json",
            models_config="configs/models-config.json",
            external_config=config,
            retro=True,
            hp_tune=True
        )
        result = automl.update_models()
        print(result.deployment)  # Check deployment decision
        """

    def __init__(self,
                 auto_ml_config,
                 models_config,
                 extractor: Optional[Extractor] = None,
                 business_metric: BaseMetric = None,
                 compare_business_metric: BaseCompareBusinessMetric=None,
                 external_config=None,
                 retro: bool = True,
                 hp_tune: bool = True,
                 async_mode: bool = False,
                 use_temp_files: bool = False,
                 model_timeout_seconds: int = None,
                 grafana_connection=None,
                 models_dict: dict=None
                 ):
        super().__init__(config_name=models_config, extractor=extractor,
                         external_config=external_config, use_temp_files=use_temp_files)
        self.models_dict = models_dict
        self.timeout = model_timeout_seconds
        self._business_metric = business_metric
        self.__default_models_config = models_config
        self._compare_business_metric = compare_business_metric
        if self._compare_business_metric is None:
            self._compare_business_metric = BaseCompareBusinessMetric()
        self.__grafana_connection = grafana_connection
        self._async_mode = async_mode
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
                                    results_path=self._external_config.results_path,
                                    mlflow_tracking_uri=self._external_config.mlflow_tracking_uri)
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


    def update_models(self, send_mail: bool = False, parameters_for_optuna: dict = None):
        """Execute the complete AutoML pipeline to update models.
        
        Orchestrates the full AutoML workflow:
        1. Feature selection (if retro=True)
        2. Hyperparameter tuning (if hp_tune=True)
        3. Model training
        4. Result saving
        5. Comparison with previous models
        6. Deployment decision
        7. Review (email or HTML report)
        8. MLFlow logging
        
        :param send_mail: Whether to send email notifications. If False, generates
            HTML report instead. Defaults to False.
        :type send_mail: bool
        :param parameters_for_optuna: Optional dictionary mapping model names to
            custom Optuna parameter functions for hyperparameter tuning.
            Defaults to None.
        :type parameters_for_optuna: dict, optional
        :return: AutoMLResult object containing all results, metrics, and metadata.
        :rtype: AutoMLResult
        
        .. note::
            - Errors are caught and logged, but results are still saved to MLFlow
            - Status dictionary is updated throughout the process
            - Execution times are recorded for each stage
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            automl = AutoMLManager(...)
            result = automl.update_models(send_mail=True)
            print(f"Deployment: {result.deployment}")
            print(f"New features: {result.new_features}")
        """
        self._init_logger()
        email = AutoMLReviewEMail(config=self._external_config)
        try:

            self.status['Loading dataset'] = True
            self.feature_selection()

            if self._hp_tune:
                new_hp = self.hp_tuning(parameters_for_optuna)
                self.automl_results.new_hp = new_hp
                self.__update_hyperparameters(new_hp)
                self.status['HP tuning'] = True
                self.automl_results.run_time['hp_tuning'] = datetime.now()
            results = self.fit_models(models_dict=self.models_dict)
            self.status['Fitting'] = True
            self.automl_results.run_time['fitting'] = datetime.now()
            self.save_results(self._results)
            self.compare_with_previous()
            self.deployment()
            self.review(email, send_mail, )
            self.mlflow.log_results(self.automl_results)
            self.automl_results.run_time['export results'] = datetime.now()
            self.status['Loading results to MLFLow'] = True
        except Exception as exc:
            logger.error(str(exc))
            try:

                self.mlflow.log_results(self.automl_results)
                self.automl_results.run_time['export results'] = datetime.now()
                self.status['Loading results to MLFLow'] = True
            except Exception as exc2:
                logger.error(str(exc2))
            finally:
                if send_mail:
                    email.error_mail(group_name=self.group_name,
                                     error=exc, status=self.status,
                                     )
        finally:
          logger.debug('Updating models is finished||'+str(self.status))
        return self.automl_results

    def feature_selection(self):
        """Perform feature selection for all models.
        
        Conducts retrospective feature selection if enabled. For each model,
        determines which features should be included based on retrospective
        analysis and feature selection criteria. Saves selected feature subsets
        for later use.
        
        :return: None
        :rtype: None
        
        .. note::
            - Only executes if retro=True
            - Uses RetroFS for retrospective analysis
            - Saves feature subsets to pickle files for each model
            - Updates automl_results.new_features with selected features
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            automl = AutoMLManager(..., retro=True)
            automl.feature_selection()
            print(automl.automl_results.new_features)
            # {'model1': ['feature1', 'feature2', ...], ...}
        """
        if self._retro:
            logger.debug('Feature selection||Started')
            data = self.dataset
            features_for_research = RetroFS(retro_columns=data.columns
                                            ).features_for_reserch(data_column_names=data.columns,
                                                                   target_columns_names=self.targets_columns_names,
                                                                   models_config=self._models_configs,
                                                                   extra_columns=self._data_preprocessor._extra_columns,
                                                                   features_list_to_exclude=self.features_list_to_exclude,
                                                                   )
            self.automl_results.features_for_research = features_for_research

            for model in self._models_configs:

                feature_selector = BaseFS(data_preprocessor=self._data_preprocessor,
                                          parameters=self._feature_selection_config,
                                          prepare_data_interface = FeatureSelectionPrepareDataset(model_config=model),
                                          new_features_list=features_for_research,
                                          feature_selection_interface=FeatureSelectionInterface(
                                              feature_selection_config=self._feature_selection_config,
                                              objective=model.objective)
                                          )

                new_prepared_data = feature_selector.select_features(params=self._feature_selection_config.params,
                                                                     model_name=model.name)
                self._data_preprocessor.save_subset_to_pickle(model_name=model.name, data_subset=new_prepared_data, rewrite=True)

                logger.info('Result subset saving to version||' + self._data_preprocessor._version)
                self._data_preprocessor._use_saved_files = True
                self.automl_results.new_features[model.name] = feature_selector.result_features

            logger.debug('Feature selection||Finished')
            self.status['Feature selection'] = True
            self.automl_results.run_time['retro'] = datetime.now()

    def hp_tuning(self, parameters_for_optuna: dict = None):
        """Perform hyperparameter tuning for all models.
        
        Uses Optuna to find optimal hyperparameters for each model through
        cross-validation. Supports custom parameter search spaces via
        parameters_for_optuna.
        
        :param parameters_for_optuna: Optional dictionary mapping model names to
            custom Optuna parameter functions. Each function should define the
            search space for that model. Defaults to None.
        :type parameters_for_optuna: dict, optional
        :return: Dictionary mapping model names to optimized hyperparameters.
            Empty dictionary for models that failed tuning.
        :rtype: dict
        
        .. note::
            - Errors during tuning are logged but don't stop the process
            - Uses the configured sampler, scoring function, and CV folds
            - Respects model_timeout_seconds if set
            - Returns empty dict for models that fail tuning
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            def custom_params(trial):
                return {'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3)}
            parameters = {'model1': custom_params}
            new_hp = automl.hp_tuning(parameters_for_optuna=parameters)
            print(new_hp['model1'])
            # {'learning_rate': 0.15, ...}
        """
        new_hp = {}
        trials = self._hp_tuning_config.trials
        for model in self._models_configs:
            new_hp[model.name] = {}
            try:
                parameters_for_optuna_func = None
                if self._hp_tuning_config.parameters:
                    logger.info(f"HP_Tune || Use parameters from config")
                    try:
                        model_params = self._hp_tuning_config.parameters[model.name]
                        parameters_for_optuna_func = lambda trial: self.__sample_parameters(trial, model_params)
                    except KeyError:
                        logger.warning(f'HP_Tune || No parameters for model in config {model.name}')
                        continue
                elif parameters_for_optuna is not None:
                    logger.info(f"HP_Tune || Use parameters from func")
                    try:
                        parameters_for_optuna_func = parameters_for_optuna[model.name]
                    except KeyError:
                        logger.warning(f'HP_Tune || No parameters for model "{model.name}"')
                new_hp[model.name] = HPTuning(data_preprocessor=self._data_preprocessor,
                                              sampler=self._hp_tuning_config.sampling,
                                              scoring_fun=self._hp_tuning_config.metric_score[model.name],
                                              folds_num_for_cv=self._hp_tuning_config.cv_folds_num,
                                              objective=model.objective,
                                              random_state=self.config.data_config.separation.random_state,
                                              work_type=self._work_type_hptune,
                                              ).best_params(model_name=model.name,
                                                            parameters_for_optuna_func=parameters_for_optuna_func,
                                                            timeout=self.timeout,
                                                            trials=trials)
                logger.info(new_hp[model.name])

            except Exception as exc:
                self.errors['HP tuning'] = exc
                logger.error(str(exc))
                logger.info('Returning {}')
        return new_hp

    def __sample_parameters(self, trial: optuna.Trial, config) -> Dict[str, Any]:
        sampled_params = {}

        for name, spec in config.items():
            if spec.type == "int":
                sampled_params[name] = trial.suggest_int(
                    name, int(spec.low), int(spec.high), step=spec.step, log=spec.log
                )
            elif spec.type == "float":
                sampled_params[name] = trial.suggest_float(
                    name, spec.low, spec.high, step=spec.step, log=spec.log
                )
            elif spec.type == "categorical":
                sampled_params[name] = trial.suggest_categorical(name, spec.choices)

        return sampled_params

    def save_results(self, results: dict):
        """Save model results to disk and export to Grafana.
        
        Saves all model results including:
        - Pickle files for service deployment
        - Model configurations as JSON
        - Artifacts via ResultExport
        - Metrics to Grafana database (if connection provided)
        
        :param results: Dictionary mapping model names to DSManagerResult objects.
        :type results: dict
        :return: None
        :rtype: None
        
        .. note::
            - Creates timestamped pickle files for production deployment
            - Exports metrics to Grafana if grafana_connection is provided
            - Errors during Grafana export are logged but don't stop the process
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            results = automl.fit_models()
            automl.save_results(results)
            # Results saved to disk and exported to Grafana
        """
        models_results = []
        self.automl_results.ds_manager_result = results

        for key in results:
            models_results.append(results[key].dict_for_prod_export())

        # артефакты и метрики
        logger.debug('Saving pickle for service')
        saving_start_time = datetime.now()
        group_name = self.group_name
        ds_manager_for_save = DataSetsManager(config_name=self.__default_models_config, external_config=self._external_config)
        ds_manager_for_save.load_dataset(data=self.dataset)
        ds_manager_for_save._results = self._results
        ds_manager_for_save.group_name = group_name
        ResultExport(ds_manager=ds_manager_for_save, config=self._external_config).save(to_mlflow=False,
                                                                                        path_to_save=self._external_config.results_path,
                                                                                        to_pickle=True)
        # Сохранение пикла для сервиса локально
        result_pickle_name = ResultPickle().generate_name(group_name, saving_start_time)
        self.automl_results.result_pickle_name = result_pickle_name
        self.automl_results.all_models_config = f"{group_name}_{saving_start_time.strftime('%Y_%m_%d_%H_%M_%S')}.json"

        with open(os.path.join(self._external_config.results_path, result_pickle_name), "wb") as f:
            pickle.dump(models_results, f)
        self._save_model_json()

        try:
            logger.debug('Loading metrics to grafana')
            df = ResultExport(ds_manager=ds_manager_for_save).grafana_export(project_name=group_name,
                                                                             date_time=datetime.now())

            GrafanaExport(df=df, table_name=self._auto_ml_config.grafana_table_name,
                          connection=self.__grafana_connection).load_data_to_db()
        except Exception as exc:
            logger.error('grafana export error||' + str(exc))

    def compare_with_previous(self):
        """Compare current models with previous versions.
        
        Loads the last saved model results and compares them with current results.
        Generates comparison metrics, plots, and business metric comparisons.
        
        :return: None
        :rtype: None
        
        .. note::
            - If no previous model exists, comparison is skipped
            - Generates comparison plots for visualization
            - Calculates business metric differences if compare_business_metric is configured
            - Updates automl_results with comparison data
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            automl.compare_with_previous()
            print(automl.automl_results.compare_metrics_df)
            # DataFrame comparing current vs previous model metrics
        """
        try:
            result_to_compare = self._load_last_result()
        except:
            logger.error('No model to compare')
            result_to_compare = None

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
        self.automl_results.run_time['comparing models'] = datetime.now()

    def deployment(self):
        """Make deployment decision based on quality criteria.
        
        Evaluates whether models meet the configured quality thresholds for
        deployment. Checks both business metrics and comparison business metrics
        against configured thresholds. Sets deployment flag to True only if all
        criteria are met.
        
        :return: None
        :rtype: None
        
        .. note::
            - Deployment is set to True only if ALL criteria are met
            - Uses metric_growth_value from inference_criteria config
            - For CompareBusinessMetric, checks if difference exceeds threshold
            - For other metrics, checks if metric value is better than threshold
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            automl.deployment()
            print(automl.automl_results.deployment)
            # True if all criteria met, False otherwise
        """
        self.automl_results.deployment = False
        metrics = self._auto_ml_config.inference_criteria.metric_growth_value
        res = []
        if metrics is not None:
            for key in metrics.keys():
                self.automl_results.deployment = False
                try:
                    if key == 'CompareBusinessMetric':
                        if self.automl_results.compare_business_metric is not None:
                            logger.info('Current compare business metric value: ' + str(
                                self.automl_results.compare_business_metric['difference']))
                            if self.automl_results.compare_business_metric['difference'] is None:
                                res.append(True)
                            elif self.automl_results.compare_business_metric['difference'] > metrics[key]:
                                res.append(True)
                            else:
                                res.append(False)
                    else:
                        logger.info(
                            'Current business metric value: ' + str(self.automl_results.compare_metrics_df[key].min()))
                        if metrics[key] < self.automl_results.compare_metrics_df[key].min():
                            res.append(True)
                        else:
                            res.append(False)
                        self.automl_results.business_metric[key] = self.automl_results.compare_metrics_df[key].min()

                except KeyError:
                    logger.error('Business metric error to complete decision for deployment')

        if all(res): self.automl_results.deployment = True
        self.status['Deployment decision'] = True
        logger.info('Deployment decision: ' + str(self.automl_results.deployment))

    def review(self, email: EMail, send_mail: bool, error: Exception = None):
        """Generate review report (email or HTML) for AutoML execution.
        
        Creates and sends either an email notification or generates an HTML report
        based on the send_mail flag. Includes success reports or error reports
        depending on execution status.
        
        :param email: Email instance for sending notifications.
        :type email: EMail
        :param send_mail: Whether to send email. If False, generates HTML report instead.
        :type send_mail: bool
        :param error: Optional exception that occurred during execution. If provided,
            generates error report. Defaults to None.
        :type error: Exception, optional
        :return: None
        :rtype: None
        
        .. note::
            - If send_mail=True and no error: sends success email
            - If send_mail=True and error: sends error email
            - If send_mail=False and no error: generates HTML success report
            - If send_mail=False and error: generates HTML error report
            - Optionally sends release notification if models were released to Git
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            email = AutoMLReviewEMail(config=config)
            automl.review(email, send_mail=True)
            # Sends email notification
        """
        if send_mail:
            if error is not None:
                email.common_error_mail(group_name=self.group_name, error=str(error))
            else:
                email.success_mail(self.automl_results)
                if self.status['Loading pickle to git']:
                    EMail(self._external_config).success_release_mail(self.automl_results.result_pickle_name,
                                                                      new_features=self.automl_results.new_features)
            self.status['EMail Review'] = True
        else:
            if error is not None:
                HTMLReport(self._external_config).error_report(group_name=self.group_name,
                                                               error=str(error),
                                                               status=self.status)
            else:
                HTMLReport(self._external_config).success_report(self.automl_results)


    def __init_auto_ml(self, ):
        """Initialize AutoML configuration and settings.
        
        Loads and validates the AutoML configuration from either a file path
        or dictionary. Initializes DataSetsManager and sets up feature selection
        and hyperparameter tuning configurations.
        
        :return: None
        :rtype: None
        
        :raises FileNotFoundError: If config file path is invalid.
        :raises ValidationError: If config validation fails.
        
        .. note::
            This is a private method, called automatically during __init__.
            Supports both file paths and dictionary configurations.
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            # Called automatically during initialization
            automl = AutoMLManager(auto_ml_config="config.json", ...)
            # Config loaded and validated
        """
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
        self._init_dsmanager()
        self._is_initialized = True
        self.group_name = f"{self._auto_ml_config.group_name}_{self._auto_ml_config.project}"
        self._feature_selection_config = self._auto_ml_config.feature_selection
        self._hp_tuning_config = self._auto_ml_config.hp_tune  # use best model = False
        if self._retro:
            self.features_list_to_exclude = self.__load_features_list()

        logger.debug('AutoML init completed')

    def __load_features_list(self):
        """Load list of features to exclude from feature selection.
        
        Retrieves the list of features that should be ignored during feature
        selection from the feature selection configuration.
        
        :return: List of feature names to exclude.
        :rtype: list
        
        .. note::
            This is a private method, called internally by __init_auto_ml().
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            # Called automatically during initialization
            features_to_exclude = automl._AutoMLManager__load_features_list()
        """
        return self._feature_selection_config.features_to_ignore

    def _load_last_result(self) -> dict:
        """Load the last saved model results for comparison.
        
        Loads the most recent pickle file containing previous model results
        and generates predictions on the current dataset for comparison.
        
        :return: Dictionary mapping model names to DSManagerResult objects
            from the previous run. Empty dict if loading fails.
        :rtype: dict
        
        .. note::
            This is a private method, typically called internally by compare_with_previous().
            Errors during loading are logged but don't raise exceptions.
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            previous_results = automl._load_last_result()
            # Returns dict with previous model results or empty dict if none found
        """
        model_to_compare = load_last_pickle_models_result(self._external_config, self.group_name)
        result_to_compare = {}
        try:
            for key in model_to_compare.keys():

                models = model_to_compare[key]

                for model_result in models:
                    model_name = model_result['model_config']['name']
                    result_to_compare[model_name] = self.model_predict(data=self.dataset,
                                                                       model_name=model_name,
                                                                       model_result=model_result)
        except Exception as exc:
            logger.error(exc)
            logger.info('Cannot get results for last model||' + str(exc))
        return result_to_compare

    def _compare_base_metrics_df(self, result_to_compare: dict=None):
        """Generate comparison metrics DataFrame for all models.
        
        Creates a DataFrame comparing metrics between current and previous models,
        or just current model metrics if no previous model exists.
        
        :param result_to_compare: Optional dictionary mapping model names to
            previous DSManagerResult objects. If None, only current metrics are included.
        :type result_to_compare: dict, optional
        :return: DataFrame with metrics comparison. Columns include metric names,
            model names, and train/test indicators.
        :rtype: pandas.DataFrame
        
        .. note::
            This is a private method, typically called internally by compare_with_previous().
            Errors during metric calculation are logged but don't stop the process.
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            previous_results = automl._load_last_result()
            metrics_df = automl._compare_base_metrics_df(result_to_compare=previous_results)
            # Returns DataFrame comparing current vs previous metrics
        """
        df = pd.DataFrame()
        for key in self._results.keys():
            model_config = self._results[key].config
            ds = DataSetsManager(config_name=self.__default_models_config)
            ds._all_models_config = model_config
            res_export = ResultExport(ds_manager=ds)
            res_export.result = self._results
            try:
                if result_to_compare is not None:
                    metrics_df = res_export.compare_metrics(model_name=key,
                                                            ds_manager_result=result_to_compare,
                                                            business_metric=self._business_metric,
                                                            only_main=True)
                else:
                    metrics_df = pd.concat([res_export.metrics_df(model_name=key,
                                                       train_test='train'),
                                            res_export.metrics_df(model_name=key,
                                                                  train_test='test'),
                                            ])
                metrics_df['Имя модели'] = key
                df = pd.concat([df, metrics_df])
            except Exception as exc:
                logger.error(exc)
        return df

    def _compare_plots(self, result_to_compare: dict):
        """Generate comparison plots for current vs previous models.
        
        Creates Plotly figures comparing current model performance with previous
        models, or standalone plots if no previous model exists.
        
        :param result_to_compare: Optional dictionary mapping model names to
            previous DSManagerResult objects. If None, generates standalone plots.
        :type result_to_compare: dict, optional
        :return: Dictionary mapping model names to Plotly figure objects.
        :rtype: dict
        
        .. note::
            This is a private method, typically called internally by compare_with_previous().
            - If result_to_compare is provided: generates comparison plots
            - If result_to_compare is None: generates standalone plots
            - Uses cohort analysis (plot_type=2) with exposure
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            previous_results = automl._load_last_result()
            figures = automl._compare_plots(result_to_compare=previous_results)
            # Returns dict of Plotly figures for each model
        """
        figures = {}
        for key in self._results.keys():
            y_graph, features_categorical, features_numerical = DataframeForPlots().df_for_plots(
                result=self._results[key],
                use_exposure=True)
            if result_to_compare is not None:
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
            else:
                figures[key] = MLPlot(model_name_1=key, y_graph=y_graph, features_categorical=features_categorical,
                                                 features_numerical=features_numerical,
                                                use_exposure=True,
                                                 show=False).make(plot_type=2,
                                                                  cut_min_value=0.1,
                                                                  cut_max_value=0.9,
                                                                  samples=100,
                                                                  cohort_base='model')


        return figures

    def __update_hyperparameters(self, new_hp: dict):
        """Update model configurations with optimized hyperparameters.
        
        Updates the hyperparameters in model configurations based on the results
        from hyperparameter tuning. Supports CatBoost and GLM models.
        
        :param new_hp: Dictionary mapping model names to optimized hyperparameters.
        :type new_hp: dict
        :return: None
        :rtype: None
        
        .. note::
            This is a private method, typically called internally by update_models().
            Only updates parameters for supported model types (CatBoost, GLM).
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            new_hp = {'model1': {'learning_rate': 0.1, 'depth': 6}}
            automl._AutoMLManager__update_hyperparameters(new_hp)
            # Model configurations updated with new hyperparameters
        """
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

    def _save_model_json(self):
        """Save model configuration as JSON to pickle file.
        
        Serializes the complete model configuration (including all models and
        features) to a JSON string and saves it as a pickle file. Ensures all
        mapping keys are strings for JSON compatibility.
        
        :return: None
        :rtype: None
        
        .. note::
            This is a private method, typically called internally by save_results().
            Converts non-string keys in feature mappings to strings for JSON compatibility.
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            # Called automatically during save_results()
            automl._save_model_json()
            # Configuration saved to pickle file
        """
        config_to_save = deepcopy(self.config)
        for model_config in config_to_save.models_configs:
            for feature in model_config.features:
                if feature.mapping is not None:
                    if isinstance(feature.mapping, dict):
                        if not all(isinstance(key, str) for key in feature.mapping):
                            feature.mapping = {str(key): value for key, value in feature.mapping.items()}
        with open(os.path.join(self._external_config.results_path, self.automl_results.all_models_config), "wb") as f:
            pickle.dump(config_to_save.json(), f)

    def _init_logger(self):
        """Initialize logger and handle log file rotation.
        
        Sets up logging to a log file in the results path. If a log file already
        exists, it is renamed with a timestamp to preserve previous logs. Handles
        cases where the file is locked by another process.
        
        :return: None
        :rtype: None
        
        .. note::
            This is a private method, typically called internally by update_models().
            - Renames existing log.log to log_<timestamp>.log
            - Handles file locking gracefully
            - Creates new log.log for current session
            
        .. rubric:: Examples
        
        .. code-block:: python
        
            # Called automatically during update_models()
            automl._init_logger()
            # Logger configured to write to results_path/log.log
        """
        log_path = Path(str(self._external_config.results_path)) / 'log.log'

        if log_path.exists():
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            new_name = f"log_{timestamp}.log"
            try:
                os.rename(log_path, log_path.parent / new_name)
            except (PermissionError, OSError) as e:
                # File is used by another process, create a new log file with a unique name
                logger.warning(f"Failed to rename log.log: {e}. Creating a new file with timestamp.")
                new_log_path = log_path.parent / new_name
                # If a file with this name already exists, add an additional suffix
                counter = 1
                while new_log_path.exists():
                    new_name = f"log_{timestamp}_{counter}.log"
                    new_log_path = log_path.parent / new_name
                    counter += 1
                # Try to copy contents if possible
                try:
                    shutil.copy2(log_path, new_log_path)
                except:
                    pass  # If copying fails, just continue

        logger.add(Path(str(self._external_config.results_path) + '/log.log'))
