import json
import os
import pickle
from copy import deepcopy
from itertools import chain
from pathlib import Path

from pydantic import ValidationError
from sklearn.base import is_classifier
import pandas as pd
import polars as pl
import numpy as np
from loguru import logger
from typing import List, Dict, Any, Optional, Union, Literal
from sklearn.preprocessing import LabelEncoder

from outboxml.feature_importance import FeatureImportance
from outboxml.monitoring_result import DataContext
from outboxml.core.enums import ModelsParams
from outboxml.data_subsets import DataPreprocessor, ModelDataSubset
from outboxml.dataset_retro import RetroDataset
from outboxml.datadrift import DataDrift
from outboxml.core.data_prepare import prepare_dataset
from outboxml.core.pydantic_models import AllModelsConfig, DataModelConfig, ModelConfig
from outboxml.extractors import Extractor, BaseExtractor, SimpleExtractor
from outboxml.metrics.base_metrics import BaseMetric, BaseMetrics
from outboxml.core.prepared_datasets import PrepareDataset, TrainTestIndexes, PrepareDatasetPl
from outboxml.metrics.processor import ModelMetrics
from outboxml.ensemble import resolve_model_reference
from outboxml.models import DefaultModels
from outboxml import config


class DSManagerResult:
    """Container class for model results.

    Stores all information about a trained model including the model object,
    data subsets, predictions, metrics, and configurations.

    :param model_name: Name of the model.
    :type model_name: str
    :param model: Trained model object.
    :type model: Any
    :param data_subset: Object containing train/test data subsets with
        X_train/test, y_train/test vectors, numerical and categorical feature
        names, and exposure vectors.
    :type data_subset: ModelDataSubset
    :param model_config: Model configuration object.
    :type model_config: ModelConfig, optional
    :param config: Configuration file with source data settings.
    :type config: AllModelsConfig, optional
    :param predictions: Dictionary with 'train' and 'test' keys containing predictions.
    :type predictions: dict, optional
    :param metrics: Dictionary with 'train' and 'test' keys containing metrics.
    :type metrics: dict, optional

    .. rubric:: Methods

    - :meth:`dict_for_prod_export` - Returns dictionary for creating pickle file for service
    - :meth:`from_pickle_model_result` - Converter from pickle service dictionary to object (class method)

    .. rubric:: Properties

    - :attr:`X` - Feature vector X
    - :attr:`y_pred` - Predictions
    - :attr:`y` - True target values (y_true)
    - :attr:`exposure` - Exposure vector
    """

    def __init__(self,
                 model_name: str,
                 model: Any,
                 data_subset: ModelDataSubset,
                 model_config: ModelConfig = None,
                 config: AllModelsConfig = None,
                 predictions: dict = None,
                 metrics: dict = None,
                 feature_importance: FeatureImportance = None
                 ):
        """Initialize DSManagerResult instance.
        
        :param model_name: Name of the model.
        :type model_name: str
        :param model: Trained model object.
        :type model: Any
        :param data_subset: Object containing train/test data subsets.
        :type data_subset: ModelDataSubset
        :param model_config: Configuration object for the model.
        :type model_config: ModelConfig, optional
        :param config: Configuration object for all models.
        :type config: AllModelsConfig, optional
        :param predictions: Dictionary with 'train' and 'test' keys containing predictions.
        :type predictions: dict, optional
        :param metrics: Dictionary with 'train' and 'test' keys containing metrics.
        :type metrics: dict, optional
        """
        if predictions is None:
            self.predictions = {'train': None, 'test': None}
        else:
            self.predictions = predictions
        self.model_name = model_name
        self.config = config
        if metrics is None:
            self.metrics = {'train': None, 'test': None}
        else:
            self.metrics = metrics
        self.model = model
        self.data_subset = data_subset
        self.model_config = model_config
        self.feature_importance = feature_importance

    def load_metrics(self, metrics: dict, ds_type: str = None):
        """Load metrics into the result object.
        
        :param metrics: Dictionary containing metrics to load.
        :type metrics: dict
        :param ds_type: Dataset type ('train' or 'test'). If None, replaces all metrics.
        :type ds_type: str, optional
        """
        if ds_type is not None:
            self.metrics[ds_type] = metrics
        else:
            self.metrics = metrics

    def load_predictions(self, df: Union[pd.DataFrame, pd.Series], ds_type: str):
        """Load predictions into the result object.
        
        :param df: DataFrame or Series containing predictions.
        :type df: Union[pd.DataFrame, pd.Series]
        :param ds_type: Dataset type ('train' or 'test').
        :type ds_type: str
        """
        self.predictions[ds_type] = df

    def dict_for_prod_export(self, ):
        """Convert DSManagerResult to dictionary for pickle export to production service.
        
        :return: Dictionary containing model configuration, model object, and feature lists.
        :rtype: dict
        
        .. rubric:: Examples
        
        >>> result = DSManagerResult(...)
        >>> export_dict = result.dict_for_prod_export()
        >>> export_dict.keys()
        dict_keys(['model_config', 'model', 'min_max_scaler', 'features_numerical', 'features_categorical'])
        """
        model = self.model
        model_results = {
            "model_config": self.model_config.model_dump(),
            "model": model,
            "min_max_scaler": None,
            "features_numerical": self.data_subset.features_numerical,
            "features_categorical": self.data_subset.features_categorical,
        }

        return model_results

    @classmethod
    def from_pickle_model_result(cls, model_result: dict, all_model_config: AllModelsConfig, ):
        """Convert pickle dictionary to DSManagerResult object.
        
        Uses library model wrapper format.
        
        :param model_result: Dictionary loaded from pickle file containing model data.
        :type model_result: dict
        :param all_model_config: Configuration object for all models.
        :type all_model_config: AllModelsConfig
        :return: DSManagerResult instance created from pickle data.
        :rtype: DSManagerResult
        
        .. rubric:: Examples
        
        >>> with open('model.pickle', 'rb') as f:
        ...     model_result = pickle.load(f)
        >>> result = DSManagerResult.from_pickle_model_result(model_result, all_models_config)
        """

        model_config = model_result['model_config']
        model_name = model_result['model_config']['name']
        model_config = ModelConfig.model_validate(model_config)
        model = model_result['model']
        features_numerical = model_result['features_numerical']
        features_categorical = model_result['features_categorical']
        return cls(model_name=model_name,
                   config=all_model_config,
                   data_subset=ModelDataSubset(model_name=model_name,
                                               features_numerical=features_numerical,
                                               features_categorical=features_categorical,
                                               ),
                   model=model,
                   model_config=model_config)

    @property
    def X(self):
        """Get combined feature matrix from train and test sets.
        
        :return: DataFrame containing all features.
        :rtype: pd.DataFrame
        """
        return pd.concat([self.data_subset.X_train, self.data_subset.X_test])

    @property
    def y(self):
        """Get combined target values from train and test sets.
        
        :return: Series containing all target values.
        :rtype: pd.Series
        """
        return pd.concat([self.data_subset.y_train, self.data_subset.y_test])

    @property
    def y_pred(self):
        """Get combined predictions from train and test sets.
        
        :return: Series containing all predictions.
        :rtype: pd.Series
        """
        y_pred = pd.concat([self.predictions['train'], self.predictions['test']])
        return y_pred

    @property
    def exposure(self):
        """Get combined exposure values from train and test sets.
        
        :return: Series containing all exposure values.
        :rtype: pd.Series
        """
        exposure = pd.concat([self.data_subset.exposure_train, self.data_subset.exposure_test])
        return exposure


class DataSetsManager:
    """Main class for working with models.

    This is the core class for managing datasets, model training, and evaluation
    in the OutBoxML framework. It handles data loading, preprocessing, model
    fitting, prediction, and result management.

    For framework usage, a properly configured config file is required.
    Out-of-the-box operation is performed using config parameters:
    ``DataSetsManager(config_name=config)``.

    For custom framework configuration, import the following modules:
    ::

        from outboxml.extractors import Extractor
        from outboxml.metrics.base_metrics import BaseMetric
        from outboxml.models import BaseWrapperModel
        from outboxml.dataset_retro import RetroDataset
        from outboxml.export_results import ResultExport

    Framework usage starts with creating an object with parameters.
    The required parameter is: path to config file or validated config file
    (AllModelsConfig). Other parameters can be set automatically "out of the box".
    Work information is output as a log file.
    Modeling results are output in the DSManagerResult container.

    :param config_name: Path to config file or validated config file (AllModelsConfig).
    :type config_name: Union[str, Dict]
    :param extractor: User-defined extractor object inheriting from Extractor interface.
        Main method ``extract_dataset()`` should return pandas DataFrame.
        Extractor should contain ``check_object()`` method with data validation
        and verification. Use library RTDMExtractor or ActuarExtractor for
        working with databases.
    :type extractor: Optional[Extractor]
    :param prepared_datasets: Dictionary ``{name: PreparedDataset}`` with model
        preparation objects. Wrapper of ``prepare_dataset`` function.
        PreparedDataset by default uses model_config features and has no prep
        and post prep functions.
    :type prepared_datasets: Optional[Dict[str, PrepareDataset]]
    :param models_dict: Dictionary ``{name: Model}`` with models for training
        and prediction inheriting from Model class. Class should have ``fit()``,
        ``predict()`` methods. By default model is chosen by group and project
        name. You can import models from library.
    :type models_dict: Optional[Dict]
    :param business_metric: User-defined business metric. Should inherit from
        BaseMetric. Main method is ``calculate_metric()``.
    :type business_metric: Optional[BaseMetric]
    :param use_baseline_model: Baseline model selection. 1 - RandomForestRegressor,
        2 - DummyRegressor median; 3 - mean. Defaults to 0 (no baseline).
    :type use_baseline_model: int
    :param retro_changes: RetroDataset object for retro analysis.
    :type retro_changes: Optional[RetroDataset]
    :param external_config: External configuration object. Defaults to None.
    :type external_config: Any, optional
    :param use_temp_files: Whether to use temporary files for data processing.
        Defaults to False.
    :type use_temp_files: bool
    :param prepare_engine: Engine to use for data preparation ('pandas' or 'polars').
        Defaults to 'pandas'.
    :type prepare_engine: Literal['pandas', 'polars']

    .. rubric:: Methods

    - :meth:`load_dataset` - Load dataset from source and path in config file
        or user-defined extractor class
    - :meth:`get_trainDfs` - Return prepared data subset to train user model
    - :meth:`get_testDfs` - Return prepared data subset to test user model
    - :meth:`fit_models` - Fit, predict and get metrics for all models in model_dict
    - :meth:`get_result` - Return container of results DSManagerResult
    - :meth:`check_datadrift` - Returns dataframe with datadrift analysis result

    .. rubric:: Examples

    Example usage with Titanic dataset:

    .. code-block:: python

        # Post prep function
        def data_post_prep_func(data: pd.DataFrame):
            data["SEX"] = pd.to_numeric(data["SEX"])
            return data

        titanic_ds_manager = DataSetsManager(
            config_name=config_name,
            extractor=TitanicExampleExtractor(path_to_file=path_to_data),
            prepared_datasets={
                'first': PrepareDataset(
                    group_name='survived1',
                    data_post_prep_func=data_post_prep_func,
                    check_prepared=True,
                    calc_corr=True
                ),
                'second': PrepareDataset(group_name='survived2')
            },
            business_metric=TitanicExampleMetric()
        )
        titanic_TrainDs = titanic_ds_manager.get_trainDfs(model_name='first')
        titanic_results = titanic_ds_manager.fit_models()

    For more examples, see the ``outboxml/examples`` repository.
    """

    def __init__(
            self,
            config_name: Union[str, Dict],
            extractor: Optional[Extractor] = None,
            prepared_datasets: Optional[Dict[str, PrepareDataset]] = None,
            models_dict: Optional[Dict] = None,
            business_metric: Optional[BaseMetric] = None,
            use_baseline_model: int = 0,
            retro_changes: Optional[RetroDataset] = None,
            external_config = None,
            use_temp_files: bool = False,
            prepare_engine: Literal['pandas', 'polars'] = 'pandas',
    ):
        """Initialize DataSetsManager instance.
        
        :param config_name: Path to config file or validated config dictionary (AllModelsConfig).
        :type config_name: Union[str, Dict]
        :param extractor: User-defined extractor object inheriting from Extractor interface.
                         Main method extract_dataset() should return pandas DataFrame.
                         Extractor should contain check_object() method for data validation.
                         Use library RTDMExtractor or ActuarExtractor for database connections.
        :type extractor: Optional[Extractor]
        :param prepared_datasets: Dictionary with model names as keys and PrepareDataset objects as values.
                                 Wrapper of prepare_dataset function. PreparedDataset by default uses
                                 model_config features and has no prep and post prep functions.
        :type prepared_datasets: Optional[Dict[str, PrepareDataset]]
        :param models_dict: Dictionary with model names as keys and Model objects as values.
                           Models should inherit from Model class and have fit(), predict() methods.
                           By default model is chosen by group and project name.
        :type models_dict: Optional[Dict]
        :param business_metric: User-defined business metric inheriting from BaseMetric.
                               Main method is calculate_metric().
        :type business_metric: Optional[BaseMetric]
        :param use_baseline_model: Baseline model selection. 0 - no baseline, 1 - RandomForestRegressor,
                                   2 - DummyRegressor median, 3 - mean.
        :type use_baseline_model: int
        :param retro_changes: RetroDataset object for retro analysis.
        :type retro_changes: Optional[RetroDataset]
        :param external_config: External configuration object. If None, uses default config.
        :type external_config: Any, optional
        :param use_temp_files: Whether to use temporary files for data subsets.
        :type use_temp_files: bool
        :param prepare_engine: Engine to use for data preparation ('pandas' or 'polars').
        :type prepare_engine: Literal['pandas', 'polars']
        """
        if external_config is None:
            self._external_config = config
        else:
            self._external_config = external_config
        self._work_type_fit = self._external_config.work_type_fit if "work_type_fit" in self._external_config.__dict__ else "CPU"
        self._work_type_hptune = self._external_config.work_type_hptune if "work_type_hptune" in self._external_config.__dict__ else "CPU"
        self._use_temp_files = use_temp_files
        self._prepare_engine = prepare_engine
        self._exposure = {}
        self._all_models_config_name: Union[str, Dict] = config_name
        self._results: Dict[str, DSManagerResult] = {}
        self._extractor: Optional[Extractor] = extractor
        self._prepare_datasets: Optional[Dict[str, PrepareDataset]] = prepared_datasets
        self._models_dict: Optional[Dict] = models_dict
        self._use_baseline_model = use_baseline_model
        self._business_metric: Optional[BaseMetric] = business_metric
        self._data_preprocessor: Optional[DataPreprocessor] = None
        self.X: Optional[pd.DataFrame] = None
        self.Y: Optional[pd.DataFrame] = None
        self.index_train: Optional[pd.Index] = None
        self.index_test: Optional[pd.Index] = None
        self.targets_columns_names = []
        self.extra_columns: Optional[pd.DataFrame] = None
        self.all_models_config: Optional[AllModelsConfig] = None
        self.group_name = 'general'
        self.data_config: Optional[DataModelConfig] = None
        self._models_configs: List[ModelConfig] = []
        self._retro = False
        self.business_metric_value = {}

        self._retro_changes = retro_changes
        self._retro_dataset = None

        self._default_name = None
        self._init_dsmanager()

    @property
    def dataset(self):
        """Get the loaded dataset.
        
        :return: DataFrame containing the loaded dataset.
        :rtype: pd.DataFrame
        """
        return self._data_preprocessor.dataset

    @property
    def config(self):
        """Get configuration with updated model configs from results.
        
        :return: Deep copy of all models configuration with updated model configs.
        :rtype: AllModelsConfig
        """
        config_to_return = deepcopy(self._all_models_config)
        if self._results != {}:
            updated_models_configs = []
            for result in self._results.values():
                updated_models_configs.append(result.model_config)
            config_to_return.models_configs = updated_models_configs

        return config_to_return

    def get_result(self) -> Dict[str, DSManagerResult]:
        """Get dictionary of all model results.
        
        :return: Dictionary with model names as keys and DSManagerResult objects as values.
        :rtype: Dict[str, DSManagerResult]
        
        .. rubric:: Examples
        
        >>> results = ds_manager.get_result()
        >>> results['model1'].metrics
        {'train': {...}, 'test': {...}}
        """
        return self._results

    def load_dataset(self, data: pd.DataFrame | pl.DataFrame = None) -> pd.DataFrame | pl.DataFrame:
        """Load data from source according to config or user-defined extractor object.
        
        Uses .env file or external config for extractor. Can also load dataset directly via parameter.
        
        :param data: Optional DataFrame to load directly. If provided, uses SimpleExtractor.
        :type data: pd.DataFrame, optional
        :return: Loaded dataset as DataFrame.
        :rtype: pd.DataFrame
        
        .. rubric:: Examples
        
        >>> dataset = ds_manager.load_dataset()
        >>> # or
        >>> dataset = ds_manager.load_dataset(data=my_dataframe)
        """

        logger.debug("Dataset loading")
        if data is not None:
            self._extractor = SimpleExtractor(data=data)
        data = self._extractor.extract_dataset()
        logger.debug('DataSet is extracted')
        return data

    def get_subset(self, model_name):
        """Get data subset for specified model.
        
        :param model_name: Name of the model. If None, uses default model name.
        :type model_name: str, optional
        :return: ModelDataSubset object containing train/test data for the model.
        :rtype: ModelDataSubset
        
        .. rubric:: Examples
        
        >>> subset = ds_manager.get_subset('model1')
        >>> subset.X_train.shape
        (800, 10)
        """
        if model_name is None: model_name = self._default_name
        logger.debug('Model ' + model_name + ' || Subset export')
        return self._data_preprocessor.get_subset(model_name)

    @property
    def data_subsets(self, ):
        """Get all data subsets for all models.
        
        :return: Dictionary with model names as keys and ModelDataSubset objects as values.
        :rtype: dict
        """
        return self._data_preprocessor.data_subsets()

    def fit_models(self, models_dict: dict = None, need_fit: bool = False, model_name: str = None, calc_feature_importance: bool = False
                  ) -> dict:
        """Fit models and calculate metrics.
        
        If 'need_fit' is True, fit methods are called for models.
        If load_subsets_from_pickle option is enabled, loads previously saved datasubsets.
        
        :param models_dict: Optional dictionary of models to fit. If None, uses default models.
        :type models_dict: dict, optional
        :param need_fit: Whether to fit models. If False, assumes models are already fitted.
        :type need_fit: bool
        :param model_name: Optional name of specific model to fit. If None, fits all models.
        :type model_name: str, optional
        :return: Dictionary with model names as keys and metrics dictionaries as values.
        :rtype: dict
        
        .. rubric:: Examples
        
        >>> metrics = ds_manager.fit_models(need_fit=True)
        >>> metrics['model1']['train']['full']
        {'mae': 0.1234, 'rmse': 0.5678, 'r2': 0.9012}
        """

        fitted = True
        logger.debug('Fitting model started')
        if models_dict is not None:
            models = models_dict
            if need_fit:
                fitted = False
            logger.info('User-defined models')
        else:
            if self._models_dict is None:
                logger.info('Setting default models')
                self.__load_models()
                fitted = True
            models = self._models_dict

        if model_name is not None:
            try:
                chosen_model = models[model_name]
                models = {chosen_model: models[model_name]}
            except KeyError:
                logger.error('Wrong model name in input')

        fitted_models = self.__get_fitted_models(models=models, fitted=fitted)
        metrics = {}
        for model_name, model in fitted_models.items():
            data_subset = self.get_subset(model_name)
            predictions_train = self._predict(model, data_subset.X_train)
            predictions_test = self._predict(model, data_subset.X_test)

            metrics[model_name] = ModelMetrics(data_config=self.data_config,
                                               model_config=self._prepare_datasets[model_name].get_model_config(),
                                               data_subset=data_subset,
                                               ).result_dict(predictions={'train': predictions_train,
                                                                          'test': predictions_test})

            self._results[model_name] = DSManagerResult(model_name=model_name,
                                                        model=model,
                                                        config=self.config,
                                                        data_subset=data_subset,
                                                        model_config=self._prepare_datasets[
                                                            model_name].get_model_config(),
                                                        predictions={'train': predictions_train,
                                                                     'test': predictions_test},
                                                        metrics=metrics[model_name])
        if calc_feature_importance:
            self._calculate_importances(self._results, use_test=True)
        try:
            metrics.update(self._calculate_business_metric())
        except Exception as exc:
            logger.error('Error while calculation business metric||'+ str(exc))
        return metrics

    def _calculate_importances(
            self,
            results: dict,
            use_test: bool = True,
    ):

        for model_name, ds_result in self._results.items():
            feature_importances = FeatureImportance(
                model_name=model_name,
                model=ds_result.model,
                data_subset=self._results[model_name].data_subset,
            )
            feature_importances.calculate_importance(use_test=use_test)
            self._results[model_name].feature_importance = feature_importances

    def check_datadrift(self, model_name: str) -> pd.DataFrame:
        """Check data drift between train and test datasets.
        
        Uses DataDrift library for analysis.
        
        :param model_name: Name of the model to check drift for.
        :type model_name: str
        :return: DataFrame containing data drift analysis results.
        :rtype: pd.DataFrame
        
        .. rubric:: Examples
        
        >>> drift_report = ds_manager.check_datadrift('model1')
        >>> drift_report.head()
        """
        subset = self.get_subset(model_name)
        report = DataDrift(full_calc=False).review(DataContext(X_train=subset.X_train, X_test=subset.X_test))

        return report

    def model_predict(self,
                      data: pd.DataFrame,
                      model_name: str,
                      model_result=None,
                      full_output: bool = True,
                      ) -> DSManagerResult:
        """Construct DSManagerResult for external model or data prediction.
        
        :param data: DataFrame with data to make predictions on.
        :type data: pd.DataFrame
        :param model_name: Name of the model to use for prediction.
        :type model_name: str
        :param model_result: Optional model result as dict from service or DSManagerResult object.
                            If None, uses inner results.
        :type model_result: dict or DSManagerResult, optional
        :param full_output: Whether to return full output with metrics or only predictions.
        :type full_output: bool
        :return: DSManagerResult object containing predictions and optionally metrics.
        :rtype: DSManagerResult
        
        .. rubric:: Examples
        
        >>> result = ds_manager.model_predict(data=new_data, model_name='model1')
        >>> result.predictions['test'].head()
        """
        logger.debug('Prediction for external data||' + model_name)
        if model_result is None:
            logger.info('No external model||Using inner results')
            result = self.get_result()
            model_result = result[model_name]
        else:
            if isinstance(model_result, dict):
                try:
                    model_result = model_result[model_name]
                except KeyError as e:
                    # logger.error(e)
                    logger.debug('Converting pickle to DSManagerResult')
                    model_result = DSManagerResult.from_pickle_model_result(model_result=model_result,
                                                                            all_model_config=self._all_models_config)

        model_config = deepcopy(model_result.model_config)
        model_config.column_exposure = None
        model_config.column_weight = None
        model = model_result.model
        features_numerical = model_result.data_subset.features_numerical
        features_categorical = model_result.data_subset.features_categorical
        data_to_predict = data.copy()

        preproc = DataPreprocessor(prepare_dataset_interface_dict={model_name:
                                                                           PrepareDataset(model_config=model_config,
                                                   check_prepared=False,
                                                   )},
                                       dataset=data_to_predict,
                                       data_config=self.data_config,
                                       prepare_engine='pandas',)
        data_subset = preproc.get_subset(model_name, from_pickle=False)
        output_model = model
        prediction = model.predict(data_subset.X[chain(features_numerical, features_categorical)])
        if isinstance(prediction, np.ndarray):
            prediction = pd.Series(prediction, index=data_subset.X.index)

        metrics = ModelMetrics(model_config=model_config,
                               data_subset=data_subset,
                               data_config=None).result_dict(
            predictions={'train': prediction[prediction.index.isin(preproc.index_train)],
                         'test': prediction[prediction.index.isin(preproc.index_test)]},
        )

        res = DSManagerResult(model_name=model_name,
                              config=model_result.config,
                              model=output_model,
                              data_subset=data_subset,
                              model_config=model_config,
                              predictions={'train': prediction.loc[prediction.index.isin(preproc.index_train)],
                                           'test': prediction.loc[prediction.index.isin(preproc.index_test)]},
                              metrics=metrics,
                              )

        logger.debug('Prediction for external data finished')
        return res

    def ensemble_predict(self, ensemble_result, config=None) -> DSManagerResult:
        """Predict one model of a stored-reference ensemble on the manager dataset.

        Resolves the model reference of each ensemble part (loading the referenced
        group pickle when ``store_references=True`` was used), filters the dataset
        by the part's ``condition``, predicts each row partition with
        :meth:`model_predict`, then stitches the partitions back together
        (row-wise, sorted by original index) into a single ``DSManagerResult``.
        Metrics are recomputed on the combined data so the result can be used for
        comparison exactly like a regular model result.

        This is the building block for comparing a candidate ensemble (with one
        model replaced) against the previous ensemble: each ``EnsembleResult``
        produces one stitched ``DSManagerResult`` keyed by ``model_name``.

        :param ensemble_result: One ensemble part for a single model, holding
            ``model_name`` and a list of ``(condition, group_name, model)`` where
            ``model`` is either a model result dict or a group name reference.
        :type ensemble_result: EnsembleResult
        :param config: Configuration providing ``prod_models_path`` for resolving
            references. Defaults to the manager's external config / global config.
        :type config: object, optional
        :return: Stitched result with combined ``data_subset``, ``predictions``
            and recomputed ``metrics``.
        :rtype: DSManagerResult
        :raises ValueError: If no rows match any condition, or conditions overlap
            (duplicate indices in the stitched result).
        """
        model_name = ensemble_result.model_name
        logger.debug('Ensemble prediction||' + model_name)
        if config is None:
            config = self._external_config

        parts = []
        for condition, group_name, model in ensemble_result.models:
            model_result = resolve_model_reference(model, model_name, config)
            data_filtered = self.dataset.query(condition)
            if data_filtered.empty:
                logger.debug('Ensemble part matched no rows||' + str(condition))
                continue
            parts.append(self.model_predict(data=data_filtered,
                                             model_name=model_name,
                                             model_result=model_result))

        if not parts:
            raise ValueError(f"Ensemble model `{model_name}`: no rows matched any condition")

        def _concat(getter):
            series = [value for p in parts if (value := getter(p)) is not None]
            if not series:
                return None
            return pd.concat(series).sort_index()

        data_subset = ModelDataSubset(
            model_name=model_name,
            X_train=_concat(lambda p: p.data_subset.X_train),
            y_train=_concat(lambda p: p.data_subset.y_train),
            X_test=_concat(lambda p: p.data_subset.X_test),
            y_test=_concat(lambda p: p.data_subset.y_test),
            features_numerical=parts[0].data_subset.features_numerical,
            features_categorical=parts[0].data_subset.features_categorical,
            exposure_train=_concat(lambda p: p.data_subset.exposure_train),
            exposure_test=_concat(lambda p: p.data_subset.exposure_test),
            sample_weight_train=_concat(lambda p: p.data_subset.sample_weight_train),
            sample_weight_test=_concat(lambda p: p.data_subset.sample_weight_test),
        )

        predictions = {
            'train': _concat(lambda p: p.predictions['train']),
            'test': _concat(lambda p: p.predictions['test']),
        }

        full_index = pd.concat([predictions['train'], predictions['test']]).index
        if full_index.duplicated().any():
            raise ValueError(f"Ensemble model `{model_name}`: overlapping conditions produce duplicate rows")

        model_config = deepcopy(parts[0].model_config)
        metrics = ModelMetrics(model_config=model_config,
                               data_subset=data_subset,
                               data_config=None).result_dict(predictions=predictions)

        res = DSManagerResult(model_name=model_name,
                              config=parts[0].config,
                              model=parts[0].model,
                              data_subset=data_subset,
                              model_config=model_config,
                              predictions=predictions,
                              metrics=metrics,
                              )
        logger.debug('Ensemble prediction finished||' + model_name)
        return res

    def __get_fitted_models(self, models: dict, fitted: bool = False) -> dict:
        """Get fitted models, fitting them if necessary.
        
        :param models: Dictionary of models to fit.
        :type models: dict
        :param fitted: Whether models are already fitted.
        :type fitted: bool
        :return: Dictionary of fitted models.
        :rtype: dict
        """
        if not fitted:
            for model_name in models.keys():
                data_subset = self.get_subset(model_name)
                models[model_name].fit(data_subset.X_train, data_subset.y_train)
        return models

    def _predict(self, model, X):
        """Make predictions using the model.
        
        Handles special case for Prophet models.
        
        :param model: Trained model object with predict method.
        :type model: Any
        :param X: Feature matrix for prediction.
        :type X: pd.DataFrame
        :return: Series of predictions with same index as X.
        :rtype: pd.Series
        """
        if X is None: return None
        if type(model).__name__ == 'Prophet':
            logger.info('Prophet in work..')
            data = model.predict(X)
            prediction_series = pd.Series(data=np.expm1(data['yhat']), index=data.index)
            logger.info('Prophet finished')

        else:
            data = model.predict(X)
            prediction_series = pd.Series(data=data, index=X.index)

        return prediction_series

    def _calculate_business_metric(self,) -> dict:
        """Calculate business metric if business_metric is configured.
        
        :return: Dictionary with business metric results.
        :rtype: dict
        """
        metric = {}
        try:
            if self._business_metric is not None:
                metric = self._business_metric.calculate_metric(self.get_result())
                self.business_metric_value = metric
                logger.info('Business metric value||'+ str(metric))
        except ModuleNotFoundError as e:
            logger.info('No business metrics')

        logger.debug('Calculating metrics finished')
        return metric

    def __load_all_models_config(self):
        """Load and validate all models configuration from file or dict.
        
        :raises FileNotFoundError: If config file is not found.
        :raises ValidationError: If config validation fails.
        """

        if isinstance(self._all_models_config_name, dict):
            logger.info("All models config from dict")
            all_models_config = json.dumps(self._all_models_config_name)

        else:
            logger.info("All models config from path")
            try:
                with open(self._all_models_config_name, "r", encoding='utf-8') as f:
                    all_models_config = f.read()
            except FileNotFoundError:
                logger.error("Invalid all models config name")
                raise FileNotFoundError("Invalid config name")

        try:
            self._all_models_config = AllModelsConfig.model_validate_json(all_models_config)
        except ValidationError as e:
            logger.error("Config validation error")
            raise ValidationError(e)
        self.data_config = self._all_models_config.data_config
        self._models_configs = self._all_models_config.models_configs
        self.version = self._all_models_config.version
        for model in self._models_configs:

            file_path = os.path.join(self._external_config.results_path, model.name + '_v' + self.version + '_subset.pickle')
            if os.path.exists(file_path):
                if not self._retro:
                    logger.warning(f'{model.name}||File {file_path} already exists. Change version in config file to for new data prepare')
                else:
                    logger.warning(
                        f'{model.name}||File {file_path} already exists. Changing version in config file for A/B test')
                    self.version = self.version + '_new'
        self.group_name = f"{self._all_models_config.project}_{self._all_models_config.version}"

        logger.info("Config is loaded")

    def __load_targets_names(self):
        """Load target column names from model configs and set random state.
        
        Extracts unique target and exposure column names from all model configurations.
        """

        self.random_state = self.data_config.separation.random_state

        if self.random_state:
            np.random.seed(self.random_state)

        self.targets_columns_names = list(set(
            [model.column_target for model in self._models_configs if model.column_target]
            + [model.column_exposure for model in self._models_configs if model.column_exposure]
            + [model.column_weight for model in self._models_configs if model.column_weight]
        ))

    def __load_prepare_datasets(self):
        """Load or create PrepareDataset objects for all models.
        
        Creates default PrepareDataset objects if not provided by user.
        Supports both pandas and polars engines.
        
        :raises ValueError: If prepare engine is unknown.
        """

        i = 0
        if self._prepare_datasets is None and self._prepare_engine == "pandas":
            self._prepare_datasets = {}
            logger.info("Load models prepare datasets")
            for model_config in self._models_configs:
                self._prepare_datasets[model_config.name] = PrepareDataset(model_config=model_config,
                                                                           check_prepared=True,
                                                                           group_name=self.group_name)

        elif self._prepare_datasets is None and self._prepare_engine == "polars":
            self._prepare_datasets = {}
            logger.info("Load models prepare datasets with polars")
            for model_config in self._models_configs:
                self._prepare_datasets[model_config.name] = PrepareDatasetPl(
                    group_name=self.group_name, model_config=model_config, check_prepared=True
                )

        elif self._prepare_datasets is None:
            logger.error("Unknown prepare engine")
            raise ValueError("Unknown prepare engine")

        else:
            logger.info("User models prepare datasets")
            for value in self._prepare_datasets.values():
                if value.get_model_config() is None:
                    logger.info(f"Load config  from DS Manager")
                    value.load_model_config(model_config=self._models_configs[i])
                    i += 1
        self._default_name = list(self._prepare_datasets.keys())[0]

    def __load_models(self):
        """Load default models if models_dict is not provided.
        
        Uses DefaultModels class to create models based on configurations.
        """
        if self._models_dict is None:
            self._models_dict = DefaultModels(dataset=self.dataset,
                                              data_subsets=self._data_preprocessor.data_subsets(),
                                              models_configs=self._models_configs,
                                              group_name=self.group_name,
                                              baseline_model=self._use_baseline_model,
                                              work_type_fit=self._work_type_fit).load_default()

    def __init_retro(self):
        """Initialize retro analysis if retro_changes is configured.
        
        Sets up retro dataset and modifies model configs for retro analysis.
        """
        logger.debug('Initializing retro')
        self._retro_dataset = self._retro_changes.get_retro_dataset()
        self._models_configs = self._retro_changes.models_config_for_retro(models_config=self._models_configs,
                                                                           target_columns_names=self.targets_columns_names)

    def _init_dsmanager(self):
        """Initialize DataSetsManager by loading configs, datasets, and extractors.
        
        This is the main initialization method called in __init__.
        """
        logger.debug('Initializing DSManager')
        self.__load_all_models_config()
        self.__load_targets_names()
        if self._retro_changes is not None:
            self.__init_retro()
        self.__load_prepare_datasets()

        if self._extractor is not None:
            logger.info("Reading user extractor")
            if self._extractor.load_config_from_env:
                logger.info("Reading config from env")
                self._extractor.load_config(connection_config=config)


        else:
            self._extractor = BaseExtractor(data_config=self.data_config)
        logger.debug('Initializing completed')
        self._data_preprocessor = DataPreprocessor(prepare_dataset_interface_dict=self._prepare_datasets,
                                                   dataset=self._extractor,
                                                   external_config=self._external_config,
                                                   version=self.version,
                                                   prepare_engine=self._prepare_engine,
                                                   use_saved_files=self._use_temp_files,
                                                   data_config=self.data_config,
                                                   retro=self._retro
                                                   )
