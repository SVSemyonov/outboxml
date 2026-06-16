import os
import json
import pickle
import shutil
from copy import deepcopy
from typing import Callable

import mlflow
import pandas as pd
from loguru import logger
from sqlalchemy import create_engine
import select

from outboxml.core.config_builders import AutoMLConfigBuilder, AllModelsConfigBuilder, feature_params, FeatureBuilder, \
    ModelConfigBuilder
from outboxml.core.enums import ModelsParams, EncodingNames, FeatureEngineering
from outboxml.core.utils import ResultPickle
from outboxml.datasets_manager import DataSetsManager, DSManagerResult
from outboxml.data_subsets import ModelDataSubset
from outboxml.ensemble import resolve_model_reference
from outboxml.metrics.processor import ModelMetrics


def load_last_pickle_models_result(config=None, group_name_json:str=None):
    """Loads the latest saved models result from a pickle file.

    The function determines the latest model group name and loads
    the corresponding pickle file from the production models directory.

    :param config: Project configuration containing the models path.
    :type config: Any

    :param group_name_json: Name of the models group (if not provided, the latest is used).
    :type group_name_json: str, optional

    :return: Dictionary with loaded model groups.
    :rtype: dict

    :raises FileNotFoundError: If the pickle file does not exist.
    :raises pickle.UnpicklingError: If the pickle file is corrupted or invalid.

    .. rubric:: Examples

    >>> results = load_last_pickle_models_result(config)
    >>> list(results.keys())
    ['group_001']
    """
    all_groups = {}
    group_name = ResultPickle(config).get_last_group_name(group_name=group_name_json)

    logger.info('Loading pickle||' + group_name)
    group = all_groups.get(group_name)
    if not group:
        with open(os.path.join(config.prod_models_path, f"{group_name}.pickle"), "rb") as f:
            group = pickle.load(f)
            all_groups.update({group_name: group})
    return all_groups


def predict_ensemble_on_dataset(ds_manager: DataSetsManager, ensemble_result, config=None) -> DSManagerResult:
    """Optional pre-handler for EnsembleResult on a DataSetsManager dataset.

    Resolves conditions, filters the dataset partition-by-partition, calls the
    standard :meth:`DataSetsManager.model_predict` for each partition, then
    stitches the results back together (row-wise, sorted by index) into a single
    :class:`~outboxml.datasets_manager.DSManagerResult` with recomputed metrics.

    :param ds_manager: Dataset and model manager whose dataset is used.
    :type ds_manager: DataSetsManager
    :param ensemble_result: One :class:`~outboxml.ensemble.EnsembleResult` for a
        single model, containing ``(condition, group_name, model)`` tuples.
    :param config: Configuration providing ``prod_models_path`` for resolving
        string model references. Defaults to the manager's external config.
    :raises ValueError: If no rows match any condition, or conditions overlap.
    :return: Stitched result ready for comparison with a regular model result.
    :rtype: DSManagerResult
    """
    model_name = ensemble_result.model_name
    logger.debug('Ensemble predict (handler)||' + model_name)
    if config is None:
        config = ds_manager._external_config

    parts = []
    for condition, group_name, model in ensemble_result.models:
        model_result = resolve_model_reference(model, model_name, config)
        data_filtered = ds_manager.dataset.query(condition)
        if data_filtered.empty:
            logger.debug('Ensemble part matched no rows||' + str(condition))
            continue
        parts.append(ds_manager.model_predict(data=data_filtered,
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
    metrics = ModelMetrics(
        model_config=model_config,
        data_subset=data_subset,
        data_config=None,
    ).result_dict(predictions=predictions)

    logger.debug('Ensemble predict (handler) finished||' + model_name)
    return DSManagerResult(
        model_name=model_name,
        config=parts[0].config,
        model=parts[0].model,
        data_subset=data_subset,
        model_config=model_config,
        predictions=predictions,
        metrics=metrics,
    )


def calculate_previous_models(ds_manager: DataSetsManager,
                              all_groups=None,
                              ensemble=None,
                              config=None,
                              ) -> dict:
    """Calculates predictions and metrics for previously trained models.

        For each model in the provided groups, predictions are generated
        using the current dataset via the DataSetsManager.

        Two input modes are supported:

        * ``all_groups`` — a dictionary of plain model groups. Each model is
          predicted directly on the current dataset.
        * ``ensemble`` — a list of ``EnsembleResult`` (the simplified ensemble
          format, including ``store_references=True``). Each ensemble part is
          resolved, predicted partition-by-partition and stitched into a single
          result via :func:`_predict_ensemble_on_dataset`. This is the mode
          used to compare a candidate ensemble (with one model replaced) against
          the previous one.

        :param ds_manager: Dataset and model manager.
        :type ds_manager: DataSetsManager

        :param all_groups: Dictionary of model groups and their results.
        :type all_groups: dict, optional

        :param ensemble: List of ``EnsembleResult`` objects describing the
            ensemble to evaluate.
        :type ensemble: list, optional

        :param config: Configuration providing ``prod_models_path`` for resolving
            ensemble model references. Defaults to the manager's config.
        :type config: object, optional

        :return: Dictionary with prediction results keyed by model name.
        :rtype: dict

        :raises ValueError: If neither ``all_groups`` nor ``ensemble`` is provided.

        .. rubric:: Examples

        >>> results = calculate_previous_models(ds_manager, all_groups)
        >>> results.keys()
        dict_keys(['xgboost', 'lightgbm'])
        """
    if all_groups is None and ensemble is None:
        raise ValueError("Provide either `all_groups` or `ensemble`")

    logger.debug('Calculating metrics for previous model')
    ds_result_to_compare = {}

    if ensemble is not None:
        for ensemble_result in ensemble:
            ds_result_to_compare[ensemble_result.model_name] = predict_ensemble_on_dataset(
                ds_manager=ds_manager,
                ensemble_result=ensemble_result,
                config=config,
            )
        return ds_result_to_compare

    for key in all_groups.keys():

        models = all_groups[key]

        for model_result in models:
            model_name = model_result['model_config']['name']
            ds_result_to_compare[model_name] = ds_manager.model_predict(data=ds_manager.dataset,
                                                                        model_name=model_name,
                                                                        model_result=model_result)
    return ds_result_to_compare

def check_postgre_transaction(script: Callable, config, waiting_time=300):
    """Waits for PostgreSQL notifications and executes a callback script.

        The function listens to the ``table_changes`` channel and waits
        for notifications within the specified timeout. When a notification
        is received, the provided callback function is executed.

        :param script: Callback function executed on notification.
        :type script: Callable

        :param config: Configuration containing database connection parameters.
        :type config: Any

        :param waiting_time: Notification waiting time in seconds.
        :type waiting_time: int

        :raises Exception: If a database connection or SQL execution error occurs.

        .. rubric:: Examples

        >>> def my_script():
        ...     print("Database updated")
        >>> check_postgre_transaction(my_script, config, waiting_time=60)
    """
    # Connecting to the database
    params = config.connection_params
    engine = create_engine(params)
    raw_conn = engine.raw_connection()  # Get raw psycopg2 connection
    try:
        raw_conn.set_isolation_level(0)  # AUTOCOMMIT
        cur = raw_conn.cursor()
        cur.execute("LISTEN table_changes;")

        logger.debug(f"Waiting for notifications for {waiting_time} seconds...")
        if select.select([raw_conn], [], [], waiting_time) == ([], [], []):
            logger.debug("No notifications received")
        else:
            raw_conn.poll()
            while raw_conn.notifies:
                notify = raw_conn.notifies.pop(0)
                logger.debug(f"Notification received: {notify.payload}")
                script()

    except Exception as e:
        print(f"Произошла ошибка: {e}")

    finally:
        raw_conn.close()


def build_default_auto_ml_config(params:dict={}):
    """Builds a default AutoML configuration.

        :param params: AutoML configuration parameters.
        :type params: dict

        :return: Built AutoML configuration.
        :rtype: dict

        .. rubric:: Examples

        >>> config = build_default_auto_ml_config()
        """
    return AutoMLConfigBuilder(**params).build()

def build_default_all_models_config(data:pd.DataFrame=None,
                                    column_target: str=None,
                                    column_exposure: str=None,
                                    column_weight: str=None,
                                    group_name:str = 'example',
                                    project:str = 'test',
                                    version: str = '1',
                                    max_category_num: int = 20,
                                    category_proportion_cut_value: float=0.01,
                                    q1:float=0.001,
                                    q2:float=0.999,
                                    model_params:dict={},
                                    features_params:dict={},
                                    ):
    """Builds a configuration for training all models with feature generation.

        The function automatically generates features from the input dataset,
        excludes target and exposure columns, and constructs a configuration
        for model training.

        :param data: Dataset used for feature generation.
        :type data: pandas.DataFrame

        :param column_target: Target column name.
        :type column_target: str, optional

        :param column_exposure: Exposure column name.
        :type column_exposure: str, optional

        :param group_name: Model group name.
        :type group_name: str

        :param project: Project name.
        :type project: str

        :param version: Configuration version.
        :type version: str

        :param max_category_num: Maximum number of categories.
        :type max_category_num: int

        :param category_proportion_cut_value: Threshold for rare category filtering.
        :type category_proportion_cut_value: float

        :param q1: Lower quantile.
        :type q1: float

        :param q2: Upper quantile.
        :type q2: float

        :param model_params: Model parameters.
        :type model_params: dict

        :param features_params: Feature generation parameters.
        :type features_params: dict

        :return: All-models configuration.
        :rtype: dict

        .. rubric:: Examples

        >>> config = build_default_all_models_config(data=df, column_target="target")
        """
    features =[]
    if data is not None:
        if column_target is not None:
            data = data.drop(columns=column_target)
            model_params['column_target'] = column_target
        if column_exposure is not None:
            data = data.drop(columns=column_exposure)
            model_params['column_exposure'] = column_exposure
        if column_weight is not None:
            data = data.drop(columns=column_weight)
            model_params['column_weight'] = column_weight

        for columns_name, series in data.items():
            params = feature_params(serie=series,
                                    max_category_num=max_category_num,
                                    depth=category_proportion_cut_value,
                                    q1=q1,
                                    q2=q2,
                                    **features_params
                                    )
            if params == {}:
                logger.info('Dropping feature||' + str(series.name))
                continue
            features.append(FeatureBuilder(**params).build())

    config_params = {'group_name': group_name,
                     'project': project,
                     'version': version,
                     'models_config': [ModelConfigBuilder(features=features,
                                                          **model_params
                                                                     ).build()]
                                 }

    return AllModelsConfigBuilder(**config_params).build()
