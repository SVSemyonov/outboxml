import os
import json
import pickle
import shutil
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
from outboxml.datasets_manager import DataSetsManager


def load_last_pickle_models_result(config=None, group_name_json:str=None):
    all_groups = {}
    group_name = ResultPickle(config).get_last_group_name(group_name=group_name_json)

    logger.info('Loading pickle||' + group_name)
    group = all_groups.get(group_name)
    if not group:
        with open(os.path.join(config.prod_models_path, f"{group_name}.pickle"), "rb") as f:
            group = pickle.load(f)
            all_groups.update({group_name: group})
    return all_groups


def calculate_previous_models(ds_manager: DataSetsManager,
                              all_groups,
                              ) -> dict:
    logger.debug('Calculating metrics for previous model')
    ds_result_to_compare = {}

    for key in all_groups.keys():

        models = all_groups[key]

        for model_result in models:
            model_name = model_result['model_config']['name']
            ds_result_to_compare[model_name] = ds_manager.model_predict(data=ds_manager.dataset,
                                                                        model_name=model_name,
                                                                        model_result=model_result)
    return ds_result_to_compare

def check_postgre_transaction(script: Callable, config, waiting_time=300):

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
    return AutoMLConfigBuilder(**params).build()

def build_default_all_models_config(data:pd.DataFrame=None,
                                    column_target: str=None,
                                    column_exposure: str=None,
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


    features =[]
    if data is not None:
        if column_target is not None:
            data = data.drop(columns=column_target)
            model_params['column_target'] = column_target
        if column_exposure is not None:
            data = data.drop(columns=column_exposure)
            model_params['column_exposure'] = column_exposure

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
