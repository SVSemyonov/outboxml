import os
import pickle
import shutil
from typing import Callable

import mlflow
from loguru import logger
from sqlalchemy import create_engine
import select

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
            try:
                model_result['model']
            #FIXME
            except KeyError:
                if model_result.get("model_sm") is not None:
                    model_result["model"] = model_result.get("model_sm")
                elif model_result.get("model_ctb") is not None:
                    model_result["model"] = model_result.get("model_ctb")

            ds_result_to_compare[model_name] = ds_manager.model_predict(data=ds_manager.dataset,
                                                                        model_name=model_name,
                                                                        model_result=model_result,
                                                                        train_ind=ds_manager.index_train,
                                                                        test_ind=ds_manager.index_test)
    return ds_result_to_compare


def load_model_to_source(group_name: str, config=None) -> None:
    source_path = config.prod_models_path
    if os.path.isfile(source_path / f"{group_name}.pickle"):
        raise FileExistsError(f"Already in {source_path}: {group_name}.pickle")

    if not os.path.isfile(config.results_path / f"{group_name}.pickle"):
        raise FileNotFoundError(f"Not found in {config.results_path}: {group_name}.pickle")

    shutil.copyfile(config.results_path / f"{group_name}.pickle", source_path / f"{group_name}.pickle")


def load_model_to_source_from_mlflow(group_name: str, config=None) -> None:
    source_path = config.prod_models_path
    if os.path.isfile(source_path / f"{group_name}.pickle"):
        raise FileExistsError(f"Already in {source_path}: {group_name}.pickle")

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment)

    mlflow_runs = mlflow.search_runs(filter_string=f"tags.mlflow.runName = '{group_name}'")
    if not mlflow_runs:
        raise FileNotFoundError(f"Not found in mlflow: {group_name}")

    artifact_uri = mlflow_runs.sort_values("start_time", ascending=False).iloc[0].artifact_uri
    mlflow.artifacts.download_artifacts(artifact_uri, dst_path="/")

    shutil.copyfile(f"./artifacts/{group_name}.pickle", source_path / f"{group_name}.pickle")

def check_postgre_transaction(script: Callable, config=None, waiting_time=300):
    global last_seen_id

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
