import sys
import time
from typing import Callable, Any

sys.path.append("/home")
sys.path.append("/home/jovyan")

import config
import external_config_for_titanic_example
from examples.titanic.titanic_basic import auto_ml_config as titanic_auto_ml_config, config_name
from outboxml.automl_utils import check_postgre_transaction
from outboxml.automl_manager import AutoMLConfig, AutoMLManager
from outboxml.extractors import Extractor
from sqlalchemy import create_engine
import json
import numpy as np
import pandas as pd

from outboxml.main_release import MLFLowRelease


class TitanicDBExtractor(Extractor):
    """Reads training data from the Postgres table created by outboxml/app/init-scripts/init.sql.

    The table has a trigger that notifies the ``table_changes`` channel,
    which ``check_postgre_transaction`` listens to.
    """

    def __init__(self, table_name: str, connection_params: str):
        super().__init__()
        self.__table_name = table_name
        self.__connection_params = connection_params

    def extract_dataset(self) -> pd.DataFrame:
        engine = create_engine(self.__connection_params)
        data = pd.read_sql(f'SELECT * FROM "{self.__table_name}";', engine)
        rng = np.random.default_rng(42)
        data["ROW_WEIGHT"] = rng.uniform(0.5, 1.5, size=len(data))
        return data


def titanic_db_example(retro: bool = True, external_config=external_config_for_titanic_example):
    with open(titanic_auto_ml_config, encoding='utf-8') as f:
        table_name = AutoMLConfig.model_validate(json.load(f)).trigger['table_name']
    AutoMLManager(auto_ml_config=titanic_auto_ml_config,
                  models_config=config_name,
                  external_config=external_config,
                  extractor=TitanicDBExtractor(table_name=table_name,
                                               connection_params=external_config.connection_params),
                  retro=retro
                  ).update_models()


def add_to_db(auto_ml_config='examples/titanic/configs/automl-titanic.json', external_config=None):
    if external_config is None:
        external_config = config
    with open(auto_ml_config, encoding='utf-8') as f:
        auto_ml_config = AutoMLConfig.model_validate(json.load(f))
    engine = create_engine(external_config.connection_params)
    trigger = auto_ml_config.trigger
    table_name = trigger['table_name']
    ID = trigger['field']
    query = f"""SELECT * FROM "{table_name}";"""
    df = pd.read_sql(query, engine)
    max_id = df[ID].max()
    row_to_add = df.tail(1).replace({ID: max_id}, max_id + 1)
    row_to_add.to_sql(table_name, con=external_config.connection_params, if_exists='append', index=False)
    print('Row successfully added')


def main(auto_ml_script: Callable, config: Any, waiting_time: float, group_name: str = None):
    params = {'script': auto_ml_script, 'config': config, 'waiting_time': waiting_time}

    while True:
        check_postgre_transaction(**params)
        MLFLowRelease(config=config).load_model_to_source_from_mlflow(group_name=group_name)
        time.sleep(waiting_time)

if __name__ == "__main__":
    main(auto_ml_script=titanic_db_example,
         config=external_config_for_titanic_example,
         waiting_time=2 * 60,
         group_name='example_titanic'
         )
