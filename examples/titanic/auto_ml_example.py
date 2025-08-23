import time
from typing import Callable, Any

from outboxml import config
from examples.titanic.titanic_basic import titanic_example
from outboxml.automl_utils import check_postgre_transaction
from outboxml.automl_manager import AutoMLConfig
from sqlalchemy import create_engine, text
import json
import pandas as pd


def add_to_db(auto_ml_config):
    with open(auto_ml_config, encoding='utf-8') as f:
        auto_ml_config = AutoMLConfig.model_validate(json.load(f))
    engine = create_engine(config.connection_params)
    trigger = auto_ml_config.trigger
    table_name = trigger['table_name']
    ID = trigger['field']
    query = f"""SELECT * FROM "{table_name}";"""
    df = pd.read_sql(query, engine)
    max_id = df[ID].max()
    row_to_add = df.tail(1).replace({ID: max_id}, max_id + 1)
    row_to_add.to_sql(table_name, con=config.connection_params, if_exists='append', index=False)
    print('Row successfully added')


def main(auto_ml_script: Callable, config: Any, waiting_time: float):
    params = {'script': auto_ml_script, 'config': config, 'waiting_time': waiting_time}

    while True:
        check_postgre_transaction(**params)
        time.sleep(1)


if __name__ == "__main__":
    main(auto_ml_script=titanic_example,
         config=config,
         waiting_time=2 * 60,
         )
