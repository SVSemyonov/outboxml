import time
from typing import Callable

import schedule
from outboxml import config
from examples.titanic_basic import titanic_example
from outboxml.automl_utils import check_for_new_data
from outboxml.automl_manager import AutoMLConfig

with open('./configs/test_configs/automl-titanic.json') as f:
    auto_ml_config = AutoMLConfig.validate(f.read())


def main(auto_ml_script: Callable = titanic_example, auto_ml_config: AutoMLConfig = auto_ml_config, config=config):
    schedule.every(5).minutes.do(
        check_for_new_data(auto_ml_config=auto_ml_config, script=auto_ml_script, config=config))
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
