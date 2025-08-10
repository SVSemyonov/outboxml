import json
import time
from typing import Callable, Any

import schedule
from outboxml import config
from examples.titanic.titanic_basic import titanic_example
from outboxml.automl_utils import check_for_new_data
from outboxml.automl_manager import AutoMLConfig




def main(auto_ml_script: Callable, auto_ml_config: str, config: Any):
    with open(auto_ml_config, encoding='utf-8') as f:
        auto_ml_config = AutoMLConfig.model_validate(json.load(f))
    params = {'auto_ml_config': auto_ml_config, 'auto_ml_script': auto_ml_script, 'config': config}
    schedule.every(5).minutes.do(check_for_new_data, *params)

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main(auto_ml_config='./configs/test_configs/automl-titanic.json',
         auto_ml_script=titanic_example,
         config=config
         )
