import numpy as np
import pandas as pd

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parents[1]
sys.path.append(str(ROOT_DIR))

from outboxml.automl_manager import AutoMLManager
from outboxml.extractors import Extractor
import config

config_name = str(BASE_DIR / 'configs' / 'config-example-titanic.json')
auto_ml_config = str(BASE_DIR / 'configs' / 'automl-titanic.json')
path_to_data = str(BASE_DIR / 'data' / 'titanic.csv')
config.mlflow_tracking_uri = (BASE_DIR / "mlruns").resolve().as_uri()
config.mlflow_experiment = "TitanicExample"
config.results_path.mkdir(parents=True, exist_ok=True)
config.prod_models_path.mkdir(parents=True, exist_ok=True)


class TitanicExampleExtractor(Extractor):

    def __init__(self, path_to_file: str):
        self.__path_to_file = path_to_file
        super().__init__()

    def extract_dataset(self) -> pd.DataFrame:
        data = pd.read_csv(self.__path_to_file)
        rng = np.random.default_rng(42)
        data["ROW_WEIGHT"] = rng.uniform(0.5, 1.5, size=len(data))
        return data


def titanic_example(retro: bool =False):
    AutoMLManager(auto_ml_config=auto_ml_config,
                  models_config=config_name,
                  external_config=config,
                  extractor=TitanicExampleExtractor(path_to_file=path_to_data),
                  hp_tune=False,
                  retro=retro
                  ).update_models()


if __name__ == "__main__":
    titanic_example()
