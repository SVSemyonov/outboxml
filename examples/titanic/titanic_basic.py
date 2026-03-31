import numpy as np
import pandas as pd

from outboxml.automl_manager import AutoMLManager
from outboxml.extractors import Extractor
import config

config_name = './configs/config-example-titanic.json'
auto_ml_config = './configs/automl-titanic.json'
path_to_data = './data/titanic.csv'


class TitanicExampleExtractor(Extractor):

    def __init__(self, path_to_file: str):
        self.__path_to_file = path_to_file
        super().__init__()

    def extract_dataset(self) -> pd.DataFrame:
        data = pd.read_csv(self.__path_to_file)
        rng = np.random.default_rng(42)
        data["ROW_WEIGHT"] = rng.uniform(0.5, 1.5, size=len(data))
        return data


def titanic_example(retro: bool =True):
    AutoMLManager(auto_ml_config=auto_ml_config,
                  models_config=config_name,
                  external_config=config,
                  extractor=TitanicExampleExtractor(path_to_file=path_to_data),
                  retro=retro
                  ).update_models()


if __name__ == "__main__":
    titanic_example()
