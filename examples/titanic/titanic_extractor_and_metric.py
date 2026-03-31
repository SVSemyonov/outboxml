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
from outboxml.metrics.base_metrics import BaseMetric
from outboxml.metrics.business_metrics import BaseCompareBusinessMetric

config_name = str(BASE_DIR / 'configs' / 'config-example-titanic.json')
auto_ml_config = str(BASE_DIR / 'configs' / 'automl-titanic.json')
path_to_data = str(BASE_DIR / 'data' / 'titanic.csv')
config.mlflow_tracking_uri = (BASE_DIR / "mlruns").resolve().as_uri()
config.mlflow_experiment = "TitanicExample"


class TitanicExampleExtractor(Extractor):

    def __init__(self,
                 path_to_file: str
                 ):
        self.__path_to_file = path_to_file
        super().__init__()

    def extract_dataset(self) -> pd.DataFrame:
        data = pd.read_csv(self.__path_to_file)
        rng = np.random.default_rng(42)
        data["ROW_WEIGHT"] = rng.uniform(0.5, 1.5, size=len(data))
        data['survived1'] = data['SURVIVED']
        data['survived2'] = data['SURVIVED']
        return data


class TitanicMetric(BaseMetric):
    def __init__(self):
        pass

    def calculate_metric(self, result1: dict, result2: dict=None) -> dict:
        y1 = (result1['first'].y_pred + result1['second'].y_pred) / 2
        y = result1['first'].y
        score1 = (y - y1).sum()
        score2 = 0
        if result2 is not None:
            y2 = (result2['first'].y_pred + result2['second'].y_pred) / 2
            score2 = (y - y2).sum()
        return {'impact': score1-score2}



def main():
    auto_ml = AutoMLManager(auto_ml_config=auto_ml_config,
                            models_config=config_name,
                            business_metric=TitanicMetric(),
                            external_config=config,
                            extractor=TitanicExampleExtractor(path_to_file=path_to_data),
                            compare_business_metric=BaseCompareBusinessMetric(),
                            hp_tune=False,
                            retro=False
                            )
    auto_ml.update_models(send_mail=False)


if __name__ == "__main__":
    main()
