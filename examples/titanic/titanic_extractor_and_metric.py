import pandas as pd

from outboxml.automl_manager import AutoMLManager
from outboxml.extractors import Extractor


import config
from outboxml.metrics.base_metrics import BaseMetric
from outboxml.metrics.business_metrics import BaseCompareBusinessMetric

config_name = './configs/config-example-titanic.json'
auto_ml_config = './configs/automl-titanic.json'
path_to_data = 'data/titanic.csv'


class TitanicExampleExtractor(Extractor):

    def __init__(self,
                 path_to_file: str
                 ):
        self.__path_to_file = path_to_file
        super().__init__()

    def extract_dataset(self) -> pd.DataFrame:
        data = pd.read_csv(self.__path_to_file)
        data['survived1'] = data['SURVIVED']
        data['survived2'] = data['SURVIVED']
        return data


class TitanicMetric(BaseMetric):
    def __init__(self):
        pass

    def calculate_metric(self, result1: dict, result2: dict) -> dict:
        y1 = (result1['first'].y_pred + result1['second'].y_pred) / 2
        y = result1['first'].y
        score1 = (y - y1).sum()
        y2 = (result2['first'].y_pred + result2['second'].y_pred) / 2
        score2 = (y - y2).sum()
        return {'impact': score2 - score1}


def main():
    auto_ml = AutoMLManager(auto_ml_config=auto_ml_config,
                            models_config=config_name,
                            business_metric=TitanicMetric(),
                            external_config=config,
                            extractor=TitanicExampleExtractor(path_to_file=path_to_data),
                            compare_business_metric=BaseCompareBusinessMetric(),
                            save_temp=False,
                            hp_tune=False,
                            )
    auto_ml.update_models(send_mail=False)


if __name__ == "__main__":
    main()
