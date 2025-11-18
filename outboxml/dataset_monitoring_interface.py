import numpy as np
from tqdm import tqdm
import pandas as pd
from itertools import chain
from loguru import logger
from outboxml.automl_utils import load_last_pickle_models_result


class DatasetMonitor:
    def __init__(self):
        pass

    def report(self, monitoring_config, dataset_name, data: pd.DataFrame = None) -> pd.DataFrame:
        model_name = monitoring_config.pickle_name
        models_config = load_last_pickle_models_result(monitoring_config)
        all_features = self._find_unique_features(monitoring_config, models_config)
        report = pd.DataFrame(columns=['FEATURE', 'NAN_SHARE'])
        for feature in tqdm(all_features):
            row = pd.DataFrame(
                {
                    'FEATURE': [feature],
                    'MODEL_NAME': [model_name],
                    'DATASET_NAME': [dataset_name],
                    'NAN_SHARE': [data[feature].isna().mean()]
                }
            )
            report = pd.concat([report, row], ignore_index=True)
        return report

    def _find_unique_features(self, monitoring_config, models_config):
        key = list(models_config.keys())[0]
        features_set = set()
        for config in models_config[key]:
            features = chain(config['features_numerical'], config['features_categorical'])
            for f in features:
                features_set.add(f)

        logger.info(f'Unique features from all models: {features_set}')
        return features_set