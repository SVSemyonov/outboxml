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
        model_name = str(monitoring_config.pickle_name).split('.')[0]
        models_config = load_last_pickle_models_result(monitoring_config)
        all_features = self._find_unique_features(monitoring_config, models_config)
        model_filter_report = pd.DataFrame()
        report_all_features = pd.DataFrame()

        def create_model_filter_report(feature, model_name, dataset_name, nan_share):
            row = pd.DataFrame(
                {
                    'FEATURE': [feature],
                    'MODEL_NAME': [model_name],
                    'DATASET_NAME': [dataset_name],
                    'NAN_SHARE': [nan_share]
                }
            )
            return row

        for feature in tqdm(data.columns):
            nan_share = data[feature].isna().mean()
            row = pd.DataFrame(
                {
                    'FEATURE': [feature],
                    'MODEL_NAME': ['all_features'],
                    'DATASET_NAME': [dataset_name],
                    'NAN_SHARE': [nan_share]
                }
            )
            if feature in all_features:
                model_filter_report = pd.concat([model_filter_report, create_model_filter_report(feature, model_name, dataset_name, nan_share)], ignore_index=True)
            report_all_features = pd.concat([report_all_features, row], ignore_index=True)
        return pd.concat([report_all_features, model_filter_report], ignore_index=True)

    def _find_unique_features(self, monitoring_config, models_config):
        key = list(models_config.keys())[0]
        features_set = set()
        for config in models_config[key]:
            features = chain(config['features_numerical'], config['features_categorical'])
            for f in features:
                features_set.add(f)

        logger.info(f'Unique features from all models: {features_set}')
        return features_set