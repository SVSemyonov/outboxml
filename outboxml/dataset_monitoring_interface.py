from tqdm import tqdm
import pandas as pd
from itertools import chain
from loguru import logger


class DatasetMonitor:
    """
    DatasetMonitor provides utilities for evaluating dataset feature completeness
    relative to the feature sets used by production models. It generates a report
    containing NaN ratios for all dataset features, as well as a filtered report
    for features required by the monitored model.

    Methods
    -------
    report(monitoring_config, dataset_name, prod_models_configs, data: pd.DataFrame = None) -> pd.DataFrame
        Generates a monitoring report with NaN share statistics for all features
        in the provided dataset, and an additional subset for features used by
        the production model.

    _find_unique_features(models_config)
        Extracts and returns a unique set of features (numerical + categorical)
        used across all production model configurations.
    """
    def __init__(self):
        pass

    def report(self, monitoring_config, dataset_name, prod_models_configs, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Generate a monitoring report for the provided dataset.

        Parameters
        ----------
        monitoring_config : object
            Configuration object containing metadata about the monitored model.
            The attribute `pickle_name` is expected to contain the model's filename.
        dataset_name : str
            Name of the dataset being analyzed.
        prod_models_configs : dict
            Dictionary with production model configuration(s). Each configuration
            must contain lists `features_numerical` and `features_categorical`.
        data : pandas.DataFrame, optional
            The dataset whose features will be analyzed. Each column is treated
            as a feature. Default is None.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing feature names and their NaN share for both:
            - all dataset features,
            - features required by the monitored model.

            The resulting DataFrame includes metadata in `result.attrs['meta']`
            that specifies uniqueness constraints for MODEL_NAME and DATASET_NAME.

        Notes
        -----
        - NaN share is computed as the mean of null values in each column.
        - Two report segments are generated:
            * All features in the dataset.
            * Features that belong to all models' configurations.
        - The two segments are concatenated into a single report.
        - Metadata is added to support downstream checks (e.g., DB validation).
        """
        model_name = str(monitoring_config.pickle_name).split('.')[0]
        all_features = self._find_unique_features(prod_models_configs)
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
        result = pd.concat([report_all_features, model_filter_report], ignore_index=True)

        # Создаем метаданные датафрейма, чтобы позже по ним делать проверку на наличие в БД
        result.attrs['meta'] = {
            'MODEL_NAME': {'uniq': True},
            'DATASET_NAME': {'uniq': True}
        }
        return result

    def _find_unique_features(self, models_config):
        """
        Extract a set of unique feature names from the provided model configurations.

        Parameters
        ----------
        models_config : dict
            Dictionary where each value is a list of model configuration dictionaries.
            Each configuration must contain:
                - 'features_numerical' : list of numerical feature names
                - 'features_categorical' : list of categorical feature names

        Returns
        -------
        set
            A set containing all unique feature names across all provided model configs.

        Notes
        -----
        - Only the first key of the dict is used to access model configurations.
        - The method logs all collected unique features.
        """
        key = list(models_config.keys())[0]
        features_set = set()
        for config in models_config[key]:
            features = chain(config['features_numerical'], config['features_categorical'])
            for f in features:
                features_set.add(f)

        logger.info(f'Unique features from all models: {features_set}')
        return features_set