from abc import ABC, abstractmethod
import pandas as pd
from loguru import logger



class RetroDataset(ABC):
    """
    Abstract class for constructing dataset for retro analysis
    Inheritanced classes should contain methods add_X_retro and  add_y_retro for loading data for retro

    ------
    Класс для работы с ретроанализом. Пользователю необходимо перегрузить метод load_retro_data с получением датафрейма для ретро анализа
    """
    def __init__(self):
        self.retro_data = None

    @abstractmethod
    def load_retro_data(self, *params):
        """ Method for extracting dataset for retro analysy. Should load data to self.retro_data"""
        pass

    def get_retro_dataset(self) -> pd.DataFrame:
        self.load_retro_data()
        return self.retro_data

    def models_config_for_retro(self, models_config: list, target_columns_names: list):
        retro_columns = self.retro_data.columns
        logger.info('Changing models config for retro')
        for model in models_config:
            i = 0
            features = model.features.copy()
            features_for_retro = model.features.copy()
            relative_features_for_retro = model.relative_features.copy()
            relative_features = model.relative_features.copy()
            for feature in features:
                if feature.name not in(retro_columns):
                    features_for_retro.remove(feature)
                    i += 1
            for relative_feature in relative_features:
                if (relative_feature.numerator not in (retro_columns)) or(relative_feature.denominator not in retro_columns):
                     relative_features_for_retro.remove(relative_feature)
                     i += 1
            logger.info('Removed ' + str(i) +' features for model ' + model.name)

            model.features = features_for_retro

        for column in target_columns_names:
            if column not in retro_columns:
                logger.error('No target ' + column + ' in retro dataset')
                if column == 'general':
                    self.retro_data['general'] = self.retro_data[target_columns_names[1]]
                    logger.info('Setting general column as '+ target_columns_names[1])
        return models_config

    def features_for_reserch(self, data_column_names: list,
                             target_columns_names: list,
                             models_config: list,
                             extra_columns: list = None,
                             features_list_to_exclude: list=None,):
        data_column_names = list(data_column_names)
        features_for_retro = data_column_names.copy() #список полей в датасете

        target_columns_names = list(target_columns_names)
        for column in target_columns_names:
            if column in features_for_retro:
                features_for_retro.remove(column) #удаляем таргет
        if extra_columns is not None:
            extra_columns = list(extra_columns)
            for column in extra_columns:
                if column in features_for_retro:
                    features_for_retro.remove(column) #удаляем экстра
        features_for_research = []
        if features_list_to_exclude is not None and features_list_to_exclude != []: #логика по внешнему списку из конфига
            logger.info('feature list to exclude from config ' + str(features_list_to_exclude))

            features = features_for_retro.copy()
            for feature in features:
                if feature in features_list_to_exclude:
                    features_for_retro.remove(feature)
        features_for_research = self._features_for_research_from_config(models_config=models_config,
                                                                        features_for_retro=features_for_retro)

        logger.info('Retro||Features for research '+ str(features_for_research))
        return features_for_research

    @staticmethod
    def _features_for_research_from_config(models_config, features_for_retro):
        using_features = []
        model_features = {}
        features_for_research = []
        for model in models_config:  # логика по конфигу
            features = model.features.copy()
            model_features[model.name] = []
            for feature in features:
                model_features[model.name].append(feature.name)

            relative_features = model.relative_features.copy()
            for relative_feature in relative_features:
                if relative_feature.numerator not in model_features[model.name]:
                    model_features[model.name].append(relative_feature.numerator)
                if relative_feature.denominator not in model_features[model.name]:
                    model_features[model.name].append(relative_feature.denominator)
            using_features = using_features + model_features[model.name]

        for feature in features_for_retro:
            if feature not in using_features:
                features_for_research.append(feature)

        return features_for_research
