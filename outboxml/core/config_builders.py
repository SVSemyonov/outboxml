from abc import ABC, abstractmethod
from typing import Optional, List, Union, Dict

import pandas as pd
from loguru import logger
from pydantic import BaseModel

from outboxml.core.enums import FeaturesTypes, FeatureEngineering
from outboxml.core.pydantic_models import FeatureModelConfig, ModelConfig, FeatureSelectionConfig, HPTuneConfig, \
    ModelInferenceConfig, AutoMLConfig, AllModelsConfig, DataConfig, DataModelConfig, SeparationModelConfig


class ConfigBuilder(ABC):
    @abstractmethod
    def build(self)->BaseModel:
        pass





class ModelConfigBuilder(ConfigBuilder):
    def __init__(self, **params):
        self.name = params.get('name', 'default')
        self.wrapper = params.get('wrapper', 'catboost')
        self.objective = params.get('objective', 'rmse')
        self.features: List[FeatureModelConfig] = params.get('features', [])
        self.column_target: Optional[str] = params.get('column_target')
        self.column_exposure: Optional[str] = params.get('column_exposure')
        self.relative_features: Optional[List] = params.get('relative_features')
        self.intersections:[] = None
        self.params_catboost: Optional[Dict[str, Optional[Union[int, float, str, bool]]]] = params.get('params_catboost')
        self.params_glm:  Optional[Dict[str, Optional[Union[int, float, str, bool]]]] = params.get('params_glm')
        self.params_xgb: Optional[Dict[str, Optional[Union[int, float, str, bool]]]] = params.get(
            'params_xgb')
        self.treatment_dict: Optional[Dict[str, str]] = params.get('treatment_dict')
        self.cat_features_catboost: Optional[List[str]] = params.get('cat_features_catboost')  # TODO: move to features
        self.data_filter_condition: Optional[str] = params.get('data_filter_condition')

    def build(self)->ModelConfig:
        return ModelConfig(name=self.name,
                           objective=self.objective,
                           wrapper=self.wrapper,
                           column_target=self.column_target,
                           column_exposure=self.column_exposure,
                           relative_features=self.relative_features,
                           features=self.features,
                           params_catboost=self.params_catboost,
                           params_glm=self.params_glm,
                           params_xgb=self.params_xgb,
                           treatment_dict=self.treatment_dict,
                           cat_features_catboost=self.cat_features_catboost,
                           data_filter_condition=self.data_filter_condition
                           )


class FeatureBuilder(ConfigBuilder):
    def __init__(self, **params):

        self.type = params.get("type")
        self.mapping = params.get("mapping")
        self.bins = params.get("bins")
        self.optbinning_params = params.get("optbinning_params")
        self.fillna = params.get("fillna")
        self.clip = params.get("clip")
        self.encoding = params.get("encoding")
        self.cut_number = params.get("cut_number")
        self.replace_map = params.get("replace_map")
        self.default_value = params.get("default")
        self.feature_values = params.get("feature_values")
        self.name = params.get("name")

    def build(self):
        logger.debug('Feature builder||'+str(self.name))
        if self.replace_map is None:
            self.replace_map = self.__get_replace_map()
        if self.default_value is None:
            self.default_value = FeatureEngineering.nan
        return FeatureModelConfig(name=self.name,
                                  default=self.default_value,
                                  replace=self.replace_map,
                                  clip=self.clip,
                                  cut_number=self.cut_number,
                                  fillna=self.fillna,
                                  encoding=self.encoding,
                                  optbinning_params=self.optbinning_params,
                                  bins=self.bins,
                                  mapping=self.mapping,
                                  )

    def __get_replace_map(self):
        if self.type == FeaturesTypes.numerical:
            return {"_TYPE_": "_NUM_"}
        else:
            if self.type == FeaturesTypes.categorical:
                return dict(
                    (value, FeatureEngineering.not_changed) for value in list(self.feature_values.unique()))


class AutoMLConfigBuilder(ConfigBuilder):
    def __init__(self, **params):
        self.project = params.get('project', 'test')
        self.group_name = params.get('group_name', 'example')
        self.feature_selection = params.get('feature_selection', FeatureSelectionConfig())
        self.hp_tune = params.get('hp_tune', HPTuneConfig())
        self.inference_criteria = params.get('feature_selection',
                                             ModelInferenceConfig(prod_models_folder='prod_models_folder',
                                                                  metric_growth_value={'neg_mean_absolute_error': 0}
                                                                  ))
        self.mlflow_experiment = 'mlflow_experiment_name'
        self.grafana_table_name = 'self.grafana_table_name'
        self.dashboard_name = 'dashboard_name'
        self.trigger: Optional[Dict[str, str]] = None

    def build(self):
        return AutoMLConfig(project=self.project,
                            group_name=self.group_name,
                            feature_selection=self.feature_selection,
                            hp_tune=self.hp_tune,
                            inference_criteria=self.inference_criteria,
                            mlflow_experiment=self.mlflow_experiment,
                            grafana_table_name=self.grafana_table_name,
                            dashboard_name=self.dashboard_name,
                            )


class AllModelsConfigBuilder(ConfigBuilder):
    def __init__(self, data: pd.DataFrame=None, **params):
        self.project = params.get('project', 'test')
        self.group_name = params.get('group_name', 'example')
        self.version = params.get('version', '1')
        self.data_config = params.get('data_config',
                                      DataModelConfig(source='database',
                                                      table_name_source='data_table_name',
                                                      separation=SeparationModelConfig(kind='random',
                                                                                      random_state=42,
                                                                                      test_train_proportion=0.2),

                                                                     ))
        self.models_config = params.get('models_config', [ModelConfigBuilder().build()])


    def build(self):
        return AllModelsConfig(project=self.project,
                               version=self.version,
                               group_name=self.group_name,
                               data_config=self.data_config,
                               models_configs=self.models_config,
                               )
