from abc import ABC
from typing import Optional, List, Union, Dict

from loguru import logger

from outboxml.core.enums import FeaturesTypes, FeatureEngineering
from outboxml.core.pydantic_models import FeatureModelConfig


class ConfigBuilder(ABC):
    def build(self):
        pass


class ModelBuilder(ConfigBuilder):
    def __init__(self, **params):
        self.name = params.get('name')
        self.wrapper = params.get('wrapper')
        self.objective = params.get('objective')
        self.column_target: Optional[str] = None
        self.column_exposure: Optional[str] = None
        self.relative_features: Optional[List]= None
        self.features: List[FeatureModelConfig]
        self.intersections:[] = None
        self.params_catboost: Optional[Dict[str, Optional[Union[int, float, str, bool]]]] = None
        self.params_glm:  Optional[Dict[str, Optional[Union[int, float, str, bool]]]] = None
        self.treatment_dict: Optional[Dict[str, str]] = None
        self.cat_features_catboost: Optional[List[str]] = None  # TODO: move to features
        self.data_filter_condition: Optional[str] = None


class FeatureBuilder(ConfigBuilder):
    def __init__(self, **params):
        """
                 name: str,
                 feature_values,
                 type: str,
                 default_value=None,
                 replace_map: dict = None,
                 cut_number=None,
                 encoding=None,
                 clip: dict = None,
                 fillna=None,
                 optbinning_params=None,
                 bins=None,
                 mapping=None,
                 ):
        """
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
                                  bin=self.bins,
                                  mapping=self.mapping,
                                  )

    def __get_replace_map(self):
        if self.type == FeaturesTypes.numerical:
            return {"_TYPE_": "_NUM_"}
        else:
            if self.type == FeaturesTypes.categorical:
                return dict(
                    (value, FeatureEngineering.not_changed) for value in list(self.feature_values.unique()))
