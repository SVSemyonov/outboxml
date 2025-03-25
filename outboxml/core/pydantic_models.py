from pydantic import BaseModel, model_validator
from typing import List, Dict, Any, Optional, Union
from typing_extensions import Literal

from outboxml.core.enums import FilesNames, SeparationParams, FeatureEngineering
from outboxml.core.errors import ConfigError


class SeparationModelConfig(BaseModel):
    kind: Literal[SeparationParams.rand, SeparationParams.period, SeparationParams.null]
    random_state: Optional[int] = None
    test_train_proportion: Optional[float] = None
    train_period: Optional[List] = None
    test_period: Optional[List] = None
    period_column: Optional[List[str]] = None


# TODO: remove class
class DataConfig(BaseModel):
    targetcolumns: List[List]  = None
    extraColumns: Optional[List] = None
    targetslices: Optional[List] = None
#    extraDicts: Optional[List[ExtraDictConfig]] = None
    queries: Optional[List[str]] = None


class DataModelConfig(BaseModel):
    project: Optional[str] = None
    version: Optional[str] = None
    processing: Optional[bool] = None
    showGraphs: Optional[bool] = None
    source: Optional[Literal[FilesNames.parquet, FilesNames.database, FilesNames.csv, FilesNames.pickle, FilesNames.hadoop]]
    table_name_source: Optional[str] = None
    local_name_source: Optional[str] = None
    extra_conditions: Optional[str] = None
    extra_params: Optional[Dict] = None
    separation: Optional[SeparationModelConfig] = None
    extra_columns: Optional[List[str]] = None
    targetslices: Optional[List] = None
    data: Optional[DataConfig] = None  # TODO: remove


class RelativeFeatureModelConfig(BaseModel):
    name: str
    numerator: str
    denominator: str
    default: Any


class FeatureModelConfig(BaseModel):
    name: str
    default: Union[int, float, str]
    replace: Dict
    clip: Optional[Dict] = None
    cut_number: Optional[str] = None
    fillna: Optional[Union[int, float, str]] = None
    encoding: Optional[str] = None
    optbinning_params: Optional[Dict[str, Optional[Union[int, float, str, bool]]]] = None
    bins: Optional[List] = None
    mapping: Optional[Dict] = None

    @model_validator(mode="after")
    def check_replace(self):
        # Numerical
        if self.replace.get(FeatureEngineering.feature_type) == FeatureEngineering.numerical:
            for key, val in self.replace.items():
                if (
                    key != FeatureEngineering.feature_type
                    and val != FeatureEngineering.nan
                ):
                    try:
                        float(val)
                    except ValueError:
                        raise ConfigError(f"{self.name}: invalid value for replace in numerical feature")

        # Categorical
        else:
            for key, val in self.replace.items():
                if (
                    key != FeatureEngineering.feature_type
                    and val != FeatureEngineering.nan
                    and not isinstance(val, (str, int))
                ):
                    raise ConfigError(f"{self.name}: invalid value for replace in categorical feature")

        return self


class IntersectionModelConfig(BaseModel):
    name: str
    features_to_intersect: List[str]


class ModelConfig(BaseModel):
    name: str
    objective: Optional[str] = None
    wrapper: Optional[str] = None
    column_target: Optional[str] = None
    column_exposure: Optional[str] = None
    relative_features: Optional[List[RelativeFeatureModelConfig]] = None
    features: List[FeatureModelConfig]
    intersections: Optional[List[IntersectionModelConfig]] = None
    params_catboost: Optional[Dict[str, Optional[Union[int, float, str, bool]]]] = None
    params_glm:  Optional[Dict[str, Optional[Union[int, float, str, bool]]]] = None
    params: Optional[Dict[str, Optional[Union[int, float, str, bool]]]] = None
    treatment_dict: Optional[Dict[str, str]] = None
    cat_features_catboost: Optional[List[str]] = None  # TODO: move to features
    data_filter_condition: Optional[str] = None

    @model_validator(mode="after")
    def check_unique_features_names(self):
        features_names = [feature.name for feature in self.features]
        seen = set()
        for name in features_names:
            if name in seen:
                raise ConfigError("Not unique features names")
            else:
                seen.add(name)
        return self


class AllModelsConfig(BaseModel):
    project: str
    version: str
    group_name: str
    data_config: DataModelConfig
    models_configs: List[ModelConfig]

    @model_validator(mode="after")
    def check_unique_models_names(self):
        models_names = [model.name for model in self.models_configs]
        seen = set()
        for name in models_names:
            if name in seen:
                raise ConfigError("Not unique models names")
            else:
                seen.add(name)
        return self


class ServiceRequest(BaseModel):
    main_model: str
    main_request: List[Dict]
    second_model: Optional[str] = None
    second_request: Optional[List[Dict]] = None


class FeatureSelectionConfig(BaseModel):
    top_feautures_to_select: int = 10
    count_category: int = 100,
    cutoff_1_category: float = 0.99,
    cutoff_nan: float = 0.7,
    max_corr_value: float = 0.6
    metric_eval: dict
    cv_diff_value: float = 0.05
    encoding_cat: str = 'WoE_cat_to_num'
    encoding_num: str = 'WoE_num_to_num'
    features_to_ignore: List[str] = []
    params: dict = {}


class HPTuneConfig(BaseModel):
    sampling: str ='TPE'
    cv_folds_num: int = 3
    parameters: dict = None
    metric_score: Dict[str, str] = None


class ModelInferenceConfig(BaseModel):
    prod_models_folder: str
    metric_growth_value: Dict[str, float]
    calculate_threshold: int = 0
    threshold: list = [0.8, 0.8]
    prod_path: Optional[str] = None



class AutoMLConfig(BaseModel):
    project: str
    group_name: str
    feature_selection: FeatureSelectionConfig
    hp_tune: HPTuneConfig
    inference_criteria: ModelInferenceConfig
    mlflow_experiment: str
    grafana_table_name: str
    dashboard_name: str
    trigger: Optional[Dict[str, str]] = None


class MonitoringConfig(BaseModel):
    group_name: str
    prod_models_path: str
    pickle_name: str
    grafana_table_name: str
    dashboard_name: str
    extrapolation_period: int = 12
    target_column: str
