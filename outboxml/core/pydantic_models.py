from pydantic import AliasChoices, BaseModel, Field, model_validator
import warnings
from typing import List, Dict, Any, Optional, Union
from typing_extensions import Literal

from outboxml.core.enums import FilesNames, SeparationParams, FeatureEngineering
from outboxml.core.errors import ConfigError


def warn_deprecated_fields(data: Any, model_name: str, fields: List[str]) -> Any:
    """Warn about config fields that are accepted for compatibility but not used anymore."""
    if isinstance(data, dict):
        for field in fields:
            if data.get(field) is not None:
                warnings.warn(f"{model_name}.{field} is deprecated and ignored, remove it from the config",
                              DeprecationWarning, stacklevel=2)
    return data


class SeparationModelConfig(BaseModel):
    kind: Literal[SeparationParams.rand, SeparationParams.period, SeparationParams.null] = SeparationParams.rand
    random_state: Optional[int] = 42
    test_train_proportion: Optional[float] = 0.2
    train_period: Optional[List] = None
    test_period: Optional[List] = None
    period_column: Optional[List[str]] = None


# TODO: remove class
class DataConfig(BaseModel):

    targetslices: Optional[List]  = []
    queries: Optional[List[str]] = None  # deprecated

    @model_validator(mode="before")
    @classmethod
    def deprecated_fields(cls, data):
        return warn_deprecated_fields(data, "DataConfig", ["queries"])


class DataModelConfig(BaseModel):
    project: Optional[str] = None
    version: Optional[str] = None
    processing: Optional[bool] = None  # deprecated
    showGraphs: Optional[bool] = None  # deprecated
    source: Optional[Literal[FilesNames.parquet, FilesNames.database, FilesNames.csv, FilesNames.pickle, FilesNames.hadoop]] = None
    table_name_source: Optional[str] = None
    local_name_source: Optional[str] = None
    extra_conditions: Optional[str] = None
    extra_params: Optional[Dict] = None
    separation: Optional[SeparationModelConfig] = Field(default_factory=SeparationModelConfig)
    extra_columns: Optional[List[str]] = None
    targetslices: Optional[List] = []
    data: Optional[DataConfig] = Field(default_factory=DataConfig)  # TODO: remove

    @model_validator(mode="before")
    @classmethod
    def deprecated_fields(cls, data):
        return warn_deprecated_fields(data, "DataModelConfig", ["processing", "showGraphs"])


class RelativeFeatureModelConfig(BaseModel):
    name: str
    numerator: str
    denominator: str
    default: int | float | Literal[FeatureEngineering.nan] = FeatureEngineering.nan

    @model_validator(mode="after")
    def check_rel_default(self):
        if not (self.default == FeatureEngineering.nan or isinstance(self.default, (float, int))):
            raise ConfigError(f"{self.name}: invalid default value for numerical feature {self.default}")
        return self


class FeatureModelConfig(BaseModel):
    name: str
    default: Union[int, float, str] = None
    replace: Dict = Field(default_factory=dict)
    clip: Optional[Dict] = None
    cut_number: Optional[str] = None
    fillna: Optional[Union[int, float, str]] = None  # For categorical features only
    encoding: Optional[str] = None
    optbinning_params: Optional[Dict[str, Optional[Union[int, float, str, bool]]]] = None
    bins: Optional[List] = None
    mapping: Optional[Dict] = None

    @model_validator(mode="after")
    def set_default(self):
        """Without `default`: the median for numerical features, NaN for categorical ones."""
        if self.default is None:
            if self.replace.get(FeatureEngineering.feature_type) == FeatureEngineering.numerical:
                self.default = FeatureEngineering.median.value
            else:
                self.default = FeatureEngineering.nan.value
        return self

    @model_validator(mode="after")
    def check_replace(self):
        # Numerical
        if self.replace.get(FeatureEngineering.feature_type) == FeatureEngineering.numerical:
            for key, val in self.replace.items():
                if (
                    key != FeatureEngineering.feature_type
                    and val not in [FeatureEngineering.nan,  FeatureEngineering.median,
                                    FeatureEngineering.mean, FeatureEngineering.max,   FeatureEngineering.min]
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

    @model_validator(mode="after")
    def check_default(self):
        # Numerical
        if self.replace.get(FeatureEngineering.feature_type) == FeatureEngineering.numerical:
            if not (
                self.default in [
                    FeatureEngineering.nan,
                    FeatureEngineering.median,
                    FeatureEngineering.mean,
                    FeatureEngineering.max,
                    FeatureEngineering.min,
                ]
                or isinstance(self.default, (float, int))
            ):
                raise ConfigError(f"{self.name}: invalid default value for numerical feature")

        # Categorical
        else:
            if not isinstance(self.default, (str, int)):
                raise ConfigError(f"{self.name}: invalid default value for categorical feature")
            if not (isinstance(self.fillna, (str, int)) or self.fillna is None):
                raise ConfigError(f"{self.name}: invalid fillna value for categorical feature")

        return self

    # @model_validator(mode="after")
    # def check_clip(self):
    #     # Numerical
    #     if self.replace.get(FeatureEngineering.feature_type) == FeatureEngineering.numerical:
    #         if not self.clip:
    #             raise ConfigError(f"{self.name}: clip value is required for numerical feature")
    #         else:
    #             for key in [FeatureEngineering.min_value, FeatureEngineering.max_value]:
    #                 val = self.clip.get(key)
    #                 if not isinstance(val, (float, int)):
    #                     raise ConfigError(f"{self.name}: invalid clip value for numerical feature")
    #
    #     return self


class IntersectionModelConfig(BaseModel):
    name: str
    features_to_intersect: List[str]


class ModelConfig(BaseModel):
    name: str
    objective: Optional[str] = None
    wrapper: Optional[str] = None
    column_target: Optional[str] = None
    column_exposure: Optional[str] = None
    column_weight: Optional[str] = None
    relative_features: Optional[List[RelativeFeatureModelConfig]] = []
    features: List[FeatureModelConfig]
    intersections: Optional[List[IntersectionModelConfig]] = None
    params_catboost: Optional[Dict[str, Optional[Union[int, float, str, bool]]]] = None
    params_xgb: Optional[Dict[str, Optional[Union[int, float, str, bool]]]] = None
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
    project: str = "default"
    version: str = "1"
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
    # 'top_feautures_to_select' is the old misspelled name, still accepted
    top_features_to_select: int = Field(default=10, validation_alias=AliasChoices("top_features_to_select",
                                                                                 "top_feautures_to_select"))
    count_category: int = 100
    cutoff_1_category: float = 0.99
    cutoff_nan: float = 0.7
    max_corr_value: float = 0.6
    metric_eval: dict = {'metric_name': 0}
    cv_diff_value:  Optional[float] = None
    encoding_cat: str = 'WoE_cat_to_num'
    encoding_num: str = 'WoE_num_to_num'
    default_cat: str = '_NAN_'
    default_num: str = '_MEDIAN_'
    depth: float = 0.01
    features_to_ignore: List[str] = []
    params: dict = {}
    use_temp_data: bool = False

    @model_validator(mode="before")
    @classmethod
    def deprecated_fields(cls, data):
        return warn_deprecated_fields(data, "FeatureSelectionConfig", ["top_feautures_to_select"])


class OptunaOptimizeConfig(BaseModel):
    type: Literal["int", "float", "categorical"]
    low: Optional[Union[int, float]] = None
    high: Optional[Union[int, float]] = None
    step: Optional[Union[int, float]] = None
    log: bool = False
    choices: Optional[List[Union[int, float, str, bool]]] = None

class HPTuneConfig(BaseModel):
    sampling: str ='TPE'
    cv_folds_num: int = 3
    parameters: Optional[Dict[str, Dict[str, OptunaOptimizeConfig]]] = None
    n_jobs: int = -1
    trials: int = 100
    metric_score: Dict[str, str] = {"default": "neg_mean_absolute_error"}


class ModelInferenceConfig(BaseModel):
    prod_models_folder: Optional[str] = None  # deprecated: the folder comes from the environment config
    metric_growth_value: Dict[str, float] = Field(default_factory=dict)
    calculate_threshold: int = 0
    threshold: list = [0.8, 0.8]
    prod_path: Optional[str] = None  # deprecated

    @model_validator(mode="before")
    @classmethod
    def deprecated_fields(cls, data):
        return warn_deprecated_fields(data, "ModelInferenceConfig", ["prod_models_folder", "prod_path"])


class AutoMLConfig(BaseModel):
    # project and group_name default to the models config (AllModelsConfig)
    project: Optional[str] = None
    group_name: Optional[str] = None
    feature_selection: FeatureSelectionConfig = Field(default_factory=FeatureSelectionConfig)
    hp_tune: HPTuneConfig = Field(default_factory=HPTuneConfig)
    inference_criteria: ModelInferenceConfig = Field(default_factory=ModelInferenceConfig)
    # mlflow_experiment defaults to the environment config, grafana_table_name to the AutoML group name
    mlflow_experiment: Optional[str] = None
    grafana_table_name: Optional[str] = None
    dashboard_name: Optional[str] = None  # deprecated
    trigger: Optional[Dict[str, str]] = None

    @model_validator(mode="before")
    @classmethod
    def deprecated_fields(cls, data):
        return warn_deprecated_fields(data, "AutoMLConfig", ["dashboard_name"])

class MonitoringFactoryConfig(BaseModel):
    type: str = "datadrift"
    report: str = "base_datadrift_report"
    group_models: Optional[bool] = False
    parameters: Optional[Dict[str, Any]] = {}
    db_table_name: Optional[str] = None

class MonitoringConfig(BaseModel):
    group_name: str
    grafana_table_name: Optional[str] = None
    dashboard_name: Optional[str] = None  # deprecated
    monitoring_factory: List[MonitoringFactoryConfig] = [MonitoringFactoryConfig()]
    extrapolation_period: int = 12  # deprecated
    target_column: Optional[str] = None  # deprecated
    data_source: Optional[str] = None  # defaults to the source of the models config

    @model_validator(mode="before")
    @classmethod
    def deprecated_fields(cls, data):
        return warn_deprecated_fields(data, "MonitoringConfig",
                                      ["dashboard_name", "extrapolation_period", "target_column"])

    @model_validator(mode="after")
    def set_grafana_table_name(self):
        if self.grafana_table_name is None:
            self.grafana_table_name = self.group_name
        return self
    prod_models_path: Optional[str] = None
    pickle_name: Optional[str] = None
    report_db_schema: Optional[str] = "public"

class UpdateRequest(BaseModel):
    auto_ml_config: AutoMLConfig
    all_model_config: AllModelsConfig
    user_parameters: Optional[Dict[str, Optional[Union[int, float, str, bool]]]] = None


class MonitoringRequest(BaseModel):
    all_model_config: AllModelsConfig
    monitoring_config: MonitoringConfig
    user_parameters: Optional[Dict[str, Optional[Union[int, float, str, bool]]]] = None


class AutoMLResultRequest(BaseModel):
    main_model: str
    request: Dict[str, bool]


class MonitoringResultRequest(BaseModel):
    main_model: str
    request: Dict[str, bool]
