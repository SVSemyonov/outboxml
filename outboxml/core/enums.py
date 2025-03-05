from enum import Enum


class FilesNames(str, Enum):
    data_init = "data_init"
    parquet = "parquet"
    database = "database"
    pickle = "pickle"
    csv = "csv"
    hadoop = "hadoop"


class FeatureEngineering(str, Enum):
    feature_type = "_TYPE_"
    numerical = "_NUM_"
    not_changed = "_NOTCHANGED_"
    min_value = "min_value"
    max_value = "max_value"
    nan = "_NAN_"
    min = "_MIN_"
    max = "_MAX_"
    mean = "_MEAN_"
    median = "_MEDIAN_"


class FeaturesTypes(str, Enum):
    feature = "feature"
    target_encoding = "target_encoding"
    numerical = "numerical"
    categorical = "categorical"


class ModelsParams(str, Enum):
    poisson = "poisson"
    gamma = "gamma"
    binary = "binary"
    rmse = 'RMSE'
    rmsewithuncertainty = "rmsewithuncertainty"
    glm = "glm"
    glm_without_scaler = "glm_without_scaler"
    catboost = "catboost"
    catboost_over_glm = "catboost_over_glm"


class ResultNames:
    model_name = 'Имя модели'
    metric = 'Метрика'
    new_model = 'Новая модель'
    new_result_train = 'Новая модель||Тренировочная выборка'
    new_result_test = 'Новая модель||Тестовая выборка'
    old_result_train = 'Предыдущая модель||Тренировочная выборка'
    old_result_test = 'Предыдущая модель||Тестовая выборка'


class SeparationParams(str, Enum):
    rand = "random"
    period = "date"
    null = "none"


class ColumnsNames(str, Enum):
    is_train = "is_train"
    is_test = "is_test"


class EncodingNames(str, Enum):

    woe_num = "WoE_num_to_cat"
    woe_cat = "WoE_cat_to_num"
    to_float = "to_float"
    to_int = "to_int"
    woe_num_num = "WoE_num_to_num"


class FeatureTypesForSelection(str, Enum):
    binary = "BINARY"#: ['EventCreatedByGIBDDFlag', 'E-Garant', < ... >]
    categorical = "CATEGORIAL"#: ['CustomerImportance', 'DTPOSAGOType', < ... >]
    numeric = "NUMERIC"#: ['LossNumber', 'InsuredSum', 'LossDateTime', < ... >]
    to_drop = "TO_DROP"#: ['EventTypeDescription', 'InsuranceTypeName', < ... >]
    object =  "OBJECT"#: ['ContractNumber', 'VictimContractNumber', < ... >]