import pandas as pd
from pandas.api.types import is_integer_dtype, is_float_dtype
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from typing_extensions import Literal
from itertools import chain
import os
from loguru import logger
from optbinning import ContinuousOptimalBinning

from outboxml.core.pydantic_models import (
    ModelConfig,
    FeatureModelConfig,
)
from outboxml.core.enums import FeatureEngineering, FeaturesTypes
from outboxml.core.utils import (
    update_model_config_default,
    update_model_config_replace,
    find_drop_values,
)
from outboxml.core.errors import ConfigError
from outboxml.core.enums import EncodingNames


class OptiBinningEncoder:
    def __init__(self,
                 X: pd.Series,
                 y: pd.Series,
                 type: str,
                 name: str,
                 train_ind: pd.Index,
                 ):
        self._X = X
        self._y = y
        self._type = type
        self._name = name
        self.mapping = {}
        self._train_ind = train_ind

    def encode_data(self, mapping: dict = None, bins: np.array = None, num_num: bool = False, optbinning_params: dict = None) -> tuple:

        if self._type == 'numerical':
            try:
                self._X = self._X.astype('float')
            except:
                logger.error('Wrong type of X for binning')
        if (mapping is None) and (bins is None):
            if optbinning_params is None:
                optbinning_params =  {}
            else:
                logger.info('User Optbinning params')
            optb = ContinuousOptimalBinning(name=self._name, dtype=self._type, **optbinning_params)

            optb.fit(
                self._X.loc[[i for i in self._train_ind if i in self._X.index]],
                self._y.loc[[i for i in self._train_ind if i in self._X.index]]
            )
            t = optb.binning_table.build()
            mapping = {}
            if self._type == 'categorical':

                for l, woe in t.loc[~t['Bin'].astype(str).isin(['Special', 'Missing', '']), ['Bin', 'WoE']].values:
                    for i in l:
                        mapping[i] = woe
            elif self._type == 'numerical':
                if not num_num:
                    bins = np.concatenate([[-np.inf], optb.binning_table.splits, [np.inf]])
                else:
                    bins = np.concatenate([[-np.inf], optb.binning_table.splits, [np.inf]])
                    optb2 = ContinuousOptimalBinning(name=self._name, dtype='categorical',  **optbinning_params)

                    optb2.fit(
                        pd.cut(self._X.loc[[i for i in self._train_ind if i in self._X.index]], bins=bins),
                        self._y.loc[[i for i in self._train_ind if i in self._X.index]]
                    )
                    t2 = optb2.binning_table.build()
                    for l, woe in t2.loc[~t2['Bin'].astype(str).isin(['Special', 'Missing', '']), ['Bin', 'WoE']].values:
                        for i in l:
                            mapping[i] = woe
        if isinstance(bins, np.ndarray):
            bins = bins.tolist()
        if mapping is not None:
            if len(mapping) == 1:
                logger.warning('Invalid WoE optibinnig params for feature ' + self._name + '||[-inf, inf] interval')
        return mapping, bins


class PrepareDatasetResult:
    def __init__(
            self,
            data: pd.DataFrame,
            features_numerical: Optional[List[str]],
            features_categorical: Optional[List[str]],
            model_config: ModelConfig,
            corr_df: Optional[pd.DataFrame] = None,
            encoding_map: dict = None
    ):
        self.data: pd.DataFrame = data
        self.features_numerical: Optional[List[str]] = features_numerical
        self.features_categorical: Optional[List[str]] = features_categorical
        self.model_config: ModelConfig = model_config
        self.corr_df: Optional[pd.DataFrame] = corr_df
        self.encoding_map = encoding_map


def map_num(v: Union[int, float], bins: List[float], mapping: Dict[pd.IntervalIndex, str]) -> Optional[str]:
    for i, j in zip(bins[:-1], bins[1:]):
        if (i < v) and (v <= j):
            for k, m in mapping.items():
                if k.left == i:
                    return m
    return None


def feature_encoding_series(
        feature_data: pd.Series,
        feature: FeatureModelConfig,
        target: pd.Series = None,
        train_ind: Optional[pd.Index] = None,
        log: bool = True,
        raise_on_error: bool = False,
) -> Tuple[pd.Series, Optional[Dict], Optional[List]]:
    mapping = feature.mapping
    bins = feature.bins
    if train_ind is None:
        train_ind = feature_data.index

    if feature.encoding == EncodingNames.to_float:
        if log:
            logger.info(f"{feature.name} || Encoding || To float")
        try:
            feature_data = feature_data.astype("float")
        except ValueError as e:
            logger.error(f"{feature.name} || Encoding error || Cannot convert to float || {str(e)}")
            if raise_on_error:
                raise ValueError(f"{feature.name} || Cannot convert to float")

    elif feature.encoding == EncodingNames.to_int:
        if log:
            logger.info(f"{feature.name} || Encoding || To int")
        try:
            feature_data = feature_data.astype("int")
        except ValueError as e:
            logger.error(f"{feature.name} || Encoding error || Cannot convert to int || {str(e)}")
            if raise_on_error:
                raise ValueError(f"{feature.name} || Cannot convert to int")


    elif feature.encoding == EncodingNames.woe_cat and not target.empty:
        if log:
            logger.info(f"{feature.name} || Encoding || WoE categorical to numerical")
        if feature.mapping is not None:
            feature_data = feature_data.map(mapping).astype("float")
        else:
            try:
                mapping, bins = OptiBinningEncoder(
                    X=feature_data,
                    y=target,
                    type="categorical",
                    name=feature.name,
                    train_ind=train_ind,
                ).encode_data(mapping=mapping, bins=bins, optbinning_params=feature.optbinning_params)
                feature_data = feature_data.map(mapping).astype("float")
            except Exception as e:
                logger.error(f"{feature.name} || Encoding error || Cannot convert to WoE || {str(e)}")
                if raise_on_error:
                    raise ValueError(f"{feature.name} || Cannot convert to WoE")

    elif feature.encoding == EncodingNames.woe_num and not target.empty:
        if log:
            logger.info(f"{feature.name} || Encoding || WoE numerical to categorical")
        if feature.bins is not None:
            feature_data = feature_data.astype("float")
            feature_data = pd.cut(feature_data, bins=feature.bins)
        else:
            try:
                feature_data = feature_data.astype("float")
                mapping, bins = OptiBinningEncoder(
                    X=feature_data,
                    y=target,
                    type="numerical",
                    name=feature.name,
                    train_ind=train_ind,
                ).encode_data(mapping=mapping, bins=bins, optbinning_params=feature.optbinning_params)
                feature_data = pd.cut(feature_data, bins=bins)
            except Exception as e:
                logger.error(f"{feature.name} || Encoding error || Cannot convert to WoE || {str(e)}")
                if raise_on_error:
                    raise ValueError(f"{feature.name} || Cannot convert to WoE")

    elif feature.encoding == EncodingNames.woe_num_num and not target.empty:
        if log:
            logger.info(f"{feature.name} || Encoding || WoE numerical to numerical")
        if feature.bins is not None and feature.mapping is not None:
            feature_data = feature_data.astype("float")
            feature_data = pd.cut(feature_data, bins=bins).map(mapping).astype("float")
        else:
            try:
                feature_data = feature_data.astype("float")
                mapping, bins = OptiBinningEncoder(
                    X=feature_data,
                    y=target,
                    type="numerical",
                    name=feature.name,
                    train_ind=train_ind,
                ).encode_data(mapping=mapping, bins=bins, num_num=True,  optbinning_params=feature.optbinning_params)
                feature_data = pd.cut(feature_data, bins=bins).map(mapping).astype("float")
            except Exception as e:
                logger.error(f"{feature.name} || Encoding error || Cannot convert to WoE || {str(e)}")
                if raise_on_error:
                    raise ValueError(f"{feature.name} || Cannot convert to WoE")
    else:
        logger.info("Unknown encoding || Return origin")
        if raise_on_error:
            raise NotImplementedError(f"{feature.name} || Unknown encoding")

    return feature_data, mapping, bins


def feature_encoding(
        feature_value: Union[int, float],
        feature: FeatureModelConfig,
) -> Union[int, float]:

    if feature.encoding == EncodingNames.to_float:
        try:
            feature_value = float(feature_value)
        except ValueError as e:
            raise ValueError(f"{feature.name} || Cannot convert to float")

    elif feature.encoding == EncodingNames.to_int:
        try:
            feature_value = int(feature_value)
        except ValueError as e:
            raise ValueError(f"{feature.name} || Cannot convert to int")

    elif feature.encoding == EncodingNames.woe_cat:
        if feature.mapping is None:
            raise NotImplementedError(f"{feature.name} || Invalid mappings")
        try:
            feature_value = feature.mapping.get(feature_value)
        except Exception as e:
            raise ValueError(f"{feature.name} || Cannot convert to WoE")

    elif feature.encoding == EncodingNames.woe_num:
        if feature.bins is None:
            raise NotImplementedError(f"{feature.name} || Invalid bins")
        try:
            if not isinstance(feature_value, (float, int)):
                feature_value = float(feature_value)
            mapping = {
                k: v for k, v in zip(
                    pd.IntervalIndex.from_arrays(feature.bins[:-1], feature.bins[1:]),
                    list(pd.IntervalIndex.from_arrays(feature.bins[:-1], feature.bins[1:]).astype(str))
                )
            }
            feature_value = map_num(feature_value, feature.bins, mapping)
        except Exception as e:
            raise ValueError(f"{feature.name} || Cannot convert to WoE")

    elif feature.encoding == EncodingNames.woe_num_num:
        if feature.bins is None or feature.mapping is None:
            raise NotImplementedError(f"{feature.name} || Invalid mappings or bins")
        try:
            if not isinstance(feature_value, (float, int)):
                feature_value = float(feature_value)
            feature_value = map_num(feature_value, feature.bins, feature.mapping)
        except Exception as e:
            raise ValueError(f"{feature.name} || Cannot convert to WoE")

    else:
        raise NotImplementedError(f"{feature.name} || Unknown encoding")

    return feature_value


def dict_replace(
        feature: FeatureModelConfig,
        dtype: Literal[FeaturesTypes.numerical, FeaturesTypes.categorical]
) -> Dict:

    dict_replace_temp = {}

    for key, val in feature.replace.items():
        if (
            key != FeatureEngineering.feature_type
            and val != FeatureEngineering.nan
        ):
            if val != FeatureEngineering.not_changed:
                dict_replace_temp[key] = val
                try:
                    key_float = float(key)
                    dict_replace_temp[key_float] = val
                except ValueError:
                    pass
            else:
                dict_replace_temp[key] = key
        elif val == FeatureEngineering.nan:
            dict_replace_temp[key] = np.nan
            try:
                key_float = float(key)
                dict_replace_temp[key_float] = np.nan
            except ValueError:
                pass

    return dict_replace_temp


def replace_categorical_values_series(
        feature_data: pd.Series,
        feature: FeatureModelConfig,
) -> pd.Series:
    dict_replace_temp = dict_replace(feature=feature, dtype=FeaturesTypes.categorical)
    ind = (~feature_data.isin(dict_replace_temp) & pd.notnull(feature_data))
    feature_data = feature_data.map(dict_replace_temp)
    feature_data.loc[ind] = feature.default

    return feature_data


def replace_categorical_values(
        feature_value: Union[int, str],
        feature: FeatureModelConfig,
) -> Union[int, float, str]:
    dict_replace_temp = dict_replace(feature=feature, dtype=FeaturesTypes.categorical)
    if pd.isnull(feature_value):
        return np.nan
    return dict_replace_temp.get(feature_value, feature.default)


def replace_numerical_values_series(
        feature_data: pd.Series,
        feature: FeatureModelConfig,
) -> pd.Series:
    dict_replace_temp = dict_replace(feature=feature, dtype=FeaturesTypes.numerical)
    return feature_data.replace(dict_replace_temp)


def replace_numerical_values(
        feature_value: Union[int, float],
        feature: FeatureModelConfig,
) -> Union[int, float]:
    dict_replace_temp = dict_replace(feature=feature, dtype=FeaturesTypes.numerical)
    return dict_replace_temp.get(feature_value, feature_value)


def replace_with_default(feature_data: pd.Series, feature: FeatureModelConfig, values: List[str]) -> pd.Series:
    dict_replace_temp = {value: feature.default for value in values}
    return feature_data.replace(dict_replace_temp)


def prepare_relative_feature_series(
        numerator: pd.Series,
        denominator: pd.Series,
        default_value: Union[float, int],
) -> pd.Series:
    if not isinstance(default_value, (int, float)):
        raise ConfigError("Invalid default value for relative feature")
    return (numerator / denominator).replace([-np.inf, np.inf], np.nan).fillna(default_value)


def prepare_relative_feature(
        numerator: Union[float, int],
        denominator: Union[float, int],
        default_value: Union[float, int],
) -> Union[float, int]:
    if not isinstance(default_value, (int, float)):
        raise ConfigError("Invalid default value for relative feature")
    if (
        pd.isnull(numerator)
        or pd.isnull(denominator)
        or denominator == 0
    ):
        feature_value = default_value
    else:
        feature_value = numerator / denominator
    return feature_value


def to_str(v):
    if pd.isnull(v):
        v = None
    elif isinstance(v, (int, float)):
        if v % 1 == 0:
            v = str(int(v))
        else:
            v = str(v)
    else:
        v = v.upper()
    return v
"""
def replace_categorical_values_pl(self,
        feature_data: pl.LazyFrame,
        feature: FeatureModelConfig,
) -> pl.LazyFrame:
    dict_replace_temp = dict_replace(feature=feature, dtype=FeaturesTypes.categorical)
    feature_data = (
        feature_data
        .with_columns(
            pl.when(~pl.col(feature.name).is_in(dict_replace_temp) & pl.col(feature.name).is_not_null())
            .then(pl.lit(feature.default))
            .otherwise(pl.col(feature.name).replace(dict_replace_temp))
            .alias(feature.name)
        )
    )
    return feature_data
"""

def prepare_categorical_feature_series(
        feature_data: pd.Series,
        feature: FeatureModelConfig,
        log: bool = True,
) -> pd.Series:

    feature_data = feature_data.apply(lambda x: to_str(x))

    if not isinstance(feature.default, (str, int)):
        raise ConfigError(f"{feature.name}: invalid default value for categorical feature")

    # Схлопывание до базового уровня и Замена значений
    if log:
        ind = (~feature_data.isin(feature.replace.keys()) & feature_data.notna())
        if len(feature_data.loc[ind]) > 0:
            logger.info(feature.name + ' || Присвоено default значений: ' + str(len(feature_data.loc[ind])))
    feature_data = replace_categorical_values_series(feature_data, feature)

    # Замена пропусков
    if pd.isnull(feature_data).sum() > 0:
        if log:
            logger.info(feature.name + ' || Исправлено пропусков: ' + str(feature_data.isna().sum()))
        if feature.fillna:
            if not isinstance(feature.fillna, (str, int)):
                raise ConfigError(f"{feature.name}: invalid fillna value for categorical feature")
            feature_data.fillna(feature.fillna, inplace=True)
        else:
            feature_data.fillna(feature.default, inplace=True)

    return feature_data


def prepare_categorical_feature(
        feature_value: Union[int, str],
        feature: FeatureModelConfig,
) -> Union[int, str]:
    feature_value = to_str(feature_value)

    if not isinstance(feature.default, (str, int)):
        raise ConfigError(f"{feature.name}: invalid default value for categorical feature")

    # Схлопывание до базового уровня и Замена значений
    feature_value = replace_categorical_values(feature_value, feature)

    # Замена пропусков
    if pd.isnull(feature_value):
        if feature.fillna:
            if not isinstance(feature.fillna, (str, int)):
                raise ConfigError(f"{feature.name}: invalid fillna value for categorical feature")
            feature_value = feature.fillna
        else:
            feature_value = feature.default

    return feature_value


def prepare_numerical_feature_series(
        feature_data: pd.Series,
        feature: FeatureModelConfig,
        train_ind: Optional[pd.Index] = None,
        log: bool = True,
) -> Tuple[pd.Series, Optional[Union[float, int]]]:
    if train_ind is None:
        train_ind = feature_data.index

    # Замена значений
    feature_data = replace_numerical_values_series(feature_data, feature)

    if feature_data.dtype not in (int, float):
        feature_data = pd.to_numeric(feature_data, downcast="float", errors="coerce")

    # Замена пропусков
    val_fill = None
    if feature.default == FeatureEngineering.min:
        val_fill = feature_data.loc[[i for i in train_ind if i in feature_data.index]].min()
    elif feature.default == FeatureEngineering.max:
        val_fill = feature_data.loc[[i for i in train_ind if i in feature_data.index]].max()
    elif feature.default == FeatureEngineering.mean:
        val_fill = feature_data.loc[[i for i in train_ind if i in feature_data.index]].mean()
    elif feature.default == FeatureEngineering.median:
        val_fill = feature_data.loc[[i for i in train_ind if i in feature_data.index]].median()

    default_value = val_fill if val_fill is not None else feature.default

    if not isinstance(default_value, (int, float)):
        raise ConfigError(f"{feature.name}: invalid default value for numerical feature")

    if pd.isnull(feature_data).sum() > 0:
        if log:
            logger.info(feature.name + ' || Исправлено пропусков: ' + str(feature_data.isna().sum()))
        feature_data.fillna(default_value, inplace=True)

    # Отсечение значений
    if feature.clip:
        if (sum(feature_data < feature.clip[FeatureEngineering.min_value])
                + sum(feature_data > feature.clip[FeatureEngineering.max_value])) > 0:
            if log:
                logger.info(feature.name + ' || Исправлено значений вне интервала: ' +
                            str(sum(feature_data < feature.clip[FeatureEngineering.min_value]) +
                                sum(feature_data > feature.clip[FeatureEngineering.max_value])))

            feature_data.clip(
                feature.clip[FeatureEngineering.min_value],
                feature.clip[FeatureEngineering.max_value],
                inplace=True,
            )

    # Группировка
    if feature.cut_number:
        val_splits = [-np.inf] + list([float(x) for x in feature.cut_number.split('_')]) + [np.inf]
        feature_data = pd.cut(feature_data, bins=val_splits).astype(str)

    return feature_data, val_fill


def prepare_numerical_feature(
        feature_value: Union[float, int, str],
        feature: FeatureModelConfig,
) -> Union[float, int, str]:

    # Замена значений
    feature_value = replace_numerical_values(feature_value, feature)

    if not isinstance(feature_value, (int, float)):
        try:
            feature_value = float(feature_value)
        except:
            feature_value = np.nan

    # Замена пропусков
    if not isinstance(feature.default, (int, float)):
        raise ConfigError(f"{feature.name}: invalid default value for numerical feature")

    if pd.isnull(feature_value):
        feature_value = feature.default

    # Отсечение значений
    if feature.clip:
        feature_value = (
            feature.clip[FeatureEngineering.min_value] if feature_value < feature.clip[FeatureEngineering.min_value]
            else feature.clip[FeatureEngineering.max_value] if feature_value > feature.clip[FeatureEngineering.max_value]
            else feature_value
        )

    # Группировка
    if feature.cut_number:
        val_splits = [-np.inf] + list([float(x) for x in feature.cut_number.split('_')]) + [np.inf]
        mapping = {
            k: v for k, v in zip(
                pd.IntervalIndex.from_arrays(val_splits[:-1], val_splits[1:]),
                list(pd.IntervalIndex.from_arrays(val_splits[:-1], val_splits[1:]).astype(str))
            )
        }
        feature_value = map_num(feature_value, val_splits, mapping)

    return feature_value


# def prepare_intersection(data: pd.DataFrame, feature_intersection: IntersectionModelConfig) -> pd.Series:
#     data[feature_intersection.name] = ""
#     for feature in feature_intersection.features_to_intersect:
#         data[feature_intersection.name] += '[' + data[feature].astype(str) + ']'
#
#     return data[feature_intersection.name]


def prepare_dataset(
        group_name: str,
        data: Union[Dict, pd.DataFrame],
        train_ind: Optional[pd.Index],
        test_ind: Optional[pd.Index],
        model_config: ModelConfig,
        check_prepared: bool = False,
        calc_corr: bool = False,
        save_data: bool = False,
        corr_threshold: Optional[float] = None,
        target: Optional[pd.Series] = None,
        log: bool = True,
        modify_dtypes: bool = True,
        raise_on_encoding_error: bool = False
) -> PrepareDatasetResult:

    as_dict: bool = False
    if isinstance(data, dict):
        as_dict = True
    elif isinstance(data, pd.DataFrame):
        pd.options.mode.chained_assignment = None
    else:
        raise TypeError("Invalid data type")

    if model_config.relative_features:
        for relative_feature in model_config.relative_features:
            if as_dict:
                data[relative_feature.name] = prepare_relative_feature(
                    numerator=data[relative_feature.numerator],
                    denominator=data[relative_feature.denominator],
                    default_value=relative_feature.default
                )
            else:
                data[relative_feature.name] = prepare_relative_feature_series(
                    numerator=data[relative_feature.numerator],
                    denominator=data[relative_feature.denominator],
                    default_value=relative_feature.default
                )

    for feature in model_config.features:
        if as_dict and feature.replace.get(FeatureEngineering.feature_type) == FeatureEngineering.numerical:
            data[feature.name] = prepare_numerical_feature(
                feature_value=data[feature.name],
                feature=feature,
            )

        elif as_dict and feature.replace.get(FeatureEngineering.feature_type) != FeatureEngineering.numerical:
            data[feature.name] = prepare_categorical_feature(
                feature_value=data[feature.name],
                feature=feature,
            )

        elif feature.replace.get(FeatureEngineering.feature_type) == FeatureEngineering.numerical:
            if train_ind is None:
                train_ind = data.index
            data[feature.name], val_fill = prepare_numerical_feature_series(
                feature_data=data[feature.name],
                feature=feature,
                train_ind=train_ind,
                log=log
            )
            if val_fill:
                model_config = update_model_config_default(model_config, feature.name, val_fill)

        elif feature.replace.get(FeatureEngineering.feature_type) != FeatureEngineering.numerical:
            data[feature.name] = prepare_categorical_feature_series(
                feature_data=data[feature.name],
                feature=feature,
                log=log
            )

    if check_prepared and not as_dict:
        logger.info('Find drop values for features')
        if train_ind is None:
            train_ind = data.index
        for feature in model_config.features:
            if feature.replace.get(FeatureEngineering.feature_type) != FeatureEngineering.numerical:
                replace_dict = dict_replace(feature=feature, dtype=FeaturesTypes.categorical)
                drop_values = find_drop_values(data[feature.name], replace_dict, train_ind)
                try:
                    if len(drop_values) > 0:
                        logger.info('Dropping unused levels in ' + feature.name + '||' + str(drop_values))
                        model_config = update_model_config_replace(model_config, {feature.name: drop_values})

                        data.loc[[i for i in train_ind if i in data.index]][feature.name] = replace_with_default(
                            data.loc[[i for i in train_ind if i in data.index]][feature.name], feature, drop_values
                        )
                except NotImplementedError as exc:
                    logger.error(exc)

    #FIXME Перевести внутрь цикла. Не записываются атрибут
    for feature in model_config.features:
        if feature.encoding is not None and as_dict:
            data[feature.name] = feature_encoding(
                feature_value=data[feature.name],
                feature=feature,
            )

        elif feature.encoding is not None:
            if log:
                logger.info('Feature preparation||Encoding from config ' + str(feature.encoding))
            data[feature.name], mapping, bins = feature_encoding_series(
                feature_data=data[feature.name],
                feature=feature,
                target=target,
                train_ind=train_ind,
                log=log,
                raise_on_error=raise_on_encoding_error,
            )
            feature.mapping = mapping
            feature.bins = bins

    # if model_config.intersections:
    #     for feature_intersection in model_config.intersections:
    #         data[feature_intersection.name] = prepare_intersection(
    #             data[feature_intersection.features_to_intersect], feature_intersection
    #         )

    features_all = list(set(chain(
        [feature.name for feature in model_config.relative_features] if model_config.relative_features else [],
        [feature.name for feature in model_config.features] if model_config.features else [],
        # [feature.name for feature in model_config.intersections] if model_config.intersections else [],
    )))
    if as_dict:
        features_categorical = [feature for feature in features_all if
                                isinstance(data[feature], str)]
        features_numerical = [feature for feature in features_all if
                              isinstance(data[feature], (int, float))]
    else:
        features_categorical = [feature for feature in features_all if
                                (data[feature].dtype == 'object' or data[feature].dtype == 'category')]
        features_numerical = [feature for feature in features_all if
                              (data[feature].dtype != 'object' and data[feature].dtype != 'category')]

    # Изменения типа выходных данных
    # TODO: добавить в конфиг автоматическое определение размера категорий
    if modify_dtypes and not as_dict:
        for column in features_all:
            # if data[column].dtype == 'object' or (check_prepared and column in nunUniqueTrainDict and nunUniqueTrainDict[column] < 30):
            if data[column].dtype == 'object':
                data[column] = data[column].astype("category")
            #             data[column] = data[column].astype("category")
            elif (is_integer_dtype(data[column].dtype)):
                data[column] = data[column].astype("int32")
            elif (is_float_dtype(data[column].dtype)):
                data[column] = data[column].astype("float32")

            if model_config.cat_features_catboost:
                if column in model_config.cat_features_catboost:
                    if is_float_dtype(data[column].dtype):
                        data[column] = data[column].astype("int32")
                    data[column] = data[column].astype("category")
                    if column not in features_categorical:
                        features_categorical.append(column)
                        features_numerical.remove(column)

    if save_data and not as_dict:
        results_path = model_config.results_path
        results_path.mkdir(exist_ok=True)
        group_path = results_path / group_name
        group_path.mkdir(exist_ok=True)
        model_path = model_config.results_path / group_name / model_config.name
        model_path.mkdir(exist_ok=True)
        data.to_parquet(
            os.path.join(model_path, model_config.name + "_data.parquet.gzip"),
            engine="pyarrow",
            compression="gzip",
        )

    if as_dict:
        data = pd.DataFrame([data])

    return PrepareDatasetResult(
        data=data[features_all],
        features_numerical=features_numerical,
        features_categorical=features_categorical,
        model_config=model_config,
        corr_df=pd.DataFrame()
    )
