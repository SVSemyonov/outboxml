import pandas as pd
from pandas.api.types import is_integer_dtype, is_float_dtype
import polars as pl
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
from outboxml.core.enums import FeatureEngineering, FeaturesTypes, ColumnsNames
from outboxml.core.utils import (
    update_model_config_default,
    update_model_config_replace,
    find_drop_values,
    find_drop_values_pl,
)
from outboxml.core.errors import ConfigError
from outboxml.core.enums import EncodingNames


class Encoder:
    """
    Base class for encoders.
    """
    def encode_data(self, *params):
        pass


class OptiBinningEncoder(Encoder):
    """
    Class for create and apply encodings. Uses ContinuousOptimalBinning method from the optbinning library.

    :param X: Feature values.
    :param y: Target values.
    :param type: The feature's type, `numerical` and `categorical` are supported.
    :param name: The feature's name.
    :param train_ind: Indices of training subset.

    Default optbinning_params: min_prebin_size=0.05, max_n_bins=5, max_n_prebins=20.
    """

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
        self._default_optbinning_params =  {'max_n_bins': 5, 'max_n_prebins': 20, 'min_prebin_size': 0.05,}

    def encode_data(
            self,
            mapping: Optional[dict] = None,
            bins: Optional[np.array] = None,
            num_num: bool = False,
            optbinning_params: Optional[dict] = None,
    ) -> tuple:
        """
        Creates bins and mappings if they are not given.

        :param mapping: External map values, not calculated if given.
        :param bins: External bins, not calculated if given.
        :param num_num: Calculates mappings for bins of numerical feature if True.
        :param optbinning_params: External optbinning params. Set default params if None.

        :return: Tuple of mappings and bins.
        """

        if self._type == 'numerical':
            try:
                self._X = self._X.astype('float')
            except:
                logger.error('Wrong type of X for binning')
        if (mapping is None) and (bins is None):
            if optbinning_params is None:
                optbinning_params = self._default_optbinning_params
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
                        pd.cut(self._X.loc[[i for i in self._train_ind if i in self._X.index]], bins=bins, precision=5),
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


class CutNumberEncoder(Encoder):
    """
    Class for create and apply cut encoding. The valuable strategy is Freedman-diaconis.

    :param max_bins: Maximum number of bins, 5 by default.
    :param rule: Cut rule, 'Freedman-diaconis' by default.
    :param round_decimals: Rounding decimals, 2 by default.
    """

    def __init__(self,
                 max_bins: int = 5,
                 rule: str = 'Freedman-diaconis',
                 round_decimals: int=2):
        self.max_bins = max_bins
        self.round_decimals = round_decimals
        self.rule = rule

    def encode_data(self, serie: pd.Series):
        """
        Creates bins.

        :param serie: Feature's values.

        :return: Bins.
        """

        opt_bins = self.calculate_optimal_bins(serie.dropna())
        cut_number = None
        if opt_bins is not None:
            result, bins = pd.qcut(serie, q=opt_bins, retbins=True, duplicates='drop')
            bins = np.round(bins, decimals=self.round_decimals)
            logger.info('bins_for_feature||' + str(bins))
            if len(bins) > 1:
                cut_number = '_'.join(map(str, bins[:-1]))
            else:
                cut_number = str(bins)
        return cut_number


    def calculate_optimal_bins(self, data: pd.Series):
        if self.rule == 'Freedman-diaconis':
            return self._freedman_diaconis_rule(data)
        else:
            logger.error('Unknown rule for cut number||Returnin None')
            return None


    def _freedman_diaconis_rule(self, data: pd.Series):
        try:
            n = len(data)
            iqr = np.percentile(data, 75) - np.percentile(data, 25)
            fd = int(np.ceil((max(data) - min(data)) / (2 * iqr / (n ** (1 / 3)))))

            return min(self.max_bins, fd)
        except Exception as exc:
            logger.error(f'No cut values for {data.name} return None||{str(exc)}')
            return None


class PrepareDatasetResult:
    """
    Class for prepared dataset.

    :param data: Pandas' or Polars' DataFrame with prepared features' values.
    :param features_numerical: List of numerical features' names.
    :param features_categorical: List of categorical features' names.
    :param model_config: Model's config.
    :param corr_df: Correlation matrix between features. (Deprecated)
    :param encoding_map: Mapping for features' bins. (Deprecated)
    """

    def __init__(
            self,
            data: pd.DataFrame | pl.DataFrame,
            features_numerical: Optional[List[str]],
            features_categorical: Optional[List[str]],
            model_config: ModelConfig,
            corr_df: Optional[pd.DataFrame] = None,
            encoding_map: Optional[dict] = None
    ):
        self.data: pd.DataFrame | pl.DataFrame = data
        self.features_numerical: Optional[List[str]] = features_numerical
        self.features_categorical: Optional[List[str]] = features_categorical
        self.model_config: ModelConfig = model_config
        self.corr_df: Optional[pd.DataFrame] = corr_df
        self.encoding_map = encoding_map


def map_num(v: Union[int, float], mapping: Dict[pd.IntervalIndex, str]) -> Optional[str]:
    """
    Apply mapping to value.

    :param v: Feature's value.
    :param mapping: Mapping.

    :return: Encoded value.
    """
    for k, m in mapping.items():
        if (k.left < v) and (v <= k.right):
            return m
    logger.warning(f"Value {v} is not in mapping.")
    return None


def feature_encoding_series(
        feature_data: pd.Series,
        feature: FeatureModelConfig,
        target: pd.Series = pd.Series(),
        train_ind: Optional[pd.Index] = None,
        log: bool = True,
        raise_on_error: bool = False,
) -> Tuple[pd.Series, Optional[Dict], Optional[List]]:
    """
    Encode feature's values, the input type should be Pandas' Series.

    :param feature_data: Feature's values.
    :param feature: Feature's config.
    :param target: Target's values.
    :param train_ind: Indices of training subset.
    :param log: Whether to log the process.
    :param raise_on_error: Whether to raise an error if occurs.

    :return: Tuple of encoded values, mapping and bins.
    """

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


    elif feature.encoding == EncodingNames.woe_cat:
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

    elif feature.encoding == EncodingNames.woe_num:
        if log:
            logger.info(f"{feature.name} || Encoding || WoE numerical to categorical")
        if feature.bins is not None:
            feature_data = feature_data.astype("float")
            feature_data = pd.cut(feature_data, bins=feature.bins, precision=5)
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
                feature_data = pd.cut(feature_data, bins=bins, precision=5)
            except Exception as e:
                logger.error(f"{feature.name} || Encoding error || Cannot convert to WoE || {str(e)}")
                if raise_on_error:
                    raise ValueError(f"{feature.name} || Cannot convert to WoE")

    elif feature.encoding == EncodingNames.woe_num_num:
        if log:
            logger.info(f"{feature.name} || Encoding || WoE numerical to numerical")
        if feature.bins is not None and feature.mapping is not None:
            feature_data = feature_data.astype("float")
            feature_data = pd.cut(feature_data, bins=bins, precision=5).map(mapping).astype("float")
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
                feature_data = pd.cut(feature_data, bins=bins, precision=5).map(mapping).astype("float")
            except Exception as e:
                logger.error(f"{feature.name} || Encoding error || Cannot convert to WoE || {str(e)}")
                if raise_on_error:
                    raise ValueError(f"{feature.name} || Cannot convert to WoE")
    elif feature.encoding == EncodingNames.cut_num:
        pass #Encoding call in prepare_numerical_feature function
    else:
        logger.error(f"Unknown encoding {feature.encoding}|| Return origin")
        if raise_on_error:
            raise NotImplementedError(f"{feature.name} || Unknown encoding")

    return feature_data, mapping, bins


def feature_encoding(
        feature_value: Union[int, float],
        feature: FeatureModelConfig,
) -> Union[int, float]:
    """
    Encode feature's one value.

    :param feature_value: Feature's value.
    :param feature: Feature's config.

    :return: Encoded value.
    """

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
            feature_value = map_num(feature_value, mapping)
        except Exception as e:
            raise ValueError(f"{feature.name} || Cannot convert to WoE")

    elif feature.encoding == EncodingNames.woe_num_num:
        if feature.bins is None or feature.mapping is None:
            raise NotImplementedError(f"{feature.name} || Invalid mappings or bins")
        try:
            if not isinstance(feature_value, (float, int)):
                feature_value = float(feature_value)
            feature_value = map_num(feature_value, feature.mapping)
        except Exception as e:
            raise ValueError(f"{feature.name} || Cannot convert to WoE")
    elif feature.encoding == EncodingNames.cut_num:
        pass
    else:
        raise NotImplementedError(f"{feature.name} || Unknown encoding")

    return feature_value


def dict_replace(
        feature: FeatureModelConfig,
        dtype: Literal[FeaturesTypes.numerical, FeaturesTypes.categorical]
) -> Dict:
    """
    Prepares replace dict for categorical and numerical features.

    :param feature: Feature's config.
    :param dtype: Feature's type. (Deprecated)

    :return: Replace dict.
    """

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


def dict_replace_pl(
        feature: FeatureModelConfig,
        is_numeric_dtype: bool
) -> Dict:
    """
    Prepares replace dict for categorical and numerical features.

    :param feature: Feature's config.
    :param is_numeric_dtype: If feature values' type is numeric, then it's True, otherwise False.

    :return: Replace dict.
    """

    dict_replace_temp = {}

    for key, val in feature.replace.items():
        if (
            key != FeatureEngineering.feature_type
            and val != FeatureEngineering.nan
        ):
            if val != FeatureEngineering.not_changed:
                new_val = val
            else:
                new_val = key
        elif val == FeatureEngineering.nan:
            new_val = None
        else:
            continue

        if is_numeric_dtype:
            try:
                key_float = float(key)
                if new_val is None:
                    new_val_numeric = None
                else:
                    try:
                        new_val_numeric = int(new_val)
                    except ValueError:
                        new_val_numeric = float(new_val)
                dict_replace_temp[key_float] = new_val_numeric
            except ValueError:
                pass
        else:
            dict_replace_temp[key.upper()] = str(new_val).upper() if new_val is not None else None

    return dict_replace_temp


def replace_categorical_values_series(
        feature_data: pd.Series,
        feature: FeatureModelConfig,
) -> pd.Series:
    """
    Replace values in categorical feature's data, the input type should be Pandas' Series.

    :param feature_data: Feature's values.
    :param feature: Feature's config.

    :return: Series with replaced values.
    """

    dict_replace_temp = dict_replace(feature=feature, dtype=FeaturesTypes.categorical)
    ind = (~feature_data.isin(dict_replace_temp) & pd.notnull(feature_data))
    feature_data = feature_data.map(dict_replace_temp)
    feature_data.loc[ind] = feature.default

    return feature_data


def replace_categorical_values(
        feature_value: Union[int, str],
        feature: FeatureModelConfig,
) -> Union[int, float, str]:
    """
    Replace categorical feature's one value.

    :param feature_value: Feature's value.
    :param feature: Feature's config.

    :return: Replaced value.
    """

    dict_replace_temp = dict_replace(feature=feature, dtype=FeaturesTypes.categorical)
    if pd.isnull(feature_value):
        return np.nan
    return dict_replace_temp.get(feature_value, feature.default)


def replace_numerical_values_series(
        feature_data: pd.Series,
        feature: FeatureModelConfig,
) -> pd.Series:
    """
    Replace values in numerical feature's data, the input type should be Pandas' Series.

    :param feature_data: Feature's values.
    :param feature: Feature's config.

    :return: Series with replaced values.
    """

    dict_replace_temp = dict_replace(feature=feature, dtype=FeaturesTypes.numerical)
    return feature_data.replace(dict_replace_temp)


def replace_numerical_values(
        feature_value: Union[int, float],
        feature: FeatureModelConfig,
) -> Union[int, float]:
    """
    Replace numerical feature's one value.

    :param feature_value: Feature's value.
    :param feature: Feature's config.

    :return: Replaced value.
    """

    dict_replace_temp = dict_replace(feature=feature, dtype=FeaturesTypes.numerical)
    return dict_replace_temp.get(feature_value, feature_value)


def replace_with_default(feature_data: pd.Series, feature: FeatureModelConfig, values: List[str]) -> pd.Series:
    """
    Replace values in feature's data with default, the input type should be Pandas' Series.

    :param feature_data: Feature's values.
    :param feature: Feature's config.
    :param values: Values to be replaced

    :return: Series with replaced values.
    """

    dict_replace_temp = {value: feature.default for value in values}
    return feature_data.replace(dict_replace_temp)


def prepare_relative_feature_series(
        numerator: pd.Series,
        denominator: pd.Series,
        default_value: Union[float, int],
) -> pd.Series:
    """
    Prepare relative feature's data, the input numerator's and denominator's types should be Pandas' Series.

    :param numerator: Numerator's values.
    :param denominator: Denominator's values.
    :param default_value: Default value for NaNs and infinities.

    :return: Series with calculated values.
    """

    return (numerator / denominator).replace([-np.inf, np.inf], np.nan).fillna(default_value)


def prepare_relative_feature_series_pl(
        lazy_data: pl.LazyFrame,
        feature_name: str,
        numerator_name: str,
        denominator_name: str,
        default_value: int | float | Literal[FeatureEngineering.nan],
) -> pl.LazyFrame:
    """
    Prepare relative feature's data, the input numerator's and denominator's types should be Polars' LazyFrame.

    :param lazy_data: Features' values in Polars' LazyFrame format.
    :param feature_name: Feature's name.
    :param numerator_name: Numerator column's name.
    :param denominator_name: Denominator column's name.
    :param default_value: Default value for NaNs and infinities.

    :return: LazyFrame with calculated values.
    """

    return (
        lazy_data
        .with_columns(
            pl.when(
                (pl.col(numerator_name).is_null())
                | (pl.col(numerator_name).is_nan())
                | (pl.col(denominator_name).is_null())
                | (pl.col(denominator_name).is_nan())
                | (pl.col(denominator_name) == 0)
            )
            .then(pl.lit(float("nan")) if default_value == FeatureEngineering.nan else pl.lit(default_value))
            .otherwise(pl.col(numerator_name) / pl.col(denominator_name))
            .alias(feature_name)
        )
    )


def prepare_relative_feature(
        numerator: Union[float, int],
        denominator: Union[float, int],
        default_value: Union[float, int],
) -> Union[float, int]:
    """
    Prepare relative feature's one value.

    :param numerator: Numerator's value.
    :param denominator: Denominator's value.
    :param default_value: Default value for NaNs and infinities.

    :return: Calculated value.
    """

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


def prepare_categorical_feature_series(
        feature_data: pd.Series,
        feature: FeatureModelConfig,
        log: bool = True,
) -> pd.Series:
    """
    Prepare values in categorical feature's data, the input type should be Pandas' Series.

    :param feature_data: Feature's values.
    :param feature: Feature's config.
    :param log: Whether to log the process.

    :return: Series with prepared values.
    """

    feature_data = feature_data.apply(lambda x: to_str(x))

    # Replace values
    if log:
        ind = (~feature_data.isin(feature.replace.keys()) & feature_data.notna())
        if len(feature_data.loc[ind]) > 0:
            logger.info(feature.name + ' || Присвоено default значений: ' + str(len(feature_data.loc[ind])))
    feature_data = replace_categorical_values_series(feature_data, feature)

    # Fill missing values
    fill_null_value = feature.fillna if feature.fillna else feature.default
    if pd.isnull(feature_data).sum() > 0:
        if log:
            logger.info(feature.name + ' || Исправлено пропусков: ' + str(feature_data.isna().sum()))
        feature_data.fillna(fill_null_value, inplace=True)

    return feature_data


def prepare_categorical_feature_pl(
        lazy_data: pl.LazyFrame,
        feature: FeatureModelConfig,
        data_dtypes: Dict[str, pl.DataType],
) -> pl.LazyFrame:
    """
    Prepare values in categorical feature's data, the input type should be Polars' LazyFrame.

    :param lazy_data: Feature's values in Polars' LazyFrame format.
    :param feature: Feature's config.
    :param data_dtypes: Features' types.

    :return: Polars' LazyFrame with prepared values.
    """

    dict_replace_temp = dict_replace_pl(feature=feature, is_numeric_dtype=data_dtypes[feature.name].is_numeric())
    fill_null_value = feature.fillna if feature.fillna else feature.default

    lazy_data = (
        lazy_data
        .with_columns(
            pl.when(
                ~(
                    pl.col(feature.name).is_in(dict_replace_temp)
                    if data_dtypes[feature.name].is_numeric()
                    else pl.col(feature.name).str.to_uppercase().is_in(dict_replace_temp)
                )
                & pl.col(feature.name).is_not_null()
                & (pl.col(feature.name).is_not_nan() if data_dtypes[feature.name].is_numeric() else True)
            )
            .then(pl.lit(feature.default))
            .when(
                data_dtypes[feature.name].is_numeric()
            )
            .then(pl.col(feature.name).cast(pl.String).replace(dict_replace_temp))
            .otherwise(pl.col(feature.name).str.to_uppercase().replace(dict_replace_temp))
            .fill_null(fill_null_value)
            .alias(feature.name)
        )
    )

    return lazy_data


def prepare_categorical_feature(
        feature_value: Union[int, str],
        feature: FeatureModelConfig,
) -> Union[int, str]:
    """
    Prepare categorical feature's one value.

    :param feature_value: Feature's value.
    :param feature: Feature's config.

    :return: Prepared value.
    """

    feature_value = to_str(feature_value)

    # Replace values
    feature_value = replace_categorical_values(feature_value, feature)

    # Fill missing values
    if pd.isnull(feature_value):
        feature_value = feature.fillna if feature.fillna else feature.default

    return feature_value


def prepare_numerical_feature_series(
        feature_data: pd.Series,
        feature: FeatureModelConfig,
        default_value: float | int,
        log: bool = True,
) -> pd.Series:
    """
    Prepare values in numerical feature's data, the input type should be Pandas' Series.

    :param feature_data: Feature's values.
    :param feature: Feature's config.
    :param default_value: Default value for NaNs.
    :param log: Whether to log the process.

    :return: Series of prepared values.
    """

    # Replace values
    feature_data = replace_numerical_values_series(feature_data, feature)

    if feature_data.dtype not in (int, float):
        feature_data = pd.to_numeric(feature_data, downcast="float", errors="coerce")

    if pd.isnull(feature_data).sum() > 0:
        if log:
            logger.info(feature.name + ' || Исправлено пропусков: ' + str(feature_data.isna().sum()))
        feature_data.fillna(default_value, inplace=True)

    # Clip values
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

    # Cut values
    if feature.encoding == EncodingNames.cut_num and feature.cut_number is None:
        feature.cut_number = CutNumberEncoder().encode_data(feature_data)

    if feature.cut_number:
        val_splits = [-np.inf] + list([float(x) for x in feature.cut_number.split('_')]) + [np.inf]
        feature_data = pd.cut(feature_data, bins=val_splits).astype(str)

    return feature_data


def prepare_numerical_feature_pl(
        lazy_data: pl.LazyFrame,
        feature: FeatureModelConfig,
        data_dtypes: Dict[str, pl.DataType],
        default_value: float | int,
) -> pl.LazyFrame:
    """
    Prepare values in numerical feature's data, the input type should be Polars' LazyFrame.

    :param lazy_data: Feature's values in Polars' LazyFrame format.
    :param feature: Feature's config.
    :param data_dtypes: Features' types.
    :param default_value: Default value for NaNs.

    :return: Polars' LazyFrame with prepared values.
    """

    dict_replace_temp = dict_replace_pl(feature=feature, is_numeric_dtype=data_dtypes[feature.name].is_numeric())

    lazy_data = (
        lazy_data
        .with_columns(
            pl.col(feature.name)
            .replace(dict_replace_temp)
            .cast(pl.Float64)
            .fill_null(float(default_value))
            .fill_nan(float(default_value))
            .alias(feature.name)
        )
    )

    # Clip values
    if feature.clip:
        lazy_data = (
            lazy_data
            .with_columns(
                pl.col(feature.name).clip(
                    feature.clip[FeatureEngineering.min_value],
                    feature.clip[FeatureEngineering.max_value],
                )
                .alias(feature.name)
            )
        )

    # Cut values
    # TODO:
    # if feature.encoding == EncodingNames.cut_num and feature.cut_number is None:
    #     feature.cut_number = CutNumberEncoder().encode_data(feature_data)

    if feature.cut_number:
        # For equal result with Pandas' cut function
        def format_cut_boundary(value):
            if value == float("-inf"):
                return "-inf"
            if value == float("inf"):
                return "inf"
            return float(value)

        val_splits = list([float(x) for x in feature.cut_number.split('_')])
        val_splits_inf = [float("-inf")] + val_splits + [float("inf")]
        lazy_data = (
            lazy_data
            .with_columns(
                pl.col(feature.name)
                .cut(
                    val_splits,
                    labels=[
                        f"({format_cut_boundary(val_splits_inf[i])}, {format_cut_boundary(val_splits_inf[i+1])}]"
                        for i in range(len(val_splits) + 1)
                    ],
                )
                .cast(pl.String)
                .alias(feature.name)
            )
        )

    return lazy_data


def prepare_numerical_feature(
        feature_value: Union[float, int, str],
        feature: FeatureModelConfig,
) -> Union[float, int, str]:
    """
    Prepare numerical feature's one value.

    :param feature_value: Feature's value.
    :param feature: Feature's config.

    :return: Prepared value.
    """

    # Replace values
    feature_value = replace_numerical_values(feature_value, feature)

    if not isinstance(feature_value, (int, float)):
        try:
            feature_value = float(feature_value)
        except:
            feature_value = np.nan

    # Fill missing values
    if pd.isnull(feature_value):
        feature_value = feature.default

    # Clip values
    if feature.clip:
        feature_value = (
            feature.clip[FeatureEngineering.min_value] if feature_value < feature.clip[FeatureEngineering.min_value]
            else feature.clip[FeatureEngineering.max_value] if feature_value > feature.clip[FeatureEngineering.max_value]
            else feature_value
        )

    # Cut values
    if feature.cut_number:
        val_splits = [-np.inf] + list([float(x) for x in feature.cut_number.split('_')]) + [np.inf]
        mapping = {
            k: v for k, v in zip(
                pd.IntervalIndex.from_arrays(val_splits[:-1], val_splits[1:]),
                list(pd.IntervalIndex.from_arrays(val_splits[:-1], val_splits[1:]).astype(str))
            )
        }
        feature_value = map_num(feature_value, mapping)

    return feature_value


def prepare_dataset(
        group_name: str,
        data: Union[Dict, pd.DataFrame, pl.DataFrame],
        train_ind: Optional[pd.Index],
        test_ind: Optional[pd.Index],
        model_config: ModelConfig,
        check_prepared: bool = False,
        calc_corr: bool = False,
        save_data: bool = False,
        corr_threshold: Optional[float] = None,
        target: Optional[pd.Series | pl.DataFrame] = None,
        log: bool = True,
        modify_dtypes: bool = True,
        raise_on_encoding_error: bool = False,
        extra_columns_list: List[str] | None = None,
) -> PrepareDatasetResult:
    """
    Prepare dataset. Input data should be Pandas' or Polars' DataFrame or dict of feature and value pairs.

    :param group_name: Models' group name.
    :param data: Input data.
    :param train_ind: Indices of training subset.
    :param test_ind: Indices of testing subset. (Deprecated)
    :param model_config: Model's config.
    :param check_prepared: Whether to check prepared dataset.
    :param calc_corr: Whether to calculate correlation matrix. (Deprecated)
    :param save_data: Whether to save prepared dataset.
    :param corr_threshold: Threshold for correlation matrix. (Deprecated)
    :param target: Target's values.
    :param log: Whether to log the process.
    :param modify_dtypes: Whether to modify dtypes to int32, float32 and category respectively.
    :param raise_on_encoding_error: Whether to raise if an encoding error occurs.
    :param extra_columns_list: List of extra columns to be added to the dataset (for Polars only).

    :return: An instance of the PrepareDatasetResult class.
    """

    as_dict: bool = False
    as_polars: bool = False
    as_pandas: bool = False
    if isinstance(data, dict):
        as_dict = True
    elif isinstance(data, pl.DataFrame) and (isinstance(target, pl.DataFrame) or target is None):
        as_polars = True
        if ColumnsNames.is_train_obml not in data.columns:
            data = data.with_columns(pl.lit(1).alias(ColumnsNames.is_train_obml))
        if target is not None and ColumnsNames.is_train_obml not in target.columns:
            target = target.with_columns(pl.lit(1).alias(ColumnsNames.is_train_obml))
        data_dtypes = data.schema
        lazy_data: pl.LazyFrame = data.lazy()
    elif isinstance(data, pd.DataFrame) and (isinstance(target, pd.Series) or target is None):
        as_pandas = True
        pd.options.mode.chained_assignment = None
        if train_ind is None:
            train_ind = data.index
        if target is None:
            target = pd.Series()
    else:
        logger.error(f"Invalid data type {type(data)}, {type(target)}")
        raise TypeError(f"Invalid data type {type(data)}, {type(target)}")

    if model_config.relative_features:
        for relative_feature in model_config.relative_features:
            if as_dict:
                data[relative_feature.name] = prepare_relative_feature(
                    numerator=data[relative_feature.numerator],
                    denominator=data[relative_feature.denominator],
                    default_value=relative_feature.default
                )

            elif as_polars:
                lazy_data = prepare_relative_feature_series_pl(
                    lazy_data=lazy_data,
                    feature_name=relative_feature.name,
                    numerator_name=relative_feature.numerator,
                    denominator_name=relative_feature.denominator,
                    default_value=relative_feature.default
                )

            elif as_pandas:
                data[relative_feature.name] = prepare_relative_feature_series(
                    numerator=data[relative_feature.numerator],
                    denominator=data[relative_feature.denominator],
                    default_value=relative_feature.default
                )

    for feature in model_config.features:

        # Numerical features
        if feature.replace.get(FeatureEngineering.feature_type) == FeatureEngineering.numerical:

            # Define missing values
            default_value = feature.default
            if not isinstance(feature.default, (int, float)):
                if as_polars:
                    feature_data_train = (
                        data
                        .filter(pl.col(ColumnsNames.is_train_obml) == 1)
                        [feature.name]
                    )

                elif as_pandas:
                    feature_data_train = (
                        data
                        .loc[[i for i in train_ind if i in data.index]]
                        [feature.name]
                    )

                match feature.default:
                    case FeatureEngineering.min:
                        val_fill = feature_data_train.min()
                        if val_fill == -float("inf"):
                            logger.warning(f"invalid default value (minimum = -inf) for {feature.name}")
                    case FeatureEngineering.max:
                        val_fill = feature_data_train.max()
                        if val_fill == float("inf"):
                            logger.warning(f"invalid default value (maximum = inf) for {feature.name}")
                    case FeatureEngineering.mean:
                        val_fill = feature_data_train.mean()
                    case FeatureEngineering.median:
                        val_fill = feature_data_train.median()
                    case FeatureEngineering.nan:
                        val_fill = np.nan
                        logger.warning(f"invalid default value (nan) for {feature.name}")
                    case _:
                        logger.error(f"invalid default value for {feature.name}: {feature.default}")
                        raise ValueError(f"invalid default value for {feature.name}: {feature.default}")

                logger.info(f"default value for {feature.name} is {val_fill}")
                model_config = update_model_config_default(model_config, feature.name, val_fill)
                default_value = val_fill

            # Prepare values
            if as_dict:
                data[feature.name] = prepare_numerical_feature(
                    feature_value=data[feature.name],
                    feature=feature,
                )

            elif as_polars:
                lazy_data = prepare_numerical_feature_pl(
                    lazy_data=lazy_data,
                    feature=feature,
                    data_dtypes=data_dtypes,
                    default_value=default_value,
                )

            elif as_pandas:
                data[feature.name] = prepare_numerical_feature_series(
                    feature_data=data[feature.name],
                    feature=feature,
                    default_value=default_value,
                    log=log,
                )

        # Categorical features
        else:
            if as_dict:
                data[feature.name] = prepare_categorical_feature(
                    feature_value=data[feature.name],
                    feature=feature,
                )

            elif as_polars:
                lazy_data = prepare_categorical_feature_pl(
                    lazy_data=lazy_data,
                    feature=feature,
                    data_dtypes=data_dtypes,
                )

            elif as_pandas:
                data[feature.name] = prepare_categorical_feature_series(
                    feature_data=data[feature.name],
                    feature=feature,
                    log=log,
                )

    if as_polars:
        data = lazy_data.collect()

    if check_prepared and not as_dict:
        logger.info('Find drop values for features')

        for feature in model_config.features:
            if feature.replace.get(FeatureEngineering.feature_type) != FeatureEngineering.numerical:

                if as_pandas:
                    replace_dict = dict_replace(feature=feature, dtype=FeaturesTypes.categorical)
                    drop_values = find_drop_values(data[feature.name], replace_dict, train_ind)

                elif as_polars:
                    replace_dict = dict_replace_pl(feature=feature, is_numeric_dtype=data_dtypes[feature.name].is_numeric())
                    drop_values = find_drop_values_pl(
                        data.filter(pl.col(ColumnsNames.is_train_obml) == 1)[feature.name],
                        replace_dict,
                    )

                try:
                    if len(drop_values) > 0:
                        logger.info('Dropping unused levels in ' + feature.name + '||' + str(drop_values))
                        model_config = update_model_config_replace(model_config, {feature.name: drop_values})

                        if as_pandas:
                            data.loc[[i for i in train_ind if i in data.index]][feature.name] = replace_with_default(
                                data.loc[[i for i in train_ind if i in data.index]][feature.name], feature, drop_values
                            )
                        elif as_polars:
                            data = (
                                data
                                .with_columns(
                                    pl.when(pl.col(feature.name).is_in(drop_values))
                                    .then(pl.lit(feature.default))
                                    .otherwise(pl.col(feature.name))
                                    .alias(feature.name)
                                )
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

        elif feature.encoding is not None and as_pandas:
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

        elif feature.encoding is not None and as_polars:
            if log:
                logger.info('Feature preparation||Encoding from config ' + str(feature.encoding))
            feature_data = data.select(feature.name, ColumnsNames.is_train_obml).to_pandas()

            encoded_data, mapping, bins = feature_encoding_series(
                feature_data=feature_data[feature.name],
                feature=feature,
                target=target.to_pandas()[model_config.column_target] if target is not None else pd.Series(),
                train_ind=feature_data.loc[feature_data[ColumnsNames.is_train_obml] == 1].index,
                log=log,
                raise_on_error=raise_on_encoding_error,
            )
            data = (
                data
                .with_columns(
                    pl.from_pandas(encoded_data).alias(feature.name)
                )
            )
            feature.mapping = mapping
            feature.bins = bins

    features_all = list(set(chain(
        [feature.name for feature in model_config.relative_features] if model_config.relative_features else [],
        [feature.name for feature in model_config.features] if model_config.features else [],
    )))

    if as_dict:
        features_categorical = [feature for feature in features_all if
                                isinstance(data[feature], str)]
        features_numerical = [feature for feature in features_all if
                              isinstance(data[feature], (int, float))]
        data = pd.DataFrame([data])

    elif as_polars:
        data = data.select(
            [ColumnsNames.is_train_obml, model_config.column_target]
            + ([model_config.column_exposure] if model_config.column_exposure is not None else [])
            + ([model_config.column_weight] if model_config.column_weight is not None else [])
            + (extra_columns_list if extra_columns_list is not None else [])
            + features_all
        )
        data_dtypes_prepared = data.schema
        features_categorical = [feature_name for feature_name in features_all if not data_dtypes_prepared[feature_name].is_numeric()]
        features_numerical = [feature_name for feature_name in features_all if data_dtypes_prepared[feature_name].is_numeric()]

    else:
        data = data[features_all]
        features_categorical = [feature for feature in features_all if
                                (data[feature].dtype == 'object' or data[feature].dtype == 'category')]
        features_numerical = [feature for feature in features_all if
                              (data[feature].dtype != 'object' and data[feature].dtype != 'category')]

    # Change output data types
    # TODO: добавить в конфиг автоматическое определение размера категорий
    if modify_dtypes and not as_dict and not as_polars:
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
        if as_polars:
            data.write_parquet(
                os.path.join(model_path, model_config.name + "_data.parquet.gzip"),
                compression="gzip",
            )
        else:
            data.to_parquet(
                os.path.join(model_path, model_config.name + "_data.parquet.gzip"),
                engine="pyarrow",
                compression="gzip",
            )

    return PrepareDatasetResult(
        data=data,
        features_numerical=features_numerical,
        features_categorical=features_categorical,
        model_config=model_config,
        corr_df=pd.DataFrame()
    )
