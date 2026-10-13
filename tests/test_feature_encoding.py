import numpy as np
import pandas as pd
import pytest

from conftest import TEST_DATA_PATH
from outboxml.core.data_prepare import feature_encoding, feature_encoding_series
from outboxml.core.enums import EncodingNames
from outboxml.core.pydantic_models import FeatureModelConfig


@pytest.fixture(scope="module")
def titanic() -> pd.DataFrame:
    return pd.read_csv(TEST_DATA_PATH / "titanic.csv")


def categorical(encoding: str, **params) -> FeatureModelConfig:
    return FeatureModelConfig(name="SEX", default="_NAN_", replace={}, encoding=encoding, **params)


def numerical(encoding: str, name: str = "FARE", **params) -> FeatureModelConfig:
    return FeatureModelConfig(name=name, default=0, replace={"_TYPE_": "_NUM_"}, encoding=encoding, **params)


def encode(data: pd.DataFrame, feature: FeatureModelConfig):
    return feature_encoding_series(feature_data=data[feature.name], feature=feature, target=data["SURVIVED"],
                                   train_ind=data.index, log=False, raise_on_error=True)


@pytest.mark.parametrize("encoding, values, expected", [
    (EncodingNames.to_float, ["1.5", "2"], [1.5, 2.0]),
    (EncodingNames.to_int, ["1", "2"], [1, 2]),
])
def test_type_conversion(encoding, values, expected):
    feature = numerical(encoding, name="X")

    encoded, mapping, bins = feature_encoding_series(pd.Series(values), feature, log=False, raise_on_error=True)

    assert encoded.tolist() == expected
    assert [feature_encoding(v, feature) for v in values] == expected


@pytest.mark.parametrize("encoding", [EncodingNames.to_float, EncodingNames.to_int])
def test_type_conversion_errors(encoding):
    feature = numerical(encoding, name="X")

    with pytest.raises(ValueError):
        feature_encoding_series(pd.Series(["abc"]), feature, log=False, raise_on_error=True)
    with pytest.raises(ValueError):
        feature_encoding("abc", feature)


def test_woe_categorical_series_matches_single_values(titanic):
    feature = categorical(EncodingNames.woe_cat)

    encoded, mapping, bins = encode(titanic, feature)

    assert set(mapping) == {"male", "female"}
    assert pd.api.types.is_numeric_dtype(encoded)
    assert mapping["female"] != mapping["male"]
    feature.mapping = mapping
    assert [feature_encoding(v, feature) for v in titanic["SEX"].head(10)] == encoded.head(10).tolist()
    # the stored mapping is reused instead of being fitted again
    reused, reused_mapping, _ = encode(titanic.head(5), feature)
    assert reused_mapping == mapping


def test_woe_numerical_to_numerical_series_matches_single_values(titanic):
    feature = numerical(EncodingNames.woe_num_num)

    encoded, mapping, bins = encode(titanic, feature)

    assert bins is not None and mapping is not None
    assert encoded.nunique() <= len(bins)
    feature.mapping, feature.bins = mapping, bins
    singles = [feature_encoding(v, feature) for v in titanic["FARE"].head(20)]
    np.testing.assert_allclose(singles, encoded.head(20).astype(float).tolist())


def test_woe_numerical_to_categorical_uses_interval_labels(titanic):
    feature = numerical(EncodingNames.woe_num)

    encoded, mapping, bins = encode(titanic, feature)

    assert bins is not None
    # string categories: models (CatBoost) reject Interval objects
    assert isinstance(encoded.dtype, pd.CategoricalDtype)
    assert all(isinstance(label, str) for label in encoded.cat.categories)
    assert encoded.iloc[0] == "(-inf, 7.73125]"


def test_woe_numerical_to_categorical_single_value_matches_series(titanic):
    feature = numerical(EncodingNames.woe_num)
    encoded, mapping, bins = encode(titanic, feature)
    feature.bins = bins

    singles = [feature_encoding(v, feature) for v in titanic["FARE"].head(20)]

    assert singles == encoded.head(20).astype(str).tolist()


def test_cut_num_is_passthrough(titanic):
    feature = numerical(EncodingNames.cut_num)

    encoded, _, _ = encode(titanic, feature)

    assert encoded.equals(titanic["FARE"])
    assert feature_encoding(7.25, feature) == 7.25


@pytest.mark.parametrize("feature", [categorical(EncodingNames.woe_cat), numerical(EncodingNames.woe_num),
                                     numerical(EncodingNames.woe_num_num)])
def test_single_value_without_fitted_encoding_raises(feature):
    with pytest.raises(NotImplementedError):
        feature_encoding("male" if feature.name == "SEX" else 1.0, feature)


def test_unknown_encoding_raises():
    feature = numerical("target_encoding", name="X")

    with pytest.raises(NotImplementedError):
        feature_encoding_series(pd.Series([1.0]), feature, log=False, raise_on_error=True)
    with pytest.raises(NotImplementedError):
        feature_encoding(1.0, feature)


def test_one_row_prediction_matches_batch_prediction_with_woe_num_to_cat(titanic):
    from outboxml.core.predict import one_model_predict
    from outboxml.datasets_manager import DataSetsManager
    from outboxml.extractors import SimpleExtractor

    config = {
        "group_name": "woe", "project": "woe", "version": "1",
        "data_config": {"source": "csv", "local_name_source": "unused.csv",
                        "separation": {"kind": "random", "random_state": 1, "test_train_proportion": 0.2},
                        "data": {"targetslices": []}},
        "models_configs": [{"name": "model", "wrapper": "catboost", "objective": "binary", "column_target": "SURVIVED",
                            "features": [{"name": "FARE", "default": 0, "replace": {"_TYPE_": "_NUM_"},
                                          "encoding": "WoE_num_to_cat"},
                                         {"name": "AGE", "default": "_MEDIAN_", "replace": {"_TYPE_": "_NUM_"}}]}],
    }
    ds_manager = DataSetsManager(config_name=config, extractor=SimpleExtractor(data=titanic.copy()))
    ds_manager.fit_models()
    model = ds_manager.get_result()["model"].dict_for_prod_export()
    rows = titanic[["FARE", "AGE"]].dropna().head(5)

    batch = one_model_predict("woe", model, rows.copy(), log=False)["result"]["model"]
    singles = [one_model_predict("woe", model, row.to_dict(), log=False)["result"]["model"][0]
               for _, row in rows.iterrows()]

    np.testing.assert_allclose(singles, [batch[i] for i in rows.index])
