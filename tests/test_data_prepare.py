from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from pandas.testing import assert_series_equal
from unittest import TestCase, main

from outboxml.core.data_prepare import (
    replace_categorical_values_series,
    replace_categorical_values,
    replace_numerical_values_series,
    replace_numerical_values,
    prepare_relative_feature_series,
    prepare_relative_feature,
    prepare_categorical_feature_series,
    prepare_categorical_feature,
    prepare_numerical_feature_series,
    prepare_categorical_feature_pl,
    prepare_numerical_feature, prepare_relative_feature_series_pl, prepare_numerical_feature_pl,
)
from outboxml.core.pydantic_models import FeatureModelConfig
from outboxml.data_subsets import ModelDataSubset


class TestDataPrepare(TestCase):

    def setUp(self):
        pass

    def test_replace_categorical_values(self):
        feature_model_config_categorical = FeatureModelConfig.model_validate(
            {
                "name": "FEATURE_CAT",
                "default": "1",
                "fillna": "11",
                "replace": {"1": "_NOTCHANGED_", "2": "1", "3": "33", "4": "_NAN_"},
            }
        )

        feature_data = pd.Series(["1", "2", "3", "4", "5", np.nan, None])
        feature_data_replaced = pd.Series(["1", "1", "33", np.nan, "1", np.nan, np.nan])

        feature_data_replace_series = replace_categorical_values_series(
            feature_data, feature_model_config_categorical
        )
        feature_data_replace_dict = pd.Series(
            [replace_categorical_values(v, feature_model_config_categorical) for v in feature_data]
        )
        assert_series_equal(feature_data_replace_series, feature_data_replace_dict)
        assert_series_equal(feature_data_replace_series, feature_data_replaced)

    def test_replace_numerical_values(self):
        feature_model_config_numerical = FeatureModelConfig.model_validate(
            {
                "name": "FEATURE_NUM",
                "default": 13,
                "clip": {"min_value": -1, "max_value": 12},
                "replace": {"_TYPE_": "_NUM_", "M": -1, "2": 3, "-100": "_NAN_"},
            }
        )

        feature_data_variants = [
            pd.Series([-100, -2, -1, 1, 2, 14, np.nan, None]),
            pd.Series(["-100", "-2", "M", "1", "2", "14", np.nan, None]),
        ]
        feature_data_replaced_variants = [
            pd.Series([np.nan, -2.0, -1.0, 1.0, 3.0, 14.0, np.nan, np.nan]),
            pd.Series([np.nan, "-2", -1, "1", 3, "14", np.nan, None]),
        ]

        for feature_data, feature_data_replaced in zip(feature_data_variants, feature_data_replaced_variants):
            feature_data_replace_series = replace_numerical_values_series(feature_data, feature_model_config_numerical)
            feature_data_replace_dict = pd.Series(
                [replace_numerical_values(v, feature_model_config_numerical) for v in feature_data]
            )
            # FIXME:
            # feature_data_replace_series_cut_num = replace_numerical_values_series(
            #     feature_data, self.feature_model_config_numerical_cut_num)
            # assert_series_equal(feature_data_replace_series_cut_num,
            #                     pd.Series([np.nan, "-2", -1, "1", 3, "14", np.nan, None]))
            assert_series_equal(feature_data_replace_series, feature_data_replace_dict)
            assert_series_equal(feature_data_replace_series, feature_data_replaced)

    def test_prepare_relative_feature(self):
        numerator = pd.Series([0, 1, 2, 4, 5, np.nan, None], name="numerator")
        denominator = pd.Series([1, 0, 2, np.nan, None, np.nan, 1], name="denominator")
        default_value = -100
        feature_data_prepared = pd.Series(
            [0.0, -100.0, 1.0, -100.0, -100.0, -100.0, -100.0],
            name="feature_relative",
        )

        feature_data_prepared_series = prepare_relative_feature_series(
            numerator, denominator, default_value
        )
        feature_data_prepared_series.name = "feature_relative"
        data_pl = pl.DataFrame(
            {
                "numerator": numerator.replace({float('nan'): None}),
                "denominator": denominator.replace({float('nan'): None}),
            }
        )
        lazy_data = data_pl.lazy()
        feature_data_prepared_pl = (
            prepare_relative_feature_series_pl(
                lazy_data,
                feature_name="feature_relative",
                numerator_name="numerator",
                denominator_name="denominator",
                default_value=default_value,
            )
            .collect()
            .to_pandas()
            ["feature_relative"]
        )
        feature_data_prepared_dict = pd.Series(
            [prepare_relative_feature(n, d, default_value) for n, d in zip(numerator.tolist(), denominator.tolist())],
            name="feature_relative",
        )
        assert_series_equal(feature_data_prepared_series, feature_data_prepared_pl)
        assert_series_equal(feature_data_prepared_series, feature_data_prepared_dict)
        assert_series_equal(feature_data_prepared_series, feature_data_prepared)

    def test_prepare_categorical_feature(self):
        feature_model_config_categorical = FeatureModelConfig.model_validate(
            {
                "name": "FEATURE_CAT",
                "default": "1",
                "fillna": "11",
                "replace": {"1": "_NOTCHANGED_", "2": "1", "3": "33", "4": "_NAN_"},
            }
        )

        feature_data_variants = [
            pd.Series(["1", "2", "3", "4", "5", np.nan, None], name=feature_model_config_categorical.name),
            pd.Series([1, 2, 3, 4, 5, np.nan, None], name=feature_model_config_categorical.name),
        ]
        feature_data_prepared = pd.Series(
            ["1", "1", "33", "11", "1", "11", "11"],
            name=feature_model_config_categorical.name,
        )

        for feature_data in feature_data_variants:
            feature_data_prepared_series = prepare_categorical_feature_series(
                feature_data, feature_model_config_categorical, log=False
            )
            data_pl = pl.DataFrame(pd.DataFrame(feature_data).replace({float('nan'): None}))
            data_dtypes = data_pl.schema
            lazy_data = data_pl.lazy()
            feature_data_prepared_pl = (
                prepare_categorical_feature_pl(
                    lazy_data,
                    feature=feature_model_config_categorical,
                    data_dtypes=data_dtypes,
                )
                .collect()
                .to_pandas()
                [feature_model_config_categorical.name]
            )
            feature_data_prepared_dict = pd.Series(
                [prepare_categorical_feature(v, feature_model_config_categorical) for v in feature_data],
                name=feature_model_config_categorical.name,
            )
            assert_series_equal(feature_data_prepared_series, feature_data_prepared_pl)
            assert_series_equal(feature_data_prepared_series, feature_data_prepared_dict)
            assert_series_equal(feature_data_prepared_series, feature_data_prepared)

    def test_prepare_numerical_feature(self):
        feature_model_config_numerical = FeatureModelConfig.model_validate(
            {
                "name": "FEATURE_NUM",
                "default": 5,
                "clip": {"min_value": -1, "max_value": 13},
                "replace": {"_TYPE_": "_NUM_", "M": -1, "2": 3, "-100": "_NAN_"},
            }
        )

        feature_data_variants = [
            pd.Series([-100, -2, -1, 1, 2, 14, np.nan, None], name=feature_model_config_numerical.name),
            pd.Series(["-100", "-2", "M", "1", "2", "14", np.nan, None], name=feature_model_config_numerical.name),
        ]
        feature_data_prepared = pd.Series(
            [5.0, -1.0, -1.0, 1.0, 3.0, 13.0, 5.0, 5.0],
            name=feature_model_config_numerical.name,
        )

        for feature_data in feature_data_variants:
            feature_data_prepared_series = prepare_numerical_feature_series(
                feature_data, feature_model_config_numerical,
                default_value=feature_model_config_numerical.default,
                log=False,
            )
            data_pl = pl.DataFrame(pd.DataFrame(feature_data).replace({float('nan'): None}))
            data_dtypes = data_pl.schema
            lazy_data = data_pl.lazy()
            feature_data_prepared_pl = (
                prepare_numerical_feature_pl(
                    lazy_data,
                    feature=feature_model_config_numerical,
                    data_dtypes=data_dtypes,
                    default_value=feature_model_config_numerical.default,
                )
                .collect()
                .to_pandas()
                [feature_model_config_numerical.name]
            )
            feature_data_prepared_dict = pd.Series(
                [prepare_numerical_feature(v, feature_model_config_numerical) for v in feature_data],
                name=feature_model_config_numerical.name,
            )
            assert_series_equal(feature_data_prepared_series.astype("float64"), feature_data_prepared_pl)
            assert_series_equal(feature_data_prepared_series.astype("float64"), feature_data_prepared_dict)
            assert_series_equal(feature_data_prepared_series.astype("float64"), feature_data_prepared)

    def test_prepare_numerical_feature_cut(self):
        feature_model_config_numerical_cut = FeatureModelConfig.model_validate(
            {
                "name": "FEATURE_NUM_CUT",
                "default": 5,
                "clip": {"min_value": -1, "max_value": 13},
                "replace": {"_TYPE_": "_NUM_", "M": -1, "2": 3, "-100": "_NAN_"},
                "cut_number": "1_3_10",
            }
        )

        feature_data = pd.Series([-100, -2, -1, 1, 2, 14, None], name=feature_model_config_numerical_cut.name)
        feature_data_prepared = pd.Series(
            # [5.0, -1.0, -1.0, 1.0, 3.0, 13.0, 5.0],
            ["(3.0, 10.0]", "(-inf, 1.0]", "(-inf, 1.0]", "(-inf, 1.0]", "(1.0, 3.0]", "(10.0, inf]", "(3.0, 10.0]"],
            name=feature_model_config_numerical_cut.name,
        )

        feature_data_prepared_series = prepare_numerical_feature_series(
            feature_data,
            feature_model_config_numerical_cut,
            default_value=feature_model_config_numerical_cut.default,
            log=False,
        )
        data_pl = pl.DataFrame(pd.DataFrame(feature_data))
        data_dtypes = data_pl.schema
        lazy_data = data_pl.lazy()
        feature_data_prepared_pl = (
            prepare_numerical_feature_pl(
                lazy_data,
                feature=feature_model_config_numerical_cut,
                data_dtypes=data_dtypes,
                default_value=feature_model_config_numerical_cut.default,
            )
            .collect()
            .to_pandas()
            [feature_model_config_numerical_cut.name]
        )
        feature_data_prepared_dict = pd.Series(
            [prepare_numerical_feature(v, feature_model_config_numerical_cut) for v in feature_data],
            name=feature_model_config_numerical_cut.name,
        )
        assert_series_equal(feature_data_prepared_series, feature_data_prepared_pl)
        assert_series_equal(feature_data_prepared_series, feature_data_prepared_dict)
        assert_series_equal(feature_data_prepared_series, feature_data_prepared)

    def test_model_data_subset(self):
        test_data_path = Path(__file__).resolve().parent / "test_data"
        path_to_data = test_data_path / 'titanic.csv'
        data = pd.read_csv(path_to_data)
        data1 = data.drop(columns=['AGE'])
        data2 = data['AGE']
        datasubset1 = ModelDataSubset(model_name='test',X_train=data1, features_categorical=list(data.columns))
        datasubset2 = ModelDataSubset(model_name='test', X_train=data2, features_numerical=['AGE'])
        dsubset = datasubset1 + datasubset2
        self.assertIsInstance(dsubset, ModelDataSubset)
        self.assertEqual(dsubset.X_train.shape, (891,12))
        self.assertEqual(datasubset1.X_train.shape, (891, 11))

if __name__ == '__main__':
    main()