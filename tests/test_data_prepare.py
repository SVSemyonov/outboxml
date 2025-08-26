import numpy as np
import pandas as pd
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
    prepare_numerical_feature,
)
from outboxml.core.pydantic_models import FeatureModelConfig


class TestDataPrepare(TestCase):

    def setUp(self):
        self.feature_model_config_categorical = FeatureModelConfig.model_validate(
            {
                "name": "VEHICLE_SUBTYPE",
                "default": "1",
                "fillna": "11",
                "replace": {"1": "_NOTCHANGED_", "2": "1", "3": "33", "4": "_NAN_"}
            }
        )

        self.feature_model_config_numerical = FeatureModelConfig.model_validate(
            {
               "name": "CBM",
                "default": 13,
                "clip": {"min_value": -1, "max_value": 13},
                "replace": {"_TYPE_": "_NUM_", "M": -1, "2": 3, "-100": "_NAN_"}
            }
        )

        self.feature_model_config_numerical_cut = FeatureModelConfig.model_validate(
            {
                "name": "CBM_cut",
                "default": 13,
                "clip": {"min_value": -1, "max_value": 13},
                "replace": {"_TYPE_": "_NUM_", "M": -1, "2": 3, "-100": "_NAN_"},
                "cut_number": "1_3"
            }
        )

    def test_replace_categorical_values(self):
        feature_data = pd.Series(["1", "2", "3", "4", "5", np.nan, None])
        feature_data_replace_series = replace_categorical_values_series(
            feature_data, self.feature_model_config_categorical
        )
        feature_data_replace_dict = pd.Series(
            [replace_categorical_values(v, self.feature_model_config_categorical) for v in feature_data]
        )
        assert_series_equal(feature_data_replace_series, feature_data_replace_dict)
        assert_series_equal(feature_data_replace_series, pd.Series(["1", "1", "33", np.nan, "1", np.nan, np.nan]))

    def test_replace_numerical_values_1(self):
        feature_data = pd.Series([-100, -2, -1, 1, 2, 14, np.nan, None])
        feature_data_replace_series = replace_numerical_values_series(feature_data, self.feature_model_config_numerical)
        feature_data_replace_dict = pd.Series(
            [replace_numerical_values(v, self.feature_model_config_numerical) for v in feature_data]
        )
        assert_series_equal(feature_data_replace_series, feature_data_replace_dict)
        assert_series_equal(feature_data_replace_series, pd.Series([np.nan, -2.0, -1.0, 1.0, 3.0, 14.0, np.nan, np.nan]))

    def test_replace_numerical_values_2(self):
        feature_data = pd.Series(["-100", "-2", "M", "1", "2", "14", np.nan, None])
        feature_data_replace_series = replace_numerical_values_series(
            feature_data, self.feature_model_config_numerical
        )
        feature_data_replace_dict = pd.Series(
            [replace_numerical_values(v, self.feature_model_config_numerical) for v in feature_data]
        )
        assert_series_equal(feature_data_replace_series, feature_data_replace_dict)
        assert_series_equal(feature_data_replace_series, pd.Series([np.nan, "-2", -1, "1", 3, "14", np.nan, None]))

    def test_prepare_relative_feature(self):
        numerator = pd.Series([0, 1, 2, 4, 5, np.nan, None])
        denominator = pd.Series([1, 0, 2, np.nan, None, np.nan, 1])
        default_value = -100
        feature_data_prepared_series = prepare_relative_feature_series(
            numerator, denominator, default_value
        )
        feature_data_prepared_dict = pd.Series(
            [prepare_relative_feature(n, d, default_value) for n, d in zip(numerator.tolist(), denominator.tolist())]
        )
        assert_series_equal(feature_data_prepared_series, feature_data_prepared_dict)
        assert_series_equal(feature_data_prepared_series, pd.Series([0.0, -100.0, 1.0, -100.0, -100.0, -100.0, -100.0]))

    def test_prepare_categorical_feature_1(self):
        feature_data = pd.Series(["1", "2", "3", "4", "5", np.nan, None])
        feature_data_prepared_series = prepare_categorical_feature_series(
            feature_data, self.feature_model_config_categorical, log=False
        )
        feature_data_prepared_dict = pd.Series(
            [prepare_categorical_feature(v, self.feature_model_config_categorical) for v in feature_data]
        )
        assert_series_equal(feature_data_prepared_series, feature_data_prepared_dict)
        assert_series_equal(feature_data_prepared_series, pd.Series(["1", "1", "33", "11", "1", "11", "11"]))

    def test_prepare_categorical_feature_2(self):
        feature_data = pd.Series([1, 2, 3, 4, 5, np.nan, None])
        feature_data_prepared_series = prepare_categorical_feature_series(
            feature_data, self.feature_model_config_categorical, log=False
        )
        feature_data_prepared_dict = pd.Series(
            [prepare_categorical_feature(v, self.feature_model_config_categorical) for v in feature_data]
        )
        assert_series_equal(feature_data_prepared_series, feature_data_prepared_dict)
        assert_series_equal(feature_data_prepared_series, pd.Series(["1", "1", "33", "11", "1", "11", "11"]))

    def test_prepare_numerical_feature_1(self):
        feature_data = pd.Series([-100, -2, -1, 1, 2, 14, np.nan, None])
        feature_data_prepared_series, _ = prepare_numerical_feature_series(
            feature_data, self.feature_model_config_numerical, train_ind=None, log=False
        )
        feature_data_prepared_dict = pd.Series(
            [prepare_numerical_feature(v, self.feature_model_config_numerical) for v in feature_data]
        )
        assert_series_equal(
            feature_data_prepared_series.astype("float64"),
            feature_data_prepared_dict.astype("float64")
        )
        assert_series_equal(
            feature_data_prepared_series.astype("float64"),
            pd.Series([13.0, -1.0, -1.0, 1.0, 3.0, 13.0, 13.0, 13.0]).astype("float64")
        )

    def test_prepare_numerical_feature_2(self):
        feature_data = pd.Series(["-100", "-2", "M", "1", "2", "14", np.nan, None])
        feature_data_prepared_series, _ = prepare_numerical_feature_series(
            feature_data, self.feature_model_config_numerical, train_ind=None, log=False
        )
        feature_data_prepared_dict = pd.Series(
            [prepare_numerical_feature(v, self.feature_model_config_numerical) for v in feature_data]
        )
        assert_series_equal(
            feature_data_prepared_series.astype("float64"),
            feature_data_prepared_dict.astype("float64")
        )
        assert_series_equal(
            feature_data_prepared_series.astype("float64"),
            pd.Series([13.0, -1., -1.0, 1.0, 3.0, 13.0, 13.0, 13.0]).astype("float64")
        )

    def test_prepare_numerical_feature_cut(self):
        feature_data = pd.Series([-100, -2, -1, 1, 2, 14, np.nan, None])
        feature_data_prepared_series, _ = prepare_numerical_feature_series(
            feature_data, self.feature_model_config_numerical_cut, train_ind=None, log=False
        )
        feature_data_prepared_dict = pd.Series(
            [prepare_numerical_feature(v, self.feature_model_config_numerical_cut) for v in feature_data]
        )
        assert_series_equal(feature_data_prepared_series, feature_data_prepared_dict)
        assert_series_equal(
            feature_data_prepared_series,
            pd.Series(["(3.0, inf]", "(-inf, 1.0]", "(-inf, 1.0]", "(-inf, 1.0]", "(1.0, 3.0]", "(3.0, inf]", "(3.0, inf]", "(3.0, inf]"])
        )


if __name__ == '__main__':
    main()