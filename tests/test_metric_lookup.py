from types import SimpleNamespace

import pytest

from outboxml.analysis_tools import CVStability
from outboxml.automl_manager import AutoMLManager

metric_for_model = AutoMLManager._AutoMLManager__metric_for_model


@pytest.mark.parametrize("metrics, expected", [
    ({"first": "r2", "default": "neg_root_mean_squared_error"}, "r2"),
    ({"default": "neg_root_mean_squared_error"}, "neg_root_mean_squared_error"),
    ({}, "neg_mean_absolute_error"),
])
def test_hp_tuning_metric_for_model(metrics, expected):
    assert metric_for_model(metrics, "first") == expected


@pytest.mark.parametrize("metric_eval, expected", [
    ({"first": "accuracy"}, "accuracy"),
    ({"default": "roc_auc"}, "roc_auc"),
    ({"metric_name": 0}, "neg_mean_absolute_error"),  # the FeatureSelectionConfig default
])
def test_cv_stability_scoring(metric_eval, expected):
    cv = object.__new__(CVStability)
    cv.config = SimpleNamespace(metric_eval=metric_eval)

    assert cv._CVStability__choose_scoring_fun("first") == expected
