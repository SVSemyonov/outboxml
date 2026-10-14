import json
from types import SimpleNamespace

import pytest

from conftest import TEST_CONFIGS_PATH
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


def test_hp_tuning_with_default_metric_score():
    with open(TEST_CONFIGS_PATH / "automl-titanic.json", encoding="utf-8") as f:
        automl_config = json.load(f)
    automl_config["hp_tune"] = {"trials": 2, "n_jobs": 1, "cv_folds_num": 2,
                                "parameters": {"first": {"iterations": {"type": "int", "low": 10, "high": 12}}}}
    automl = AutoMLManager(auto_ml_config=automl_config,
                           models_config=str(TEST_CONFIGS_PATH / "config-example-titanic.json"),
                           retro=False, hp_tune=True)

    params = automl.hp_tuning()

    assert set(params["first"]) == {"iterations"}
