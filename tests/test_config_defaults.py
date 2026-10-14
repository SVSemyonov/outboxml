import warnings

import pytest

from outboxml.core.config_builders import AutoMLConfigBuilder
from outboxml.core.enums import SeparationParams
from outboxml.core.pydantic_models import (AllModelsConfig, AutoMLConfig, DataConfig, DataModelConfig,
                                           FeatureModelConfig, ModelInferenceConfig, MonitoringConfig,
                                           RelativeFeatureModelConfig)


def test_minimal_all_models_config():
    config = AllModelsConfig.model_validate({
        "group_name": "titanic",
        "data_config": {"source": "csv", "local_name_source": "titanic.csv"},
        "models_configs": [{"name": "first", "features": [{"name": "SEX"},
                                                          {"name": "AGE", "replace": {"_TYPE_": "_NUM_"}}]}],
    })

    assert (config.project, config.version) == ("default", "1")
    separation = config.data_config.separation
    assert (separation.kind, separation.random_state, separation.test_train_proportion) == \
        (SeparationParams.rand, 42, 0.2)
    assert config.data_config.data.targetslices == []
    sex, age = config.models_configs[0].features
    assert (sex.default, sex.replace) == ("_NAN_", {})
    assert age.default == "_MEDIAN_"


def test_explicit_values_are_kept():
    feature = FeatureModelConfig(name="AGE", default=0, replace={"_TYPE_": "_NUM_"})
    data_config = DataModelConfig(source="csv", separation={"kind": "random", "test_train_proportion": 0.3})

    assert feature.default == 0
    assert data_config.separation.test_train_proportion == 0.3


def test_relative_feature_default():
    assert RelativeFeatureModelConfig(name="R", numerator="A", denominator="B").default == "_NAN_"


def test_minimal_automl_config(monkeypatch):
    from conftest import TEST_CONFIGS_PATH
    from outboxml import config
    from outboxml.automl_manager import AutoMLManager

    monkeypatch.setattr(config, "mlflow_experiment", "env_experiment")
    automl = AutoMLManager(auto_ml_config={}, models_config=str(TEST_CONFIGS_PATH / "config-example-titanic.json"),
                           retro=False, hp_tune=False)

    assert automl.group_name == "example_titanic"
    assert automl._auto_ml_config.mlflow_experiment == "env_experiment"
    assert automl.mlflow.experiment_name == "env_experiment"
    assert automl._auto_ml_config.grafana_table_name == "example_titanic"
    assert automl._auto_ml_config.inference_criteria.metric_growth_value == {}
    assert automl._auto_ml_config.hp_tune.trials == 100


def test_minimal_monitoring_config():
    import pandas as pd

    from conftest import TEST_CONFIGS_PATH, TEST_DATA_PATH
    from outboxml.extractors import SimpleExtractor
    from outboxml.monitoring_manager import MonitoringManager

    logs = pd.read_csv(TEST_DATA_PATH / "titanic.csv").head(300)
    manager = MonitoringManager(monitoring_config={"group_name": "example"},
                                models_config=str(TEST_CONFIGS_PATH / "config-example-titanic.json"),
                                logs_extractor=SimpleExtractor(data=logs))

    assert manager._monitoring_config.grafana_table_name == "example"
    assert manager.result.dataset_name == "titanic"


@pytest.mark.parametrize("model, data, field", [
    (DataModelConfig, {"source": "csv", "showGraphs": True}, "showGraphs"),
    (DataModelConfig, {"source": "csv", "processing": True}, "processing"),
    (DataConfig, {"queries": ["select 1"]}, "queries"),
    (ModelInferenceConfig, {"prod_models_folder": "models"}, "prod_models_folder"),
    (AutoMLConfig, {"dashboard_name": "dash"}, "dashboard_name"),
    (MonitoringConfig, {"group_name": "g", "target_column": "SURVIVED"}, "target_column"),
    (MonitoringConfig, {"group_name": "g", "extrapolation_period": 6}, "extrapolation_period"),
])
def test_deprecated_fields_are_accepted_with_warning(model, data, field):
    with pytest.warns(DeprecationWarning, match=field):
        model.model_validate(data)


def test_current_fields_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        AutoMLConfig.model_validate({"group_name": "g", "hp_tune": {"trials": 5}})
        MonitoringConfig.model_validate({"group_name": "g"})
        AutoMLConfigBuilder().build()
