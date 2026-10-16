from outboxml.automl_utils import build_default_auto_ml_config
from outboxml.core.config_builders import AutoMLConfigBuilder
from outboxml.core.pydantic_models import ModelInferenceConfig


def test_automl_builder_uses_given_parameters():
    criteria = ModelInferenceConfig(metric_growth_value={"accuracy": 0.7})
    trigger = {"table_name": "titanic_data", "field": "PASSENGERID"}

    config = AutoMLConfigBuilder(group_name="g", inference_criteria=criteria, mlflow_experiment="exp",
                                 grafana_table_name="table", trigger=trigger).build()

    assert config.inference_criteria.metric_growth_value == {"accuracy": 0.7}
    assert (config.mlflow_experiment, config.grafana_table_name, config.trigger) == ("exp", "table", trigger)


def test_automl_builder_leaves_environment_defaults_to_manager():
    config = build_default_auto_ml_config({"group_name": "g"})

    assert config.mlflow_experiment is None
    assert config.grafana_table_name is None
    assert config.inference_criteria.metric_growth_value == {"neg_mean_absolute_error": 0}
