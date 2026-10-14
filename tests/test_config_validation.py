import json

import pytest
from pydantic import ValidationError

from conftest import TEST_CONFIGS_PATH
from outboxml.automl_manager import AutoMLManager
from outboxml.datasets_manager import DataSetsManager

models_config_path = TEST_CONFIGS_PATH / "config-example-titanic.json"


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_invalid_models_config_raises_validation_error():
    config = load_json(models_config_path)
    del config["models_configs"]

    with pytest.raises(ValidationError, match="models_configs"):
        DataSetsManager(config_name=config)


def test_invalid_automl_config_raises_validation_error():
    automl_config = load_json(TEST_CONFIGS_PATH / "automl-titanic.json")
    automl_config["hp_tune"] = "not a section"

    with pytest.raises(ValidationError, match="hp_tune"):
        AutoMLManager(auto_ml_config=automl_config, models_config=str(models_config_path), retro=False,
                      hp_tune=False)
