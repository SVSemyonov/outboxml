import json

import pandas as pd

from conftest import TEST_DATA_PATH
from outboxml.core.pydantic_models import AllModelsConfig
from outboxml.extractors import PolarsExtractor, get_useful_columns


def models_config(local_name_source: str = "unused.csv") -> dict:
    return {
        "group_name": "g",
        "data_config": {"source": "csv", "local_name_source": local_name_source},
        "models_configs": [{"name": "m", "column_target": "SURVIVED", "column_weight": "ROW_WEIGHT",
                            "relative_features": [{"name": "FARE_PER_AGE", "numerator": "FARE",
                                                   "denominator": "AGE"}],
                            "features": [{"name": "SEX"}]}],
    }


def test_useful_columns_include_weight_and_relative_feature_sources():
    config = AllModelsConfig.model_validate(models_config())

    features, targets, extra_columns = get_useful_columns(config.data_config, config.models_configs)

    assert {"FARE", "AGE", "SEX"} <= set(features)
    assert {"SURVIVED", "ROW_WEIGHT"} <= set(targets)


def test_polars_extractor_keeps_weight_and_relative_feature_columns(tmp_path):
    data = pd.read_csv(TEST_DATA_PATH / "titanic.csv")
    data["ROW_WEIGHT"] = 1.0
    data.to_csv(tmp_path / "titanic.csv", index=False)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(models_config(str(tmp_path / "titanic.csv"))), encoding="utf-8")

    dataset = PolarsExtractor(str(config_path)).extract_dataset()

    assert {"ROW_WEIGHT", "FARE", "AGE", "SEX", "SURVIVED"} <= set(dataset.columns)
