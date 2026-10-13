import json

import numpy as np
import pytest

from conftest import TEST_CONFIGS_PATH
from outboxml.datasets_manager import DataSetsManager, DSManagerResult


def fitted_result(wrapper: str) -> DSManagerResult:
    with open(TEST_CONFIGS_PATH / "config-example-titanic.json", encoding="utf-8") as f:
        config = json.load(f)
    config["models_configs"] = config["models_configs"][:1]
    config["models_configs"][0]["wrapper"] = wrapper
    ds_manager = DataSetsManager(config_name=config)
    ds_manager.fit_models()
    return ds_manager.get_result()["first"]


@pytest.mark.parametrize("wrapper", [
    "catboost", "glm", "glm_without_scaler", "xgboost",
    pytest.param("catboost_over_glm", marks=pytest.mark.xfail(
        strict=True, reason="dict_for_export writes 'catbosot_model', from_dict reads 'catboost_model'")),
])
def test_export_and_import_keep_predictions(wrapper):
    result = fitted_result(wrapper)

    restored = DSManagerResult.from_dict(result.dict_for_export())

    X_test = result.data_subset.X_test
    assert restored.model_config == result.model_config
    assert restored.data_subset.features_numerical == result.data_subset.features_numerical
    np.testing.assert_allclose(np.asarray(restored.model.predict(X_test)), np.asarray(result.model.predict(X_test)))
