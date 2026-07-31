"""
Tests the round-trip export/import of a model:
    DSManagerResult.dict_for_export() -> DSManagerResult.from_dict()

Test idea:
1. Train models via DataSetsManager on titanic.csv using config-example-titanic.json.
2. For each resulting DSManagerResult, call dict_for_export() and check the
   dict structure (this mirrors what gets pickled for production).
3. Restore the result via DSManagerResult.from_dict(exported_dict).
4. Compare:
   - model name and feature lists (numerical/categorical) match;
   - predictions from the restored model match the original model's
     predictions on the same data (X_test), since from_dict should return
     a model equivalent to the original in terms of weights/parameters.

Note: test file paths are resolved relative to this file
(tests/test_data/titanic.csv and tests/test_config/config-example-titanic.json).
If your repo layout differs, adjust TEST_DATA_DIR/TEST_CONFIG_DIR.
"""

import os

import numpy as np
import pandas as pd
import pytest

from outboxml.datasets_manager import DataSetsManager, DSManagerResult

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(HERE, "test_data")
TEST_CONFIG_DIR = os.path.join(HERE, "test_config")

CSV_PATH = os.path.join(TEST_DATA_DIR, "titanic.csv")
CONFIG_PATH = os.path.join(TEST_CONFIG_DIR, "config-example-titanic.json")


@pytest.fixture(scope="module")
def titanic_df() -> pd.DataFrame:
    assert os.path.exists(CSV_PATH), f"Data file not found: {CSV_PATH}"
    return pd.read_csv(CSV_PATH)


@pytest.fixture(scope="module")
def ds_manager(titanic_df: pd.DataFrame) -> DataSetsManager:
    assert os.path.exists(CONFIG_PATH), f"Config file not found: {CONFIG_PATH}"

    manager = DataSetsManager(config_name=CONFIG_PATH)
    # Explicitly route the dataset through SimpleExtractor (load_dataset(data=...)
    # uses SimpleExtractor under the hood when no extractor was passed to the constructor).
    manager.load_dataset(data=titanic_df)
    return manager


@pytest.fixture(scope="module")
def fitted_results(ds_manager: DataSetsManager):
    ds_manager.fit_models(need_fit=True)
    results = ds_manager.get_result()
    assert len(results) > 0, "fit_models returned no results"
    return results


def test_dict_for_export_structure(fitted_results):
    """Check that dict_for_export returns the expected structure for each model."""
    expected_keys = {
        "model_config",
        "wrapper",
        "model",
        "glm_model",
        "catboost_model",
        "xgm_model",
        "min_max_scaler",
        "features_numerical",
        "features_categorical",
    }

    for model_name, result in fitted_results.items():
        exported = result.dict_for_export()

        assert expected_keys.issubset(exported.keys())
        assert exported["model_config"]["name"] == model_name
        assert exported["features_numerical"] == result.data_subset.features_numerical
        assert exported["features_categorical"] == result.data_subset.features_categorical


def test_from_dict_roundtrip_predictions(fitted_results):
    """Check that after export -> from_dict the model produces the same predictions."""
    for model_name, result in fitted_results.items():
        exported = result.dict_for_export()
        restored = DSManagerResult.from_dict(exported)

        # Basic metadata must match
        assert restored.model_name == model_name
        assert restored.data_subset.features_numerical == result.data_subset.features_numerical
        assert restored.data_subset.features_categorical == result.data_subset.features_categorical

        X_test = result.data_subset.X_test
        assert X_test is not None and len(X_test) > 0, (
            f"Empty X_test for model {model_name}, cannot compare predictions"
        )

        feature_cols = list(result.data_subset.features_numerical) + list(
            result.data_subset.features_categorical
        )

        original_preds = np.asarray(result.model.predict(X_test[feature_cols]))
        restored_preds = np.asarray(restored.model.predict(X_test[feature_cols]))

        np.testing.assert_allclose(
            original_preds,
            restored_preds,
            rtol=1e-6,
            atol=1e-8,
            err_msg=f"Predictions diverged after export/import for model {model_name}",
        )


def test_from_dict_preserves_model_config_fields(fitted_results):
    """Check that key model_config fields survive the export/import round-trip."""
    for model_name, result in fitted_results.items():
        exported = result.dict_for_export()
        restored = DSManagerResult.from_dict(exported)

        original_config = result.model_config
        restored_config = restored.model_config

        assert restored_config.name == original_config.name
        assert restored_config.wrapper == original_config.wrapper