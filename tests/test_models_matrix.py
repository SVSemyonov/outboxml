import numpy as np
import pytest

from conftest import models_config
from outboxml.datasets_manager import DataSetsManager
from outboxml.extractors import SimpleExtractor


def fit_and_predict(data, **config):
    ds_manager = DataSetsManager(config_name=models_config(**config), extractor=SimpleExtractor(data=data.copy()))
    ds_manager.fit_models()
    result = ds_manager.get_result()["model"]
    return result, np.asarray(result.model.predict(result.data_subset.X_test), dtype=float)


@pytest.mark.parametrize("wrapper, objective, target, exposure", [
    ("catboost", "poisson", "CLAIMS", "EXPOSURE"),
    ("catboost", "gamma", "FARE_TARGET", None),
    ("glm", "poisson", "CLAIMS", "EXPOSURE"),
    ("glm", "gamma", "FARE_TARGET", None),
    ("glm", "RMSE", "FARE_TARGET", None),
    ("glm_without_scaler", "poisson", "CLAIMS", None),
    ("xgboost", "poisson", "CLAIMS", "EXPOSURE"),
    ("xgboost", "gamma", "FARE_TARGET", None),
    ("xgboost", "binary", "SURVIVED", None),
    ("catboost_over_glm", "poisson", "CLAIMS", "EXPOSURE"),
    ("catboost_over_glm", "gamma", "FARE_TARGET", None),
])
def test_wrapper_objective_fits_and_predicts(titanic_synthetic, wrapper, objective, target, exposure):
    result, predictions = fit_and_predict(titanic_synthetic, wrapper=wrapper, objective=objective, target=target,
                                          exposure=exposure)

    assert len(predictions) == len(result.data_subset.X_test)
    assert np.isfinite(predictions).all()
    if objective in ("poisson", "gamma"):
        assert (predictions > 0).all()
    assert set(result.metrics) == {"train", "test"}


@pytest.mark.parametrize("treatment_dict", [{"SEX": "'female'"}, {"SEX": "'unknown_level'"}])
def test_glm_treatment_base_level(titanic_synthetic, treatment_dict):
    result, predictions = fit_and_predict(titanic_synthetic, wrapper="glm", objective="poisson", target="CLAIMS",
                                          treatment_dict=treatment_dict)

    assert np.isfinite(predictions).all()
    assert any("SEX" in name for name in result.model.model.params.index)
