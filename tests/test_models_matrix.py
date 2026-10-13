import numpy as np
import pandas as pd
import pytest

from conftest import TEST_DATA_PATH
from outboxml.datasets_manager import DataSetsManager
from outboxml.extractors import SimpleExtractor


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    data = pd.read_csv(TEST_DATA_PATH / "titanic.csv")
    rng = np.random.default_rng(1)
    data["EXPOSURE"] = rng.uniform(0.5, 1.5, size=len(data))
    data["FARE_TARGET"] = data["FARE"] + 1.0  # strictly positive target for gamma
    data["CLAIMS"] = rng.poisson(1.0, size=len(data))  # count target for poisson
    return data


def models_config(wrapper: str, objective: str, target: str, exposure: str = None, **model_params) -> dict:
    model = {
        "name": "model",
        "wrapper": wrapper,
        "objective": objective,
        "column_target": target,
        "column_exposure": exposure,
        "features": [
            {"name": "SEX", "default": "male", "replace": {"MALE": "male", "FEMALE": "female"}},
            {"name": "AGE", "default": "_MEDIAN_", "replace": {"_TYPE_": "_NUM_"}},
            {"name": "PCLASS", "default": 3, "replace": {"_TYPE_": "_NUM_"}},
        ],
        **model_params,
    }
    return {
        "group_name": "matrix", "project": "matrix", "version": "1",
        "data_config": {"source": "csv", "local_name_source": "unused.csv",
                        "separation": {"kind": "random", "random_state": 1, "test_train_proportion": 0.2},
                        "data": {"targetslices": []}},
        "models_configs": [model],
    }


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
def test_wrapper_objective_fits_and_predicts(data, wrapper, objective, target, exposure):
    result, predictions = fit_and_predict(data, wrapper=wrapper, objective=objective, target=target,
                                          exposure=exposure)

    assert len(predictions) == len(result.data_subset.X_test)
    assert np.isfinite(predictions).all()
    if objective in ("poisson", "gamma"):
        assert (predictions > 0).all()
    assert set(result.metrics) == {"train", "test"}


@pytest.mark.parametrize("treatment_dict", [{"SEX": "'female'"}, {"SEX": "'unknown_level'"}])
def test_glm_treatment_base_level(data, treatment_dict):
    result, predictions = fit_and_predict(data, wrapper="glm", objective="poisson", target="CLAIMS",
                                          treatment_dict=treatment_dict)

    assert np.isfinite(predictions).all()
    assert any("SEX" in name for name in result.model.model.params.index)
