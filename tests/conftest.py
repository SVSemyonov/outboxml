import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from outboxml import config
from outboxml.core.utils import ResultPickle
from outboxml.datasets_manager import DataSetsManager

TESTS_PATH = Path(__file__).resolve().parent
TEST_CONFIGS_PATH = TESTS_PATH / "test_configs"
TEST_DATA_PATH = TESTS_PATH / "test_data"

PROD_GROUP_NAME = "example_titanic"


@pytest.fixture(scope="session", autouse=True)
def isolated_config(tmp_path_factory):
    """Redirect outboxml.config outputs away from the repository and external services.

    Results and production models go to a session temp dir,
    MLflow uses a local file store instead of a tracking server,
    database access goes to a SQLite file instead of Postgres.
    """
    root = tmp_path_factory.mktemp("outboxml")
    results_path = root / "results"
    results_path.mkdir()
    prod_models_path = root / config.prod_models_folder
    prod_models_path.mkdir()
    mlflow_store = root / "mlruns"

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "results_path", results_path)
        mp.setattr(config, "prod_path", root)
        mp.setattr(config, "prod_models_path", prod_models_path)
        mp.setattr(config, "mlflow_tracking_uri", mlflow_store.as_uri())
        mp.setattr(config, "connection_params", f"sqlite:///{(root / 'outboxml.sqlite').as_posix()}")
        yield config


@pytest.fixture(scope="session")
def prod_pickle_name(isolated_config) -> str:
    """Train models from the titanic test config and save them as a production pickle.

    :return: Pickle name without extension, as expected by ``main_predict`` and monitoring config.
    """
    ds_manager = DataSetsManager(config_name=str(TEST_CONFIGS_PATH / "config-example-titanic.json"))
    ds_manager.fit_models()
    group = [result.dict_for_prod_export() for result in ds_manager.get_result().values()]

    # ResultPickle ignores pickles dated now or later
    file_name = ResultPickle().generate_name(PROD_GROUP_NAME, datetime.now() - timedelta(minutes=1))
    with open(isolated_config.prod_models_path / file_name, "wb") as f:
        pickle.dump(group, f)
    return file_name.replace(".pickle", "")


@pytest.fixture(scope="session")
def fitted_ds_manager(isolated_config) -> DataSetsManager:
    """Titanic DataSetsManager with fitted models, shared read-only across tests."""
    ds_manager = DataSetsManager(config_name=str(TEST_CONFIGS_PATH / "config-example-titanic.json"))
    ds_manager.fit_models()
    return ds_manager


@pytest.fixture(scope="class")
def prod_pickle_cls(request, prod_pickle_name):
    """Expose the production pickle name to unittest.TestCase classes as ``self.prod_pickle_name``."""
    request.cls.prod_pickle_name = prod_pickle_name


@pytest.fixture(scope="session")
def test_configs_path() -> Path:
    return TEST_CONFIGS_PATH


@pytest.fixture(scope="session")
def test_data_path() -> Path:
    return TEST_DATA_PATH


@pytest.fixture(scope="session")
def titanic_synthetic() -> pd.DataFrame:
    """Titanic with synthetic EXPOSURE, ROW_WEIGHT, positive FARE_TARGET and count CLAIMS columns."""
    data = pd.read_csv(TEST_DATA_PATH / "titanic.csv")
    rng = np.random.default_rng(1)
    data["EXPOSURE"] = rng.uniform(0.5, 1.5, size=len(data))
    data["FARE_TARGET"] = data["FARE"] + 1.0  # strictly positive target for gamma
    data["CLAIMS"] = rng.poisson(1.0, size=len(data))  # count target for poisson
    data["ROW_WEIGHT"] = rng.uniform(0.5, 1.5, size=len(data))
    return data


def models_config(wrapper: str, objective: str, target: str, exposure: str = None, **model_params) -> dict:
    """All models config with one model `model` on SEX, AGE, PCLASS of titanic_synthetic."""
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
