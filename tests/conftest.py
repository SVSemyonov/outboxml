import pickle
from datetime import datetime, timedelta
from pathlib import Path

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
