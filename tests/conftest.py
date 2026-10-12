from pathlib import Path

import pytest

from outboxml import config

TESTS_PATH = Path(__file__).resolve().parent
TEST_CONFIGS_PATH = TESTS_PATH / "test_configs"
TEST_DATA_PATH = TESTS_PATH / "test_data"


@pytest.fixture(scope="session", autouse=True)
def isolated_config(tmp_path_factory):
    """Redirect outboxml.config outputs away from the repository and external services.

    Results go to a session temp dir, MLflow uses a local file store instead of a tracking server.
    """
    root = tmp_path_factory.mktemp("outboxml")
    results_path = root / "results"
    results_path.mkdir()
    mlflow_store = root / "mlruns"

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "results_path", results_path)
        mp.setattr(config, "mlflow_tracking_uri", mlflow_store.as_uri())
        yield config


@pytest.fixture(scope="session")
def test_configs_path() -> Path:
    return TEST_CONFIGS_PATH


@pytest.fixture(scope="session")
def test_data_path() -> Path:
    return TEST_DATA_PATH
