from pathlib import Path

import pytest

TESTS_PATH = Path(__file__).resolve().parent
TEST_CONFIGS_PATH = TESTS_PATH / "test_configs"
TEST_DATA_PATH = TESTS_PATH / "test_data"


@pytest.fixture(scope="session")
def test_configs_path() -> Path:
    return TEST_CONFIGS_PATH


@pytest.fixture(scope="session")
def test_data_path() -> Path:
    return TEST_DATA_PATH
