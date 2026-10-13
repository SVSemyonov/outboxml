from conftest import TESTS_PATH
from outboxml import config


def test_results_path_outside_repository():
    assert not config.results_path.is_relative_to(TESTS_PATH.parent)


def test_prod_models_path_outside_repository():
    assert not config.prod_models_path.is_relative_to(TESTS_PATH.parent)
    assert not config.prod_path.is_relative_to(TESTS_PATH.parent)


def test_mlflow_uses_local_file_store():
    assert config.mlflow_tracking_uri.startswith("file:")


def test_database_is_local_sqlite():
    assert config.connection_params.startswith("sqlite:///")
