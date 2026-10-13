from unittest.mock import MagicMock, patch

import mlflow
import pytest

from outboxml.automl_manager import MLFlowKeycloakWrapper, MLFlowWrapper

UNREACHABLE_URI = "http://127.0.0.1:1"


@pytest.fixture
def restore_mlflow(monkeypatch):
    """MLFlow keeps tracking URI and token in process state; restore them after each test."""
    tracking_uri = mlflow.get_tracking_uri()
    monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)
    yield
    if mlflow.active_run():
        mlflow.end_run()
    mlflow.set_tracking_uri(tracking_uri)


def test_init_does_not_contact_server(restore_mlflow):
    with patch("outboxml.automl_manager.mlflow.set_experiment") as set_experiment:
        MLFlowWrapper(experiment_name="test", mlflow_tracking_uri=UNREACHABLE_URI)
    set_experiment.assert_not_called()


def test_start_run_sets_up_tracking(restore_mlflow, tmp_path):
    store_uri = (tmp_path / "mlruns").as_uri()
    wrapper = MLFlowWrapper(experiment_name="lazy_init", group_name="group", mlflow_tracking_uri=store_uri)

    wrapper.start_run()
    run = mlflow.active_run()
    wrapper.end_run()

    assert mlflow.get_tracking_uri() == store_uri
    assert mlflow.get_experiment(run.info.experiment_id).name == "lazy_init"
    assert run.info.run_name == "group"


def test_keycloak_login_is_deferred_until_run(restore_mlflow):
    response = MagicMock()
    response.json.return_value = {"access_token": "token", "refresh_token": "refresh", "expires_in": 300}

    with patch("outboxml.automl_manager.requests.post", return_value=response) as post:
        wrapper = MLFlowKeycloakWrapper(mlflow_tracking_uri=UNREACHABLE_URI,
                                        keycloak_url="https://keycloak.local",
                                        realm="realm",
                                        client_id="client",
                                        username="user",
                                        password="password")
        post.assert_not_called()

        wrapper._ensure_valid_token()
        wrapper._ensure_valid_token()

    post.assert_called_once()
    assert post.call_args.kwargs["data"]["grant_type"] == "password"


def test_keycloak_without_settings_skips_login(restore_mlflow):
    with patch("outboxml.automl_manager.requests.post") as post:
        wrapper = MLFlowKeycloakWrapper(mlflow_tracking_uri=UNREACHABLE_URI)
        wrapper._ensure_valid_token()
    post.assert_not_called()
