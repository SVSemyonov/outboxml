import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from conftest import TEST_CONFIGS_PATH, TEST_DATA_PATH
from outboxml.app.fastapi import main as mlflow_app_module
from outboxml.app.fastapi import main_app
from outboxml.main_predict import app as predict_app


def records(df: pd.DataFrame) -> list:
    """JSON-ready rows: missing values become null, as an API client sends them."""
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


def load_json(name: str) -> dict:
    with open(TEST_CONFIGS_PATH / name, encoding="utf-8") as f:
        return json.load(f)


# --- prediction service (outboxml.main_predict) ---

@pytest.fixture
def predict_client():
    return TestClient(predict_app)


def test_health(predict_client):
    response = predict_client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"health": True}


def test_predict_route(predict_client, prod_pickle_name):
    rows = records(pd.read_csv(TEST_DATA_PATH / "titanic.csv").head(3))

    response = predict_client.post("/api/predict", json={"main_model": "example_titanic", "main_request": rows})

    assert response.status_code == 200, response.json()
    body = response.json()
    assert body["main_response"]["usage_model"] == prod_pickle_name
    assert set(body["main_response"]["result"]) == {"first", "second"}


def test_predict_route_unknown_model_returns_400(predict_client, prod_pickle_name):
    response = predict_client.post("/api/predict", json={"main_model": "missing_group", "main_request": [{}]})

    assert response.status_code == 400
    assert "error" in response.json()


# --- AutoML / monitoring application (outboxml.app.fastapi.main_app) ---

@pytest.fixture
def app_client():
    return TestClient(main_app.app)


def update_request(user_parameters):
    return {"auto_ml_config": load_json("automl-titanic.json"),
            "all_model_config": load_json("config-example-titanic.json"),
            "user_parameters": user_parameters}


def test_health_app(app_client):
    assert app_client.get("/api/health_app").json() == {"health_app": True}


def test_update_runs_automl(app_client):
    with patch.object(main_app, "AutoMLManager") as manager_cls:
        manager_cls.return_value.status = {"Fitting": True}
        response = app_client.post("/api/update", json=update_request({"hp_tune": True}))

    assert response.status_code == 200
    assert response.json() == {"Fitting": True}
    assert manager_cls.call_args.kwargs["hp_tune"] is True
    manager_cls.return_value.update_models.assert_called_once()


def test_update_error_returns_400(app_client):
    with patch.object(main_app, "AutoMLManager", side_effect=RuntimeError("no data")):
        response = app_client.post("/api/update", json=update_request({}))

    assert response.status_code == 400
    assert "no data" in response.json()["error"]


def test_update_without_user_parameters(app_client):
    with patch.object(main_app, "AutoMLManager") as manager_cls:
        manager_cls.return_value.status = {}
        response = app_client.post("/api/update", json=update_request(None))

    assert response.status_code == 200


def test_update_passes_retro(app_client):
    with patch.object(main_app, "AutoMLManager") as manager_cls:
        manager_cls.return_value.status = {}
        app_client.post("/api/update", json=update_request({"retro": True}))

    assert manager_cls.call_args.kwargs["retro"] is True


def monitoring_request():
    return {"monitoring_config": load_json("monitoring_test_config.json"),
            "all_model_config": load_json("config-example-titanic.json")}


def test_monitoring_route(app_client):
    with patch.object(main_app, "MonitoringManager") as manager_cls:
        response = app_client.post("/api/monitoring", json=monitoring_request())

    assert response.status_code == 200
    assert response.json() == {"Monitoring status": "OK"}
    manager_cls.return_value.review.assert_called_once()


def test_monitoring_route_error_returns_400(app_client):
    with patch.object(main_app, "MonitoringManager", side_effect=FileNotFoundError("pickle")):
        response = app_client.post("/api/monitoring", json=monitoring_request())

    assert response.status_code == 400


@pytest.mark.parametrize("path, key", [("/api/auto_ml_result", "Auto ml results"),
                                       ("/api/monitoring_result", "Monitoring results")])
def test_result_routes(app_client, path, key):
    response = app_client.request("GET", path, json={"main_model": "example_titanic", "request": {"metrics": True}})

    assert response.status_code == 200
    assert response.json() == {key: "OK"}


def test_default_automl_config(app_client):
    response = app_client.post("/api/default_automl_config", json={"group_name": "api_group"})

    assert response.status_code == 200
    assert response.json()["group_name"] == "api_group"


def test_default_all_models_config(app_client):
    rows = pd.read_csv(TEST_DATA_PATH / "titanic.csv").head(50)[["SURVIVED", "SEX", "AGE", "FARE"]]
    params = {"data": records(rows), "column_target": "SURVIVED", "group_name": "api_group"}

    response = app_client.post("/api/default_all_models_config", json=params)

    assert response.status_code == 200, response.json()
    features = {f["name"] for f in response.json()["models_configs"][0]["features"]}
    assert {"SEX", "AGE", "FARE"} <= features


def test_default_all_models_config_error_returns_400(app_client):
    response = app_client.post("/api/default_all_models_config", json={"unknown_param": 1})

    assert response.status_code == 400


# --- MLflow REST proxy demo (outboxml.app.fastapi.main) ---

def mlflow_response(status_code=200, payload=None):
    response = MagicMock(status_code=status_code)
    response.json.return_value = payload or {}
    return response


def test_mlflow_proxy_create_run_and_log_metrics():
    with patch.object(mlflow_app_module.requests, "post",
                      return_value=mlflow_response(payload={"run": {"info": {"run_id": "r1"}}})) as post:
        with TestClient(mlflow_app_module.app) as client:
            run = client.get("/create-run/").json()
            logged = client.post("/log-metrics/", params={"run_id": "r1"}, json={"mae": 0.1, "rmse": 0.2})

    assert run["run"]["info"]["run_id"] == "r1"
    assert logged.json() == {"message": "Metrics logged successfully"}
    metric_calls = [c for c in post.call_args_list if c.args[0].endswith("/runs/log-metric")]
    assert [c.kwargs["json"]["key"] for c in metric_calls] == ["mae", "rmse"]


def test_mlflow_proxy_errors_return_500():
    with patch.object(mlflow_app_module.requests, "post", return_value=mlflow_response(500, {"error": "down"})):
        with TestClient(mlflow_app_module.app) as client:
            assert client.get("/create-run/").status_code == 500
            assert client.post("/log-metrics/", params={"run_id": "r1"}, json={"mae": 0.1}).status_code == 500
