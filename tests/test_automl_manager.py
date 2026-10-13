from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from conftest import TEST_CONFIGS_PATH
from outboxml.automl_manager import AutoMLManager, MLFlowKeycloakWrapper


@pytest.fixture
def automl() -> AutoMLManager:
    return AutoMLManager(auto_ml_config=str(TEST_CONFIGS_PATH / "automl-titanic.json"),
                         models_config=str(TEST_CONFIGS_PATH / "config-example-titanic.json"),
                         retro=False, hp_tune=False)


def decide(automl, criteria: dict, compare_business_metric=None, compare_metrics_df=None) -> bool:
    automl._auto_ml_config.inference_criteria.metric_growth_value = criteria
    automl.automl_results.compare_business_metric = compare_business_metric
    if compare_metrics_df is not None:
        automl.automl_results.compare_metrics_df = compare_metrics_df
    automl.deployment()
    assert automl.status["Deployment decision"] is True
    return automl.automl_results.deployment


@pytest.mark.parametrize("difference, expected", [(None, True), (0.5, True), (0.0001, False), (-1.0, False)])
def test_deployment_by_compare_business_metric(automl, difference, expected):
    assert decide(automl, {"CompareBusinessMetric": 0.001}, {"difference": difference}) is expected


@pytest.mark.parametrize("values, expected", [([0.9, 0.8], True), ([0.9, 0.1], False)])
def test_deployment_by_metric_minimum(automl, values, expected):
    deployed = decide(automl, {"accuracy": 0.5}, compare_metrics_df=pd.DataFrame({"accuracy": values}))

    assert deployed is expected
    assert automl.automl_results.business_metric["accuracy"] == min(values)


def test_deployment_requires_all_criteria(automl):
    deployed = decide(automl, {"CompareBusinessMetric": 0.001, "accuracy": 0.5}, {"difference": 0.5},
                      pd.DataFrame({"accuracy": [0.1]}))

    assert deployed is False


@pytest.mark.xfail(strict=True, reason="a criterion metric missing from compare_metrics_df is skipped on KeyError "
                                       "and all([]) is True, so a misspelled metric name deploys the model")
def test_deployment_with_unknown_metric_is_rejected(automl):
    assert decide(automl, {"misspelled_metric": 0.5}, compare_metrics_df=pd.DataFrame({"accuracy": [0.9]})) is False


def test_review_html_reports(automl):
    with patch("outboxml.automl_manager.HTMLReport") as report:
        automl.review(email=MagicMock(), send_mail=False)
        automl.review(email=MagicMock(), send_mail=False, error=RuntimeError("fit failed"))

    report.return_value.success_report.assert_called_once_with(automl.automl_results)
    assert report.return_value.error_report.call_args.kwargs["error"] == "fit failed"


def test_review_error_mail(automl):
    email = MagicMock()

    automl.review(email=email, send_mail=True, error=RuntimeError("fit failed"))

    email.common_error_mail.assert_called_once_with(group_name=automl.group_name, error="fit failed")
    assert automl.status["EMail Review"] is True


@pytest.mark.xfail(strict=True, reason="review reads status['Loading pickle to git'], a key that does not exist")
def test_review_success_mail(automl):
    email = MagicMock()

    automl.review(email=email, send_mail=True)

    email.success_mail.assert_called_once_with(automl.automl_results)
    assert automl.status["EMail Review"] is True


def test_update_models_failure_logs_and_sends_error_mail(automl):
    automl.mlflow = MagicMock()
    with patch.object(automl, "feature_selection", side_effect=RuntimeError("selection failed")), \
            patch("outboxml.automl_manager.AutoMLReviewEMail") as email_cls:
        result = automl.update_models(send_mail=True)

    assert result is automl.automl_results
    assert automl.status["Fitting"] is False
    automl.mlflow.log_results.assert_called_once_with(automl.automl_results)
    error_mail = email_cls.return_value.error_mail
    error_mail.assert_called_once()
    assert str(error_mail.call_args.kwargs["error"]) == "selection failed"


def keycloak_response(token: str):
    response = MagicMock()
    response.json.return_value = {"access_token": token, "refresh_token": f"refresh-{token}", "expires_in": 0}
    return response


def test_keycloak_refreshes_expired_token_and_reauthenticates_on_failure(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)
    wrapper = MLFlowKeycloakWrapper(mlflow_tracking_uri="http://127.0.0.1:1", keycloak_url="https://keycloak.local",
                                    realm="realm", client_id="client", client_secret="secret",
                                    username="user", password="password")

    with patch("outboxml.automl_manager.requests.post", return_value=keycloak_response("t1")) as post:
        wrapper._ensure_valid_token()  # first login
        wrapper._ensure_valid_token()  # expired (expires_in=0): refresh
    grants = [call.kwargs["data"]["grant_type"] for call in post.call_args_list]
    assert grants == ["password", "refresh_token"]
    assert post.call_args.kwargs["data"]["client_secret"] == "secret"

    failing = MagicMock()
    failing.raise_for_status.side_effect = RuntimeError("expired refresh token")
    with patch("outboxml.automl_manager.requests.post", side_effect=[failing, keycloak_response("t2")]) as post:
        wrapper._ensure_valid_token()
    assert [call.kwargs["data"]["grant_type"] for call in post.call_args_list] == ["refresh_token", "password"]
