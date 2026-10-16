from email import message_from_string
from types import SimpleNamespace
from unittest.mock import patch

import pathlib

import pandas as pd
import plotly.graph_objects as go
import pytest

from outboxml.automl_manager import AutoMLResult
from outboxml.core.email import (AutoMLReviewEMail, EMail, EMailDSCompareResult, EMailDSResult, EMailMonitoring,
                                 HTMLReport)

RECEIVERS = ["ds-team@example.com", "mlops@example.com"]


@pytest.fixture
def mail_config(tmp_path):
    return SimpleNamespace(email_smtp_server="smtp.example.com", email_port=2525,
                           email_sender="outboxml@example.com", email_login="login", email_pass="password",
                           email_receivers=RECEIVERS, results_path=tmp_path)


@pytest.fixture
def smtp():
    with patch("outboxml.dsml.mailing.smtplib.SMTP") as smtp_cls:
        yield smtp_cls.return_value.__enter__.return_value


def sent_parts(smtp_server, index: int = -1):
    """MIME parts of a message passed to SMTP.sendmail (the last one by default)."""
    sender, receivers, message = smtp_server.sendmail.call_args_list[index].args
    assert receivers == RECEIVERS
    return list(message_from_string(message).walk())


def sent_html(smtp_server, index: int = -1) -> str:
    """HTML body of a sent message, attachments excluded."""
    for part in sent_parts(smtp_server, index):
        if part.get_content_type() == "text/html" and part.get_filename() is None:
            return part.get_payload(decode=True).decode()
    raise AssertionError("no html body in the sent message")


def sent_attachments(smtp_server, index: int = -1) -> list:
    return [part.get_filename() for part in sent_parts(smtp_server, index) if part.get_filename()]


def test_send_saves_html_and_sends_to_receivers(mail_config, smtp):
    email = EMail(config=mail_config)
    email.base_mail(header_name="Test", text="Hello")

    email.send()

    smtp.starttls.assert_called_once()
    smtp.login.assert_called_once_with("login", "password")
    assert (mail_config.results_path / "email.html").exists()
    assert "Hello" in sent_html(smtp)


def test_common_error_mail_contains_error(mail_config, smtp):
    EMail(config=mail_config).common_error_mail(group_name="example", error=ValueError("boom"))

    assert "boom" in sent_html(smtp)


def test_success_release_mail_lists_new_features(mail_config, smtp):
    EMail(config=mail_config).success_release_mail(group_name="example", new_features={"first": ["CABIN"]})

    assert "CABIN" in sent_html(smtp)


@pytest.fixture
def automl_result():
    result = AutoMLResult(group_name="example_titanic")
    result.new_features = {"first": ["CABIN"]}
    result.deployment = True
    result.compare_metrics_df = pd.DataFrame({"metric": ["mae"], "new": [0.1], "old": [0.2]})
    result.figures = {"first": go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))}
    return result


@pytest.mark.parametrize("deployment", [True, False])
def test_automl_review_success_mail(mail_config, smtp, automl_result, deployment):
    automl_result.deployment = deployment

    AutoMLReviewEMail(config=mail_config).success_mail(automl_result)

    assert (mail_config.results_path / "first figure.html").exists()
    assert sent_attachments(smtp) == ["first figure.html"]
    assert "first figure.html" in sent_html(smtp)
    assert "CABIN" in sent_html(smtp)


def test_automl_review_error_mail(mail_config, smtp):
    AutoMLReviewEMail(config=mail_config).error_mail(group_name="example", error=RuntimeError("fit failed"),
                                                     status={"Fitting": False})

    assert "fit failed" in sent_html(smtp, index=0)


def test_monitoring_mail_reports_only_drifted_features(mail_config, smtp):
    report = pd.DataFrame({"model_name": ["first", "first"], "col": ["AGE", "SEX"], "PSI": [0.5, 0.1],
                           "KL": [0.3, 0.1], "JS": [0.1, 0.05], "model_version": ["1", "1"]})
    monitoring_result = SimpleNamespace(group_name="example", report=report, grafana_dashboard="TitanicMonitoring")

    EMailMonitoring(config=mail_config).success_mail(monitoring_result)

    html = sent_html(smtp)
    assert "AGE" in html and "TitanicMonitoring" in html
    assert EMailMonitoring(config=mail_config)._prepare_drift_df(report)["col"].tolist() == ["AGE"]
    assert EMailMonitoring(config=mail_config)._prepare_drift_df(None).empty


def test_monitoring_error_mail(mail_config, smtp):
    EMailMonitoring(config=mail_config).error_mail(group_name="example", error=KeyError("PSI"))

    smtp.sendmail.assert_called_once()


@pytest.mark.parametrize("deployment", [True, False])
def test_html_report(mail_config, automl_result, deployment):
    automl_result.deployment = deployment

    HTMLReport(config=mail_config).success_report(automl_result)

    report = (mail_config.results_path / "automl_report.html").read_text(encoding="utf-8")
    assert "example_titanic" in report


def test_html_error_report(mail_config):
    HTMLReport(config=mail_config).error_report(group_name="example", error=RuntimeError("fit failed"),
                                                status={"Fitting": False})

    assert "fit failed" in (mail_config.results_path / "automl_report.html").read_text(encoding="utf-8")


def test_mails_do_not_render_images(mail_config, smtp, automl_result, monkeypatch):
    def write_image(*args, **kwargs):
        raise AssertionError("PNG export needs kaleido, which can hang on Windows")
    monkeypatch.setattr(go.Figure, "write_image", write_image)

    AutoMLReviewEMail(config=mail_config).success_mail(automl_result)

    assert sent_attachments(smtp) == ["first figure.html"]
