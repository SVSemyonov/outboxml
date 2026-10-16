from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from outboxml.core.email import AutoMLReviewEMail, EMail, EMailDSCompareResult, EMailDSResult, EMailMonitoring
from outboxml.datasets_manager import DataSetsManager

RECEIVERS = ["ds-team@example.com"]


@pytest.fixture
def mail_config(tmp_path):
    return SimpleNamespace(email_smtp_server="smtp.example.com", email_port=2525,
                           email_sender="outboxml@example.com", email_login="login", email_pass="password",
                           email_receivers=RECEIVERS, results_path=tmp_path)


@pytest.fixture
def smtp():
    with patch("outboxml.dsml.mailing.smtplib.SMTP") as smtp_cls:
        yield smtp_cls.return_value.__enter__.return_value


def saved_html(mail_config) -> str:
    return (mail_config.results_path / "email.html").read_text(encoding="utf-8")


def test_saved_html_contains_mail_body(mail_config, smtp):
    email = EMail(config=mail_config)
    email.base_mail(header_name="Report", text="Model metrics are ready")

    email.send()

    assert "Model metrics are ready" in saved_html(mail_config)


def test_release_mail_is_saved(mail_config, smtp):
    EMail(config=mail_config).success_release_mail(group_name="example", new_features={"first": ["CABIN"]})

    assert "CABIN" in saved_html(mail_config)
    smtp.sendmail.assert_called_once()


@pytest.fixture(scope="module")
def ds_result():
    config_name = Path(__file__).resolve().parent / "test_configs" / "config-example-titanic.json"
    ds_manager = DataSetsManager(config_name=str(config_name))
    ds_manager.fit_models()
    return ds_manager.get_result()


def test_ds_result_mail(mail_config, smtp, ds_result):
    EMailDSResult(config=mail_config, ds_manager_result=ds_result).success_mail(group_name="titanic_group")

    assert "Характеристики моделей" in saved_html(mail_config)
    assert sorted(p.name for p in mail_config.results_path.glob("* cohort.html")) == ["first cohort.html",
                                                                                      "second cohort.html"]
    smtp.sendmail.assert_called_once()


def test_ds_compare_result_mail(mail_config, smtp, ds_result):
    EMailDSCompareResult(config=mail_config, ds_manager_result=ds_result,
                         ds_result_to_compare=ds_result).success_mail(group_name="titanic_group")

    assert "Характеристики моделей" in saved_html(mail_config)
    smtp.sendmail.assert_called_once()


def test_automl_error_mail_is_one_mail_with_error_and_status(mail_config, smtp):
    AutoMLReviewEMail(config=mail_config).error_mail(group_name="example", error=RuntimeError("fit failed"),
                                                     status={"Fitting": False})

    smtp.sendmail.assert_called_once()
    html = saved_html(mail_config)
    assert "fit failed" in html and "Fitting" in html


def test_monitoring_error_mail_is_sent(mail_config, smtp):
    EMailMonitoring(config=mail_config).error_mail(group_name="example", error=KeyError("PSI"))

    smtp.sendmail.assert_called_once()
    assert "PSI" in saved_html(mail_config)
