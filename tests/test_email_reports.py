from types import SimpleNamespace
from unittest.mock import patch

import pytest

from outboxml.core.email import EMail

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
