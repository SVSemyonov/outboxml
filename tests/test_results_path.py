from types import SimpleNamespace
from unittest.mock import patch

import pytest

from conftest import TEST_CONFIGS_PATH
from outboxml import config
from outboxml.automl_manager import AutoMLResult
from outboxml.core.email import EMail, HTMLReport
from outboxml.datasets_manager import DataSetsManager
from outboxml.export_results import ResultExport

models_config = str(TEST_CONFIGS_PATH / "config-example-titanic.json")


@pytest.fixture
def missing_results_path(tmp_path, monkeypatch):
    path = tmp_path / "not" / "created" / "results"
    monkeypatch.setattr(config, "results_path", path)
    return path


def test_fit_models_with_temp_files_creates_results_path(missing_results_path):
    DataSetsManager(config_name=models_config, use_temp_files=True).fit_models()

    assert list(missing_results_path.glob("*.pickle")) or list(missing_results_path.glob("*.parquet"))


def test_result_export_creates_results_path(missing_results_path):
    ResultExport(ds_manager=DataSetsManager(config_name=models_config), config=config)

    assert missing_results_path.is_dir()


def test_mail_and_html_report_create_results_path(missing_results_path):
    mail_config = SimpleNamespace(email_smtp_server="smtp", email_port=25, email_sender="a@b", email_login="l",
                                  email_pass="p", email_receivers=["c@d"], results_path=missing_results_path)
    with patch("outboxml.dsml.mailing.smtplib.SMTP"):
        email = EMail(config=mail_config)
        email.base_mail(header_name="Test", text="text")
        email.send()
    HTMLReport(config=mail_config).error_report(group_name="g", error="boom", status={})

    assert (missing_results_path / "email.html").exists()
    assert (missing_results_path / "automl_report.html").exists()
