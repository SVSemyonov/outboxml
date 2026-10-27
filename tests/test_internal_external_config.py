from types import SimpleNamespace
from unittest.mock import patch

import pytest

from conftest import TEST_CONFIGS_PATH
from outboxml import config
from outboxml.core.email import EMailDSCompareResult, EMailDSResult


@pytest.fixture
def package_results_path(tmp_path, monkeypatch):
    """The default results_path of outboxml.config: nothing may be written there."""
    path = tmp_path / "package_results"
    monkeypatch.setattr(config, "results_path", path)
    return path


@pytest.mark.parametrize("mail_cls", [EMailDSResult, EMailDSCompareResult])
def test_ds_result_mails_use_their_config(fitted_ds_manager, package_results_path, tmp_path, mail_cls):
    mail_config = SimpleNamespace(email_smtp_server="smtp", email_port=25, email_sender="a@b", email_login="l",
                                  email_pass="p", email_receivers=["c@d"], results_path=tmp_path / "mail_results")
    result = fitted_ds_manager.get_result()
    kwargs = {"ds_result_to_compare": result} if mail_cls is EMailDSCompareResult else {}

    with patch("outboxml.dsml.mailing.smtplib.SMTP"):
        mail_cls(config=mail_config, ds_manager_result=result, **kwargs).success_mail(group_name="g")

    assert (tmp_path / "mail_results" / "email.html").exists()
    assert not package_results_path.exists()


def test_automl_compare_metrics_uses_external_config(fitted_ds_manager, package_results_path, tmp_path):
    from outboxml.automl_manager import AutoMLManager

    external = SimpleNamespace(**{k: getattr(config, k) for k in dir(config) if not k.startswith("_")})
    external.results_path = tmp_path / "automl_results"
    automl = AutoMLManager(auto_ml_config={}, models_config=str(TEST_CONFIGS_PATH / "config-example-titanic.json"),
                           external_config=external, retro=False, hp_tune=False)
    automl._results = fitted_ds_manager.get_result()

    automl._compare_base_metrics_df(result_to_compare=fitted_ds_manager.get_result())

    assert not package_results_path.exists()
