import json
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, event

from conftest import TEST_CONFIGS_PATH, TEST_DATA_PATH
from outboxml import config
from outboxml.extractors import SimpleExtractor
from outboxml.monitoring_manager import MonitoringManager
from outboxml.monitoring_result import MonitoringResult

models_config = str(TEST_CONFIGS_PATH / "config-example-titanic.json")


@pytest.fixture
def monitoring_config(prod_pickle_name) -> dict:
    with open(TEST_CONFIGS_PATH / "monitoring_test_config.json", encoding="utf-8") as f:
        monitoring = json.load(f)
    monitoring["prod_models_path"] = str(config.prod_models_path)
    monitoring["pickle_name"] = prod_pickle_name
    monitoring["monitoring_factory"] = [{"type": "datadrift", "report": "base_datadrift_report",
                                         "group_models": False, "parameters": {"full_calc": True},
                                         "db_table_name": "titanic_drift"}]
    return monitoring


@pytest.fixture
def grafana_engine(tmp_path):
    """SQLite engine with a `public` schema, as GrafanaExport writes to public.<table>."""
    engine = create_engine(f"sqlite:///{(tmp_path / 'main.sqlite').as_posix()}")
    public_db = (tmp_path / "public.sqlite").as_posix()

    @event.listens_for(engine, "connect")
    def attach_public(dbapi_connection, connection_record):
        dbapi_connection.execute(f"ATTACH DATABASE '{public_db}' AS public")

    return engine


def manager(monitoring_config, **kwargs) -> MonitoringManager:
    logs = pd.read_csv(TEST_DATA_PATH / "titanic.csv").head(500)
    return MonitoringManager(monitoring_config=monitoring_config, models_config=models_config,
                             logs_extractor=SimpleExtractor(data=logs), **kwargs)


def test_review_exports_reports_to_grafana_and_sends_mail(monitoring_config, grafana_engine):
    email = MagicMock()

    result = manager(monitoring_config, grafana_connection=grafana_engine, email=email).review(send_mail=True)

    assert isinstance(result, MonitoringResult)
    assert result.dataset_name == "titanic"
    email.success_mail.assert_called_once_with(result)
    exported = pd.read_sql("select * from public.titanic_drift", grafana_engine)
    assert not exported.empty


def test_review_sends_error_mail_when_success_mail_fails(monitoring_config):
    email = MagicMock()
    email.success_mail.side_effect = RuntimeError("smtp down")

    result = manager(monitoring_config, email=email).review(send_mail=True, to_grafana=False)

    assert isinstance(result, MonitoringResult)
    email.error_mail.assert_called_once()
    assert "smtp down" in str(email.error_mail.call_args.kwargs["error"])


def test_monitoring_config_from_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        manager(str(TEST_CONFIGS_PATH / "missing.json"))


def test_invalid_monitoring_config_raises(monitoring_config):
    del monitoring_config["group_name"]

    with pytest.raises((ValidationError, TypeError)):
        manager(monitoring_config)
