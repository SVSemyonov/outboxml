from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from outboxml import config
from outboxml.core.errors import DataLoadError
from outboxml.core.pydantic_models import DataConfig, DataModelConfig
from outboxml.extractors import BaseExtractor

path_to_data = Path(__file__).resolve().parent / "test_data" / "titanic.csv"


def test_local_extraction_does_not_touch_database():
    data_config = DataModelConfig(source="csv",
                                  local_name_source=str(path_to_data),
                                  table_name_source="public.'TitanicExample'",
                                  data=DataConfig())

    with patch.object(pd.DataFrame, "to_sql") as to_sql, \
            patch("outboxml.extractors.create_engine") as create_engine:
        dataset = BaseExtractor(data_config=data_config).extract_dataset()

    assert dataset.shape[0] == 891
    to_sql.assert_not_called()
    create_engine.assert_not_called()


@pytest.fixture
def sqlite_connection(tmp_path, monkeypatch):
    connection = f"sqlite:///{(tmp_path / 'data.sqlite').as_posix()}"
    monkeypatch.setattr(config, "connection_params", connection)
    return connection


def db_config(table_name: str) -> DataModelConfig:
    return DataModelConfig(source="database", table_name_source=table_name, data=DataConfig())


def test_database_extraction(sqlite_connection):
    pd.read_csv(path_to_data).to_sql("titanic", sqlite_connection, index=False)

    dataset = BaseExtractor(data_config=db_config("titanic")).extract_dataset()

    assert dataset.shape == (891, 12)


def test_database_query_error_raises_data_load_error(sqlite_connection):
    with pytest.raises(DataLoadError, match="Sql query error"):
        BaseExtractor(data_config=db_config("missing_table")).extract_dataset()


def test_empty_database_table_raises_data_load_error(sqlite_connection):
    pd.read_csv(path_to_data).head(0).to_sql("empty", sqlite_connection, index=False)

    with pytest.raises(DataLoadError, match="No Data loaded"):
        BaseExtractor(data_config=db_config("empty")).extract_dataset()


def test_invalid_connection_raises_data_load_error(monkeypatch):
    monkeypatch.setattr(config, "connection_params", "unknown-dialect://host/db")

    with pytest.raises(DataLoadError, match="Cannot connect"):
        BaseExtractor(data_config=db_config("titanic")).extract_dataset()


def test_missing_local_file_raises_file_not_found(tmp_path):
    data_config = DataModelConfig(source="csv", local_name_source=str(tmp_path / "missing.csv"), data=DataConfig())

    with pytest.raises(FileNotFoundError):
        BaseExtractor(data_config=data_config).extract_dataset()


def test_empty_local_file_raises_data_load_error(tmp_path):
    empty_csv = tmp_path / "empty.csv"
    pd.read_csv(path_to_data).head(0).to_csv(empty_csv, index=False)
    data_config = DataModelConfig(source="csv", local_name_source=str(empty_csv), data=DataConfig())

    with pytest.raises(DataLoadError, match="No Data loaded"):
        BaseExtractor(data_config=data_config).extract_dataset()
