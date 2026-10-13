import pandas as pd
import pytest
from sqlalchemy import create_engine

from outboxml.export_results import GrafanaExport


@pytest.fixture
def engine(tmp_path):
    return create_engine(f"sqlite:///{(tmp_path / 'grafana.sqlite').as_posix()}")


def test_empty_dataframe_raises_value_error(engine):
    with pytest.raises(ValueError, match="Empty dataframe"):
        GrafanaExport(df=pd.DataFrame(), connection=engine)


def test_load_data_appends_rows(engine):
    df = pd.DataFrame({"metric": ["mae", "rmse"], "value": [0.1, 0.2]})
    exporter = GrafanaExport(df=df, table_name="metrics", schema=None, connection=engine)

    exporter.load_data_to_db()
    exporter.load_data_to_db()

    assert len(pd.read_sql("select * from metrics", engine)) == 4
