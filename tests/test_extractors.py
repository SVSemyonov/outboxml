from pathlib import Path
from unittest.mock import patch

import pandas as pd

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
