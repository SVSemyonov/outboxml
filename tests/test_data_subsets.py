import pandas as pd
import pytest

from outboxml.core.errors import ConfigError
from outboxml.core.pydantic_models import DataConfig, DataModelConfig
from outboxml.data_subsets import DataPreprocessor


def test_unknown_prepare_engine_raises_config_error():
    with pytest.raises(ConfigError, match="Unknown engine for data preparation: spark"):
        DataPreprocessor(prepare_dataset_interface_dict={},
                         dataset=pd.DataFrame(),
                         data_config=DataModelConfig(source="csv", data=DataConfig()),
                         prepare_engine="spark")
