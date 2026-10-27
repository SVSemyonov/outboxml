import pytest

from conftest import TEST_CONFIGS_PATH
from outboxml.core.errors import ConfigError
from outboxml.core.prepared_datasets import PrepareDataset
from outboxml.datasets_manager import DataSetsManager

config_name = str(TEST_CONFIGS_PATH / "config-example-titanic.json")


def test_user_prepared_datasets_get_their_model_config_by_name():
    ds_manager = DataSetsManager(config_name=config_name,
                                 prepared_datasets={"second": PrepareDataset(), "first": PrepareDataset()})

    for name in ("first", "second"):
        assert ds_manager._prepare_datasets[name].get_model_config().name == name
    assert sorted(ds_manager.get_subset("second").X_train.columns) == sorted(
        f.name for f in ds_manager._prepare_datasets["second"].get_model_config().features)


def test_prepared_dataset_for_unknown_model_raises():
    with pytest.raises(ConfigError, match="third"):
        DataSetsManager(config_name=config_name, prepared_datasets={"third": PrepareDataset()}).get_subset("third")
