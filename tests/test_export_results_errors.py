from pathlib import Path
from unittest.mock import MagicMock

import pytest

from outboxml.datasets_manager import DataSetsManager
from outboxml.export_results import ResultExport

config_name = str(Path(__file__).resolve().parent / "test_configs" / "config-example-titanic.json")


@pytest.fixture(scope="module")
def ds_manager() -> DataSetsManager:
    ds_manager = DataSetsManager(config_name=config_name)
    ds_manager.fit_models()
    return ds_manager


def test_compare_without_second_result_raises_value_error(ds_manager):
    export = ResultExport(ds_manager=ds_manager)

    with pytest.raises(ValueError, match="No results to compare"):
        export.compare_metrics(model_name="first")
    with pytest.raises(ValueError, match="No results to compare"):
        export.compare_models_plot(model_name="first")


def test_result_to_compare_of_wrong_format_raises_type_error(ds_manager):
    wrong = MagicMock()
    wrong.get_result.return_value = ["not", "a", "dict"]

    with pytest.raises(TypeError, match="Wrong format of results"):
        ResultExport(ds_manager=ds_manager, ds_manager_to_compare=wrong)
