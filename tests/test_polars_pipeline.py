from pathlib import Path

import pandas as pd

from outboxml.datasets_manager import DataSetsManager
from outboxml.extractors import PolarsExtractor

# extra_columns contains the target column SURVIVED
config_name = str(Path(__file__).resolve().parent / "test_configs" / "config-example-titanic.json")


def polars_manager() -> DataSetsManager:
    return DataSetsManager(config_name=config_name, extractor=PolarsExtractor(config_name), prepare_engine="polars")


def test_polars_subset_with_target_in_extra_columns():
    pandas_subset = DataSetsManager(config_name=config_name).get_subset("first")
    polars_subset = polars_manager().get_subset("first")

    assert isinstance(polars_subset.X_train, pd.DataFrame)
    assert polars_subset.X_train.shape == pandas_subset.X_train.shape
    assert polars_subset.X_test.shape == pandas_subset.X_test.shape
    assert sorted(polars_subset.features_numerical) == sorted(pandas_subset.features_numerical)


def test_polars_fit_models_with_target_slices():
    manager = polars_manager()
    subset = manager.get_subset("first")

    results = manager.fit_models()

    assert subset.X is not None
    assert len(subset.X) == len(subset.X_train) + len(subset.X_test)
    slices = manager.data_config.data.targetslices
    assert slices and len(results["first"]["train"]) == slices[0]["slices"] + 1
