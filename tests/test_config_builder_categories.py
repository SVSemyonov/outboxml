import pandas as pd

from conftest import TEST_DATA_PATH
from outboxml.automl_utils import build_default_all_models_config
from outboxml.core.config_builders import FeatureBuilder
from outboxml.datasets_manager import DataSetsManager
from outboxml.extractors import SimpleExtractor


def test_replace_keys_match_prepared_values():
    feature = FeatureBuilder(name="SEX", type="categorical",
                             feature_values=pd.Series(["male", "female", None, 1.0])).build()

    assert set(feature.replace) == {"MALE", "FEMALE", "1"}


def test_default_config_keeps_lower_case_categories():
    data = pd.read_csv(TEST_DATA_PATH / "titanic.csv")[["SURVIVED", "SEX", "AGE"]]
    data["SEX"] = data["SEX"].str.lower()
    config = build_default_all_models_config(data=data, group_name="g", column_target="SURVIVED",
                                             model_params={"wrapper": "catboost", "name": "m", "objective": "binary"})

    subset = DataSetsManager(config_name=config.model_dump(mode="json"),
                             extractor=SimpleExtractor(data=data)).get_subset("m")

    assert set(subset.X_train["SEX"].unique()) == {"MALE", "FEMALE"}
