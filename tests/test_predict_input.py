import pandas as pd
import pytest

from conftest import TEST_DATA_PATH, models_config
from outboxml.core.predict import one_model_predict
from outboxml.datasets_manager import DataSetsManager
from outboxml.extractors import SimpleExtractor


@pytest.fixture(scope="module")
def model(titanic_synthetic) -> dict:
    config = models_config(wrapper="catboost", objective="binary", target="SURVIVED")
    ds_manager = DataSetsManager(config_name=config, extractor=SimpleExtractor(data=titanic_synthetic.copy()))
    ds_manager.fit_models()
    return ds_manager.get_result()["model"].dict_for_prod_export()


def test_prediction_does_not_modify_input_dataframe(model):
    data = pd.read_csv(TEST_DATA_PATH / "titanic.csv").head(10)
    original = data.copy()

    one_model_predict("titanic", model, data, log=False)

    pd.testing.assert_frame_equal(data, original)


def test_prediction_does_not_modify_input_row(model):
    row = {"SEX": "MALE", "AGE": None, "PCLASS": 3}
    original = dict(row)

    one_model_predict("titanic", model, row, log=False)

    assert row == original
