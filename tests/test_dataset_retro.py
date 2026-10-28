import pandas as pd

from outboxml.core.pydantic_models import ModelConfig
from outboxml.dataset_retro import RetroDataset


class Retro(RetroDataset):
    def load_retro_data(self):
        self.retro_data = pd.DataFrame({"A": [1.0], "B": [2.0], "TARGET": [0.0]})


def test_retro_config_drops_features_without_retro_columns():
    retro = Retro()
    retro.load_retro_data()
    model = ModelConfig.model_validate({
        "name": "m", "column_target": "TARGET",
        "features": [{"name": "A", "replace": {"_TYPE_": "_NUM_"}}, {"name": "C", "replace": {"_TYPE_": "_NUM_"}}],
        "relative_features": [{"name": "A_B", "numerator": "A", "denominator": "B"},
                              {"name": "A_C", "numerator": "A", "denominator": "C"}],
    })

    [retro_model] = retro.models_config_for_retro([model], target_columns_names=["TARGET"])

    assert [f.name for f in retro_model.features] == ["A"]
    assert [f.name for f in retro_model.relative_features] == ["A_B"]
