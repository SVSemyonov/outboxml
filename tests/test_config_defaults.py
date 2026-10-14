from outboxml.core.enums import SeparationParams
from outboxml.core.pydantic_models import (AllModelsConfig, DataModelConfig, FeatureModelConfig,
                                           RelativeFeatureModelConfig)


def test_minimal_all_models_config():
    config = AllModelsConfig.model_validate({
        "group_name": "titanic",
        "data_config": {"source": "csv", "local_name_source": "titanic.csv"},
        "models_configs": [{"name": "first", "features": [{"name": "SEX"},
                                                          {"name": "AGE", "replace": {"_TYPE_": "_NUM_"}}]}],
    })

    assert (config.project, config.version) == ("default", "1")
    separation = config.data_config.separation
    assert (separation.kind, separation.random_state, separation.test_train_proportion) == \
        (SeparationParams.rand, 42, 0.2)
    assert config.data_config.data.targetslices == []
    sex, age = config.models_configs[0].features
    assert (sex.default, sex.replace) == ("_NAN_", {})
    assert age.default == "_MEDIAN_"


def test_explicit_values_are_kept():
    feature = FeatureModelConfig(name="AGE", default=0, replace={"_TYPE_": "_NUM_"})
    data_config = DataModelConfig(source="csv", separation={"kind": "random", "test_train_proportion": 0.3})

    assert feature.default == 0
    assert data_config.separation.test_train_proportion == 0.3


def test_relative_feature_default():
    assert RelativeFeatureModelConfig(name="R", numerator="A", denominator="B").default == "_NAN_"
