import plotly.graph_objects as go
import pytest

from conftest import models_config
from outboxml.datasets_manager import DataSetsManager
from outboxml.extractors import SimpleExtractor
from outboxml.feature_importance import FeatureImportance


def fitted(data, **config):
    ds_manager = DataSetsManager(config_name=models_config(**config), extractor=SimpleExtractor(data=data.copy()))
    ds_manager.fit_models()
    result = ds_manager.get_result()["model"]
    return FeatureImportance(model_name="model", model=result.model, data_subset=result.data_subset)


@pytest.mark.parametrize("config", [
    dict(wrapper="catboost", objective="poisson", target="CLAIMS", exposure="EXPOSURE"),
    dict(wrapper="xgboost", objective="gamma", target="FARE_TARGET"),
])
def test_importance_with_directions_and_plot(titanic_synthetic, config):
    importance = fitted(titanic_synthetic, **config)

    result = importance.calculate_importance(use_test=True, calculate_directions=True, zero_corr_threshold=0.01)

    assert {item["FEATURE"] for item in result} >= {"AGE", "PCLASS"}
    assert all(item["SHAP"] >= 0 for item in result)
    figure = importance.plot(show=False)
    assert isinstance(figure, go.Figure)


def test_importance_on_train_without_directions(titanic_synthetic):
    importance = fitted(titanic_synthetic, wrapper="catboost", objective="gamma", target="FARE_TARGET")

    result = importance.calculate_importance(use_test=False, calculate_directions=False)

    assert result and all("SIGN" not in item for item in result)
    assert isinstance(importance.plot(show=False), go.Figure)


def test_importance_not_supported_for_catboost_over_glm(titanic_synthetic):
    importance = fitted(titanic_synthetic, wrapper="catboost_over_glm", objective="gamma", target="FARE_TARGET")

    assert importance.calculate_importance() == []
    assert importance.plot(show=False) is None


def test_plot_before_calculation_returns_none(titanic_synthetic):
    importance = fitted(titanic_synthetic, wrapper="catboost", objective="gamma", target="FARE_TARGET")

    assert importance.plot(show=False) is None
