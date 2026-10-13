import optuna
import pytest
from optuna.samplers import CmaEsSampler, RandomSampler, TPESampler
from optuna.trial import FixedTrial

from conftest import models_config
from outboxml.datasets_manager import DataSetsManager
from outboxml.extractors import SimpleExtractor
from outboxml.hyperparameter_tuning import HPTuning

optuna.logging.set_verbosity(optuna.logging.WARNING)


def catboost_space(trial):
    return {"iterations": trial.suggest_int("iterations", 10, 12), "depth": trial.suggest_int("depth", 1, 3)}


def xgboost_space(trial):
    return {"iterations": trial.suggest_int("iterations", 10, 12), "max_depth": trial.suggest_int("max_depth", 1, 3),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3, log=True)}


def tuning(data, sampler="TPE", pass_objective=True, **config) -> HPTuning:
    """HPTuning as AutoMLManager.hp_tuning creates it: with the objective of the model config."""
    ds_manager = DataSetsManager(config_name=models_config(**config), extractor=SimpleExtractor(data=data.copy()))
    ds_manager.get_subset("model")
    objective = {"objective": config["objective"]} if pass_objective else {}
    return HPTuning(data_preprocessor=ds_manager._data_preprocessor, sampler=sampler, folds_num_for_cv=2, **objective)


@pytest.mark.parametrize("config, space, sampler", [
    (dict(wrapper="catboost", objective="gamma", target="FARE_TARGET", exposure="EXPOSURE",
          column_weight="ROW_WEIGHT"), catboost_space, "RandomSampler"),
    (dict(wrapper="catboost", objective="poisson", target="CLAIMS", exposure="EXPOSURE"), catboost_space, "TPE"),
    (dict(wrapper="catboost", objective="binary", target="SURVIVED"), catboost_space, "RandomSampler"),
    (dict(wrapper="catboost", objective="RMSE", target="FARE_TARGET"), catboost_space, "unknown_sampler"),
    (dict(wrapper="xgboost", objective="poisson", target="CLAIMS"), xgboost_space, "TPE"),
    (dict(wrapper="xgboost", objective="gamma", target="FARE_TARGET"), xgboost_space, "TPE"),
    (dict(wrapper="xgboost", objective="binary", target="SURVIVED"), xgboost_space, "TPE"),
])
def test_best_params_for_wrapper_and_objective(titanic_synthetic, config, space, sampler):
    params = tuning(titanic_synthetic, sampler=sampler, **config).best_params(
        model_name="model", trials=2, n_jobs=1, direction="maximize", parameters_for_optuna_func=space)

    assert set(params) == set(space(FixedTrial({"iterations": 10, "depth": 1, "max_depth": 1,
                                                "learning_rate": 0.1})))


@pytest.mark.xfail(strict=True, reason="model class is chosen by HPTuning(objective=...) (default 'RMSE'), the loss "
                                       "by the model config objective: CatBoostRegressor with Logloss")
def test_best_params_uses_model_config_objective(titanic_synthetic):
    hp_tuning = tuning(titanic_synthetic, pass_objective=False, wrapper="catboost", objective="binary",
                       target="SURVIVED")

    params = hp_tuning.best_params(model_name="model", trials=1, n_jobs=1, parameters_for_optuna_func=catboost_space)

    assert set(params) == {"iterations", "depth"}


@pytest.mark.parametrize("sampler, sampler_cls", [("TPE", TPESampler), ("RandomSampler", RandomSampler),
                                                  ("CmaEsSampler", CmaEsSampler), ("unknown", type(None))])
def test_sampler_selection(titanic_synthetic, sampler, sampler_cls):
    hp_tuning = tuning(titanic_synthetic, sampler=sampler, wrapper="catboost", objective="poisson", target="CLAIMS")

    hp_tuning._sample_parameters()

    assert isinstance(hp_tuning._sampler, sampler_cls)


@pytest.mark.parametrize("work_type", ["CPU", "GPU"])
def test_default_catboost_search_space(work_type):
    trial = FixedTrial({"iterations": 100, "depth": 3, "learning_rate": 0.05, "l2_leaf_reg": 1.0, "subsample": 0.5,
                        "colsample_bylevel": 0.5, "min_data_in_leaf": 11, "bootstrap_type": "Bayesian"})

    params = HPTuning.parameters_for_optuna(trial, work_type=work_type)

    if work_type == "GPU":
        assert "colsample_bylevel" not in params
        assert params["rsm"] == 1 and params["bootstrap_type"] == "Bayesian"
    else:
        assert params["colsample_bylevel"] == 0.5
