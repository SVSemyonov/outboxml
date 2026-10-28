import asyncio
import pickle
import shutil
from types import SimpleNamespace

import mlflow
import pandas as pd
import pytest

from conftest import TEST_DATA_PATH
from outboxml import config
from outboxml.core.errors import EnsembleError
from outboxml.core.predict import ensemble_predict
from outboxml.core.validators import GroupValidator
from outboxml.ensemble import Ensemble, EnsembleResult
from outboxml import main_predict as main_predict_module
from outboxml.main_predict import main_predict

CONDITIONS = [("SEX == 'male'", "group_male"), ("SEX == 'female'", "group_female")]


@pytest.fixture
def ensemble_config(tmp_path, prod_pickle_name):
    """Two copies of the production group, one per condition of the ensemble."""
    for _, group_name in CONDITIONS:
        shutil.copy(config.prod_models_path / f"{prod_pickle_name}.pickle", tmp_path / f"{group_name}.pickle")
    return SimpleNamespace(prod_models_path=tmp_path, results_path=tmp_path,
                           mlflow_tracking_uri=config.mlflow_tracking_uri, mlflow_experiment="ensemble_test")


@pytest.fixture
def ensemble(ensemble_config) -> Ensemble:
    ensemble = Ensemble(config=ensemble_config)
    ensemble.make_ensemble(ensemble_name="titanic_by_sex", models_names=["first", "second"], groups=CONDITIONS)
    return ensemble


@pytest.fixture
def titanic() -> pd.DataFrame:
    return pd.read_csv(TEST_DATA_PATH / "titanic.csv")


def test_save_ensemble_locally_and_to_mlflow(ensemble, ensemble_config):
    ensemble.save_ensemble(to_mlflow=True)

    saved = list(ensemble_config.results_path.glob("titanic_by_sex_*.pickle"))
    assert len(saved) == 1
    with open(saved[0], "rb") as f:
        result = pickle.load(f)
    GroupValidator(result).validate()
    assert [r.model_name for r in result] == ["first", "second"]
    mlflow.set_tracking_uri(ensemble_config.mlflow_tracking_uri)
    runs = mlflow.search_runs(experiment_names=["ensemble_test"])
    assert runs["tags.mlflow.runName"].tolist() == [saved[0].stem]


def test_save_ensemble_before_make_raises(ensemble_config):
    with pytest.raises(EnsembleError):
        Ensemble(config=ensemble_config).save_ensemble()


def test_ensemble_predict_dataframe_combines_conditions(ensemble, titanic):
    data = titanic.head(20)
    first = ensemble._result_pickle[0]

    result = asyncio.run(ensemble_predict("titanic_by_sex", first, data, log=False))

    predictions = pd.Series(result["result"]["first"])
    assert len(predictions) == len(data)
    assert predictions.index.tolist() == data.index.tolist()


def test_ensemble_predict_single_row_uses_matching_condition(ensemble, titanic):
    row = titanic.iloc[[0]].astype(object).where(titanic.iloc[[0]].notna(), None).to_dict(orient="records")[0]

    result = asyncio.run(ensemble_predict("titanic_by_sex", ensemble._result_pickle[0], row, log=False))

    assert set(result["result"]) == {"first"}


def test_ensemble_predict_without_matching_condition_raises(ensemble):
    with pytest.raises(NotImplementedError):
        asyncio.run(ensemble_predict("titanic_by_sex", ensemble._result_pickle[0], {"SEX": "unknown"}, log=False))


def test_ensemble_predict_rejects_invalid_features_type(ensemble):
    with pytest.raises(TypeError):
        asyncio.run(ensemble_predict("titanic_by_sex", ensemble._result_pickle[0], "SEX=male", log=False))


def test_group_validator_rejects_invalid_groups():
    with pytest.raises(NotImplementedError):
        GroupValidator([]).validate()
    with pytest.raises(NotImplementedError):
        GroupValidator(["not a model"]).validate()


def test_main_predict_with_ensemble_pickle(ensemble, ensemble_config, titanic, monkeypatch):
    ensemble.save_ensemble()
    saved = next(ensemble_config.results_path.glob("titanic_by_sex_*.pickle"))
    monkeypatch.setattr(config, "prod_models_path", ensemble_config.results_path)
    monkeypatch.setattr(main_predict_module, "all_groups", {})  # module-level cache of loaded groups

    result = asyncio.run(main_predict(config=config, group_name="titanic_by_sex", features_values=titanic.head(10)))

    assert result["main_response"]["usage_model"] == saved.stem
    assert set(result["main_response"]["result"]) == {"first", "second"}


@pytest.mark.parametrize("condition, expected", [
    ("SEX == 'male'", True),
    ("SEX=='male'", True),
    ("SEX == 'male' and AGE > 30", False),
    ("SEX == 'male' and AGE <= 30", True),
    ("PCLASS in [1, 2]", False),
    ("MISSING_FEATURE > 1", False),
])
def test_one_row_condition_matching(condition, expected):
    from outboxml.core.predict import _row_matches

    row = pd.DataFrame([{"SEX": "male", "AGE": 22.0, "PCLASS": 3}])

    assert _row_matches(row, condition) is expected


def test_ensemble_predict_single_row_with_compound_conditions(ensemble_config, titanic):
    ensemble = Ensemble(config=ensemble_config)
    ensemble.make_ensemble(ensemble_name="titanic_by_sex_age", models_names=["first"],
                           groups=[("SEX=='male' and AGE > 30", "group_male"),
                                   ("not (SEX=='male' and AGE > 30)", "group_female")])
    row = {"SEX": "male", "AGE": 22.0, "SIBSP": 1}

    result = asyncio.run(ensemble_predict("titanic_by_sex_age", ensemble._result_pickle[0], row, log=False))

    assert set(result["result"]) == {"first"}


def test_one_group_under_several_conditions(ensemble_config, titanic):
    ensemble = Ensemble(config=ensemble_config)
    ensemble.make_ensemble(ensemble_name="same_group", models_names=["first"],
                           groups=[("SEX == 'male'", "group_male"), ("SEX == 'female'", "group_male")])
    result_pickle = ensemble._result_pickle
    GroupValidator(result_pickle).validate()
    data = titanic.head(30)

    result = asyncio.run(ensemble_predict("same_group", result_pickle[0], data, log=False))

    predictions = pd.Series(result["result"]["first"])
    assert sorted(predictions.index) == sorted(data.index)


def test_repeated_condition_and_group_is_rejected(ensemble_config):
    with pytest.raises(EnsembleError, match="repeated"):
        Ensemble(config=ensemble_config).make_ensemble(
            ensemble_name="dup", models_names=["first"],
            groups=[("SEX == 'male'", "group_male"), ("SEX == 'male'", "group_male")])
    duplicated = EnsembleResult(model_name="first", models=[("A > 1", "g", {}), ("A > 1", "g", {})])
    with pytest.raises(NotImplementedError):
        GroupValidator([duplicated]).validate()
