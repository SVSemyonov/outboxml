from datetime import datetime, timedelta
from types import SimpleNamespace

import mlflow
import pytest

from outboxml.automl_manager import AutoMLResult, MLFlowWrapper
from outboxml.main_release import MLFLowRelease, Release

EXPERIMENT = "release_test"


@pytest.fixture
def release_config(tmp_path):
    tracking_uri = mlflow.get_tracking_uri()
    results_path = tmp_path / "results"
    results_path.mkdir()
    yield SimpleNamespace(mlflow_tracking_uri=(tmp_path / "mlruns").as_uri(),
                          mlflow_experiment=EXPERIMENT,
                          results_path=results_path,
                          prod_path=tmp_path / "prod",
                          prod_models_folder="models")
    if mlflow.active_run():
        mlflow.end_run()
    mlflow.set_tracking_uri(tracking_uri)


def log_automl_run(config, group_name: str, deployment: bool, start: datetime) -> str:
    """Log a run the way AutoMLManager does and return its result pickle name."""
    result = AutoMLResult(group_name=group_name)
    result.run_time["start"] = start
    result.result_pickle_name = f"{group_name}_{start.strftime('%Y_%m_%d_%H_%M_%S')}.pickle"
    result.deployment = deployment
    result.compare_business_metric = None
    for name in ("log.log", "automl_report.html", result.result_pickle_name):
        (config.results_path / name).write_text(name)
    MLFlowWrapper(experiment_name=config.mlflow_experiment, group_name=group_name,
                  results_path=str(config.results_path),
                  mlflow_tracking_uri=config.mlflow_tracking_uri).log_results(result)
    return result.result_pickle_name


def prod_pickles(config) -> list:
    return sorted(p.name for p in (config.prod_path / config.prod_models_folder).glob("*.pickle"))


def test_mlflow_release_downloads_latest_deployed_run_of_group(release_config):
    now = datetime.now()
    log_automl_run(release_config, "example_titanic", True, now - timedelta(hours=2))
    latest = log_automl_run(release_config, "example_titanic", True, now - timedelta(hours=1))
    log_automl_run(release_config, "example_titanic", False, now)
    log_automl_run(release_config, "other_group", True, now)

    MLFLowRelease(config=release_config).load_model_to_source_from_mlflow(group_name="example_titanic")

    assert prod_pickles(release_config) == [latest]


def test_mlflow_release_without_group_takes_latest_deployed_run(release_config):
    now = datetime.now()
    log_automl_run(release_config, "example_titanic", True, now - timedelta(hours=1))
    latest = log_automl_run(release_config, "other_group", True, now)

    MLFLowRelease(config=release_config).load_model_to_source_from_mlflow()

    assert prod_pickles(release_config) == [latest]


def test_mlflow_release_without_deployed_runs_copies_nothing(release_config):
    log_automl_run(release_config, "example_titanic", False, datetime.now())

    MLFLowRelease(config=release_config).load_model_to_source_from_mlflow(group_name="example_titanic")

    assert prod_pickles(release_config) == []


def test_release_from_mlflow_copies_named_pickle(release_config):
    mlflow.set_tracking_uri(release_config.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    pickle_file = release_config.results_path / "example_titanic.pickle"
    pickle_file.write_text("model")
    with mlflow.start_run(run_name="example_titanic"):
        mlflow.log_artifact(str(pickle_file))

    Release(config=release_config).load_model_to_source_from_mlflow(group_name="example_titanic")

    assert prod_pickles(release_config) == ["example_titanic.pickle"]


def test_release_copies_local_pickle(release_config):
    (release_config.results_path / "example_titanic.pickle").write_text("model")

    Release(config=release_config).load_model_to_source(group_name="example_titanic")

    assert prod_pickles(release_config) == ["example_titanic.pickle"]
    with pytest.raises(FileExistsError):
        Release(config=release_config).load_model_to_source(group_name="example_titanic")
