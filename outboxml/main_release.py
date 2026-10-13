import fnmatch
import os
import shutil
import tempfile
from pathlib import Path

import mlflow
from loguru import logger
from mlflow.entities import ViewType



class Release:

    def __init__(self, config):
        """
        :param config: `config` should contain:
         `results_path`, Path: directory with models' pickles, is used if source is local (`from_mlflow = False`);
         `prod_path`, Path: git directory of models' register;
         `prod_models_folder`, str: subdirectory in `prod_path` for models' pickles;
         `prod_git`, str: git url;
         `mlflow_tracking_uri`, str: is used if source is mlflow (`from_mlflow = True`);
         `mlflow_experiment`, str: is used if source is mlflow (`from_mlflow = True`);
         `email_smtp_server`, str;
         `email_port`, int;
         `email_sender`, str;
         `email_login`, str;
         `email_pass`, str;
         `email_receivers`, List[str].
        """
        self.config = config

    def _source_path(self) -> Path:
        source_path = Path(self.config.prod_path) / self.config.prod_models_folder
        source_path.mkdir(parents=True, exist_ok=True)
        return source_path

    def load_model_to_source(self, group_name: str) -> None:
        source_path = self._source_path()
        results_path = Path(self.config.results_path)
        if os.path.isfile(source_path / f"{group_name}.pickle"):
            raise FileExistsError(f"Already in {source_path}: {group_name}.pickle")

        if not os.path.isfile(results_path / f"{group_name}.pickle"):
            raise FileNotFoundError(f"Not found in {results_path}: {group_name}.pickle")

        shutil.copyfile(results_path / f"{group_name}.pickle", source_path / f"{group_name}.pickle")

    def load_model_to_source_from_mlflow(self, group_name: str) -> None:
        source_path = self._source_path()
        if os.path.isfile(source_path / f"{group_name}.pickle"):
            raise FileExistsError(f"Already in {source_path}: {group_name}.pickle")

        os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment)

        mlflow_runs = mlflow.search_runs(filter_string=f"tags.mlflow.runName = '{group_name}'")
        if mlflow_runs.empty:
            raise FileNotFoundError(f"Not found in mlflow: {group_name}")

        artifact_uri = mlflow_runs.sort_values("start_time", ascending=False).iloc[0].artifact_uri
        with tempfile.TemporaryDirectory() as download_dir:
            pickle_path = mlflow.artifacts.download_artifacts(f"{artifact_uri}/{group_name}.pickle",
                                                              dst_path=download_dir)
            shutil.copyfile(pickle_path, source_path / f"{group_name}.pickle")


class MLFLowRelease(Release):
    def __init__(self, config):
        super().__init__(config)

    def load_model_to_source_from_mlflow(self, group_name: str = None) -> None:
        """
        Download artifacts from MLFlow.

        Args:
            group_name: str. Name of model group. If None, the latest deployed run of the experiment is used.

        Returns:
            None
        """
        curr_experiment = self.config.mlflow_experiment
        source_path = self._source_path()

        os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)

        # Find experiment
        experiment = mlflow.search_experiments(
            view_type=ViewType.ACTIVE_ONLY,  # don't take into account DELETED
            filter_string=f"name = '{curr_experiment}'"
        )
        # Check experiment exists
        if experiment:
            mlflow.set_experiment(curr_experiment)

            # Find runs
            filter_runs = "tags.Deployment_decision = 'True'"
            if group_name is not None:
                filter_runs += f" AND tags.mlflow.runName LIKE '{group_name}_%'"
            mlflow_runs = mlflow.search_runs(
                filter_string=filter_runs,
                run_view_type=ViewType.ACTIVE_ONLY,  # don't take into account DELETED
            )

            # Check runs exist
            if not mlflow_runs.empty:
                artifact_uri = mlflow_runs.sort_values("start_time", ascending=False).iloc[0].artifact_uri
                is_pickle = False
                with tempfile.TemporaryDirectory() as download_dir:
                    run_dir = mlflow.artifacts.download_artifacts(artifact_uri, dst_path=download_dir)
                    for file in os.listdir(run_dir):
                        file_path = os.path.join(run_dir, file)
                        if not os.path.isfile(file_path):
                            continue
                        shutil.copy(file_path, source_path)
                        if fnmatch.fnmatch(file, '*.pickle'):
                            is_pickle = True
                if not is_pickle:  # if model not exist
                    logger.info(f"Not found model in run '{group_name}'")
            else:  # If run not exist
                logger.info(f"Deployed run for group '{group_name}' "
                            f"doesn't exist in mlflow in experiment '{curr_experiment}'")
        else:
            logger.info(f"Experiment '{curr_experiment}' doesn't exist in mlflow")
