import os
import shutil
import mlflow



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

    def load_model_to_source(self, group_name: str) -> None:
        source_path = self.config.prod_path / self.config.prod_models_folder
        if os.path.isfile(source_path / f"{group_name}.pickle"):
            raise FileExistsError(f"Already in {source_path}: {group_name}.pickle")

        if not os.path.isfile(self.config.results_path / f"{group_name}.pickle"):
            raise FileNotFoundError(f"Not found in {self.config.results_path}: {group_name}.pickle")

        shutil.copyfile(self.config.results_path / f"{group_name}.pickle", source_path / f"{group_name}.pickle")

    def load_model_to_source_from_mlflow(self, group_name: str) -> None:
        source_path = self.config.prod_path / self.config.prod_models_folder
        if os.path.isfile(source_path / f"{group_name}.pickle"):
            raise FileExistsError(f"Already in {source_path}: {group_name}.pickle")

        os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment)

        mlflow_runs = mlflow.search_runs(filter_string=f"tags.mlflow.runName = '{group_name}'")
        if mlflow_runs.empty:
            raise FileNotFoundError(f"Not found in mlflow: {group_name}")

        artifact_uri = mlflow_runs.sort_values("start_time", ascending=False).iloc[0].artifact_uri
        mlflow.artifacts.download_artifacts(f"{artifact_uri}/{group_name}.pickle", dst_path="/")

        shutil.copyfile(f"./artifacts/{group_name}.pickle", source_path / f"{group_name}.pickle")


class MLFLowRelease(Release):
    def __init__(self, config):
        super().__init__(config)

    def load_model_to_source_from_mlflow(self, group_name=None) -> None:
        source_path = self.config.prod_path / self.config.prod_models_folder

        os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment)

        mlflow_runs = mlflow.search_runs(filter_string=f"tags.Deployment_decision = 'True'")
        if mlflow_runs.empty:
            raise FileNotFoundError(f"Not found in mlflow: {group_name}")

        artifact_uri = mlflow_runs.sort_values("start_time", ascending=False).iloc[0].artifact_uri
        dir = mlflow.artifacts.download_artifacts(artifact_uri, dst_path=source_path)
        for file in os.listdir(dir):
            shutil.copy(dir + '/' + file, source_path)

