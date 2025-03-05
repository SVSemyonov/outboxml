from datetime import datetime
import mlflow
import os
import pickle
from pydantic import ValidationError
from typing import List, Dict, Tuple, Optional, Any

from outboxml.core.pydantic_models import ModelConfig
from outboxml.core.errors import EnsembleError
from outboxml.core.utils import ResultPickle


class EnsembleResult:

    def __init__(self, model_name: str, models: List[Tuple[str, str, Any]]):
        self.model_name: str = model_name
        self.models = models


class Ensemble:

    def __init__(self, config=None):
        """
        :param config: `config` should contain `prod_models_path`, `results_path` and if mlflow is used `mlflow_tracking_uri`, `mlflow_experiment`
        """
        self.config = config
        self.ensemble_name: Optional[str] = None
        self.models_names: Optional[List[str]] = None
        self.all_groups: Dict = {}
        self.is_maked: bool = False
        self.result_pickle: Optional[List] = None

    def make_ensemble(self, ensemble_name: str, models_names: List[str], groups: List[Tuple[str, str]]) -> None:
        """
        Make an ensemble of models groups.
        :param ensemble_name: Ensemble name.
        :param models_names: Models names, e.g. [`frequency`, `severity`].
        :param groups: List[(condition, group_name)], conditions should be valid for pandas.query().
        :return: None
        """

        if self.is_maked:
            raise EnsembleError("ensemble is already maked")

        if not isinstance(ensemble_name, str) or ensemble_name == "":
            raise EnsembleError("invalid `ensemble_name`")
        self.ensemble_name = ensemble_name

        if not isinstance(models_names, list) or len(models_names) == 0:
            raise EnsembleError("invalid `models_names`")
        self.models_names = models_names

        unique_models_names = set()
        for name in self.models_names:
            if not isinstance(name, str):
                raise EnsembleError("invalid `models_names`")
            if name in unique_models_names:
                raise EnsembleError("not unique models names in ensemble")
            else:
                unique_models_names.add(name)

        if not isinstance(groups, list) or len(groups) == 0:
            raise EnsembleError("invalid groups")

        unique_group_names = set()
        for group in groups:
            if not isinstance(group, tuple) or len(group) != 2:
                raise EnsembleError("invalid groups")
            condition, group_name = group
            if not isinstance(condition, str):
                raise EnsembleError("invalid condition")
            if not isinstance(group_name, str):
                raise EnsembleError("invalid group_name")
            group_name = group_name.replace(".pickle", "")

            if group_name in unique_group_names:
                raise EnsembleError("not unique group names in `groups`")
            else:
                unique_group_names.add(group_name)

            if group_name not in self.all_groups:
                self.load_group(group_name)

        self.result_pickle = []
        for name in self.models_names:
            self.result_pickle.append(
                EnsembleResult(
                    model_name=name,
                    models=[
                        (condition, group_name, model)
                        for condition, group_name in groups
                        for model in self.all_groups[group_name]
                        if ModelConfig.model_validate(model["model_config"]).name == name
                    ]
                )
            )

        self.is_maked = True

    def load_group(self, group_name):
        try:
            with open(os.path.join(self.config.prod_models_path, f"{group_name}.pickle"), "rb") as f:
                group = pickle.load(f)
        except FileNotFoundError:
            raise EnsembleError(f"file `{group_name}.pickle` is not found in {self.config.prod_models_path}")
        self.validate_group(group, group_name)
        self.all_groups.update({group_name: group})

    def validate_group(self, group: List, group_name: str) -> None:

        if not isinstance(group, list):
            raise EnsembleError(f"invalid group `{group_name}`")

        models_names = []
        for model in group:
            if not isinstance(model, dict):
                raise EnsembleError(f"invalid model in group `{group_name}`")
            try:
                model_config = ModelConfig.model_validate(model["model_config"])
            except ValidationError as exc:
                raise EnsembleError(exc)
            models_names.append(model_config.name)

        unique_models_names = set()
        for name in models_names:
            if name in unique_models_names:
                raise EnsembleError(f"not unique models names in group `{group_name}`")
            else:
                unique_models_names.add(name)

        for name in self.models_names:
            if name not in unique_models_names:
                raise EnsembleError(f"no `{name}` model in group `{group_name}`")

    def save_ensemble(self, to_mlflow: bool = False) -> None:

        if not self.is_maked:
            raise EnsembleError("ensemble is not maked")

        # Сохранение пикла для сервиса локально
        now_time = datetime.utcnow()
        result_pickle_name = ResultPickle().generate_name(self.ensemble_name, now_time)
        with open(os.path.join(self.config.results_path, result_pickle_name), "wb") as f:
            pickle.dump(self.result_pickle, f)

        # Сохранение пикла для сервиса в MLFlow
        if to_mlflow:
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            mlflow.set_experiment(self.config.mlflow_experiment)
            with mlflow.start_run(run_name=result_pickle_name.replace(".pickle", "")):
                mlflow.log_artifact(os.path.join(self.config.results_path, result_pickle_name))
