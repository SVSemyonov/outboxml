from datetime import datetime
from loguru import logger
import mlflow
import os
import pickle
from pydantic import ValidationError
from typing import List, Dict, Tuple, Optional, Any

from outboxml.core.pydantic_models import ModelConfig
from outboxml.core.errors import EnsembleError
from outboxml.core.utils import ResultPickle
from outboxml import config as lib_config

class EnsembleResult:
    """Class for storing one part of the models' ensemble.

    The final structure of the pickle file is ``List[EnsembleResult]``.
    Each result contains a model name and a list of conditional models.

    :var model_name: The name of the model, it should be in all elements
        of ``models``.
    :var models: List of tuples ``(condition: str, group_name: str, model: Any)``.
        Condition should be a valid string for ``pandas.query()``.
        Group_name is a name of models' group that should be applied for
        the given condition. Model is a fitted model object.

    Example::

        result = EnsembleResult(
            model_name="my_model",
            models=[
                ("region == 'A'", "model_group_a", fitted_model_a),
                ("region == 'B'", "model_group_b", fitted_model_b)
            ]
        )
    """

    def __init__(self, model_name: str, models: List[Tuple[str, str, Any]]):
        """Initialize EnsembleResult instance.

        :param model_name: The name of the model, it should be in all elements
            of ``models``.
        :type model_name: str
        :param models: List of tuples ``(condition: str, group_name: str, model: Any)``.
            Condition should be a valid string for ``pandas.query()``.
            Group_name is a name of models' group that should be applied for
            the given condition. Model is a fitted model object.
        :type models: List[Tuple[str, str, Any]]
        """
        self.model_name: str = model_name
        self.models = models


class Ensemble:
    """Class for creating and saving models' ensemble.

    This class allows creating ensembles of model groups that can be applied
    conditionally based on data characteristics. The ensemble is saved as
    a pickle file for use in production.

    :var config: Configuration object that should contain ``prod_models_path``,
        ``results_path`` and optionally ``mlflow_tracking_uri``, ``mlflow_experiment``
        if MLflow is used.
    :var _ensemble_name: Name of the ensemble.
    :var _models_names: List of model names in the ensemble.
    :var _all_groups: Dictionary of all model groups.
    :var _is_maked: Flag indicating if ensemble has been created.
    :var _result_pickle: List of EnsembleResult objects.

    .. rubric:: Examples

    .. code-block:: python

        class external_config:
            prod_models_path = "example_prod_path"
            results_path = "example_results_path"
            mlflow_tracking_uri = "https://mlflow.company.my"
            mlflow_experiment = "example_experiment"

        ens = Ensemble(config=external_config)

        ens.make_ensemble(
            ensemble_name="example_ensemble",
            models_names=[
                "example_model_a",
                "example_model_b",
            ],
            groups=[
                ("rule_column == 'rule_a'", "model_a.pickle"),
                ("rule_column == 'rule_b'", "model_b.pickle"),
            ]
        )
    """

    def __init__(self, config=None):
        """Initialize Ensemble instance.

        :param config: Configuration object that should contain ``prod_models_path``,
            ``results_path`` and optionally ``mlflow_tracking_uri``, ``mlflow_experiment``
            if MLflow is used. Defaults to None.
        :type config: Any, optional
        """
        self.config = config
        if config is None:
            self.config = lib_config
        self._ensemble_name: Optional[str] = None
        self._models_names: Optional[List[str]] = None
        self._all_groups: Optional[Dict] = None
        self._is_maked: bool = False
        self._result_pickle: Optional[List] = None

    def make_ensemble(self, ensemble_name: str, models_names: List[str], groups: List[Tuple[str, str]]) -> None:
        """Make an ensemble of models' groups.

        Creates an ensemble by loading model groups and associating them
        with conditional rules. The ensemble is saved as a pickle file.

        :param ensemble_name: Ensemble name.
        :type ensemble_name: str
        :param models_names: Model names that should be present in all groups.
        :type models_names: List[str]
        :param groups: List of tuples ``(condition: str, group_name: str)``.
            Condition should be a valid string for ``pandas.query()``.
            Group_name is a name of models' group that should be applied
            for the given condition.
        :type groups: List[Tuple[str, str]]
        :return: None
        :rtype: None

        :raises EnsembleError: If ensemble is already made, or if parameters
            are invalid.

        Example::

            ens.make_ensemble(
                ensemble_name="my_ensemble",
                models_names=["model_1", "model_2"],
                groups=[
                    ("region == 'A'", "group_a.pickle"),
                    ("region == 'B'", "group_b.pickle")
                ]
            )
        """

        logger.info(f"making ensemble {ensemble_name} ...")

        if self._is_maked:
            raise EnsembleError("ensemble is already maked")

        if not isinstance(ensemble_name, str) or ensemble_name == "":
            raise EnsembleError("invalid `ensemble_name`")
        self._ensemble_name = ensemble_name

        if not isinstance(models_names, list) or len(models_names) == 0:
            raise EnsembleError("invalid `models_names`")
        self._models_names = models_names

        unique_models_names = set()
        for name in self._models_names:
            if not isinstance(name, str):
                raise EnsembleError("invalid `models_names`")
            if name in unique_models_names:
                raise EnsembleError("not unique models names in ensemble")
            else:
                unique_models_names.add(name)

        if not isinstance(groups, list) or len(groups) == 0:
            raise EnsembleError("invalid groups")

        self._all_groups = {}
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

            if group_name not in self._all_groups:
                self._load_group(group_name)
                logger.info(f"loaded group `{group_name}`")

        self._result_pickle = []
        for name in self._models_names:
            self._result_pickle.append(
                EnsembleResult(
                    model_name=name,
                    models=[
                        (condition, group_name, model)
                        for condition, group_name in groups
                        for model in self._all_groups[group_name]
                        if ModelConfig.model_validate(model["model_config"]).name == name
                    ]
                )
            )

        self._is_maked = True
        logger.info(f"ensemble {ensemble_name} is maked")

    def _load_group(self, group_name):
        try:
            with open(os.path.join(self.config.prod_models_path, f"{group_name}.pickle"), "rb") as f:
                group = pickle.load(f)
        except FileNotFoundError:
            raise EnsembleError(f"file `{group_name}.pickle` is not found in {self.config.prod_models_path}")
        self._validate_group(group, group_name)
        self._all_groups.update({group_name: group})

    def _validate_group(self, group: List, group_name: str) -> None:

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

        for name in self._models_names:
            if name not in unique_models_names:
                raise EnsembleError(f"no `{name}` model in group `{group_name}`")

    def save_ensemble(self, to_mlflow: bool = False) -> None:
        """
        Saves maked ensemble's pickle to a local file and optionally to MLFlow.

        :param to_mlflow: Whether to save the ensemble to MLFlow.

        :return: None
        """

        if not self._is_maked:
            raise EnsembleError("ensemble is not maked")

        # Save pickle file locally
        now_time = datetime.now()
        result_pickle_name = ResultPickle().generate_name(self._ensemble_name, now_time)
        with open(os.path.join(self.config.results_path, result_pickle_name), "wb") as f:
            pickle.dump(self._result_pickle, f)
        logger.info(f"saved ensemble to `{result_pickle_name}`")

        # Save pickle file to MLFlow
        if to_mlflow:
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            mlflow.set_experiment(self.config.mlflow_experiment)
            with mlflow.start_run(run_name=result_pickle_name.replace(".pickle", "")):
                mlflow.log_artifact(os.path.join(self.config.results_path, result_pickle_name))
        logger.info(f"saved ensemble to MLFlow")
