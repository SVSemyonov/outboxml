import abc
from abc import ABC
from typing import List, Dict

from outboxml.ensemble import EnsembleResult


class BaseValidator(ABC):

    @abc.abstractmethod
    def validate(self):
        raise NotImplementedError("Method validate() is not implemented")


class GroupValidator(BaseValidator):

    def __init__(self, group: List[Dict]):
        self.group = group

    @staticmethod
    def validate_ensemble(ensemble: EnsembleResult):
        models = ensemble.models  # List[tuple[condition: str, group_name: str,  model: ModelsResult]]
        unique_group_names = set()
        for condition, group_name, model in models:
            if group_name in unique_group_names:
                raise NotImplementedError(f"Not unique group names in ensemble: {group_name}")
            else:
                unique_group_names.add(group_name)

    def validate(self):

        if not (isinstance(self.group, list) and len(self.group) > 0):
            raise NotImplementedError(f"Invalid group type")

        for ensemble in self.group:

            # Validate Ensemble
            if isinstance(ensemble, EnsembleResult):
                self.validate_ensemble(ensemble)

            elif isinstance(ensemble, dict):
                pass

            else:
                raise NotImplementedError(f"Invalid model type in group")
