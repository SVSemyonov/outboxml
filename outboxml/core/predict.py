from copy import deepcopy
from itertools import chain
import pandas as pd
from typing import Dict, List, Union

from outboxml.core.prepared_datasets import PrepareDataset
from outboxml.data_subsets import DataPreprocessor
from outboxml.datasets_manager import DSManagerResult
from outboxml.ensemble import EnsembleResult
from outboxml.core.pydantic_models import ModelConfig, DataModelConfig, SeparationModelConfig
from outboxml.core.data_prepare import prepare_dataset


async def ensemble_predict(
        ensemble_name: str,
        ensemble: Union[Dict, EnsembleResult],
        features_values: Union[Dict, List[Dict], pd.DataFrame],
        log: bool = True,
        modify_dtypes: bool = True,
        raise_on_encoding_error: bool = True,
) -> Dict:

    as_dict: bool = False
    if isinstance(features_values, dict):
        as_dict = True
    elif isinstance(features_values, list) and len(features_values) == 1:
        features_values = features_values[0]
        as_dict = True
    elif isinstance(features_values, list):
        data = pd.DataFrame(features_values)
    elif isinstance(features_values, pd.DataFrame):
        data = features_values
    else:
        raise TypeError("Invalid features_values type")

    # ensemble
    if isinstance(ensemble, EnsembleResult):

        model_name = ensemble.model_name
        models = ensemble.models  # List[tuple[condition: str, group_name: str,  model: ModelsResult]]

        if as_dict:
            # data = pd.DataFrame([features_values])
            for condition, group_name, model in models:
                # FIXME
                condition = condition.split(" ")
                # if not data.query(condition).empty:
                if eval(f"""features_values.get("{condition[0]}") {condition[1]} {" ".join(condition[2:])}"""):
                    return one_model_predict(
                        group_name=ensemble_name,
                        model_result=model,
                        features_values=deepcopy(features_values),
                        log=log,
                        modify_dtypes=modify_dtypes,
                        raise_on_encoding_error=raise_on_encoding_error,
                    )
            raise NotImplementedError("Invalid ensemble: imprecise conditions")

        result_filtered = {}
        version_model_filtered = {}
        df_filtered = {}

        for condition, group_name, model in models:
            if not (data_filtered := data.query(condition)).empty:
                result = one_model_predict(
                    group_name=group_name,
                    model_result=model,
                    features_values=data_filtered,
                    log=log,
                    modify_dtypes=modify_dtypes,
                    raise_on_encoding_error=raise_on_encoding_error,
                )
                result_filtered[group_name] = pd.Series(result["result"][model_name])
                version_model_filtered[group_name] = pd.Series(result["version_model"][model_name])
                df_filtered[group_name] = pd.DataFrame.from_dict(result["df"][model_name])

        if len(result_filtered) > 1:
            result = pd.concat(
                [predictions for group_name, predictions in result_filtered.items()],
                axis=0,
                join="outer",
                ignore_index=False,
            ).sort_index()

            version_model = pd.concat(
                [versions for group_name, versions in version_model_filtered.items()],
                axis=0,
                join="outer",
                ignore_index=False,
            ).sort_index()

            df_result = pd.concat(
                [df for group_name, df in df_filtered.items()],
                axis=0,
                join="outer",
                ignore_index=False,
            ).sort_index()

        else:
            result = result_filtered[list(result_filtered.keys())[0]]
            version_model = version_model_filtered[list(version_model_filtered.keys())[0]]
            df_result = df_filtered[list(df_filtered.keys())[0]]

        if data.shape[0] != result.shape[0] or len(result.index) != len(result.index.drop_duplicates()):
            raise NotImplementedError("Invalid ensemble: imprecise conditions")

        return {
            "group_name": ensemble_name,
            "result": {model_name: result.to_dict()},
            "version_model": {model_name: version_model.to_dict()},
            "df": {model_name: df_result.to_dict()},
        }

    # not ensemble
    else:
        return one_model_predict(
            group_name=ensemble_name,
            model_result=ensemble,
            features_values=deepcopy(features_values) if as_dict else data,
            log=log,
            modify_dtypes=modify_dtypes,
            raise_on_encoding_error=raise_on_encoding_error,
        )


def one_model_predict(
        group_name: str,
        model_result: Union[Dict, DSManagerResult],
        features_values: Union[Dict, pd.DataFrame],
        log: bool = True,
        modify_dtypes: bool = True,
        raise_on_encoding_error: bool = True,
) -> Dict:

    if isinstance(model_result, dict):

        model_config = model_result["model_config"]
        model_config = ModelConfig.model_validate(model_config)

        model = model_result.get("model")

        features_numerical = (
            model_result.get("features_numerical")
            if model_result.get("features_numerical")
            else []
        )

        features_categorical = (
            model_result.get("features_categorical")
            if model_result.get("features_categorical")
            else []
        )

    elif isinstance(model_result, DSManagerResult):

        model_config = model_result.model_config
        model_config = ModelConfig.model_validate(model_config)
        features_numerical = model_result.data_subset.features_numerical
        features_categorical = model_result.data_subset.features_categorical
        model = model_result.model

    else:
        raise TypeError("Wrong input model_result")
    data_config=DataModelConfig(separation=SeparationModelConfig(kind='none'), source='csv')
    dataset = DataPreprocessor(prepare_dataset_interface_dict={'default_predict':
                                                                       PrepareDataset(model_config=model_config,
                                                                                      check_prepared=False,
                                                                                      )},
                                   dataset=features_values,
                               data_config=data_config).get_subset('default_predict', from_pickle=False).X_train

    prediction = model.predict(dataset[chain(features_numerical, features_categorical)])

    return {
        "group_name": group_name,
        "result": {model_config.name: pd.Series(prediction, index=dataset.index).to_dict()},
        "version_model": {model_config.name: pd.Series(group_name, index=dataset.index).to_dict()},
        "df": {model_config.name: dataset.to_dict()},
    }
