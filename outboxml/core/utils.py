from copy import deepcopy
from pathlib import Path
import mlflow
import pandas as pd
import os
from typing import Any, List, Dict, Optional

from loguru import logger
import json
import pickle
from datetime import timedelta, datetime

from outboxml.core.pydantic_models import ModelConfig
from outboxml.core.enums import FilesNames, FeatureEngineering


def update_model_config_default(model_config: ModelConfig, feature_name: str, new_value: Any) -> ModelConfig:
    model_config = model_config.model_dump()

    for i in range(len(model_config["features"])):
        if model_config["features"][i]["name"] == feature_name:
            model_config["features"][i]["default"] = new_value
            break

    return ModelConfig.model_validate(model_config)


def update_model_config_replace(model_config: ModelConfig, absent_levels: Dict[str, List]) -> ModelConfig:
    model_config = model_config.model_dump()

    for feature_name, levels in absent_levels.items():
        for i in range(len(model_config["features"])):
            if model_config["features"][i]["name"] == feature_name:
                for value, level in list(model_config["features"][i]["replace"].items()):
                    if level == FeatureEngineering.not_changed:
                        level = value
                    if model_config["features"][i]["default"] in levels:
                        raise NotImplementedError(
                            f"""Try to delete default value {model_config["features"][i]["default"]} of {feature_name}"""
                        )
                    if level in levels:
                        model_config["features"][i]["replace"].pop(value)
                break

    return ModelConfig.model_validate(model_config)


def check_prepared_dataset(data: pd.DataFrame, features: List[str], max_unique_values=100,
                           max_corr_coef=0.8, min_group_rate=0.05) -> None:
    numUniqueDict = {}
    for feature in features:
        print(f"  ** {feature}")
        print(f"   *** dtype: {data[feature].dtypes}")

        na_count = data[feature].isna().sum()
        if na_count == 0:
            print(f"   *** (+) no nan values")
        else:
            print(f"   *** (!!!) {na_count} nan values")

        uniqueNum= data[feature].nunique()
        if data[feature].dtype in (int, float) and uniqueNum <= max_unique_values:
            print(f"   *** {uniqueNum} unique values")

        elif data[feature].dtype == object:
            print(f"   *** {uniqueNum} unique values")
            min_group_rate_1 = (data[feature].value_counts() / data.shape[0]).min()
            if min_group_rate_1 < min_group_rate:
                print(f"   *** (!!!) min group rate {round(min_group_rate_1, 3)} less than {min_group_rate}")
            else:
                print(f"   *** (+) min group rate {min_group_rate_1}")
        numUniqueDict[feature] = uniqueNum
    return numUniqueDict


def calculate_correlations(data: pd.DataFrame, features: List[str], max_corr_coef=0.8) -> pd.DataFrame:
    """
    Calculate correlation matrix.

    :param data: Dataset of features.
    :param features: List of features.
    :return: Correlation matrix.
    """
    corr_df = pd.get_dummies(data[features]).corr()

    for i in range(corr_df.shape[0]):
        for j in range(i, corr_df.shape[0]):
            if i != j:
                coef_corr = corr_df.iloc[i, j]
                if abs(coef_corr) >= max_corr_coef:
                    name_i = corr_df.index[i]
                    name_j = corr_df.index[j]
                    if '_'.join(name_i.split('_')[:-1]) != '_'.join(name_j.split('_')[:-1]):
                        print(f"  ** {round(coef_corr, 3)}: {name_i} <-> {name_j}")

    return corr_df


def find_drop_values(serie: pd.Series, replace_dict: dict, train_ind: pd.Index) -> List:

    values_train = serie.loc[[i for i in train_ind if i in serie.index]].unique()
    values_config = replace_dict.values()

    values_diff = list(set(values_config) - set(values_train))
    drop_values = values_diff

    return drop_values


def td_to_hms(td: timedelta) -> str:
    return str(td.seconds // 3600) + "h " + str((td.seconds // 60) % 60) + "m " + str(td.seconds % 60) + "s"


class ResultPickle:

    def __init__(self, config=None):
        self.date_time_format = "%Y_%m_%d_%H_%M_%S"
        self.date_time_len = 19
        self.date_time_default = datetime(2001, 1, 1, 0, 0, 0)
        self.config = config
    def generate_name(self, group_name: str, date_time: datetime) -> str:
        return f"{group_name}_{date_time.strftime(self.date_time_format)}.pickle"

    def get_date_time(self, pickle_name: str) -> datetime:
        pickle_name = pickle_name.replace(".pickle", "")
        try:
            pickle_date_time = datetime.strptime(pickle_name[-self.date_time_len:], self.date_time_format)
        except:
            pickle_date_time = self.date_time_default
        if pickle_date_time > datetime.now():
            pickle_date_time = self.date_time_default
        return pickle_date_time

    def get_last_group_name(self) -> str:
        prod_models_path = Path(self.config.prod_models_path)
        if not prod_models_path.exists():
            raise FileNotFoundError(f"Directory {prod_models_path} does not exist.")

        models_list = [f for f in os.listdir(prod_models_path) if f.endswith(FilesNames.pickle)]
        if not models_list:
            raise FileNotFoundError(f"Directory {prod_models_path} is empty.")

        models_date_time = [self.get_date_time(pickle_name) for pickle_name in models_list]
        if max(models_date_time) == self.date_time_default:
            raise FileNotFoundError(f"There are no actual models in {prod_models_path}.")

        models_date_time_index_max = max(range(len(models_date_time)), key=models_date_time.__getitem__)
        last_group_name = models_list[models_date_time_index_max].replace(".pickle", "")

        return last_group_name


def save_results(
        group_name: str,
        model_config: ModelConfig,
        features_numerical: List[str],
        features_categorical: List[str],
        model: Optional[Any],
        min_max_scaler: Optional[Any],
        metrics_train: pd.DataFrame,
        metrics_test: pd.DataFrame,
        predictions: pd.DataFrame,
        to_mlflow: bool,
        to_pickle: bool,
        config=None
) -> None:
    os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
    results_path = config.results_path
    results_path.mkdir(exist_ok=True)
    group_path = results_path / group_name
    group_path.mkdir(exist_ok=True)
    model_path = config.results_path / group_name / model_config.name
    model_path.mkdir(exist_ok=True)
    logger.debug('Saving started to' + str(results_path))
    if to_mlflow:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment)
        mlflow.start_run(run_name=f"{group_name}_{model_config.name}")
    logger.info('Saving||Configs')
    try:
        path_model_config = os.path.join(model_path, f"{model_config.name}_model_config.json")
        model_config_to_save = deepcopy(model_config)

        with open(path_model_config, "w") as f:
            json.dump(model_config_to_save.model_dump(), f)
    except:
        logger.error('Wrong format for json')
    if to_mlflow:
        mlflow.log_artifact(path_model_config)
    logger.info('Saving||Features')
    path_features_numerical = os.path.join(model_path, f"{model_config.name}_features_numerical.json")
    with open(path_features_numerical, "w") as f:
        json.dump({"features": features_numerical}, f)
    if to_mlflow:
        mlflow.log_artifact(path_features_numerical)

    path_features_categorical = os.path.join(model_path, f"{model_config.name}_features_categorical.json")
    with open(path_features_categorical, "w") as f:
        json.dump({"features": features_categorical}, f)
    if to_mlflow:
        mlflow.log_artifact(path_features_categorical)
    if to_pickle:
        if min_max_scaler:
            path_min_max_scaler = os.path.join(model_path, f"{model_config.name}_min_max_scaler.pickle")
            with open(path_min_max_scaler, "wb") as f:
                pickle.dump(min_max_scaler, f)
            if to_mlflow:
                mlflow.log_artifact(path_min_max_scaler)
        logger.info('Saving||Model')
        if model:
            path_model_ctb = os.path.join(model_path, f"{model_config.name}_model.pickle")
            with open(path_model_ctb, "wb") as f:
                pickle.dump(model, f)
            if to_mlflow:
                mlflow.log_artifact(path_model_ctb)
    logger.info('Saving||Metrics')

    metrics_train.to_excel(os.path.join(model_path, f"{model_config.name}_metrics_train.xlsx"))
    if to_mlflow:
        metrics_dict = metrics_train.to_dict()
        mlflow.log_metrics({f"train{k}": v for k, v in metrics_dict.items()})

    if not metrics_test.empty:
        pd.DataFrame([metrics_test]).to_excel(os.path.join(model_path, f"{model_config.name}_metrics_test.xlsx"))
        if to_mlflow:
            metrics_dict = metrics_test.to_dict()
            mlflow.log_metrics({f"test{k}": v for k, v in metrics_dict.items()})

    logger.info('Saving||Predictions')
    try:
        if not predictions.empty:
            predictions.to_parquet(
                os.path.join(model_path, f"{model_config.name}_predictions_sample.parquet.gzip"),
                compression="gzip",
            )
    except Exception as exc:
        logger.error(str(exc))

    if to_mlflow:
        mlflow.end_run()
