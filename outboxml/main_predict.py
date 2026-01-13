"""Module for implementation service of models."""
import asyncio
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from loguru import logger
import os
import pandas as pd
import pickle
import traceback
from typing import Dict, List, Optional, Union

from outboxml import config
from outboxml.core.predict import ensemble_predict
from outboxml.core.pydantic_models import ServiceRequest
from outboxml.core.utils import ResultPickle
from outboxml.core.validators import GroupValidator


app = FastAPI()


all_groups: Dict = {}


async def main_predict(
        config,
        group_name: Optional[str],
        features_values: Union[List[Dict], pd.DataFrame],
        second_group_name: Optional[str] = None,
        second_features_values: Optional[List[Dict]] = None,
        async_mode: bool = True,
) -> Dict:
    """Calculate model predictions.

    This function loads model groups from pickle files and performs
    predictions on the provided features. Supports single or dual
    model group predictions with optional async execution.

    :param config: Configuration module. Should contain ``prod_models_path``.
    :type config: module
    :param group_name: Name of the main model group. If None, uses the
        latest group from the production models path.
    :type group_name: Optional[str]
    :param features_values: Data for main model. Can be a list of dictionaries
        or a pandas DataFrame.
    :type features_values: Union[List[Dict], pd.DataFrame]
    :param second_group_name: Name of second model group. Defaults to None.
    :type second_group_name: Optional[str]
    :param second_features_values: Data for second model. Can be a list of
        dictionaries or a pandas DataFrame. Defaults to None.
    :type second_features_values: Optional[List[Dict]]
    :param async_mode: Whether to use async mode for predictions. Defaults to True.
    :type async_mode: bool
    :return: Dictionary containing predictions with keys:
        - ``usage_model``: Name of the main model group used
        - ``result``: Dictionary of predictions by model name
        - ``version_model``: Dictionary of model versions
        - ``df``: Dictionary of DataFrames with predictions
        If ``second_group_name`` is provided, also includes ``second_response``
        with the same structure for the second model group.
    :rtype: Dict

    :raises FileNotFoundError: If model group pickle file is not found.
    :raises ValidationError: If model group structure is invalid.

    Example::

        result = await main_predict(
            config=config,
            group_name="my_model_group",
            features_values=df,
            async_mode=True
        )
    """

    predict_tasks = []

    group_name = ResultPickle(config).get_last_group_name(group_name)
    group = all_groups.get(group_name)
    if not group:
        logger.info("Loading pickle || " + group_name)
        with open(os.path.join(config.prod_models_path, f"{group_name}.pickle"), "rb") as f:
            group = pickle.load(f)
            GroupValidator(group).validate()
            all_groups.update({group_name: group})

    for model in group:
        predict_tasks.append(ensemble_predict(
            group_name, model, features_values, log=False, modify_dtypes=False, raise_on_encoding_error=True
        ))

    if second_group_name:
        second_group = all_groups.get(second_group_name)
        if not second_group:
            logger.info("Loading pickle||" + second_group_name)
            with open(os.path.join(config.prod_models_path, f"{second_group_name}.pickle"), "rb") as f:
                second_group = pickle.load(f)
                GroupValidator(second_group).validate()
                all_groups.update({second_group_name: second_group})

        for model in second_group:
            predict_tasks.append(ensemble_predict(
                second_group_name, model, second_features_values, log=False, modify_dtypes=False, raise_on_encoding_error=True
            ))

    if async_mode:
        predictions = await asyncio.gather(*predict_tasks)
    else:
        predictions = []
        for task in predict_tasks:
            predictions.append(await task)

    main_response = {
        "usage_model": group_name,  # название сборки
        "result": {},
        "version_model": {},
        "df": {}
    }
    for predict in predictions:
        if predict["group_name"] == group_name:
            main_response["result"].update(predict["result"])
            main_response["version_model"].update(predict["version_model"])
            main_response["df"].update(predict["df"])

    second_response = {}
    if second_group_name:
        second_response = {
            "usage_model": second_group_name,  # название сборки
            "result": {},
            "version_model": {},
            "df": {}
        }
        for predict in predictions:
            if predict["group_name"] == second_group_name:
                second_response["result"].update(predict["result"])
                second_response["version_model"].update(predict["version_model"])
                second_response["df"].update(predict["df"])

    return {"main_response": main_response, "second_response": second_response}


@app.get("/api/health")
async def health_route():
    """Check service running.

    :return: Service response in JSON format.
    :rtype: JSONResponse

    .. rubric:: Example

    import requests

    response = requests.post(url='https://service_url/api/health')
    """
    return JSONResponse(content=jsonable_encoder({"health": True}), status_code=status.HTTP_200_OK)


@app.post("/api/predict")
async def predict_route(service_request: ServiceRequest):
    """Request models predictions.

    :param service_request: Service request
    :type service_request: ServiceRequest
    :return: Service response in JSON format.
    :rtype: JSONResponse

    .. rubric:: Example

    import requests

    request_data = {
        'main_model': 'titanic'
        'main_request': [{
            'FEATURE1': 100,
            'FEATIRE2': 200
        }]
    }

    response = requests.post(
        url='https://service_url/api/predict',
        headers={'Content-Type': 'application/json'},
        json=request_data
    )
    """
    try:
        group_name = service_request.main_model
        features_values = service_request.main_request

        second_group_name = service_request.second_model
        second_features_values = service_request.second_request

        prediction = await main_predict(
                config=config,
                group_name=group_name,
                features_values=features_values,
                second_group_name=second_group_name,
                second_features_values=second_features_values,
                async_mode=True,
            )

        response = prediction
        status_code = status.HTTP_200_OK

    except Exception as exc:
        response = {"error": traceback.format_exc()}
        status_code = status.HTTP_400_BAD_REQUEST

    return JSONResponse(content=jsonable_encoder(response), status_code=status_code)


