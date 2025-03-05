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
import uvicorn

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
    """
    :param config: `config` should contain `prod_models_path`
    :param group_name: name of the main model
    :param features_values: data for main model
    :param second_group_name: name of second model
    :param second_features_values: data for second model
    :param async_mode:
    """

    predict_tasks = []

    if not group_name:
        group_name = ResultPickle(config).get_last_group_name()

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
    return JSONResponse(content=jsonable_encoder({"health": True}), status_code=status.HTTP_200_OK)


@app.post("/api/predict")
async def predict_route(service_request: ServiceRequest):

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5100)
