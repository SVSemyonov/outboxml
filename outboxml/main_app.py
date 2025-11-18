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

from outboxml.automl_manager import AutoMLManager
from outboxml import config
from outboxml.core.pydantic_models import UpdateRequest, MonitoringRequest
from outboxml.core.utils import ResultPickle
from outboxml.core.validators import GroupValidator
from outboxml.monitoring_manager import MonitoringManager

app = FastAPI()

@app.get("/api/health_app")
async def health_route():
    return JSONResponse(content=jsonable_encoder({"health_app": True}), status_code=status.HTTP_200_OK)


@app.post("/api/update")
async def update_route(update_request: UpdateRequest):

    try:
        auto_ml_config = update_request.auto_ml_config
        all_model_config = update_request.all_model_config
        auto_ml  = AutoMLManager(auto_ml_config=auto_ml_config,
                                models_config=all_model_config,
                         )
        auto_ml.update_models()
        response = auto_ml.status
        status_code = status.HTTP_200_OK

    except Exception as exc:
        response = {"error": traceback.format_exc()}
        status_code = status.HTTP_400_BAD_REQUEST

    return JSONResponse(content=jsonable_encoder(response), status_code=status_code)

@app.post("/api/monitoring")
async def update_route(monitoring_request: MonitoringRequest):

    try:
        monitoring_config = monitoring_request.monitoring_config
        all_model_config = monitoring_request.all_model_config
        if monitoring_config is not None and all_model_config is not None:

           monitoring  = MonitoringManager(monitoring_config=monitoring_config,
                                        models_config=all_model_config,
                         )
           monitoring.review()
        else:
            raise f'Wrong input'
        response = {'Monitoring status': 'OK'}
        status_code = status.HTTP_200_OK

    except Exception as exc:
        response = {"error": traceback.format_exc()}
        status_code = status.HTTP_400_BAD_REQUEST

    return JSONResponse(content=jsonable_encoder(response), status_code=status_code)