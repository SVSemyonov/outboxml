import mlflow
import pandas as pd
import uvicorn
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware

import traceback

import config
from outboxml.export_results import ResultExport

from outboxml.automl_manager import AutoMLManager
from outboxml.automl_utils import build_default_auto_ml_config, build_default_all_models_config
from outboxml.core.pydantic_models import UpdateRequest, MonitoringRequest, AutoMLResultRequest, MonitoringResultRequest

from outboxml.monitoring_manager import MonitoringManager
from outboxml.plots import FactorsPlot

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В разработке можно "*", в продакшене укажите домены
    allow_credentials=True,
    allow_methods=["*"],  # Или ["GET", "POST", "PUT", "DELETE"]
    allow_headers=["*"],  # Или ["Content-Type", "Authorization"]
)

@app.get("/api/health_app")
async def health_route():
    return JSONResponse(content=jsonable_encoder({"health_app": True}), status_code=status.HTTP_200_OK)


@app.post("/api/update")
async def update_route(update_request: UpdateRequest):
    user_paramters = update_request.user_parameters
    retro =  user_paramters.get('retro', False)
    hp_tune = user_paramters.get('hp_tune', False)
    use_temp_files = user_paramters.get('use_temp_files', False)
    #TODO other parameters
    try:
        auto_ml_config = update_request.auto_ml_config
        all_model_config = update_request.all_model_config
        auto_ml  = AutoMLManager(auto_ml_config=auto_ml_config.dict(),
                                 models_config=all_model_config.dict(),
                                 retro=retro,
                                 hp_tune=hp_tune,
                                 use_temp_files=use_temp_files,

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
        monitoring  = MonitoringManager(monitoring_config=monitoring_config.dict(),
                                        models_config=all_model_config.dict(),
                         )
        monitoring.review()

        response = {'Monitoring status': 'OK'}
        status_code = status.HTTP_200_OK

    except Exception as exc:
        response = {"error": traceback.format_exc()}
        status_code = status.HTTP_400_BAD_REQUEST

    return JSONResponse(content=jsonable_encoder(response), status_code=status_code)


@app.get("/api/auto_ml_result")
async def update_route(auto_ml_result_request: AutoMLResultRequest):
    try:
        model_name = auto_ml_result_request.main_model

        response = {'Auto ml results': 'OK'}
        status_code = status.HTTP_200_OK

    except Exception as exc:
        response = {"error": traceback.format_exc()}
        status_code = status.HTTP_400_BAD_REQUEST

    return JSONResponse(content=jsonable_encoder(response), status_code=status_code)


@app.get("/api/monitoring_result")
async def update_route(monitoring_result_request: MonitoringResultRequest):
    try:
        model_name = monitoring_result_request.main_model

        response = {'Monitoring results': 'OK'}
        status_code = status.HTTP_200_OK

    except Exception as exc:
        response = {"error": traceback.format_exc()}
        status_code = status.HTTP_400_BAD_REQUEST

    return JSONResponse(content=jsonable_encoder(response), status_code=status_code)


@app.post("/api/default_automl_config")
async def default_automl_config(params: dict={}):
    try:

        response = build_default_auto_ml_config(params)
        status_code = status.HTTP_200_OK

    except Exception as exc:
        response = {"error": traceback.format_exc()}
        status_code = status.HTTP_400_BAD_REQUEST

    return JSONResponse(content=jsonable_encoder(response), status_code=status_code)


@app.post("/api/default_all_models_config")
async def default_model_config(params: dict={}):
    try:

        data = params.get('data', None)
        # Конвертируем входящий список диктов в DataFrame
        data = pd.DataFrame(data) if data is not None else None

        func_params = {k: v for k, v in params.items() if k != 'data'}
        response = build_default_all_models_config(data=data,
                                                    max_category_num= 20,
                                                    category_proportion_cut_value=0.01,
                                                    q1=0.001,
                                                    q2=0.999,**func_params)
        status_code = status.HTTP_200_OK

    except Exception as exc:

        response = {"error": traceback.format_exc()}
        status_code = status.HTTP_400_BAD_REQUEST

    return JSONResponse(content=jsonable_encoder(response), status_code=status_code)


def run_app(host="127.0.0.2", port=8000):
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_app()
