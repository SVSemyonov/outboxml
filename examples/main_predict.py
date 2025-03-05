import os
import shutil
import requests
import mlflow
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import traceback
import uvicorn

from outboxml.main_predict import main_predict

import config
from outboxml.core.pydantic_models import ServiceRequest
from outboxml.main_release import MLFLowRelease



app = FastAPI()


@app.get("/api/health")
async def health_route():
    return JSONResponse(content=jsonable_encoder({"health": True}), status_code=status.HTTP_200_OK)


@app.post("/api/predict")
async def predict_route(service_request: ServiceRequest):
    try:
        # service_request = ServiceRequest.model_validate_json(json.dumps(service_request))

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
    dir = MLFLowRelease(config=config).load_model_to_source_from_mlflow()

    uvicorn.run(app, host="0.0.0.0", port=8080)
    request = requests.post(
        "http://0.0.0.0:8080/api/predict",
        headers={"Content-Type": "application/json"},
        json={

            "main_request": []
        }
    )
