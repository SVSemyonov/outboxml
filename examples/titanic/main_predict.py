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

from outboxml.service import main

main(group_name='example_titanic',
     host="0.0.0.0",
     port=8080)

if __name__ == "__main__":
    main()
