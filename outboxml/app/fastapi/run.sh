#!/bin/sh

pip install requests
uvicorn main_app:app --host 0.0.0.0 --port 8000 --reload
