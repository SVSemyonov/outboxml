from fastapi import FastAPI, HTTPException
import requests

app = FastAPI()

MLFLOW_TRACKING_URI = "http://mlflow:5000"  # URL сервиса MLflow внутри Docker
EXPERIMENT_NAME = "fastapi_integration"

# Инициализация эксперимента при старте приложения
@app.on_event("startup")
def init_experiment():
    response = requests.post(
        f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/create",
        json={"name": EXPERIMENT_NAME}
    )
    if response.status_code != 200:
        print(f"Experiment creation skipped: {response.json()}")

@app.post("/log-metrics/")
def log_metrics(run_id: str, metrics: dict):
    """
    Логирование метрик в MLflow
    :param run_id: ID текущего эксперимента
    :param metrics: Словарь с метриками (ключ — название метрики, значение — значение)
    """
    for key, value in metrics.items():
        response = requests.post(
            f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/runs/log-metric",
            json={"run_id": run_id, "key": key, "value": value, "timestamp": 0}
        )
        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to log metric {key}: {response.json()}"
            )
    return {"message": "Metrics logged successfully"}

@app.get("/create-run/")
def create_run():
    """
    Создаёт новый запуск в MLflow
    """
    response = requests.post(
        f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/runs/create",
        json={"experiment_id": "0", "start_time": 0}
    )
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to create run")
    return response.json()
