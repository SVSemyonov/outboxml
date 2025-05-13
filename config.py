from pathlib import Path
from environs import Env

env_reader = Env()
env_reader.read_env()

base_path = Path(__file__).resolve().parent
results_path = base_path / "results"
prod_path = base_path
prod_models_folder = "prod_models_from_mlflow"
prod_models_path = base_path / "examples" /"models_to_compare"

email_smtp_server = env_reader.str("email_smtp_server", "")
email_port = env_reader.str("email_port", "")

email_sender = env_reader.str("email_sender", "")
email_login = env_reader.str("email_login", "")
email_pass = env_reader.str("email_pass", "")
email_receivers = []

mlflow_tracking_uri = env_reader.str("mlflow_tracking_uri", "http://localhost:5000")
mlflow_experiment = env_reader.str("mlflow_experiment", "FrameworkTest")

connection_params = env_reader.str("connection", f"postgresql+psycopg2://mlflow:mlflowpassword@127.0.0.1:5433/mlflow")
