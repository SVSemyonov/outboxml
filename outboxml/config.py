from pathlib import Path
from environs import Env

env_reader = Env()
env_reader.read_env()

base_path = Path(__file__).resolve().parent.parent
results_path = base_path / "results"

prod_models_folder = env_reader.str("prod_models_folder", "prod_models_from_mlflow")
prod_models_path = base_path / prod_models_folder
prod_path = Path(__file__).resolve().parent.parent/ prod_models_folder


mlflow_tracking_uri = env_reader.str("mlflow_tracking_uri", "http://mlflow:5000")
mlflow_experiment = env_reader.str("mlflow_experiment", "FrameworkTest")


connection_params = env_reader.str("connection", f"postgresql+psycopg2://mlflow:mlflowpassword@postgre:5432/mlflow")


email_smtp_server = ""
email_port = 2525
email_sender = env_reader.str("email_sender", "")
email_login = env_reader.str("email_login", "")
email_pass = env_reader.str("email_pass", "")
email_receivers = []

