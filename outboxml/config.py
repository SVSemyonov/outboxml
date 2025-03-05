from pathlib import Path
from environs import Env

env_reader = Env()
env_reader.read_env()

base_path = Path(__file__).resolve().parent.parent
results_path = base_path / "results"

prod_models_folder = "examples/models_to_compare"
prod_path = Path(__file__).resolve().parent.parent/ prod_models_folder
prod_models_path = prod_path



connection_params = env_reader.str("connection", f"postgresql+psycopg2://mlflow:mlflowpassword@127.0.0.1:5433/mlflow")


email_smtp_server = ""
email_port = 2525
email_sender = env_reader.str("email_sender", "")
email_login = env_reader.str("email_login", "")
email_pass = env_reader.str("email_pass", "")
email_receivers = []

