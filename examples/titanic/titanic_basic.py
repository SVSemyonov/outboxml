import sys
sys.path.append("../../")
sys.path.append("../../outboxml")

from outboxml.automl_manager import AutoMLManager
import config
config_name = './configs/config-example-titanic.json'
auto_ml_config = './configs/automl-titanic.json'


def titanic_example():
    AutoMLManager(auto_ml_config=auto_ml_config,
                  models_config=config_name,
                  external_config=config).update_models()


if __name__ == "__main__":
    titanic_example()
