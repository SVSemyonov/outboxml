from outboxml.automl_manager import AutoMLManager

config_name = './configs/test_configs/config-example-titanic.json'
auto_ml_config = './configs/test_configs/automl-titanic.json'


def titanic_example():
    AutoMLManager(auto_ml_config=auto_ml_config,
                  models_config=config_name,
                  ).update_models()


if __name__ == "__main__":
    titanic_example()
