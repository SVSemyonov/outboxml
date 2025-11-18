import pandas as pd
import config

from outboxml.monitoring_manager import MonitoringManager

m = MonitoringManager(
    monitoring_config='configs/monitoring_test_config.json',
    models_config='configs/config-example-titanic.json',
    superset_connection=config.connection_params_superset
)

m.review(check_datadrift=False, to_superset=True, to_grafana=False, send_mail=False)

for model_name, report in m.result.dataset_monitor.items():
    print(model_name)
    print(report)