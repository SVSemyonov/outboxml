import pandas as pd
import config

from outboxml.monitoring_manager import MonitoringManager

m = MonitoringManager(
    monitoring_config='configs/monitoring_test_config.json',
    models_config='configs/config-example-titanic.json',
    dashboard_connection=config.connection_params
)

m.review(
    check_datadrift=False,
    check_rolling_datadrift = False,
    check_dataset=True,
    to_dashboard=False,
    prepare_datadrift_data=True,
    send_mail=False)

for model_name, report in m.result.dataset_monitor.items():
    print(model_name)
    print(report)