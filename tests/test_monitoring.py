from pathlib import Path
from unittest import TestCase
import pandas as pd

from outboxml.monitoring_manager import MonitoringManager
from outboxml.monitoring_result import MonitoringResult, DataContext
from outboxml.core.monitoring_factory import DataReviewerRegistry, DataReviewerComponent
from outboxml.extractors import Extractor
from outboxml.metrics.base_metrics import BaseMetric

test_configs_path = Path(__file__).resolve().parent/ "test_configs"
test_data_path = Path(__file__).resolve().parent/"test_data"
config_name = test_configs_path / 'config-example-titanic.json'
monitoring_config = test_configs_path / 'monitoring_test_config.json'

path_to_data = test_data_path / 'titanic.csv'

@DataReviewerRegistry.register('test_dataset_review')
class DataSetReview(DataReviewerComponent):
    def __init__(self):
        super().__init__()

    def review(self, data_context: DataContext)-> pd.DataFrame:
        df = data_context.base
        res = []

        for column in df.columns:
            missing = df[column].isnull().mean()

            res.append({
                'column': column,
                'missing': missing,
                'X_train_length': len(data_context.X_train_index),
                'X_test_length': len(data_context.X_test_index),
            })

        return pd.DataFrame(res)

class LogsExtractor(Extractor):
    def extract_dataset(self) -> pd.DataFrame:
        return pd.read_csv(path_to_data)[:500]

class BusinessMetricsExample(BaseMetric):
    def calculate_metric(self, result1: dict, result2: dict) -> dict:
        return {'Test metric': 1}

class TestMonitoringManger(TestCase):
    def setUp(self):
        pass

    def test_monitoring(self):
        review_result = MonitoringManager(monitoring_config=str(monitoring_config),
                                   models_config=str(config_name),
                                   business_metric=BusinessMetricsExample(),
                                   logs_extractor=LogsExtractor()
                                   ).review(send_mail=False, )
        self.assertIsInstance(review_result, MonitoringResult)
        self.assertAlmostEqual(review_result.reviews['datadrift']['first']['PSI']['SEX'], 0.002, 2)
        self.assertEqual(review_result.reviews['test_dataset_review']['X_train_length'].max(), 712)