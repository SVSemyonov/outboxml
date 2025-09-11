import asyncio
from copy import deepcopy
from pathlib import Path
from unittest import TestCase

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from outboxml.automl_manager import RetroFS, AutoMLManager
from outboxml.core.email import EMailDSResult, EMailDSCompareResult
from outboxml.core.prepared_datasets import FeatureSelectionPrepareDataset
from outboxml.core.pydantic_models import FeatureSelectionConfig
from outboxml.datadrift import DataDrift
from outboxml.datasets_manager import DataSetsManager, DSManagerResult
from outboxml.export_results import ResultExport
from outboxml import config
from outboxml.automl_utils import load_last_pickle_models_result, calculate_previous_models
from outboxml.extractors import Extractor
from outboxml.feature_selection import BaseFS, FeatureSelectionInterface
from outboxml.hyperparameter_tuning import HPTuning
from outboxml.main_predict import main_predict
from outboxml.metrics.business_metrics import BaseCompareBusinessMetric
from outboxml.metrics.base_metrics import BaseMetric
from outboxml.monitoring_manager import MonitoringManager, MonitoringReport, MonitoringResult
from outboxml.target_extrapolation import TargetModel

test_configs_path = Path(__file__).resolve().parent/ "test_configs"
test_data_path = Path(__file__).resolve().parent/"test_data"
config_name = test_configs_path / 'config-example-titanic.json'
monitoring_config = test_configs_path / 'monitoring_test_config.json'

path_to_data = test_data_path / 'titanic.csv'
auto_ml_config = test_configs_path / 'automl-titanic.json'

path_to_target = test_data_path / 'target_extrapolation_test.gzip'


class TitanicMetric(BaseMetric):
    def __init__(self):
        pass

    def calculate_metric(self, result1: dict, result2: dict) -> dict:
        y1 = (result1['first'].y_pred + result1['second'].y_pred) / 2
        y = result1['first'].y
        score1 = (y - y1).sum()
        y2 = (result2['first'].y_pred + result2['second'].y_pred) / 2
        score2 = (y - y2).sum()
        return {'impact': score2 - score1}


class FeatureSelection(TestCase):
    def setUp(self):

        self.data = pd.read_csv(path_to_data)

        self.dsManager = DataSetsManager(config_name=str(config_name)
                                         )
        self.dsManager._retro = True
        self.dsManager.load_dataset(self.data,)
        self.dsManager._make_test_train()
        self.feature_for_research = ['PCLASS', 'NAME', 'TICKET', 'CABIN', 'EMBARKED']
        self._fs_config = FeatureSelectionConfig(
            metric_eval={"first": "accuracy", "second": "accuracy"},
            top_feautures_to_select=4,
            count_category=100,
            cutoff_1_category=0.9,
            cutoff_nan=0.7,
            max_corr_value=0.6,
            cv_diff_value=0.1,
            encoding_cat='WoE_cat_to_num',
            encoding_num='WoE_num_to_num',
        )

    def test_new_features_list(self):
        features_for_research = RetroFS(retro_columns=self.data.columns,
                                        ).features_for_reserch(data_column_names=self.dsManager.X.columns,
                                                               target_columns_names=self.dsManager.Y.columns,
                                                               models_config=self.dsManager._models_configs,
                                                               extra_columns=self.dsManager.extra_columns,
                                                               features_list_to_exclude=['SEX', 'SIBSP', 'AGE', 'PARCH',
                                                                                         'FARE']
                                                               )
        self.assertIsInstance(features_for_research, list)
        self.assertEqual(features_for_research, self.feature_for_research)

    def test_BaseFS(self):
        data = BaseFS( new_features_list=self.feature_for_research,
                      parameters=self._fs_config,
                        data_preprocessor=self.dsManager._data_preprocessor,
                      prepare_data_interface=FeatureSelectionPrepareDataset(model_config=self.dsManager._models_configs[
                          0]),
                      feature_selection_interface=FeatureSelectionInterface(feature_selection_config=self._fs_config,
                                                                            objective='binomial')
                      ).select_features(params={"iterations": 30}, model_name='first')

        self.assertEqual(len(data.features_categorical), 1)
        self.assertEqual(len(data.features_numerical), 4)


class HPTune(TestCase):

    def setUp(self):
        self.ds_manager = DataSetsManager(config_name=str(config_name)
                                          )
        self.ds_manager.get_TrainDfs()

    def test_hp_tune_catboost(self):
        self.ds_manager._prepare_datasets['first']._model_config.objective = 'poisson'
        self.ds_manager._prepare_datasets['first']._model_config.wrapper = 'catboost'

        def parameters_for_optuna(trial):
            return {
                'iterations': trial.suggest_int('iterations', 10, 12, step=1),
                'depth': trial.suggest_int('depth', 1, 15, step=2),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 1e-5, 100.0, log=True),
                'subsample': trial.suggest_float("subsample", 0.05, 1.0),
                'colsample_bylevel': trial.suggest_float("colsample_bylevel", 0.05, 1.0),
                'min_data_in_leaf': trial.suggest_int("min_data_in_leaf", 1, 101, step=10),
            }

        params = HPTuning(data_preprocessor=self.ds_manager._data_preprocessor, folds_num_for_cv=5, ).best_params(model_name='first',
                                                                                        trials=5,
                                                                                        direction='maximize',
                                                                                        parameters_for_optuna_func=parameters_for_optuna)
        self.assertIsInstance(params, dict)

    def test_hp_tune_glm(self):
        self.ds_manager._prepare_datasets['second']._model_config.objective = 'poisson'
        self.ds_manager._prepare_datasets['second']._model_config.wrapper = 'glm'
        def parameters_for_optuna(trial):
            return {
                'maxiter': trial.suggest_int('maxiter', 10, 120, step=10),
                'tol': trial.suggest_float('tol', 1e-8, 1e-4, log=True),
                'method': trial.suggest_categorical("method", ["nm", "lbfgs"]),
            }

        params = HPTuning(data_preprocessor=self.ds_manager._data_preprocessor, folds_num_for_cv=5, ).best_params(model_name='second',
                                                                                        trials=5,
                                                                                        direction='maximize',
                                                                                        parameters_for_optuna_func=parameters_for_optuna)
        self.assertIsInstance(params, dict)


class AutoMLTest(TestCase):

    def setUp(self):
        self.ds_manager1 = DataSetsManager(config_name=str(config_name)
                                           )
        self.ds_manager1.fit_models()
        self.ds_manager2 = DataSetsManager(config_name=str(config_name)
                                           )
        self.ds_manager2._separateTestTrain()
        self.ds_manager2._results = deepcopy(self.ds_manager1.get_result())

        for key in self.ds_manager2._results:
            self.ds_manager2._results[key].predictions['train'] = 0.6 * self.ds_manager2._results[key].predictions[
                'train']
            self.ds_manager2._results[key].predictions['test'] = 0.9 * self.ds_manager2._results[key].predictions[
                'test']
        self.result1 = ResultExport(self.ds_manager1)
        self.result2 = ResultExport(self.ds_manager1, self.ds_manager2)

    def test_compare_business_metric(self):
        result = BaseCompareBusinessMetric(calculate_threshold=True,
                                           use_exposure=False,
                                           metric_function=mean_absolute_error,
                                           direction='minimize').calculate_metric(self.ds_manager1.get_result(),
                                                                                  self.ds_manager2.get_result())
        self.assertIsInstance(result, dict)
        value = result['first_model']['metric'] - result['second_model']['metric']
        self.assertAlmostEqual(value, 0.002, 2)

    def test_titanic_example(self):
        auto_ml = AutoMLManager(auto_ml_config=str(auto_ml_config),
                                models_config=str(config_name),
                                business_metric=TitanicMetric(),
                                compare_business_metric=BaseCompareBusinessMetric(calculate_threshold=True),
                                save_temp=False,
                                hp_tune=True,
                                retro=True
                                )
        self.assertEqual(auto_ml.update_models(send_mail=False), {'Loading dataset': True,
                                                                  'Feature selection': False,
                                                                  'HP tuning': True,
                                                                  'Fitting': True,
                                                                  'Compare with previous': True,
                                                                  'Deployment decision': True,
                                                                  'Loading results to MLFLow': True,
                                                                  'EMail Review': False})

        self.assertGreater(auto_ml.automl_results.compare_business_metric['difference'], 0)

    def test_previous_model(self):
        group = load_last_pickle_models_result(config=config)
        res = calculate_previous_models(ds_manager=self.ds_manager1, all_groups=group)
        self.assertIsInstance(res, dict)
    """
    def test_email_result(self):
        EMailDSCompareResult(config=config, ds_manager_result=self.ds_manager1.get_result(),
                             ds_result_to_compare=self.ds_manager2.get_result()).success_mail(group_name='test')
        EMailDSResult(config=config, ds_manager_result=self.ds_manager1.get_result(),
                      ).success_mail(group_name='test')
        self.assertIsInstance(1, int)
    

class TestTargetExtrapolation(TestCase):
    def setUp(self):
        self.data = pd.read_csv(path_to_data)

    def test_extrapolate_data(self):
        X_train, y_train = DataSetsManager(str(config_name)).get_TrainDfs('second')
        res = TargetModel().extrapolate_target(model_name='second',
                                               X_train=X_train,
                                               y_train=np.reshape(y_train, (-1, 1)))
        self.assertIsInstance(res, DSManagerResult)
    """

class LogsExtractor(Extractor):
    def extract_dataset(self) -> pd.DataFrame:
        return pd.read_csv(path_to_data)[:500]


class TargetExtractor(Extractor):
    def extract_dataset(self) -> pd.DataFrame:
        return pd.read_csv(path_to_data)[500:]


class BusinessMetricsExample(BaseMetric):
    def calculate_metric(self, result1: dict, result2: dict) -> dict:
        return {'Test metric': 1}


class TestMonitoringManger(TestCase):
    def setUp(self):
        pass

    def test_monitoring(self):
        review = MonitoringManager(monitoring_config=str(monitoring_config),
                                   models_config=str(config_name),
                                   business_metric=BusinessMetricsExample(),
                                   datadrift_interface=DataDrift(full_calc=True),
                                   logs_extractor=LogsExtractor(),
                                   monitoring_report=MonitoringReport()).review(send_mail=False, )
        self.assertIsInstance(review, MonitoringResult)
        self.assertAlmostEqual(review.datadrift['first']['PSI']['SEX'], 0.002, 2)


class TestPredict(TestCase):
    def setUp(self):
        pass

    def test_predict(self):
        result = asyncio.run(
            main_predict(config=config, group_name=None, features_values=LogsExtractor().extract_dataset()[:100],
                         second_group_name=None, second_features_values=LogsExtractor().extract_dataset()[700:]))
        self.assertIsInstance(result, dict)
        self.assertIsInstance(result['main_response'], dict)
        self.assertIsInstance(result['main_response']['result'], dict)


if __name__ == '__main__':
    main()
