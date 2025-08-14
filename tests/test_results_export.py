from copy import deepcopy
from unittest import TestCase
import pandas as pd
from unittest import main

import config
from outboxml.export_results import ResultExport

from outboxml.core.prepared_datasets import PrepareDataset
from outboxml.dataset_retro import RetroDataset
from outboxml.datasets_manager import DataSetsManager

from pathlib import Path

test_configs_path = Path(__file__).resolve().parent/ "test_configs"
test_data_path = Path(__file__).resolve().parent/"test_data"

config_name = str(test_configs_path / 'config-example-titanic.json')


path_to_data = test_data_path / 'titanic.csv'
path_to_parquet = test_data_path / 'titanic.csv'


class DSRetro(RetroDataset):
    def __init__(self, path_to_parquet: str):
        super().__init__()
        self._path_to_parquet = path_to_parquet

    def load_retro_data(self):
        self.retro_data = pd.read_parquet(self._path_to_parquet)


class ResultExportSaveTest(TestCase):
    def setUp(self):
        def data_post_prep_func(data: pd.DataFrame):
            data["SEX"] = pd.to_numeric(data["SEX"])
            return data

        dsManager = DataSetsManager(config_name=str(config_name),
                                    prepared_datasets={
                                        'survived1': PrepareDataset(group_name='survived1',
                                                                    data_post_prep_func=data_post_prep_func),
                                        'survived2': PrepareDataset(group_name='survived2',
                                                                    )
                                    },

                                    use_baseline_model=True)
        dsManager.fit_models()
        self.result = ResultExport(ds_manager=dsManager, config=config)

    def test_init(self):
        self.assertIsNotNone(self.result.result)

    def test_save_data(self):
        self.result.save(to_pickle=True, to_mlflow=False, save_ds_manager=False)


class TestPlotExport(TestCase):

    def setUp(self):

        self.ds_manager1 = DataSetsManager(config_name=str(config_name),

                                           use_baseline_model=True)
        self.ds_manager1.fit_models()
        self.ds_manager2 = DataSetsManager(config_name=str(config_name),

                                           use_baseline_model=True)
        self.ds_manager2.fit_models()
        for key in self.ds_manager2._results.keys():
            self.ds_manager2._results[key].predictions['train'] = 0.6 * self.ds_manager1._results[key].predictions[
                'train']
            self.ds_manager2._results[key].predictions['test'] = 0.9 * self.ds_manager1._results[key].predictions[
                'test']
        self.result1 = ResultExport(self.ds_manager1, config=config)
        self.result2 = ResultExport(self.ds_manager1, self.ds_manager2)



    def test_metrics_df(self):
        df = self.result1.metrics_df(model_name='first')
        self.assertEqual(df.shape, (3, 6))

    def test_metics_plot(self):
        metrics = self.result1.plots(model_name='first', plot_type=0, use_exposure=False,  )

        metrics = self.result1.plots(model_name='first', plot_type=0, use_exposure=True,  )

        metrics = self.result1.plots(model_name='second', plot_type=0, features=["AGE"],
                                     use_exposure=False,
                                     bins_for_numerical_features=2,
                                      )
        self.assertEqual(metrics.shape, (4, 3))


    def test_factors_plot(self):
        self.result1.plots(model_name='first', plot_type=1, use_exposure=True,  )
        self.result1.plots(model_name='second', plot_type=1, features=["AGE", 'SEX'],
                           use_exposure=True,  )

    def test_cohort_plot(self):
        self.result1.plots(model_name='first', plot_type=2, cohort_base='fact',
                           use_exposure=False,  )
        self.result1.plots(model_name='second', plot_type=2,
                           )

        self.result1.plots(model_name='first', plot_type=2,
                           use_exposure=False,  cut_min_value=0.1, cut_max_value=0.8, samples=100)



    def test_compare_models_factors(self):
        df5 = self.result2.compare_metrics(model_name='first', )
        df = self.result2.compare_models_plot(model_name='first',  plot_type=1)
        df2 = self.result2.compare_models_plot(model_name='first')

        df3 = self.result2.compare_models_plot(model_name='first', features=['AGE'],
                                               bins_for_numerical_features=2)


    def test_compare_models_cohorts(self):
        df1 = self.result2.compare_models_plot(model_name='first', plot_type=2, cohort_base='fact',
                                               )
        df2 = self.result2.compare_models_plot(model_name='first', plot_type=2,
                                               use_exposure=True,  cut_min_value=0.1, cut_max_value=0.9,
                                               samples=10)

        df3 = self.result2.compare_models_plot(model_name='second', plot_type=2,
                                               use_exposure=True, cut_min_value=0.001, cut_max_value=1,
                                               samples=200, cohort_base='fact')

    def test_compare_models_relative(self):
        df1 = self.result2.compare_models_plot(model_name='first', plot_type=3, samples=1,
                                              )
        df2 = self.result2.compare_models_plot(model_name='second', plot_type=3,
                                               use_exposure=True, cut_min_value=0.1, cut_max_value=0.9,
                                               samples=0.01)

        df3 = self.result2.compare_models_plot(model_name='second', plot_type=3,
                                               use_exposure=False,  cut_min_value=0.001, cut_max_value=1,
                                               samples=200, cohort_base='fact')


    def test_grafana_export(self):
        df = self.result1.grafana_export()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (24, 7))


if __name__ == '__main__':
    main()
